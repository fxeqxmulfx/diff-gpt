//! Recheck original and corrected native losses on unchanged checkpoint weights.
use crate::*;

pub const LOGIT_RELATIVE_TOLERANCE: f64 = 1e-5;
pub const PARAMETER_RELATIVE_TOLERANCE: f64 = 1e-4;

pub fn run(cp: &Checkpoint, name: &str) -> Result<Value> {
    let task = data::build(&cp.protocol["case"])?;
    let mut legacy = cp.engine()?;
    let mut corrected = cp.engine()?;
    corrected.enable_stable_loss()?;
    corrected.enable_stable_attention()?;
    let mut splits = serde_json::Map::new();
    for split in ["train", "validation"] {
        let examples = &task.splits[split];
        let n = examples.len();
        let vocab = cp.cfg.vocab;
        let mut old_gradient = vec![0.; cp.p.len()];
        let mut new_gradient = vec![0.; cp.p.len()];
        let mut ref_gradient = vec![0.; cp.p.len()];
        let mut packed = metrics::Packed::default();
        let mut counts = [0usize; 5]; // exact, old zero target, new zero target, old zero loss, new zero loss
        let mut losses = [0.; 3];
        let mut errors = [0.; 2];
        let mut reference_energy = 0.;
        let mut max_row_error: f64 = 0.;
        let mut max_conservation: f64 = 0.;
        for start in (0..n).step_by(cp.cfg.max_batch) {
            let end = (start + cp.cfg.max_batch).min(n);
            let batch = end - start;
            let ids: Vec<_> = (start as u32..end as u32).collect();
            packed.fill(examples, &ids);
            legacy.forward(&packed.tokens, &packed.labels, batch, 2)?;
            corrected.forward(&packed.tokens, &packed.labels, batch, 2)?;
            let old_logits = legacy.logits.read()?;
            let new_logits = corrected.logits.read()?;
            let old_pred = legacy.predictions.read()?;
            let new_pred = corrected.predictions.read()?;
            require(
                old_logits[..batch * 2 * vocab] == new_logits[..batch * 2 * vocab],
                "loss fix changed forward logits",
            )?;
            require(
                old_pred[..batch * 2] == new_pred[..batch * 2],
                "loss fix changed predictions",
            )?;
            let old_d = legacy.dlogits.read()?;
            let new_d = corrected.dlogits.read()?;
            let old_loss = legacy.losses(batch * 2)?;
            let new_loss = corrected.losses(batch * 2)?;
            let mut ref_d = new_d.clone();
            for local in 0..batch {
                let masked = 2 * local;
                require(
                    new_loss[masked] == 0.
                        && new_d[masked * vocab..(masked + 1) * vocab]
                            .iter()
                            .all(|&v| v == 0.),
                    "corrected mask changed",
                )?;
                let row = masked + 1;
                let gold = examples[start + local].targets[1] as usize;
                let (loss, gradient) = loss_precision::reference_softmax(
                    &new_logits[row * vocab..(row + 1) * vocab],
                    gold,
                );
                counts[0] += usize::from(new_pred[row] as usize == gold);
                losses[0] += old_loss[row] as f64;
                losses[1] += new_loss[row] as f64;
                losses[2] += loss;
                counts[3] += usize::from(old_loss[row] == 0. && loss as f32 != 0.);
                counts[4] += usize::from(new_loss[row] == 0. && loss as f32 != 0.);
                require(
                    (new_loss[row] as f64 - loss).abs()
                        <= LOGIT_RELATIVE_TOLERANCE * loss.abs() + 1e-44,
                    "corrected loss differs from reference",
                )?;
                let want_gold = (gradient[gold] / batch as f64) as f32;
                counts[1] += usize::from(old_d[row * vocab + gold] == 0. && want_gold != 0.);
                counts[2] += usize::from(new_d[row * vocab + gold] == 0. && want_gold != 0.);
                let mut row_ref = 0.;
                let mut row_err = 0.;
                let mut row_sum = 0.;
                let mut row_abs = 0.;
                for (c, &g) in gradient.iter().enumerate() {
                    let index = row * vocab + c;
                    let old = old_d[index] as f64 * batch as f64;
                    let new = new_d[index] as f64 * batch as f64;
                    errors[0] += (old - g).powi(2);
                    errors[1] += (new - g).powi(2);
                    reference_energy += g * g;
                    row_ref += g * g;
                    row_err += (new - g).powi(2);
                    row_sum += new;
                    row_abs += new.abs();
                    ref_d[index] = (g / batch as f64) as f32;
                }
                if row_ref > 0. {
                    max_row_error = max_row_error.max((row_err / row_ref).sqrt());
                }
                if row_abs > 0. {
                    max_conservation = max_conservation.max(row_sum.abs() / row_abs);
                }
            }
            legacy.backward(batch, 2)?;
            corrected.backward(batch, 2)?;
            let old_g = legacy.gradients.read()?;
            let new_g = corrected.gradients.read()?;
            corrected.dlogits.write(&ref_d)?;
            corrected.backward(batch, 2)?;
            let ref_g = corrected.gradients.read()?;
            require(
                old_g
                    .iter()
                    .chain(&new_g)
                    .chain(&ref_g)
                    .all(|v| v.is_finite()),
                "nonfinite corrected gradient",
            )?;
            for i in 0..cp.p.len() {
                let fraction = batch as f64 / n as f64;
                old_gradient[i] += old_g[i] as f64 * fraction;
                new_gradient[i] += new_g[i] as f64 * fraction;
                ref_gradient[i] += ref_g[i] as f64 * fraction;
            }
        }
        require(
            cp.result["endpoint"][split]["exact"] == counts[0]
                && (cp.result["endpoint"][split]["loss"].as_f64().unwrap() - losses[0] / n as f64)
                    .abs()
                    < 1e-7,
            "legacy checkpoint replay failed",
        )?;
        require(
            counts[2] == 0 && counts[4] == 0,
            "corrected kernel lost a representable target gradient or loss",
        )?;
        require(
            max_row_error <= LOGIT_RELATIVE_TOLERANCE && max_conservation <= 2e-7,
            "corrected logit gradient failed tolerance",
        )?;
        let old = projection_gradient(&cp.cfg, &old_gradient);
        let new = projection_gradient(&cp.cfg, &new_gradient);
        let oracle = projection_gradient(&cp.cfg, &ref_gradient);
        let mut block_reports = Vec::new();
        for b in blocks(&cp.cfg) {
            let ids = owner_indices(&cp.cfg, &b);
            let a: Vec<_> = ids.iter().map(|&i| old[i]).collect();
            let n: Vec<_> = ids.iter().map(|&i| new[i]).collect();
            let r: Vec<_> = ids.iter().map(|&i| oracle[i]).collect();
            let ea: Vec<_> = a.iter().zip(&r).map(|(a, r)| a - r).collect();
            let en: Vec<_> = n.iter().zip(&r).map(|(a, r)| a - r).collect();
            let denom = stats::norm(&r);
            let rel = if denom > 0. {
                stats::norm(&en) / denom
            } else {
                stats::norm(&en)
            };
            require(
                rel <= PARAMETER_RELATIVE_TOLERANCE,
                "corrected full-parameter gradient differs from reference",
            )?;
            let momentum: Vec<_> = ids
                .iter()
                .map(|&i| {
                    cp.m[i] as f64 / ((cp.v[i] as f64).sqrt() + 1e-8)
                        + cp.cfg.decay as f64 * cp.p[i] as f64
                })
                .collect();
            block_reports.push(json!({"block":b.name,"reference_gradient_l2":denom,
                "legacy_relative_error":if denom>0.{stats::norm(&ea)/denom}else{stats::norm(&ea)},
                "corrected_relative_error":rel,"legacy_reference_cosine":stats::cosine(&a,&r),
                "corrected_reference_cosine":stats::cosine(&n,&r),
                "saved_momentum_corrected_gradient_cosine":stats::cosine(&momentum,&n)}));
        }
        splits.insert(split.into(),json!({"examples":n,"exact":counts[0],"accuracy":counts[0] as f64/n as f64,
            "logits_and_predictions_bitwise_unchanged":true,"legacy_mean_loss":losses[0]/n as f64,
            "corrected_mean_loss":losses[1]/n as f64,"reference_mean_loss":losses[2]/n as f64,
            "legacy_lost_target_gradients":counts[1],"corrected_lost_target_gradients":counts[2],
            "legacy_false_zero_losses":counts[3],"corrected_false_zero_losses":counts[4],
            "legacy_logit_gradient_relative_error":(errors[0]/reference_energy).sqrt(),
            "corrected_logit_gradient_relative_error":(errors[1]/reference_energy).sqrt(),
            "corrected_max_row_gradient_relative_error":max_row_error,"corrected_max_relative_gradient_sum":max_conservation,
            "parameter_gradients":block_reports,"checks":"passed"}));
    }
    for engine in [&legacy, &corrected] {
        require(
            engine.parameters.read()? == cp.p
                && engine.first_moment.read()? == cp.m
                && engine.second_moment.read()? == cp.v,
            "loss recheck changed parameters or optimizer state",
        )?;
    }
    Ok(
        json!({"snapshot":name,"checkpoint_sha256":cp.checkpoint["state_sha256"],"splits":splits,
        "weights_and_optimizer_unchanged":true,"training_updates":0}),
    )
}
