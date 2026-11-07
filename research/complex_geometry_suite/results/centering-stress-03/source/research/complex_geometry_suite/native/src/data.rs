//! Native ports of frozen synthetic generators. JSON and tensor fingerprints
//! are checked against the original manifest before any scientific training.
use crate::{Result, hash, random::Random};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone)]
pub struct Example {
    pub task: String,
    pub tokens: Vec<i64>,
    pub targets: Vec<i64>,
    pub changes: Vec<bool>,
    pub prompt: Vec<i64>,
    pub answer: Vec<i64>,
    pub limit: usize,
    pub field: Option<(i64, i64)>,
    pub row_index: Option<i64>,
}
impl Example {
    fn prefix(task: &str, tokens: Vec<i64>, targets: Vec<i64>) -> Self {
        let mut prior = None;
        let changes = targets
            .iter()
            .map(|&y| {
                let changed = y != -100 && prior != Some(y);
                if y != -100 {
                    prior = Some(y);
                }
                changed
            })
            .collect();
        Self {
            task: task.into(),
            tokens,
            targets,
            changes,
            prompt: vec![],
            answer: vec![],
            limit: 0,
            field: None,
            row_index: None,
        }
    }
    pub(crate) fn generation(task: &str, prompt: Vec<i64>, answer: Vec<i64>) -> Self {
        let mut tokens = prompt.clone();
        tokens.extend_from_slice(&answer[..answer.len() - 1]);
        let mut targets = vec![-100; prompt.len() - 1];
        targets.extend(&answer);
        let mut out = Self::prefix(task, tokens, targets);
        out.limit = answer.len();
        out.prompt = prompt;
        out.answer = answer;
        out
    }
    fn key(&self) -> &[i64] {
        if self.prompt.is_empty() {
            &self.tokens
        } else {
            &self.prompt
        }
    }
    fn value(&self) -> Value {
        json!({"task":self.task,"tokens":self.tokens,"targets":self.targets,"changes":self.changes,"prompt":self.prompt,"answer":self.answer,"generation_limit":self.limit})
    }
}
pub struct Task {
    pub splits: BTreeMap<String, Vec<Example>>,
    pub vocab: usize,
    pub selection: String,
}
pub fn json_text(v: &Value) -> String {
    match v {
        Value::Array(a) => format!(
            "[{}]",
            a.iter().map(json_text).collect::<Vec<_>>().join(", ")
        ),
        Value::Object(o) => format!(
            "{{{}}}",
            o.iter()
                .map(|(k, v)| format!("{}: {}", serde_json::to_string(k).unwrap(), json_text(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        _ => v.to_string(),
    }
}
fn tensor_hash(examples: &[Example]) -> String {
    let width = examples.iter().map(|e| e.tokens.len()).max().unwrap();
    let mut raw = format!("torch.int64:({}, 3, {}):", examples.len(), width).into_bytes();
    for e in examples {
        for part in 0..3 {
            for col in 0..width {
                let x = match part {
                    0 => e.tokens.get(col).copied().unwrap_or(0),
                    1 => e.targets.get(col).copied().unwrap_or(-100),
                    _ => {
                        if let (0, Some(index)) = (col, e.row_index) {
                            index
                        } else {
                            i64::from(e.changes.get(col).copied().unwrap_or(false))
                        }
                    }
                };
                raw.extend_from_slice(&x.to_le_bytes());
            }
        }
    }
    hash(&raw)
}
fn rng(spec: &Value, split: &str, formats: &[&str]) -> Random {
    let mut spec = spec.clone();
    for key in formats {
        spec.as_object_mut().unwrap().remove(*key);
    }
    let bytes = Sha256::digest(
        json_text(&json!({"version":2,"seed":1,"split":split,"spec":spec})).as_bytes(),
    );
    let words: Vec<u32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .rev()
        .map(|x| u32::from_be_bytes(*x))
        .collect();
    Random::words(&words)
}
fn split_hash(spec: &Value, split: &str, examples: &[Example]) -> String {
    hash(json_text(&json!({"version":2,"name":split,"spec":spec,"seed":1,"examples":examples.iter().map(Example::value).collect::<Vec<_>>()})).as_bytes())
}
fn distribute(n: usize, parts: usize, rng: &mut Random) -> Vec<usize> {
    let mut values = vec![0; parts];
    for _ in 0..n {
        values[rng.below(parts as u32) as usize] += 1;
    }
    values
}
fn blocks_example(spec: &Value, rng: &mut Random, length: usize, matched: bool) -> Vec<Example> {
    let blocks = spec["blocks"].as_u64().unwrap() as usize;
    let size = ((length - 1) as f64 * 0.75).round_ties_even() as usize;
    let word = |rng: &mut Random| {
        let runs = 1 + rng.below(size.min(2 * blocks + 3) as u32) as usize;
        let extras = distribute(size - runs, runs, rng);
        let start = 9 + rng.below(2) as i64;
        extras
            .iter()
            .enumerate()
            .flat_map(|(i, &n)| {
                std::iter::repeat_n(if i % 2 == 0 { start } else { 19 - start }, n + 1)
            })
            .collect::<Vec<_>>()
    };
    let make = |word: &[i64], gaps: &[usize]| {
        let mut tokens = vec![1];
        tokens.extend(std::iter::repeat_n(11, gaps[0]));
        for (i, &t) in word.iter().enumerate() {
            tokens.push(t);
            tokens.extend(std::iter::repeat_n(11, gaps[i + 1]));
        }
        let (mut first, mut prev, mut runs) = (0, 0, 0);
        let mut targets = vec![-100];
        for &t in &tokens[1..] {
            if t != 11 {
                if first == 0 {
                    first = t;
                }
                if t != prev {
                    runs += 1;
                }
                prev = t;
            }
            targets.push(if first == 9 && runs == blocks { 16 } else { 15 });
        }
        Example::prefix("blocks", tokens, targets)
    };
    if !matched {
        (0..2)
            .map(|_| {
                let w = word(rng);
                let gaps = distribute(length - 1 - size, size + 1, rng);
                make(&w, &gaps)
            })
            .collect()
    } else {
        let mut lengths = vec![1; blocks];
        lengths[0] = 2;
        lengths[1] = 2;
        let extra = distribute(size - lengths.iter().sum::<usize>(), blocks, rng);
        let first: Vec<_> = lengths
            .iter()
            .zip(&extra)
            .enumerate()
            .flat_map(|(i, (base, extra))| {
                std::iter::repeat_n(if i % 2 == 0 { 9 } else { 10 }, base + extra)
            })
            .collect();
        let mut second = first.clone();
        let boundary = lengths[0] + extra[0];
        second.swap(boundary - 1, boundary);
        let gaps = distribute(length - 1 - size, size + 1, rng);
        vec![make(&first, &gaps), make(&second, &gaps)]
    }
}
fn mqar_example(spec: &Value, rng: &mut Random, length: usize) -> Example {
    let symbols = spec["symbols"].as_u64().unwrap() as usize;
    let pairs = spec["pairs"].as_u64().unwrap() as usize;
    let queries = spec["queries"].as_u64().unwrap() as usize;
    let keys = rng.sample(symbols, pairs);
    let values: Vec<i64> = (0..pairs)
        .map(|_| 36 + symbols as i64 + rng.below(symbols as u32) as i64)
        .collect();
    let mut tokens = vec![1];
    tokens.extend((1..length).map(|_| 36 + symbols as i64 + rng.below(symbols as u32) as i64));
    for i in 0..pairs {
        tokens[1 + 2 * i] = 36 + keys[i] as i64;
        tokens[2 + 2 * i] = values[i];
    }
    let end = 1 + 2 * pairs;
    let mut positions: Vec<_> = (end..length)
        .map(|p| {
            (
                (-(rng.random().max(f64::MIN_POSITIVE)).ln()).ln() + 0.1 * (p as f64).ln(),
                p,
            )
        })
        .collect();
    positions.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let selected = rng.sample(pairs, queries);
    let mut targets = vec![-100; length];
    for ((_, pos), &key) in positions.iter().take(queries).zip(&selected) {
        tokens[*pos] = 36 + keys[key] as i64;
        targets[*pos] = values[key];
    }
    Example::prefix("mqar", tokens, targets)
}
fn bit_example(spec: &Value, rng: &mut Random, length: usize) -> Example {
    for _ in 0..4 {
        let _ = rng.bits(32);
    }
    let parity = spec["task"] == "parity";
    let mut bits = if parity {
        (0..length)
            .map(|_| 21 + rng.below(2) as i64)
            .collect::<Vec<_>>()
    } else {
        vec![22; length]
    };
    if !parity && rng.random() < 0.5 {
        let cut = length - (length / 4).max(1);
        let (start, end) = if spec["and_shift"] == true {
            if spec["and_region"] == "early" {
                (0, cut)
            } else {
                (cut, length)
            }
        } else {
            (0, length)
        };
        bits[start + rng.below((end - start) as u32) as usize] = 21;
    }
    let answer = if parity {
        24 + (bits.iter().filter(|&&b| b == 22).count() % 2) as i64
    } else {
        if bits.iter().all(|&b| b == 22) {
            16
        } else {
            15
        }
    };
    let mut prompt = vec![1];
    prompt.extend(bits);
    prompt.push(18);
    Example::generation(
        if parity { "parity" } else { "boolean_and" },
        prompt,
        vec![answer, 17],
    )
}
fn generate(spec: &Value, split: &str, count: usize) -> Vec<Example> {
    let name = spec["task"].as_str().unwrap();
    let formats = if name == "parity" {
        vec!["scratchpad", "index_hints", "symbols", "number_limit"]
    } else {
        vec![]
    };
    let mut rng = rng(spec, split, &formats);
    let length = spec["length"].as_u64().unwrap() as usize;
    let min = spec["min_length"].as_u64().unwrap_or(length as u64) as usize;
    let mut out = Vec::new();
    while out.len() < count {
        let n = min + rng.below((length - min + 1) as u32) as usize;
        let mut pair = match name {
            "blocks" => {
                let matched = rng.random() >= 1. / 3.;
                blocks_example(spec, &mut rng, n, matched)
            }
            "mqar" => (0..2).map(|_| mqar_example(spec, &mut rng, n)).collect(),
            _ => (0..2).map(|_| bit_example(spec, &mut rng, n)).collect(),
        };
        if rng.random() < 0.5 {
            pair.reverse();
        }
        out.extend(pair);
    }
    out.truncate(count);
    out
}
fn numeric(row: &Value) -> Result<Task> {
    let name = row["case"]["data"]["task"].as_str().ok_or("task missing")?;
    let mut splits = BTreeMap::new();
    for (split, count, seed) in [
        ("train", 1024, 1001),
        ("validation", 256, 2001),
        ("test", 256, 3001),
    ] {
        let mut rng = Random::torch_seed(seed);
        let mut examples = Vec::new();
        for _ in 0..count {
            let signs: Vec<i64> = (0..64)
                .map(|_| 2 * (rng.next_u32() % 2) as i64 - 1)
                .collect();
            let inputs = if name == "linear-transport" {
                &signs[..32]
            } else {
                &signs[..]
            };
            let mut prompt = vec![1];
            prompt.extend(inputs.iter().map(|s| 52 + 4 * s));
            prompt.push(18);
            let mut answer: Vec<i64> = (0..32)
                .map(|i| {
                    52 + if name == "linear-transport" {
                        signs[i]
                    } else {
                        4 * signs[i] + signs[32 + i]
                    }
                })
                .collect();
            answer.push(17);
            examples.push(Example::generation(name, prompt, answer));
        }
        let actual = hash(
            json_text(&Value::Array(examples.iter().map(Example::value).collect())).as_bytes(),
        );
        if row["data_fingerprints"][split] != actual {
            return Err(format!("{split} numeric fingerprint {actual}").into());
        }
        splits.insert(split.into(), examples);
    }
    Ok(Task {
        splits,
        vocab: 69,
        selection: "validation".into(),
    })
}
pub fn build(row: &Value) -> Result<Task> {
    if row["case"]["kind"] == "arithmetic" {
        let data = crate::arithmetic_data::arithmetic(row)?;
        let mut splits = BTreeMap::new();
        for (name, rows) in [("train", data.train), ("validation", data.validation)] {
            splits.insert(
                name.into(),
                rows.into_iter()
                    .map(|[a, b, y]| {
                        let mut e = Example::prefix(
                            "arithmetic",
                            vec![a as i64, b as i64],
                            vec![-100, y as i64],
                        );
                        e.changes = vec![false, false];
                        e
                    })
                    .collect(),
            );
        }
        return Ok(Task {
            splits,
            vocab: data.vocab,
            selection: "validation".into(),
        });
    }
    let case = row["name"].as_str().ok_or("name missing")?;
    if case == "recurrence-fibonacci-k1-c2-lgt" {
        return recurrence(row);
    }
    if case == "associative-divisor-n2-l1-w0" {
        return associative(row);
    }
    if matches!(
        case,
        "gpt-probe-linear-transport" | "gpt-probe-branch-amplitudes"
    ) {
        return numeric(row);
    }
    let (spec, vocab) = match case {
        "suite-boolean-and-shift" => (
            json!({"task":"boolean_and","length":16,"min_length":8,"and_shift":true,"and_region":"early"}),
            100,
        ),
        "suite-boolean-and-random" => (
            json!({"task":"boolean_and","length":16,"min_length":8,"and_shift":false}),
            100,
        ),
        "basis-parity16" | "basis-parity20" | "basis-parity24" => (
            json!({"task":"parity","length":row["case"]["data"]["bits"],"min_length":1,"scratchpad":"none","index_hints":false}),
            68,
        ),
        "basis-depth-easy" | "basis-depth-hard" => (
            json!({"task":"blocks","length":64,"min_length":32,"neutral_fraction":0.25,"max_neutral_gap":null,"blocks":if case.ends_with("hard"){4}else{2}}),
            36,
        ),
        "basis-recall-easy" => (
            json!({"task":"mqar","length":64,"min_length":32,"symbols":256,"pairs":8,"queries":8,"query_gap":0,"alpha":0.1}),
            548,
        ),
        "gpt-probe-key-value-binding" => (
            json!({"task":"mqar","length":12,"min_length":null,"symbols":32,"pairs":2,"queries":1,"query_gap":0,"alpha":0.1}),
            100,
        ),
        _ => return Err("native data family not yet implemented".into()),
    };
    let mut splits = BTreeMap::new();
    for (name, expected) in row["data_fingerprints"]
        .as_object()
        .ok_or("fingerprints missing")?
    {
        if name.ends_with("/novel") {
            continue;
        }
        let source = if name == "train_clean" {
            "train"
        } else {
            name.split('/').next().unwrap()
        };
        let count = if source == "train" {
            row["train_examples"].as_u64().unwrap() as usize
        } else if case.starts_with("suite-") {
            if source == "validation" { 32 } else { 64 }
        } else {
            512
        };
        let mut current = spec.clone();
        if let Some((_, extra)) = name.split_once('/') {
            if let Some((_, length)) = extra.rsplit_once("length-") {
                current["length"] = length.parse::<usize>()?.into();
                current["min_length"] = Value::Null;
            }
            if extra.starts_with("position") {
                current["and_region"] = "late".into();
            }
        }
        let examples = generate(&current, source, count);
        let actual = split_hash(&current, source, &examples);
        if expected != &Value::String(actual.clone()) {
            return Err(format!("{name} synthetic fingerprint {actual}").into());
        }
        splits.insert(name.clone(), examples);
    }
    let selection = row["selection"].as_str().ok_or("selection missing")?;
    if let Some(base) = selection.strip_suffix("/novel") {
        let train: BTreeSet<Vec<i64>> = splits["train"].iter().map(|e| e.key().to_vec()).collect();
        let novel: Vec<_> = splits[base]
            .iter()
            .filter(|e| !train.contains(e.key()))
            .cloned()
            .collect();
        if let Some(expected) = row["data_fingerprints"].get(selection) {
            let actual = tensor_hash(&novel);
            if expected != &Value::String(actual.clone()) {
                return Err(format!("{selection} tensor fingerprint {actual}").into());
            }
        }
        splits.insert(selection.into(), novel);
    }
    if splits["train"].len() != row["train_examples"].as_u64().unwrap() as usize
        || splits[selection].len() != row["heldout_examples"].as_u64().unwrap() as usize
    {
        return Err("frozen example counts disagree".into());
    }
    Ok(Task {
        splits,
        vocab,
        selection: selection.into(),
    })
}

fn recurrence(row: &Value) -> Result<Task> {
    let mut remaining: BTreeSet<(i64, i64)> = (0..97)
        .flat_map(|a| (0..97).map(move |b| (a, b)))
        .filter(|&p| p != (0, 0))
        .collect();
    let mut orbits = Vec::new();
    while let Some(&first) = remaining.first() {
        let mut state = first;
        let mut orbit = Vec::new();
        while remaining.remove(&state) {
            orbit.push(state);
            state = (state.1, (state.0 + state.1) % 97);
        }
        if state != first {
            return Err("recurrence orbit failed to close".into());
        }
        orbits.push(orbit);
    }
    let mut ids: Vec<usize> = (0..orbits.len()).collect();
    crate::arithmetic_data::SplitMix64(1 ^ 0x6F72626974730001).shuffle(&mut ids);
    let (mut train, mut heldout) = (Vec::new(), Vec::new());
    for i in ids {
        if train.len() <= heldout.len() {
            train.extend_from_slice(&orbits[i]);
        } else {
            heldout.extend_from_slice(&orbits[i]);
        }
    }
    crate::arithmetic_data::SplitMix64(1 ^ 0x747261696E000002).shuffle(&mut train);
    crate::arithmetic_data::SplitMix64(1 ^ 0x7465737400000003).shuffle(&mut heldout);
    let mut raw = format!("torch.int64:({}, 4):", train.len() + heldout.len()).into_bytes();
    let mut splits = BTreeMap::new();
    let mut index = 0;
    for (name, states) in [("train", train), ("validation", heldout)] {
        let mut examples = Vec::new();
        for (a, b) in states {
            let c = (a + b) % 97;
            let future = (b + c) % 97;
            for x in [a, b, c, future] {
                raw.extend_from_slice(&x.to_le_bytes());
            }
            let mut e = Example::prefix(
                "recurrence",
                vec![b - a + 128, c - b + 128],
                vec![-100, future - c + 128],
            );
            e.changes.fill(false);
            e.row_index = Some(index);
            e.field = Some((c, future));
            index += 1;
            examples.push(e);
        }
        let actual = tensor_hash(&examples);
        if row["data_fingerprints"][name] != actual {
            return Err(format!("{name} recurrence fingerprint {actual}").into());
        }
        splits.insert(name.into(), examples);
    }
    let actual = hash(&raw);
    if row["data_fingerprints"]["raw"] != actual {
        return Err(format!("raw recurrence fingerprint {actual}").into());
    }
    Ok(Task {
        splits,
        vocab: 256,
        selection: "validation".into(),
    })
}

fn associative(row: &Value) -> Result<Task> {
    let mut splits = BTreeMap::new();
    for (name, stream, wanted) in [("train", 0u32, 0u8), ("validation", 2, 1)] {
        let mut rng = crate::pcg::Pcg::new(&[1, stream]);
        let mut examples = Vec::new();
        while examples.len() < 4656 {
            let pending = 4656 - examples.len();
            let size = 16.max(pending.div_ceil(4) * 3);
            let mut keys = Vec::new();
            for _ in 0..size {
                let mut alphabet: Vec<i64> = (1..=32).collect();
                rng.shuffle(&mut alphabet);
                keys.push([alphabet[0], alphabet[1]]);
            }
            let mut orders = Vec::new();
            for _ in 0..size {
                let mut order = [0usize, 1];
                rng.shuffle(&mut order);
                orders.push(order);
            }
            let finals: Vec<[i64; 2]> = (0..size)
                .map(|_| [rng.below(97) as i64, rng.below(97) as i64])
                .collect();
            // The source draws values for both backwards write passes even
            // when every write is the latest and the draws are discarded.
            for _ in 0..size * 2 {
                let _ = rng.below(96);
            }
            for i in 0..size {
                let mut memory = [(keys[i][0], finals[i][0]), (keys[i][1], finals[i][1])];
                memory.sort_unstable();
                let raw: Vec<u8> = memory
                    .iter()
                    .flat_map(|&(k, v)| [k, v])
                    .flat_map(i64::to_le_bytes)
                    .collect();
                if crate::blake::digest8(&raw, b"mqar-copy-v1")[0] % 2 != wanted {
                    continue;
                }
                let mut queries = [(0usize, 0usize), (0, 1), (1, 0), (1, 1)];
                rng.shuffle(&mut queries);
                for (a, b) in queries {
                    let mut values = Vec::new();
                    for slot in orders[i] {
                        values.extend([-keys[i][slot], -96, finals[i][slot]]);
                    }
                    values.extend([-keys[i][a], -keys[i][b], finals[i][b]]);
                    // The exact public order-zero LGT scale is the largest
                    // binary64 below 4/3; compute its floor without rounding.
                    let codes: Vec<i64> = values
                        .iter()
                        .map(|&x| {
                            128 + (x as i128 * 6004799503160661i128).div_euclid(1i128 << 52) as i64
                        })
                        .collect();
                    let mut labels = vec![-100; 8];
                    labels[7] = codes[8];
                    let mut e = Example::prefix("associative", codes[..8].to_vec(), labels);
                    e.changes.fill(false);
                    examples.push(e);
                    if examples.len() == 4656 {
                        break;
                    }
                }
                if examples.len() == 4656 {
                    break;
                }
            }
        }
        let actual = tensor_hash(&examples);
        if row["data_fingerprints"][name] != actual {
            return Err(format!("{name} associative fingerprint {actual}").into());
        }
        splits.insert(name.into(), examples);
    }
    Ok(Task {
        splits,
        vocab: 256,
        selection: "validation".into(),
    })
}
