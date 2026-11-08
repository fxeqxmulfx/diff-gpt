use super::{
    Difference, Domain, Encoded, Grid, Matrix, Result, Tokenizer, level, level_grid, vocabulary,
};

fn differences(values: &[i64], lags: &[usize]) -> Result<Vec<i64>> {
    let mut values = values.to_vec();
    for &lag in lags {
        if lag > values.len() {
            return Err("difference exceeds series length".into());
        }
        values = values[lag..]
            .iter()
            .zip(&values[..values.len() - lag])
            .map(|(&a, &b)| {
                a.checked_sub(b)
                    .ok_or_else(|| "int64 difference overflow".into())
            })
            .collect::<Result<_>>()?;
    }
    Ok(values)
}

fn invert(prefix: &[i64], steps: &[i64], lags: &[usize]) -> Result<Vec<i64>> {
    let mut firsts = Vec::new();
    let mut d = prefix.to_vec();
    for &lag in lags {
        if lag > d.len() {
            return Err("difference exceeds prefix length".into());
        }
        firsts.push(d[..lag].to_vec());
        d = differences(&d, &[lag])?;
    }
    let mut values = steps.to_vec();
    for (&lag, first) in lags.iter().zip(firsts).rev() {
        values = [first, values].concat();
        for i in lag..values.len() {
            values[i] = values[i]
                .checked_add(values[i - lag])
                .ok_or("decoded levels leave int64")?;
        }
    }
    Ok(values)
}

fn invert_wrapping(prefix: &[u64], steps: &[u64], lags: &[usize]) -> Vec<u64> {
    let mut firsts = Vec::new();
    let mut d = prefix.to_vec();
    for &lag in lags {
        firsts.push(d[..lag].to_vec());
        d = d[lag..]
            .iter()
            .zip(&d[..d.len() - lag])
            .map(|(a, b)| a.wrapping_sub(*b))
            .collect();
    }
    let mut values = steps.to_vec();
    for (&lag, first) in lags.iter().zip(firsts).rev() {
        values = [first, values].concat();
        for i in lag..values.len() {
            values[i] = values[i].wrapping_add(values[i - lag]);
        }
    }
    values
}

pub(super) fn anchor(
    scaled_prefix: &[f64],
    rows: usize,
    op: &Difference,
    tokenizer: Tokenizer,
) -> Result<(Vec<f64>, Vec<i64>)> {
    if scaled_prefix.len() != op.degree() {
        return Err("wrong anchor prefix length".into());
    }
    if op.lags.is_empty() {
        return Ok((vec![0.5; rows], vec![0; rows]));
    }
    if tokenizer == Tokenizer::Lgt {
        return Ok((vec![0.; rows + op.degree()], vec![0; rows]));
    }
    if scaled_prefix.iter().any(|x| !x.is_finite()) {
        return Err("nonfinite scaled prefix".into());
    }
    let bits = 53.min(64 - op.lags.len());
    let unit = 1_u64 << bits;
    let seeds: Vec<u64> = scaled_prefix
        .iter()
        .map(|x| ((x - x.floor()) * unit as f64).round_ties_even() as u64)
        .collect();
    let step = if tokenizer == Tokenizer::MashHalf {
        unit / 2
    } else {
        0
    };
    let numerators: Vec<i64> = invert_wrapping(&seeds, &vec![step; rows], &op.lags)
        .iter()
        .map(|n| (n & (unit - 1)) as i64)
        .collect();
    let fractions = numerators.iter().map(|&n| n as f64 / unit as f64).collect();
    let steps = differences(&numerators, &op.lags)?
        .iter()
        .map(|n| n >> bits)
        .collect();
    Ok((fractions, steps))
}

fn shape(columns: usize, operators: &[Difference]) -> Result<usize> {
    if columns != operators.len() {
        return Err("one operator per column is required".into());
    }
    Ok(operators.iter().map(Difference::degree).max().unwrap_or(0))
}

/// Encode finite rows on each column's declared level grid.
/// Out-of-domain tokens clip to [0,V-1]; the returned count is never hidden.
pub fn encode(
    input: &Matrix<f64>,
    vocab: i64,
    domains: &[Domain],
    operators: &[Difference],
    tokenizer: Tokenizer,
) -> Result<Encoded> {
    let prefix_rows = shape(input.columns, operators)?;
    if input.values.iter().any(|x| !x.is_finite()) {
        return Err("series holds NaN or infinity".into());
    }
    if prefix_rows > input.rows() {
        return Err("difference exceeds series length".into());
    }
    let grid = level_grid(vocab, domains, operators, tokenizer)?;
    let rows = input.rows() - prefix_rows;
    let mut tokens = vec![0; rows * input.columns];
    let mut clipped_tokens = 0;
    for (column, op) in operators.iter().enumerate() {
        let values: Vec<_> = (prefix_rows - op.degree()..input.rows())
            .map(|r| input.values[r * input.columns + column])
            .collect();
        let s = grid.scale[column];
        let scaled: Vec<_> = values[..op.degree()].iter().map(|x| x * s).collect();
        let (fraction, fraction_steps) = anchor(&scaled, rows, op, tokenizer)?;
        let levels: Vec<_> = values
            .iter()
            .zip(&fraction)
            .map(|(&x, &f)| level(x, s, f, op.lags.len()))
            .collect::<Result<_>>()?;
        for (r, (&d, &a)) in differences(&levels, &op.lags)?
            .iter()
            .zip(&fraction_steps)
            .enumerate()
        {
            let token = i128::from(d) + i128::from(a) + i128::from(grid.zero[column]);
            if token < 0 || token >= i128::from(vocab) {
                clipped_tokens += 1;
            }
            tokens[r * input.columns + column] = token.clamp(0, i128::from(vocab - 1)) as i64;
        }
    }
    Ok(Encoded {
        prefix: Matrix::new(
            input.columns,
            input.values[..prefix_rows * input.columns].to_vec(),
        )?,
        grid,
        tokens: Matrix::new(input.columns, tokens)?,
        clipped_tokens,
    })
}

/// Invert tokens from the supplied raw prefix. A new forecast prefix re-anchors
/// MASH just as DiffGPT does; it does not reuse an earlier frame's anchor.
pub fn decode(
    prefix: &Matrix<f64>,
    grid: &Grid,
    tokens: &Matrix<i64>,
    vocab: i64,
    operators: &[Difference],
    tokenizer: Tokenizer,
) -> Result<Matrix<f64>> {
    vocabulary(vocab)?;
    let columns = tokens.columns;
    let max_degree = shape(columns, operators)?;
    if prefix.columns != columns
        || prefix.rows() < max_degree
        || grid.scale.len() != columns
        || grid.zero.len() != columns
    {
        return Err("prefix,grid and tokens have incompatible dimensions".into());
    }
    if prefix.values.iter().any(|x| !x.is_finite())
        || grid.scale.iter().any(|s| !s.is_finite() || *s <= 0.)
        || grid.zero.iter().any(|z| z.unsigned_abs() >= (1_u64 << 62))
    {
        return Err("nonfinite prefix or invalid level grid".into());
    }
    if tokens.values.iter().any(|q| *q < 0 || *q >= vocab) {
        return Err("token outside vocabulary".into());
    }
    let count = (prefix.rows() + tokens.rows())
        .checked_mul(columns)
        .ok_or("decoded shape overflow")?;
    let mut result = prefix.values.clone();
    result.resize(count, 0.);
    for (column, op) in operators.iter().enumerate() {
        let s = grid.scale[column];
        let raw: Vec<_> = (prefix.rows() - op.degree()..prefix.rows())
            .map(|r| prefix.values[r * columns + column])
            .collect();
        let scaled: Vec<_> = raw.iter().map(|x| x * s).collect();
        let (fraction, fraction_steps) = anchor(&scaled, tokens.rows(), op, tokenizer)?;
        let levels: Vec<_> = raw
            .iter()
            .zip(&fraction)
            .map(|(&x, &f)| level(x, s, f, op.lags.len()))
            .collect::<Result<_>>()?;
        let steps: Vec<i64> = (0..tokens.rows())
            .map(|r| {
                tokens.values[r * columns + column]
                    .checked_sub(grid.zero[column])
                    .and_then(|v| v.checked_sub(fraction_steps[r]))
                    .ok_or_else(|| "token difference overflow".to_string())
            })
            .collect::<Result<_>>()?;
        let values = invert(&levels, &steps, &op.lags)?;
        for row in 0..tokens.rows() {
            let value = (values[row + op.degree()] as f64 + fraction[row + op.degree()]) / s;
            if !value.is_finite() {
                return Err("decoded value exceeds binary64 range".into());
            }
            result[(prefix.rows() + row) * columns + column] = value;
        }
    }
    Matrix::new(columns, result)
}

#[cfg(test)]
pub(super) fn check_inverse(prefix: &[i64], steps: &[i64], lags: &[usize]) -> Result<Vec<i64>> {
    invert(prefix, steps, lags)
}
