// Tiny reference model for the parity harness (#T-rust-skel).
//
// The SAME math is implemented in `training/python/parity/reference.py`.
// `tests/parity.rs` runs both and asserts agreement across the stack §30
// matrix (short/long states, 2–32 options, choice/boolean × FP32/FP16).
// INT8 runs only when a quantized fixture exists (none yet → skip).

/// Deterministic pseudo-random source shared with the Python reference.
/// LCG: state = (1103515245 * state + 12345) mod 2^31, value in [-1, 1).
pub struct Lcg(u32);

impl Lcg {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }

    pub fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345) & 0x7fff_ffff;
        f64::from(self.0) / 2147483648.0 * 2.0 - 1.0
    }
}

/// Reference semantics in f64: logits = W·x + b, then softmax.
pub fn logits_f64(weights: &[f64], bias: &[f64], input: &[f64], n_options: usize) -> Vec<f64> {
    let dim = input.len();
    (0..n_options)
        .map(|i| {
            weights[i * dim..(i + 1) * dim]
                .iter()
                .zip(input)
                .map(|(w, x)| w * x)
                .sum::<f64>()
                + bias[i]
        })
        .collect()
}

pub fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Production-shaped path in f32 (what Candle executes).
pub fn probs_f32(weights: &[f32], bias: &[f32], input: &[f32], n_options: usize) -> Vec<f32> {
    let dim = input.len();
    let logits: Vec<f32> = (0..n_options)
        .map(|i| {
            weights[i * dim..(i + 1) * dim]
                .iter()
                .zip(input)
                .map(|(w, x)| w * x)
                .sum::<f32>()
                + bias[i]
        })
        .collect();
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// FP16-rounded path (f32 → f16 → f32 around every op input).
pub fn probs_f16(weights: &[f32], bias: &[f32], input: &[f32], n_options: usize) -> Vec<f32> {
    let q = |v: f32| half::f16::from_f32(v).to_f32();
    let w: Vec<f32> = weights.iter().map(|&v| q(v)).collect();
    let b: Vec<f32> = bias.iter().map(|&v| q(v)).collect();
    let x: Vec<f32> = input.iter().map(|&v| q(v)).collect();
    probs_f32(&w, &b, &x, n_options)
}

/// Tolerances from stack §30.
pub const TOL_F32: f64 = 1e-5;
pub const TOL_F16: f64 = 1e-2;

pub fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn f32_matches_f64_reference() {
        let mut rng = Lcg::new(7);
        let (dim, n) = (8, 5);
        let w: Vec<f64> = (0..n * dim).map(|_| rng.next_f64()).collect();
        let b: Vec<f64> = (0..n).map(|_| rng.next_f64()).collect();
        let x: Vec<f64> = (0..dim).map(|_| rng.next_f64()).collect();
        let reference = softmax(&logits_f64(&w, &b, &x, n));
        let wf: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let bf: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let got: Vec<f64> = probs_f32(&wf, &bf, &xf, n)
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        assert!(max_abs_diff(&reference, &got) < TOL_F32);
    }

    #[test]
    fn lcg_matches_documented_sequence() {
        // First value for seed 1, fixed by the formula above.
        let mut rng = Lcg::new(1);
        let v = rng.next_f64();
        let expected =
            f64::from(1103515245u32.wrapping_add(12345) & 0x7fff_ffff) / 2147483648.0 * 2.0 - 1.0;
        assert!((v - expected).abs() < 1e-15);
    }
}
