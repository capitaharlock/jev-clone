// Latency bench harness skeleton (#T-rust-skel): times batched inference
// on the tiny model and reports p50/p95 over N iterations.
use std::time::Instant;

pub struct BenchResult {
    pub iters: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

pub fn bench_latency(iters: usize) -> BenchResult {
    let weights = vec![0.5f32; 8 * 16];
    let bias = vec![0.0f32; 8];
    let input = vec![0.25f32; 16];
    let mut rt = jev_runtime::Runtime::new();
    let mut dts = Vec::with_capacity(iters);
    for i in 0..iters {
        let t = Instant::now();
        // Fresh key per iter so we time compute, not the cache.
        rt.infer(&format!("bench#{i}"), &weights, &bias, &input, 8);
        dts.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    dts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    BenchResult {
        iters,
        p50_ms: dts[dts.len() / 2],
        p95_ms: dts[(dts.len() * 95 / 100).min(dts.len() - 1)],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bench_runs_and_reports() {
        let r = bench_latency(20);
        assert_eq!(r.iters, 20);
        assert!(r.p50_ms >= 0.0 && r.p95_ms >= r.p50_ms);
    }
}
