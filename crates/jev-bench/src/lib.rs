// Latency bench harness skeleton (#T-rust-skel): times batched inference
// on the tiny model and reports p50/p95 over N iterations.
//
// #T-local-infer adds the device-aware report: p50/p95/p99 with raw samples
// so the gate artifact is reproducible, not a bare summary number.
use std::time::Instant;

pub struct BenchResult {
    pub iters: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
}

pub fn bench_latency(iters: usize) -> BenchResult {
    let r = bench_device(iters);
    BenchResult {
        iters: r.iters,
        p50_ms: r.p50_ms,
        p95_ms: r.p95_ms,
    }
}

/// Full reproducible report for the gate artifact.
#[derive(Debug, Clone, serde::Serialize)]
pub struct BenchReport {
    pub device: &'static str,
    pub iters: usize,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub samples_ms: Vec<f64>,
}

fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    sorted[(sorted.len() * pct / 100).min(sorted.len() - 1)]
}

pub fn bench_device(iters: usize) -> BenchReport {
    bench_device_on(iters, jev_model::backend::Device::Cpu)
}

fn bench_device_on(iters: usize, device: jev_model::backend::Device) -> BenchReport {
    let weights = vec![0.5f32; 8 * 16];
    let bias = vec![0.0f32; 8];
    let input = vec![0.25f32; 16];
    let mut rt = jev_runtime::Runtime::with_device(device);
    let mut samples = Vec::with_capacity(iters);
    for i in 0..iters {
        let t = Instant::now();
        // Fresh key per iter so we time compute, not the cache.
        rt.infer(&format!("bench#{i}"), &weights, &bias, &input, 8);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    BenchReport {
        device: device.as_str(),
        iters,
        p50_ms: percentile(&samples, 50),
        p95_ms: percentile(&samples, 95),
        p99_ms: percentile(&samples, 99),
        samples_ms: samples,
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

    #[test]
    fn device_report_carries_raw_samples() {
        let r = bench_device(20);
        assert_eq!(r.device, "cpu");
        assert_eq!(r.samples_ms.len(), 20);
        assert!(r.p50_ms <= r.p95_ms && r.p95_ms <= r.p99_ms);
        assert!(r.samples_ms.windows(2).all(|w| w[0] <= w[1]));
    }
}
