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

/// Encode-once + N packs benchmark (§177, #T-state-cache).
///
/// One shared state, `packs` different question packs: exactly one
/// cache miss, the rest hits. Reports cold (first encode) vs warm
/// (cached) latency with raw samples, scheduler queue latency, and
/// hit-rate. `verdict` is GO only when warm p50 stays within the
/// smoke threshold below; a miss would be a measured NO-GO, never
/// an omitted gate.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StateCacheReport {
    pub mode: &'static str,
    pub device: &'static str,
    pub packs: usize,
    pub cold_ms: f64,
    pub warm_p50_ms: f64,
    pub warm_p95_ms: f64,
    pub warm_p99_ms: f64,
    pub warm_samples_ms: Vec<f64>,
    pub queue_p50_ms: f64,
    pub batches: u64,
    pub hits: u64,
    pub misses: u64,
    pub hit_rate: f64,
    pub smoke_threshold_ms: f64,
    pub verdict: &'static str,
}

pub fn bench_state_cache(packs: usize) -> StateCacheReport {
    bench_state_cache_on(packs, jev_model::backend::Device::Cpu)
}

fn bench_state_cache_on(packs: usize, device: jev_model::backend::Device) -> StateCacheReport {
    use jev_runtime::{BatchScheduler, CacheKey, Job, Runtime, SchedConfig, Submit};
    let key = CacheKey::new("v1", "bench-state", "tok1");
    let input = vec![0.25f32; 16];
    let mut rt = Runtime::with_device(device);

    // Cold: first encode of this state (miss).
    let w0 = vec![0.5f32; 8 * 16];
    let b0 = vec![0.0f32; 8];
    let t = Instant::now();
    let _ = rt.infer(&key, &w0, &b0, &input, 8);
    let cold_ms = t.elapsed().as_secs_f64() * 1000.0;

    // Warm: same state, fresh packs (hits).
    let mut samples = Vec::with_capacity(packs);
    for i in 0..packs {
        let w: Vec<f32> = (0..8 * 16)
            .map(|j| 0.05 * ((i + j) % 11) as f32 - 0.25)
            .collect();
        let t = Instant::now();
        let (got, cached) = rt.infer(&key, &w, &b0, &input, 8);
        assert!(cached, "warm pack {i} must hit the state cache");
        assert_eq!(got.len(), 8);
        samples.push(t.elapsed().as_secs_f64() * 1000.0);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Scheduler overhead: queue the same packs, flush as one batch.
    let mut sched = BatchScheduler::new(SchedConfig::default());
    let mut rt2 = Runtime::with_device(device);
    for i in 0..packs {
        let w: Vec<f32> = (0..8 * 16)
            .map(|j| 0.05 * ((i + j) % 11) as f32 - 0.25)
            .collect();
        let fp = format!("bench-state:{i}");
        match sched.submit(
            &mut rt2,
            Job {
                key: key.clone(),
                fingerprint: fp,
                critical: false,
                weights: w,
                bias: b0.clone(),
                input: input.clone(),
                n_options: 8,
            },
        ) {
            Submit::Queued(_) => {}
            _ => panic!("bench queue must accept pack {i}"),
        }
    }
    let batches_before = sched.batches;
    sched.flush(&mut rt2).expect("bench flush");
    let threshold = 5.0;
    let warm_p50_ms = percentile(&sorted, 50);
    StateCacheReport {
        mode: "encode-once-100-packs",
        device: device.as_str(),
        packs,
        cold_ms,
        warm_p50_ms,
        warm_p95_ms: percentile(&sorted, 95),
        warm_p99_ms: percentile(&sorted, 99),
        warm_samples_ms: samples,
        queue_p50_ms: sched.queue_p50_ms(),
        batches: sched.batches - batches_before,
        hits: rt.hits,
        misses: rt.misses,
        hit_rate: rt.hit_rate(),
        smoke_threshold_ms: threshold,
        verdict: if warm_p50_ms < threshold {
            "GO"
        } else {
            "NO-GO"
        },
    }
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
        let k = jev_runtime::CacheKey::new("v0", &format!("bench#{i}"), "tok0");
        rt.infer(&k, &weights, &bias, &input, 8);
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

    #[test]
    fn state_cache_bench_is_encode_once() {
        let r = bench_state_cache(100);
        assert_eq!((r.hits, r.misses), (100, 1));
        assert_eq!(r.warm_samples_ms.len(), 100);
        assert_eq!(r.batches, 1, "scheduler must flush 100 packs in one launch");
        assert!((r.hit_rate - 100.0 / 101.0).abs() < 1e-12);
        assert_eq!(r.verdict, "GO");
    }
}
