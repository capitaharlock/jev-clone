// Production observability (#T-release): Prometheus counters for the server.
//
// Only counts and latency sums are stored — never request bodies, weights,
// state hashes, option IDs or any user content. `render()` therefore cannot
// leak a state even if scraped by a third party.
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Metrics {
    requests: AtomicU64,
    errors: AtomicU64,
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,
    infer_us_sum: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self {
            requests: AtomicU64::new(0),
            errors: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            infer_us_sum: AtomicU64::new(0),
        }
    }
}

impl Metrics {
    /// Record one answered query. `latency_us` is the server-side handling
    /// time; `cached`/`error` are outcome flags. No payload is retained.
    pub fn record(&self, latency_us: u64, cached: bool, error: bool) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.infer_us_sum.fetch_add(latency_us, Ordering::Relaxed);
        if cached {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
        if error {
            self.errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> (u64, u64, u64, u64, u64) {
        (
            self.requests.load(Ordering::Relaxed),
            self.errors.load(Ordering::Relaxed),
            self.cache_hits.load(Ordering::Relaxed),
            self.cache_misses.load(Ordering::Relaxed),
            self.infer_us_sum.load(Ordering::Relaxed),
        )
    }

    /// Prometheus text exposition (counters only, no labels from requests).
    pub fn render(&self) -> String {
        let (req, err, hits, miss, us) = self.snapshot();
        format!(
            "# HELP jev_requests_total Answered decision queries.\n\
             # TYPE jev_requests_total counter\n\
             jev_requests_total {req}\n\
             # HELP jev_errors_total Queries answered with an error status.\n\
             # TYPE jev_errors_total counter\n\
             jev_errors_total {err}\n\
             # HELP jev_cache_hits_total State-cache hits.\n\
             # TYPE jev_cache_hits_total counter\n\
             jev_cache_hits_total {hits}\n\
             # HELP jev_cache_misses_total State-cache misses.\n\
             # TYPE jev_cache_misses_total counter\n\
             jev_cache_misses_total {miss}\n\
             # HELP jev_infer_us_sum Sum of server-side handling microseconds.\n\
             # TYPE jev_infer_us_sum counter\n\
             jev_infer_us_sum {us}\n"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_accumulate() {
        let m = Metrics::default();
        m.record(100, false, false);
        m.record(50, true, false);
        m.record(10, false, true);
        let out = m.render();
        assert!(out.contains("jev_requests_total 3"));
        assert!(out.contains("jev_errors_total 1"));
        assert!(out.contains("jev_cache_hits_total 1"));
        assert!(out.contains("jev_cache_misses_total 2"));
        assert!(out.contains("jev_infer_us_sum 160"));
    }

    #[test]
    fn render_carries_no_request_content() {
        // Even a hostile state string must never surface in the exposition:
        // record() takes only flags, so there is nothing to echo.
        let m = Metrics::default();
        let hostile = "SECRET-STATE-7f3a weights=[9.99,8.88]";
        m.record(hostile.len() as u64, false, false);
        let out = m.render();
        assert!(!out.contains("SECRET"));
        assert!(!out.contains("9.99"));
        assert!(!out.contains("weights"));
    }
}
