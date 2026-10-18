// Batched inference with a composite-key state cache (#T-state-cache).
//
// Product shape of the runtime: cache entries are keyed by
// (model_version, state_hash, tokenizer_hash), evicted LRU, and
// invalidated per component. `BatchScheduler` sits in front with a
// dynamic batching window (0-2 ms, critical requests bypass it),
// a bounded queue (backpressure), a bounded number of concurrent GPU
// launches, dedup of identical queued jobs, and cancellation before
// execution.
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// Composite cache key: (model_version, state_hash, tokenizer_hash).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub model_version: String,
    pub state_hash: String,
    pub tokenizer_hash: String,
}

impl CacheKey {
    pub fn new(model_version: &str, state_hash: &str, tokenizer_hash: &str) -> Self {
        Self {
            model_version: model_version.to_string(),
            state_hash: state_hash.to_string(),
            tokenizer_hash: tokenizer_hash.to_string(),
        }
    }
}

pub struct Runtime {
    cache: HashMap<CacheKey, Vec<f32>>,
    /// LRU order, front = least recently used. Linear touch is fine at
    /// V1 capacities; replace with an intrusive list only if profiled.
    order: VecDeque<CacheKey>,
    capacity: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    device: jev_model::backend::Device,
    model_version: String,
    tokenizer_hash: String,
}

impl Runtime {
    pub fn new() -> Self {
        Self::with_capacity(128)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cache: HashMap::new(),
            order: VecDeque::new(),
            capacity: capacity.max(1),
            hits: 0,
            misses: 0,
            evictions: 0,
            device: jev_model::backend::Device::Cpu,
            model_version: "dev".to_string(),
            tokenizer_hash: "dev".to_string(),
        }
    }

    /// Pin this runtime to an already-resolved backend device (#T-local-infer).
    pub fn with_device(device: jev_model::backend::Device) -> Self {
        let mut rt = Self::new();
        rt.device = device;
        rt
    }

    pub fn device_name(&self) -> &'static str {
        self.device.as_str()
    }

    pub fn set_model_version(&mut self, v: &str) {
        self.model_version = v.to_string();
    }

    pub fn set_tokenizer_hash(&mut self, h: &str) {
        self.tokenizer_hash = h.to_string();
    }

    /// Key for a state under the runtime's current model + tokenizer.
    pub fn key_for(&self, state_hash: &str) -> CacheKey {
        CacheKey::new(&self.model_version, state_hash, &self.tokenizer_hash)
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    fn touch(&mut self, key: &CacheKey) {
        if let Some(pos) = self.order.iter().position(|k| k == key) {
            self.order.remove(pos);
        }
        self.order.push_back(key.clone());
    }

    fn insert(&mut self, key: CacheKey, encoding: Vec<f32>) {
        if self.cache.contains_key(&key) {
            self.cache.insert(key.clone(), encoding);
            self.touch(&key);
            return;
        }
        while self.cache.len() >= self.capacity {
            if let Some(old) = self.order.pop_front() {
                self.cache.remove(&old);
                self.evictions += 1;
            } else {
                break;
            }
        }
        self.cache.insert(key.clone(), encoding);
        self.order.push_back(key);
    }

    /// Drop every entry built from `version`. Returns entries removed.
    pub fn invalidate_model(&mut self, version: &str) -> usize {
        self.invalidate(|k| k.model_version == version)
    }

    /// Drop every entry built with tokenizer `hash`. Returns entries removed.
    pub fn invalidate_tokenizer(&mut self, hash: &str) -> usize {
        self.invalidate(|k| k.tokenizer_hash == hash)
    }

    /// Drop every entry for state `hash`. Returns entries removed.
    pub fn invalidate_state(&mut self, hash: &str) -> usize {
        self.invalidate(|k| k.state_hash == hash)
    }

    fn invalidate(&mut self, mut pred: impl FnMut(&CacheKey) -> bool) -> usize {
        let doomed: Vec<CacheKey> = self.cache.keys().filter(|k| pred(k)).cloned().collect();
        let n = doomed.len();
        for k in &doomed {
            self.cache.remove(k);
        }
        self.order.retain(|k| !pred(k));
        n
    }

    /// Encode-once: return the cached state encoding plus whether it hit.
    pub fn encode(&mut self, key: &CacheKey, input: &[f32]) -> (Vec<f32>, bool) {
        if let Some(hit) = self.cache.get(key) {
            self.hits += 1;
            let enc = hit.clone();
            self.touch(key);
            return (enc, true);
        }
        self.misses += 1;
        let enc = input.to_vec();
        self.insert(key.clone(), enc.clone());
        (enc, false)
    }

    /// Answer one pack of questions from an encoding (never touches cache).
    pub fn answer(encoding: &[f32], weights: &[f32], bias: &[f32], n_options: usize) -> Vec<f32> {
        jev_model::probs_f32(weights, bias, encoding, n_options)
    }

    /// Full path: encode (maybe cached) + answer. Returns (probs, cached).
    pub fn infer(
        &mut self,
        key: &CacheKey,
        weights: &[f32],
        bias: &[f32],
        input: &[f32],
        n_options: usize,
    ) -> (Vec<f32>, bool) {
        let (enc, cached) = self.encode(key, input);
        (Self::answer(&enc, weights, bias, n_options), cached)
    }

    /// Reference path with no cache: the golden vector a cached call
    /// must reproduce within tolerance.
    pub fn infer_uncached(
        weights: &[f32],
        bias: &[f32],
        input: &[f32],
        n_options: usize,
    ) -> Vec<f32> {
        jev_model::probs_f32(weights, bias, input, n_options)
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
    }
}

/// One question batch item: its own weights/bias over the shared state.
#[derive(Clone, Debug)]
pub struct Job {
    pub key: CacheKey,
    /// Must cover key + weights + bias + input + n_options; the
    /// scheduler dedups queued jobs on this string alone.
    pub fingerprint: String,
    /// Critical jobs bypass the batching window (never the GPU budget).
    pub critical: bool,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub input: Vec<f32>,
    pub n_options: usize,
}

#[derive(Clone, Debug)]
pub struct SchedConfig {
    /// Batching window in ms (0-2 per spec); critical jobs skip it.
    pub window_ms: u64,
    pub max_queue: usize,
    pub max_inflight: usize,
}

impl Default for SchedConfig {
    fn default() -> Self {
        Self {
            window_ms: 2,
            max_queue: 1024,
            max_inflight: 1,
        }
    }
}

enum Slot {
    Pending,
    Done(Vec<f32>),
    Cancelled,
}

/// Handle to a queued job; `wait` blocks until it resolves.
#[derive(Clone)]
pub struct JobHandle {
    ticket: u64,
    slot: Arc<(Mutex<Slot>, Condvar)>,
}

impl JobHandle {
    pub fn ticket(&self) -> u64 {
        self.ticket
    }

    pub fn wait(&self) -> Result<Vec<f32>, &'static str> {
        let (lock, cvar) = &*self.slot;
        let mut guard = lock.lock().expect("slot lock");
        loop {
            match &*guard {
                Slot::Pending => guard = cvar.wait(guard).expect("slot wait"),
                Slot::Done(v) => return Ok(v.clone()),
                Slot::Cancelled => return Err("cancelled"),
            }
        }
    }

    pub fn poll(&self) -> Option<Result<Vec<f32>, &'static str>> {
        match &*self.slot.0.lock().expect("slot lock") {
            Slot::Pending => None,
            Slot::Done(v) => Some(Ok(v.clone())),
            Slot::Cancelled => Some(Err("cancelled")),
        }
    }

    fn resolve(&self, probs: Vec<f32>) {
        let (lock, cvar) = &*self.slot;
        *lock.lock().expect("slot lock") = Slot::Done(probs);
        cvar.notify_all();
    }

    fn cancel_slot(&self) {
        let (lock, cvar) = &*self.slot;
        *lock.lock().expect("slot lock") = Slot::Cancelled;
        cvar.notify_all();
    }
}

pub enum Submit {
    Immediate { probs: Vec<f32>, cached: bool },
    Queued(JobHandle),
    Rejected { reason: &'static str },
}

struct PendingJob {
    ticket: u64,
    job: Job,
    slot: Arc<(Mutex<Slot>, Condvar)>,
    submitted: Instant,
}

pub struct BatchScheduler {
    config: SchedConfig,
    next_ticket: u64,
    pending: VecDeque<PendingJob>,
    deadline: Option<Instant>,
    inflight: usize,
    peak_inflight: usize,
    pub batches: u64,
    pub dedup_hits: u64,
    pub rejected: u64,
    pub cancelled: u64,
    queue_wait_ms: Vec<f64>,
}

impl BatchScheduler {
    pub fn new(config: SchedConfig) -> Self {
        Self {
            config,
            next_ticket: 0,
            pending: VecDeque::new(),
            deadline: None,
            inflight: 0,
            peak_inflight: 0,
            batches: 0,
            dedup_hits: 0,
            rejected: 0,
            cancelled: 0,
            queue_wait_ms: Vec::new(),
        }
    }

    pub fn pending_len(&self) -> usize {
        self.pending.len()
    }

    pub fn peak_inflight(&self) -> usize {
        self.peak_inflight
    }

    pub fn queue_p50_ms(&self) -> f64 {
        if self.queue_wait_ms.is_empty() {
            return 0.0;
        }
        let mut v = self.queue_wait_ms.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() * 50 / 100]
    }

    fn gpu_enter(&mut self) -> bool {
        if self.inflight >= self.config.max_inflight {
            return false;
        }
        self.inflight += 1;
        self.peak_inflight = self.peak_inflight.max(self.inflight);
        true
    }

    fn gpu_exit(&mut self) {
        self.inflight = self.inflight.saturating_sub(1);
    }

    pub fn submit(&mut self, rt: &mut Runtime, job: Job) -> Submit {
        if job.critical {
            if !self.gpu_enter() {
                self.rejected += 1;
                return Submit::Rejected { reason: "gpu-busy" };
            }
            let (probs, cached) =
                rt.infer(&job.key, &job.weights, &job.bias, &job.input, job.n_options);
            self.gpu_exit();
            return Submit::Immediate { probs, cached };
        }
        if self.pending.len() >= self.config.max_queue {
            self.rejected += 1;
            return Submit::Rejected {
                reason: "queue-full",
            };
        }
        if let Some(p) = self
            .pending
            .iter()
            .find(|p| p.job.fingerprint == job.fingerprint)
        {
            self.dedup_hits += 1;
            return Submit::Queued(JobHandle {
                ticket: p.ticket,
                slot: p.slot.clone(),
            });
        }
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        let handle = JobHandle {
            ticket,
            slot: Arc::new((Mutex::new(Slot::Pending), Condvar::new())),
        };
        if self.pending.is_empty() {
            self.deadline =
                Some(Instant::now() + std::time::Duration::from_millis(self.config.window_ms));
        }
        self.pending.push_back(PendingJob {
            ticket,
            job,
            slot: handle.slot.clone(),
            submitted: Instant::now(),
        });
        Submit::Queued(handle)
    }

    /// Drop a queued job before it executes. Returns false if the
    /// ticket already ran or never existed.
    pub fn cancel(&mut self, ticket: u64) -> bool {
        if let Some(pos) = self.pending.iter().position(|p| p.ticket == ticket) {
            let p = self.pending.remove(pos).expect("pending position");
            JobHandle {
                ticket,
                slot: p.slot,
            }
            .cancel_slot();
            self.cancelled += 1;
            if self.pending.is_empty() {
                self.deadline = None;
            }
            true
        } else {
            false
        }
    }

    /// Execute everything queued as ONE GPU launch. Returns jobs run.
    pub fn flush(&mut self, rt: &mut Runtime) -> Result<usize, &'static str> {
        if !self.gpu_enter() {
            return Err("gpu-busy");
        }
        let mut ran = 0;
        while let Some(p) = self.pending.pop_front() {
            self.queue_wait_ms
                .push(p.submitted.elapsed().as_secs_f64() * 1000.0);
            let handle = JobHandle {
                ticket: p.ticket,
                slot: p.slot,
            };
            let (probs, _) = rt.infer(
                &p.job.key,
                &p.job.weights,
                &p.job.bias,
                &p.job.input,
                p.job.n_options,
            );
            handle.resolve(probs);
            ran += 1;
        }
        self.deadline = None;
        self.batches += 1;
        self.gpu_exit();
        Ok(ran)
    }

    /// Flush only once the batching window has elapsed.
    pub fn flush_if_due(&mut self, rt: &mut Runtime) -> Result<usize, &'static str> {
        match self.deadline {
            Some(t) if Instant::now() >= t => self.flush(rt),
            _ => Ok(0),
        }
    }
}

impl Default for BatchScheduler {
    fn default() -> Self {
        Self::new(SchedConfig::default())
    }
}

#[cfg(test)]
impl BatchScheduler {
    /// Test-only: pretend another worker holds GPU launches, so the
    /// gpu-busy path can be exercised deterministically.
    pub fn hold_gpu_for_test(&mut self) {
        self.inflight = self.config.max_inflight;
    }

    pub fn release_gpu_for_test(&mut self) {
        self.inflight = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (
            vec![0.5, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
            vec![0.1, -0.1],
            vec![1.0, 0.5, -0.5, 0.25],
        )
    }

    fn key(s: &str) -> CacheKey {
        CacheKey::new("v1", s, "tok1")
    }

    fn job(key: &CacheKey, fp: &str, critical: bool) -> Job {
        let (w, b, x) = tiny();
        Job {
            key: key.clone(),
            fingerprint: format!("{}:{fp}", key.state_hash),
            critical,
            weights: w,
            bias: b,
            input: x,
            n_options: 2,
        }
    }

    #[test]
    fn second_call_hits_cache() {
        let (w, b, x) = tiny();
        let mut rt = Runtime::new();
        let k = key("s");
        let (first, c1) = rt.infer(&k, &w, &b, &x, 2);
        let (second, c2) = rt.infer(&k, &w, &b, &x, 2);
        assert!(!c1);
        assert!(c2);
        assert_eq!(first, second);
        assert_eq!((rt.hits, rt.misses), (1, 1));
        assert!((first.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cached_matches_uncached_golden() {
        let (w, b, x) = tiny();
        let mut rt = Runtime::new();
        let k = key("s");
        let (got, _) = rt.infer(&k, &w, &b, &x, 2);
        assert_eq!(got, Runtime::infer_uncached(&w, &b, &x, 2));
    }

    #[test]
    fn key_components_isolate_entries() {
        let (w, b, x) = tiny();
        let mut rt = Runtime::new();
        let (_, c1) = rt.infer(&CacheKey::new("v1", "s", "tok1"), &w, &b, &x, 2);
        assert!(!c1);
        // Same state bytes, different model: miss.
        let (_, c2) = rt.infer(&CacheKey::new("v2", "s", "tok1"), &w, &b, &x, 2);
        assert!(!c2);
        // Same state bytes, different tokenizer: miss.
        let (_, c3) = rt.infer(&CacheKey::new("v1", "s", "tok2"), &w, &b, &x, 2);
        assert!(!c3);
        assert_eq!(rt.cache_size(), 3);
    }

    #[test]
    fn invalidation_drops_only_matching_entries() {
        let (w, b, x) = tiny();
        let mut rt = Runtime::new();
        rt.infer(&CacheKey::new("v1", "s1", "tok1"), &w, &b, &x, 2);
        rt.infer(&CacheKey::new("v1", "s2", "tok1"), &w, &b, &x, 2);
        assert_eq!(rt.invalidate_state("s1"), 1);
        let (_, c) = rt.infer(&CacheKey::new("v1", "s1", "tok1"), &w, &b, &x, 2);
        assert!(!c, "invalidated state must miss again");
        let (_, c) = rt.infer(&CacheKey::new("v1", "s2", "tok1"), &w, &b, &x, 2);
        assert!(c, "untouched entry must still hit");
        assert_eq!(rt.invalidate_tokenizer("tok1"), 2);
        assert_eq!(rt.cache_size(), 0);
        rt.infer(&CacheKey::new("v9", "s", "tok9"), &w, &b, &x, 2);
        assert_eq!(rt.invalidate_model("v9"), 1);
    }

    #[test]
    fn lru_evicts_least_recently_used() {
        let (w, b, x) = tiny();
        let mut rt = Runtime::with_capacity(2);
        rt.infer(&key("k1"), &w, &b, &x, 2);
        rt.infer(&key("k2"), &w, &b, &x, 2);
        // Touch k1 so k2 is least recently used.
        let (_, c) = rt.infer(&key("k1"), &w, &b, &x, 2);
        assert!(c);
        rt.infer(&key("k3"), &w, &b, &x, 2);
        assert_eq!(rt.evictions, 1);
        let (_, c) = rt.infer(&key("k1"), &w, &b, &x, 2);
        assert!(c, "recently used entry must survive");
        let (_, c) = rt.infer(&key("k2"), &w, &b, &x, 2);
        assert!(!c, "evicted entry must miss");
    }

    #[test]
    fn encode_once_answers_many_packs() {
        // §177 shape: one state, 100 packs, exactly one miss.
        let mut rt = Runtime::with_device(jev_model::backend::Device::Cpu);
        assert_eq!(rt.device_name(), "cpu");
        let k = key("shared");
        let x = vec![0.25f32; 4];
        for i in 0..100 {
            let w: Vec<f32> = (0..8).map(|j| 0.1 * ((i + j) % 7) as f32 - 0.3).collect();
            let b = vec![0.05, -0.05];
            let (got, _) = rt.infer(&k, &w, &b, &x, 2);
            assert_eq!(got, Runtime::infer_uncached(&w, &b, &x, 2));
        }
        assert_eq!((rt.hits, rt.misses), (99, 1));
        assert!((rt.hit_rate() - 99.0 / 100.0).abs() < 1e-12);
    }

    fn sched() -> (Runtime, BatchScheduler) {
        (Runtime::new(), BatchScheduler::default())
    }

    #[test]
    fn critical_bypasses_window_but_not_gpu_budget() {
        let (mut rt, mut sc) = sched();
        let k = key("s");
        match sc.submit(&mut rt, job(&k, "a", true)) {
            Submit::Immediate { cached, .. } => assert!(!cached),
            _ => panic!("critical job must execute immediately"),
        }
        sc.hold_gpu_for_test();
        match sc.submit(&mut rt, job(&k, "b", true)) {
            Submit::Rejected { reason } => assert_eq!(reason, "gpu-busy"),
            _ => panic!("contended GPU must reject, even critical"),
        }
        assert!(sc.flush(&mut rt).is_err());
        sc.release_gpu_for_test();
    }

    #[test]
    fn queue_full_is_backpressure_not_loss() {
        let (mut rt, mut sc) = (
            Runtime::new(),
            BatchScheduler::new(SchedConfig {
                max_queue: 2,
                ..Default::default()
            }),
        );
        let k = key("s");
        for fp in ["a", "b"] {
            assert!(matches!(
                sc.submit(&mut rt, job(&k, fp, false)),
                Submit::Queued(_)
            ));
        }
        match sc.submit(&mut rt, job(&k, "c", false)) {
            Submit::Rejected { reason } => assert_eq!(reason, "queue-full"),
            _ => panic!("overfull queue must reject"),
        }
        assert_eq!(sc.rejected, 1);
        assert_eq!(sc.flush(&mut rt).unwrap(), 2);
    }

    #[test]
    fn identical_queued_jobs_share_one_execution() {
        let (mut rt, mut sc) = sched();
        let k = key("s");
        let h1 = match sc.submit(&mut rt, job(&k, "same", false)) {
            Submit::Queued(h) => h,
            _ => panic!("must queue"),
        };
        let h2 = match sc.submit(&mut rt, job(&k, "same", false)) {
            Submit::Queued(h) => h,
            _ => panic!("must queue"),
        };
        assert_eq!(sc.pending_len(), 1, "dedup must not enqueue twice");
        assert_eq!(sc.dedup_hits, 1);
        assert_eq!(sc.flush(&mut rt).unwrap(), 1);
        let (w, b, x) = tiny();
        assert_eq!(h1.wait().unwrap(), Runtime::infer_uncached(&w, &b, &x, 2));
        assert_eq!(h2.wait().unwrap(), h1.wait().unwrap());
    }

    #[test]
    fn cancelled_job_never_runs_and_waiter_errors() {
        let (mut rt, mut sc) = sched();
        let k = key("s");
        let h = match sc.submit(&mut rt, job(&k, "doomed", false)) {
            Submit::Queued(h) => h,
            _ => panic!("must queue"),
        };
        assert!(sc.cancel(h.ticket()));
        assert!(!sc.cancel(h.ticket()), "double cancel must fail");
        assert_eq!(sc.flush(&mut rt).unwrap(), 0);
        assert_eq!(h.wait().unwrap_err(), "cancelled");
        assert_eq!(
            (rt.hits, rt.misses),
            (0, 0),
            "cancelled job must not touch cache"
        );
    }

    #[test]
    fn window_holds_until_due_or_flushed() {
        let (mut rt, mut sc) = sched();
        let k = key("s");
        let h = match sc.submit(&mut rt, job(&k, "w", false)) {
            Submit::Queued(h) => h,
            _ => panic!("must queue"),
        };
        assert!(h.poll().is_none(), "unflushed job must stay pending");
        assert_eq!(
            sc.flush_if_due(&mut rt).unwrap(),
            0,
            "window not elapsed: no flush"
        );
        assert_eq!(sc.flush(&mut rt).unwrap(), 1);
        assert!(h.poll().is_some());
    }

    #[test]
    fn concurrent_hammering_respects_gpu_bound() {
        use std::sync::Mutex as StdMutex;
        let shared = Arc::new(StdMutex::new((Runtime::new(), BatchScheduler::default())));
        let mut threads = Vec::new();
        for t in 0..8 {
            let shared = shared.clone();
            threads.push(std::thread::spawn(move || {
                for i in 0..25 {
                    let k = CacheKey::new("v1", &format!("t{t}"), "tok1");
                    let (w, b, x) = tiny();
                    let _ = i;
                    let mut guard = shared.lock().expect("shared lock");
                    let (probs, _) = guard.0.infer(&k, &w, &b, &x, 2);
                    assert_eq!(probs, Runtime::infer_uncached(&w, &b, &x, 2));
                }
            }));
        }
        for th in threads {
            th.join().expect("worker thread");
        }
        let guard = shared.lock().expect("shared lock");
        assert_eq!((guard.0.hits, guard.0.misses), (8 * 25 - 8, 8));
    }
}
