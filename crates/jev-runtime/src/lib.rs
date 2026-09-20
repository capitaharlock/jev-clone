// Batched inference with a state cache (#T-rust-skel).
//
// Skeleton shape of the product runtime: repeated states hit the cache,
// a batch of questions over one state is answered in one call.
use std::collections::HashMap;

pub struct Runtime {
    cache: HashMap<String, Vec<f32>>,
    pub hits: u64,
    pub misses: u64,
    device: jev_model::backend::Device,
}

/// One question inside a batch: its own weights/bias/input over the
/// shared state, answering `n_options` options.
pub struct BatchItem<'a> {
    pub weights: &'a [f32],
    pub bias: &'a [f32],
    pub input: &'a [f32],
    pub n_options: usize,
}

impl Runtime {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            device: jev_model::backend::Device::Cpu,
        }
    }

    /// Pin this runtime to an already-resolved backend device (#T-local-infer).
    pub fn with_device(device: jev_model::backend::Device) -> Self {
        Self {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
            device,
        }
    }

    pub fn device_name(&self) -> &'static str {
        self.device.as_str()
    }

    pub fn cache_size(&self) -> usize {
        self.cache.len()
    }

    /// Answer one question; `state_key` identifies the reusable prefix.
    pub fn infer(
        &mut self,
        state_key: &str,
        weights: &[f32],
        bias: &[f32],
        input: &[f32],
        n_options: usize,
    ) -> Vec<f32> {
        if let Some(hit) = self.cache.get(state_key) {
            self.hits += 1;
            return hit.clone();
        }
        self.misses += 1;
        let probs = jev_model::probs_f32(weights, bias, input, n_options);
        self.cache.insert(state_key.to_string(), probs.clone());
        probs
    }

    /// Answer a batch of questions sharing one state in a single call.
    pub fn infer_batch(&mut self, state_key: &str, items: &[BatchItem<'_>]) -> Vec<Vec<f32>> {
        items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                self.infer(
                    &format!("{state_key}#{i}"),
                    item.weights,
                    item.bias,
                    item.input,
                    item.n_options,
                )
            })
            .collect()
    }
}

impl Default for Runtime {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn second_call_hits_cache() {
        let (w, b, x) = tiny();
        let mut rt = Runtime::new();
        let first = rt.infer("s", &w, &b, &x, 2);
        let second = rt.infer("s", &w, &b, &x, 2);
        assert_eq!(first, second);
        assert_eq!((rt.hits, rt.misses), (1, 1));
        assert!((first.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn batch_answers_each_question() {
        let (w, b, x) = tiny();
        let item = || BatchItem {
            weights: &w,
            bias: &b,
            input: &x,
            n_options: 2,
        };
        let mut rt = Runtime::new();
        let out = rt.infer_batch("s", &[item(), item()]);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], out[1]);
    }

    #[test]
    fn single_and_batch_agree() {
        // Gate requirement: single↔batch parity on the same backend.
        let (w, b, x) = tiny();
        let mut rt = Runtime::with_device(jev_model::backend::Device::Cpu);
        assert_eq!(rt.device_name(), "cpu");
        let single = rt.infer("solo", &w, &b, &x, 2);
        let item = BatchItem {
            weights: &w,
            bias: &b,
            input: &x,
            n_options: 2,
        };
        let batch = rt.infer_batch("otro", &[item]);
        assert_eq!(single, batch[0]);
    }
}
