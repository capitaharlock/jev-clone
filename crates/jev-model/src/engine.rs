//! The real decision engine: backbone + pointer head, in Rust
//! (#T-candle-infer).
//!
//! This is the Rust twin of `model/decision_head.py::DecisionEngine` —
//! encode the state ONCE, reuse that memory for every question of the
//! row, embed question and option texts through the same backbone with a
//! per-string cache, and score with the pointer head.
//!
//! Loading refuses to proceed on a hash mismatch, exactly like
//! `model/weights.py::require_verified`: the checkpoint's own
//! `weights_sha256` and `tokenizer_hash` are re-derived from the bytes on
//! disk, and the backbone is checked against its fetch manifest. A
//! runtime that silently loads the wrong weights would make every number
//! downstream — parity, latency, the gate — a lie.
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

use crate::backend::{Device, Unavailable};
use crate::modernbert::{self, ModernBert};
use crate::pointer::{softmax, HeadConfig, PointerHead};

/// `model/encoder.py::DEFAULT_MAX_LENGTH`.
pub const DEFAULT_MAX_LENGTH: usize = 512;
/// The id the head's last logit stands for.
pub const UNKNOWN_ID: &str = "unknown";

#[derive(Debug, Clone, Deserialize)]
pub struct Architecture {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BackboneRef {
    pub id: String,
    pub hidden_size: usize,
    pub revision: String,
}

/// The fields of a #T-train-real checkpoint manifest the runtime needs.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub model_version: String,
    pub tokenizer_hash: String,
    pub weights_sha256: String,
    pub weights_file: String,
    pub architecture: Architecture,
    pub backbone: BackboneRef,
    #[serde(default)]
    pub samples_seen: u64,
    #[serde(default)]
    pub run_id: String,
}

impl Manifest {
    pub fn read(dir: &Path) -> Result<Self, String> {
        let path = dir.join("manifest.json");
        let bytes =
            std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        serde_json::from_slice(&bytes).map_err(|e| format!("bad {}: {e}", path.display()))
    }
}

/// `H_s` — one state encoded once, reused by every question and option.
#[derive(Clone)]
pub struct StateMemory {
    pub text: String,
    /// `[1, T, hidden_size]`
    pub tokens: Tensor,
    /// `[1, T]` f32, 1.0 for a real token
    pub mask: Tensor,
    pub n_tokens: usize,
}

/// One option as the V1 schema carries it.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Opt {
    pub id: String,
    pub text: String,
}

/// The distribution over THIS row's options plus `unknown`.
#[derive(Debug, Clone, Serialize)]
pub struct Decision {
    pub option_ids: Vec<String>,
    pub logits: Vec<f32>,
    pub probs: Vec<f32>,
    pub unknown_logit: f32,
    pub unknown_prob: f32,
    pub argmax_id: String,
    pub ranking: Vec<String>,
    pub k: usize,
}

pub struct Engine {
    pub device: Device,
    dev: candle_core::Device,
    tokenizer: Tokenizer,
    backbone: ModernBert,
    head: PointerHead,
    pub manifest: Manifest,
    pub checkpoint_dir: PathBuf,
    states: HashMap<String, StateMemory>,
    texts: HashMap<String, Tensor>,
    max_length: usize,
}

fn sha256_file(path: &Path) -> Result<String, String> {
    jev_core::sha256_file(path)
}

/// Re-derive a file's SHA-256 and refuse the load on a mismatch.
fn require_hash(path: &Path, want: &str, what: &str) -> Result<(), String> {
    let got = sha256_file(path)?;
    if got != want {
        return Err(format!(
            "{what}: refusing to load — SHA-256 {got} != manifest {want} ({})",
            path.display()
        ));
    }
    Ok(())
}

impl Engine {
    /// Load a checkpoint directory plus the backbone it names.
    ///
    /// `weights_root` is `artifacts/weights/`, the store
    /// `model/weights.py` fills and hashes.
    pub fn load(
        checkpoint_dir: &Path,
        weights_root: &Path,
        requested_device: &str,
    ) -> Result<Self, String> {
        let manifest = Manifest::read(checkpoint_dir)?;
        let device = crate::backend::resolve(requested_device).map_err(|e: Unavailable| e.to_string())?;
        let dev = device.to_candle()?;

        let weights = checkpoint_dir.join(&manifest.weights_file);
        require_hash(&weights, &manifest.weights_sha256, "head weights")?;
        let tok_path = checkpoint_dir.join("tokenizer.json");
        require_hash(&tok_path, &manifest.tokenizer_hash, "tokenizer")?;

        let backbone_dir = weights_root.join(&manifest.backbone.id);
        verify_backbone(&backbone_dir, &manifest.backbone)?;

        let mut tokenizer = Tokenizer::from_file(&tok_path)
            .map_err(|e| format!("cannot load {}: {e}", tok_path.display()))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: DEFAULT_MAX_LENGTH,
                ..Default::default()
            }))
            .map_err(|e| format!("cannot set truncation: {e}"))?;

        let cfg = modernbert::read_config(&backbone_dir.join("config.json"))?;
        if cfg.hidden_size != manifest.backbone.hidden_size {
            return Err(format!(
                "backbone hidden_size {} != manifest {}",
                cfg.hidden_size, manifest.backbone.hidden_size
            ));
        }
        let vb = modernbert::pth_var_builder(&backbone_dir.join("pytorch_model.bin"), &dev)?;
        let backbone = ModernBert::load(cfg, vb, &dev, DEFAULT_MAX_LENGTH)
            .map_err(|e| format!("cannot build backbone: {e}"))?;

        let head_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights.clone()], DType::F32, &dev)
                .map_err(|e| format!("cannot load {}: {e}", weights.display()))?
        };
        let head = PointerHead::load(
            HeadConfig {
                d_state: backbone.hidden_size(),
                d_model: manifest.architecture.d_model,
                n_layers: manifest.architecture.n_layers,
                n_heads: manifest.architecture.n_heads,
            },
            head_vb,
        )
        .map_err(|e| format!("cannot build head: {e}"))?;

        Ok(Self {
            device,
            dev,
            tokenizer,
            backbone,
            head,
            manifest,
            checkpoint_dir: checkpoint_dir.to_path_buf(),
            states: HashMap::new(),
            texts: HashMap::new(),
            max_length: DEFAULT_MAX_LENGTH,
        })
    }

    pub fn model_version(&self) -> &str {
        &self.manifest.model_version
    }

    pub fn tokenizer_hash(&self) -> &str {
        &self.manifest.tokenizer_hash
    }

    pub fn candle_device(&self) -> &candle_core::Device {
        &self.dev
    }

    /// Drop both caches: what a cold-latency measurement must do per row.
    pub fn clear_caches(&mut self) {
        self.states.clear();
        self.texts.clear();
    }

    pub fn cached_states(&self) -> usize {
        self.states.len()
    }

    pub fn cached_texts(&self) -> usize {
        self.texts.len()
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, String> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| format!("tokenizer failed: {e}"))?;
        let mut ids = enc.get_ids().to_vec();
        ids.truncate(self.max_length);
        Ok(ids)
    }

    /// One real forward over the state, memoised per exact string.
    pub fn encode_state(&mut self, state: &str) -> Result<StateMemory, String> {
        if let Some(hit) = self.states.get(state) {
            return Ok(hit.clone());
        }
        let mem = self.encode_state_uncached(state)?;
        self.states.insert(state.to_string(), mem.clone());
        Ok(mem)
    }

    /// The same forward with no cache lookup and no insert.
    pub fn encode_state_uncached(&self, state: &str) -> Result<StateMemory, String> {
        if state.trim().is_empty() {
            return Err("state must be a non-empty string".to_string());
        }
        let ids = self.tokenize(state)?;
        let out = self
            .backbone
            .forward(&[ids])
            .map_err(|e| format!("backbone forward failed: {e}"))?;
        Ok(StateMemory {
            text: state.to_string(),
            tokens: out.tokens,
            mask: out.mask,
            n_tokens: out.n_tokens[0],
        })
    }

    /// Pooled text embeddings `[n, hidden_size]`, memoised per exact string.
    ///
    /// The misses are embedded in ONE padded batch, as the Python engine
    /// does; padding is masked out, so the cached vector is the same one a
    /// single-text forward would produce.
    pub fn embed_texts(&mut self, texts: &[String]) -> Result<Tensor, String> {
        let mut missing: Vec<String> = Vec::new();
        for t in texts {
            if !self.texts.contains_key(t) && !missing.contains(t) {
                missing.push(t.clone());
            }
        }
        if !missing.is_empty() {
            let batch: Vec<Vec<u32>> = missing
                .iter()
                .map(|t| self.tokenize(t))
                .collect::<Result<_, _>>()?;
            let out = self
                .backbone
                .forward(&batch)
                .map_err(|e| format!("backbone forward failed: {e}"))?;
            for (i, text) in missing.iter().enumerate() {
                let emb = out
                    .pooled
                    .i(i)
                    .map_err(|e| format!("cannot slice pooled embedding: {e}"))?;
                self.texts.insert(text.clone(), emb);
            }
        }
        let rows: Vec<Tensor> = texts
            .iter()
            .map(|t| self.texts[t].clone())
            .collect::<Vec<_>>();
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::stack(&refs, 0).map_err(|e| format!("cannot stack embeddings: {e}"))
    }

    /// Distribution over the options of one question, plus `unknown`.
    pub fn score(
        &mut self,
        mem: &StateMemory,
        question: &str,
        options: &[Opt],
    ) -> Result<Decision, String> {
        if options.is_empty() {
            return Err("a question needs at least one option".to_string());
        }
        let q_emb = {
            let stacked = self.embed_texts(&[question.to_string()])?;
            stacked
                .i(0)
                .map_err(|e| format!("cannot slice question embedding: {e}"))?
        };
        let texts: Vec<String> = options.iter().map(|o| o.text.clone()).collect();
        let opt_embs = self.embed_texts(&texts)?;
        let logits = self
            .head
            .forward(&mem.tokens, &mem.mask, &q_emb, &opt_embs)
            .map_err(|e| format!("head forward failed: {e}"))?;
        let logits: Vec<f32> = logits
            .to_vec1()
            .map_err(|e| format!("cannot read logits: {e}"))?;
        Ok(decision_from_logits(options, &logits))
    }

    /// Encode + score in one call, the shape `predict_row` uses per question.
    pub fn decide(
        &mut self,
        state: &str,
        question: &str,
        options: &[Opt],
    ) -> Result<Decision, String> {
        let mem = self.encode_state(state)?;
        self.score(&mem, question, options)
    }
}

use candle_core::IndexOp;

/// Turn `[K + 1]` logits into the reported decision.
pub fn decision_from_logits(options: &[Opt], logits: &[f32]) -> Decision {
    let k = options.len();
    let probs_all = softmax(logits);
    let ids: Vec<String> = options.iter().map(|o| o.id.clone()).collect();
    let probs: Vec<f32> = probs_all[..k].to_vec();
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|a, b| {
        probs[*b]
            .partial_cmp(&probs[*a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let best = probs_all
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(k);
    Decision {
        option_ids: ids.clone(),
        logits: logits[..k].to_vec(),
        probs,
        unknown_logit: logits[k],
        unknown_prob: probs_all[k],
        argmax_id: if best == k {
            UNKNOWN_ID.to_string()
        } else {
            ids[best].clone()
        },
        ranking: order.into_iter().map(|i| ids[i].clone()).collect(),
        k,
    }
}

/// Check a backbone directory against the manifest `model/weights.py` wrote.
fn verify_backbone(dir: &Path, want: &BackboneRef) -> Result<(), String> {
    let path = dir.join("manifest.json");
    let bytes = std::fs::read(&path).map_err(|e| {
        format!(
            "cannot read {}: {e} — run `.venv-train/bin/python -m model.weights fetch`",
            path.display()
        )
    })?;
    let manifest: serde_json::Value =
        serde_json::from_slice(&bytes).map_err(|e| format!("bad {}: {e}", path.display()))?;
    let revision = manifest["revision"].as_str().unwrap_or_default();
    if revision != want.revision {
        return Err(format!(
            "backbone {}: manifest revision {revision} != checkpoint {}",
            want.id, want.revision
        ));
    }
    let files = manifest["files"]
        .as_object()
        .ok_or_else(|| format!("bad {}: no files map", path.display()))?;
    for (name, entry) in files {
        let want_sha = entry["sha256"].as_str().unwrap_or_default();
        require_hash(&dir.join(name), want_sha, &format!("backbone {name}"))?;
    }
    Ok(())
}

/// The state hash half of `jev_runtime::CacheKey`.
pub fn state_hash(state: &str) -> String {
    jev_core::sha256_str(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(n: usize) -> Vec<Opt> {
        (0..n)
            .map(|i| Opt {
                id: format!("o{i}"),
                text: format!("option {i}"),
            })
            .collect()
    }

    #[test]
    fn decision_reports_unknown_when_it_wins() {
        let d = decision_from_logits(&opts(3), &[0.1, 0.2, 0.3, 5.0]);
        assert_eq!(d.argmax_id, UNKNOWN_ID);
        assert_eq!(d.k, 3);
        assert_eq!(d.probs.len(), 3);
        assert!(d.unknown_prob > 0.9);
        assert!((d.probs.iter().sum::<f32>() + d.unknown_prob - 1.0).abs() < 1e-5);
    }

    #[test]
    fn ranking_is_by_descending_probability() {
        let d = decision_from_logits(&opts(3), &[0.1, 2.0, 1.0, -5.0]);
        assert_eq!(d.argmax_id, "o1");
        assert_eq!(d.ranking, vec!["o1", "o2", "o0"]);
    }

    #[test]
    fn state_hash_is_stable_and_content_addressed() {
        assert_eq!(state_hash("abc"), state_hash("abc"));
        assert_ne!(state_hash("abc"), state_hash("abd"));
        assert_eq!(state_hash("abc").len(), 64);
    }
}
