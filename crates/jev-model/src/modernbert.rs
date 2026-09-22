//! ModernBERT encoder on Candle (#T-candle-infer).
//!
//! `model/encoder.py` runs `transformers`' `ModernBertModel` over a state
//! string; this is the same graph in Candle so the Rust runtime stops
//! depending on a Python process. The pieces that have to agree for the
//! parity test to pass, all read off `config.json`:
//!
//! * RoPE on every layer, with the local/global alternation
//!   (`layer_id % global_attn_every_n_layers == 0` is global, the rest see
//!   a `local_attention // 2` sliding window on each side);
//! * GeGLU MLP — `Wi` emits `2 * intermediate_size` and the halves are
//!   `gelu(a) * b`, with the exact erf GELU, not the tanh approximation;
//! * pre-norm residuals, with layer 0's attention norm an identity;
//! * mask-aware mean pooling, the same one `encode_state` returns.
//!
//! Padding is bit-neutral: a pad key is masked out of every attention row
//! and out of the pooling denominator, so a batched forward and a
//! one-row forward return the same numbers for the real tokens.
use std::collections::HashMap;

use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};
use serde::Deserialize;

/// Additive mask value for a forbidden key. Mirrors `torch.finfo(f32).min`.
const MASK_NEG: f32 = f32::MIN;

fn default_eps() -> f64 {
    1e-5
}

/// The subset of HF's `ModernBertConfig` this encoder actually reads.
#[derive(Debug, Clone, Deserialize)]
pub struct Config {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    #[serde(default = "default_eps")]
    pub norm_eps: f64,
    #[serde(default)]
    pub norm_bias: bool,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub mlp_bias: bool,
    pub local_attention: usize,
    pub global_attn_every_n_layers: usize,
    pub global_rope_theta: f64,
    #[serde(default)]
    pub local_rope_theta: Option<f64>,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub pad_token_id: u32,
}

impl Config {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    fn is_global(&self, layer: usize) -> bool {
        self.global_attn_every_n_layers == 0 || layer % self.global_attn_every_n_layers == 0
    }

    /// Half-width of the sliding window, in tokens, on each side.
    pub fn window(&self) -> usize {
        self.local_attention / 2
    }
}

/// Rotary tables for one theta, materialised once per load.
struct Rope {
    cos: Tensor,
    sin: Tensor,
}

impl Rope {
    fn new(head_dim: usize, theta: f64, max_len: usize, dev: &Device) -> Result<Self> {
        let half = head_dim / 2;
        // torch builds `inv_freq` in f32 and multiplies by an f32 position;
        // rounding the frequency the same way keeps us on its grid.
        let inv: Vec<f32> = (0..half)
            .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
            .collect();
        let mut cos = Vec::with_capacity(max_len * head_dim);
        let mut sin = Vec::with_capacity(max_len * head_dim);
        for p in 0..max_len {
            // `emb = cat([freqs, freqs], -1)`: the table repeats twice.
            for _ in 0..2 {
                for f in &inv {
                    let a = p as f64 * f64::from(*f);
                    cos.push(a.cos() as f32);
                    sin.push(a.sin() as f32);
                }
            }
        }
        Ok(Self {
            cos: Tensor::from_vec(cos, (max_len, head_dim), dev)?,
            sin: Tensor::from_vec(sin, (max_len, head_dim), dev)?,
        })
    }

    /// `x` is `[B, heads, T, head_dim]`.
    fn apply(&self, x: &Tensor) -> Result<Tensor> {
        let (_, _, t, d) = x.dims4()?;
        let cos = self.cos.narrow(0, 0, t)?.reshape((1, 1, t, d))?;
        let sin = self.sin.narrow(0, 0, t)?.reshape((1, 1, t, d))?;
        let x1 = x.narrow(D::Minus1, 0, d / 2)?;
        let x2 = x.narrow(D::Minus1, d / 2, d / 2)?;
        let rotated = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        x.broadcast_mul(&cos)?
            .add(&rotated.broadcast_mul(&sin)?)
    }
}

struct Attention {
    wqkv: Linear,
    wo: Linear,
    n_heads: usize,
    head_dim: usize,
    global: bool,
}

impl Attention {
    fn load(cfg: &Config, layer: usize, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let wqkv = load_linear(h, 3 * h, cfg.attention_bias, vb.pp("Wqkv"))?;
        let wo = load_linear(h, h, cfg.attention_bias, vb.pp("Wo"))?;
        Ok(Self {
            wqkv,
            wo,
            n_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            global: cfg.is_global(layer),
        })
    }

    fn forward(&self, x: &Tensor, masks: &Masks, rope: &Rope) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let h = self.n_heads * self.head_dim;
        let qkv = self.wqkv.forward(x)?;
        // `Wqkv`'s output is viewed as (3, heads, head_dim): three
        // contiguous blocks of `hidden_size`, q first.
        let split = |i: usize| -> Result<Tensor> {
            qkv.narrow(D::Minus1, i * h, h)?
                .reshape((b, t, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = rope.apply(&split(0)?)?;
        let k = rope.apply(&split(1)?)?;
        let v = split(2)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let mask = if self.global {
            &masks.global
        } else {
            &masks.local
        };
        let scores = scores.broadcast_add(mask)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, h))?;
        self.wo.forward(&out)
    }
}

struct Mlp {
    wi: Linear,
    wo: Linear,
    intermediate: usize,
}

impl Mlp {
    fn load(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            wi: load_linear(
                cfg.hidden_size,
                2 * cfg.intermediate_size,
                cfg.mlp_bias,
                vb.pp("Wi"),
            )?,
            wo: load_linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                cfg.mlp_bias,
                vb.pp("Wo"),
            )?,
            intermediate: cfg.intermediate_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let wi = self.wi.forward(x)?;
        let input = wi.narrow(D::Minus1, 0, self.intermediate)?;
        let gate = wi.narrow(D::Minus1, self.intermediate, self.intermediate)?;
        self.wo.forward(&input.gelu_erf()?.mul(&gate)?)
    }
}

struct Layer {
    attn_norm: Option<LayerNorm>,
    attn: Attention,
    mlp_norm: LayerNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &Config, layer: usize, vb: VarBuilder) -> Result<Self> {
        // HF makes layer 0's attention norm an `nn.Identity`, so the
        // checkpoint has no `layers.0.attn_norm.weight` to load.
        let attn_norm = if layer == 0 {
            None
        } else {
            Some(load_norm(cfg, vb.pp("attn_norm"))?)
        };
        Ok(Self {
            attn_norm,
            attn: Attention::load(cfg, layer, vb.pp("attn"))?,
            mlp_norm: load_norm(cfg, vb.pp("mlp_norm"))?,
            mlp: Mlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, x: &Tensor, masks: &Masks, rope: &Rope) -> Result<Tensor> {
        let normed = match &self.attn_norm {
            Some(n) => n.forward(x)?,
            None => x.clone(),
        };
        let x = x.add(&self.attn.forward(&normed, masks, rope)?)?;
        let normed = self.mlp_norm.forward(&x)?;
        x.add(&self.mlp.forward(&normed)?)
    }
}

/// Additive attention masks for one batch shape.
struct Masks {
    global: Tensor,
    local: Tensor,
}

fn load_linear(in_dim: usize, out_dim: usize, bias: bool, vb: VarBuilder) -> Result<Linear> {
    if bias {
        candle_nn::linear(in_dim, out_dim, vb)
    } else {
        candle_nn::linear_no_bias(in_dim, out_dim, vb)
    }
}

fn load_norm(cfg: &Config, vb: VarBuilder) -> Result<LayerNorm> {
    let lnc = candle_nn::LayerNormConfig {
        eps: cfg.norm_eps,
        remove_mean: true,
        affine: cfg.norm_bias,
    };
    candle_nn::layer_norm(cfg.hidden_size, lnc, vb)
}

/// One encoded batch: the tensors `encode_state` hands the decision head.
#[derive(Clone)]
pub struct Encoded {
    /// `[B, T, hidden_size]` last hidden states.
    pub tokens: Tensor,
    /// `[B, T]`, 1.0 for a real token and 0.0 for padding.
    pub mask: Tensor,
    /// `[B, hidden_size]` mask-aware mean pooling.
    pub pooled: Tensor,
    pub n_tokens: Vec<usize>,
}

pub struct ModernBert {
    tok_embeddings: Embedding,
    emb_norm: LayerNorm,
    layers: Vec<Layer>,
    final_norm: LayerNorm,
    rope_global: Rope,
    rope_local: Rope,
    cfg: Config,
    device: Device,
}

impl ModernBert {
    /// Load from a `pytorch_model.bin` state dict under the `model.` prefix.
    pub fn load(cfg: Config, vb: VarBuilder, device: &Device, max_len: usize) -> Result<Self> {
        let vb = vb.pp("model");
        let emb = vb.pp("embeddings");
        let tok_embeddings =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, emb.pp("tok_embeddings"))?;
        let emb_norm = load_norm(&cfg, emb.pp("norm"))?;
        let layers_vb = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::load(&cfg, i, layers_vb.pp(i))?);
        }
        let final_norm = load_norm(&cfg, vb.pp("final_norm"))?;
        let head_dim = cfg.head_dim();
        let local_theta = cfg.local_rope_theta.unwrap_or(cfg.global_rope_theta);
        Ok(Self {
            tok_embeddings,
            emb_norm,
            layers,
            final_norm,
            rope_global: Rope::new(head_dim, cfg.global_rope_theta, max_len, device)?,
            rope_local: Rope::new(head_dim, local_theta, max_len, device)?,
            cfg,
            device: device.clone(),
        })
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn hidden_size(&self) -> usize {
        self.cfg.hidden_size
    }

    fn masks(&self, lens: &[usize], t: usize) -> Result<Masks> {
        let b = lens.len();
        let window = self.cfg.window();
        let mut global = vec![0f32; b * t * t];
        let mut local = vec![0f32; b * t * t];
        for (bi, &len) in lens.iter().enumerate() {
            for i in 0..t {
                let row = (bi * t + i) * t;
                for j in 0..t {
                    let pad = j >= len;
                    let far = i.abs_diff(j) > window;
                    global[row + j] = if pad { MASK_NEG } else { 0.0 };
                    local[row + j] = if pad || far { MASK_NEG } else { 0.0 };
                }
            }
        }
        Ok(Masks {
            global: Tensor::from_vec(global, (b, 1, t, t), &self.device)?,
            local: Tensor::from_vec(local, (b, 1, t, t), &self.device)?,
        })
    }

    /// Forward one batch of already-tokenised, already-truncated rows.
    ///
    /// Rows are right-padded to the longest one; every pad position is
    /// masked out of attention and out of the pooling denominator, which
    /// is why a padded batch and a single row agree to the last bit that
    /// floating point allows.
    pub fn forward(&self, batch: &[Vec<u32>]) -> Result<Encoded> {
        if batch.is_empty() {
            candle_core::bail!("modernbert: empty batch");
        }
        let lens: Vec<usize> = batch.iter().map(|ids| ids.len()).collect();
        let t = lens.iter().copied().max().unwrap_or(0).max(1);
        let b = batch.len();
        let pad = self.cfg.pad_token_id;
        let mut ids = Vec::with_capacity(b * t);
        let mut mask = Vec::with_capacity(b * t);
        for row in batch {
            for j in 0..t {
                ids.push(row.get(j).copied().unwrap_or(pad));
                mask.push(if j < row.len() { 1f32 } else { 0f32 });
            }
        }
        let ids = Tensor::from_vec(ids, (b, t), &self.device)?;
        let mask = Tensor::from_vec(mask, (b, t), &self.device)?;
        let masks = self.masks(&lens, t)?;

        let mut x = self.emb_norm.forward(&self.tok_embeddings.forward(&ids)?)?;
        for (i, layer) in self.layers.iter().enumerate() {
            let rope = if self.cfg.is_global(i) {
                &self.rope_global
            } else {
                &self.rope_local
            };
            x = layer.forward(&x, &masks, rope)?;
        }
        let tokens = self.final_norm.forward(&x)?;
        let m = mask.reshape((b, t, 1))?;
        let summed = tokens.broadcast_mul(&m)?.sum(1)?;
        let denom = m.sum(1)?.clamp(1e-6f64, f64::INFINITY)?;
        let pooled = summed.broadcast_div(&denom)?;
        Ok(Encoded {
            tokens,
            mask,
            pooled,
            n_tokens: lens,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Read a `config.json` from a backbone weights directory.
pub fn read_config(path: &std::path::Path) -> std::result::Result<Config, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    serde_json::from_slice(&bytes).map_err(|e| format!("bad {}: {e}", path.display()))
}

/// `pytorch_model.bin` tensors, keyed as in the state dict.
pub fn pth_tensors(path: &std::path::Path) -> std::result::Result<HashMap<String, Tensor>, String> {
    let tensors = candle_core::pickle::read_all(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    Ok(tensors.into_iter().collect())
}

/// A `VarBuilder` over a `.bin` state dict, cast to f32 on `device`.
pub fn pth_var_builder<'a>(
    path: &std::path::Path,
    device: &Device,
) -> std::result::Result<VarBuilder<'a>, String> {
    VarBuilder::from_pth(path, DType::F32, device)
        .map_err(|e| format!("cannot load {}: {e}", path.display()))
}

/// Row `i` of `[B, ...]`, kept as a batch of one.
pub fn row(t: &Tensor, i: usize) -> Result<Tensor> {
    t.narrow(0, i, 1)
}

/// Row `i` of a `[B, D]` matrix, as a `[D]` vector.
pub fn vector(t: &Tensor, i: usize) -> Result<Tensor> {
    t.i(i)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> Config {
        serde_json::from_str(
            r#"{"hidden_size":8,"intermediate_size":4,"num_attention_heads":2,
                "num_hidden_layers":3,"vocab_size":16,"local_attention":4,
                "global_attn_every_n_layers":3,"global_rope_theta":10000.0,
                "max_position_embeddings":32,"pad_token_id":0}"#,
        )
        .unwrap()
    }

    #[test]
    fn global_layers_are_every_nth() {
        let c = cfg();
        assert!(c.is_global(0) && c.is_global(3));
        assert!(!c.is_global(1) && !c.is_global(2));
        assert_eq!(c.window(), 2);
        assert_eq!(c.head_dim(), 4);
    }

    #[test]
    fn rope_tables_repeat_twice() {
        let dev = Device::Cpu;
        let rope = Rope::new(4, 10000.0, 3, &dev).unwrap();
        let cos: Vec<f32> = rope.cos.flatten_all().unwrap().to_vec1().unwrap();
        // `emb = cat([freqs, freqs])`: entry i and i + head_dim/2 agree.
        for p in 0..3 {
            assert!((cos[p * 4] - cos[p * 4 + 2]).abs() < 1e-6);
            assert!((cos[p * 4 + 1] - cos[p * 4 + 3]).abs() < 1e-6);
        }
        // Position 0 is all ones.
        assert!((cos[0] - 1.0).abs() < 1e-7);
    }
}
