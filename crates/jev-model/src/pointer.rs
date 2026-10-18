//! The pointer decision head on Candle (#T-candle-infer).
//!
//! A straight port of `model/decision_head.py::PointerDecisionHead`, the
//! head #T-train-real trained: state memory projected once, 1-3
//! cross-attention blocks in which the options read the state and then
//! each other, and a content-based pointer whose key comes from the
//! option's OWN text embedding. The output is `[K + 1]` logits with
//! `unknown` last.
//!
//! No parameter here is shaped by a label count or by K, so the Rust
//! side inherits the label-free property structurally: the weights it
//! loads simply have no `num_labels` axis to load.
//!
//! `nn.MultiheadAttention` stores q/k/v fused in `in_proj_weight`
//! `[3d, d]`; the three `[d, d]` blocks are unpacked at load time so the
//! forward is three ordinary linears, exactly what torch does.
use candle_core::{IndexOp, Result, Tensor, D};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder};

/// Architecture as the checkpoint manifest records it.
#[derive(Debug, Clone, Copy)]
pub struct HeadConfig {
    pub d_state: usize,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

fn norm(d: usize, vb: VarBuilder) -> Result<LayerNorm> {
    candle_nn::layer_norm(
        d,
        candle_nn::LayerNormConfig {
            eps: 1e-5,
            remove_mean: true,
            affine: true,
        },
        vb,
    )
}

/// One `nn.MultiheadAttention(batch_first=True)` layer.
struct MultiheadAttention {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    n_heads: usize,
    head_dim: usize,
}

impl MultiheadAttention {
    fn load(d: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get((3 * d, d), "in_proj_weight")?;
        let b = vb.get(3 * d, "in_proj_bias")?;
        let part = |i: usize| -> Result<Linear> {
            Ok(Linear::new(
                w.narrow(0, i * d, d)?.contiguous()?,
                Some(b.narrow(0, i * d, d)?.contiguous()?),
            ))
        };
        Ok(Self {
            q: part(0)?,
            k: part(1)?,
            v: part(2)?,
            out: candle_nn::linear(d, d, vb.pp("out_proj"))?,
            n_heads,
            head_dim: d / n_heads,
        })
    }

    /// `query` `[B, Q, d]`, `key_value` `[B, S, d]`, `key_mask` an additive
    /// `[B, 1, 1, S]` mask (0 keep, very negative drop).
    fn forward(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        key_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, qlen, d) = query.dims3()?;
        let slen = key_value.dim(1)?;
        let heads = |t: &Tensor, len: usize| -> Result<Tensor> {
            t.reshape((b, len, self.n_heads, self.head_dim))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = heads(&self.q.forward(query)?, qlen)?;
        let k = heads(&self.k.forward(key_value)?, slen)?;
        let v = heads(&self.v.forward(key_value)?, slen)?;
        let scale = (self.head_dim as f64).powf(-0.5);
        let mut scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        if let Some(m) = key_mask {
            scores = scores.broadcast_add(m)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, qlen, d))?;
        self.out.forward(&out)
    }
}

/// One cross-attention block: options read `H_s`, then read each other.
struct CrossBlock {
    ln_cross: LayerNorm,
    cross: MultiheadAttention,
    ln_set: LayerNorm,
    set_attn: MultiheadAttention,
    ln_ff: LayerNorm,
    ff_in: Linear,
    ff_out: Linear,
}

impl CrossBlock {
    fn load(cfg: &HeadConfig, vb: VarBuilder) -> Result<Self> {
        let d = cfg.d_model;
        Ok(Self {
            ln_cross: norm(d, vb.pp("ln_cross"))?,
            cross: MultiheadAttention::load(d, cfg.n_heads, vb.pp("cross"))?,
            ln_set: norm(d, vb.pp("ln_set"))?,
            set_attn: MultiheadAttention::load(d, cfg.n_heads, vb.pp("set_attn"))?,
            ln_ff: norm(d, vb.pp("ln_ff"))?,
            // `nn.Sequential(Linear, GELU, Dropout, Linear)`: indices 0 and 3.
            ff_in: candle_nn::linear(d, 4 * d, vb.pp("ff").pp(0))?,
            ff_out: candle_nn::linear(4 * d, d, vb.pp("ff").pp(3))?,
        })
    }

    fn forward(&self, x: &Tensor, memory: &Tensor, pad: Option<&Tensor>) -> Result<Tensor> {
        let h = self.ln_cross.forward(x)?;
        let x = x.add(&self.cross.forward(&h, memory, pad)?)?;
        let h = self.ln_set.forward(&x)?;
        let mixed = self.set_attn.forward(&h, &h, None)?;
        let x = x.add(&mixed)?;
        let h = self.ln_ff.forward(&x)?;
        let ff = self.ff_out.forward(&self.ff_in.forward(&h)?.gelu_erf()?)?;
        x.add(&ff)
    }
}

pub struct PointerHead {
    state_proj: Linear,
    state_ln: LayerNorm,
    q_proj: Linear,
    opt_proj: Linear,
    blocks: Vec<CrossBlock>,
    ln_out: LayerNorm,
    ptr_query: Linear,
    ptr_key: Linear,
    ptr_bias: Tensor,
    unknown_ln: LayerNorm,
    unknown_in: Linear,
    unknown_out: Linear,
    cfg: HeadConfig,
}

impl PointerHead {
    pub fn load(cfg: HeadConfig, vb: VarBuilder) -> Result<Self> {
        if !(1..=3).contains(&cfg.n_layers) {
            candle_core::bail!("n_layers must be 1..3 (plan: 1-3 layers)");
        }
        let (ds, d) = (cfg.d_state, cfg.d_model);
        let blocks_vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            blocks.push(CrossBlock::load(&cfg, blocks_vb.pp(i))?);
        }
        Ok(Self {
            state_proj: candle_nn::linear(ds, d, vb.pp("state_proj"))?,
            state_ln: norm(d, vb.pp("state_ln"))?,
            q_proj: candle_nn::linear(ds, d, vb.pp("q_proj"))?,
            opt_proj: candle_nn::linear(ds, d, vb.pp("opt_proj"))?,
            blocks,
            ln_out: norm(d, vb.pp("ln_out"))?,
            ptr_query: candle_nn::linear(d, d, vb.pp("ptr_query"))?,
            ptr_key: candle_nn::linear_no_bias(d, d, vb.pp("ptr_key"))?,
            ptr_bias: vb.get(1, "ptr_bias")?,
            unknown_ln: norm(d, vb.pp("unknown_ln"))?,
            unknown_in: candle_nn::linear(d, d, vb.pp("unknown").pp(0))?,
            unknown_out: candle_nn::linear(d, 1, vb.pp("unknown").pp(2))?,
            cfg,
        })
    }

    pub fn config(&self) -> HeadConfig {
        self.cfg
    }

    /// `[K + 1]` logits, `unknown` last.
    ///
    /// * `memory` `[1, T, d_state]` — the state encoded once;
    /// * `memory_mask` `[1, T]` f32, 1.0 for a real token;
    /// * `question_emb` `[d_state]`, `option_embs` `[K, d_state]`.
    pub fn forward(
        &self,
        memory: &Tensor,
        memory_mask: &Tensor,
        question_emb: &Tensor,
        option_embs: &Tensor,
    ) -> Result<Tensor> {
        if option_embs.rank() != 2 {
            candle_core::bail!("option_embs must be [K, d_state]");
        }
        let k = option_embs.dim(0)?;
        if k == 0 {
            candle_core::bail!("a question needs at least one option");
        }
        let d = self.cfg.d_model;
        let t = memory.dim(1)?;
        let mem = self.state_ln.forward(&self.state_proj.forward(memory)?)?;
        // `pad = ~memory_mask`: an additive mask over the memory keys.
        let pad = ((memory_mask.reshape((1, 1, 1, t))? - 1.0)? * 1e30)?;
        let opt = self.opt_proj.forward(option_embs)?; // [K, d]
        let q = self.q_proj.forward(question_emb)?.reshape((1, 1, d))?;
        let mut x = opt.reshape((1, k, d))?.broadcast_add(&q)?;
        for block in &self.blocks {
            x = block.forward(&x, &mem, Some(&pad))?;
        }
        let ctx = self.ln_out.forward(&x)?.i(0)?; // [K, d]
        let keys = self.ptr_key.forward(&opt)?; // key from the option TEXT
        let scores = (self.ptr_query.forward(&ctx)?.mul(&keys)?.sum(D::Minus1)?
            * (d as f64).powf(-0.5))?;
        let scores = scores.broadcast_add(&self.ptr_bias)?; // [K]

        // Permutation-invariant row summary: masked mean over H_s, mean
        // over option contexts, plus the question.
        let m = memory_mask.reshape((1, t, 1))?;
        let denom = m.sum(1)?.clamp(1e-6f64, f64::INFINITY)?;
        let mem_pooled = mem.broadcast_mul(&m)?.sum(1)?.broadcast_div(&denom)?;
        let summary = mem_pooled.i(0)?.add(&ctx.mean(0)?)?.add(&q.reshape(d)?)?;
        let unk = self
            .unknown_out
            .forward(&self.unknown_in.forward(&self.unknown_ln.forward(&summary)?)?.gelu_erf()?)?;
        Tensor::cat(&[&scores, &unk.reshape(1)?], 0)
    }
}

/// Softmax in f32, max-subtracted — the same shape as `torch.softmax`.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_is_a_distribution() {
        let p = softmax(&[3.0, 1.0, -2.0, 0.5]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[0] > p[1] && p[1] > p[3] && p[3] > p[2]);
    }

    #[test]
    fn softmax_survives_large_logits() {
        let p = softmax(&[1000.0, 999.0]);
        assert!(p.iter().all(|x| x.is_finite()));
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
}
