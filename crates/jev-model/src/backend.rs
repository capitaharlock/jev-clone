// Local inference backends behind one trait (#T-local-infer).
//
// V1 scope: CPU is the only compiled backend in this workspace. Metal/CUDA
// are represented as `not_available` with a reason, never as faked numbers:
// the gate requires them to stay pending until real hardware validates them.
// `select("auto")` picks the best available backend (CPU today).
use std::fmt;

use super::{probs_f16, probs_f32};

/// Backends this workspace can address.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Metal,
    Cuda,
}

impl Device {
    pub fn as_str(self) -> &'static str {
        match self {
            Device::Cpu => "cpu",
            Device::Metal => "metal",
            Device::Cuda => "cuda",
        }
    }

    /// Whether this backend is compiled in and usable on this machine.
    pub fn is_available(self) -> bool {
        match self {
            Device::Cpu => true,
            Device::Metal | Device::Cuda => false,
        }
    }

    /// Why an unavailable backend cannot run here.
    pub fn unavailable_reason(self) -> Option<&'static str> {
        match self {
            Device::Cpu => None,
            Device::Metal => Some("metal backend not compiled in this workspace build"),
            Device::Cuda => Some("cuda backend not compiled in this workspace build"),
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "cpu" => Some(Device::Cpu),
            "metal" => Some(Device::Metal),
            "cuda" => Some(Device::Cuda),
            _ => None,
        }
    }
}

/// One row of the device inventory (`jevclone devices` prints these).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InventoryEntry {
    pub device: &'static str,
    pub available: bool,
    pub reason: Option<&'static str>,
}

/// Every backend this workspace addresses, present or pending.
pub fn inventory() -> Vec<InventoryEntry> {
    [Device::Cpu, Device::Metal, Device::Cuda]
        .iter()
        .map(|d| InventoryEntry {
            device: d.as_str(),
            available: d.is_available(),
            reason: d.unavailable_reason(),
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unavailable {
    pub device: &'static str,
    pub reason: &'static str,
}

impl fmt::Display for Unavailable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} not_available: {}", self.device, self.reason)
    }
}

impl std::error::Error for Unavailable {}

/// Common inference surface every backend implements (stack §§8–9).
pub trait Backend {
    fn device(&self) -> Device;
    fn name(&self) -> &'static str;
    /// `decide` in FP32: logits = W·x + b, then softmax.
    fn decide(&self, weights: &[f32], bias: &[f32], input: &[f32], n_options: usize) -> Vec<f32>;
    /// FP16-rounded path; the frozen reference before any INT8/ONNX work.
    fn decide_f16(
        &self,
        weights: &[f32],
        bias: &[f32],
        input: &[f32],
        n_options: usize,
    ) -> Vec<f32>;
}

/// The CPU backend (natively executed here, stands in for Candle CPU).
#[derive(Debug)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    fn device(&self) -> Device {
        Device::Cpu
    }

    fn name(&self) -> &'static str {
        "cpu"
    }

    fn decide(&self, weights: &[f32], bias: &[f32], input: &[f32], n_options: usize) -> Vec<f32> {
        probs_f32(weights, bias, input, n_options)
    }

    fn decide_f16(
        &self,
        weights: &[f32],
        bias: &[f32],
        input: &[f32],
        n_options: usize,
    ) -> Vec<f32> {
        probs_f16(weights, bias, input, n_options)
    }
}

/// Resolve `--device auto|cpu|metal|cuda` to a usable backend.
/// `auto` picks the best available one; naming an unavailable backend is an
/// explicit error carrying its reason (never a silent CPU fallback).
pub fn select(requested: &str) -> Result<(Device, CpuBackend), Unavailable> {
    if requested.eq_ignore_ascii_case("auto") {
        return Ok((Device::Cpu, CpuBackend));
    }
    match Device::parse(requested) {
        Some(Device::Cpu) => Ok((Device::Cpu, CpuBackend)),
        Some(d) => Err(Unavailable {
            device: d.as_str(),
            reason: d.unavailable_reason().unwrap_or("unavailable"),
        }),
        None => Err(Unavailable {
            device: "unknown",
            reason: "expected one of: auto, cpu, metal, cuda",
        }),
    }
}

/// `unknown` semantics shared by every backend: the argmax wins only when
/// its mass clears `threshold`, otherwise the answer is `None` (unknown).
/// A threshold at or below 0 disables abstention; at or above 1 always
/// abstains on finite distributions.
pub fn decide_with_unknown(probs: &[f32], threshold: f32) -> Option<usize> {
    let (best, mass) = probs
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
    if *mass >= threshold {
        Some(best)
    } else {
        None
    }
}

/// Rough working-set estimate (MiB) for one loaded state of `dim` × options
/// in FP32/FP16. Documents the memory budget; not a substitute for measuring
/// the real backend allocation on the primary machine.
pub fn memory_budget_mib(dim: usize, n_options: usize, use_f16: bool) -> f64 {
    let bytes_per = if use_f16 { 2.0 } else { 4.0 };
    (n_options * dim + n_options + dim) as f64 * bytes_per / (1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_selects_cpu() {
        let (d, _) = select("auto").expect("auto must resolve");
        assert_eq!(d, Device::Cpu);
    }

    #[test]
    fn metal_and_cuda_stay_not_available() {
        for req in ["metal", "cuda"] {
            let err = select(req).unwrap_err();
            assert!(err.reason.contains("not compiled"), "got {err}");
        }
    }

    #[test]
    fn unknown_request_is_rejected() {
        assert!(select("tpu").is_err());
    }

    #[test]
    fn cpu_decide_matches_production_path() {
        let be = CpuBackend;
        let (w, b, x) = (
            vec![0.5f32, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
            vec![0.1f32, -0.1],
            vec![1.0f32, 0.5, -0.5, 0.25],
        );
        assert_eq!(
            be.decide(&w, &b, &x, 2),
            super::super::probs_f32(&w, &b, &x, 2)
        );
        let d = be
            .decide_f16(&w, &b, &x, 2)
            .iter()
            .zip(super::super::probs_f16(&w, &b, &x, 2).iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(d < 1e-6, "f16 path diverged: {d}");
    }

    #[test]
    fn unknown_abstains_below_threshold() {
        assert_eq!(decide_with_unknown(&[0.6, 0.4], 0.5), Some(0));
        assert_eq!(decide_with_unknown(&[0.6, 0.4], 0.9), None);
        assert!(decide_with_unknown(&[0.5, 0.5], 0.0).is_some());
        assert!(decide_with_unknown(&[], 0.5).is_none());
    }

    #[test]
    fn inventory_lists_cpu_available_only() {
        let inv = inventory();
        assert_eq!(inv.len(), 3);
        let cpu = inv.iter().find(|e| e.device == "cpu").unwrap();
        assert!(cpu.available && cpu.reason.is_none());
        for e in inv.iter().filter(|e| e.device != "cpu") {
            assert!(!e.available && e.reason.is_some());
        }
    }
}
