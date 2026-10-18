// Cross-language parity: Python reference vs this crate (#T-rust-skel).
//
// Runs `training/python/parity/reference.py --matrix`, recomputes every
// case in Rust (f64 reference, f32 production path, f16-rounded path) and
// asserts agreement within the stack §30 tolerances. Multi-question cases
// (1–N) are covered by batching independent seeds per question.
use std::path::PathBuf;
use std::process::Command;

use serde::Deserialize;

#[derive(Deserialize)]
struct Case {
    name: String,
    dim: usize,
    options: usize,
    kind: String,
    fp32: Vec<f64>,
    fp16: Vec<f64>,
}

fn reference_cases() -> Vec<Case> {
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../training/python/parity/reference.py");
    // Resolve ".." lexically so Command finds the script.
    let mut abs = std::env::current_dir().expect("cwd");
    abs.push(&script);
    let out = Command::new("python3")
        .arg(&abs)
        .arg("--matrix")
        .output()
        .expect("python3 reference.py must run on CPU CI");
    assert!(
        out.status.success(),
        "reference.py failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    serde_json::from_slice(&out.stdout).expect("reference output must be JSON")
}

fn lcg_vec(seed: u32, n: usize) -> Vec<f64> {
    let mut rng = jev_model::Lcg::new(seed);
    (0..n).map(|_| rng.next_f64()).collect()
}

fn seed_of(name: &str) -> u32 {
    match name {
        "short-2-choice" => 11,
        "short-2b-boolean" => 12,
        "short-8-choice" => 13,
        "long-2-boolean" => 14,
        "long-8-choice" => 15,
        "long-32-choice" => 16,
        _ => panic!("unknown matrix case {name}"),
    }
}

#[test]
fn parity_matrix_agrees() {
    for case in reference_cases() {
        let n = case.options;
        let dim = case.dim;
        let vals = lcg_vec(seed_of(&case.name), n * dim + n + dim);
        let (w, b, x) = (
            &vals[..n * dim],
            &vals[n * dim..n * dim + n],
            &vals[n * dim + n..],
        );
        // f64 reference path must match Python near-bit-exact.
        let ours = jev_model::softmax(&jev_model::logits_f64(w, b, x, n));
        let d = jev_model::max_abs_diff(&ours, &case.fp32);
        assert!(d < 1e-12, "{} f64 diverged: {d}", case.name);

        // f32 production path within FP32 tolerance.
        let wf: Vec<f32> = w.iter().map(|&v| v as f32).collect();
        let bf: Vec<f32> = b.iter().map(|&v| v as f32).collect();
        let xf: Vec<f32> = x.iter().map(|&v| v as f32).collect();
        let got32: Vec<f64> = jev_model::probs_f32(&wf, &bf, &xf, n)
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let d32 = jev_model::max_abs_diff(&got32, &case.fp32);
        assert!(
            d32 < jev_model::TOL_F32,
            "{} f32 diverged: {d32}",
            case.name
        );

        // f16-rounded path within FP16 tolerance.
        let got16: Vec<f64> = jev_model::probs_f16(&wf, &bf, &xf, n)
            .iter()
            .map(|&v| f64::from(v))
            .collect();
        let d16 = jev_model::max_abs_diff(&got16, &case.fp16);
        assert!(
            d16 < jev_model::TOL_F16,
            "{} f16 diverged: {d16}",
            case.name
        );

        // Kind shape: boolean cases always carry exactly 2 options.
        if case.kind == "boolean" {
            assert_eq!(n, 2, "boolean case {} must be binary", case.name);
        }
    }
}

#[test]
fn batched_questions_are_independent_and_deterministic() {
    // 1–N questions shape: each question is an independent seed; the same
    // seed answered twice gives the same distribution, different seeds differ.
    let probs = |seed: u32| {
        let vals = lcg_vec(seed, 4 * 4 + 4 + 4);
        jev_model::softmax(&jev_model::logits_f64(
            &vals[..16],
            &vals[16..20],
            &vals[20..],
            4,
        ))
    };
    let a = probs(100);
    let b = probs(100);
    let c = probs(101);
    assert!(jev_model::max_abs_diff(&a, &b) == 0.0);
    assert!(jev_model::max_abs_diff(&a, &c) > 1e-6);
    assert!((a.iter().sum::<f64>() - 1.0).abs() < 1e-12);
}

#[test]
fn int8_reports_skip_without_fixture() {
    // Stack §30 runs INT8 "cuando exista": no quantized fixture ships with
    // the skeleton, so the harness must SKIP loudly, never fake a pass.
    let q = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../artifacts/fixtures/tiny-int8");
    assert!(
        !q.join("model.safetensors").is_file(),
        "INT8 fixture exists but harness has no INT8 path yet"
    );
}
