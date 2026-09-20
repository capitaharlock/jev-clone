//! Public decision API V1 (#T-cloud-api).
//!
//! V1 covers `choice` (any N, including binary = `boolean`) plus
//! `unknown` via abstention threshold; `score` is deferred to V2.
//! Contract rules enforced here:
//! - schema/version echoed, order + option IDs preserved;
//! - errors never echo state (no weights/input/hashes in bodies);
//! - hard limits on options, batch size and vector length.

use serde::{Deserialize, Serialize};

use crate::{AppState, InferRequest};

pub const SCHEMA: &str = "v1";
pub const MAX_OPTIONS: usize = 128;
pub const MAX_BATCH: usize = 64;
pub const MAX_VEC: usize = 1 << 20;
pub const DEFAULT_THRESHOLD: f32 = 0.5;
/// Handler-level compute budget per query.
pub const QUERY_TIMEOUT_MS: u64 = 30_000;

#[derive(Debug, Clone, Deserialize)]
pub struct ApiOption {
    pub id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChoiceRequest {
    pub model_version: String,
    #[serde(default)]
    pub schema: Option<String>,
    pub options: Vec<ApiOption>,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub input: Vec<f32>,
    #[serde(default)]
    pub threshold: Option<f32>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ScoredOption {
    pub id: String,
    pub prob: f32,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ChoiceResponse {
    pub schema: String,
    pub model_version: String,
    pub distribution: Vec<ScoredOption>,
    pub choice: Option<String>,
    pub unknown: bool,
    pub cached: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BatchChoiceRequest {
    #[serde(default)]
    pub schema: Option<String>,
    pub queries: Vec<ChoiceRequest>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct BatchChoiceResponse {
    pub schema: String,
    pub results: Vec<BatchItem>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "status", rename_all = "lowercase")]
pub enum BatchItem {
    Ok(ChoiceResponse),
    Err(ApiError),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ApiError {
    pub code: String,
    pub message: String,
}

impl ApiError {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.to_string(),
            message: message.into(),
        }
    }
}

fn check_schema(schema: &Option<String>) -> Result<(), ApiError> {
    match schema {
        None => Ok(()),
        Some(s) if s == SCHEMA => Ok(()),
        Some(_) => Err(ApiError::new("bad-schema", "unsupported schema version")),
    }
}

fn check_vec(name: &str, v: &[f32]) -> Result<(), ApiError> {
    if v.is_empty() {
        return Err(ApiError::new("bad-vector", "empty vector"));
    }
    if v.len() > MAX_VEC {
        return Err(ApiError::new("too-large", "vector exceeds limit"));
    }
    if v.iter().any(|x| !x.is_finite()) {
        return Err(ApiError::new("bad-vector", "non-finite value"));
    }
    let _ = name;
    Ok(())
}

/// Validate without echoing any state back in the error.
pub fn validate(req: &ChoiceRequest) -> Result<f32, ApiError> {
    check_schema(&req.schema)?;
    if req.model_version.is_empty() || req.model_version.len() > 64 {
        return Err(ApiError::new("bad-version", "invalid model_version"));
    }
    if req.options.is_empty() || req.options.len() > MAX_OPTIONS {
        return Err(ApiError::new("bad-options", "options count out of range"));
    }
    let mut seen = std::collections::HashSet::new();
    for o in &req.options {
        if o.id.is_empty() || o.id.len() > 128 || !seen.insert(o.id.as_str()) {
            return Err(ApiError::new(
                "bad-options",
                "invalid or duplicate option id",
            ));
        }
    }
    check_vec("weights", &req.weights)?;
    check_vec("bias", &req.bias)?;
    check_vec("input", &req.input)?;
    match req.threshold {
        None => Ok(DEFAULT_THRESHOLD),
        Some(t) if (0.0..=1.0).contains(&t) => Ok(t),
        Some(_) => Err(ApiError::new("bad-threshold", "threshold must be in [0,1]")),
    }
}

pub fn choose(state: &AppState, req: &ChoiceRequest) -> Result<ChoiceResponse, ApiError> {
    let threshold = validate(req)?;
    // Internal keys only; never leave this function in an error body.
    let inner = InferRequest {
        model_version: req.model_version.clone(),
        state_hash: String::new(),
        tokenizer_hash: String::new(),
        weights: req.weights.clone(),
        bias: req.bias.clone(),
        input: req.input.clone(),
        n_options: req.options.len(),
    };
    let out = crate::infer(state, &inner);
    let idx = jev_model::backend::decide_with_unknown(&out.probs, threshold);
    let distribution = req
        .options
        .iter()
        .zip(out.probs.iter())
        .map(|(o, p)| ScoredOption {
            id: o.id.clone(),
            prob: *p,
        })
        .collect();
    Ok(ChoiceResponse {
        schema: SCHEMA.to_string(),
        model_version: req.model_version.clone(),
        distribution,
        choice: idx.map(|i| req.options[i].id.clone()),
        unknown: idx.is_none(),
        cached: out.cached,
    })
}

pub fn choose_batch(
    state: &AppState,
    req: &BatchChoiceRequest,
) -> Result<BatchChoiceResponse, ApiError> {
    check_schema(&req.schema)?;
    if req.queries.is_empty() || req.queries.len() > MAX_BATCH {
        return Err(ApiError::new("bad-batch", "batch size out of range"));
    }
    let results = req
        .queries
        .iter()
        .map(|q| match choose(state, q) {
            Ok(r) => BatchItem::Ok(r),
            Err(e) => BatchItem::Err(e),
        })
        .collect();
    Ok(BatchChoiceResponse {
        schema: SCHEMA.to_string(),
        results,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::new_state;

    fn req() -> ChoiceRequest {
        ChoiceRequest {
            model_version: "m1".to_string(),
            schema: None,
            options: vec![
                ApiOption {
                    id: "a".to_string(),
                },
                ApiOption {
                    id: "b".to_string(),
                },
            ],
            weights: vec![0.5, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
            bias: vec![0.1, -0.1],
            input: vec![1.0, 0.5, -0.5, 0.25],
            threshold: None,
        }
    }

    #[test]
    fn choice_preserves_order_and_ids() {
        let s = new_state();
        let r = choose(&s, &req()).unwrap();
        assert_eq!(r.schema, "v1");
        assert_eq!(r.model_version, "m1");
        assert_eq!(r.distribution.len(), 2);
        assert_eq!(r.distribution[0].id, "a");
        assert_eq!(r.distribution[1].id, "b");
        let sum: f32 = r.distribution.iter().map(|d| d.prob).sum();
        assert!((sum - 1.0).abs() < 1e-5);
        assert_eq!(r.unknown, r.choice.is_none());
    }

    #[test]
    fn unknown_when_threshold_impossible() {
        let s = new_state();
        let mut q = req();
        q.threshold = Some(1.0);
        let r = choose(&s, &q).unwrap();
        assert!(r.unknown);
        assert_eq!(r.choice, None);
    }

    #[test]
    fn boolean_is_binary_choice() {
        let s = new_state();
        let mut q = req();
        q.threshold = Some(0.0);
        let r = choose(&s, &q).unwrap();
        assert!(!r.unknown);
        assert!(r.choice == Some("a".to_string()) || r.choice == Some("b".to_string()));
    }

    #[test]
    fn bad_schema_rejected() {
        let s = new_state();
        let mut q = req();
        q.schema = Some("v9".to_string());
        assert_eq!(choose(&s, &q).unwrap_err().code, "bad-schema");
    }

    #[test]
    fn dup_ids_rejected_without_state_echo() {
        let s = new_state();
        let mut q = req();
        q.options[1].id = "a".to_string();
        let e = choose(&s, &q).unwrap_err();
        assert_eq!(e.code, "bad-options");
        let body = serde_json::to_string(&e).unwrap();
        assert!(!body.contains("0.5"));
    }

    #[test]
    fn empty_options_and_oversize_batch() {
        let s = new_state();
        let mut q = req();
        q.options.clear();
        assert!(choose(&s, &q).is_err());
        let big = BatchChoiceRequest {
            schema: None,
            queries: vec![req(); MAX_BATCH + 1],
        };
        assert_eq!(choose_batch(&s, &big).unwrap_err().code, "bad-batch");
    }

    #[test]
    fn batch_partial_failure_keeps_order() {
        let s = new_state();
        let mut bad = req();
        bad.options.clear();
        let r = choose_batch(
            &s,
            &BatchChoiceRequest {
                schema: None,
                queries: vec![req(), bad],
            },
        )
        .unwrap();
        assert!(matches!(r.results[0], BatchItem::Ok(_)));
        assert!(matches!(r.results[1], BatchItem::Err(_)));
    }

    #[test]
    fn nonfinite_rejected() {
        let s = new_state();
        let mut q = req();
        q.weights[0] = f32::NAN;
        assert_eq!(choose(&s, &q).unwrap_err().code, "bad-vector");
    }
}
