//! RunPod Serverless adapter (#T-cloud-api).
//!
//! Thin envelope translation only: RunPod wraps the worker payload as
//! `{"input": <ChoiceRequest>, "id": <jobId>}` and expects the result as
//! `{"output": <ChoiceResponse|ApiError>}`. The runtime is never coupled
//! to the provider — this module converts envelope ↔ core V1 types and
//! nothing else. Live validation is conditional on credentials; the stub
//! path below runs without them.

use serde_json::Value;

use crate::v1::{ApiError, ChoiceRequest, ChoiceResponse};

/// Split a RunPod envelope into (core request, job id).
pub fn unwrap_envelope(body: &Value) -> Result<(ChoiceRequest, Option<String>), ApiError> {
    let input = body.get("input").ok_or(ApiError {
        code: "bad-envelope".to_string(),
        message: "missing input envelope".to_string(),
    })?;
    let req: ChoiceRequest = serde_json::from_value(input.clone()).map_err(|_| ApiError {
        code: "bad-envelope".to_string(),
        message: "input is not a V1 choice request".to_string(),
    })?;
    let id = body.get("id").and_then(|v| v.as_str()).map(str::to_string);
    Ok((req, id))
}

/// Wrap a successful core response for RunPod.
pub fn wrap_ok(job_id: Option<&str>, resp: &ChoiceResponse) -> Value {
    let mut out = serde_json::json!({ "output": resp });
    if let Some(id) = job_id {
        out["id"] = Value::String(id.to_string());
    }
    out
}

/// Wrap a core error for RunPod (error carries no state by construction).
pub fn wrap_err(job_id: Option<&str>, err: &ApiError) -> Value {
    let mut out = serde_json::json!({ "output": { "error": err } });
    if let Some(id) = job_id {
        out["id"] = Value::String(id.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{new_state, v1};

    fn stub_envelope() -> Value {
        serde_json::json!({
            "id": "job-123",
            "input": {
                "model_version": "m1",
                "options": [{"id": "yes"}, {"id": "no"}],
                "weights": [0.5, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
                "bias": [0.1, -0.1],
                "input": [1.0, 0.5, -0.5, 0.25]
            }
        })
    }

    #[test]
    fn stub_roundtrip_envelope_to_core() {
        let state = new_state();
        let (req, id) = unwrap_envelope(&stub_envelope()).unwrap();
        assert_eq!(id.as_deref(), Some("job-123"));
        let resp = v1::choose(&state, &req).unwrap();
        let wrapped = wrap_ok(id.as_deref(), &resp);
        assert_eq!(wrapped["id"], "job-123");
        assert_eq!(wrapped["output"]["schema"], "v1");
        assert_eq!(
            wrapped["output"]["distribution"].as_array().unwrap().len(),
            2
        );
    }

    #[test]
    fn missing_input_rejected() {
        assert!(unwrap_envelope(&serde_json::json!({"id": "x"})).is_err());
    }

    #[test]
    fn non_v1_input_rejected() {
        let bad = serde_json::json!({"input": {"nonsense": 1}});
        assert!(unwrap_envelope(&bad).is_err());
    }

    #[test]
    fn error_wrap_carries_no_state() {
        let err = ApiError {
            code: "bad-options".to_string(),
            message: "x".to_string(),
        };
        let w = wrap_err(Some("j"), &err).to_string();
        assert!(w.contains("bad-options"));
    }
}
