// Axum server: `jevclone serve` boots this (#T-rust-skel, #T-state-cache).
//
// `/infer` answers immediately against the state cache; `/infer_batch`
// queues one HTTP batch as one scheduler batch (a single GPU launch)
// and flushes it before responding.
use axum::{
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

pub mod metrics;
pub mod runpod;
pub mod v1;

pub struct Inner {
    pub runtime: jev_runtime::Runtime,
    pub sched: jev_runtime::BatchScheduler,
}

#[derive(Clone)]
pub struct AppState {
    pub inner: Arc<Mutex<Inner>>,
    pub metrics: Arc<metrics::Metrics>,
}

#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub model_version: String,
    pub state_hash: String,
    pub tokenizer_hash: String,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
    pub input: Vec<f32>,
    pub n_options: usize,
}

#[derive(Debug, Serialize, PartialEq)]
pub struct InferResponse {
    pub probs: Vec<f32>,
    pub cached: bool,
}

#[derive(Debug, Deserialize)]
pub struct BatchRequest {
    pub jobs: Vec<InferRequest>,
}

#[derive(Debug, Serialize)]
pub struct BatchResponse {
    pub results: Vec<Vec<f32>>,
    /// Scheduler batches executed for this HTTP batch: 1 when healthy.
    pub batches: u64,
}

pub fn infer(state: &AppState, req: &InferRequest) -> InferResponse {
    let t0 = std::time::Instant::now();
    let (key, weights, bias, input, n_options) = (
        jev_runtime::CacheKey::new(&req.model_version, &req.state_hash, &req.tokenizer_hash),
        req.weights.clone(),
        req.bias.clone(),
        req.input.clone(),
        req.n_options,
    );
    let (probs, cached) = {
        let mut inner = state.inner.lock().expect("runtime lock");
        inner
            .runtime
            .infer(&key, &weights, &bias, &input, n_options)
    };
    state
        .metrics
        .record(t0.elapsed().as_micros() as u64, cached, false);
    InferResponse { probs, cached }
}

pub fn infer_batch(state: &AppState, req: &BatchRequest) -> Result<BatchResponse, &'static str> {
    let mut inner = state.inner.lock().expect("runtime lock");
    let Inner { runtime: rt, sched } = &mut *inner;
    let before = sched.batches;
    let mut handles = Vec::with_capacity(req.jobs.len());
    for j in &req.jobs {
        let job = jev_runtime::Job {
            key: jev_runtime::CacheKey::new(&j.model_version, &j.state_hash, &j.tokenizer_hash),
            fingerprint: format!(
                "{}:{}:{}:{}:{}:{}",
                j.model_version,
                j.state_hash,
                j.tokenizer_hash,
                j.weights.len(),
                j.bias.len(),
                j.input.len(),
            ),
            critical: false,
            weights: j.weights.clone(),
            bias: j.bias.clone(),
            input: j.input.clone(),
            n_options: j.n_options,
        };
        match sched.submit(rt, job) {
            jev_runtime::Submit::Queued(h) => handles.push(h),
            jev_runtime::Submit::Immediate { .. } => return Err("unexpected immediate"),
            jev_runtime::Submit::Rejected { reason } => return Err(reason),
        }
    }
    sched.flush(rt)?;
    let mut results = Vec::with_capacity(handles.len());
    for h in &handles {
        results.push(h.wait()?);
    }
    Ok(BatchResponse {
        results,
        batches: sched.batches - before,
    })
}

async fn health() -> &'static str {
    "ok"
}

async fn ready(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> Json<serde_json::Value> {
    // Readiness self-check: fixed dummy vectors, reserved key. The body
    // reports capability only — never state.
    let key = jev_runtime::CacheKey::new("__ready__", "__ready__", "__ready__");
    let ok = {
        let mut inner = state.inner.lock().expect("runtime lock");
        let w = vec![0.25f32; 8];
        let b = vec![0.0f32; 2];
        let x = vec![0.5f32; 4];
        inner.runtime.infer(&key, &w, &b, &x, 2).0.len() == 2
    };
    let device = state
        .inner
        .lock()
        .expect("runtime lock")
        .runtime
        .device_name();
    Json(serde_json::json!({"ready": ok, "device": device, "schema": v1::SCHEMA}))
}

fn api_status(err: &v1::ApiError) -> StatusCode {
    match err.code.as_str() {
        "too-large" => StatusCode::PAYLOAD_TOO_LARGE,
        "bad-batch" => StatusCode::UNPROCESSABLE_ENTITY,
        "bad-envelope" => StatusCode::BAD_REQUEST,
        _ => StatusCode::UNPROCESSABLE_ENTITY,
    }
}

async fn choice_route(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<v1::ChoiceRequest>,
) -> Result<Json<v1::ChoiceResponse>, (StatusCode, Json<v1::ApiError>)> {
    let t0 = std::time::Instant::now();
    let fut = async { v1::choose(&state, &req) };
    match tokio::time::timeout(std::time::Duration::from_millis(v1::QUERY_TIMEOUT_MS), fut).await {
        Ok(Ok(r)) => {
            state
                .metrics
                .record(t0.elapsed().as_micros() as u64, false, false);
            Ok(Json(r))
        }
        Ok(Err(e)) => {
            state
                .metrics
                .record(t0.elapsed().as_micros() as u64, false, true);
            Err((api_status(&e), Json(e)))
        }
        Err(_) => {
            state
                .metrics
                .record(t0.elapsed().as_micros() as u64, false, true);
            Err((
                StatusCode::REQUEST_TIMEOUT,
                Json(v1::ApiError {
                    code: "timeout".to_string(),
                    message: "query budget exceeded".to_string(),
                }),
            ))
        }
    }
}

async fn batch_choice_route(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<v1::BatchChoiceRequest>,
) -> Result<Json<v1::BatchChoiceResponse>, (StatusCode, Json<v1::ApiError>)> {
    let t0 = std::time::Instant::now();
    match v1::choose_batch(&state, &req) {
        Ok(r) => {
            state
                .metrics
                .record(t0.elapsed().as_micros() as u64, false, false);
            Ok(Json(r))
        }
        Err(e) => {
            state
                .metrics
                .record(t0.elapsed().as_micros() as u64, false, true);
            Err((api_status(&e), Json(e)))
        }
    }
}

async fn runpod_route(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (req, id) = match runpod::unwrap_envelope(&body) {
        Ok(t) => t,
        Err(e) => return (StatusCode::BAD_REQUEST, Json(runpod::wrap_err(None, &e))),
    };
    match v1::choose(&state, &req) {
        Ok(r) => (StatusCode::OK, Json(runpod::wrap_ok(id.as_deref(), &r))),
        Err(e) => (api_status(&e), Json(runpod::wrap_err(id.as_deref(), &e))),
    }
}

async fn infer_route(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    Json(infer(&state, &req))
}

async fn batch_route(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<BatchRequest>,
) -> Result<Json<BatchResponse>, StatusCode> {
    match infer_batch(&state, &req) {
        Ok(r) => Ok(Json(r)),
        Err("queue-full" | "gpu-busy") => Err(StatusCode::TOO_MANY_REQUESTS),
        Err(_) => Err(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

async fn metrics_route(
    axum::extract::State(state): axum::extract::State<AppState>,
) -> ([(&'static str, &'static str); 1], String) {
    (
        [("content-type", "text/plain; version=0.0.4")],
        state.metrics.render(),
    )
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/ready", get(ready))
        .route("/metrics", get(metrics_route))
        .route("/infer", post(infer_route))
        .route("/infer_batch", post(batch_route))
        .route("/v1/choice", post(choice_route))
        .route("/v1/batch", post(batch_choice_route))
        .route("/runpod", post(runpod_route))
        .with_state(state)
}

pub fn new_state() -> AppState {
    AppState {
        inner: Arc::new(Mutex::new(Inner {
            runtime: jev_runtime::Runtime::new(),
            sched: jev_runtime::BatchScheduler::default(),
        })),
        metrics: Arc::new(metrics::Metrics::default()),
    }
}

/// Bind + serve forever; returns only on error.
pub async fn serve(listen: &str) -> Result<(), String> {
    let listener = tokio::net::TcpListener::bind(listen)
        .await
        .map_err(|e| format!("cannot bind {listen}: {e}"))?;
    axum::serve(listener, router(new_state()))
        .await
        .map_err(|e| format!("server error: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(state_hash: &str) -> InferRequest {
        InferRequest {
            model_version: "v1".to_string(),
            state_hash: state_hash.to_string(),
            tokenizer_hash: "tok1".to_string(),
            weights: vec![0.5, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
            bias: vec![0.1, -0.1],
            input: vec![1.0, 0.5, -0.5, 0.25],
            n_options: 2,
        }
    }

    #[test]
    fn infer_and_cache_flag() {
        let state = new_state();
        let r = req("s");
        let first = infer(&state, &r);
        assert!(!first.cached);
        let second = infer(&state, &r);
        assert!(second.cached);
        assert_eq!(first.probs, second.probs);
    }

    #[test]
    fn batch_runs_as_one_gpu_launch() {
        let state = new_state();
        let jobs = vec![req("shared"), req("shared"), req("shared")];
        let out = infer_batch(&state, &BatchRequest { jobs }).expect("batch");
        assert_eq!(out.results.len(), 3);
        assert_eq!(out.batches, 1, "one HTTP batch must be one GPU launch");
        assert_eq!(out.results[0], out.results[1]);
        assert_eq!(out.results[1], out.results[2]);
    }

    #[test]
    fn router_builds() {
        let _ = router(new_state());
    }
}
