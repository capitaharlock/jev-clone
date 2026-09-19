// Axum server skeleton: `jevclone serve` boots this (#T-rust-skel).
use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct AppState {
    pub runtime: Arc<Mutex<jev_runtime::Runtime>>,
}

#[derive(Debug, Deserialize)]
pub struct InferRequest {
    pub state_key: String,
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

pub fn infer(state: &AppState, req: &InferRequest) -> InferResponse {
    let mut rt = state.runtime.lock().expect("runtime lock");
    let before = rt.hits;
    let probs = rt.infer(
        &req.state_key,
        &req.weights,
        &req.bias,
        &req.input,
        req.n_options,
    );
    InferResponse {
        probs,
        cached: rt.hits > before,
    }
}

async fn health() -> &'static str {
    "ok"
}

async fn infer_route(
    axum::extract::State(state): axum::extract::State<AppState>,
    Json(req): Json<InferRequest>,
) -> Json<InferResponse> {
    Json(infer(&state, &req))
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/infer", post(infer_route))
        .with_state(state)
}

pub fn new_state() -> AppState {
    AppState {
        runtime: Arc::new(Mutex::new(jev_runtime::Runtime::new())),
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

    #[test]
    fn infer_and_cache_flag() {
        let state = new_state();
        let req = InferRequest {
            state_key: "s".to_string(),
            weights: vec![0.5, -0.25, 0.1, 0.0, 0.2, 0.3, -0.1, 0.4],
            bias: vec![0.1, -0.1],
            input: vec![1.0, 0.5, -0.5, 0.25],
            n_options: 2,
        };
        let first = infer(&state, &req);
        assert!(!first.cached);
        let second = infer(&state, &req);
        assert!(second.cached);
        assert_eq!(first.probs, second.probs);
    }

    #[test]
    fn router_builds() {
        let _ = router(new_state());
    }
}
