// Moved from main.rs to allow reuse by backend.

use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use pyo3::prelude::*;
use serde_json::Value;
use std::{net::SocketAddr, sync::Arc};
use tokio::sync::Mutex;

#[derive(Clone)]
struct EngineHandle {
    engine: Arc<Mutex<Option<Py<PyAny>>>>,
}

#[derive(Clone)]
struct AppState {
    engine: EngineHandle,
}

impl EngineHandle {
    fn new() -> anyhow::Result<Self> {
        Python::with_gil(|py| -> anyhow::Result<Self> {
            // Add engine directory to sys.path so we can import heyi
            let sys = py.import_bound("sys")?;
            let path = sys.getattr("path")?;
            path.call_method1("append", (concat!(env!("CARGO_MANIFEST_DIR"), "/../engine"),))?;
            Ok(Self {
                engine: Arc::new(Mutex::new(None)),
            })
        })
    }

    async fn start(&self, model_path: &str) -> anyhow::Result<()> {
        let engine_instance = Python::with_gil(|py| -> anyhow::Result<Py<PyAny>> {
            let engine_mod = py.import_bound("heyi.engine")?;
            let engine_class = engine_mod.getattr("Engine")?;
            let engine_instance = engine_class.call1((model_path,))?;
            engine_instance.call_method0("boot")?;
            Ok(engine_instance.into())
        })?;
        *self.engine.lock().await = Some(engine_instance);
        Ok(())
    }

    async fn stop(&self) -> anyhow::Result<()> {
        *self.engine.lock().await = None;
        Ok(())
    }

    async fn status(&self) -> &'static str {
        if self.engine.lock().await.is_some() {
            "running"
        } else {
            "stopped"
        }
    }
}

async fn engine_start(State(state): State<AppState>, Json(payload): Json<Value>) -> Result<Json<Value>, StatusCode> {
    let model_path = payload
        .get("model_path")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;
    state
        .engine
        .start(model_path)
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({"status": "started"})))
}

async fn engine_stop(State(state): State<AppState>) -> Result<Json<Value>, StatusCode> {
    state
        .engine
        .stop()
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({"status": "stopped"})))
}

async fn engine_status(State(state): State<AppState>) -> Json<Value> {
    Json(serde_json::json!({"status": state.engine.status().await}))
}

pub async fn serve(addr: SocketAddr) -> anyhow::Result<()> {
    let engine = EngineHandle::new()?;
    let state = AppState { engine };

    let app = Router::new()
        .route("/engine/start", post(engine_start))
        .route("/engine/stop", post(engine_stop))
        .route("/engine/status", get(engine_status))
        .with_state(state);

    println!("gateway listening on {addr}");
    axum::serve(tokio::net::TcpListener::bind(addr).await?, app).await?;
    Ok(())
}
