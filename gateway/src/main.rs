use pyo3::prelude::*;
use std::sync::Arc;
use tokio::sync::Mutex;
use axum::{
    extract::State,
    http::StatusCode,
    response::Json,
    routing::{get, post},
    Router,
};
use serde_json::Value;

#[derive(Clone)]
struct EngineHandle {
    _py: Python,
    engine: Arc<Mutex<Option<Py<PyAny>>>>,
}

#[derive(Clone)]
struct AppState {
    engine: EngineHandle,
}

impl EngineHandle {
    fn new() -> anyhow::Result<Self> {
        let py = Python::acquire_gil().python();
        // Add engine directory to sys.path so we can import heyi
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        path.call_method1("append", (concat!(env!("CARGO_MANIFEST_DIR"), "/../engine"),))?;
        Ok(Self {
            _py: py,
            engine: Arc::new(Mutex::new(None)),
        })
    }

    async fn start(&self, model_path: &str) -> anyhow::Result<()> {
        let py = Python::acquire_gil().python();
        let engine_mod = py.import_bound("heyi.engine")?;
        let engine_class = engine_mod.getattr("Engine")?;
        let engine_instance = engine_class.call1((model_path,))?;
        engine_instance.call_method0("boot")?;
        *self.engine.lock().await = Some(engine_instance.into());
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

async fn engine_start(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, StatusCode> {
    let model_path = payload.get("model_path")
        .and_then(|v| v.as_str())
        .ok_or(StatusCode::BAD_REQUEST)?;
    state.engine.start(model_path).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({"status": "started"})))
}

async fn engine_stop(
    State(state): State<AppState>,
) -> Result<Json<Value>, StatusCode> {
    state.engine.stop().await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(Json(serde_json::json!({"status": "stopped"})))
}

async fn engine_status(
    State(state): State<AppState>,
) -> Json<Value> {
    Json(serde_json::json!({"status": state.engine.status().await}))
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = EngineHandle::new()?;
    let state = AppState { engine };

    let app = Router::new()
        .route("/engine/start", post(engine_start))
        .route("/engine/stop", post(engine_stop))
        .route("/engine/status", get(engine_status))
        .with_state(state);

    let addr = "127.0.0.1:8080";
    println!("Gateway listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
