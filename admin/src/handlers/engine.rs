use axum::{
    extract::State,
    response::Json as JsonResponse,
    http::StatusCode,
    Json,
};
use serde::Serialize;
use std::process::{Command, Child};
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::models::{ApiResponse, EngineControlRequest, EngineConfig, EngineAction};
use crate::services::AppState;

pub struct EngineProcess {
    pub process: Option<Child>,
    pub config: EngineConfig,
}

impl EngineProcess {
    pub fn new(config: EngineConfig) -> Self {
        Self {
            process: None,
            config,
        }
    }
}

#[derive(Clone)]
pub struct EngineManager {
    pub process: Arc<RwLock<EngineProcess>>,
}

impl EngineManager {
    pub fn new() -> Self {
        let config = EngineConfig::default();
        Self {
            process: Arc::new(RwLock::new(EngineProcess::new(config))),
        }
    }

    pub async fn start_engine(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut process_guard = self.process.write().await;

        // Check if already running
        if process_guard.process.is_some() {
            return Ok(());
        }

        // Build the command to start the engine
        let mut cmd = Command::new("python");
        cmd.arg("server/main.py")
           .arg(&process_guard.config.model_path)
           .arg("--model-name")
           .arg(&process_guard.config.model_name)
           .arg("--host")
           .arg(&process_guard.config.host)
           .arg("--port")
           .arg(process_guard.config.port.to_string())
           .arg("--num-cpu-threads")
           .arg(process_guard.config.num_cpu_threads.to_string())
           .arg("--max-batch-size")
           .arg(process_guard.config.max_batch_size.to_string());

        if let Some(api_key) = &process_guard.config.api_key {
            cmd.arg("--api-key").arg(api_key);
        }

        // Set working directory to the project root
        cmd.current_dir("/Users/xinference/Resilio Sync/Heyi/heyipython/heyi");

        // Set PYTHONPATH
        cmd.env("PYTHONPATH", "/Users/xinference/Resilio Sync/Heyi/heyipython");

        // Start the process
        let child = cmd.spawn()?;
        process_guard.process = Some(child);

        Ok(())
    }

    pub async fn stop_engine(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut process_guard = self.process.write().await;

        if let Some(mut child) = process_guard.process.take() {
            child.kill()?;
            child.wait()?;
        }

        Ok(())
    }

    pub async fn restart_engine(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.stop_engine().await?;
        self.start_engine().await?;
        Ok(())
    }

    pub async fn get_status(&self) -> EngineProcessStatus {
        let process_guard = self.process.read().await;

        EngineProcessStatus {
            is_running: process_guard.process.is_some(),
            config: process_guard.config.clone(),
        }
    }
}

#[derive(Serialize)]
pub struct EngineProcessStatus {
    pub is_running: bool,
    pub config: EngineConfig,
}

pub async fn control_engine(
    State(state): State<AppState>,
    Json(payload): Json<EngineControlRequest>,
) -> Result<JsonResponse<ApiResponse<String>>, StatusCode> {
    let manager = &state.engine_manager;

    match payload.action {
        EngineAction::Start => {
            match manager.start_engine().await {
                Ok(_) => Ok(JsonResponse(ApiResponse {
                    success: true,
                    message: "Engine started successfully".to_string(),
                    data: Some("started".to_string()),
                })),
                Err(e) => Ok(JsonResponse(ApiResponse {
                    success: false,
                    message: format!("Failed to start engine: {}", e),
                    data: None,
                })),
            }
        }
        EngineAction::Stop => {
            match manager.stop_engine().await {
                Ok(_) => Ok(JsonResponse(ApiResponse {
                    success: true,
                    message: "Engine stopped successfully".to_string(),
                    data: Some("stopped".to_string()),
                })),
                Err(e) => Ok(JsonResponse(ApiResponse {
                    success: false,
                    message: format!("Failed to stop engine: {}", e),
                    data: None,
                })),
            }
        }
        EngineAction::Restart => {
            match manager.restart_engine().await {
                Ok(_) => Ok(JsonResponse(ApiResponse {
                    success: true,
                    message: "Engine restarted successfully".to_string(),
                    data: Some("restarted".to_string()),
                })),
                Err(e) => Ok(JsonResponse(ApiResponse {
                    success: false,
                    message: format!("Failed to restart engine: {}", e),
                    data: None,
                })),
            }
        }
        _ => Err(StatusCode::BAD_REQUEST),
    }
}

pub async fn get_engine_config(
    State(state): State<AppState>,
) -> JsonResponse<EngineConfig> {
    let status = state.engine_manager.get_status().await;
    JsonResponse(status.config)
}

pub async fn update_engine_config(
    State(state): State<AppState>,
    Json(config): Json<EngineConfig>,
) -> JsonResponse<ApiResponse<String>> {
    let mut process_guard = state.engine_manager.process.write().await;
    process_guard.config = config.clone();

    JsonResponse(ApiResponse {
        success: true,
        message: "Engine configuration updated".to_string(),
        data: Some("updated".to_string()),
    })
}