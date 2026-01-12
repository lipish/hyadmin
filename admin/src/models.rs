use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub path: String,
    pub model_type: ModelType,
    pub status: ModelStatus,
    pub loaded_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
    pub config: ModelConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelType {
    DeepSeekV2,
    DeepSeekV3,
    Qwen3Moe,
    Other(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Unloaded,
    Loading,
    Loaded,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelConfig {
    pub max_length: Option<usize>,
    pub max_new_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiEndpoint {
    pub id: String,
    pub name: String,
    pub api_type: ApiType,
    pub base_url: String,
    pub enabled: bool,
    pub config: ApiConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApiType {
    OpenAI,
    Anthropic,
    Codex,
    OpenCode,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub model_name: String,
    pub api_key: Option<String>,
    pub max_tokens: Option<usize>,
    pub timeout: Option<u64>,
    pub retry_count: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineStatus {
    pub state: EngineState,
    pub uptime: Option<u64>,
    pub request_count: u64,
    pub error_count: u64,
    pub throughput: Option<f32>,
    pub memory_usage: Option<f64>,
    pub gpu_usage: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineState {
    Stopped,
    Starting,
    Running,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f32,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_rx: u64,
    pub network_tx: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestLog {
    pub id: String,
    pub timestamp: DateTime<Utc>,
    pub method: String,
    pub url: String,
    pub status_code: u16,
    pub response_time: u64,
    pub client_ip: String,
    pub user_agent: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: String,
    pub username: String,
    pub role: UserRole,
    pub created_at: DateTime<Utc>,
    pub last_login: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UserRole {
    Admin,
    Operator,
    Viewer,
}

// Request/Response types for API communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadModelRequest {
    pub model_path: String,
    pub model_name: Option<String>,
    pub config: Option<ModelConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineControlRequest {
    pub action: EngineAction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EngineAction {
    Start,
    Stop,
    Restart,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub message: String,
    pub data: Option<T>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EngineConfig {
    pub model_path: String,
    pub model_name: String,
    pub host: String,
    pub port: u16,
    pub num_cpu_threads: usize,
    pub max_batch_size: usize,
    pub api_key: Option<String>,
    pub max_length: Option<usize>,
    pub max_new_tokens: Option<usize>,
    pub prefill_chunk_size: Option<usize>,
    pub use_cuda_graph: Option<bool>,
}
