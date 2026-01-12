use axum::{
    extract::{Path, Json},
    response::{Json as JsonResponse},
    http::StatusCode,
};
use uuid::Uuid;

use crate::models::{ApiEndpoint, ApiType, ApiConfig, ApiResponse};

// Static data for demo purposes
static mut APIS: Vec<ApiEndpoint> = Vec::new();

pub async fn list_apis() -> JsonResponse<Vec<ApiEndpoint>> {
    unsafe {
        if APIS.is_empty() {
            // Initialize with some demo data
            APIS.push(ApiEndpoint {
                id: "demo-api-1".to_string(),
                name: "OpenAI Compatible".to_string(),
                api_type: ApiType::OpenAI,
                base_url: "/v1".to_string(),
                enabled: true,
                config: ApiConfig {
                    model_name: "deepseek-chat".to_string(),
                    api_key: None,
                    max_tokens: Some(4096),
                    timeout: Some(30),
                    retry_count: Some(3),
                },
            });
        }
        JsonResponse(APIS.clone())
    }
}

pub async fn get_api(
    Path(id): Path<String>,
) -> Result<JsonResponse<ApiEndpoint>, StatusCode> {
    unsafe {
        if let Some(api) = APIS.iter().find(|a| a.id == id) {
            Ok(JsonResponse(api.clone()))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn create_api(
    Json(api_data): Json<serde_json::Value>,
) -> Result<JsonResponse<ApiEndpoint>, StatusCode> {
    let api = ApiEndpoint {
        id: Uuid::new_v4().to_string(),
        name: api_data["name"].as_str().unwrap_or("Unnamed API").to_string(),
        api_type: match api_data["api_type"].as_str() {
            Some("openai") => ApiType::OpenAI,
            Some("anthropic") => ApiType::Anthropic,
            Some("codex") => ApiType::Codex,
            Some("opencode") => ApiType::OpenCode,
            _ => ApiType::Custom("unknown".to_string()),
        },
        base_url: api_data["base_url"].as_str().unwrap_or("").to_string(),
        enabled: api_data["enabled"].as_bool().unwrap_or(false),
        config: ApiConfig {
            model_name: api_data["config"]["model_name"].as_str().unwrap_or("").to_string(),
            api_key: api_data["config"]["api_key"].as_str().map(|s| s.to_string()),
            max_tokens: api_data["config"]["max_tokens"].as_u64().map(|n| n as usize),
            timeout: api_data["config"]["timeout"].as_u64(),
            retry_count: api_data["config"]["retry_count"].as_u64().map(|n| n as usize),
        },
    };

    unsafe {
        APIS.push(api.clone());
    }

    Ok(JsonResponse(api))
}

pub async fn update_api(
    Path(id): Path<String>,
    Json(update_data): Json<serde_json::Value>,
) -> Result<JsonResponse<ApiEndpoint>, StatusCode> {
    unsafe {
        if let Some(api) = APIS.iter_mut().find(|a| a.id == id) {
            if let Some(name) = update_data["name"].as_str() {
                api.name = name.to_string();
            }
            if let Some(base_url) = update_data["base_url"].as_str() {
                api.base_url = base_url.to_string();
            }
            if let Some(enabled) = update_data["enabled"].as_bool() {
                api.enabled = enabled;
            }
            Ok(JsonResponse(api.clone()))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn delete_api(
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    unsafe {
        if let Some(pos) = APIS.iter().position(|a| a.id == id) {
            APIS.remove(pos);
            Ok(StatusCode::NO_CONTENT)
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn enable_api(
    Path(id): Path<String>,
) -> Result<JsonResponse<ApiResponse<ApiEndpoint>>, StatusCode> {
    unsafe {
        if let Some(api) = APIS.iter_mut().find(|a| a.id == id) {
            api.enabled = true;
            let response = ApiResponse {
                success: true,
                message: format!("API {} enabled", api.name),
                data: Some(api.clone()),
            };
            Ok(JsonResponse(response))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn disable_api(
    Path(id): Path<String>,
) -> Result<JsonResponse<ApiResponse<ApiEndpoint>>, StatusCode> {
    unsafe {
        if let Some(api) = APIS.iter_mut().find(|a| a.id == id) {
            api.enabled = false;
            let response = ApiResponse {
                success: true,
                message: format!("API {} disabled", api.name),
                data: Some(api.clone()),
            };
            Ok(JsonResponse(response))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}
