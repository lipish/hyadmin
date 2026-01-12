use axum::{
    extract::{Path, Json},
    response::{Json as JsonResponse},
    http::StatusCode,
};
use uuid::Uuid;
use chrono::Utc;

use crate::models::{Model, ModelStatus, ModelType, ApiResponse};

// Static data for demo purposes
static mut MODELS: Vec<Model> = Vec::new();

pub async fn list_models() -> JsonResponse<Vec<Model>> {
    unsafe {
        if MODELS.is_empty() {
            // Initialize with some demo data
            MODELS.push(Model {
                id: "demo-model-1".to_string(),
                name: "DeepSeek-V3".to_string(),
                path: "/models/deepseek-v3".to_string(),
                model_type: ModelType::DeepSeekV3,
                status: ModelStatus::Unloaded,
                loaded_at: None,
                error_message: None,
                config: Default::default(),
            });
        }
        JsonResponse(MODELS.clone())
    }
}

pub async fn get_model(
    Path(id): Path<String>,
) -> Result<JsonResponse<Model>, StatusCode> {
    unsafe {
        if let Some(model) = MODELS.iter().find(|m| m.id == id) {
            Ok(JsonResponse(model.clone()))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn create_model(
    Json(model_data): Json<serde_json::Value>,
) -> Result<JsonResponse<Model>, StatusCode> {
    let model = Model {
        id: Uuid::new_v4().to_string(),
        name: model_data["name"].as_str().unwrap_or("Unnamed").to_string(),
        path: model_data["path"].as_str().unwrap_or("").to_string(),
        model_type: ModelType::Other("unknown".to_string()),
        status: ModelStatus::Unloaded,
        loaded_at: None,
        error_message: None,
        config: Default::default(),
    };

    unsafe {
        MODELS.push(model.clone());
    }

    Ok(JsonResponse(model))
}

pub async fn update_model(
    Path(id): Path<String>,
    Json(update_data): Json<serde_json::Value>,
) -> Result<JsonResponse<Model>, StatusCode> {
    unsafe {
        if let Some(model) = MODELS.iter_mut().find(|m| m.id == id) {
            if let Some(name) = update_data["name"].as_str() {
                model.name = name.to_string();
            }
            if let Some(path) = update_data["path"].as_str() {
                model.path = path.to_string();
            }
            Ok(JsonResponse(model.clone()))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn delete_model(
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    unsafe {
        if let Some(pos) = MODELS.iter().position(|m| m.id == id) {
            MODELS.remove(pos);
            Ok(StatusCode::NO_CONTENT)
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn scan_models(
    Json(_scan_request): Json<serde_json::Value>,
) -> JsonResponse<ApiResponse<Vec<Model>>> {
    let response = ApiResponse {
        success: true,
        message: "Models scanned successfully".to_string(),
        data: Some(vec![]),
    };
    JsonResponse(response)
}

pub async fn load_model(
    Path(id): Path<String>,
) -> Result<JsonResponse<ApiResponse<Model>>, StatusCode> {
    unsafe {
        if let Some(model) = MODELS.iter_mut().find(|m| m.id == id) {
            model.status = ModelStatus::Loading;
            model.error_message = None;
            model.status = ModelStatus::Loaded;
            model.loaded_at = Some(Utc::now());

            let response = ApiResponse {
                success: true,
                message: format!("Model {} loaded successfully", model.name),
                data: Some(model.clone()),
            };

            Ok(JsonResponse(response))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}

pub async fn unload_model(
    Path(id): Path<String>,
) -> Result<JsonResponse<ApiResponse<Model>>, StatusCode> {
    unsafe {
        if let Some(model) = MODELS.iter_mut().find(|m| m.id == id) {
            model.status = ModelStatus::Unloaded;
            model.loaded_at = None;

            let response = ApiResponse {
                success: true,
                message: format!("Model {} unloaded successfully", model.name),
                data: Some(model.clone()),
            };

            Ok(JsonResponse(response))
        } else {
            Err(StatusCode::NOT_FOUND)
        }
    }
}
