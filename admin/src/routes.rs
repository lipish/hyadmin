use axum::{
    Router,
    routing::{get, post, put, delete},
};
use tower_http::services::ServeDir;

use crate::handlers;
use crate::services::AppState;

pub fn dashboard_routes() -> Router {
    Router::new()
        .route("/", get(handlers::dashboard::dashboard))
        .route("/dashboard", get(handlers::dashboard::dashboard))
        .route("/login", get(handlers::auth::login_page))
        .route("/login", post(handlers::auth::login))
        .route("/logout", post(handlers::auth::logout))
}

pub fn model_routes() -> Router {
    Router::new()
        .route("/api/models", get(handlers::models::list_models))
        .route("/api/models", post(handlers::models::create_model))
        .route("/api/models/scan", post(handlers::models::scan_models))
        .route("/api/models/:id/load", post(handlers::models::load_model))
        .route("/api/models/:id/unload", post(handlers::models::unload_model))
        .route("/api/models/:id", get(handlers::models::get_model))
        .route("/api/models/:id", put(handlers::models::update_model))
        .route("/api/models/:id", delete(handlers::models::delete_model))
}

pub fn api_routes() -> Router {
    Router::new()
        .route("/api/apis", get(handlers::apis::list_apis))
        .route("/api/apis", post(handlers::apis::create_api))
        .route("/api/apis/:id", get(handlers::apis::get_api))
        .route("/api/apis/:id", put(handlers::apis::update_api))
        .route("/api/apis/:id", delete(handlers::apis::delete_api))
        .route("/api/apis/:id/enable", post(handlers::apis::enable_api))
        .route("/api/apis/:id/disable", post(handlers::apis::disable_api))
}

pub fn monitoring_routes() -> Router {
    Router::new()
        .route("/api/monitoring", get(handlers::monitoring::dashboard))
        .route("/api/monitoring/metrics", get(handlers::monitoring::get_metrics))
        .route("/api/monitoring/logs", get(handlers::monitoring::get_logs))
        .route("/api/monitoring/engine", get(handlers::monitoring::get_engine_status))
}

pub fn engine_routes() -> Router<AppState> {
    Router::new()
        .route("/api/engine/control", post(handlers::engine::control_engine))
        .route("/api/engine/config", get(handlers::engine::get_engine_config))
        .route("/api/engine/config", post(handlers::engine::update_engine_config))
}

pub fn static_routes() -> Router {
    Router::new()
        .nest_service("/static", ServeDir::new("frontend/build/static"))
}
