mod config;
mod handlers;
mod models;
mod routes;
mod services;

use std::net::SocketAddr;

use axum::Router;
use hyper::Server;
use tower_http::cors::CorsLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use crate::config::AppConfig;
use crate::services::AppState;

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "heyi_admin=debug,tower_http=debug".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .init();

    // Load configuration
    let config = AppConfig::load().expect("Failed to load configuration");

    // Create application state
    let state = AppState::new(config.clone());

    // Build the application router
    let app = Router::new()
        .merge(routes::dashboard_routes())
        .merge(routes::model_routes())
        .merge(routes::api_routes())
        .merge(routes::monitoring_routes())
        .layer(CorsLayer::permissive());

    // Serve static files
    let app = app.merge(routes::static_routes());

    // Start the server
    let addr: SocketAddr = format!("{}:{}", config.host, config.port).parse().unwrap();
    println!("🚀 Heyi Admin server running at http://{}", addr);
    println!("📖 Visit http://{} in your browser", addr);
    println!("🎨 React frontend available at http://{}:3005", config.host);

    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}
