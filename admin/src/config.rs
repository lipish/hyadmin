use serde::{Deserialize, Serialize};
use std::env;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    pub host: String,
    pub port: u16,
    pub heyi_api_url: String,
    pub heyi_api_key: Option<String>,
    pub database_url: Option<String>,
    pub redis_url: Option<String>,
    pub log_level: String,
    pub admin_username: Option<String>,
    pub admin_password: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            heyi_api_url: "http://localhost:10814".to_string(),
            heyi_api_key: None,
            database_url: None,
            redis_url: None,
            log_level: "info".to_string(),
            admin_username: Some("admin".to_string()),
            admin_password: Some("admin".to_string()),
        }
    }
}

impl AppConfig {
    pub fn load() -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        // Try to load from environment variables first
        let config = Self {
            host: env::var("ADMIN_HOST").unwrap_or_else(|_| "127.0.0.1".to_string()),
            port: env::var("ADMIN_PORT")
                .unwrap_or_default()
                .parse()
                .unwrap_or(8080),
            heyi_api_url: env::var("HEYI_API_URL")
                .unwrap_or_else(|_| "http://localhost:10814".to_string()),
            heyi_api_key: env::var("HEYI_API_KEY").ok(),
            database_url: env::var("DATABASE_URL").ok(),
            redis_url: env::var("REDIS_URL").ok(),
            log_level: env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
            admin_username: env::var("ADMIN_USERNAME").ok().or(Some("admin".to_string())),
            admin_password: env::var("ADMIN_PASSWORD").ok().or(Some("admin".to_string())),
        };

        Ok(config)
    }
}
