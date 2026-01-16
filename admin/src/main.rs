use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::Response,
    response::IntoResponse,
    routing::{delete, get, post},
    Json, Router,
};
use chrono::{DateTime, Utc};
use rand::{distributions::Alphanumeric, Rng};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::{sqlite::SqliteConnectOptions, sqlite::SqlitePoolOptions, SqlitePool};
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use uuid::Uuid;

#[derive(Clone)]
struct AppState {
    pool: SqlitePool,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
struct Engine {
    id: String,
    name: String,
    base_url: String,
    kind: String,
    status: String,
    group_name: String,
    weight: i64,
    priority: i64,
    enabled: bool,
    created_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EnginePayload {
    name: String,
    base_url: String,
    kind: String,
    status: Option<String>,
    group_name: Option<String>,
    weight: Option<i64>,
    priority: Option<i64>,
    enabled: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
struct ApiKey {
    id: String,
    name: String,
    key: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ApiKeyPayload {
    name: String,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
struct AdminUser {
    id: String,
    username: String,
    password_hash: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
struct AdminSession {
    id: String,
    user_id: String,
    token: String,
    created_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct LoginPayload {
    username: String,
    password: String,
}

#[derive(Debug, Serialize)]
struct LoginResponse {
    token: String,
}

#[derive(Debug, Serialize, Deserialize, sqlx::FromRow)]
struct GatewaySettings {
    id: String,
    strategy: String,
    failover_enabled: bool,
    vip_enabled: bool,
    notes: String,
    updated_at: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
struct GatewaySettingsPayload {
    strategy: String,
    failover_enabled: bool,
    vip_enabled: bool,
    notes: String,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: &'static str,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    std::fs::create_dir_all("data")?;
    let connect_opts = SqliteConnectOptions::new()
        .filename("data/admin.db")
        .create_if_missing(true);
    let pool = SqlitePoolOptions::new()
        .max_connections(5)
        .connect_with(connect_opts)
        .await?;

    init_db(&pool).await?;

    ensure_admin_user(&pool).await?;
    ensure_gateway_settings(&pool).await?;

    let state = AppState { pool };

    let public_routes = Router::new()
        .route("/health", get(health))
        .route("/auth/login", post(login));

    let protected_routes = Router::new()
        .route("/engines", get(list_engines).post(create_engine))
        .route(
            "/engines/:id",
            get(get_engine).put(update_engine).delete(delete_engine),
        )
        .route("/api-keys", get(list_api_keys).post(create_api_key))
        .route("/api-keys/:id", delete(delete_api_key))
        .route("/api-keys/:id/rotate", post(rotate_api_key))
        .route("/gateway/settings", get(get_gateway_settings).put(update_gateway_settings))
        .layer(middleware::from_fn_with_state(state.clone(), auth_guard));

    let app = Router::new()
        .merge(public_routes)
        .merge(protected_routes)
        .layer(CorsLayer::new().allow_origin(Any).allow_headers(Any))
        .with_state(state);

    let addr = SocketAddr::from(([0, 0, 0, 0], 9001));
    println!("admin api listening on {addr}");

    axum::serve(
        tokio::net::TcpListener::bind(addr).await?,
        app,
    )
    .await?;

    Ok(())
}

async fn ensure_gateway_settings(pool: &SqlitePool) -> anyhow::Result<()> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM gateway_settings")
        .fetch_one(pool)
        .await?;

    if count.0 > 0 {
        return Ok(());
    }

    sqlx::query(
        "INSERT INTO gateway_settings (id, strategy, failover_enabled, vip_enabled, notes, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind("default")
    .bind("weighted")
    .bind(true)
    .bind(false)
    .bind("默认策略")
    .bind(Utc::now())
    .execute(pool)
    .await?;

    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse { status: "ok" })
}

async fn auth_guard(
    State(state): State<AppState>,
    req: axum::http::Request<axum::body::Body>,
    next: Next,
) -> Result<Response, ApiError> {
    let token = extract_bearer(req.headers())
        .ok_or_else(|| ApiError::unauthorized("missing_token"))?;

    let session = sqlx::query_as::<_, AdminSession>(
        "SELECT id, user_id, token, created_at FROM admin_sessions WHERE token = ?",
    )
    .bind(token)
    .fetch_optional(&state.pool)
    .await?
    .ok_or_else(|| ApiError::unauthorized("invalid_token"))?;

    drop(session);
    Ok(next.run(req).await)
}

async fn login(
    State(state): State<AppState>,
    Json(payload): Json<LoginPayload>,
) -> Result<Json<LoginResponse>, ApiError> {
    let user = sqlx::query_as::<_, AdminUser>(
        "SELECT id, username, password_hash, created_at FROM admin_users WHERE username = ?",
    )
    .bind(&payload.username)
    .fetch_optional(&state.pool)
    .await?
    .ok_or_else(|| ApiError::unauthorized("invalid_credentials"))?;

    let incoming_hash = hash_password(&payload.password);
    if incoming_hash != user.password_hash {
        return Err(ApiError::unauthorized("invalid_credentials"));
    }

    let session = AdminSession {
        id: Uuid::new_v4().to_string(),
        user_id: user.id,
        token: generate_key(),
        created_at: Utc::now(),
    };

    sqlx::query(
        "INSERT INTO admin_sessions (id, user_id, token, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(&session.id)
    .bind(&session.user_id)
    .bind(&session.token)
    .bind(session.created_at)
    .execute(&state.pool)
    .await?;

    Ok(Json(LoginResponse { token: session.token }))
}

async fn list_engines(State(state): State<AppState>) -> Result<Json<Vec<Engine>>, ApiError> {
    let items = sqlx::query_as::<_, Engine>(
        "SELECT id, name, base_url, kind, status, group_name, weight, priority, enabled, created_at, updated_at FROM engines",
    )
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(items))
}

async fn get_gateway_settings(
    State(state): State<AppState>,
) -> Result<Json<GatewaySettings>, ApiError> {
    let item = sqlx::query_as::<_, GatewaySettings>(
        "SELECT id, strategy, failover_enabled, vip_enabled, notes, updated_at FROM gateway_settings WHERE id = ?",
    )
    .bind("default")
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(item))
}

async fn update_gateway_settings(
    State(state): State<AppState>,
    Json(payload): Json<GatewaySettingsPayload>,
) -> Result<Json<GatewaySettings>, ApiError> {
    let now = Utc::now();
    let result = sqlx::query(
        "UPDATE gateway_settings SET strategy = ?, failover_enabled = ?, vip_enabled = ?, notes = ?, updated_at = ? WHERE id = ?",
    )
    .bind(&payload.strategy)
    .bind(payload.failover_enabled)
    .bind(payload.vip_enabled)
    .bind(&payload.notes)
    .bind(now)
    .bind("default")
    .execute(&state.pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found("settings_not_found"));
    }

    let item = sqlx::query_as::<_, GatewaySettings>(
        "SELECT id, strategy, failover_enabled, vip_enabled, notes, updated_at FROM gateway_settings WHERE id = ?",
    )
    .bind("default")
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(item))
}

async fn get_engine(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Engine>, ApiError> {
    let item = sqlx::query_as::<_, Engine>(
        "SELECT id, name, base_url, kind, status, group_name, weight, priority, enabled, created_at, updated_at FROM engines WHERE id = ?",
    )
    .bind(&id)
    .fetch_optional(&state.pool)
    .await?
    .ok_or(ApiError::not_found("engine_not_found"))?;

    Ok(Json(item))
}

async fn create_engine(
    State(state): State<AppState>,
    Json(payload): Json<EnginePayload>,
) -> Result<(StatusCode, Json<Engine>), ApiError> {
    let now = Utc::now();
    let engine = Engine {
        id: Uuid::new_v4().to_string(),
        name: payload.name,
        base_url: payload.base_url,
        kind: payload.kind,
        status: payload.status.unwrap_or_else(|| "active".to_string()),
        group_name: payload.group_name.unwrap_or_else(|| "default".to_string()),
        weight: payload.weight.unwrap_or(100),
        priority: payload.priority.unwrap_or(0),
        enabled: payload.enabled.unwrap_or(true),
        created_at: now,
        updated_at: now,
    };

    sqlx::query(
        "INSERT INTO engines (id, name, base_url, kind, status, group_name, weight, priority, enabled, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    )
    .bind(&engine.id)
    .bind(&engine.name)
    .bind(&engine.base_url)
    .bind(&engine.kind)
    .bind(&engine.status)
    .bind(&engine.group_name)
    .bind(engine.weight)
    .bind(engine.priority)
    .bind(engine.enabled)
    .bind(engine.created_at)
    .bind(engine.updated_at)
    .execute(&state.pool)
    .await?;

    Ok((StatusCode::CREATED, Json(engine)))
}

async fn update_engine(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Json(payload): Json<EnginePayload>,
) -> Result<Json<Engine>, ApiError> {
    let now = Utc::now();
    let status = payload.status.unwrap_or_else(|| "active".to_string());
    let group_name = payload.group_name.unwrap_or_else(|| "default".to_string());
    let weight = payload.weight.unwrap_or(100);
    let priority = payload.priority.unwrap_or(0);
    let enabled = payload.enabled.unwrap_or(true);

    let result = sqlx::query(
        "UPDATE engines SET name = ?, base_url = ?, kind = ?, status = ?, group_name = ?, weight = ?, priority = ?, enabled = ?, updated_at = ? WHERE id = ?",
    )
    .bind(&payload.name)
    .bind(&payload.base_url)
    .bind(&payload.kind)
    .bind(&status)
    .bind(&group_name)
    .bind(weight)
    .bind(priority)
    .bind(enabled)
    .bind(now)
    .bind(&id)
    .execute(&state.pool)
    .await?;

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found("engine_not_found"));
    }

    let item = sqlx::query_as::<_, Engine>(
        "SELECT id, name, base_url, kind, status, created_at, updated_at FROM engines WHERE id = ?",
    )
    .bind(&id)
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(item))
}

async fn delete_engine(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let result = sqlx::query("DELETE FROM engines WHERE id = ?")
        .bind(&id)
        .execute(&state.pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found("engine_not_found"));
    }

    Ok(StatusCode::NO_CONTENT)
}

async fn list_api_keys(State(state): State<AppState>) -> Result<Json<Vec<ApiKey>>, ApiError> {
    let items = sqlx::query_as::<_, ApiKey>(
        "SELECT id, name, key, created_at FROM api_keys ORDER BY created_at DESC",
    )
    .fetch_all(&state.pool)
    .await?;

    Ok(Json(items))
}

async fn create_api_key(
    State(state): State<AppState>,
    Json(payload): Json<ApiKeyPayload>,
) -> Result<(StatusCode, Json<ApiKey>), ApiError> {
    let now = Utc::now();
    let api_key = ApiKey {
        id: Uuid::new_v4().to_string(),
        name: payload.name,
        key: generate_key(),
        created_at: now,
    };

    sqlx::query("INSERT INTO api_keys (id, name, key, created_at) VALUES (?, ?, ?, ?)")
        .bind(&api_key.id)
        .bind(&api_key.name)
        .bind(&api_key.key)
        .bind(api_key.created_at)
        .execute(&state.pool)
        .await?;

    Ok((StatusCode::CREATED, Json(api_key)))
}

async fn delete_api_key(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<StatusCode, ApiError> {
    let result = sqlx::query("DELETE FROM api_keys WHERE id = ?")
        .bind(&id)
        .execute(&state.pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found("api_key_not_found"));
    }

    Ok(StatusCode::NO_CONTENT)
}

async fn rotate_api_key(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<ApiKey>, ApiError> {
    let new_key = generate_key();

    let result = sqlx::query("UPDATE api_keys SET key = ? WHERE id = ?")
        .bind(&new_key)
        .bind(&id)
        .execute(&state.pool)
        .await?;

    if result.rows_affected() == 0 {
        return Err(ApiError::not_found("api_key_not_found"));
    }

    let item = sqlx::query_as::<_, ApiKey>(
        "SELECT id, name, key, created_at FROM api_keys WHERE id = ?",
    )
    .bind(&id)
    .fetch_one(&state.pool)
    .await?;

    Ok(Json(item))
}

async fn init_db(pool: &SqlitePool) -> anyhow::Result<()> {
    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS engines (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            base_url TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            group_name TEXT NOT NULL,
            weight INTEGER NOT NULL,
            priority INTEGER NOT NULL,
            enabled INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS api_keys (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            key TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS admin_users (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS admin_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            token TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES admin_users(id)
        );
        "#,
    )
    .execute(pool)
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS gateway_settings (
            id TEXT PRIMARY KEY,
            strategy TEXT NOT NULL,
            failover_enabled INTEGER NOT NULL,
            vip_enabled INTEGER NOT NULL,
            notes TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .execute(pool)
    .await?;

    Ok(())
}

fn generate_key() -> String {
    rand::thread_rng()
        .sample_iter(&Alphanumeric)
        .take(32)
        .map(char::from)
        .collect()
}

fn hash_password(password: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(password.as_bytes());
    hex::encode(hasher.finalize())
}

async fn ensure_admin_user(pool: &SqlitePool) -> anyhow::Result<()> {
    let count: (i64,) = sqlx::query_as("SELECT COUNT(*) FROM admin_users")
        .fetch_one(pool)
        .await?;

    if count.0 > 0 {
        return Ok(());
    }

    let username = std::env::var("ADMIN_USERNAME").unwrap_or_else(|_| "admin".to_string());
    let password = std::env::var("ADMIN_PASSWORD").unwrap_or_else(|_| "admin123".to_string());
    let user = AdminUser {
        id: Uuid::new_v4().to_string(),
        username,
        password_hash: hash_password(&password),
        created_at: Utc::now(),
    };

    sqlx::query(
        "INSERT INTO admin_users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
    )
    .bind(&user.id)
    .bind(&user.username)
    .bind(&user.password_hash)
    .bind(user.created_at)
    .execute(pool)
    .await?;

    Ok(())
}

fn extract_bearer(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.strip_prefix("Bearer "))
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn not_found(message: &str) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.to_string(),
        }
    }

    fn unauthorized(message: &str) -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            message: message.to_string(),
        }
    }
}

impl From<sqlx::Error> for ApiError {
    fn from(err: sqlx::Error) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: format!("db_error: {err}"),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let body = serde_json::json!({
            "error": self.message,
        });
        (self.status, Json(body)).into_response()
    }
}
