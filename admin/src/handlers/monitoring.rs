use axum::{
    response::{Json as JsonResponse, Html},
};

use crate::models::{SystemMetrics, EngineStatus, RequestLog};

pub async fn dashboard() -> Html<String> {
    let html = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Heyi Admin - Monitoring</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .metric-card { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
        .metric-value { font-size: 2em; font-weight: bold; color: #007bff; }
        .logs { max-height: 400px; overflow-y: auto; border: 1px solid #ddd; padding: 10px; }
        .log-entry { margin-bottom: 5px; padding: 5px; border-bottom: 1px solid #eee; }
    </style>
</head>
<body>
    <h1>System Monitoring</h1>

    <div class="metrics" id="metrics">
        <!-- Metrics will be loaded via JavaScript -->
    </div>

    <h2>Recent Logs</h2>
    <div class="logs" id="logs">
        <!-- Logs will be loaded via JavaScript -->
    </div>

    <script>
        async function updateMetrics() {
            try {
                const response = await fetch('/api/monitoring/metrics');
                const metrics = await response.json();
                document.getElementById('metrics').innerHTML = metrics.map(m => `
                    <div class="metric-card">
                        <h3>System Metrics</h3>
                        <div class="metric-value">${m.cpu_usage.toFixed(1)}%</div>
                        <small>CPU Usage</small>
                        <div class="metric-value">${(m.memory_usage / 1024 / 1024 / 1024).toFixed(1)}GB</div>
                        <small>Memory Usage</small>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load metrics:', error);
            }
        }

        async function updateLogs() {
            try {
                const response = await fetch('/api/monitoring/logs?limit=50');
                const logs = await response.json();
                document.getElementById('logs').innerHTML = logs.map(log => `
                    <div class="log-entry">
                        <strong>${log.method} ${log.url}</strong> - ${log.status_code} (${log.response_time}ms)
                        <br><small>${log.timestamp} - ${log.client_ip}</small>
                    </div>
                `).join('');
            } catch (error) {
                console.error('Failed to load logs:', error);
            }
        }

        // Update every 5 seconds
        updateMetrics();
        updateLogs();
        setInterval(() => {
            updateMetrics();
            updateLogs();
        }, 5000);
    </script>
</body>
</html>
"#;

    Html(html.to_string())
}

pub async fn get_metrics() -> JsonResponse<Vec<SystemMetrics>> {
    // Return demo metrics
    let metrics = vec![SystemMetrics {
        timestamp: chrono::Utc::now(),
        cpu_usage: 45.2,
        memory_usage: 8.5 * 1024.0 * 1024.0 * 1024.0, // 8.5GB in bytes
        disk_usage: 65.3,
        network_rx: 1024 * 1024 * 50, // 50MB
        network_tx: 1024 * 1024 * 25, // 25MB
    }];
    JsonResponse(metrics)
}

pub async fn get_logs() -> JsonResponse<Vec<RequestLog>> {
    // Return demo logs
    let logs = vec![
        RequestLog {
            id: "log-1".to_string(),
            timestamp: chrono::Utc::now(),
            method: "POST".to_string(),
            url: "/api/chat/completions".to_string(),
            status_code: 200,
            response_time: 150,
            client_ip: "127.0.0.1".to_string(),
            user_agent: Some("Heyi-Client/1.0".to_string()),
        },
        RequestLog {
            id: "log-2".to_string(),
            timestamp: chrono::Utc::now(),
            method: "GET".to_string(),
            url: "/api/models".to_string(),
            status_code: 200,
            response_time: 45,
            client_ip: "127.0.0.1".to_string(),
            user_agent: Some("Mozilla/5.0".to_string()),
        },
    ];
    JsonResponse(logs)
}

pub async fn get_engine_status() -> JsonResponse<EngineStatus> {
    // Return demo engine status
    let status = EngineStatus {
        state: crate::models::EngineState::Running,
        uptime: Some(3600), // 1 hour
        request_count: 1250,
        error_count: 3,
        throughput: Some(15.7),
        memory_usage: Some(6.2 * 1024.0 * 1024.0 * 1024.0), // 6.2GB
        gpu_usage: Some(78.5),
    };
    JsonResponse(status)
}
