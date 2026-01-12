export interface Model {
  id: string;
  name: string;
  path: string;
  model_type: ModelType;
  status: ModelStatus;
  loaded_at?: string;
  error_message?: string;
  config: ModelConfig;
}

export type ModelType = 'DeepSeekV2' | 'DeepSeekV3' | 'Qwen3Moe' | 'Other';

export type ModelStatus = 'unloaded' | 'loading' | 'loaded' | 'error';

export interface ModelConfig {
  max_length?: number;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
}

export interface ApiEndpoint {
  id: string;
  name: string;
  api_type: ApiType;
  base_url: string;
  enabled: boolean;
  config: ApiConfig;
}

export type ApiType = 'OpenAI' | 'Anthropic' | 'Codex' | 'OpenCode' | 'Custom';

export interface ApiConfig {
  model_name?: string;
  api_key?: string;
  max_tokens?: number;
  timeout?: number;
  retry_count?: number;
}

export interface EngineStatus {
  state: EngineState;
  uptime?: number;
  request_count: number;
  error_count: number;
  throughput?: number;
  memory_usage?: number;
  gpu_usage?: number;
}

export type EngineState = 'stopped' | 'starting' | 'running' | 'error';

export interface SystemMetrics {
  timestamp: string;
  cpu_usage: number;
  memory_usage: number;
  disk_usage: number;
  network_rx: number;
  network_tx: number;
}

export interface RequestLog {
  id: string;
  timestamp: string;
  method: string;
  url: string;
  status_code: number;
  response_time: number;
  client_ip: string;
  user_agent?: string;
}

export interface ApiResponse<T> {
  success: boolean;
  message: string;
  data?: T;
}

export interface LoadModelRequest {
  model_path: string;
  model_name?: string;
  config?: ModelConfig;
}

export interface EngineControlRequest {
  action: EngineAction;
}

export type EngineAction = 'start' | 'stop' | 'restart';