# Heyi Admin

Heyi Admin 是一个现代化的管理界面，用于管理 Heyi 推理引擎。它采用 Rust + React + Shadcn UI 构建，提供模型管理、API 配置和系统监控等功能。

## 功能特性

### 🎯 核心功能
- **模型管理**：动态加载、切换和卸载 AI 模型
- **API 管理**：配置多种 API 格式（OpenAI、Anthropic、Codex、OpenCode等）
- **系统监控**：实时查看引擎状态、性能指标和系统资源
- **现代化UI**：基于 React + Shadcn UI 的美观界面

### 🏗️ 架构设计
```
┌─────────────────┐    HTTP API    ┌─────────────────┐
│   Heyi Admin    │◄──────────────►│  Heyi Engine    │
│ Rust + React    │                │  (Python)       │
│ ├─ Axum API     │                └─────────────────┘
│ └─ React UI     │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Web Browser   │
│   (Admin UI)    │
└─────────────────┘
```

## 安装和运行

### 环境要求
- Rust 1.70+
- Node.js 18+
- Python 3.8+ (Heyi Engine)

### 快速开始

#### 开发模式
```bash
cd admin
./dev.sh
```
这将启动 React 开发服务器和 Rust 后端。

#### 生产构建
```bash
cd admin
./build.sh
cargo run --release
```

### 配置
复制配置模板并修改：
```bash
cp config.env.example .env
```

主要配置项：
```bash
# 服务器端口
ADMIN_PORT=8080

# Heyi Engine API 地址
HEYI_API_URL="http://localhost:10814"

# 管理员认证
ADMIN_USERNAME="admin"
ADMIN_PASSWORD="admin"
```

### 手动构建

#### 构建前端
```bash
cd admin/frontend
npm install
npm run build
```

#### 构建后端
```bash
cd admin
cargo build --release
```

### 运行
```bash
# 开发模式
cd admin && ./dev.sh

# 或生产模式
cd admin && cargo run --release
```

服务将在 `http://localhost:8080` 启动。

## API 接口

### 模型管理
```bash
# 获取所有模型
GET /models

# 添加新模型
POST /models
{
  "name": "DeepSeek-V3",
  "path": "/path/to/model"
}

# 加载模型
POST /models/{id}/load

# 卸载模型
POST /models/{id}/unload
```

### API 管理
```bash
# 获取所有 API 配置
GET /apis

# 添加 API 配置
POST /apis
{
  "name": "OpenAI Compatible",
  "api_type": "openai",
  "base_url": "/v1",
  "enabled": true,
  "config": {
    "model_name": "deepseek-chat",
    "api_key": "sk-...",
    "max_tokens": 4096
  }
}

# 启用/禁用 API
POST /apis/{id}/enable
POST /apis/{id}/disable
```

### 监控
```bash
# 获取系统指标
GET /monitoring/metrics

# 获取引擎状态
GET /monitoring/engine

# 获取请求日志
GET /monitoring/logs
```

## 开发

### 项目结构
```
admin/
├── frontend/             # 🎨 React 前端应用
│   ├── src/
│   │   ├── components/   # ⚛️ React 组件
│   │   │   ├── ui/       # Shadcn UI 组件
│   │   │   └── Layout.tsx
│   │   ├── pages/        # 📄 页面组件
│   │   │   ├── Dashboard.tsx
│   │   │   ├── Models.tsx
│   │   │   ├── APIs.tsx
│   │   │   ├── Monitoring.tsx
│   │   │   └── Login.tsx
│   │   ├── hooks/        # 🎣 自定义 hooks
│   │   ├── lib/          # 🛠️ 工具库
│   │   └── types/        # 📝 TypeScript 类型
│   ├── public/
│   ├── package.json
│   └── tailwind.config.js
├── src/                  # 🦀 Rust 后端
│   ├── main.rs           # 🚀 应用入口点
│   ├── config.rs         # ⚙️ 配置管理
│   ├── models.rs         # 📋 数据模型定义
│   ├── services.rs       # 🔧 业务逻辑服务
│   ├── handlers/         # 🎯 HTTP 请求处理器
│   │   ├── mod.rs
│   │   ├── auth.rs       # 🔐 用户认证
│   │   ├── dashboard.rs  # 📊 主仪表板
│   │   ├── models.rs     # 🤖 模型管理
│   │   ├── apis.rs       # 🌐 API 配置管理
│   │   └── monitoring.rs # 📈 系统监控
│   └── routes.rs         # 🛣️ 路由定义
├── build.sh              # 🏗️ 构建脚本
├── dev.sh                # 🚀 开发脚本
├── config.env.example    # ⚙️ 配置示例
├── Cargo.toml            # 📦 Rust 依赖配置
└── README.md             # 📖 项目文档
```

### 开发指南

#### 前端开发 (React + TypeScript)
```bash
cd admin/frontend
npm install
npm start          # 启动开发服务器
npm run build      # 生产构建
```

#### 后端开发 (Rust)
```bash
cd admin
cargo build        # 开发构建
cargo run          # 运行开发版本
cargo test         # 运行测试
```

#### 全栈开发
```bash
cd admin
./dev.sh           # 同时启动前后端
```

### 添加新功能

#### 后端 (Rust)
1. 在 `models.rs` 中定义数据结构
2. 在 `handlers/` 中实现业务逻辑
3. 在 `routes.rs` 中添加路由

#### 前端 (React)
1. 在 `types/` 中定义 TypeScript 类型
2. 在 `components/` 中创建 UI 组件
3. 在 `pages/` 中创建页面组件
4. 更新路由配置

## 部署

### Docker 部署
```dockerfile
FROM rust:1.70-slim as builder
WORKDIR /app
COPY . .
RUN cargo build --release

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/heyi-admin /usr/local/bin/
EXPOSE 8080
CMD ["heyi-admin"]
```

### 系统服务
```systemd
[Unit]
Description=Heyi Admin Service
After=network.target

[Service]
Type=simple
User=heyi
ExecStart=/usr/local/bin/heyi-admin
Restart=always
Environment=HEYI_API_URL=http://localhost:10814

[Install]
WantedBy=multi-user.target
```

## 安全性

- 默认启用管理员认证
- 支持 HTTPS（推荐生产环境）
- API 密钥管理
- 请求日志记录

## 许可证

[待定]