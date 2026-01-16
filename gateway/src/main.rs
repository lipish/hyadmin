#[tokio::main]
async fn main() -> anyhow::Result<()> {
    gateway::serve(std::net::SocketAddr::from(([127, 0, 0, 1], 8080))).await
}
