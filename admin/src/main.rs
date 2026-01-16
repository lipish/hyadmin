#[tokio::main]
async fn main() -> anyhow::Result<()> {
    admin::serve(std::net::SocketAddr::from(([0, 0, 0, 0], 9001))).await
}
