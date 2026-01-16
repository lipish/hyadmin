mod admin;
mod gateway;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let admin_addr = std::net::SocketAddr::from(([0, 0, 0, 0], 9001));
    let gateway_addr = std::net::SocketAddr::from(([127, 0, 0, 1], 8080));

    println!("backend starting: admin on {admin_addr}, gateway on {gateway_addr}");

    let admin_task = tokio::spawn(async move { admin::serve(admin_addr).await });
    let gateway_task = tokio::spawn(async move { gateway::serve(gateway_addr).await });

    admin_task.await??;
    gateway_task.await??;

    Ok(())
}
