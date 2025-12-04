use artp::artp::ARTPNode;
use artp::grpc_service::ArtpService;
use tonic::transport::Server;
use artp_service::artp_service_server::ArtpServiceServer;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let node = ARTPNode::new(1, "0.0.0.0:0").await?;
    let artp_service = ArtpService::new(node.clone());
    let addr = "0.0.0.0:50051".parse()?;
    println!("ARTP Service running on {}", addr);
    tokio::spawn(async move {
        node.start_receiving().await.unwrap();
    });
    Server::builder()
        .add_service(ArtpServiceServer::new(artp_service))
        .serve(addr)
        .await?;
    Ok(())
}