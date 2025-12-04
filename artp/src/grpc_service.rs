use tonic::{Request, Response, Status};
use tokio_stream::StreamExt;
use futures::Stream;
use std::pin::Pin;
use std::time::{SystemTime, UNIX_EPOCH};
use bincode;
use crate::artp::{ARTPNode, Payload};
use artp_service::{artp_service_server::ArtpService as ArtpServiceTrait, SendDataRequest, SendDataResponse, ReceiveDataRequest, ReceiveDataResponse};

pub struct ArtpService {
    node: ARTPNode,
}

impl ArtpService {
    pub fn new(node: ARTPNode) -> Self {
        Self { node }
    }
}

#[tonic::async_trait]
impl ArtpServiceTrait for ArtpService {
    type ReceiveDataStream = Pin<Box<dyn Stream<Item = Result<ReceiveDataResponse, Status>> + Send>>;

    async fn send_data(&self, request: Request<SendDataRequest>) -> Result<Response<SendDataResponse>, Status> {
        let req = request.into_inner();
        let payload = Payload {
            node_id: self.node.node_id,
            task_type: req.task_type,
            data: req.data,
            route: self.node.find_optimal_route(req.target_id).ok_or_else(|| Status::not_found("No route"))?,
            priority: req.priority,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| Status::internal("Time error"))?.as_secs(),
            #[cfg(feature = "energy")]
            energy_used: fixed::types::I32F32::from_num(0),
        };
        self.node.send_data(payload).await.map_err(|_| Status::internal("Send failed"))?;
        Ok(Response::new(SendDataResponse { status: "sent".to_string() }))
    }

    async fn receive_data(&self, request: Request<ReceiveDataRequest>) -> Result<Response<Self::ReceiveDataStream>, Status> {
        let task_type = request.into_inner().task_type;
        let tx = self.node.task_channels.entry(task_type.clone()).or_insert_with(|| tokio::sync::broadcast::channel(100).0).clone();
        let mut rx = tx.subscribe();
        let stream = async_stream::stream! {
            while let Ok((data, sender_id)) = rx.recv().await {
                yield Ok(ReceiveDataResponse { data, sender_id });
            }
        };
        Ok(Response::new(Box::pin(stream)))
    }
}