use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
#[cfg(feature = "energy")]
use fixed::types::I32F32;
use pathfinding::prelude::astar;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;
use dashmap::DashMap;

#[derive(Serialize, Deserialize, Clone)]
pub struct Payload {
    pub node_id: u32,
    pub task_type: String,
    pub data: Vec<u8>,
    pub route: SmallVec<[u32; 4]>,
    pub priority: u8,
    pub timestamp: u64,
    #[cfg(feature = "energy")]
    pub energy_used: I32F32,
}

pub struct ARTPNode {
    pub node_id: u32,
    pub endpoint: quinn::Endpoint,
    pub connections: Arc<Mutex<HashMap<u32, quinn::Connection>>>,
    pub task_channels: Arc<DashMap<String, tokio::sync::broadcast::Sender<(Vec<u8>, u32)>>>,
}

impl ARTPNode {
    pub async fn new(node_id: u32, addr: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut client_config = quinn::ClientConfig::with_native_roots();
        // Add Kyber512 config here (placeholder)
        let mut endpoint = quinn::Endpoint::client(addr.parse()?)?;
        endpoint.set_default_client_config(client_config);

        Ok(Self {
            node_id,
            endpoint,
            connections: Arc::new(Mutex::new(HashMap::new())),
            task_channels: Arc::new(DashMap::new()),
        })
    }

    pub fn find_optimal_route(&self, target_id: u32) -> Option<SmallVec<[u32; 4]>> {
        // Simplified A* pathfinding
        let start = self.node_id;
        let goal = target_id;
        let result = astar(
            &start,
            |&n| vec![(n + 1, 1)].into_iter(), // Placeholder neighbors
            |&n| ((goal as i32 - n as i32).abs()) as u32,
            |&n| n == goal,
        );
        result.map(|(path, _)| path.into_iter().collect())
    }

    pub async fn send_data(&self, payload: Payload) -> Result<(), Box<dyn std::error::Error>> {
        let data = bincode::serialize(&payload)?;
        let conn = self.connections.lock().await.get(&payload.route[0]).cloned();
        if let Some(conn) = conn {
            let (mut send, _) = conn.open_bi().await?;
            send.write_all(&data).await?;
            send.finish().await?;
        }
        Ok(())
    }

    pub async fn start_receiving(&self) -> Result<(), Box<dyn std::error::Error>> {
        while let Some(conn) = self.endpoint.accept().await {
            let conn = conn.await?;
            let task_channels = self.task_channels.clone();
            tokio::spawn(async move {
                while let Ok((mut send, mut recv)) = conn.accept_bi().await {
                    let data = recv.read_to_end(usize::MAX).await.unwrap();
                    let payload: Payload = bincode::deserialize(&data).unwrap();
                    if let Some(tx) = task_channels.get(&payload.task_type) {
                        let _ = tx.send((payload.data, payload.node_id));
                    }
                }
            });
        }
        Ok(())
    }
}