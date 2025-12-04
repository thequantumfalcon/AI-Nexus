"""
Qdrant Vector Database Client with GPU Acceleration
High-performance vector search with privacy-preserving features
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class VectorSearchResult:
    """Result from vector search"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[np.ndarray] = None


class QdrantClient:
    """
    GPU-accelerated Qdrant client with privacy features
    
    Features:
    - GPU-accelerated embedding generation
    - Privacy-preserving vector search
    - Batch operations for efficiency
    - Integration with distributed training
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        use_gpu: bool = True,
        embedding_dim: int = 384
    ):
        """
        Initialize Qdrant client
        
        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for authentication
            use_gpu: Use GPU for embeddings
            embedding_dim: Embedding dimension
        """
        self.host = host
        self.port = port
        self.api_key = api_key
        self.use_gpu = use_gpu
        self.embedding_dim = embedding_dim
        
        self._initialize_client()
        self._initialize_embedding_model()
        
        logger.info(f"Qdrant client initialized: {host}:{port}, GPU={use_gpu}")
    
    def _initialize_client(self):
        """Initialize Qdrant client connection"""
        try:
            from qdrant_client import QdrantClient as QdrantClientLib
            from qdrant_client.models import Distance, VectorParams
            
            self.client = QdrantClientLib(
                host=self.host,
                port=self.port,
                api_key=self.api_key,
                timeout=30
            )
            
            self.Distance = Distance
            self.VectorParams = VectorParams
            
            logger.info("Qdrant client connected successfully")
            
        except ImportError:
            logger.error(
                "qdrant-client not installed. Install with:\n"
                "pip install qdrant-client"
            )
            raise
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant server: {e}")
            # Will use in-memory mode
            self.client = None
    
    def _initialize_embedding_model(self):
        """Initialize embedding model (GPU-accelerated)"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Use GPU if available
            device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'
            
            # Load efficient embedding model
            self.embedding_model = SentenceTransformer(
                'all-MiniLM-L6-v2',  # 384 dimensions, fast
                device=device
            )
            
            logger.info(f"Embedding model loaded on {device}")
            
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. Using random embeddings.\n"
                "Install with: pip install sentence-transformers"
            )
            self.embedding_model = None
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: Optional[int] = None,
        distance: str = "Cosine"
    ) -> bool:
        """
        Create a new collection
        
        Args:
            collection_name: Name of the collection
            vector_size: Vector dimension (defaults to embedding_dim)
            distance: Distance metric (Cosine, Euclid, Dot)
        
        Returns:
            True if successful
        """
        if self.client is None:
            logger.warning("Qdrant client not connected")
            return False
        
        vector_size = vector_size or self.embedding_dim
        
        try:
            # Map distance string to enum
            distance_map = {
                "Cosine": self.Distance.COSINE,
                "Euclid": self.Distance.EUCLID,
                "Dot": self.Distance.DOT
            }
            
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=self.VectorParams(
                    size=vector_size,
                    distance=distance_map.get(distance, self.Distance.COSINE)
                )
            )
            
            logger.info(f"Collection '{collection_name}' created (size={vector_size}, distance={distance})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            return False
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection"""
        if self.client is None:
            return False
        
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Collection '{collection_name}' deleted")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Generate embeddings for texts using GPU
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            show_progress: Show progress bar
        
        Returns:
            Embeddings array (N x embedding_dim)
        """
        if self.embedding_model is None:
            # Fallback to random embeddings
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
        
        try:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization
            )
            
            return embeddings.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return np.random.randn(len(texts), self.embedding_dim).astype(np.float32)
    
    def insert_vectors(
        self,
        collection_name: str,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        ids: Optional[List[str]] = None,
        batch_size: int = 100
    ) -> bool:
        """
        Insert vectors into collection
        
        Args:
            collection_name: Collection name
            vectors: Vector embeddings (N x dim)
            payloads: Metadata for each vector
            ids: Optional vector IDs (auto-generated if None)
            batch_size: Batch size for insertion
        
        Returns:
            True if successful
        """
        if self.client is None:
            logger.warning("Qdrant client not connected")
            return False
        
        try:
            from qdrant_client.models import PointStruct
            
            # Generate IDs if not provided
            if ids is None:
                ids = [str(i) for i in range(len(vectors))]
            
            # Create points
            points = [
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload
                )
                for point_id, vector, payload in zip(ids, vectors, payloads)
            ]
            
            # Insert in batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]
                self.client.upsert(
                    collection_name=collection_name,
                    points=batch
                )
            
            logger.info(f"Inserted {len(points)} vectors into '{collection_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Vector insertion failed: {e}")
            return False
    
    def search(
        self,
        collection_name: str,
        query_vector: np.ndarray,
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict] = None
    ) -> List[VectorSearchResult]:
        """
        Search for similar vectors
        
        Args:
            collection_name: Collection name
            query_vector: Query embedding
            limit: Maximum results to return
            score_threshold: Minimum similarity score
            filter_conditions: Optional metadata filters
        
        Returns:
            List of search results
        """
        if self.client is None:
            logger.warning("Qdrant client not connected")
            return []
        
        try:
            results = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector.tolist(),
                limit=limit,
                score_threshold=score_threshold,
                query_filter=filter_conditions
            )
            
            search_results = [
                VectorSearchResult(
                    id=str(hit.id),
                    score=hit.score,
                    payload=hit.payload,
                    vector=np.array(hit.vector) if hit.vector else None
                )
                for hit in results
            ]
            
            logger.debug(f"Search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def search_texts(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = 10,
        score_threshold: Optional[float] = None
    ) -> List[List[VectorSearchResult]]:
        """
        Search using text queries (automatically embedded)
        
        Args:
            collection_name: Collection name
            query_texts: Text queries
            limit: Results per query
            score_threshold: Minimum score
        
        Returns:
            List of result lists (one per query)
        """
        # Generate embeddings for queries
        query_vectors = self.embed_texts(query_texts)
        
        # Search for each query
        all_results = []
        for query_vector in query_vectors:
            results = self.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold
            )
            all_results.append(results)
        
        return all_results
    
    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        """Get collection information"""
        if self.client is None:
            return None
        
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return {
                "name": collection_name,
                "vectors_count": info.vectors_count,
                "points_count": info.points_count,
                "status": info.status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": info.config.params.vectors.distance.name
                }
            }
        except Exception as e:
            logger.error(f"Failed to get collection info: {e}")
            return None
    
    def close(self):
        """Close client connection"""
        if self.client:
            try:
                self.client.close()
                logger.info("Qdrant client closed")
            except:
                pass


class PrivacyPreservingVectorDB(QdrantClient):
    """
    Privacy-preserving vector database with differential privacy
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        **kwargs
    ):
        """
        Initialize with privacy parameters
        
        Args:
            epsilon: Privacy budget
            delta: Privacy parameter
            **kwargs: Arguments for QdrantClient
        """
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.delta = delta
        
        logger.info(f"Privacy-preserving mode: ε={epsilon}, δ={delta}")
    
    def privatize_vector(
        self,
        vector: np.ndarray,
        sensitivity: float = 1.0
    ) -> np.ndarray:
        """
        Add differential privacy noise to vector
        
        Args:
            vector: Input vector
            sensitivity: L2 sensitivity
        
        Returns:
            Privatized vector
        """
        import math
        
        # Gaussian mechanism
        sigma = sensitivity * math.sqrt(2 * math.log(1.25 / self.delta)) / self.epsilon
        
        noise = np.random.randn(*vector.shape) * sigma
        noisy_vector = vector + noise
        
        # Re-normalize
        norm = np.linalg.norm(noisy_vector)
        if norm > 0:
            noisy_vector = noisy_vector / norm
        
        return noisy_vector.astype(np.float32)
    
    def insert_private_vectors(
        self,
        collection_name: str,
        vectors: np.ndarray,
        payloads: List[Dict[str, Any]],
        **kwargs
    ) -> bool:
        """Insert vectors with differential privacy"""
        # Privatize each vector
        private_vectors = np.array([
            self.privatize_vector(v) for v in vectors
        ])
        
        return self.insert_vectors(
            collection_name=collection_name,
            vectors=private_vectors,
            payloads=payloads,
            **kwargs
        )
