"""
Retrieval-Augmented Generation (RAG) Pipeline
Integrates Qdrant vector database with ML inference for enhanced AI responses
GPU-accelerated embeddings + vector search + LLM generation
"""

import torch
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging
from pathlib import Path

from services.vectordb.qdrant_client import QdrantClient, PrivacyPreservingVectorDB
from services.vectordb.embeddings import GPUEmbeddingService, get_embedding_service


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    collection_name: str = "knowledge_base"
    top_k: int = 5  # Number of documents to retrieve
    min_score: float = 0.7  # Minimum similarity score
    chunk_size: int = 512  # Document chunk size
    chunk_overlap: int = 50  # Overlap between chunks
    enable_privacy: bool = False
    privacy_epsilon: float = 1.0
    embedding_model: str = "balanced"  # fast, balanced, or quality
    use_gpu: bool = True
    cache_size: int = 10000


class DocumentChunker:
    """Split documents into overlapping chunks for embedding"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.logger = logging.getLogger(__name__)
    
    def chunk_text(self, text: str, metadata: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to each chunk
            
        Returns:
            List of chunk dictionaries with text and metadata
        """
        if not text:
            return []
        
        chunks = []
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Create chunk metadata
            chunk_meta = {
                'chunk_id': chunk_id,
                'start_pos': start,
                'end_pos': end,
                'chunk_size': len(chunk_text)
            }
            
            # Add user metadata
            if metadata:
                chunk_meta.update(metadata)
            
            chunks.append({
                'text': chunk_text,
                'metadata': chunk_meta
            })
            
            # Move to next chunk with overlap
            start = end - self.overlap
            chunk_id += 1
        
        self.logger.info(f"Split text into {len(chunks)} chunks")
        return chunks


class RAGPipeline:
    """
    Retrieval-Augmented Generation Pipeline
    
    Combines:
    1. GPU-accelerated embedding generation
    2. Qdrant vector similarity search
    3. Context injection for LLM generation
    4. Optional differential privacy
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        qdrant_url: str = "http://localhost:6333",
        api_key: Optional[str] = None
    ):
        """
        Initialize RAG pipeline
        
        Args:
            config: RAG configuration
            qdrant_url: Qdrant server URL
            api_key: Optional Qdrant API key
        """
        self.config = config or RAGConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize GPU embedding service
        self.embedding_service = get_embedding_service(
            model_type=self.config.embedding_model,
            use_gpu=self.config.use_gpu,
            cache_size=self.config.cache_size
        )
        
        # Parse URL
        if "://" in qdrant_url:
            qdrant_url = qdrant_url.split("://")[1]
        
        if ":" in qdrant_url:
            host, port_str = qdrant_url.rsplit(":", 1)
            port = int(port_str)
        else:
            host = qdrant_url
            port = 6333
        
        # Initialize vector database client
        if self.config.enable_privacy:
            self.vector_db = PrivacyPreservingVectorDB(
                host=host,
                port=port,
                api_key=api_key,
                epsilon=self.config.privacy_epsilon,
                use_gpu=self.config.use_gpu,
                embedding_dim=self.embedding_service.embedding_dim
            )
        else:
            self.vector_db = QdrantClient(
                host=host,
                port=port,
                api_key=api_key,
                use_gpu=self.config.use_gpu,
                embedding_dim=self.embedding_service.embedding_dim
            )
        
        # Initialize document chunker
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            overlap=self.config.chunk_overlap
        )
        
        self.logger.info(
            f"RAG Pipeline initialized - "
            f"Model: {self.config.embedding_model}, "
            f"GPU: {self.config.use_gpu}, "
            f"Privacy: {self.config.enable_privacy}"
        )
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Index documents into vector database
        
        Args:
            documents: List of documents with 'text' and optional 'metadata'
            collection_name: Optional collection name (default from config)
            
        Returns:
            Indexing statistics
        """
        collection = collection_name or self.config.collection_name
        
        # Create collection if it doesn't exist
        vector_size = self.embedding_service.embedding_dim
        self.vector_db.create_collection(collection, vector_size)
        
        # Chunk all documents
        all_chunks = []
        for doc_id, doc in enumerate(documents):
            doc_metadata = doc.get('metadata', {})
            doc_metadata['doc_id'] = doc_id
            
            chunks = self.chunker.chunk_text(doc['text'], doc_metadata)
            all_chunks.extend(chunks)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk['text'] for chunk in all_chunks]
        embeddings = self.embedding_service.embed(chunk_texts)
        
        # Prepare vectors with metadata
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(all_chunks, embeddings)):
            vectors.append({
                'id': i,
                'vector': embedding,
                'payload': chunk['metadata']
            })
        
        # Insert into vector database
        self.vector_db.insert_vectors(collection, vectors)
        
        stats = {
            'documents': len(documents),
            'chunks': len(all_chunks),
            'collection': collection,
            'vector_size': vector_size,
            'privacy_enabled': self.config.enable_privacy
        }
        
        self.logger.info(f"Indexed {stats['documents']} docs → {stats['chunks']} chunks")
        return stats
    
    def retrieve(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: Search query
            collection_name: Optional collection name
            top_k: Number of results to return
            min_score: Minimum similarity score
            
        Returns:
            List of retrieved documents with scores
        """
        collection = collection_name or self.config.collection_name
        k = top_k or self.config.top_k
        threshold = min_score or self.config.min_score
        
        # Search vector database
        results = self.vector_db.search(
            collection_name=collection,
            query_text=query,
            limit=k
        )
        
        # Filter by minimum score
        filtered_results = [
            r for r in results
            if r.get('score', 0) >= threshold
        ]
        
        self.logger.info(
            f"Retrieved {len(filtered_results)}/{len(results)} docs "
            f"(threshold: {threshold})"
        )
        
        return filtered_results
    
    def generate_context(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Generate context string for LLM from retrieved documents
        
        Args:
            query: User query
            collection_name: Optional collection name
            top_k: Number of documents to retrieve
            
        Returns:
            Tuple of (context_string, retrieved_documents)
        """
        # Retrieve relevant documents
        docs = self.retrieve(query, collection_name, top_k)
        
        if not docs:
            return "No relevant context found.", []
        
        # Build context string
        context_parts = ["Context Information:"]
        for i, doc in enumerate(docs, 1):
            score = doc.get('score', 0)
            text = doc.get('text', '')
            metadata = doc.get('payload', {})
            
            context_parts.append(
                f"\n[Document {i}] (relevance: {score:.2f})\n{text}"
            )
        
        context = "\n".join(context_parts)
        return context, docs
    
    def query(
        self,
        user_query: str,
        collection_name: Optional[str] = None,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Full RAG query pipeline
        
        Args:
            user_query: User's question/query
            collection_name: Optional collection name
            include_metadata: Include retrieval metadata in response
            
        Returns:
            Dictionary with context, retrieved docs, and metadata
        """
        context, docs = self.generate_context(user_query, collection_name)
        
        response = {
            'query': user_query,
            'context': context,
            'num_docs': len(docs)
        }
        
        if include_metadata:
            response['retrieved_documents'] = docs
            response['config'] = {
                'top_k': self.config.top_k,
                'min_score': self.config.min_score,
                'privacy_enabled': self.config.enable_privacy,
                'model': self.config.embedding_model
            }
        
        return response
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics"""
        return self.embedding_service.get_cache_stats()
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.embedding_service.clear_cache()
        self.logger.info("Embedding cache cleared")


class HybridSearch:
    """
    Hybrid search combining:
    - Dense vector similarity (embeddings)
    - Sparse keyword matching (BM25/TF-IDF)
    - GPU-accelerated ranking fusion
    """
    
    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3
    ):
        """
        Initialize hybrid search
        
        Args:
            rag_pipeline: RAG pipeline for dense retrieval
            dense_weight: Weight for vector similarity
            sparse_weight: Weight for keyword matching
        """
        self.rag = rag_pipeline
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.logger = logging.getLogger(__name__)
    
    def search(
        self,
        query: str,
        collection_name: Optional[str] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search with dense + sparse retrieval
        
        Args:
            query: Search query
            collection_name: Collection to search
            top_k: Number of results
            
        Returns:
            Ranked results from hybrid search
        """
        # Dense vector search
        dense_results = self.rag.retrieve(
            query,
            collection_name,
            top_k=top_k * 2  # Get more for re-ranking
        )
        
        # TODO: Implement sparse keyword search (BM25)
        # For now, just use dense results with adjusted weights
        
        # Re-rank with combined scores
        for result in dense_results:
            dense_score = result.get('score', 0)
            # sparse_score would come from keyword matching
            sparse_score = 0  # Placeholder
            
            result['hybrid_score'] = (
                self.dense_weight * dense_score +
                self.sparse_weight * sparse_score
            )
        
        # Sort by hybrid score and take top k
        ranked = sorted(
            dense_results,
            key=lambda x: x.get('hybrid_score', 0),
            reverse=True
        )[:top_k]
        
        return ranked


# Global RAG pipeline instance
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline(
    config: Optional[RAGConfig] = None,
    force_new: bool = False
) -> RAGPipeline:
    """
    Get global RAG pipeline instance (singleton pattern)
    
    Args:
        config: Optional RAG configuration
        force_new: Force creation of new instance
        
    Returns:
        Global RAG pipeline instance
    """
    global _rag_pipeline
    
    if _rag_pipeline is None or force_new:
        _rag_pipeline = RAGPipeline(config)
    
    return _rag_pipeline
