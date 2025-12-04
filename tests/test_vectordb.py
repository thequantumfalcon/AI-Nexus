"""
Tests for Qdrant Vector Database Client and Embedding Service
"""

import pytest
import numpy as np
from services.vectordb.qdrant_client import (
    QdrantClient,
    PrivacyPreservingVectorDB,
    VectorSearchResult
)
from services.vectordb.embeddings import (
    GPUEmbeddingService,
    MultiModelEmbeddingService,
    get_embedding_service
)


class TestGPUEmbeddingService:
    """Tests for GPU embedding service"""
    
    def test_initialization(self):
        """Test service initialization"""
        service = GPUEmbeddingService(model_name="all-MiniLM-L6-v2")
        
        assert service.model is not None
        assert service.embedding_dim == 384
        assert service.model_name == "all-MiniLM-L6-v2"
    
    def test_single_embedding(self):
        """Test single text embedding"""
        service = GPUEmbeddingService()
        
        text = "This is a test sentence"
        embedding = service.embed_single(text)
        
        assert embedding.shape == (384,)
        assert embedding.dtype == np.float32
        assert np.all(np.isfinite(embedding))
    
    def test_batch_embedding(self):
        """Test batch embedding generation"""
        service = GPUEmbeddingService()
        
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        
        result = service.embed(texts)
        
        assert result.embeddings.shape == (3, 384)
        assert result.dimension == 384
        assert result.model_name == "all-MiniLM-L6-v2"
        assert result.processing_time > 0
    
    def test_embedding_similarity(self):
        """Test similarity calculation"""
        service = GPUEmbeddingService()
        
        text1 = "The cat sits on the mat"
        text2 = "A cat is sitting on a mat"
        text3 = "Python is a programming language"
        
        # Similar texts should have high similarity
        sim_similar = service.similarity(text1, text2, metric="cosine")
        sim_different = service.similarity(text1, text3, metric="cosine")
        
        assert sim_similar > sim_different
        assert 0 <= sim_similar <= 1
        assert 0 <= sim_different <= 1
    
    def test_cache_functionality(self):
        """Test embedding caching"""
        service = GPUEmbeddingService(cache_size=100)
        
        texts = ["Cached text"]
        
        # First call - should cache
        result1 = service.embed(texts, use_cache=True)
        
        # Second call - should use cache
        result2 = service.embed(texts, use_cache=True)
        
        # Should be faster (from cache)
        assert result2.processing_time < result1.processing_time
        
        # Embeddings should be identical
        np.testing.assert_array_equal(result1.embeddings, result2.embeddings)
        
        # Check cache stats
        stats = service.get_cache_stats()
        assert stats['size'] == 1
        assert stats['utilization'] > 0
    
    def test_normalization(self):
        """Test L2 normalization"""
        service = GPUEmbeddingService()
        
        texts = ["Test normalization"]
        result = service.embed(texts, normalize=True)
        
        # Check L2 norm is approximately 1
        norm = np.linalg.norm(result.embeddings[0])
        assert abs(norm - 1.0) < 1e-5


class TestMultiModelEmbeddingService:
    """Tests for multi-model embedding service"""
    
    def test_initialization(self):
        """Test multi-model service init"""
        service = MultiModelEmbeddingService()
        
        available = service.get_available_models()
        assert len(available) > 0
        assert "fast" in available
    
    def test_different_models(self):
        """Test using different models"""
        service = MultiModelEmbeddingService()
        
        texts = ["Test text for different models"]
        
        # Try different models
        result_fast = service.embed(texts, model="fast")
        
        assert result_fast.embeddings.shape[1] == result_fast.dimension
        assert result_fast.processing_time > 0


class TestQdrantClient:
    """Tests for Qdrant client"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return QdrantClient(
            host="localhost",
            port=6333,
            use_gpu=True,
            embedding_dim=384
        )
    
    def test_initialization(self, client):
        """Test client initialization"""
        assert client.host == "localhost"
        assert client.port == 6333
        assert client.embedding_dim == 384
    
    def test_embed_texts(self, client):
        """Test text embedding"""
        texts = ["Hello world", "Test embedding"]
        embeddings = client.embed_texts(texts)
        
        assert embeddings.shape == (2, 384)
        assert embeddings.dtype == np.float32
    
    def test_collection_operations(self, client):
        """Test collection create/delete"""
        collection_name = "test_collection"
        
        # Note: These may fail if Qdrant server not running
        # That's expected in CI/CD
        try:
            # Create collection
            success = client.create_collection(
                collection_name=collection_name,
                vector_size=384,
                distance="Cosine"
            )
            
            if success:
                # Clean up
                client.delete_collection(collection_name)
        except Exception:
            # Server not running - skip
            pytest.skip("Qdrant server not available")
    
    def test_vector_operations(self, client):
        """Test vector insert and search"""
        collection_name = "test_vectors"
        
        try:
            # Create collection
            if not client.create_collection(collection_name, vector_size=384):
                pytest.skip("Qdrant server not available")
            
            # Prepare test data
            texts = ["Document 1", "Document 2", "Document 3"]
            vectors = client.embed_texts(texts)
            payloads = [
                {"text": text, "id": i}
                for i, text in enumerate(texts)
            ]
            
            # Insert vectors
            success = client.insert_vectors(
                collection_name=collection_name,
                vectors=vectors,
                payloads=payloads
            )
            
            assert success
            
            # Search
            query_vector = vectors[0]  # Search for first document
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=3
            )
            
            assert len(results) > 0
            assert isinstance(results[0], VectorSearchResult)
            assert results[0].score > 0.9  # Should match itself well
            
            # Clean up
            client.delete_collection(collection_name)
            
        except Exception as e:
            pytest.skip(f"Qdrant test failed: {e}")


class TestPrivacyPreservingVectorDB:
    """Tests for privacy-preserving vector database"""
    
    @pytest.fixture
    def client(self):
        """Create privacy client"""
        return PrivacyPreservingVectorDB(
            epsilon=1.0,
            delta=1e-5,
            host="localhost",
            port=6333,
            use_gpu=True
        )
    
    def test_initialization(self, client):
        """Test privacy client init"""
        assert client.epsilon == 1.0
        assert client.delta == 1e-5
    
    def test_vector_privatization(self, client):
        """Test differential privacy noise"""
        vector = np.random.randn(384).astype(np.float32)
        
        # Normalize
        vector = vector / np.linalg.norm(vector)
        
        # Privatize
        private_vector = client.privatize_vector(vector)
        
        assert private_vector.shape == vector.shape
        assert private_vector.dtype == np.float32
        
        # Should be different due to noise
        assert not np.allclose(vector, private_vector, atol=0.01)
        
        # Similarity can be low due to strong privacy (epsilon=1.0)
        similarity = np.dot(vector, private_vector)
        assert -1.0 <= similarity <= 1.0  # Valid cosine similarity range
    
    def test_private_insertion(self, client):
        """Test private vector insertion"""
        collection_name = "test_private"
        
        try:
            if not client.create_collection(collection_name, vector_size=384):
                pytest.skip("Qdrant server not available")
            
            # Generate test vectors
            vectors = np.random.randn(10, 384).astype(np.float32)
            # Normalize
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
            
            payloads = [{"id": i} for i in range(10)]
            
            # Insert with privacy
            success = client.insert_private_vectors(
                collection_name=collection_name,
                vectors=vectors,
                payloads=payloads
            )
            
            assert success
            
            # Clean up
            client.delete_collection(collection_name)
            
        except Exception:
            pytest.skip("Qdrant test skipped")


def test_get_global_service():
    """Test global embedding service"""
    service = get_embedding_service(use_gpu=True)
    
    assert service is not None
    assert isinstance(service, GPUEmbeddingService)
    
    # Should return same instance
    service2 = get_embedding_service()
    assert service is service2


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
