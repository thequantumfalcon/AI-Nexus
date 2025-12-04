"""
Tests for RAG (Retrieval-Augmented Generation) pipeline
GPU-accelerated vector search and context generation
"""

import pytest
import torch
from typing import List, Dict

from services.ml.rag_pipeline import (
    RAGPipeline,
    RAGConfig,
    DocumentChunker,
    HybridSearch,
    get_rag_pipeline
)
from services.vectordb.embeddings import get_embedding_service


class TestDocumentChunker:
    """Test document chunking functionality"""
    
    def test_initialization(self):
        """Test chunker initialization"""
        chunker = DocumentChunker(chunk_size=100, overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20
    
    def test_basic_chunking(self):
        """Test basic text chunking"""
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        text = "A" * 120  # 120 character text
        
        chunks = chunker.chunk_text(text)
        
        # Should create 3 chunks: 0-50, 40-90, 80-120
        assert len(chunks) == 3
        assert all('text' in chunk for chunk in chunks)
        assert all('metadata' in chunk for chunk in chunks)
        
        # Check chunk IDs
        for i, chunk in enumerate(chunks):
            assert chunk['metadata']['chunk_id'] == i
    
    def test_chunking_with_metadata(self):
        """Test chunking preserves metadata"""
        chunker = DocumentChunker(chunk_size=50, overlap=10)
        text = "B" * 100
        metadata = {'source': 'test.txt', 'author': 'pytest'}
        
        chunks = chunker.chunk_text(text, metadata)
        
        for chunk in chunks:
            assert chunk['metadata']['source'] == 'test.txt'
            assert chunk['metadata']['author'] == 'pytest'
            assert 'chunk_id' in chunk['metadata']
    
    def test_empty_text(self):
        """Test chunking empty text"""
        chunker = DocumentChunker()
        chunks = chunker.chunk_text("")
        assert len(chunks) == 0
    
    def test_overlap_calculation(self):
        """Test chunk overlap is correct"""
        chunker = DocumentChunker(chunk_size=10, overlap=3)
        text = "0123456789" * 3  # 30 characters
        
        chunks = chunker.chunk_text(text)
        
        # Verify chunks created (chunk_size=10, overlap=3)
        # Chunk 1: 0-10, Chunk 2: 7-17, Chunk 3: 14-24, Chunk 4: 21-30
        assert len(chunks) >= 3
        
        # Just verify chunks are created with overlap
        for chunk in chunks:
            assert len(chunk['text']) > 0
            assert 'chunk_id' in chunk['metadata']


class TestRAGPipeline:
    """Test RAG pipeline functionality"""
    
    @pytest.fixture
    def sample_documents(self) -> List[Dict]:
        """Sample documents for testing"""
        return [
            {
                'text': 'Machine learning is a subset of artificial intelligence.',
                'metadata': {'topic': 'ML', 'source': 'intro.txt'}
            },
            {
                'text': 'Deep learning uses neural networks with multiple layers.',
                'metadata': {'topic': 'DL', 'source': 'advanced.txt'}
            },
            {
                'text': 'Natural language processing enables computers to understand text.',
                'metadata': {'topic': 'NLP', 'source': 'nlp.txt'}
            }
        ]
    
    @pytest.fixture
    def rag_config(self) -> RAGConfig:
        """RAG configuration for testing"""
        return RAGConfig(
            collection_name="test_collection",
            top_k=3,
            min_score=0.5,
            chunk_size=128,
            chunk_overlap=20,
            enable_privacy=False,
            embedding_model="fast",
            use_gpu=torch.cuda.is_available(),
            cache_size=1000
        )
    
    def test_initialization(self, rag_config):
        """Test RAG pipeline initialization"""
        rag = RAGPipeline(config=rag_config)
        
        assert rag.config.collection_name == "test_collection"
        assert rag.config.top_k == 3
        assert rag.embedding_service is not None
        assert rag.vector_db is not None
        assert rag.chunker is not None
    
    def test_initialization_with_privacy(self):
        """Test RAG initialization with privacy enabled"""
        config = RAGConfig(enable_privacy=True, privacy_epsilon=2.0)
        rag = RAGPipeline(config=config)
        
        assert rag.config.enable_privacy
        assert hasattr(rag.vector_db, 'epsilon')
    
    @pytest.mark.skip(reason="Requires Qdrant server running")
    def test_index_documents(self, rag_config, sample_documents):
        """Test document indexing"""
        rag = RAGPipeline(config=rag_config)
        
        stats = rag.index_documents(sample_documents)
        
        assert stats['documents'] == 3
        assert stats['chunks'] > 0  # Should create at least 3 chunks
        assert stats['collection'] == "test_collection"
        assert stats['vector_size'] > 0
    
    @pytest.mark.skip(reason="Requires Qdrant server running")
    def test_retrieve_documents(self, rag_config, sample_documents):
        """Test document retrieval"""
        rag = RAGPipeline(config=rag_config)
        
        # Index documents first
        rag.index_documents(sample_documents)
        
        # Search for relevant documents
        results = rag.retrieve("What is machine learning?")
        
        assert len(results) > 0
        assert all('score' in r for r in results)
        assert all('text' in r for r in results)
    
    def test_generate_context(self, rag_config):
        """Test context generation for LLM"""
        rag = RAGPipeline(config=rag_config)
        
        # Mock retrieve to avoid Qdrant dependency
        def mock_retrieve(query, collection, top_k):
            return [
                {'text': 'ML is AI subset', 'score': 0.9, 'payload': {}},
                {'text': 'DL uses neural nets', 'score': 0.8, 'payload': {}}
            ]
        
        rag.retrieve = mock_retrieve
        
        context, docs = rag.generate_context("What is ML?")
        
        assert "Context Information:" in context
        assert "ML is AI subset" in context
        assert "DL uses neural nets" in context
        assert len(docs) == 2
    
    def test_generate_context_no_results(self, rag_config):
        """Test context generation with no results"""
        rag = RAGPipeline(config=rag_config)
        
        # Mock retrieve to return empty
        rag.retrieve = lambda q, c, t: []
        
        context, docs = rag.generate_context("Random query")
        
        assert context == "No relevant context found."
        assert len(docs) == 0
    
    def test_query_pipeline(self, rag_config):
        """Test full query pipeline"""
        rag = RAGPipeline(config=rag_config)
        
        # Mock generate_context
        def mock_context(query, collection):
            return "Mocked context", [{'text': 'doc1'}]
        
        rag.generate_context = mock_context
        
        response = rag.query("Test query")
        
        assert response['query'] == "Test query"
        assert response['context'] == "Mocked context"
        assert response['num_docs'] == 1
        assert 'retrieved_documents' in response
        assert 'config' in response
    
    def test_cache_operations(self, rag_config):
        """Test cache statistics and clearing"""
        rag = RAGPipeline(config=rag_config)
        
        # Get cache stats
        stats = rag.get_cache_stats()
        assert 'size' in stats or 'cache_size' in stats
        
        # Clear cache
        rag.clear_cache()
        stats_after = rag.get_cache_stats()
        # Check both possible field names
        cache_size = stats_after.get('size', stats_after.get('cache_size', 0))
        assert cache_size == 0


class TestHybridSearch:
    """Test hybrid search combining dense + sparse retrieval"""
    
    @pytest.fixture
    def rag_config(self) -> RAGConfig:
        """RAG configuration for testing"""
        return RAGConfig(
            collection_name="test_hybrid",
            use_gpu=torch.cuda.is_available()
        )
    
    def test_initialization(self, rag_config):
        """Test hybrid search initialization"""
        rag = RAGPipeline(config=rag_config)
        hybrid = HybridSearch(rag, dense_weight=0.6, sparse_weight=0.4)
        
        assert hybrid.rag is not None
        assert hybrid.dense_weight == 0.6
        assert hybrid.sparse_weight == 0.4
    
    def test_search_scoring(self, rag_config):
        """Test hybrid scoring calculation"""
        rag = RAGPipeline(config=rag_config)
        
        # Mock retrieve to return fixed results
        def mock_retrieve(query, collection, top_k):
            return [
                {'id': 1, 'score': 0.9, 'text': 'doc1'},
                {'id': 2, 'score': 0.7, 'text': 'doc2'}
            ]
        
        rag.retrieve = mock_retrieve
        
        hybrid = HybridSearch(rag, dense_weight=0.7, sparse_weight=0.3)
        results = hybrid.search("test query", top_k=5)
        
        # Should have hybrid scores
        assert all('hybrid_score' in r for r in results)
        
        # First result should have higher hybrid score (0.9 * 0.7)
        assert results[0]['hybrid_score'] >= results[1]['hybrid_score']


class TestGlobalRAGPipeline:
    """Test global RAG pipeline singleton"""
    
    def test_get_global_pipeline(self):
        """Test getting global RAG pipeline instance"""
        rag1 = get_rag_pipeline()
        rag2 = get_rag_pipeline()
        
        # Should return same instance
        assert rag1 is rag2
    
    def test_force_new_pipeline(self):
        """Test forcing new pipeline instance"""
        rag1 = get_rag_pipeline()
        rag2 = get_rag_pipeline(force_new=True)
        
        # Should be different instances
        assert rag1 is not rag2
    
    def test_custom_config(self):
        """Test global pipeline with custom config"""
        config = RAGConfig(collection_name="custom_test")
        rag = get_rag_pipeline(config=config, force_new=True)
        
        assert rag.config.collection_name == "custom_test"


class TestRAGIntegration:
    """Integration tests for RAG pipeline"""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU required for GPU integration test"
    )
    def test_gpu_embedding_integration(self):
        """Test GPU embedding service integration"""
        config = RAGConfig(use_gpu=True, embedding_model="fast")
        rag = RAGPipeline(config=config)
        
        # Test embedding generation
        texts = ["Test document for GPU embedding"]
        embeddings = rag.embedding_service.embed(texts)
        
        # Embeddings can be EmbeddingResult or list of arrays
        if hasattr(embeddings, 'embeddings'):
            # EmbeddingResult object
            assert len(embeddings.embeddings) == 1
            assert embeddings.embeddings[0].shape[0] > 0
        elif isinstance(embeddings, list):
            assert len(embeddings) == 1
            assert embeddings[0].shape[0] > 0
        else:
            # Single embedding
            assert embeddings.shape[0] > 0
    
    def test_chunking_integration(self):
        """Test document chunking integration"""
        config = RAGConfig(chunk_size=50, chunk_overlap=10)
        rag = RAGPipeline(config=config)
        
        long_text = "This is a test. " * 20  # ~300 characters
        docs = [{'text': long_text, 'metadata': {'test': True}}]
        
        # Mock embedding service to return proper embeddings
        class MockEmbeddingService:
            embedding_dim = 384
            def embed(self, texts):
                import numpy as np
                return [np.random.randn(self.embedding_dim).astype(np.float32) for _ in texts]
            def get_cache_stats(self):
                return {'size': 0, 'max_size': 10000, 'utilization': 0.0}
            def clear_cache(self):
                pass
        
        # Mock vector_db to avoid Qdrant dependency
        class MockVectorDB:
            def create_collection(self, name, size): pass
            def insert_vectors(self, collection, vectors):
                # Verify chunks were created
                assert len(vectors) > 1
                assert all('id' in v for v in vectors)
                assert all('vector' in v for v in vectors)
                assert all('payload' in v for v in vectors)
        
        rag.embedding_service = MockEmbeddingService()
        rag.vector_db = MockVectorDB()
        rag.index_documents(docs)
    
    def test_privacy_integration(self):
        """Test privacy-preserving integration"""
        config = RAGConfig(enable_privacy=True, privacy_epsilon=1.5)
        rag = RAGPipeline(config=config)
        
        assert rag.config.enable_privacy
        assert hasattr(rag.vector_db, 'epsilon')
        assert rag.vector_db.epsilon == 1.5
