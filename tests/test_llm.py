"""
Tests for LLM Service
GPU-accelerated text generation with quantization
"""

import pytest
import torch
from typing import List

from services.ml.llm_service import (
    LLMService,
    LLMConfig,
    GenerationConfig,
    get_llm_service
)
from services.ml.rag_llm_service import (
    RAGLLMService,
    RAGLLMConfig,
    get_rag_llm_service
)


class TestLLMConfig:
    """Test LLM configuration"""
    
    def test_default_config(self):
        """Test default configuration"""
        config = LLMConfig()
        
        assert config.model_name == "meta-llama/Llama-3.3-70B-Instruct"
        assert config.load_in_4bit == True
        assert config.max_length == 4096
        assert config.temperature == 0.7
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = LLMConfig(
            model_name="custom-model",
            load_in_4bit=False,
            load_in_8bit=True,
            temperature=0.9
        )
        
        assert config.model_name == "custom-model"
        assert config.load_in_4bit == False
        assert config.load_in_8bit == True
        assert config.temperature == 0.9


class TestGenerationConfig:
    """Test generation configuration"""
    
    def test_default_generation_config(self):
        """Test default generation settings"""
        config = GenerationConfig()
        
        assert config.max_new_tokens == 512
        assert config.temperature == 0.7
        assert config.do_sample == True
    
    def test_custom_generation_config(self):
        """Test custom generation settings"""
        config = GenerationConfig(
            max_new_tokens=1024,
            temperature=0.5,
            top_p=0.95,
            do_sample=False
        )
        
        assert config.max_new_tokens == 1024
        assert config.temperature == 0.5
        assert config.top_p == 0.95
        assert config.do_sample == False


class TestLLMService:
    """Test LLM service functionality"""
    
    @pytest.fixture
    def llm_config(self) -> LLMConfig:
        """LLM configuration for testing (mock mode)"""
        return LLMConfig(
            model_name="test-model",
            load_in_4bit=False,
            load_in_8bit=False
        )
    
    def test_initialization(self, llm_config):
        """Test LLM service initialization"""
        llm = LLMService(
            config=llm_config,
            use_local_model=True  # Don't download model
        )
        
        assert llm.config.model_name == "test-model"
        assert llm.device is not None
    
    def test_gpu_check(self, llm_config):
        """Test GPU availability check"""
        llm = LLMService(config=llm_config, use_local_model=True)
        
        if torch.cuda.is_available():
            assert llm.device.type == "cuda"
        else:
            assert llm.device.type == "cpu"
    
    def test_mock_generation(self, llm_config):
        """Test text generation with mock model"""
        llm = LLMService(config=llm_config, use_local_model=True)
        
        # Mock model returns mock response
        result = llm.generate("Test prompt")
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert "Mock response" in result or "Test prompt" in result
    
    def test_mock_streaming(self, llm_config):
        """Test streaming generation with mock model"""
        llm = LLMService(config=llm_config, use_local_model=True)
        
        tokens = []
        for token in llm.generate_stream("Test prompt"):
            tokens.append(token)
        
        assert len(tokens) > 0
        full_text = "".join(tokens)
        assert len(full_text) > 0
    
    def test_mock_batch_generation(self, llm_config):
        """Test batch generation with mock model"""
        llm = LLMService(config=llm_config, use_local_model=True)
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = llm.batch_generate(prompts)
        
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
    
    def test_mock_chat(self, llm_config):
        """Test chat completion with mock model"""
        llm = LLMService(config=llm_config, use_local_model=True)
        
        messages = [
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"}
        ]
        
        response = llm.chat(messages)
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_get_model_info(self, llm_config):
        """Test model information retrieval"""
        llm = LLMService(config=llm_config, use_local_model=True)
        
        info = llm.get_model_info()
        
        assert 'status' in info or 'model_name' in info
        assert info.get('model_name') == "test-model" or info.get('status') == "mock"
    
    def test_generation_with_config(self, llm_config):
        """Test generation with custom config"""
        llm = LLMService(config=llm_config, use_local_model=True)
        
        gen_config = GenerationConfig(
            max_new_tokens=100,
            temperature=0.5
        )
        
        result = llm.generate("Test", gen_config)
        
        assert isinstance(result, str)


class TestRAGLLMService:
    """Test RAG-LLM integration"""
    
    @pytest.fixture
    def rag_llm_config(self) -> RAGLLMConfig:
        """RAG-LLM configuration for testing"""
        llm_config = LLMConfig(
            model_name="test-model",
            load_in_4bit=False
        )
        
        return RAGLLMConfig(
            llm_config=llm_config,
            use_rag=True
        )
    
    def test_initialization(self, rag_llm_config):
        """Test RAG-LLM service initialization"""
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        assert rag_llm.config.use_rag == True
        assert rag_llm.rag is not None
        assert rag_llm.llm is not None
    
    def test_query_without_rag(self, rag_llm_config):
        """Test query without RAG (pure LLM)"""
        rag_llm_config.use_rag = False
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        result = rag_llm.query("What is AI?", use_rag=False)
        
        assert 'query' in result
        assert 'answer' in result
        assert result['used_rag'] == False
    
    def test_query_with_rag_mock(self, rag_llm_config):
        """Test query with RAG (mocked retrieval)"""
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        # Mock RAG retrieval
        def mock_query(user_query, collection_name, include_metadata):
            return {
                'context': 'Mocked context',
                'num_docs': 1,
                'retrieved_documents': [
                    {'text': 'Mock doc', 'score': 0.9, 'payload': {}}
                ]
            }
        
        rag_llm.rag.query = mock_query
        
        result = rag_llm.query("Test question", use_rag=True)
        
        assert 'query' in result
        assert 'answer' in result
        assert result['used_rag'] == True
    
    def test_chat_without_rag(self, rag_llm_config):
        """Test chat without RAG"""
        rag_llm_config.use_rag = False
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        messages = [
            {"role": "user", "content": "Hello!"}
        ]
        
        result = rag_llm.chat(messages, use_rag=False)
        
        assert 'answer' in result
        assert result['used_rag'] == False
    
    def test_ask_with_context(self, rag_llm_config):
        """Test asking with manual context"""
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        context = "AI stands for Artificial Intelligence."
        query = "What is AI?"
        
        answer = rag_llm.ask_with_context(query, context)
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_enable_disable_rag(self, rag_llm_config):
        """Test enabling/disabling RAG"""
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        # Initially enabled
        assert rag_llm.config.use_rag == True
        
        # Disable
        rag_llm.disable_rag()
        assert rag_llm.config.use_rag == False
        
        # Enable
        rag_llm.enable_rag()
        assert rag_llm.config.use_rag == True
    
    def test_get_status(self, rag_llm_config):
        """Test status retrieval"""
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        status = rag_llm.get_status()
        
        assert 'rag_enabled' in status
        assert 'llm_model' in status
        assert 'llm_loaded' in status
    
    def test_streaming_mock(self, rag_llm_config):
        """Test streaming query"""
        rag_llm = RAGLLMService(config=rag_llm_config)
        
        chunks = list(rag_llm.query_stream("Test query", use_rag=False))
        
        assert len(chunks) > 0
        # Should have metadata, tokens, and done message
        assert any(chunk.get('type') == 'metadata' for chunk in chunks)
        assert any(chunk.get('type') == 'done' for chunk in chunks)


class TestGlobalLLMService:
    """Test global LLM service singleton"""
    
    def test_get_global_service(self):
        """Test getting global LLM service"""
        llm1 = get_llm_service()
        llm2 = get_llm_service()
        
        # Should return same instance
        assert llm1 is llm2
    
    def test_force_new_service(self):
        """Test forcing new service instance"""
        config = LLMConfig(load_in_4bit=False)
        llm1 = get_llm_service(config)
        llm2 = get_llm_service(config, force_new=True)
        
        # Should be different instances
        assert llm1 is not llm2


class TestGlobalRAGLLMService:
    """Test global RAG-LLM service singleton"""
    
    def test_get_global_rag_llm(self):
        """Test getting global RAG-LLM service"""
        rag_llm1 = get_rag_llm_service()
        rag_llm2 = get_rag_llm_service()
        
        # Should return same instance
        assert rag_llm1 is rag_llm2
    
    def test_force_new_rag_llm(self):
        """Test forcing new RAG-LLM instance"""
        rag_llm1 = get_rag_llm_service()
        rag_llm2 = get_rag_llm_service(force_new=True)
        
        # Should be different instances
        assert rag_llm1 is not rag_llm2


class TestLLMIntegration:
    """Integration tests for LLM service"""
    
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="GPU required for GPU integration test"
    )
    def test_gpu_device_allocation(self):
        """Test GPU device allocation"""
        config = LLMConfig(load_in_4bit=False)
        llm = LLMService(config=config, use_local_model=True)
        
        assert llm.device.type == "cuda"
    
    def test_context_template_formatting(self):
        """Test RAG context template formatting"""
        config = RAGLLMConfig()
        rag_llm = RAGLLMService(config=config)
        
        context = "Test context information"
        query = "Test query"
        
        formatted = rag_llm.config.context_template.format(
            context=context,
            query=query
        )
        
        assert context in formatted
        assert query in formatted
    
    def test_chat_message_formatting(self):
        """Test chat message formatting"""
        config = LLMConfig(load_in_4bit=False)
        llm = LLMService(config=config, use_local_model=True)
        
        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"}
        ]
        
        formatted = llm._format_chat_messages(messages)
        
        assert "System:" in formatted or "system" in formatted.lower()
        assert "User:" in formatted or "user" in formatted.lower()
    
    def test_rag_source_attribution(self):
        """Test RAG source attribution"""
        config = RAGLLMConfig(include_sources=True)
        rag_llm = RAGLLMService(config=config)
        
        # Mock RAG retrieval
        def mock_query(user_query, collection_name, include_metadata):
            return {
                'context': 'Context',
                'num_docs': 2,
                'retrieved_documents': [
                    {'text': 'Doc 1', 'score': 0.9, 'payload': {'source': 'file1.txt'}},
                    {'text': 'Doc 2', 'score': 0.8, 'payload': {'source': 'file2.txt'}}
                ]
            }
        
        rag_llm.rag.query = mock_query
        
        result = rag_llm.query("Test", use_rag=True)
        
        assert 'sources' in result
        assert len(result['sources']) == 2
        assert result['sources'][0]['metadata']['source'] == 'file1.txt'
