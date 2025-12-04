"""
Test suite for NLP Service
===========================

Tests natural language processing with privacy preservation.
"""

import pytest
import numpy as np

from services.nlp.nlp_engine import SecureNLPEngine


class TestSecureNLPEngine:
    """Test NLP engine functionality"""
    
    @pytest.fixture
    def nlp_engine(self):
        """Create NLP engine fixture"""
        return SecureNLPEngine()
    
    def test_initialization(self, nlp_engine):
        """Test NLP engine initializes correctly"""
        assert nlp_engine is not None
        assert nlp_engine.encryption is not None
        assert nlp_engine.dp is not None
    
    def test_sentiment_analysis(self, nlp_engine):
        """Test sentiment analysis"""
        result = nlp_engine.analyze_sentiment(
            "This is a wonderful and amazing product!"
        )
        
        assert 'sentiment' in result
        assert 'score' in result
        assert result['sentiment'] in ['positive', 'negative', 'neutral']
        assert 0 <= result['score'] <= 1
    
    def test_entity_extraction(self, nlp_engine):
        """Test named entity recognition"""
        text = "Apple Inc. is located in Cupertino, California."
        entities = nlp_engine.extract_entities(text)
        
        assert isinstance(entities, list)
        # At least should detect organization or location
        if entities:
            assert 'text' in entities[0]
            assert 'type' in entities[0]
    
    def test_text_generation(self, nlp_engine):
        """Test text generation"""
        result = nlp_engine.generate_text(
            "The future of AI is",
            max_length=50,
            temperature=0.7
        )
        
        assert 'generated_text' in result
        assert len(result['generated_text']) > 0
        assert 'generation_time_ms' in result
    
    def test_process_text_classification(self, nlp_engine):
        """Test text processing with classification"""
        result = nlp_engine.process_text(
            "AI-Nexus is revolutionary!",
            task_type="classification",
            compliance_mode="HIPAA",
            security_level="high"
        )
        
        assert result.result is not None
        assert 0 <= result.confidence <= 1
        assert result.processing_time_ms > 0
        assert result.model_version is not None
    
    def test_privacy_preservation_hipaa(self, nlp_engine):
        """Test HIPAA compliance mode"""
        text = "Patient John Doe, SSN 123-45-6789, email john@example.com"
        result = nlp_engine.process_text(
            text,
            compliance_mode="HIPAA",
            security_level="high"
        )
        
        assert 'compliance_proof' in result.metadata
    
    def test_get_capabilities(self, nlp_engine):
        """Test getting model capabilities"""
        capabilities = nlp_engine.get_capabilities()
        
        assert 'supported_tasks' in capabilities
        assert 'supported_languages' in capabilities
        assert 'compliance_modes' in capabilities
        assert 'classification' in capabilities['supported_tasks']
        assert 'HIPAA' in capabilities['compliance_modes']
    
    def test_batch_processing(self, nlp_engine):
        """Test processing multiple texts"""
        texts = [
            "This is great!",
            "This is terrible.",
            "This is okay."
        ]
        
        results = []
        for text in texts:
            result = nlp_engine.analyze_sentiment(text)
            results.append(result)
        
        assert len(results) == 3
        assert all('sentiment' in r for r in results)
    
    def test_explainability(self, nlp_engine):
        """Test model explainability"""
        result = nlp_engine.process_text(
            "AI-Nexus provides excellent privacy protection.",
            task_type="classification",
            enable_explainability=True
        )
        
        # Should have some explanation (even if mock)
        assert isinstance(result.explanation, dict)


@pytest.mark.integration
class TestNLPIntegration:
    """Integration tests for NLP service"""
    
    def test_end_to_end_processing(self):
        """Test complete NLP pipeline"""
        engine = SecureNLPEngine()
        
        # Process text
        result = engine.process_text(
            "The AI-Nexus platform is secure and decentralized.",
            task_type="sentiment",
            compliance_mode="GDPR",
            enable_explainability=True
        )
        
        # Verify all components worked
        assert result.result is not None
        assert result.confidence > 0
        assert result.processing_time_ms > 0
        assert 'compliance_proof' in result.metadata


class TestNLPPerformance:
    """Performance tests (non-benchmark)"""
    
    def test_sentiment_performance(self):
        """Test sentiment analysis performance"""
        engine = SecureNLPEngine()
        text = "This is a test sentence for benchmarking performance."
        
        import time
        start = time.time()
        result = engine.analyze_sentiment(text)
        duration = time.time() - start
        
        assert 'sentiment' in result
        assert duration < 5.0
    
    def test_entity_extraction_performance(self):
        """Test entity extraction performance"""
        engine = SecureNLPEngine()
        text = "Apple Inc. and Microsoft Corporation are in the United States."
        
        import time
        start = time.time()
        result = engine.extract_entities(text)
        duration = time.time() - start
        
        assert isinstance(result, list)
        assert duration < 5.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
