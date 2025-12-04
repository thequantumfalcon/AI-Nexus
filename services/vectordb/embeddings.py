"""
GPU-Accelerated Embedding Service for Vector Database
Generates high-quality embeddings using transformer models on GPU
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingResult:
    """Result from embedding generation"""
    embeddings: np.ndarray
    processing_time: float
    model_name: str
    dimension: int


class GPUEmbeddingService:
    """
    GPU-accelerated embedding generation service
    
    Features:
    - Multiple embedding models
    - Batch processing for efficiency
    - GPU acceleration
    - Caching for repeated queries
    """
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        use_gpu: bool = True,
        cache_size: int = 10000
    ):
        """
        Initialize embedding service
        
        Args:
            model_name: HuggingFace model name
            use_gpu: Use GPU if available
            cache_size: Maximum cache entries
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.cache_size = cache_size
        self.cache: Dict[str, np.ndarray] = {}
        
        self._initialize_model()
        
        logger.info(f"Embedding service initialized: {model_name}, GPU={use_gpu}")
    
    def _initialize_model(self):
        """Initialize embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Determine device
            if self.use_gpu and torch.cuda.is_available():
                device = 'cuda'
                logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = 'cpu'
                logger.info("Using CPU for embeddings")
            
            # Load model
            self.model = SentenceTransformer(self.model_name, device=device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            
            logger.info(f"Model loaded: {self.model_name} (dim={self.embedding_dim})")
            
        except ImportError:
            logger.error(
                "sentence-transformers not installed. Install with:\n"
                "pip install sentence-transformers"
            )
            raise
    
    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False,
        use_cache: bool = True
    ) -> EmbeddingResult:
        """
        Generate embeddings for texts
        
        Args:
            texts: List of text strings
            batch_size: Processing batch size
            normalize: L2 normalize embeddings
            show_progress: Show progress bar
            use_cache: Use cached embeddings
        
        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.perf_counter()
        
        # Check cache
        if use_cache:
            cached_embeddings = []
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self.cache:
                    cached_embeddings.append(self.cache[text])
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # If all cached, return immediately
            if len(uncached_texts) == 0:
                embeddings = np.array(cached_embeddings)
                processing_time = time.perf_counter() - start_time
                
                return EmbeddingResult(
                    embeddings=embeddings,
                    processing_time=processing_time,
                    model_name=self.model_name,
                    dimension=self.embedding_dim
                )
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate new embeddings
        if len(uncached_texts) > 0:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            # Update cache
            if use_cache:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    if len(self.cache) < self.cache_size:
                        self.cache[text] = embedding
        
        # Combine cached and new embeddings
        if use_cache and len(cached_embeddings) > 0:
            all_embeddings = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
            
            # Fill cached
            cached_idx = 0
            for i in range(len(texts)):
                if i not in uncached_indices:
                    all_embeddings[i] = cached_embeddings[cached_idx]
                    cached_idx += 1
            
            # Fill new
            for i, idx in enumerate(uncached_indices):
                all_embeddings[idx] = new_embeddings[i]
            
            embeddings = all_embeddings
        else:
            embeddings = new_embeddings
        
        processing_time = time.perf_counter() - start_time
        
        logger.info(
            f"Generated {len(embeddings)} embeddings in {processing_time:.3f}s "
            f"({len(embeddings)/processing_time:.1f} texts/sec)"
        )
        
        return EmbeddingResult(
            embeddings=embeddings.astype(np.float32),
            processing_time=processing_time,
            model_name=self.model_name,
            dimension=self.embedding_dim
        )
    
    def embed_single(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding for a single text"""
        result = self.embed([text], **kwargs)
        return result.embeddings[0]
    
    def similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric (cosine, dot, euclidean)
        
        Returns:
            Similarity score
        """
        embeddings = self.embed([text1, text2]).embeddings
        
        if metric == "cosine":
            # Cosine similarity (already normalized)
            return float(np.dot(embeddings[0], embeddings[1]))
        
        elif metric == "dot":
            # Dot product
            return float(np.dot(embeddings[0], embeddings[1]))
        
        elif metric == "euclidean":
            # Negative Euclidean distance (higher = more similar)
            return float(-np.linalg.norm(embeddings[0] - embeddings[1]))
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "size": len(self.cache),
            "max_size": self.cache_size,
            "utilization": len(self.cache) / self.cache_size if self.cache_size > 0 else 0.0
        }


class MultiModelEmbeddingService:
    """
    Service supporting multiple embedding models
    Useful for different use cases (speed vs quality)
    """
    
    def __init__(self, use_gpu: bool = True):
        """Initialize multi-model service"""
        self.use_gpu = use_gpu
        self.models: Dict[str, GPUEmbeddingService] = {}
        
        # Pre-load common models
        self._load_default_models()
    
    def _load_default_models(self):
        """Load commonly used models"""
        model_configs = [
            ("all-MiniLM-L6-v2", "fast"),      # 384dim, very fast
            ("all-mpnet-base-v2", "balanced"), # 768dim, balanced
            ("all-MiniLM-L12-v2", "quality"),  # 384dim, better quality
        ]
        
        for model_name, alias in model_configs:
            try:
                service = GPUEmbeddingService(
                    model_name=model_name,
                    use_gpu=self.use_gpu
                )
                self.models[alias] = service
                logger.info(f"Loaded model '{alias}': {model_name}")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
    
    def embed(
        self,
        texts: List[str],
        model: str = "fast",
        **kwargs
    ) -> EmbeddingResult:
        """
        Generate embeddings using specified model
        
        Args:
            texts: List of texts
            model: Model alias (fast, balanced, quality)
            **kwargs: Additional arguments
        
        Returns:
            EmbeddingResult
        """
        if model not in self.models:
            raise ValueError(f"Unknown model: {model}. Available: {list(self.models.keys())}")
        
        return self.models[model].embed(texts, **kwargs)
    
    def get_available_models(self) -> List[str]:
        """Get list of available model aliases"""
        return list(self.models.keys())


# Global embedding service instance
_embedding_service = None


def get_embedding_service(
    model_type: str = "balanced",
    use_gpu: bool = True,
    cache_size: int = 10000
) -> GPUEmbeddingService:
    """
    Get global embedding service instance
    
    Args:
        model_type: Model tier (fast, balanced, quality) - ignored for now
        use_gpu: Use GPU acceleration
        cache_size: Cache size - ignored for now
    
    Returns:
        Global embedding service instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = GPUEmbeddingService(use_gpu=use_gpu)
    return _embedding_service
