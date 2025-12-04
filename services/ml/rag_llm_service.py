"""
RAG-Enhanced LLM Service
Combines vector retrieval with LLM generation for context-aware responses
"""

import logging
from typing import List, Dict, Optional, Iterator, Any
from dataclasses import dataclass

from services.ml.rag_pipeline import RAGPipeline, RAGConfig, get_rag_pipeline
from services.ml.llm_service import LLMService, LLMConfig, GenerationConfig, get_llm_service


logger = logging.getLogger(__name__)


@dataclass
class RAGLLMConfig:
    """Combined configuration for RAG + LLM"""
    rag_config: Optional[RAGConfig] = None
    llm_config: Optional[LLMConfig] = None
    use_rag: bool = True
    include_sources: bool = True
    max_context_docs: int = 5
    context_template: str = """Context information from knowledge base:
{context}

Based on the above context, please answer the following question:
{query}

Answer:"""


class RAGLLMService:
    """
    RAG-Enhanced LLM Service
    
    Workflow:
    1. User query → RAG retrieval (vector search)
    2. Retrieved documents → Context formatting
    3. Context + Query → LLM generation
    4. LLM response → User
    
    Features:
    - Automatic context injection
    - Source attribution
    - Streaming responses
    - Fallback to pure LLM (no RAG)
    """
    
    def __init__(
        self,
        config: Optional[RAGLLMConfig] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        llm_service: Optional[LLMService] = None
    ):
        """
        Initialize RAG-LLM service
        
        Args:
            config: RAG-LLM configuration
            rag_pipeline: Optional RAG pipeline (uses global if not provided)
            llm_service: Optional LLM service (uses global if not provided)
        """
        self.config = config or RAGLLMConfig()
        
        # Initialize RAG pipeline
        if rag_pipeline is None:
            self.rag = get_rag_pipeline(self.config.rag_config)
        else:
            self.rag = rag_pipeline
        
        # Initialize LLM service
        if llm_service is None:
            self.llm = get_llm_service(self.config.llm_config)
        else:
            self.llm = llm_service
        
        logger.info(
            f"RAG-LLM service initialized - "
            f"RAG: {self.config.use_rag}, "
            f"LLM: {self.llm.config.model_name}"
        )
    
    def query(
        self,
        query: str,
        collection_name: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_rag: Optional[bool] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Query with RAG-enhanced LLM
        
        Args:
            query: User question
            collection_name: Collection to search
            generation_config: LLM generation config
            use_rag: Override RAG usage (default from config)
            stream: Stream response (not implemented in dict return)
            
        Returns:
            Response dict with answer, sources, and metadata
        """
        use_rag = use_rag if use_rag is not None else self.config.use_rag
        
        # Step 1: Retrieve context (if RAG enabled)
        context = ""
        retrieved_docs = []
        
        if use_rag:
            logger.info(f"Retrieving context for query: {query[:50]}...")
            rag_result = self.rag.query(
                user_query=query,
                collection_name=collection_name,
                include_metadata=True
            )
            
            context = rag_result['context']
            retrieved_docs = rag_result.get('retrieved_documents', [])
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Step 2: Format prompt with context
        if use_rag and retrieved_docs:
            prompt = self.config.context_template.format(
                context=context,
                query=query
            )
        else:
            prompt = query
        
        # Step 3: Generate response
        logger.info("Generating LLM response...")
        answer = self.llm.generate(
            prompt=prompt,
            generation_config=generation_config,
            return_full_text=False
        )
        
        # Step 4: Format response
        response = {
            'query': query,
            'answer': answer,
            'used_rag': use_rag and len(retrieved_docs) > 0
        }
        
        if self.config.include_sources and retrieved_docs:
            response['sources'] = [
                {
                    'text': doc.get('text', '')[:200],  # First 200 chars
                    'score': doc.get('score', 0),
                    'metadata': doc.get('payload', {})
                }
                for doc in retrieved_docs[:self.config.max_context_docs]
            ]
            response['num_sources'] = len(retrieved_docs)
        
        return response
    
    def query_stream(
        self,
        query: str,
        collection_name: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_rag: Optional[bool] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Query with streaming response
        
        Args:
            query: User question
            collection_name: Collection to search
            generation_config: LLM generation config
            use_rag: Override RAG usage
            
        Yields:
            Response chunks with metadata
        """
        use_rag = use_rag if use_rag is not None else self.config.use_rag
        
        # Retrieve context
        context = ""
        retrieved_docs = []
        
        if use_rag:
            rag_result = self.rag.query(
                user_query=query,
                collection_name=collection_name,
                include_metadata=True
            )
            
            context = rag_result['context']
            retrieved_docs = rag_result.get('retrieved_documents', [])
        
        # Format prompt
        if use_rag and retrieved_docs:
            prompt = self.config.context_template.format(
                context=context,
                query=query
            )
        else:
            prompt = query
        
        # Yield metadata first
        yield {
            'type': 'metadata',
            'query': query,
            'used_rag': use_rag and len(retrieved_docs) > 0,
            'num_sources': len(retrieved_docs)
        }
        
        # Yield sources if requested
        if self.config.include_sources and retrieved_docs:
            yield {
                'type': 'sources',
                'sources': [
                    {
                        'text': doc.get('text', '')[:200],
                        'score': doc.get('score', 0),
                        'metadata': doc.get('payload', {})
                    }
                    for doc in retrieved_docs[:self.config.max_context_docs]
                ]
            }
        
        # Stream answer tokens
        for token in self.llm.generate_stream(prompt, generation_config):
            yield {
                'type': 'token',
                'content': token
            }
        
        # Final message
        yield {
            'type': 'done'
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        collection_name: Optional[str] = None,
        generation_config: Optional[GenerationConfig] = None,
        use_rag: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Conversational chat with RAG
        
        Args:
            messages: Chat history [{"role": "user", "content": "..."}]
            collection_name: Collection to search
            generation_config: LLM generation config
            use_rag: Override RAG usage
            
        Returns:
            Response dict with answer and sources
        """
        use_rag = use_rag if use_rag is not None else self.config.use_rag
        
        # Extract last user message for RAG
        last_user_msg = None
        for msg in reversed(messages):
            if msg['role'] == 'user':
                last_user_msg = msg['content']
                break
        
        if last_user_msg is None:
            raise ValueError("No user message found in chat history")
        
        # Retrieve context
        retrieved_docs = []
        if use_rag:
            rag_result = self.rag.query(
                user_query=last_user_msg,
                collection_name=collection_name,
                include_metadata=True
            )
            
            context = rag_result['context']
            retrieved_docs = rag_result.get('retrieved_documents', [])
            
            # Inject context as system message
            if retrieved_docs:
                context_msg = {
                    'role': 'system',
                    'content': f"Context from knowledge base:\n{context}"
                }
                # Insert before last user message
                messages = messages[:-1] + [context_msg, messages[-1]]
        
        # Generate chat response
        answer = self.llm.chat(
            messages=messages,
            generation_config=generation_config
        )
        
        # Format response
        response = {
            'answer': answer,
            'used_rag': use_rag and len(retrieved_docs) > 0
        }
        
        if self.config.include_sources and retrieved_docs:
            response['sources'] = [
                {
                    'text': doc.get('text', '')[:200],
                    'score': doc.get('score', 0),
                    'metadata': doc.get('payload', {})
                }
                for doc in retrieved_docs[:self.config.max_context_docs]
            ]
        
        return response
    
    def ask_with_context(
        self,
        query: str,
        context: str,
        generation_config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Ask question with manually provided context
        
        Args:
            query: User question
            context: Context to use
            generation_config: Generation config
            
        Returns:
            Generated answer
        """
        prompt = self.config.context_template.format(
            context=context,
            query=query
        )
        
        return self.llm.generate(
            prompt=prompt,
            generation_config=generation_config,
            return_full_text=False
        )
    
    def disable_rag(self):
        """Disable RAG (use pure LLM)"""
        self.config.use_rag = False
        logger.info("RAG disabled - using pure LLM mode")
    
    def enable_rag(self):
        """Enable RAG"""
        self.config.use_rag = True
        logger.info("RAG enabled")
    
    def get_status(self) -> Dict[str, Any]:
        """Get service status"""
        return {
            'rag_enabled': self.config.use_rag,
            'llm_model': self.llm.config.model_name,
            'llm_loaded': self.llm.model is not None,
            'rag_collection': self.rag.config.collection_name,
            'embedding_model': self.rag.config.embedding_model,
            'max_context_docs': self.config.max_context_docs
        }


# Global RAG-LLM service instance
_rag_llm_service: Optional[RAGLLMService] = None


def get_rag_llm_service(
    config: Optional[RAGLLMConfig] = None,
    force_new: bool = False
) -> RAGLLMService:
    """
    Get global RAG-LLM service instance (singleton)
    
    Args:
        config: RAG-LLM configuration
        force_new: Force creation of new instance
        
    Returns:
        Global RAG-LLM service instance
    """
    global _rag_llm_service
    
    if _rag_llm_service is None or force_new:
        _rag_llm_service = RAGLLMService(config)
    
    return _rag_llm_service
