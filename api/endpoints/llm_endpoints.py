"""
FastAPI endpoints for LLM operations
GPU-accelerated text generation with RAG integration
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import logging
import json

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


# API Models
class GenerateRequest(BaseModel):
    """Request for text generation"""
    prompt: str = Field(..., description="Input prompt for generation")
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0, le=200)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    do_sample: bool = Field(default=True)
    stream: bool = Field(default=False, description="Stream response")


class GenerateResponse(BaseModel):
    """Response from text generation"""
    prompt: str
    generated_text: str
    model: str
    generation_time_ms: Optional[float] = None


class ChatMessage(BaseModel):
    """Chat message"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat completion"""
    messages: List[ChatMessage] = Field(..., description="Chat history")
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    stream: bool = Field(default=False)


class ChatResponse(BaseModel):
    """Response from chat completion"""
    message: ChatMessage
    model: str


class RAGQueryRequest(BaseModel):
    """Request for RAG-enhanced query"""
    query: str = Field(..., description="User question")
    collection_name: Optional[str] = Field(default=None)
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    use_rag: Optional[bool] = Field(default=True, description="Use RAG retrieval")
    include_sources: Optional[bool] = Field(default=True)
    stream: bool = Field(default=False)


class RAGQueryResponse(BaseModel):
    """Response from RAG query"""
    query: str
    answer: str
    used_rag: bool
    sources: Optional[List[Dict[str, Any]]] = None
    num_sources: Optional[int] = None


class BatchGenerateRequest(BaseModel):
    """Request for batch generation"""
    prompts: List[str] = Field(..., description="List of prompts")
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class BatchGenerateResponse(BaseModel):
    """Response from batch generation"""
    results: List[Dict[str, str]]
    count: int


class ModelInfoResponse(BaseModel):
    """Model information"""
    model_name: str
    loaded: bool
    device: Optional[str] = None
    quantization: Optional[str] = None
    gpu_count: Optional[int] = None
    gpu_memory: Optional[Dict] = None


# Create router
router = APIRouter(prefix="/llm", tags=["LLM"])
logger = logging.getLogger(__name__)


@router.post("/generate", response_model=GenerateResponse)
async def generate_text(request: GenerateRequest):
    """
    Generate text from prompt
    
    - **prompt**: Input text prompt
    - **max_new_tokens**: Maximum tokens to generate (1-4096)
    - **temperature**: Sampling temperature (0.0-2.0)
    - **top_p**: Nucleus sampling parameter
    - **top_k**: Top-k sampling parameter
    - **stream**: Enable streaming response (returns text/event-stream)
    
    Returns generated text with metadata
    """
    try:
        # Handle streaming
        if request.stream:
            return await generate_stream(request)
        
        llm = get_llm_service()
        
        # Create generation config
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample
        )
        
        # Generate
        import time
        start = time.time()
        
        generated = llm.generate(
            prompt=request.prompt,
            generation_config=gen_config,
            return_full_text=False
        )
        
        elapsed_ms = (time.time() - start) * 1000
        
        logger.info(f"Generated {len(generated)} chars in {elapsed_ms:.1f}ms")
        
        return GenerateResponse(
            prompt=request.prompt,
            generated_text=generated,
            model=llm.config.model_name,
            generation_time_ms=elapsed_ms
        )
    
    except Exception as e:
        logger.error(f"Generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


async def generate_stream(request: GenerateRequest):
    """Stream generation response"""
    try:
        llm = get_llm_service()
        
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample
        )
        
        async def event_generator():
            for token in llm.generate_stream(request.prompt, gen_config):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Streaming error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Streaming failed: {str(e)}"
        )


@router.post("/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """
    Chat completion with conversation history
    
    - **messages**: List of chat messages with role and content
    - **max_new_tokens**: Maximum tokens to generate
    - **temperature**: Sampling temperature
    - **stream**: Enable streaming (returns text/event-stream)
    
    Supports system, user, and assistant roles
    """
    try:
        if request.stream:
            return await chat_stream(request)
        
        llm = get_llm_service()
        
        # Convert Pydantic models to dicts
        messages = [msg.dict() for msg in request.messages]
        
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        # Generate response
        response = llm.chat(
            messages=messages,
            generation_config=gen_config
        )
        
        logger.info(f"Chat completion: {len(response)} chars")
        
        return ChatResponse(
            message=ChatMessage(role="assistant", content=response),
            model=llm.config.model_name
        )
    
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


async def chat_stream(request: ChatRequest):
    """Stream chat response"""
    try:
        llm = get_llm_service()
        
        messages = [msg.dict() for msg in request.messages]
        
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            do_sample=True
        )
        
        # Format messages into prompt
        if hasattr(llm.tokenizer, 'apply_chat_template'):
            prompt = llm.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = llm._format_chat_messages(messages)
        
        async def event_generator():
            for token in llm.generate_stream(prompt, gen_config):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"Chat streaming error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat streaming failed: {str(e)}"
        )


@router.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    RAG-enhanced query (Retrieval-Augmented Generation)
    
    - **query**: User question
    - **collection_name**: Vector database collection to search
    - **use_rag**: Enable/disable RAG retrieval
    - **include_sources**: Include source documents in response
    - **stream**: Enable streaming (returns text/event-stream)
    
    Retrieves relevant context from vector database and generates
    answer with LLM using retrieved context
    """
    try:
        if request.stream:
            return await rag_query_stream(request)
        
        rag_llm = get_rag_llm_service()
        
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True
        )
        
        # Query with RAG
        result = rag_llm.query(
            query=request.query,
            collection_name=request.collection_name,
            generation_config=gen_config,
            use_rag=request.use_rag
        )
        
        logger.info(
            f"RAG query: {request.query[:50]}... - "
            f"RAG used: {result['used_rag']}"
        )
        
        return RAGQueryResponse(**result)
    
    except Exception as e:
        logger.error(f"RAG query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


async def rag_query_stream(request: RAGQueryRequest):
    """Stream RAG query response"""
    try:
        rag_llm = get_rag_llm_service()
        
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True
        )
        
        async def event_generator():
            for chunk in rag_llm.query_stream(
                query=request.query,
                collection_name=request.collection_name,
                generation_config=gen_config,
                use_rag=request.use_rag
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
    
    except Exception as e:
        logger.error(f"RAG streaming error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG streaming failed: {str(e)}"
        )


@router.post("/batch/generate", response_model=BatchGenerateResponse)
async def batch_generate(request: BatchGenerateRequest):
    """
    Batch text generation
    
    - **prompts**: List of input prompts
    - **max_new_tokens**: Maximum tokens per generation
    - **temperature**: Sampling temperature
    
    Efficiently generates text for multiple prompts in parallel
    """
    try:
        llm = get_llm_service()
        
        gen_config = GenerationConfig(
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            do_sample=True
        )
        
        # Batch generate
        results = llm.batch_generate(
            prompts=request.prompts,
            generation_config=gen_config
        )
        
        logger.info(f"Batch generated {len(results)} responses")
        
        formatted_results = [
            {"prompt": prompt, "generated_text": result}
            for prompt, result in zip(request.prompts, results)
        ]
        
        return BatchGenerateResponse(
            results=formatted_results,
            count=len(results)
        )
    
    except Exception as e:
        logger.error(f"Batch generation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch generation failed: {str(e)}"
        )


@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get LLM model information
    
    Returns model name, load status, device, quantization,
    and GPU memory usage
    """
    try:
        llm = get_llm_service()
        info = llm.get_model_info()
        
        return ModelInfoResponse(**info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check for LLM service
    
    Returns service status and configuration
    """
    try:
        llm = get_llm_service()
        rag_llm = get_rag_llm_service()
        
        return {
            "status": "healthy",
            "llm_loaded": llm.model is not None,
            "llm_model": llm.config.model_name,
            "rag_enabled": rag_llm.config.use_rag,
            "gpu_available": llm.device.type == "cuda" if llm.device else False
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.post("/rag/enable", status_code=status.HTTP_200_OK)
async def enable_rag():
    """
    Enable RAG (Retrieval-Augmented Generation)
    
    Enables automatic context retrieval from vector database
    """
    try:
        rag_llm = get_rag_llm_service()
        rag_llm.enable_rag()
        
        return {"status": "success", "rag_enabled": True}
    
    except Exception as e:
        logger.error(f"Error enabling RAG: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enable RAG: {str(e)}"
        )


@router.post("/rag/disable", status_code=status.HTTP_200_OK)
async def disable_rag():
    """
    Disable RAG (use pure LLM mode)
    
    Disables automatic context retrieval, uses LLM directly
    """
    try:
        rag_llm = get_rag_llm_service()
        rag_llm.disable_rag()
        
        return {"status": "success", "rag_enabled": False}
    
    except Exception as e:
        logger.error(f"Error disabling RAG: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disable RAG: {str(e)}"
        )


@router.get("/rag/status")
async def rag_status():
    """
    Get RAG-LLM service status
    
    Returns RAG configuration and service status
    """
    try:
        rag_llm = get_rag_llm_service()
        status_info = rag_llm.get_status()
        
        return status_info
    
    except Exception as e:
        logger.error(f"Error getting RAG status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get RAG status: {str(e)}"
        )
