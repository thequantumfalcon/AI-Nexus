"""
FastAPI endpoints for RAG (Retrieval-Augmented Generation) operations
GPU-accelerated vector search and context generation
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, status
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging

from services.ml.rag_pipeline import (
    RAGPipeline,
    RAGConfig,
    HybridSearch,
    get_rag_pipeline
)


# API Models
class Document(BaseModel):
    """Document for indexing"""
    text: str = Field(..., description="Document text content")
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata"
    )


class IndexRequest(BaseModel):
    """Request to index documents"""
    documents: List[Document] = Field(..., description="Documents to index")
    collection_name: Optional[str] = Field(
        default=None,
        description="Collection name (default: knowledge_base)"
    )


class IndexResponse(BaseModel):
    """Response from document indexing"""
    status: str
    documents_indexed: int
    chunks_created: int
    collection: str
    vector_size: int
    privacy_enabled: bool


class SearchRequest(BaseModel):
    """Request to search documents"""
    query: str = Field(..., description="Search query")
    collection_name: Optional[str] = Field(
        default=None,
        description="Collection to search"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=100,
        description="Number of results"
    )
    min_score: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include document metadata"
    )


class RetrievedDocument(BaseModel):
    """Retrieved document from search"""
    id: int
    text: str
    score: float
    payload: Dict[str, Any]


class SearchResponse(BaseModel):
    """Response from document search"""
    query: str
    num_results: int
    documents: List[Dict[str, Any]]
    config: Optional[Dict[str, Any]] = None


class RAGQueryRequest(BaseModel):
    """Request for full RAG pipeline query"""
    query: str = Field(..., description="User question")
    collection_name: Optional[str] = Field(
        default=None,
        description="Collection to search"
    )
    include_metadata: bool = Field(
        default=True,
        description="Include retrieval metadata"
    )


class RAGQueryResponse(BaseModel):
    """Response from RAG query"""
    query: str
    context: str
    num_docs: int
    retrieved_documents: Optional[List[Dict[str, Any]]] = None
    config: Optional[Dict[str, Any]] = None


class ConfigUpdateRequest(BaseModel):
    """Request to update RAG configuration"""
    top_k: Optional[int] = Field(default=None, ge=1, le=100)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    chunk_size: Optional[int] = Field(default=None, ge=100, le=2048)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=500)
    enable_privacy: Optional[bool] = None
    privacy_epsilon: Optional[float] = Field(default=None, ge=0.1, le=10.0)


class CacheStatsResponse(BaseModel):
    """Embedding cache statistics"""
    cache_size: int
    max_size: int
    hit_rate: float
    total_requests: int


# Create router
router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])
logger = logging.getLogger(__name__)


@router.post("/index", response_model=IndexResponse, status_code=status.HTTP_201_CREATED)
async def index_documents(
    request: IndexRequest,
    background_tasks: BackgroundTasks
):
    """
    Index documents into vector database
    
    - **documents**: List of documents with text and optional metadata
    - **collection_name**: Optional collection name (default: knowledge_base)
    
    Returns indexing statistics including number of chunks created
    """
    try:
        rag = get_rag_pipeline()
        
        # Convert Pydantic models to dicts
        docs = [doc.dict() for doc in request.documents]
        
        # Index documents
        stats = rag.index_documents(docs, request.collection_name)
        
        logger.info(
            f"Indexed {stats['documents']} documents "
            f"into collection '{stats['collection']}'"
        )
        
        return IndexResponse(
            status="success",
            documents_indexed=stats['documents'],
            chunks_created=stats['chunks'],
            collection=stats['collection'],
            vector_size=stats['vector_size'],
            privacy_enabled=stats['privacy_enabled']
        )
    
    except Exception as e:
        logger.error(f"Error indexing documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index documents: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """
    Search for relevant documents
    
    - **query**: Search query text
    - **collection_name**: Collection to search (default: knowledge_base)
    - **top_k**: Number of results to return (1-100)
    - **min_score**: Minimum similarity score (0.0-1.0)
    - **include_metadata**: Include document metadata in response
    
    Returns ranked list of relevant documents with similarity scores
    """
    try:
        rag = get_rag_pipeline()
        
        # Search documents
        results = rag.retrieve(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k,
            min_score=request.min_score
        )
        
        logger.info(
            f"Search query: '{request.query}' - "
            f"Found {len(results)} results"
        )
        
        response = SearchResponse(
            query=request.query,
            num_results=len(results),
            documents=results
        )
        
        if request.include_metadata:
            response.config = {
                'top_k': request.top_k,
                'min_score': request.min_score,
                'collection': request.collection_name or rag.config.collection_name
            }
        
        return response
    
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Full RAG pipeline query - retrieve context for LLM generation
    
    - **query**: User question/query
    - **collection_name**: Collection to search (default: knowledge_base)
    - **include_metadata**: Include retrieval details
    
    Returns formatted context string ready for LLM input along with
    retrieved documents and metadata
    """
    try:
        rag = get_rag_pipeline()
        
        # Execute RAG query
        result = rag.query(
            user_query=request.query,
            collection_name=request.collection_name,
            include_metadata=request.include_metadata
        )
        
        logger.info(
            f"RAG query: '{request.query}' - "
            f"Retrieved {result['num_docs']} documents"
        )
        
        return RAGQueryResponse(**result)
    
    except Exception as e:
        logger.error(f"Error executing RAG query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG query failed: {str(e)}"
        )


@router.get("/collections", response_model=List[str])
async def list_collections():
    """
    List all available collections in vector database
    
    Returns list of collection names
    """
    try:
        rag = get_rag_pipeline()
        collections = rag.vector_db.list_collections()
        
        logger.info(f"Found {len(collections)} collections")
        return collections
    
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list collections: {str(e)}"
        )


@router.delete("/collections/{collection_name}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_collection(collection_name: str):
    """
    Delete a collection from vector database
    
    - **collection_name**: Name of collection to delete
    
    WARNING: This permanently deletes all vectors in the collection
    """
    try:
        rag = get_rag_pipeline()
        rag.vector_db.delete_collection(collection_name)
        
        logger.info(f"Deleted collection: {collection_name}")
        return None
    
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete collection: {str(e)}"
        )


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats():
    """
    Get embedding cache statistics
    
    Returns cache size, hit rate, and usage metrics
    """
    try:
        rag = get_rag_pipeline()
        stats = rag.get_cache_stats()
        
        return CacheStatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.post("/cache/clear", status_code=status.HTTP_204_NO_CONTENT)
async def clear_cache():
    """
    Clear embedding cache
    
    This will force re-computation of all embeddings on next use
    """
    try:
        rag = get_rag_pipeline()
        rag.clear_cache()
        
        logger.info("Embedding cache cleared")
        return None
    
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.get("/health")
async def health_check():
    """
    Health check for RAG service
    
    Returns service status and configuration
    """
    try:
        rag = get_rag_pipeline()
        
        return {
            "status": "healthy",
            "gpu_available": rag.config.use_gpu,
            "privacy_enabled": rag.config.enable_privacy,
            "embedding_model": rag.config.embedding_model,
            "collection_name": rag.config.collection_name
        }
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@router.patch("/config", status_code=status.HTTP_200_OK)
async def update_config(request: ConfigUpdateRequest):
    """
    Update RAG pipeline configuration
    
    Updates only provided fields, leaves others unchanged
    """
    try:
        rag = get_rag_pipeline()
        updates = {}
        
        # Update only provided fields
        if request.top_k is not None:
            rag.config.top_k = request.top_k
            updates['top_k'] = request.top_k
        
        if request.min_score is not None:
            rag.config.min_score = request.min_score
            updates['min_score'] = request.min_score
        
        if request.chunk_size is not None:
            rag.config.chunk_size = request.chunk_size
            rag.chunker.chunk_size = request.chunk_size
            updates['chunk_size'] = request.chunk_size
        
        if request.chunk_overlap is not None:
            rag.config.chunk_overlap = request.chunk_overlap
            rag.chunker.overlap = request.chunk_overlap
            updates['chunk_overlap'] = request.chunk_overlap
        
        if request.enable_privacy is not None:
            rag.config.enable_privacy = request.enable_privacy
            updates['enable_privacy'] = request.enable_privacy
        
        if request.privacy_epsilon is not None:
            rag.config.privacy_epsilon = request.privacy_epsilon
            updates['privacy_epsilon'] = request.privacy_epsilon
        
        logger.info(f"Updated RAG config: {updates}")
        
        return {
            "status": "success",
            "updated_fields": updates,
            "current_config": {
                "top_k": rag.config.top_k,
                "min_score": rag.config.min_score,
                "chunk_size": rag.config.chunk_size,
                "chunk_overlap": rag.config.chunk_overlap,
                "enable_privacy": rag.config.enable_privacy,
                "privacy_epsilon": rag.config.privacy_epsilon
            }
        }
    
    except Exception as e:
        logger.error(f"Error updating config: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update config: {str(e)}"
        )
