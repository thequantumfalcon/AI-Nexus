"""
FastAPI Production Server
==========================

High-performance async REST API with authentication, rate limiting,
and comprehensive endpoint coverage.
"""

from fastapi import FastAPI, HTTPException, Depends, Security, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any
import asyncio
import time
import numpy as np
from datetime import datetime, timedelta, timezone
import jwt
from collections import defaultdict
import threading

from services.ml.ml_engine import PrivacyPreservingMLEngine, TrainResult, PredictResult
from services.nlp.nlp_engine import SecureNLPEngine, NLPResult
from services.blockchain.blockchain import AIBlockchain, Block
from core.logger import get_logger
from core.metrics import get_metrics_collector
from core.config import get_config

logger = get_logger(__name__)
config = get_config()
metrics = get_metrics_collector()

# Initialize FastAPI app
app = FastAPI(
    title="AI-Nexus API",
    description="Privacy-preserving AI platform with blockchain governance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Security
security = HTTPBearer()
SECRET_KEY = config.get("api.secret_key", "your-secret-key-change-in-production")
ALGORITHM = "HS256"

# Rate limiting
rate_limit_store = defaultdict(list)
rate_limit_lock = threading.Lock()

# Initialize services (lazy loading)
_ml_engine = None
_nlp_engine = None
_blockchain = None


def get_ml_engine():
    global _ml_engine
    if _ml_engine is None:
        _ml_engine = PrivacyPreservingMLEngine()
    return _ml_engine


def get_nlp_engine():
    global _nlp_engine
    if _nlp_engine is None:
        _nlp_engine = SecureNLPEngine()
    return _nlp_engine


def get_blockchain():
    global _blockchain
    if _blockchain is None:
        _blockchain = AIBlockchain(difficulty=4)
    return _blockchain


# Pydantic Models
class TokenRequest(BaseModel):
    api_key: str = Field(..., description="API key for authentication")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = 3600


class TrainRequest(BaseModel):
    model_type: str = Field(..., description="Model type: 'linear' or 'neural_net'")
    X: List[List[float]] = Field(..., description="Training features")
    y: List[float] = Field(..., description="Training labels")
    hyperparameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    enable_dp: bool = Field(default=True, description="Enable differential privacy")


class PredictRequest(BaseModel):
    model_id: str = Field(..., description="Model ID from training")
    X: List[List[float]] = Field(..., description="Input features")
    return_confidence: bool = True
    return_explanation: bool = False


class NLPRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    task_type: str = Field(default="classification")
    compliance_mode: str = Field(default="HIPAA")
    enable_explainability: bool = True


class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    language: str = "en"


class EntityRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000)
    entity_types: Optional[List[str]] = None


class TextGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000)
    max_length: int = Field(default=100, ge=10, le=500)
    temperature: float = Field(default=0.7, ge=0.1, le=2.0)


class BlockchainTransactionRequest(BaseModel):
    data: Dict[str, Any]
    transaction_type: str = "data"


class GovernanceProposalRequest(BaseModel):
    proposal_type: str
    description: str
    parameters: Dict[str, Any]


class VoteRequest(BaseModel):
    proposal_id: str
    vote: bool
    voter_address: str


# Authentication & Rate Limiting
def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired"
        )
    except jwt.JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def rate_limit(request: Request, max_requests: int = 100, window_seconds: int = 60):
    """Simple rate limiting"""
    client_ip = request.client.host
    current_time = time.time()
    
    with rate_limit_lock:
        # Clean old entries
        rate_limit_store[client_ip] = [
            t for t in rate_limit_store[client_ip]
            if current_time - t < window_seconds
        ]
        
        # Check limit
        if len(rate_limit_store[client_ip]) >= max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {max_requests} requests per {window_seconds}s"
            )
        
        rate_limit_store[client_ip].append(current_time)


# Health & Status Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "AI-Nexus API",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "services": {
            "ml": "ready",
            "nlp": "ready",
            "blockchain": "ready"
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "metrics": metrics.get_summary(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


# Authentication Endpoints
@app.post("/auth/token", response_model=TokenResponse)
async def login(token_request: TokenRequest):
    """Get JWT token"""
    # In production, validate against database
    if token_request.api_key == "demo-api-key":
        access_token = create_access_token(data={"sub": "user_id"})
        return TokenResponse(access_token=access_token)
    raise HTTPException(status_code=401, detail="Invalid API key")


# ML Endpoints
@app.post("/ml/train")
async def train_model(
    request: TrainRequest,
    user: dict = Depends(verify_token)
):
    """Train a machine learning model"""
    try:
        ml_engine = get_ml_engine()
        
        X = np.array(request.X)
        y = np.array(request.y)
        
        result = ml_engine.train_model(
            request.model_type,
            X,
            y,
            request.hyperparameters,
            request.enable_dp
        )
        
        return {
            "model_id": result.model_id,
            "final_loss": float(result.final_loss),
            "final_accuracy": float(result.final_accuracy),
            "training_time_ms": result.training_time_ms,
            "epochs_completed": result.epochs_completed
        }
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/predict")
async def predict(
    request: PredictRequest,
    user: dict = Depends(verify_token)
):
    """Make predictions with trained model"""
    try:
        ml_engine = get_ml_engine()
        
        X = np.array(request.X)
        
        result = ml_engine.predict(
            model_id=request.model_id,
            input_data=X,
            return_confidence=request.return_confidence,
            return_explanation=request.return_explanation
        )
        
        return {
            "prediction": result.prediction.tolist() if hasattr(result.prediction, 'tolist') else result.prediction,
            "confidence": float(result.confidence),
            "explanation": result.explanation,
            "inference_time_ms": result.inference_time_ms
        }
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ml/evaluate")
async def evaluate_model(
    model_id: str,
    X: List[List[float]],
    y: List[float],
    user: dict = Depends(verify_token)
):
    """Evaluate model performance"""
    try:
        ml_engine = get_ml_engine()
        
        X_arr = np.array(X)
        y_arr = np.array(y)
        
        metrics = ml_engine.evaluate_model(model_id, X_arr, y_arr)
        
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# NLP Endpoints
@app.post("/nlp/process")
async def process_text(
    request: NLPRequest,
    user: dict = Depends(verify_token)
):
    """Process text with NLP"""
    try:
        nlp_engine = get_nlp_engine()
        
        result = nlp_engine.process_text(
            text=request.text,
            task_type=request.task_type,
            compliance_mode=request.compliance_mode,
            enable_explainability=request.enable_explainability
        )
        
        return {
            "result": result.result,
            "confidence": float(result.confidence),
            "explanation": result.explanation,
            "processing_time_ms": result.processing_time_ms,
            "model_version": result.model_version,
            "metadata": result.metadata
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/sentiment")
async def analyze_sentiment(
    request: SentimentRequest,
    user: dict = Depends(verify_token)
):
    """Analyze sentiment"""
    try:
        nlp_engine = get_nlp_engine()
        result = nlp_engine.analyze_sentiment(request.text, request.language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/entities")
async def extract_entities(
    request: EntityRequest,
    user: dict = Depends(verify_token)
):
    """Extract named entities"""
    try:
        nlp_engine = get_nlp_engine()
        entities = nlp_engine.extract_entities(request.text, request.entity_types)
        return {"entities": entities}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/nlp/generate")
async def generate_text(
    request: TextGenRequest,
    user: dict = Depends(verify_token)
):
    """Generate text"""
    try:
        nlp_engine = get_nlp_engine()
        result = nlp_engine.generate_text(
            request.prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/nlp/capabilities")
async def get_nlp_capabilities(user: dict = Depends(verify_token)):
    """Get NLP capabilities"""
    nlp_engine = get_nlp_engine()
    return nlp_engine.get_capabilities()


# Blockchain Endpoints
@app.post("/blockchain/transaction")
async def add_transaction(
    request: BlockchainTransactionRequest,
    user: dict = Depends(verify_token)
):
    """Add transaction to blockchain"""
    try:
        blockchain = get_blockchain()
        block = blockchain.add_block(request.data, request.transaction_type)
        
        return {
            "block_index": block.index,
            "block_hash": block.hash,
            "timestamp": block.timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/blockchain/chain")
async def get_chain(user: dict = Depends(verify_token)):
    """Get blockchain"""
    blockchain = get_blockchain()
    return {
        "chain": [
            {
                "index": b.index,
                "timestamp": b.timestamp,
                "data": b.data,
                "hash": b.hash,
                "previous_hash": b.previous_hash,
                "nonce": b.nonce
            }
            for b in blockchain.chain
        ],
        "length": len(blockchain.chain)
    }


@app.get("/blockchain/validate")
async def validate_chain(user: dict = Depends(verify_token)):
    """Validate blockchain"""
    blockchain = get_blockchain()
    is_valid = blockchain.is_chain_valid()
    return {"valid": is_valid}


@app.post("/governance/proposal")
async def create_proposal(
    request: GovernanceProposalRequest,
    user: dict = Depends(verify_token)
):
    """Create governance proposal"""
    try:
        blockchain = get_blockchain()
        proposal_id = blockchain.governance.create_proposal(
            request.proposal_type,
            request.description,
            request.parameters
        )
        return {"proposal_id": proposal_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/governance/vote")
async def vote_on_proposal(
    request: VoteRequest,
    user: dict = Depends(verify_token)
):
    """Vote on proposal"""
    try:
        blockchain = get_blockchain()
        blockchain.governance.vote(
            request.proposal_id,
            request.voter_address,
            request.vote
        )
        return {"status": "vote_recorded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/governance/proposals")
async def get_proposals(user: dict = Depends(verify_token)):
    """Get all proposals"""
    blockchain = get_blockchain()
    return {"proposals": blockchain.governance.proposals}


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now(timezone.utc).isoformat()}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now(timezone.utc).isoformat()}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
