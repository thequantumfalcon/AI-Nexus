"""
Token usage and cost tracking for LLM operations.

Features:
- Token counting for all LLM operations
- Cost calculation with configurable pricing
- SQLite-backed persistence
- Per-user and per-model tracking
- Daily/monthly/total statistics
- Budget alerts
"""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from functools import wraps


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PricingConfig:
    """LLM pricing configuration (per 1M tokens)."""
    # OpenAI GPT-4 pricing (as reference)
    gpt4_input: float = 30.0   # $30/1M tokens
    gpt4_output: float = 60.0  # $60/1M tokens
    
    # OpenAI GPT-3.5 pricing
    gpt35_input: float = 0.5   # $0.50/1M tokens
    gpt35_output: float = 1.5  # $1.50/1M tokens
    
    # LLaMA pricing (self-hosted, GPU costs)
    llama_input: float = 2.0   # $2/1M tokens (GPU amortized)
    llama_output: float = 4.0  # $4/1M tokens
    
    # Embedding models
    embedding_ada: float = 0.1   # $0.10/1M tokens
    embedding_custom: float = 0.05  # $0.05/1M tokens


@dataclass
class BudgetConfig:
    """Budget alert configuration."""
    daily_limit: Optional[float] = None  # Daily spend limit ($)
    monthly_limit: Optional[float] = None  # Monthly spend limit ($)
    total_limit: Optional[float] = None  # Total spend limit ($)
    
    alert_threshold: float = 0.8  # Alert at 80% of limit


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class TokenUsage:
    """Token usage record."""
    id: Optional[int] = None
    timestamp: datetime = None
    user_id: str = ""
    model: str = ""
    operation: str = ""  # generate, embed, rag_query, etc.
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class UsageStats:
    """Usage statistics."""
    total_requests: int
    total_tokens: int
    input_tokens: int
    output_tokens: int
    total_cost: float
    avg_tokens_per_request: float
    avg_cost_per_request: float


# ============================================================================
# Token Usage Tracker
# ============================================================================

class TokenUsageTracker:
    """
    Track token usage and costs.
    
    Features:
    - SQLite persistence
    - Per-user/model/operation tracking
    - Cost calculation
    - Budget alerts
    """
    
    def __init__(
        self,
        db_path: str = "data/token_usage.db",
        pricing: PricingConfig = None
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.pricing = pricing or PricingConfig()
        self.budget = BudgetConfig()
        
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                model TEXT NOT NULL,
                operation TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                total_tokens INTEGER NOT NULL,
                cost REAL NOT NULL,
                metadata TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_id
            ON token_usage(user_id)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp
            ON token_usage(timestamp)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_model
            ON token_usage(model)
        """)
        
        conn.commit()
        conn.close()
    
    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> float:
        """
        Calculate cost for token usage.
        
        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
        
        Returns:
            Cost in dollars
        """
        # Determine pricing based on model
        if "gpt-4" in model.lower():
            input_cost = (input_tokens / 1_000_000) * self.pricing.gpt4_input
            output_cost = (output_tokens / 1_000_000) * self.pricing.gpt4_output
        elif "gpt-3.5" in model.lower():
            input_cost = (input_tokens / 1_000_000) * self.pricing.gpt35_input
            output_cost = (output_tokens / 1_000_000) * self.pricing.gpt35_output
        elif "llama" in model.lower() or "nexus" in model.lower():
            input_cost = (input_tokens / 1_000_000) * self.pricing.llama_input
            output_cost = (output_tokens / 1_000_000) * self.pricing.llama_output
        elif "embed" in model.lower():
            input_cost = (input_tokens / 1_000_000) * self.pricing.embedding_custom
            output_cost = 0.0
        else:
            # Default to LLaMA pricing
            input_cost = (input_tokens / 1_000_000) * self.pricing.llama_input
            output_cost = (output_tokens / 1_000_000) * self.pricing.llama_output
        
        return input_cost + output_cost
    
    def record_usage(
        self,
        user_id: str,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int = 0,
        metadata: Optional[Dict] = None
    ) -> TokenUsage:
        """
        Record token usage.
        
        Args:
            user_id: User identifier
            model: Model name
            operation: Operation type
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            metadata: Additional metadata
        
        Returns:
            TokenUsage record
        """
        # Calculate cost
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Create usage record
        usage = TokenUsage(
            timestamp=datetime.utcnow(),
            user_id=user_id,
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=cost,
            metadata=metadata,
        )
        
        # Store in database
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO token_usage (
                timestamp, user_id, model, operation,
                input_tokens, output_tokens, total_tokens, cost, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            usage.timestamp.isoformat(),
            usage.user_id,
            usage.model,
            usage.operation,
            usage.input_tokens,
            usage.output_tokens,
            usage.total_tokens,
            usage.cost,
            json.dumps(usage.metadata) if usage.metadata else None,
        ))
        
        usage.id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Check budget alerts
        self._check_budget_alerts(user_id)
        
        return usage
    
    def get_usage_stats(
        self,
        user_id: Optional[str] = None,
        model: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> UsageStats:
        """
        Get usage statistics.
        
        Args:
            user_id: Filter by user ID
            model: Filter by model
            start_date: Filter by start date
            end_date: Filter by end date
        
        Returns:
            UsageStats object
        """
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Build query
        query = """
            SELECT
                COUNT(*) as total_requests,
                SUM(total_tokens) as total_tokens,
                SUM(input_tokens) as input_tokens,
                SUM(output_tokens) as output_tokens,
                SUM(cost) as total_cost
            FROM token_usage
            WHERE 1=1
        """
        params = []
        
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)
        
        if model:
            query += " AND model = ?"
            params.append(model)
        
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        
        cursor.execute(query, params)
        row = cursor.fetchone()
        conn.close()
        
        total_requests = row[0] or 0
        total_tokens = row[1] or 0
        input_tokens = row[2] or 0
        output_tokens = row[3] or 0
        total_cost = row[4] or 0.0
        
        avg_tokens = total_tokens / total_requests if total_requests > 0 else 0
        avg_cost = total_cost / total_requests if total_requests > 0 else 0
        
        return UsageStats(
            total_requests=total_requests,
            total_tokens=total_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_cost=total_cost,
            avg_tokens_per_request=avg_tokens,
            avg_cost_per_request=avg_cost,
        )
    
    def get_daily_cost(self, user_id: Optional[str] = None) -> float:
        """Get today's cost."""
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        stats = self.get_usage_stats(user_id=user_id, start_date=today)
        return stats.total_cost
    
    def get_monthly_cost(self, user_id: Optional[str] = None) -> float:
        """Get this month's cost."""
        month_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        stats = self.get_usage_stats(user_id=user_id, start_date=month_start)
        return stats.total_cost
    
    def get_total_cost(self, user_id: Optional[str] = None) -> float:
        """Get total cost."""
        stats = self.get_usage_stats(user_id=user_id)
        return stats.total_cost
    
    def _check_budget_alerts(self, user_id: str):
        """Check if budget limits are exceeded."""
        alerts = []
        
        if self.budget.daily_limit:
            daily_cost = self.get_daily_cost(user_id)
            threshold = self.budget.daily_limit * self.budget.alert_threshold
            if daily_cost >= threshold:
                alerts.append(f"Daily budget alert: ${daily_cost:.2f} / ${self.budget.daily_limit:.2f}")
        
        if self.budget.monthly_limit:
            monthly_cost = self.get_monthly_cost(user_id)
            threshold = self.budget.monthly_limit * self.budget.alert_threshold
            if monthly_cost >= threshold:
                alerts.append(f"Monthly budget alert: ${monthly_cost:.2f} / ${self.budget.monthly_limit:.2f}")
        
        if self.budget.total_limit:
            total_cost = self.get_total_cost(user_id)
            threshold = self.budget.total_limit * self.budget.alert_threshold
            if total_cost >= threshold:
                alerts.append(f"Total budget alert: ${total_cost:.2f} / ${self.budget.total_limit:.2f}")
        
        # Log alerts (could integrate with alerting system)
        for alert in alerts:
            print(f"⚠️  {alert}")
    
    def get_top_users(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top users by cost."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, SUM(cost) as total_cost
            FROM token_usage
            GROUP BY user_id
            ORDER BY total_cost DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results
    
    def get_top_models(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get top models by cost."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT model, SUM(cost) as total_cost
            FROM token_usage
            GROUP BY model
            ORDER BY total_cost DESC
            LIMIT ?
        """, (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        return results


# ============================================================================
# Global Instance
# ============================================================================

_token_tracker: Optional[TokenUsageTracker] = None


def get_token_tracker() -> TokenUsageTracker:
    """Get global token tracker instance."""
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenUsageTracker()
    return _token_tracker


# ============================================================================
# Decorator
# ============================================================================

def track_tokens(
    user_id_func=None,
    model_func=None,
    operation: str = "api_call"
):
    """
    Decorator to track token usage.
    
    Args:
        user_id_func: Function to extract user_id from args
        model_func: Function to extract model from args
        operation: Operation name
    
    Example:
        @track_tokens(
            user_id_func=lambda user, **kwargs: user.id,
            model_func=lambda **kwargs: kwargs.get('model', 'llama-3.3'),
            operation="generate"
        )
        async def generate_text(user, prompt, model='llama-3.3'):
            ...
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Get tracker
            tracker = get_token_tracker()
            
            # Extract user_id and model
            user_id = user_id_func(*args, **kwargs) if user_id_func else "unknown"
            model = model_func(*args, **kwargs) if model_func else "unknown"
            
            # Call function and track usage
            result = await func(*args, **kwargs)
            
            # Extract token counts from result (if available)
            if hasattr(result, 'usage'):
                usage = result.usage
                tracker.record_usage(
                    user_id=user_id,
                    model=model,
                    operation=operation,
                    input_tokens=getattr(usage, 'prompt_tokens', 0),
                    output_tokens=getattr(usage, 'completion_tokens', 0),
                )
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Get tracker
            tracker = get_token_tracker()
            
            # Extract user_id and model
            user_id = user_id_func(*args, **kwargs) if user_id_func else "unknown"
            model = model_func(*args, **kwargs) if model_func else "unknown"
            
            # Call function and track usage
            result = func(*args, **kwargs)
            
            # Extract token counts from result (if available)
            if hasattr(result, 'usage'):
                usage = result.usage
                tracker.record_usage(
                    user_id=user_id,
                    model=model,
                    operation=operation,
                    input_tokens=getattr(usage, 'prompt_tokens', 0),
                    output_tokens=getattr(usage, 'completion_tokens', 0),
                )
            
            return result
        
        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
