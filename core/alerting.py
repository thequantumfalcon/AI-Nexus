"""
Error alerting and logging framework.

Features:
- Structured logging with severity levels
- Error aggregation and deduplication
- Alert thresholds and rate limiting
- Multiple notification channels (email, Slack, PagerDuty)
- Contextual error tracking
"""

import logging
import json
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict
import hashlib


# ============================================================================
# Configuration
# ============================================================================

class AlertSeverity(str, Enum):
    """Alert severity levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AlertConfig:
    """Alert configuration."""
    # Thresholds
    error_threshold: int = 10  # Alert after N errors
    critical_threshold: int = 5  # Alert after N critical errors
    time_window_minutes: int = 5  # Within time window
    
    # Deduplication
    deduplicate: bool = True
    dedupe_window_minutes: int = 60
    
    # Rate limiting
    max_alerts_per_hour: int = 10
    
    # Channels (implement as needed)
    enable_email: bool = False
    enable_slack: bool = False
    enable_pagerduty: bool = False
    enable_console: bool = True


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class ErrorEvent:
    """Error event record."""
    timestamp: datetime
    severity: AlertSeverity
    message: str
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    stack_trace: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    user_id: Optional[str] = None
    endpoint: Optional[str] = None
    request_id: Optional[str] = None
    
    def get_fingerprint(self) -> str:
        """Get unique fingerprint for deduplication."""
        key = f"{self.exception_type}:{self.message}:{self.endpoint}"
        return hashlib.md5(key.encode()).hexdigest()


@dataclass
class Alert:
    """Alert notification."""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    count: int  # Number of occurrences
    events: List[ErrorEvent] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "count": self.count,
            "events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "message": e.message,
                    "exception": e.exception_type,
                    "endpoint": e.endpoint,
                }
                for e in self.events[:5]  # Include first 5 events
            ]
        }


# ============================================================================
# Alert Manager
# ============================================================================

class AlertManager:
    """
    Manage error alerts and notifications.
    
    Features:
    - Error aggregation
    - Deduplication
    - Threshold-based alerting
    - Multiple notification channels
    """
    
    def __init__(self, config: AlertConfig = None):
        self.config = config or AlertConfig()
        
        # Error tracking
        self.error_buffer: List[ErrorEvent] = []
        self.alert_history: List[Alert] = []
        
        # Deduplication
        self.seen_fingerprints: Dict[str, datetime] = {}
        
        # Rate limiting
        self.alerts_sent_count: Dict[str, int] = defaultdict(int)  # hour -> count
        
        # Custom handlers
        self.handlers: List[Callable[[Alert], None]] = []
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup structured logging."""
        # Create custom logger
        self.logger = logging.getLogger("ai_nexus_alerts")
        self.logger.setLevel(logging.DEBUG)
        
        # Console handler
        if self.config.enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.WARNING)
            
            # Custom formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler (JSON structured logs)
        file_handler = logging.FileHandler('logs/alerts.log')
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
    
    def record_error(
        self,
        message: str,
        severity: AlertSeverity = AlertSeverity.ERROR,
        exception: Optional[Exception] = None,
        context: Optional[Dict] = None,
        **kwargs
    ):
        """
        Record an error event.
        
        Args:
            message: Error message
            severity: Alert severity
            exception: Exception object (if applicable)
            context: Additional context
            **kwargs: Additional metadata (user_id, endpoint, etc.)
        """
        # Create error event
        event = ErrorEvent(
            timestamp=datetime.utcnow(),
            severity=severity,
            message=message,
            context=context or {},
            **kwargs
        )
        
        # Add exception details
        if exception:
            event.exception_type = type(exception).__name__
            event.exception_message = str(exception)
            event.stack_trace = traceback.format_exc()
        
        # Add to buffer
        self.error_buffer.append(event)
        
        # Log to file
        self._log_event(event)
        
        # Check if we should trigger alert
        self._check_thresholds()
        
        # Cleanup old events
        self._cleanup_buffer()
    
    def _log_event(self, event: ErrorEvent):
        """Log event to file (structured JSON)."""
        log_data = {
            "timestamp": event.timestamp.isoformat(),
            "severity": event.severity.value,
            "message": event.message,
            "exception_type": event.exception_type,
            "exception_message": event.exception_message,
            "context": event.context,
            "user_id": event.user_id,
            "endpoint": event.endpoint,
        }
        
        # Log based on severity
        if event.severity == AlertSeverity.CRITICAL:
            self.logger.critical(json.dumps(log_data))
        elif event.severity == AlertSeverity.ERROR:
            self.logger.error(json.dumps(log_data))
        elif event.severity == AlertSeverity.WARNING:
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))
    
    def _check_thresholds(self):
        """Check if alert thresholds are met."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.config.time_window_minutes)
        
        # Get recent events
        recent_events = [
            e for e in self.error_buffer
            if e.timestamp >= window_start
        ]
        
        # Count by severity
        error_count = sum(
            1 for e in recent_events
            if e.severity == AlertSeverity.ERROR
        )
        critical_count = sum(
            1 for e in recent_events
            if e.severity == AlertSeverity.CRITICAL
        )
        
        # Check thresholds
        should_alert = False
        alert_title = ""
        alert_severity = AlertSeverity.ERROR
        
        if critical_count >= self.config.critical_threshold:
            should_alert = True
            alert_title = f"{critical_count} critical errors in {self.config.time_window_minutes} minutes"
            alert_severity = AlertSeverity.CRITICAL
        elif error_count >= self.config.error_threshold:
            should_alert = True
            alert_title = f"{error_count} errors in {self.config.time_window_minutes} minutes"
            alert_severity = AlertSeverity.ERROR
        
        if should_alert:
            self._trigger_alert(alert_title, alert_severity, recent_events)
    
    def _trigger_alert(
        self,
        title: str,
        severity: AlertSeverity,
        events: List[ErrorEvent]
    ):
        """Trigger an alert."""
        # Check rate limiting
        current_hour = datetime.utcnow().strftime("%Y-%m-%d-%H")
        if self.alerts_sent_count[current_hour] >= self.config.max_alerts_per_hour:
            self.logger.warning("Alert rate limit exceeded, suppressing alert")
            return
        
        # Check deduplication
        if self.config.deduplicate:
            # Create fingerprint from events
            fingerprint = events[0].get_fingerprint() if events else ""
            
            if fingerprint in self.seen_fingerprints:
                last_seen = self.seen_fingerprints[fingerprint]
                if datetime.utcnow() - last_seen < timedelta(minutes=self.config.dedupe_window_minutes):
                    self.logger.debug(f"Suppressing duplicate alert: {fingerprint}")
                    return
            
            self.seen_fingerprints[fingerprint] = datetime.utcnow()
        
        # Create alert
        alert = Alert(
            id=f"alert_{datetime.utcnow().timestamp()}",
            timestamp=datetime.utcnow(),
            severity=severity,
            title=title,
            message=self._format_alert_message(events),
            count=len(events),
            events=events[:10],  # Keep first 10 events
        )
        
        # Add to history
        self.alert_history.append(alert)
        
        # Send alert
        self._send_alert(alert)
        
        # Increment rate limit counter
        self.alerts_sent_count[current_hour] += 1
    
    def _format_alert_message(self, events: List[ErrorEvent]) -> str:
        """Format alert message from events."""
        if not events:
            return "No events"
        
        # Group by exception type
        by_type: Dict[str, int] = defaultdict(int)
        for event in events:
            exception_type = event.exception_type or "Unknown"
            by_type[exception_type] += 1
        
        # Format message
        lines = ["Error summary:"]
        for exc_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            lines.append(f"  - {exc_type}: {count} occurrences")
        
        # Add sample messages
        lines.append("\nSample errors:")
        for event in events[:3]:
            lines.append(f"  - {event.message}")
        
        return "\n".join(lines)
    
    def _send_alert(self, alert: Alert):
        """Send alert through configured channels."""
        # Console
        if self.config.enable_console:
            self._send_console_alert(alert)
        
        # Custom handlers
        for handler in self.handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}")
        
        # TODO: Implement additional channels
        # if self.config.enable_email:
        #     self._send_email_alert(alert)
        # if self.config.enable_slack:
        #     self._send_slack_alert(alert)
        # if self.config.enable_pagerduty:
        #     self._send_pagerduty_alert(alert)
    
    def _send_console_alert(self, alert: Alert):
        """Print alert to console."""
        severity_emoji = {
            AlertSeverity.CRITICAL: "🔴",
            AlertSeverity.ERROR: "🟠",
            AlertSeverity.WARNING: "🟡",
            AlertSeverity.INFO: "🔵",
        }
        
        emoji = severity_emoji.get(alert.severity, "⚪")
        
        print(f"\n{'='*80}")
        print(f"{emoji} ALERT: {alert.title}")
        print(f"Severity: {alert.severity.value.upper()}")
        print(f"Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Count: {alert.count}")
        print(f"\n{alert.message}")
        print(f"{'='*80}\n")
    
    def _cleanup_buffer(self):
        """Remove old events from buffer."""
        cutoff = datetime.utcnow() - timedelta(hours=1)
        self.error_buffer = [
            e for e in self.error_buffer
            if e.timestamp >= cutoff
        ]
    
    def add_handler(self, handler: Callable[[Alert], None]):
        """Add custom alert handler."""
        self.handlers.append(handler)
    
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]:
        """Get recent alerts."""
        return sorted(
            self.alert_history,
            key=lambda a: a.timestamp,
            reverse=True
        )[:limit]
    
    def get_error_stats(self) -> Dict:
        """Get error statistics."""
        now = datetime.utcnow()
        
        # Last hour
        hour_ago = now - timedelta(hours=1)
        errors_last_hour = [e for e in self.error_buffer if e.timestamp >= hour_ago]
        
        # By severity
        by_severity = {
            severity: sum(1 for e in errors_last_hour if e.severity == severity)
            for severity in AlertSeverity
        }
        
        # By endpoint
        by_endpoint: Dict[str, int] = defaultdict(int)
        for event in errors_last_hour:
            if event.endpoint:
                by_endpoint[event.endpoint] += 1
        
        return {
            "total_errors_last_hour": len(errors_last_hour),
            "by_severity": by_severity,
            "by_endpoint": dict(by_endpoint),
            "total_alerts_sent": sum(self.alerts_sent_count.values()),
        }


# ============================================================================
# Global Instance
# ============================================================================

_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager


# ============================================================================
# Helper Functions
# ============================================================================

def log_error(
    message: str,
    exception: Optional[Exception] = None,
    severity: AlertSeverity = AlertSeverity.ERROR,
    **context
):
    """
    Log an error.
    
    Args:
        message: Error message
        exception: Exception object
        severity: Alert severity
        **context: Additional context
    """
    manager = get_alert_manager()
    manager.record_error(
        message=message,
        severity=severity,
        exception=exception,
        context=context
    )


def log_critical(message: str, exception: Optional[Exception] = None, **context):
    """Log a critical error."""
    log_error(message, exception, AlertSeverity.CRITICAL, **context)


def log_warning(message: str, **context):
    """Log a warning."""
    log_error(message, None, AlertSeverity.WARNING, **context)


# ============================================================================
# Decorator
# ============================================================================

def catch_and_log(severity: AlertSeverity = AlertSeverity.ERROR):
    """
    Decorator to catch and log exceptions.
    
    Example:
        @catch_and_log(severity=AlertSeverity.CRITICAL)
        async def critical_operation():
            ...
    """
    def decorator(func):
        from functools import wraps
        import asyncio
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                log_error(
                    f"Exception in {func.__name__}: {str(e)}",
                    exception=e,
                    severity=severity,
                    function=func.__name__
                )
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_error(
                    f"Exception in {func.__name__}: {str(e)}",
                    exception=e,
                    severity=severity,
                    function=func.__name__
                )
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
