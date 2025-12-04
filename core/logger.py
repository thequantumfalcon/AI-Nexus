"""
Logging utilities for AI-Nexus
===============================

Provides structured logging with multiple outputs and formats.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from logging.handlers import RotatingFileHandler
import json
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Format logs as JSON"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        if hasattr(record, 'node_id'):
            log_data['node_id'] = record.node_id
        
        if hasattr(record, 'task_id'):
            log_data['task_id'] = record.task_id
            
        return json.dumps(log_data)


class ColoredFormatter(logging.Formatter):
    """Add colors to console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    json_format: bool = False,
    console: bool = True
) -> logging.Logger:
    """
    Set up a logger with specified configuration
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        json_format: Use JSON formatting
        console: Enable console output
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        if json_format:
            console_handler.setFormatter(JSONFormatter())
        else:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            console_handler.setFormatter(ColoredFormatter(fmt))
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            file_handler.setFormatter(logging.Formatter(fmt))
        
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger"""
    return logging.getLogger(name)


# Create default logger
default_logger = setup_logger('ainexus', level='INFO')
