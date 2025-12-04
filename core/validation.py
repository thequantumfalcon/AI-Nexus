"""
Input validation and sanitization for API endpoints.

Features:
- SQL injection prevention
- XSS prevention
- Path traversal prevention
- Command injection prevention
- JSON schema validation
- File upload validation
"""

import re
import html
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from pydantic import BaseModel, Field, validator, ValidationError


# ============================================================================
# Validation Rules
# ============================================================================

class ValidationRules:
    """Common validation patterns."""
    
    # Regex patterns
    EMAIL_PATTERN = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    USERNAME_PATTERN = r'^[a-zA-Z0-9_-]{3,32}$'
    API_KEY_PATTERN = r'^nexus_[a-zA-Z0-9_-]{8,}_[a-zA-Z0-9_-]{32,}$'
    
    # Dangerous patterns (for sanitization)
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|EXECUTE)\b)",
        r"(--|\#|\/\*|\*\/)",
        r"(\bOR\b.*=.*)",
        r"(\bAND\b.*=.*)",
        r"('|\"|\`)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe",
        r"<object",
        r"<embed",
    ]
    
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e/",
        r"%2e%2e\\",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$()]",
        r"\$\{.*\}",
        r"\$\(.*\)",
    ]
    
    # Allowed file extensions
    ALLOWED_FILE_EXTENSIONS = {
        "image": [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"],
        "document": [".pdf", ".txt", ".md", ".csv", ".json"],
        "code": [".py", ".js", ".ts", ".java", ".cpp", ".rs"],
    }
    
    # Max sizes (bytes)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
    MAX_JSON_SIZE = 1 * 1024 * 1024   # 1 MB
    MAX_STRING_LENGTH = 10000


# ============================================================================
# Sanitizers
# ============================================================================

class Sanitizer:
    """Input sanitization utilities."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = None) -> str:
        """
        Sanitize string input.
        
        - Removes null bytes
        - Trims whitespace
        - Limits length
        """
        if not isinstance(value, str):
            raise ValueError("Input must be a string")
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Trim whitespace
        value = value.strip()
        
        # Limit length
        if max_length:
            value = value[:max_length]
        elif len(value) > ValidationRules.MAX_STRING_LENGTH:
            value = value[:ValidationRules.MAX_STRING_LENGTH]
        
        return value
    
    @staticmethod
    def sanitize_html(value: str) -> str:
        """Escape HTML to prevent XSS."""
        return html.escape(value)
    
    @staticmethod
    def sanitize_sql(value: str) -> str:
        """
        Sanitize SQL input (basic).
        Note: Use parameterized queries instead when possible.
        """
        # Escape single quotes
        value = value.replace("'", "''")
        
        # Remove dangerous SQL keywords
        for pattern in ValidationRules.SQL_INJECTION_PATTERNS:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        return value
    
    @staticmethod
    def sanitize_path(value: str) -> str:
        """Sanitize file path to prevent traversal attacks."""
        # Remove path traversal patterns
        for pattern in ValidationRules.PATH_TRAVERSAL_PATTERNS:
            value = re.sub(pattern, "", value, flags=re.IGNORECASE)
        
        # Resolve to absolute path
        path = Path(value).resolve()
        
        return str(path)
    
    @staticmethod
    def sanitize_filename(value: str) -> str:
        """Sanitize filename."""
        # Remove path separators
        value = value.replace('/', '_').replace('\\', '_')
        
        # Remove dangerous characters
        value = re.sub(r'[^\w\s.-]', '', value)
        
        # Limit length
        return value[:255]
    
    @staticmethod
    def sanitize_json(value: Union[str, Dict]) -> Dict:
        """Validate and sanitize JSON input."""
        if isinstance(value, str):
            # Check size
            if len(value) > ValidationRules.MAX_JSON_SIZE:
                raise ValueError("JSON payload too large")
            
            try:
                data = json.loads(value)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {str(e)}")
        else:
            data = value
        
        if not isinstance(data, dict):
            raise ValueError("JSON must be an object")
        
        return data


# ============================================================================
# Validators
# ============================================================================

class Validator:
    """Input validation utilities."""
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format."""
        return bool(re.match(ValidationRules.EMAIL_PATTERN, email))
    
    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username format."""
        return bool(re.match(ValidationRules.USERNAME_PATTERN, username))
    
    @staticmethod
    def validate_api_key(api_key: str) -> bool:
        """Validate API key format."""
        return bool(re.match(ValidationRules.API_KEY_PATTERN, api_key))
    
    @staticmethod
    def validate_file_extension(
        filename: str,
        allowed_types: List[str] = None
    ) -> bool:
        """
        Validate file extension.
        
        Args:
            filename: File name
            allowed_types: List of allowed types (e.g., ["image", "document"])
        """
        ext = Path(filename).suffix.lower()
        
        if allowed_types:
            allowed_exts = []
            for file_type in allowed_types:
                allowed_exts.extend(
                    ValidationRules.ALLOWED_FILE_EXTENSIONS.get(file_type, [])
                )
        else:
            # Allow all defined extensions
            allowed_exts = [
                ext for exts in ValidationRules.ALLOWED_FILE_EXTENSIONS.values()
                for ext in exts
            ]
        
        return ext in allowed_exts
    
    @staticmethod
    def validate_file_size(size: int) -> bool:
        """Validate file size."""
        return size <= ValidationRules.MAX_FILE_SIZE
    
    @staticmethod
    def detect_sql_injection(value: str) -> bool:
        """Detect potential SQL injection."""
        for pattern in ValidationRules.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def detect_xss(value: str) -> bool:
        """Detect potential XSS."""
        for pattern in ValidationRules.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def detect_path_traversal(value: str) -> bool:
        """Detect path traversal attempt."""
        for pattern in ValidationRules.PATH_TRAVERSAL_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False
    
    @staticmethod
    def detect_command_injection(value: str) -> bool:
        """Detect command injection attempt."""
        for pattern in ValidationRules.COMMAND_INJECTION_PATTERNS:
            if re.search(pattern, value):
                return True
        return False


# ============================================================================
# Pydantic Models for API Validation
# ============================================================================

class SecureBaseModel(BaseModel):
    """Base model with automatic sanitization."""
    
    class Config:
        # Strict validation
        validate_assignment = True
        use_enum_values = True
        
        # Custom JSON encoders
        json_encoders = {
            Path: str,
        }
    
    def dict(self, *args, **kwargs):
        """Override dict to sanitize output."""
        data = super().dict(*args, **kwargs)
        return self._sanitize_dict(data)
    
    @classmethod
    def _sanitize_dict(cls, data: Dict) -> Dict:
        """Recursively sanitize dictionary."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = Sanitizer.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = cls._sanitize_dict(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    cls._sanitize_dict(item) if isinstance(item, dict)
                    else Sanitizer.sanitize_string(item) if isinstance(item, str)
                    else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized


class ValidatedTextInput(SecureBaseModel):
    """Validated text input."""
    text: str = Field(..., min_length=1, max_length=10000)
    
    @validator('text')
    def validate_text(cls, v):
        """Validate text input."""
        # Check for malicious patterns
        if Validator.detect_xss(v):
            raise ValueError("Potential XSS detected")
        if Validator.detect_sql_injection(v):
            raise ValueError("Potential SQL injection detected")
        if Validator.detect_command_injection(v):
            raise ValueError("Potential command injection detected")
        
        return Sanitizer.sanitize_string(v)


class ValidatedFileUpload(SecureBaseModel):
    """Validated file upload."""
    filename: str
    content_type: str
    size: int
    
    @validator('filename')
    def validate_filename(cls, v):
        """Validate filename."""
        if Validator.detect_path_traversal(v):
            raise ValueError("Invalid filename: path traversal detected")
        
        return Sanitizer.sanitize_filename(v)
    
    @validator('size')
    def validate_size(cls, v):
        """Validate file size."""
        if not Validator.validate_file_size(v):
            raise ValueError(f"File too large (max {ValidationRules.MAX_FILE_SIZE} bytes)")
        return v


class ValidatedUserInput(SecureBaseModel):
    """Validated user registration input."""
    username: str = Field(..., min_length=3, max_length=32)
    email: str
    password: str = Field(..., min_length=8)
    
    @validator('username')
    def validate_username(cls, v):
        """Validate username."""
        if not Validator.validate_username(v):
            raise ValueError("Invalid username format")
        return v
    
    @validator('email')
    def validate_email(cls, v):
        """Validate email."""
        if not Validator.validate_email(v):
            raise ValueError("Invalid email format")
        return v.lower()


class ValidatedModelInput(SecureBaseModel):
    """Validated LLM model input."""
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Validate prompt."""
        # Check for malicious patterns
        if Validator.detect_command_injection(v):
            raise ValueError("Invalid prompt: potential command injection")
        
        return Sanitizer.sanitize_string(v)


# ============================================================================
# Helper Functions
# ============================================================================

def validate_and_sanitize(
    data: Dict[str, Any],
    model_class: type[BaseModel]
) -> BaseModel:
    """
    Validate and sanitize input data.
    
    Args:
        data: Input data dictionary
        model_class: Pydantic model class to validate against
    
    Returns:
        Validated and sanitized model instance
    
    Raises:
        ValidationError: If validation fails
    """
    try:
        return model_class(**data)
    except ValidationError as e:
        raise ValueError(f"Validation failed: {e}")


def safe_json_loads(data: str) -> Dict:
    """Safely load JSON with size limits."""
    return Sanitizer.sanitize_json(data)


def safe_filename(filename: str) -> str:
    """Get safe filename."""
    return Sanitizer.sanitize_filename(filename)


def is_safe_input(value: str) -> bool:
    """Check if input is safe (no malicious patterns)."""
    return not any([
        Validator.detect_sql_injection(value),
        Validator.detect_xss(value),
        Validator.detect_path_traversal(value),
        Validator.detect_command_injection(value),
    ])
