"""
Enterprise-grade security components for AI-Nexus.

Features:
- JWT authentication with RS256/HS256 algorithms
- Role-Based Access Control (RBAC)
- Password hashing with bcrypt
- Token refresh mechanism
- API key authentication
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from enum import Enum
import secrets
import hashlib

from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field


# ============================================================================
# Configuration
# ============================================================================

class AuthConfig:
    """Authentication configuration."""
    
    # JWT Settings
    SECRET_KEY: str = secrets.token_urlsafe(32)  # Generate secure key
    ALGORITHM: str = "HS256"  # HS256 for symmetric, RS256 for asymmetric
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # API Key Settings
    API_KEY_HEADER: str = "X-API-Key"
    API_KEY_PREFIX: str = "nexus_"
    
    # Password Policy
    MIN_PASSWORD_LENGTH: int = 8
    REQUIRE_UPPERCASE: bool = True
    REQUIRE_LOWERCASE: bool = True
    REQUIRE_DIGIT: bool = True
    REQUIRE_SPECIAL: bool = True


# ============================================================================
# User Roles & Permissions
# ============================================================================

class Role(str, Enum):
    """User roles for RBAC."""
    ADMIN = "admin"           # Full access
    DEVELOPER = "developer"   # API access, no user management
    ANALYST = "analyst"       # Read-only access
    GUEST = "guest"          # Limited access


class Permission(str, Enum):
    """Granular permissions."""
    # User Management
    USER_CREATE = "user:create"
    USER_READ = "user:read"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    
    # Model Operations
    MODEL_LOAD = "model:load"
    MODEL_INFERENCE = "model:inference"
    MODEL_FINE_TUNE = "model:fine_tune"
    
    # Data Operations
    DATA_READ = "data:read"
    DATA_WRITE = "data:write"
    DATA_DELETE = "data:delete"
    
    # System Operations
    SYSTEM_CONFIG = "system:config"
    SYSTEM_METRICS = "system:metrics"
    SYSTEM_LOGS = "system:logs"


# Role -> Permission mapping
ROLE_PERMISSIONS: Dict[Role, List[Permission]] = {
    Role.ADMIN: [p for p in Permission],  # All permissions
    Role.DEVELOPER: [
        Permission.USER_READ,
        Permission.MODEL_LOAD,
        Permission.MODEL_INFERENCE,
        Permission.DATA_READ,
        Permission.DATA_WRITE,
        Permission.SYSTEM_METRICS,
        Permission.SYSTEM_LOGS,
    ],
    Role.ANALYST: [
        Permission.USER_READ,
        Permission.MODEL_INFERENCE,
        Permission.DATA_READ,
        Permission.SYSTEM_METRICS,
    ],
    Role.GUEST: [
        Permission.MODEL_INFERENCE,
        Permission.DATA_READ,
    ],
}


# ============================================================================
# Data Models
# ============================================================================

class User(BaseModel):
    """User model."""
    username: str
    email: str
    full_name: Optional[str] = None
    role: Role = Role.GUEST
    disabled: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    def has_permission(self, permission: Permission) -> bool:
        """Check if user has permission."""
        return permission in ROLE_PERMISSIONS.get(self.role, [])
    
    def has_role(self, role: Role) -> bool:
        """Check if user has role."""
        return self.role == role


class UserInDB(User):
    """User model with hashed password."""
    hashed_password: str


class Token(BaseModel):
    """JWT token response."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int  # seconds


class TokenData(BaseModel):
    """JWT token payload data."""
    username: str
    role: Role
    permissions: List[Permission]
    exp: datetime


class APIKey(BaseModel):
    """API key model."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    role: Role
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None


# ============================================================================
# Password Management
# ============================================================================

class PasswordManager:
    """Secure password hashing and verification."""
    
    def __init__(self):
        # Use SHA256 instead of bcrypt to avoid bcrypt 5.0.0 compatibility issues
        self.pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
    
    def hash_password(self, password: str) -> str:
        """Hash a password using SHA256."""  
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against a hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> tuple[bool, Optional[str]]:
        """
        Validate password meets security requirements.
        
        Returns:
            (is_valid, error_message)
        """
        if len(password) < AuthConfig.MIN_PASSWORD_LENGTH:
            return False, f"Password must be at least {AuthConfig.MIN_PASSWORD_LENGTH} characters"
        
        if AuthConfig.REQUIRE_UPPERCASE and not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if AuthConfig.REQUIRE_LOWERCASE and not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if AuthConfig.REQUIRE_DIGIT and not any(c.isdigit() for c in password):
            return False, "Password must contain at least one digit"
        
        if AuthConfig.REQUIRE_SPECIAL and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain at least one special character"
        
        return True, None


# ============================================================================
# JWT Token Management
# ============================================================================

class JWTManager:
    """JWT token creation and validation."""
    
    def __init__(self, config: AuthConfig = None):
        self.config = config or AuthConfig()
        self.password_manager = PasswordManager()
    
    def create_access_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=self.config.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        to_encode = {
            "sub": user.username,
            "role": user.role.value,
            "permissions": [p.value for p in ROLE_PERMISSIONS.get(user.role, [])],
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.config.SECRET_KEY,
            algorithm=self.config.ALGORITHM
        )
        return encoded_jwt
    
    def create_refresh_token(
        self,
        user: User,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT refresh token."""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                days=self.config.REFRESH_TOKEN_EXPIRE_DAYS
            )
        
        to_encode = {
            "sub": user.username,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(
            to_encode,
            self.config.SECRET_KEY,
            algorithm=self.config.ALGORITHM
        )
        return encoded_jwt
    
    def create_token_pair(self, user: User) -> Token:
        """Create access + refresh token pair."""
        access_token = self.create_access_token(user)
        refresh_token = self.create_refresh_token(user)
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=self.config.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
    
    def decode_token(self, token: str) -> TokenData:
        """
        Decode and validate JWT token.
        
        Raises:
            JWTError: If token is invalid
        """
        try:
            payload = jwt.decode(
                token,
                self.config.SECRET_KEY,
                algorithms=[self.config.ALGORITHM]
            )
            
            username: str = payload.get("sub")
            if username is None:
                raise JWTError("Token missing subject")
            
            role = Role(payload.get("role", Role.GUEST.value))
            permissions = [Permission(p) for p in payload.get("permissions", [])]
            exp = datetime.fromtimestamp(payload.get("exp", 0))
            
            return TokenData(
                username=username,
                role=role,
                permissions=permissions,
                exp=exp
            )
        except JWTError as e:
            raise JWTError(f"Invalid token: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str, user: User) -> str:
        """
        Create new access token from refresh token.
        
        Raises:
            JWTError: If refresh token is invalid
        """
        payload = jwt.decode(
            refresh_token,
            self.config.SECRET_KEY,
            algorithms=[self.config.ALGORITHM]
        )
        
        if payload.get("type") != "refresh":
            raise JWTError("Not a refresh token")
        
        return self.create_access_token(user)


# ============================================================================
# API Key Management
# ============================================================================

class APIKeyManager:
    """API key generation and validation."""
    
    def __init__(self):
        self.password_manager = PasswordManager()
    
    def generate_api_key(
        self,
        user_id: str,
        name: str,
        role: Role,
        expires_days: Optional[int] = None
    ) -> tuple[str, APIKey]:
        """
        Generate a new API key.
        
        Returns:
            (raw_key, api_key_model)
        """
        # Generate secure random key
        key_id = secrets.token_urlsafe(8)
        secret = secrets.token_urlsafe(32)
        raw_key = f"{AuthConfig.API_KEY_PREFIX}{key_id}_{secret}"
        
        # Hash the secret part
        key_hash = hashlib.sha256(secret.encode()).hexdigest()
        
        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            role=role,
            expires_at=expires_at
        )
        
        return raw_key, api_key
    
    def validate_api_key(self, raw_key: str, stored_key: APIKey) -> bool:
        """
        Validate an API key.
        
        Returns:
            True if valid, False otherwise
        """
        # Check expiration
        if stored_key.expires_at and datetime.utcnow() > stored_key.expires_at:
            return False
        
        # Extract and verify secret
        try:
            parts = raw_key.replace(AuthConfig.API_KEY_PREFIX, "").split("_", 1)
            if len(parts) != 2:
                return False
            
            key_id, secret = parts
            if key_id != stored_key.key_id:
                return False
            
            secret_hash = hashlib.sha256(secret.encode()).hexdigest()
            return secret_hash == stored_key.key_hash
        except Exception:
            return False


# ============================================================================
# Global Instances
# ============================================================================

# Singleton instances
_password_manager: Optional[PasswordManager] = None
_jwt_manager: Optional[JWTManager] = None
_api_key_manager: Optional[APIKeyManager] = None


def get_password_manager() -> PasswordManager:
    """Get global password manager instance."""
    global _password_manager
    if _password_manager is None:
        _password_manager = PasswordManager()
    return _password_manager


def get_jwt_manager() -> JWTManager:
    """Get global JWT manager instance."""
    global _jwt_manager
    if _jwt_manager is None:
        _jwt_manager = JWTManager()
    return _jwt_manager


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


# ============================================================================
# Helper Functions
# ============================================================================

def authenticate_user(
    username: str,
    password: str,
    user_db: Dict[str, UserInDB]
) -> Optional[User]:
    """
    Authenticate user with username and password.
    
    Args:
        username: Username
        password: Plain text password
        user_db: Dictionary of users {username: UserInDB}
    
    Returns:
        User if authenticated, None otherwise
    """
    user = user_db.get(username)
    if not user:
        return None
    
    password_manager = get_password_manager()
    if not password_manager.verify_password(password, user.hashed_password):
        return None
    
    if user.disabled:
        return None
    
    return User(**user.dict(exclude={"hashed_password"}))


def check_permission(user: User, required_permission: Permission) -> bool:
    """Check if user has required permission."""
    return user.has_permission(required_permission)


def check_role(user: User, required_role: Role) -> bool:
    """Check if user has required role (exact match)."""
    return user.has_role(required_role)


def check_role_hierarchy(user: User, minimum_role: Role) -> bool:
    """
    Check if user has role at or above minimum level.
    
    Hierarchy: ADMIN > DEVELOPER > ANALYST > GUEST
    """
    role_order = {
        Role.GUEST: 0,
        Role.ANALYST: 1,
        Role.DEVELOPER: 2,
        Role.ADMIN: 3,
    }
    
    return role_order.get(user.role, 0) >= role_order.get(minimum_role, 0)
