"""
Authentication API endpoints for Orchestral.

Provides JWT-based authentication with secure user management.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
import jwt

from orchestral.core.config import get_settings
from orchestral.billing.api_keys import get_api_key_manager, KeyTier

logger = structlog.get_logger()

# Security setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

router = APIRouter(prefix="/auth", tags=["authentication"])

# JWT configuration
JWT_SECRET = get_settings().server.admin_api_key.get_secret_value() if get_settings().server.admin_api_key else secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# In-memory user store (replace with database in production)
users_db: dict[str, dict[str, Any]] = {}
sessions_db: dict[str, dict[str, Any]] = {}


# ==================== REQUEST/RESPONSE MODELS ====================

class UserSignup(BaseModel):
    """User signup request."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: str = Field(..., min_length=2)
    company: str | None = None

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Ensure password meets security requirements."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserLogin(BaseModel):
    """User login request."""
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    """JWT token response."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int = JWT_EXPIRATION_HOURS * 3600
    user: dict[str, Any]


class UserResponse(BaseModel):
    """User response."""
    id: str
    email: str
    full_name: str
    company: str | None
    created_at: str
    api_key_id: str | None
    subscription_status: str | None
    tier: str


class PasswordReset(BaseModel):
    """Password reset request."""
    email: EmailStr


class PasswordUpdate(BaseModel):
    """Password update request."""
    token: str
    new_password: str = Field(..., min_length=8)


# ==================== HELPER FUNCTIONS ====================

def hash_password(password: str) -> str:
    """Hash a password for storing."""
    # BCrypt has a maximum password length of 72 bytes
    # Truncate if necessary (though passwords should be reasonable length)
    password_bytes = password.encode('utf-8')
    if len(password_bytes) > 72:
        password_bytes = password_bytes[:72]
    return pwd_context.hash(password_bytes.decode('utf-8'))


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict[str, Any], expires_delta: timedelta | None = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(hours=JWT_EXPIRATION_HOURS))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    """Decode and verify a JWT access token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )


async def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials, Depends(security)]
) -> dict[str, Any]:
    """Get current user from JWT token."""
    token = credentials.credentials
    payload = decode_access_token(token)

    user_id = payload.get("sub")
    if not user_id or user_id not in users_db:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return users_db[user_id]


# ==================== AUTH ENDPOINTS ====================

@router.post("/signup", response_model=TokenResponse)
async def signup(request: UserSignup):
    """
    Create a new user account.

    This endpoint:
    1. Creates a user account
    2. Generates an API key
    3. Returns a JWT token for immediate login
    """
    # Check if email already exists
    if any(u["email"] == request.email for u in users_db.values()):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered",
        )

    # Create user
    user_id = f"user_{secrets.token_urlsafe(16)}"
    user = {
        "id": user_id,
        "email": request.email,
        "full_name": request.full_name,
        "company": request.company,
        "password_hash": hash_password(request.password),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "api_key_id": None,
        "subscription_status": None,
        "tier": "free",
    }

    # Generate API key for the user
    api_key_manager = get_api_key_manager()
    raw_key, api_key = api_key_manager.generate_key(
        name=f"{request.full_name}'s API Key",
        tier=KeyTier.FREE,
        owner_id=user_id,
    )

    user["api_key_id"] = api_key.key_id
    user["api_key"] = raw_key  # Store temporarily for initial response

    users_db[user_id] = user

    # Create session token
    access_token = create_access_token({"sub": user_id})

    # Return user data without password hash
    user_response = {k: v for k, v in user.items() if k != "password_hash"}

    logger.info("User signed up", user_id=user_id, email=request.email)

    return TokenResponse(
        access_token=access_token,
        user=user_response,
    )


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLogin):
    """
    Login with email and password.

    Returns a JWT token for API access.
    """
    # Find user by email
    user = None
    for u in users_db.values():
        if u["email"] == request.email:
            user = u
            break

    if not user or not verify_password(request.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    # Create session token
    access_token = create_access_token({"sub": user["id"]})

    # Return user data without sensitive fields
    user_response = {
        k: v for k, v in user.items()
        if k not in ["password_hash", "api_key"]
    }

    logger.info("User logged in", user_id=user["id"], email=request.email)

    return TokenResponse(
        access_token=access_token,
        user=user_response,
    )


@router.post("/logout")
async def logout(current_user: dict[str, Any] = Depends(get_current_user)):
    """
    Logout current user.

    In a production system, this would invalidate the token on the server side.
    """
    logger.info("User logged out", user_id=current_user["id"])
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: dict[str, Any] = Depends(get_current_user)):
    """Get current user profile."""
    return UserResponse(
        id=current_user["id"],
        email=current_user["email"],
        full_name=current_user["full_name"],
        company=current_user.get("company"),
        created_at=current_user["created_at"],
        api_key_id=current_user.get("api_key_id"),
        subscription_status=current_user.get("subscription_status"),
        tier=current_user.get("tier", "free"),
    )


@router.put("/me")
async def update_profile(
    full_name: str | None = None,
    company: str | None = None,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Update user profile."""
    if full_name:
        current_user["full_name"] = full_name
    if company is not None:
        current_user["company"] = company

    users_db[current_user["id"]] = current_user

    logger.info("User profile updated", user_id=current_user["id"])

    return {
        "message": "Profile updated successfully",
        "user": {
            k: v for k, v in current_user.items()
            if k not in ["password_hash", "api_key"]
        }
    }


@router.post("/password-reset")
async def request_password_reset(request: PasswordReset):
    """
    Request a password reset email.

    In production, this would send an email with a reset link.
    """
    # Find user by email
    user = None
    for u in users_db.values():
        if u["email"] == request.email:
            user = u
            break

    if not user:
        # Don't reveal if email exists
        return {"message": "If the email exists, a reset link has been sent"}

    # Generate reset token
    reset_token = secrets.token_urlsafe(32)
    user["reset_token"] = reset_token
    user["reset_token_expires"] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat()

    logger.info("Password reset requested", user_id=user["id"])

    # In production, send email here
    # For development, return the token
    if get_settings().env == "development":
        return {"message": "Reset token generated", "token": reset_token}

    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-update")
async def update_password(request: PasswordUpdate):
    """Update password with reset token."""
    # Find user with matching reset token
    user = None
    for u in users_db.values():
        if u.get("reset_token") == request.token:
            # Check if token is expired
            expires_at = datetime.fromisoformat(u.get("reset_token_expires", ""))
            if expires_at < datetime.now(timezone.utc):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Reset token has expired",
                )
            user = u
            break

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token",
        )

    # Update password
    user["password_hash"] = hash_password(request.new_password)
    user.pop("reset_token", None)
    user.pop("reset_token_expires", None)

    logger.info("Password updated", user_id=user["id"])

    return {"message": "Password updated successfully"}


@router.post("/verify-token")
async def verify_token(current_user: dict[str, Any] = Depends(get_current_user)):
    """Verify if a token is valid."""
    return {
        "valid": True,
        "user_id": current_user["id"],
        "email": current_user["email"],
    }


# ==================== API KEY MANAGEMENT ====================

@router.get("/api-keys")
async def list_api_keys(current_user: dict[str, Any] = Depends(get_current_user)):
    """List user's API keys."""
    api_key_manager = get_api_key_manager()
    keys = api_key_manager.list_keys(owner_id=current_user["id"])

    return {
        "keys": [
            {
                "key_id": k.key_id,
                "name": k.name,
                "tier": k.tier.value,
                "created_at": k.created_at.isoformat(),
                "last_used": k.last_used.isoformat() if k.last_used else None,
                "is_active": k.is_active,
            }
            for k in keys
        ]
    }


@router.post("/api-keys")
async def create_api_key(
    name: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Create a new API key."""
    api_key_manager = get_api_key_manager()

    # Determine tier based on subscription
    tier = KeyTier(current_user.get("tier", "free").upper())

    raw_key, api_key = api_key_manager.generate_key(
        name=name,
        tier=tier,
        owner_id=current_user["id"],
    )

    logger.info("API key created", user_id=current_user["id"], key_id=api_key.key_id)

    return {
        "key": raw_key,
        "key_id": api_key.key_id,
        "name": api_key.name,
        "tier": api_key.tier.value,
        "warning": "Store this key securely. You won't be able to see it again.",
    }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Revoke an API key."""
    api_key_manager = get_api_key_manager()

    # Verify ownership
    key = api_key_manager.get_key(key_id)
    if not key or key.owner_id != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    api_key_manager.revoke_key(key_id)

    logger.info("API key revoked", user_id=current_user["id"], key_id=key_id)

    return {"message": "API key revoked successfully"}