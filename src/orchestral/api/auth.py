"""
Authentication API endpoints for Orchestral.

Provides JWT-based authentication with secure user management.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any

import bcrypt
import structlog
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, field_validator
import jwt

from orchestral.core.config import get_settings
from orchestral.billing.api_keys import get_api_key_manager, KeyTier
from orchestral.api.db import db

logger = structlog.get_logger()

# Security setup
security = HTTPBearer()

router = APIRouter(prefix="/auth", tags=["authentication"])

# JWT configuration
JWT_SECRET = get_settings().server.admin_api_key.get_secret_value() if get_settings().server.admin_api_key else secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


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
    """
    Hash a password for storing - supports UNLIMITED password length.

    This implementation uses SHA256 pre-hashing + bcrypt directly,
    completely bypassing passlib to avoid any 72-byte limitations.

    Industry standard approach used by Django, Rails, and others.
    """
    # Step 1: SHA256 hash the password (handles ANY length)
    # This produces a 64-character hex string
    sha256_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()

    # Step 2: Use bcrypt directly on the SHA256 hash
    # The SHA256 hash is always 64 bytes, well under bcrypt's 72-byte limit
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(sha256_hash.encode('utf-8'), salt)

    # Return as string for storage
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash - supports UNLIMITED password length.

    Uses the same SHA256 pre-hashing approach for consistency.
    """
    # Step 1: SHA256 hash the password (same as during hashing)
    sha256_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()

    # Step 2: Check against the stored bcrypt hash
    try:
        return bcrypt.checkpw(
            sha256_hash.encode('utf-8'),
            hashed_password.encode('utf-8')
        )
    except Exception as e:
        logger.error(f"Password verification failed: {e}")
        return False


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
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
        )

    user = db.get_user(user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    return user


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
    try:
        # Check if email already exists
        if db.get_user_by_email(request.email):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )

        # Create user - NO password length limitations!
        user_id = f"user_{secrets.token_urlsafe(16)}"
        user = {
            "id": user_id,
            "email": request.email,
            "full_name": request.full_name,
            "company": request.company,
            "password_hash": hash_password(request.password),  # Handles ANY password length
            "created_at": datetime.now(timezone.utc).isoformat(),
            "api_key_id": None,
            "subscription_status": None,
            "tier": "free",
        }

        # Generate API key for the user
        try:
            api_key_manager = get_api_key_manager()
            raw_key, api_key = api_key_manager.generate_key(
                name=f"{request.full_name}'s API Key",
                tier=KeyTier.FREE,
                owner_id=user_id,
            )
            user["api_key_id"] = api_key.key_id
            user["api_key"] = raw_key  # Store temporarily for initial response

            # Save API key to database
            db.save_api_key(api_key.key_id, {
                "key_id": api_key.key_id,
                "name": api_key.name,
                "owner_id": user_id,
                "tier": "free",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "is_active": True,
                "raw_key": raw_key  # Store for development only
            })
        except Exception as e:
            # If API key generation fails, create a simple fallback key
            logger.warning(f"API key generation failed: {e}, using fallback")
            fallback_key = f"orch_{secrets.token_hex(8)}_{secrets.token_urlsafe(32)}"
            key_id = f"orch_{secrets.token_hex(8)}"
            user["api_key_id"] = key_id
            user["api_key"] = fallback_key

            # Save fallback key to database
            db.save_api_key(key_id, {
                "key_id": key_id,
                "name": f"{request.full_name}'s API Key",
                "owner_id": user_id,
                "tier": "free",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "is_active": True,
                "raw_key": fallback_key  # Store for development only
            })

        # Save user to database
        db.create_user(user_id, user)

        # Send welcome notification
        try:
            from orchestral.api.notifications import notify_welcome
            notify_welcome(user_id, request.full_name)
        except Exception as e:
            logger.warning(f"Failed to send welcome notification: {e}")

        # Create session token
        access_token = create_access_token({"sub": user_id})

        # Return user data without password hash
        user_response = {k: v for k, v in user.items() if k != "password_hash"}

        logger.info("User signed up", user_id=user_id, email=request.email)

        return TokenResponse(
            access_token=access_token,
            user=user_response,
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Signup failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Signup failed: {str(e)}",
        )


@router.post("/login", response_model=TokenResponse)
async def login(request: UserLogin):
    """
    Login with email and password.

    Returns a JWT token for API access.
    Supports passwords of ANY length thanks to SHA256 pre-hashing.
    """
    # Find user by email
    user = db.get_user_by_email(request.email)

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

    db.update_user(current_user["id"], current_user)

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
    user = db.get_user_by_email(request.email)

    if not user:
        # Don't reveal if email exists
        return {"message": "If the email exists, a reset link has been sent"}

    # Generate reset token
    reset_token = secrets.token_urlsafe(32)
    user["reset_token"] = reset_token
    user["reset_token_expires"] = (
        datetime.now(timezone.utc) + timedelta(hours=1)
    ).isoformat()

    # Update user in database
    db.update_user(user["id"], user)

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
    all_users = db.get_all_users()
    for u in all_users.values():
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

    # Update user in database
    db.update_user(user["id"], user)

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
    # Get keys from database
    keys = db.get_user_api_keys(current_user["id"])

    return {
        "keys": [
            {
                "key_id": k.get("key_id"),
                "name": k.get("name"),
                "tier": k.get("tier", "free"),
                "created_at": k.get("created_at"),
                "last_used": k.get("last_used"),
                "is_active": k.get("is_active", True),
            }
            for k in keys
        ]
    }


class CreateAPIKeyRequest(BaseModel):
    """Request to create an API key."""
    name: str = Field(..., description="Key name")


@router.post("/api-keys")
async def create_api_key(
    request: CreateAPIKeyRequest,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Create a new API key."""
    try:
        api_key_manager = get_api_key_manager()

        # Determine tier based on subscription
        tier = KeyTier(current_user.get("tier", "free").upper())

        raw_key, api_key = api_key_manager.generate_key(
            name=request.name,
            tier=tier,
            owner_id=current_user["id"],
        )

        # Save to database
        db.save_api_key(api_key.key_id, {
            "key_id": api_key.key_id,
            "name": request.name,
            "owner_id": current_user["id"],
            "tier": tier.value.lower(),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_active": True,
            "raw_key": raw_key  # Store for development only
        })

        logger.info("API key created", user_id=current_user["id"], key_id=api_key.key_id)

        return {
            "key": raw_key,
            "key_id": api_key.key_id,
            "name": api_key.name,
            "tier": api_key.tier.value,
            "warning": "Store this key securely. You won't be able to see it again.",
        }
    except Exception as e:
        # Fallback to simple key generation
        logger.warning(f"API key generation failed: {e}, using fallback")
        key_id = f"orch_{secrets.token_hex(8)}"
        raw_key = f"orch_{secrets.token_hex(8)}_{secrets.token_urlsafe(32)}"

        # Save to database
        db.save_api_key(key_id, {
            "key_id": key_id,
            "name": request.name,
            "owner_id": current_user["id"],
            "tier": current_user.get("tier", "free"),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "is_active": True,
            "raw_key": raw_key  # Store for development only
        })

        return {
            "key": raw_key,
            "key_id": key_id,
            "name": request.name,
            "tier": current_user.get("tier", "free"),
            "warning": "Store this key securely. You won't be able to see it again.",
        }


@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: str,
    current_user: dict[str, Any] = Depends(get_current_user),
):
    """Revoke an API key."""
    # Verify ownership from database
    key = db.get_api_key(key_id)
    if not key or key.get("owner_id") != current_user["id"]:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found",
        )

    # Delete from database
    db.delete_api_key(key_id)

    # Try to revoke in API key manager if available
    try:
        api_key_manager = get_api_key_manager()
        api_key_manager.revoke_key(key_id)
    except Exception:
        pass  # Ignore errors from API key manager

    logger.info("API key revoked", user_id=current_user["id"], key_id=key_id)

    return {"message": "API key revoked successfully"}