"""
Vercel serverless function entrypoint.

This module exports the FastAPI application for Vercel deployment.
"""

import os
import sys
from pathlib import Path

from fastapi import FastAPI

# Set default environment variables for Vercel if not already set
if not os.getenv("ORCHESTRAL_BILLING_API_KEY_SECRET"):
    # Use a default secret for Vercel deployments without proper config
    # This allows the app to start but should be replaced in production
    os.environ["ORCHESTRAL_BILLING_API_KEY_SECRET"] = "0" * 64

if not os.getenv("ORCHESTRAL_SERVER_ADMIN_API_KEY"):
    # Default admin key for JWT signing
    os.environ["ORCHESTRAL_SERVER_ADMIN_API_KEY"] = "default-admin-key-replace-in-production"

# Add src directory to Python path for package resolution
# Use resolve() for absolute path and is_dir() for specific directory check
src_path = Path(__file__).resolve().parent.parent / "src"
if src_path.is_dir() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from orchestral.api.server import app
except Exception as e:
    # Log the error for debugging
    print(f"Failed to import app: {e}", file=sys.stderr)
    # Create a minimal error app
    app = FastAPI(title="Orchestral API - Error State")

    @app.get("/")
    async def error_root():
        return {"error": "Application failed to initialize. Check logs for details."}

    @app.get("/health")
    async def error_health():
        return {"status": "error", "message": "Application initialization failed"}

# Validate app is properly initialized
if not isinstance(app, FastAPI):
    raise TypeError(f"app must be a FastAPI instance, got {type(app).__name__}")

# Export app for Vercel
__all__ = ["app"]
