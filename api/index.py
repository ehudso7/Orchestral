"""
Vercel serverless function entrypoint.

This module exports the FastAPI application for Vercel deployment.
"""

import sys
from pathlib import Path

# Add src directory to Python path for package resolution
src_path = Path(__file__).parent.parent / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from orchestral.api.server import app

# Validate app is properly initialized
if not hasattr(app, "router"):
    raise RuntimeError("FastAPI app is not properly initialized")

# Export app for Vercel
__all__ = ["app"]
