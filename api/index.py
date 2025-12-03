"""
Vercel serverless function entrypoint.

This module exports the FastAPI application for Vercel deployment.
"""

import sys
from pathlib import Path

from fastapi import FastAPI

# Add src directory to Python path for package resolution
# Use resolve() for absolute path and is_dir() for specific directory check
src_path = Path(__file__).resolve().parent.parent / "src"
if src_path.is_dir() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from orchestral.api.server import app

# Validate app is properly initialized
if not isinstance(app, FastAPI):
    raise TypeError(f"app must be a FastAPI instance, got {type(app).__name__}")

# Export app for Vercel
__all__ = ["app"]
