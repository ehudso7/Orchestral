"""
Vercel serverless function entrypoint.

This module exports the FastAPI application for Vercel deployment.
"""

from orchestral.api.server import app

# Export app for Vercel
__all__ = ["app"]
