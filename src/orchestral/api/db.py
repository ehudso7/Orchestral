"""
Simple database layer for persistence during development.

This module provides a simple file-based persistence layer
for user data and API keys during development.
In production, this should be replaced with a proper database.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional
import threading

class SimpleDB:
    """Simple file-based database for development."""

    def __init__(self, db_path: str = "/tmp/orchestral_db.json"):
        """Initialize the database."""
        self.db_path = Path(db_path)
        self.lock = threading.Lock()
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Ensure the database file exists."""
        if not self.db_path.exists():
            self._save_data({
                "users": {},
                "api_keys": {},
                "sessions": {}
            })

    def _load_data(self) -> Dict[str, Any]:
        """Load data from the database file."""
        try:
            with open(self.db_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "users": {},
                "api_keys": {},
                "sessions": {}
            }

    def _save_data(self, data: Dict[str, Any]):
        """Save data to the database file."""
        with self.lock:
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID."""
        data = self._load_data()
        return data["users"].get(user_id)

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Get a user by email."""
        data = self._load_data()
        for user in data["users"].values():
            if user.get("email") == email:
                return user
        return None

    def create_user(self, user_id: str, user_data: Dict[str, Any]):
        """Create a new user."""
        data = self._load_data()
        data["users"][user_id] = user_data
        self._save_data(data)

    def update_user(self, user_id: str, user_data: Dict[str, Any]):
        """Update a user."""
        data = self._load_data()
        if user_id in data["users"]:
            data["users"][user_id].update(user_data)
            self._save_data(data)

    def get_all_users(self) -> Dict[str, Dict[str, Any]]:
        """Get all users."""
        data = self._load_data()
        return data["users"]

    def save_api_key(self, key_id: str, key_data: Dict[str, Any]):
        """Save an API key."""
        data = self._load_data()
        data["api_keys"][key_id] = key_data
        self._save_data(data)

    def get_api_key(self, key_id: str) -> Optional[Dict[str, Any]]:
        """Get an API key."""
        data = self._load_data()
        return data["api_keys"].get(key_id)

    def get_user_api_keys(self, user_id: str) -> list[Dict[str, Any]]:
        """Get all API keys for a user."""
        data = self._load_data()
        user_keys = []
        for key_data in data["api_keys"].values():
            if key_data.get("owner_id") == user_id:
                user_keys.append(key_data)
        return user_keys

    def delete_api_key(self, key_id: str):
        """Delete an API key."""
        data = self._load_data()
        if key_id in data["api_keys"]:
            del data["api_keys"][key_id]
            self._save_data(data)

# Global database instance
db = SimpleDB()