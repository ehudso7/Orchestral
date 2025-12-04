"""
Configuration management for Orchestral.

Supports environment variables, .env files, and YAML configuration.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderSettings(BaseSettings):
    """Settings for AI provider API keys and endpoints."""

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: SecretStr | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_org_id: str | None = Field(default=None, alias="OPENAI_ORG_ID")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        alias="OPENAI_BASE_URL"
    )

    # Anthropic Configuration
    anthropic_api_key: SecretStr | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    anthropic_base_url: str = Field(
        default="https://api.anthropic.com",
        alias="ANTHROPIC_BASE_URL"
    )

    # Google Configuration
    google_api_key: SecretStr | None = Field(default=None, alias="GOOGLE_API_KEY")
    google_project_id: str | None = Field(default=None, alias="GOOGLE_PROJECT_ID")
    google_location: str = Field(default="us-central1", alias="GOOGLE_LOCATION")

    @property
    def has_openai(self) -> bool:
        """Check if OpenAI is configured."""
        return self.openai_api_key is not None

    @property
    def has_anthropic(self) -> bool:
        """Check if Anthropic is configured."""
        return self.anthropic_api_key is not None

    @property
    def has_google(self) -> bool:
        """Check if Google is configured."""
        return self.google_api_key is not None

    @property
    def available_providers(self) -> list[str]:
        """List of configured providers."""
        providers = []
        if self.has_openai:
            providers.append("openai")
        if self.has_anthropic:
            providers.append("anthropic")
        if self.has_google:
            providers.append("google")
        return providers


class OrchestratorSettings(BaseSettings):
    """Core orchestrator settings."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Default models per provider
    default_openai_model: str = "gpt-4o"
    default_anthropic_model: str = "claude-sonnet-4-5-20250929"
    default_google_model: str = "gemini-3-pro-preview"

    # Execution settings
    default_timeout: float = 120.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_parallel_requests: int = 5

    # Context and token limits
    max_context_tokens: int = 100_000
    default_max_output_tokens: int = 4096

    # Caching
    enable_response_cache: bool = True
    cache_ttl_seconds: int = 3600

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v


class RedisSettings(BaseSettings):
    """Redis configuration for commercial features."""

    model_config = SettingsConfigDict(
        env_prefix="REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str | None = Field(default=None, alias="REDIS_URL")
    host: str = "localhost"
    port: int = 6379
    password: SecretStr | None = None
    db: int = 0
    ssl: bool = False

    # Connection pool settings
    max_connections: int = 10
    socket_timeout: float = 5.0

    @property
    def is_configured(self) -> bool:
        """Check if Redis is configured."""
        return self.url is not None or self.host != "localhost"


class BillingSettings(BaseSettings):
    """Billing and commercial settings."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRAL_BILLING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Feature flags
    enabled: bool = True
    usage_tracking_enabled: bool = True
    budget_alerts_enabled: bool = True

    # API Key Management - CRITICAL: This secret must be persistent across restarts
    # Generate once with: python -c "import secrets; print(secrets.token_hex(32))"
    api_key_secret: SecretStr | None = Field(
        default=None,
        description="Secret key for API key hashing. Must be persistent across restarts!"
    )

    # Caching
    cache_enabled: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_entries: int = 10000

    # Budget defaults
    default_monthly_budget_usd: float = 100.0
    budget_alert_threshold: float = 0.8  # Alert at 80%
    hard_budget_limit: bool = False  # If True, block requests over budget

    # Webhook for alerts
    alert_webhook_url: str | None = None


class ServerSettings(BaseSettings):
    """API server settings."""

    model_config = SettingsConfigDict(
        env_prefix="ORCHESTRAL_SERVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_seconds: int = 60
    rate_limit_by_key: bool = True  # Use per-API-key limits if True

    # Authentication
    api_key_header: str = "X-API-Key"
    require_auth: bool = True  # Default to True for commercial
    api_keys: list[str] = Field(default_factory=list)

    # Admin API
    admin_api_enabled: bool = True
    admin_api_key: SecretStr | None = None


class Settings(BaseSettings):
    """Combined application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    providers: ProviderSettings = Field(default_factory=ProviderSettings)
    orchestrator: OrchestratorSettings = Field(default_factory=OrchestratorSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    billing: BillingSettings = Field(default_factory=BillingSettings)

    # Environment
    environment: str = Field(default="development", alias="ENVIRONMENT")
    debug: bool = Field(default=False, alias="DEBUG")

    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.environment.lower() == "production"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Settings":
        """Load settings from a YAML file."""
        import yaml

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls(**data)


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


@lru_cache()
def get_provider_settings() -> ProviderSettings:
    """Get cached provider settings."""
    return ProviderSettings()


def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    get_provider_settings.cache_clear()
    return get_settings()
