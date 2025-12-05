"""
Prompt management and versioning for Orchestral.

Provides a centralized system for managing, versioning, and deploying
prompts - a critical feature for production LLM applications.
Prompt template management with versioning and variable substitution.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger()


class PromptStatus(str, Enum):
    """Status of a prompt version."""

    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    TESTING = "testing"  # In A/B test
import re
import json
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from pathlib import Path

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """Configuration for a prompt template."""

    name: str = Field(..., description="Prompt name")
    description: str = Field(default="", description="Prompt description")
    category: str = Field(default="general", description="Prompt category")
    model_hint: str | None = Field(default=None, description="Recommended model")
    temperature_hint: float | None = Field(default=None, description="Recommended temperature")
    max_tokens_hint: int | None = Field(default=None, description="Recommended max tokens")
    tags: list[str] = Field(default_factory=list, description="Tags for organization")


@dataclass
class PromptVersion:
    """A specific version of a prompt."""

    version_id: str
    prompt_id: str
    version: int
    content: str
    variables: list[str]  # Extracted template variables
    status: PromptStatus
    created_at: datetime
    created_by: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    total_uses: int = 0
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    avg_cost_usd: float = 0.0
    success_rate: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "version_id": self.version_id,
            "prompt_id": self.prompt_id,
            "version": self.version,
            "content": self.content,
            "variables": self.variables,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "description": self.description,
            "metadata": self.metadata,
            "total_uses": self.total_uses,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_tokens": self.avg_tokens,
            "avg_cost_usd": self.avg_cost_usd,
            "success_rate": self.success_rate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PromptVersion":
        """Create from dictionary."""
        return cls(
            version_id=data["version_id"],
            prompt_id=data["prompt_id"],
            version=data["version"],
            content=data["content"],
            variables=data["variables"],
            status=PromptStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data.get("created_by"),
            description=data.get("description"),
            metadata=data.get("metadata", {}),
            total_uses=data.get("total_uses", 0),
            avg_latency_ms=data.get("avg_latency_ms", 0.0),
            avg_tokens=data.get("avg_tokens", 0.0),
            avg_cost_usd=data.get("avg_cost_usd", 0.0),
            success_rate=data.get("success_rate", 1.0),
        )


@dataclass
class Prompt:
    """A managed prompt with version history."""

    prompt_id: str
    name: str
    description: str | None = None
    tenant_id: str = "global"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Current active version
    active_version: int | None = None

    # Version history
    versions: list[PromptVersion] = field(default_factory=list)

    def get_active(self) -> PromptVersion | None:
        """Get the active version."""
        for v in self.versions:
            if v.status == PromptStatus.ACTIVE:
                return v
        return None

    def get_version(self, version: int) -> PromptVersion | None:
        """Get a specific version."""
        for v in self.versions:
            if v.version == version:
                return v
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "prompt_id": self.prompt_id,
            "name": self.name,
            "description": self.description,
            "tenant_id": self.tenant_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tags": self.tags,
            "metadata": self.metadata,
            "active_version": self.active_version,
            "versions": [v.to_dict() for v in self.versions],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prompt":
        """Create from dictionary."""
        prompt = cls(
            prompt_id=data["prompt_id"],
            name=data["name"],
            description=data.get("description"),
            tenant_id=data.get("tenant_id", "global"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
            active_version=data.get("active_version"),
        )
        prompt.versions = [
            PromptVersion.from_dict(v) for v in data.get("versions", [])
        ]
        return prompt


class PromptTemplate:
    """
    Template engine for prompt rendering.

    Supports:
    - Variable substitution: {{variable}}
    - Conditionals: {{#if condition}}...{{/if}}
    - Loops: {{#each items}}...{{/each}}
    - Filters: {{variable|uppercase}}
    """

    VARIABLE_PATTERN = re.compile(r"\{\{(\w+)(?:\|(\w+))?\}\}")
    CONDITIONAL_PATTERN = re.compile(
        r"\{\{#if\s+(\w+)\}\}(.*?)\{\{/if\}\}", re.DOTALL
    )
    LOOP_PATTERN = re.compile(
        r"\{\{#each\s+(\w+)\}\}(.*?)\{\{/each\}\}", re.DOTALL
    )

    FILTERS = {
        "uppercase": lambda x: str(x).upper(),
        "lowercase": lambda x: str(x).lower(),
        "capitalize": lambda x: str(x).capitalize(),
        "trim": lambda x: str(x).strip(),
        "json": lambda x: json.dumps(x),
        "length": lambda x: str(len(x)),
    }

    @classmethod
    def extract_variables(cls, template: str) -> list[str]:
        """Extract all variable names from a template."""
        variables = set()

        # Simple variables
        for match in cls.VARIABLE_PATTERN.finditer(template):
            variables.add(match.group(1))

        # Conditional variables
        for match in cls.CONDITIONAL_PATTERN.finditer(template):
            variables.add(match.group(1))

        # Loop variables
        for match in cls.LOOP_PATTERN.finditer(template):
            variables.add(match.group(1))

        return sorted(list(variables))

    @classmethod
    def render(
        cls,
        template: str,
        variables: dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render a template with variables.

        Args:
            template: Template string
            variables: Variable values
            strict: Raise error for missing variables

        Returns:
            Rendered string
        """
        result = template

        # Process conditionals first
        def replace_conditional(match: re.Match) -> str:
            var_name = match.group(1)
            content = match.group(2)
            value = variables.get(var_name)
            if value:
                return content
            return ""

        result = cls.CONDITIONAL_PATTERN.sub(replace_conditional, result)

        # Process loops
        def replace_loop(match: re.Match) -> str:
            var_name = match.group(1)
            content = match.group(2)
            items = variables.get(var_name, [])
            if not isinstance(items, (list, tuple)):
                items = [items]
            rendered_items = []
            for i, item in enumerate(items):
                item_content = content
                item_content = item_content.replace("{{this}}", str(item))
                item_content = item_content.replace("{{@index}}", str(i))
                rendered_items.append(item_content)
            return "".join(rendered_items)

        result = cls.LOOP_PATTERN.sub(replace_loop, result)

        # Process simple variables with filters
        def replace_variable(match: re.Match) -> str:
            var_name = match.group(1)
            filter_name = match.group(2)

            if var_name not in variables:
                if strict:
                    raise ValueError(f"Missing variable: {var_name}")
                return match.group(0)  # Return unchanged

            value = variables[var_name]

            if filter_name and filter_name in cls.FILTERS:
                value = cls.FILTERS[filter_name](value)

            return str(value)

        result = cls.VARIABLE_PATTERN.sub(replace_variable, result)

        return result

    @classmethod
    def validate(cls, template: str, variables: dict[str, Any]) -> list[str]:
        """
        Validate template against provided variables.

        Returns list of missing variables.
        """
        required = cls.extract_variables(template)
        provided = set(variables.keys())
        return [v for v in required if v not in provided]


class PromptManager:
    """
    Central manager for prompts and versions.

    Features:
    - Create and version prompts
    - Activate/deactivate versions
    - Track usage and performance
    - Tenant isolation
    """

    PROMPT_PREFIX = "orch:prompt:"
    INDEX_PREFIX = "orch:prompts:idx:"

    def __init__(
        self,
        redis_client: Any | None = None,
        enabled: bool = True,
    ):
        self._redis = redis_client
        self._enabled = enabled
        self._local_store: dict[str, Prompt] = {}

    async def create_prompt(
        self,
        name: str,
        content: str,
        tenant_id: str = "global",
        description: str | None = None,
        tags: list[str] | None = None,
        created_by: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Prompt:
        """
        Create a new prompt with initial version.

        Args:
            name: Prompt name
            content: Initial prompt content
            tenant_id: Tenant ID
            description: Prompt description
            tags: Tags for categorization
            created_by: Creator identifier
            metadata: Additional metadata

        Returns:
            Created Prompt object
        """
        now = datetime.now(timezone.utc)

        # Generate IDs
        prompt_id = hashlib.sha256(
            f"{tenant_id}:{name}:{now.timestamp()}".encode()
        ).hexdigest()[:16]

        version_id = hashlib.sha256(
            f"{prompt_id}:1:{content}".encode()
        ).hexdigest()[:16]

        # Extract variables
        variables = PromptTemplate.extract_variables(content)

        # Create initial version
        version = PromptVersion(
            version_id=version_id,
            prompt_id=prompt_id,
            version=1,
            content=content,
            variables=variables,
            status=PromptStatus.ACTIVE,
            created_at=now,
            created_by=created_by,
            description="Initial version",
            metadata=metadata or {},
        )

        # Create prompt
        prompt = Prompt(
            prompt_id=prompt_id,
            name=name,
            description=description,
            tenant_id=tenant_id,
            created_at=now,
            updated_at=now,
            tags=tags or [],
            metadata=metadata or {},
            active_version=1,
            versions=[version],
        )

        await self._store_prompt(prompt)

        logger.info(
            "Prompt created",
            prompt_id=prompt_id,
            name=name,
            tenant_id=tenant_id,
        )

        return prompt

    async def add_version(
        self,
        prompt_id: str,
        content: str,
        description: str | None = None,
        created_by: str | None = None,
        activate: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> PromptVersion:
        """
        Add a new version to an existing prompt.

        Args:
            prompt_id: Prompt ID
            content: New version content
            description: Version description
            created_by: Creator
            activate: Activate this version immediately
            metadata: Additional metadata

        Returns:
            Created PromptVersion
        """
        prompt = await self.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_id}")

        now = datetime.now(timezone.utc)
        new_version_num = max(v.version for v in prompt.versions) + 1

        version_id = hashlib.sha256(
            f"{prompt_id}:{new_version_num}:{content}".encode()
        ).hexdigest()[:16]

        variables = PromptTemplate.extract_variables(content)

        version = PromptVersion(
            version_id=version_id,
            prompt_id=prompt_id,
            version=new_version_num,
            content=content,
            variables=variables,
            status=PromptStatus.ACTIVE if activate else PromptStatus.DRAFT,
            created_at=now,
            created_by=created_by,
            description=description,
            metadata=metadata or {},
        )

        # Deactivate current active if activating new
        if activate:
            for v in prompt.versions:
                if v.status == PromptStatus.ACTIVE:
                    v.status = PromptStatus.ARCHIVED
            prompt.active_version = new_version_num

        prompt.versions.append(version)
        prompt.updated_at = now

        await self._store_prompt(prompt)

        logger.info(
            "Prompt version added",
            prompt_id=prompt_id,
            version=new_version_num,
            activated=activate,
        )

        return version

    async def activate_version(
        self,
        prompt_id: str,
        version: int,
    ) -> PromptVersion:
        """Activate a specific version of a prompt."""
        prompt = await self.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_id}")

        target_version = prompt.get_version(version)
        if not target_version:
            raise ValueError(f"Version {version} not found")

        # Deactivate current active
        for v in prompt.versions:
            if v.status == PromptStatus.ACTIVE:
                v.status = PromptStatus.ARCHIVED

        # Activate target
        target_version.status = PromptStatus.ACTIVE
        prompt.active_version = version
        prompt.updated_at = datetime.now(timezone.utc)

        await self._store_prompt(prompt)

        logger.info(
            "Prompt version activated",
            prompt_id=prompt_id,
            version=version,
        )

        return target_version

    async def get_prompt(self, prompt_id: str) -> Prompt | None:
        """Get a prompt by ID."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                data = await loop.run_in_executor(
                    None, self._redis.get, f"{self.PROMPT_PREFIX}{prompt_id}"
                )
                if data:
                    return Prompt.from_dict(json.loads(data))
            except Exception as e:
                logger.warning("Failed to get prompt", error=str(e))
        else:
            return self._local_store.get(prompt_id)
        return None

    async def get_prompt_by_name(
        self,
        name: str,
        tenant_id: str = "global",
    ) -> Prompt | None:
        """Get a prompt by name within a tenant."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                prompt_id = await loop.run_in_executor(
                    None,
                    self._redis.hget,
                    f"{self.INDEX_PREFIX}name:{tenant_id}",
                    name,
                )
                if prompt_id:
                    pid = (
                        prompt_id.decode("utf-8")
                        if isinstance(prompt_id, bytes)
                        else prompt_id
                    )
                    return await self.get_prompt(pid)
            except Exception as e:
                logger.warning("Failed to get prompt by name", error=str(e))
        else:
            for prompt in self._local_store.values():
                if prompt.name == name and prompt.tenant_id == tenant_id:
                    return prompt
        return None

    async def list_prompts(
        self,
        tenant_id: str = "global",
        tags: list[str] | None = None,
        limit: int = 100,
    ) -> list[Prompt]:
        """List prompts for a tenant."""
        prompts = []

        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                pattern = f"{self.PROMPT_PREFIX}*"
                keys = await loop.run_in_executor(
                    None, lambda: list(self._redis.scan_iter(pattern, count=1000))
                )
                for key in keys[:limit]:
                    data = await loop.run_in_executor(None, self._redis.get, key)
                    if data:
                        prompt = Prompt.from_dict(json.loads(data))
                        if prompt.tenant_id == tenant_id:
                            if tags is None or any(t in prompt.tags for t in tags):
                                prompts.append(prompt)
            except Exception as e:
                logger.warning("Failed to list prompts", error=str(e))
        else:
            for prompt in self._local_store.values():
                if prompt.tenant_id == tenant_id:
                    if tags is None or any(t in prompt.tags for t in tags):
                        prompts.append(prompt)

        return prompts[:limit]

    async def render(
        self,
        prompt_id: str,
        variables: dict[str, Any],
        version: int | None = None,
    ) -> str:
        """
        Render a prompt template with variables.

        Args:
            prompt_id: Prompt ID
            variables: Template variables
            version: Specific version (uses active if None)

        Returns:
            Rendered prompt string
        """
        prompt = await self.get_prompt(prompt_id)
        if not prompt:
            raise ValueError(f"Prompt not found: {prompt_id}")

        if version:
            pv = prompt.get_version(version)
        else:
            pv = prompt.get_active()

        if not pv:
            raise ValueError("No active version found")

        return PromptTemplate.render(pv.content, variables)

    async def record_usage(
        self,
        prompt_id: str,
        version: int,
        latency_ms: float,
        tokens: int,
        cost_usd: float,
        success: bool = True,
    ) -> None:
        """Record usage metrics for a prompt version."""
        prompt = await self.get_prompt(prompt_id)
        if not prompt:
            return

        pv = prompt.get_version(version)
        if not pv:
            return

        # Update rolling averages
        n = pv.total_uses
        pv.total_uses = n + 1
        pv.avg_latency_ms = (pv.avg_latency_ms * n + latency_ms) / (n + 1)
        pv.avg_tokens = (pv.avg_tokens * n + tokens) / (n + 1)
        pv.avg_cost_usd = (pv.avg_cost_usd * n + cost_usd) / (n + 1)

        if not success:
            success_count = pv.success_rate * n
            pv.success_rate = success_count / (n + 1)

        await self._store_prompt(prompt)

    async def _store_prompt(self, prompt: Prompt) -> None:
        """Store a prompt."""
        if self._redis:
            loop = asyncio.get_running_loop()
            try:
                # Store prompt
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.set(
                        f"{self.PROMPT_PREFIX}{prompt.prompt_id}",
                        json.dumps(prompt.to_dict()),
                    ),
                )
                # Index by name
                await loop.run_in_executor(
                    None,
                    lambda: self._redis.hset(
                        f"{self.INDEX_PREFIX}name:{prompt.tenant_id}",
                        prompt.name,
                        prompt.prompt_id,
                    ),
                )
            except Exception as e:
                logger.warning("Failed to store prompt", error=str(e))
        else:
            self._local_store[prompt.prompt_id] = prompt


# Global prompt manager
_manager: PromptManager | None = None


def get_prompt_manager() -> PromptManager:
    """Get the global prompt manager."""
    global _manager
    if _manager is None:
        _manager = PromptManager()
    return _manager


def configure_prompt_manager(
    redis_client: Any | None = None,
    enabled: bool = True,
) -> PromptManager:
    """Configure the global prompt manager."""
    global _manager
    _manager = PromptManager(redis_client=redis_client, enabled=enabled)
    return _manager
    """A specific version of a prompt template."""

    version: int
    template: str
    system_prompt: str | None
    variables: list[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    created_by: str = "system"
    commit_message: str = ""
    is_active: bool = False

    # Performance tracking
    uses: int = 0
    avg_quality_score: float | None = None
    total_quality_samples: int = 0

    @property
    def content_hash(self) -> str:
        """Hash of the template content for comparison."""
        content = f"{self.template}|{self.system_prompt or ''}"
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "template": self.template,
            "system_prompt": self.system_prompt,
            "variables": self.variables,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "commit_message": self.commit_message,
            "is_active": self.is_active,
            "uses": self.uses,
            "avg_quality_score": round(self.avg_quality_score, 3) if self.avg_quality_score else None,
            "content_hash": self.content_hash,
        }


@dataclass
class PromptTemplate:
    """A managed prompt template with version history."""

    id: str
    config: PromptConfig
    versions: list[PromptVersion] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def active_version(self) -> PromptVersion | None:
        """Get the currently active version."""
        for v in reversed(self.versions):
            if v.is_active:
                return v
        return self.versions[-1] if self.versions else None

    @property
    def latest_version(self) -> PromptVersion | None:
        """Get the latest version."""
        return self.versions[-1] if self.versions else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.config.name,
            "description": self.config.description,
            "category": self.config.category,
            "tags": self.config.tags,
            "version_count": len(self.versions),
            "active_version": self.active_version.version if self.active_version else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


class PromptManager:
    """
    Manages prompt templates with versioning and variable substitution.

    Features:
    - Version control for prompts
    - Variable substitution with {{variable}} syntax
    - Performance tracking per version
    - Rollback capability
    - Import/export

    Example:
        manager = PromptManager()

        # Create a template
        template = manager.create(
            config=PromptConfig(name="code_review", category="coding"),
            template="Review this {{language}} code:\n\n{{code}}\n\nFocus on: {{focus}}",
            system_prompt="You are an expert code reviewer.",
        )

        # Render with variables
        prompt = manager.render("code_review", {
            "language": "Python",
            "code": "def foo(): pass",
            "focus": "best practices",
        })

        # Create new version
        manager.update("code_review",
            template="Review this {{language}} code for {{focus}}:\n\n```{{language}}\n{{code}}\n```",
            commit_message="Added code block formatting",
        )

        # Rollback if needed
        manager.rollback("code_review", version=1)
    """

    def __init__(self):
        self._templates: dict[str, PromptTemplate] = {}
        self._counter = 0

    def _extract_variables(self, template: str) -> list[str]:
        """Extract variable names from template."""
        pattern = r"\{\{(\w+)\}\}"
        return list(set(re.findall(pattern, template)))

    def _generate_id(self, name: str) -> str:
        """Generate a unique ID for a template."""
        self._counter += 1
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return f"prompt_{slug}_{self._counter}"

    def create(
        self,
        config: PromptConfig,
        template: str,
        system_prompt: str | None = None,
        created_by: str = "system",
    ) -> PromptTemplate:
        """Create a new prompt template."""
        template_id = self._generate_id(config.name)

        variables = self._extract_variables(template)
        if system_prompt:
            variables.extend(self._extract_variables(system_prompt))
        variables = list(set(variables))

        version = PromptVersion(
            version=1,
            template=template,
            system_prompt=system_prompt,
            variables=variables,
            created_by=created_by,
            commit_message="Initial version",
            is_active=True,
        )

        prompt_template = PromptTemplate(
            id=template_id,
            config=config,
            versions=[version],
        )

        self._templates[template_id] = prompt_template

        # Also index by name for convenience
        self._templates[config.name] = prompt_template

        return prompt_template

    def get(self, template_id: str) -> PromptTemplate | None:
        """Get a template by ID or name."""
        return self._templates.get(template_id)

    def update(
        self,
        template_id: str,
        template: str | None = None,
        system_prompt: str | None = None,
        commit_message: str = "",
        created_by: str = "system",
        activate: bool = True,
    ) -> PromptVersion | None:
        """Create a new version of a template."""
        prompt = self._templates.get(template_id)
        if not prompt:
            return None

        current = prompt.active_version
        if not current:
            return None

        new_template = template if template is not None else current.template
        new_system = system_prompt if system_prompt is not None else current.system_prompt

        variables = self._extract_variables(new_template)
        if new_system:
            variables.extend(self._extract_variables(new_system))
        variables = list(set(variables))

        new_version = PromptVersion(
            version=len(prompt.versions) + 1,
            template=new_template,
            system_prompt=new_system,
            variables=variables,
            created_by=created_by,
            commit_message=commit_message,
            is_active=activate,
        )

        if activate:
            # Deactivate previous active version
            for v in prompt.versions:
                v.is_active = False

        prompt.versions.append(new_version)
        prompt.updated_at = datetime.now(timezone.utc)

        return new_version

    def rollback(self, template_id: str, version: int) -> bool:
        """Rollback to a specific version."""
        prompt = self._templates.get(template_id)
        if not prompt:
            return False

        target = next((v for v in prompt.versions if v.version == version), None)
        if not target:
            return False

        for v in prompt.versions:
            v.is_active = (v.version == version)

        prompt.updated_at = datetime.now(timezone.utc)
        return True

    def render(
        self,
        template_id: str,
        variables: dict[str, Any],
        version: int | None = None,
    ) -> tuple[str, str | None] | None:
        """
        Render a template with variables.

        Args:
            template_id: Template ID or name
            variables: Variable values to substitute
            version: Specific version to use (default: active version)

        Returns:
            Tuple of (rendered_prompt, rendered_system_prompt) or None if not found
        """
        prompt = self._templates.get(template_id)
        if not prompt:
            return None

        if version:
            ver = next((v for v in prompt.versions if v.version == version), None)
        else:
            ver = prompt.active_version

        if not ver:
            return None

        # Track usage
        ver.uses += 1

        # Render template
        rendered = ver.template
        for var_name, var_value in variables.items():
            rendered = rendered.replace(f"{{{{{var_name}}}}}", str(var_value))

        # Render system prompt
        rendered_system = None
        if ver.system_prompt:
            rendered_system = ver.system_prompt
            for var_name, var_value in variables.items():
                rendered_system = rendered_system.replace(f"{{{{{var_name}}}}}", str(var_value))

        return rendered, rendered_system

    def record_quality(
        self,
        template_id: str,
        quality_score: float,
        version: int | None = None,
    ) -> None:
        """Record quality score for a version."""
        prompt = self._templates.get(template_id)
        if not prompt:
            return

        if version:
            ver = next((v for v in prompt.versions if v.version == version), None)
        else:
            ver = prompt.active_version

        if ver:
            if ver.avg_quality_score is None:
                ver.avg_quality_score = quality_score
            else:
                # Running average
                ver.avg_quality_score = (
                    (ver.avg_quality_score * ver.total_quality_samples + quality_score)
                    / (ver.total_quality_samples + 1)
                )
            ver.total_quality_samples += 1

    def list_templates(self, category: str | None = None) -> list[dict[str, Any]]:
        """List all templates, optionally filtered by category."""
        seen_ids = set()
        templates = []

        for template in self._templates.values():
            if template.id in seen_ids:
                continue
            seen_ids.add(template.id)

            if category and template.config.category != category:
                continue

            templates.append(template.to_dict())

        return templates

    def get_version_history(self, template_id: str) -> list[dict[str, Any]]:
        """Get version history for a template."""
        prompt = self._templates.get(template_id)
        if not prompt:
            return []

        return [v.to_dict() for v in prompt.versions]

    def compare_versions(
        self, template_id: str, version1: int, version2: int
    ) -> dict[str, Any] | None:
        """Compare two versions of a template."""
        prompt = self._templates.get(template_id)
        if not prompt:
            return None

        v1 = next((v for v in prompt.versions if v.version == version1), None)
        v2 = next((v for v in prompt.versions if v.version == version2), None)

        if not v1 or not v2:
            return None

        return {
            "version1": v1.to_dict(),
            "version2": v2.to_dict(),
            "template_changed": v1.template != v2.template,
            "system_prompt_changed": v1.system_prompt != v2.system_prompt,
            "variables_added": list(set(v2.variables) - set(v1.variables)),
            "variables_removed": list(set(v1.variables) - set(v2.variables)),
            "quality_comparison": {
                "v1_quality": v1.avg_quality_score,
                "v2_quality": v2.avg_quality_score,
                "v1_uses": v1.uses,
                "v2_uses": v2.uses,
            },
        }

    def export_template(self, template_id: str) -> dict[str, Any] | None:
        """Export a template to JSON-serializable dict."""
        prompt = self._templates.get(template_id)
        if not prompt:
            return None

        return {
            "id": prompt.id,
            "config": prompt.config.model_dump(),
            "versions": [v.to_dict() for v in prompt.versions],
            "created_at": prompt.created_at.isoformat(),
            "updated_at": prompt.updated_at.isoformat(),
        }

    def import_template(self, data: dict[str, Any]) -> PromptTemplate | None:
        """Import a template from exported data."""
        try:
            config = PromptConfig(**data["config"])

            versions = []
            for v_data in data["versions"]:
                version = PromptVersion(
                    version=v_data["version"],
                    template=v_data["template"],
                    system_prompt=v_data.get("system_prompt"),
                    variables=v_data["variables"],
                    created_at=datetime.fromisoformat(v_data["created_at"]),
                    created_by=v_data.get("created_by", "imported"),
                    commit_message=v_data.get("commit_message", ""),
                    is_active=v_data.get("is_active", False),
                )
                versions.append(version)

            template = PromptTemplate(
                id=data["id"],
                config=config,
                versions=versions,
                created_at=datetime.fromisoformat(data["created_at"]),
                updated_at=datetime.fromisoformat(data["updated_at"]),
            )

            self._templates[template.id] = template
            self._templates[config.name] = template

            return template
        except Exception:
            return None
