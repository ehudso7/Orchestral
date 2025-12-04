"""
Prompt template management with versioning and variable substitution.
"""

from __future__ import annotations

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
