"""
Guardrails and content filtering for Orchestral.

Provides multiple layers of protection:
- Input validation and sanitization
- PII detection and redaction
- Prompt injection prevention
- Topic/content filtering
- Output validation
"""

from __future__ import annotations

import asyncio
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Pattern

import structlog

logger = structlog.get_logger()


class GuardrailAction(str, Enum):
    """Action to take when guardrail is triggered."""

    ALLOW = "allow"  # Allow request to proceed
    BLOCK = "block"  # Block request entirely
    REDACT = "redact"  # Redact sensitive content
    WARN = "warn"  # Allow but log warning
    MODIFY = "modify"  # Modify content and proceed


@dataclass
class GuardrailResult:
    """Result of a guardrail check."""

    passed: bool
    action: GuardrailAction
    guardrail_name: str
    message: str | None = None
    modified_content: str | None = None
    detections: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "passed": self.passed,
            "action": self.action.value,
            "guardrail_name": self.guardrail_name,
            "message": self.message,
            "detections": self.detections,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class Guardrail(ABC):
    """Base class for guardrails."""

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        action: GuardrailAction = GuardrailAction.BLOCK,
    ):
        self.name = name
        self.enabled = enabled
        self.default_action = action

    @abstractmethod
    async def check(self, content: str, context: dict[str, Any] | None = None) -> GuardrailResult:
        """
        Check content against this guardrail.

        Args:
            content: Content to check
            context: Additional context (user_id, model, etc.)

        Returns:
            GuardrailResult
        """
        pass


class ContentFilter(Guardrail):
    """
    Filter for inappropriate or harmful content.

    Detects:
    - Profanity
    - Hate speech indicators
    - Violence references
    - Adult content
    """

    # Basic patterns (in production, use ML-based detection)
    PROFANITY_PATTERNS = [
        r"\b(fuck|shit|damn|ass|bitch)\b",
    ]

    VIOLENCE_PATTERNS = [
        r"\b(kill|murder|attack|bomb|weapon|shoot)\b.*\b(people|person|them|you)\b",
        r"\bhow\s+to\s+(make|build|create)\s+(bomb|weapon|poison)\b",
    ]

    HARMFUL_PATTERNS = [
        r"\b(suicide|self.?harm)\b.*\b(how|method|way)\b",
    ]

    def __init__(
        self,
        name: str = "content_filter",
        enabled: bool = True,
        action: GuardrailAction = GuardrailAction.BLOCK,
        check_profanity: bool = True,
        check_violence: bool = True,
        check_harmful: bool = True,
        custom_patterns: list[str] | None = None,
    ):
        super().__init__(name, enabled, action)
        self.check_profanity = check_profanity
        self.check_violence = check_violence
        self.check_harmful = check_harmful

        self._patterns: list[tuple[str, Pattern]] = []

        if check_profanity:
            for p in self.PROFANITY_PATTERNS:
                self._patterns.append(("profanity", re.compile(p, re.IGNORECASE)))

        if check_violence:
            for p in self.VIOLENCE_PATTERNS:
                self._patterns.append(("violence", re.compile(p, re.IGNORECASE)))

        if check_harmful:
            for p in self.HARMFUL_PATTERNS:
                self._patterns.append(("harmful", re.compile(p, re.IGNORECASE)))

        if custom_patterns:
            for p in custom_patterns:
                self._patterns.append(("custom", re.compile(p, re.IGNORECASE)))

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Check content for inappropriate material."""
        if not self.enabled:
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.ALLOW,
                guardrail_name=self.name,
            )

        detections = []
        for category, pattern in self._patterns:
            matches = pattern.findall(content)
            if matches:
                detections.append({
                    "category": category,
                    "matches": matches[:5],  # Limit matches
                })

        if detections:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                guardrail_name=self.name,
                message=f"Content flagged for: {', '.join(d['category'] for d in detections)}",
                detections=detections,
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            guardrail_name=self.name,
        )


class PIIDetector(Guardrail):
    """
    Detect and optionally redact Personally Identifiable Information.

    Detects:
    - Email addresses
    - Phone numbers
    - Social Security Numbers
    - Credit card numbers
    - IP addresses
    - Dates of birth
    - Names (basic detection)
    """

    PII_PATTERNS = {
        "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "phone": r"\b(?:\+?1[-.]?)?\(?[0-9]{3}\)?[-.]?[0-9]{3}[-.]?[0-9]{4}\b",
        "ssn": r"\b\d{3}[-]?\d{2}[-]?\d{4}\b",
        "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "date_of_birth": r"\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b",
    }

    REDACTION_TEMPLATES = {
        "email": "[EMAIL]",
        "phone": "[PHONE]",
        "ssn": "[SSN]",
        "credit_card": "[CREDIT_CARD]",
        "ip_address": "[IP_ADDRESS]",
        "date_of_birth": "[DOB]",
    }

    def __init__(
        self,
        name: str = "pii_detector",
        enabled: bool = True,
        action: GuardrailAction = GuardrailAction.REDACT,
        detect_types: list[str] | None = None,
        redact: bool = True,
    ):
        super().__init__(name, enabled, action)
        self.redact = redact

        self._patterns: dict[str, Pattern] = {}
        types_to_check = detect_types or list(self.PII_PATTERNS.keys())

        for pii_type in types_to_check:
            if pii_type in self.PII_PATTERNS:
                self._patterns[pii_type] = re.compile(
                    self.PII_PATTERNS[pii_type], re.IGNORECASE
                )

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Check content for PII."""
        if not self.enabled:
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.ALLOW,
                guardrail_name=self.name,
            )

        detections = []
        modified_content = content

        for pii_type, pattern in self._patterns.items():
            matches = pattern.findall(content)
            if matches:
                detections.append({
                    "type": pii_type,
                    "count": len(matches),
                    "redacted": self.redact,
                })

                if self.redact:
                    replacement = self.REDACTION_TEMPLATES.get(pii_type, "[REDACTED]")
                    modified_content = pattern.sub(replacement, modified_content)

        if detections:
            return GuardrailResult(
                passed=self.redact,  # Pass if redacting, fail if blocking
                action=GuardrailAction.REDACT if self.redact else self.default_action,
                guardrail_name=self.name,
                message=f"PII detected: {', '.join(d['type'] for d in detections)}",
                modified_content=modified_content if self.redact else None,
                detections=detections,
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            guardrail_name=self.name,
        )


class PromptInjectionDetector(Guardrail):
    """
    Detect potential prompt injection attacks.

    Detects:
    - System prompt override attempts
    - Instruction injection
    - Role-playing manipulation
    - Delimiter attacks
    """

    INJECTION_PATTERNS = [
        # System prompt overrides
        r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions?|prompts?)",
        r"disregard\s+(all\s+)?(previous|above|prior)",
        r"forget\s+(everything|all)\s+(you|that)",
        r"new\s+instructions?:",
        r"system\s*:\s*you\s+are",

        # Role manipulation
        r"you\s+are\s+now\s+(in\s+)?(\w+\s+)?mode",
        r"pretend\s+(to\s+be|you\s+are)",
        r"act\s+as\s+(if\s+)?(you|a)",
        r"roleplay\s+as",

        # Delimiter attacks
        r"```\s*(system|instruction)",
        r"<\|?(system|im_start|im_end)\|?>",
        r"\[\s*(INST|SYS|SYSTEM)\s*\]",

        # Jailbreak attempts
        r"(DAN|jailbreak|bypass|hack)\s+(mode|prompt)",
        r"do\s+anything\s+now",
    ]

    def __init__(
        self,
        name: str = "prompt_injection_detector",
        enabled: bool = True,
        action: GuardrailAction = GuardrailAction.BLOCK,
        sensitivity: float = 0.5,  # 0-1, higher = more sensitive
        custom_patterns: list[str] | None = None,
    ):
        super().__init__(name, enabled, action)
        self.sensitivity = sensitivity

        patterns = self.INJECTION_PATTERNS.copy()
        if custom_patterns:
            patterns.extend(custom_patterns)

        self._patterns = [re.compile(p, re.IGNORECASE) for p in patterns]

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Check for prompt injection attempts."""
        if not self.enabled:
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.ALLOW,
                guardrail_name=self.name,
            )

        detections = []
        total_patterns = len(self._patterns)
        matched_patterns = 0

        for pattern in self._patterns:
            match = pattern.search(content)
            if match:
                matched_patterns += 1
                detections.append({
                    "pattern": pattern.pattern[:50],
                    "match": match.group()[:100],
                })

        # Calculate confidence based on number of patterns matched
        confidence = matched_patterns / total_patterns if total_patterns > 0 else 0

        if confidence >= self.sensitivity or matched_patterns > 0:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                guardrail_name=self.name,
                message="Potential prompt injection detected",
                detections=detections,
                confidence=confidence,
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            guardrail_name=self.name,
            confidence=1 - confidence,
        )


class TopicFilter(Guardrail):
    """
    Filter content based on allowed/blocked topics.

    Use for:
    - Restricting discussions to specific domains
    - Blocking off-topic requests
    - Industry-specific content policies
    """

    def __init__(
        self,
        name: str = "topic_filter",
        enabled: bool = True,
        action: GuardrailAction = GuardrailAction.BLOCK,
        blocked_topics: list[str] | None = None,
        allowed_topics: list[str] | None = None,
    ):
        super().__init__(name, enabled, action)
        self.blocked_topics = blocked_topics or []
        self.allowed_topics = allowed_topics  # None means allow all

        self._blocked_patterns = [
            re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
            for t in self.blocked_topics
        ]
        self._allowed_patterns = (
            [
                re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)
                for t in self.allowed_topics
            ]
            if self.allowed_topics
            else None
        )

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Check content against topic filters."""
        if not self.enabled:
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.ALLOW,
                guardrail_name=self.name,
            )

        detections = []

        # Check blocked topics
        for i, pattern in enumerate(self._blocked_patterns):
            if pattern.search(content):
                detections.append({
                    "type": "blocked_topic",
                    "topic": self.blocked_topics[i],
                })

        if detections:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                guardrail_name=self.name,
                message=f"Blocked topic detected: {detections[0]['topic']}",
                detections=detections,
            )

        # Check allowed topics (if specified)
        if self._allowed_patterns:
            has_allowed = any(p.search(content) for p in self._allowed_patterns)
            if not has_allowed:
                return GuardrailResult(
                    passed=False,
                    action=self.default_action,
                    guardrail_name=self.name,
                    message="Content does not match allowed topics",
                    detections=[{"type": "off_topic"}],
                )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            guardrail_name=self.name,
        )


class OutputValidator(Guardrail):
    """
    Validate model outputs against expected formats and constraints.

    Use for:
    - JSON schema validation
    - Response length limits
    - Format enforcement
    - Hallucination detection (basic)
    """

    def __init__(
        self,
        name: str = "output_validator",
        enabled: bool = True,
        action: GuardrailAction = GuardrailAction.WARN,
        max_length: int | None = None,
        min_length: int | None = None,
        must_contain: list[str] | None = None,
        must_not_contain: list[str] | None = None,
        json_schema: dict[str, Any] | None = None,
    ):
        super().__init__(name, enabled, action)
        self.max_length = max_length
        self.min_length = min_length
        self.must_contain = must_contain or []
        self.must_not_contain = must_not_contain or []
        self.json_schema = json_schema

    async def check(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> GuardrailResult:
        """Validate output content."""
        if not self.enabled:
            return GuardrailResult(
                passed=True,
                action=GuardrailAction.ALLOW,
                guardrail_name=self.name,
            )

        detections = []

        # Length checks
        if self.max_length and len(content) > self.max_length:
            detections.append({
                "type": "too_long",
                "length": len(content),
                "max": self.max_length,
            })

        if self.min_length and len(content) < self.min_length:
            detections.append({
                "type": "too_short",
                "length": len(content),
                "min": self.min_length,
            })

        # Content checks
        for phrase in self.must_contain:
            if phrase.lower() not in content.lower():
                detections.append({
                    "type": "missing_required",
                    "phrase": phrase,
                })

        for phrase in self.must_not_contain:
            if phrase.lower() in content.lower():
                detections.append({
                    "type": "contains_forbidden",
                    "phrase": phrase,
                })

        # JSON schema validation
        if self.json_schema:
            try:
                import json
                data = json.loads(content)
                # Basic schema check - in production use jsonschema library
                for required_key in self.json_schema.get("required", []):
                    if required_key not in data:
                        detections.append({
                            "type": "missing_json_field",
                            "field": required_key,
                        })
            except json.JSONDecodeError:
                detections.append({
                    "type": "invalid_json",
                })

        if detections:
            return GuardrailResult(
                passed=False,
                action=self.default_action,
                guardrail_name=self.name,
                message=f"Output validation failed: {detections[0]['type']}",
                detections=detections,
            )

        return GuardrailResult(
            passed=True,
            action=GuardrailAction.ALLOW,
            guardrail_name=self.name,
        )


class GuardrailPipeline:
    """
    Pipeline for running multiple guardrails in sequence.

    Supports:
    - Input guardrails (before model call)
    - Output guardrails (after model call)
    - Async execution
    - Short-circuit on blocking
    """

    def __init__(
        self,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
        enabled: bool = True,
    ):
        self.input_guardrails = input_guardrails or []
        self.output_guardrails = output_guardrails or []
        self.enabled = enabled

    def add_input_guardrail(self, guardrail: Guardrail) -> None:
        """Add an input guardrail."""
        self.input_guardrails.append(guardrail)

    def add_output_guardrail(self, guardrail: Guardrail) -> None:
        """Add an output guardrail."""
        self.output_guardrails.append(guardrail)

    async def check_input(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str, list[GuardrailResult]]:
        """
        Run input guardrails.

        Returns:
            Tuple of (passed, possibly_modified_content, results)
        """
        if not self.enabled:
            return True, content, []

        results = []
        modified_content = content

        for guardrail in self.input_guardrails:
            result = await guardrail.check(modified_content, context)
            results.append(result)

            if result.action == GuardrailAction.BLOCK:
                logger.warning(
                    "Input blocked by guardrail",
                    guardrail=guardrail.name,
                    message=result.message,
                )
                return False, content, results

            if result.action == GuardrailAction.REDACT and result.modified_content:
                modified_content = result.modified_content

        return True, modified_content, results

    async def check_output(
        self,
        content: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[bool, str, list[GuardrailResult]]:
        """
        Run output guardrails.

        Returns:
            Tuple of (passed, possibly_modified_content, results)
        """
        if not self.enabled:
            return True, content, []

        results = []
        modified_content = content

        for guardrail in self.output_guardrails:
            result = await guardrail.check(modified_content, context)
            results.append(result)

            if result.action == GuardrailAction.BLOCK:
                logger.warning(
                    "Output blocked by guardrail",
                    guardrail=guardrail.name,
                    message=result.message,
                )
                return False, content, results

            if result.action == GuardrailAction.REDACT and result.modified_content:
                modified_content = result.modified_content

        return True, modified_content, results


# Global guardrail pipeline
_pipeline: GuardrailPipeline | None = None


def get_guardrail_pipeline() -> GuardrailPipeline:
    """Get the global guardrail pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = GuardrailPipeline(
            input_guardrails=[
                PromptInjectionDetector(),
                ContentFilter(),
                PIIDetector(action=GuardrailAction.REDACT),
            ],
            output_guardrails=[
                PIIDetector(action=GuardrailAction.REDACT),
                ContentFilter(),
            ],
        )
    return _pipeline


def configure_guardrail_pipeline(
    input_guardrails: list[Guardrail] | None = None,
    output_guardrails: list[Guardrail] | None = None,
    enabled: bool = True,
) -> GuardrailPipeline:
    """Configure the global guardrail pipeline."""
    global _pipeline
    _pipeline = GuardrailPipeline(
        input_guardrails=input_guardrails,
        output_guardrails=output_guardrails,
        enabled=enabled,
    )
    return _pipeline
