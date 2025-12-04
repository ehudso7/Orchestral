"""
Task classification for intelligent routing.

Automatically detects task type from prompts to enable optimal model selection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    """Detected task types."""

    CODING = "coding"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYSIS = "analysis"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    CONVERSATION = "conversation"
    MATH = "math"
    RESEARCH = "research"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    GENERAL = "general"


@dataclass
class ClassificationResult:
    """Result of task classification."""

    primary_type: TaskType
    confidence: float
    secondary_types: list[tuple[TaskType, float]]
    detected_features: list[str]
    complexity: str  # "simple", "moderate", "complex"
    estimated_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_type": self.primary_type.value,
            "confidence": f"{self.confidence:.0%}",
            "secondary_types": [
                {"type": t.value, "confidence": f"{c:.0%}"}
                for t, c in self.secondary_types
            ],
            "detected_features": self.detected_features,
            "complexity": self.complexity,
            "estimated_tokens": self.estimated_tokens,
        }


class TaskClassifier:
    """
    Classifies tasks based on prompt content.

    Uses keyword matching and heuristics for fast, local classification.
    No API calls required.

    Example:
        classifier = TaskClassifier()
        result = classifier.classify("Write a Python function to sort a list")
        print(result.primary_type)  # TaskType.CODING
    """

    # Task type indicators
    TASK_PATTERNS = {
        TaskType.CODING: {
            "keywords": [
                "code", "function", "class", "method", "implement", "debug",
                "fix", "bug", "error", "exception", "syntax", "compile",
                "programming", "script", "algorithm", "api", "endpoint",
                "database", "query", "sql", "html", "css", "javascript",
                "python", "java", "typescript", "rust", "go", "c++",
                "refactor", "optimize", "test", "unit test", "integration",
            ],
            "patterns": [
                r"```\w*\n",  # Code blocks
                r"def\s+\w+",  # Python function
                r"function\s+\w+",  # JS function
                r"class\s+\w+",  # Class definition
                r"\w+\(\)",  # Function calls
                r"import\s+\w+",  # Imports
                r"from\s+\w+\s+import",  # Python imports
            ],
            "weight": 1.2,
        },
        TaskType.REASONING: {
            "keywords": [
                "why", "explain", "reason", "because", "therefore", "logic",
                "deduce", "infer", "conclude", "argument", "premise",
                "hypothesis", "proof", "evidence", "justify", "rationale",
                "think through", "step by step", "logical", "reasoning",
            ],
            "patterns": [
                r"why\s+(is|are|do|does|did|would|should)",
                r"explain\s+(why|how|what)",
                r"what\s+(is|are)\s+the\s+reason",
            ],
            "weight": 1.1,
        },
        TaskType.CREATIVE: {
            "keywords": [
                "write", "story", "poem", "creative", "imagine", "fiction",
                "narrative", "character", "plot", "dialogue", "scene",
                "novel", "essay", "blog", "article", "content", "copy",
                "slogan", "tagline", "brainstorm", "ideas", "creative",
            ],
            "patterns": [
                r"write\s+(a|an)\s+(story|poem|essay|article)",
                r"create\s+(a|an)\s+(story|narrative|character)",
                r"imagine\s+",
            ],
            "weight": 1.0,
        },
        TaskType.ANALYSIS: {
            "keywords": [
                "analyze", "analysis", "evaluate", "assess", "examine",
                "investigate", "study", "review", "critique", "compare",
                "contrast", "pros", "cons", "advantages", "disadvantages",
                "strengths", "weaknesses", "swot", "breakdown", "insights",
            ],
            "patterns": [
                r"analyze\s+(this|the|these)",
                r"what\s+are\s+the\s+(pros|cons|advantages|disadvantages)",
                r"compare\s+(and\s+)?contrast",
            ],
            "weight": 1.0,
        },
        TaskType.SUMMARIZATION: {
            "keywords": [
                "summarize", "summary", "tldr", "brief", "overview",
                "key points", "main points", "highlights", "condensed",
                "short version", "in brief", "recap", "digest",
            ],
            "patterns": [
                r"summarize\s+(this|the|these)",
                r"give\s+(me\s+)?a\s+summary",
                r"what\s+are\s+the\s+(key|main)\s+points",
            ],
            "weight": 1.0,
        },
        TaskType.TRANSLATION: {
            "keywords": [
                "translate", "translation", "convert", "from english",
                "to english", "in spanish", "in french", "in german",
                "in chinese", "in japanese", "language", "localize",
            ],
            "patterns": [
                r"translate\s+(this|the|these|from|to|into)",
                r"(from|to|into)\s+(english|spanish|french|german|chinese|japanese)",
            ],
            "weight": 1.3,
        },
        TaskType.MATH: {
            "keywords": [
                "calculate", "compute", "solve", "equation", "formula",
                "math", "mathematical", "arithmetic", "algebra", "calculus",
                "statistics", "probability", "geometry", "number", "sum",
                "product", "derivative", "integral", "matrix", "vector",
            ],
            "patterns": [
                r"\d+\s*[\+\-\*\/\^]\s*\d+",  # Basic math
                r"solve\s+(for|the)",
                r"calculate\s+(the)?",
                r"\$?\d+[\.,]?\d*",  # Numbers/money
            ],
            "weight": 1.1,
        },
        TaskType.RESEARCH: {
            "keywords": [
                "research", "find", "search", "look up", "information",
                "data", "facts", "sources", "references", "citations",
                "literature", "study", "papers", "articles", "learn about",
            ],
            "patterns": [
                r"(find|search|look\s+up)\s+(information|data|facts)",
                r"what\s+(is|are|do\s+you\s+know\s+about)",
                r"tell\s+me\s+about",
            ],
            "weight": 0.9,
        },
        TaskType.EXTRACTION: {
            "keywords": [
                "extract", "pull", "get", "retrieve", "parse", "scrape",
                "find all", "list all", "identify", "locate", "names",
                "dates", "numbers", "entities", "ner", "regex",
            ],
            "patterns": [
                r"extract\s+(all|the)",
                r"(find|list|get)\s+all",
                r"parse\s+(this|the)",
            ],
            "weight": 1.0,
        },
        TaskType.CLASSIFICATION: {
            "keywords": [
                "classify", "categorize", "label", "tag", "sort",
                "group", "bucket", "sentiment", "positive", "negative",
                "spam", "not spam", "topic", "category",
            ],
            "patterns": [
                r"classify\s+(this|the|these)",
                r"(is\s+this|what\s+type)",
                r"categorize\s+",
            ],
            "weight": 1.0,
        },
        TaskType.CONVERSATION: {
            "keywords": [
                "chat", "talk", "hello", "hi", "hey", "how are you",
                "what's up", "thanks", "thank you", "please", "help",
                "can you", "could you", "would you",
            ],
            "patterns": [
                r"^(hi|hello|hey|thanks|thank\s+you)",
                r"how\s+are\s+you",
                r"^(can|could|would)\s+you",
            ],
            "weight": 0.8,
        },
    }

    def __init__(self):
        # Precompile regex patterns
        self._compiled_patterns: dict[TaskType, list[re.Pattern]] = {}
        for task_type, config in self.TASK_PATTERNS.items():
            self._compiled_patterns[task_type] = [
                re.compile(p, re.IGNORECASE) for p in config.get("patterns", [])
            ]

    def _count_keywords(self, text: str, keywords: list[str]) -> int:
        """Count keyword occurrences in text."""
        text_lower = text.lower()
        count = 0
        for keyword in keywords:
            if keyword.lower() in text_lower:
                count += 1
        return count

    def _check_patterns(self, text: str, task_type: TaskType) -> int:
        """Check regex pattern matches."""
        patterns = self._compiled_patterns.get(task_type, [])
        return sum(1 for p in patterns if p.search(text))

    def _estimate_complexity(self, text: str) -> str:
        """Estimate task complexity."""
        word_count = len(text.split())
        sentence_count = len(re.split(r"[.!?]+", text))
        has_code = bool(re.search(r"```", text))
        has_multiple_questions = text.count("?") > 1

        complexity_score = 0

        if word_count > 200:
            complexity_score += 2
        elif word_count > 50:
            complexity_score += 1

        if sentence_count > 5:
            complexity_score += 1

        if has_code:
            complexity_score += 2

        if has_multiple_questions:
            complexity_score += 1

        if complexity_score >= 4:
            return "complex"
        elif complexity_score >= 2:
            return "moderate"
        else:
            return "simple"

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Rough estimate: 1 token ≈ 4 characters
        return len(text) // 4

    def classify(self, prompt: str) -> ClassificationResult:
        """
        Classify a prompt into task types.

        Args:
            prompt: The prompt to classify

        Returns:
            ClassificationResult with primary and secondary types
        """
        scores: dict[TaskType, float] = {}
        features: list[str] = []

        for task_type, config in self.TASK_PATTERNS.items():
            keywords = config.get("keywords", [])
            weight = config.get("weight", 1.0)

            keyword_count = self._count_keywords(prompt, keywords)
            pattern_count = self._check_patterns(prompt, task_type)

            # Calculate score
            score = (keyword_count * 0.3 + pattern_count * 0.7) * weight
            scores[task_type] = score

            # Track detected features
            if keyword_count > 0:
                features.append(f"{task_type.value}_keywords:{keyword_count}")
            if pattern_count > 0:
                features.append(f"{task_type.value}_patterns:{pattern_count}")

        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            for task_type in scores:
                scores[task_type] /= total_score
        else:
            # Default to general if no signals
            scores[TaskType.GENERAL] = 1.0

        # Sort by score
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        primary_type = sorted_scores[0][0]
        primary_confidence = sorted_scores[0][1]

        # Get secondary types (above threshold)
        secondary_types = [
            (t, s) for t, s in sorted_scores[1:5] if s > 0.1
        ]

        return ClassificationResult(
            primary_type=primary_type,
            confidence=primary_confidence,
            secondary_types=secondary_types,
            detected_features=features,
            complexity=self._estimate_complexity(prompt),
            estimated_tokens=self._estimate_tokens(prompt),
        )

    def classify_batch(self, prompts: list[str]) -> list[ClassificationResult]:
        """Classify multiple prompts."""
        return [self.classify(prompt) for prompt in prompts]
