"""
Prompt optimization using various strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class OptimizationStrategy(str, Enum):
    """Strategies for prompt optimization."""

    CLARITY = "clarity"  # Improve clarity and reduce ambiguity
    CONCISENESS = "conciseness"  # Reduce token count while maintaining meaning
    SPECIFICITY = "specificity"  # Add more specific instructions
    STRUCTURE = "structure"  # Improve formatting and organization
    CHAIN_OF_THOUGHT = "chain_of_thought"  # Add reasoning steps
    FEW_SHOT = "few_shot"  # Add examples


@dataclass
class OptimizationResult:
    """Result of prompt optimization."""

    original: str
    optimized: str
    strategy: OptimizationStrategy
    changes: list[str]
    token_reduction: int | None
    confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_length": len(self.original),
            "optimized_length": len(self.optimized),
            "strategy": self.strategy.value,
            "changes": self.changes,
            "token_reduction": self.token_reduction,
            "confidence": f"{self.confidence:.0%}",
            "created_at": self.created_at.isoformat(),
        }


class PromptOptimizer:
    """
    Optimizes prompts using heuristic strategies.

    Provides rule-based optimizations without requiring LLM calls.
    For production use, consider augmenting with LLM-based optimization.

    Example:
        optimizer = PromptOptimizer()

        # Optimize for clarity
        result = optimizer.optimize(
            prompt="do the thing with the code make it better",
            strategy=OptimizationStrategy.CLARITY,
        )
        print(result.optimized)

        # Optimize for conciseness
        result = optimizer.optimize_for_tokens(
            prompt="I would really appreciate it if you could...",
            max_tokens=50,
        )
    """

    # Common filler phrases to remove
    FILLER_PHRASES = [
        "I would like you to",
        "I want you to",
        "Could you please",
        "Would you mind",
        "I would appreciate it if you",
        "It would be great if you could",
        "Please try to",
        "Make sure to",
        "Be sure to",
        "I need you to",
        "I'm asking you to",
        "What I need is",
        "What I want is",
    ]

    # Vague phrases to flag
    VAGUE_PHRASES = [
        "kind of",
        "sort of",
        "maybe",
        "perhaps",
        "possibly",
        "somewhat",
        "a bit",
        "a little",
        "pretty much",
        "more or less",
        "basically",
        "essentially",
        "generally",
    ]

    # Structure markers
    STRUCTURE_MARKERS = {
        "step_markers": ["First,", "Second,", "Third,", "Finally,", "Step 1:", "Step 2:"],
        "section_markers": ["##", "###", "**", "1.", "2.", "3.", "-", "•"],
    }

    def optimize(
        self,
        prompt: str,
        strategy: OptimizationStrategy,
    ) -> OptimizationResult:
        """
        Optimize a prompt using a specific strategy.

        Args:
            prompt: The prompt to optimize
            strategy: Optimization strategy to apply

        Returns:
            OptimizationResult with original and optimized prompts
        """
        if strategy == OptimizationStrategy.CLARITY:
            return self._optimize_clarity(prompt)
        elif strategy == OptimizationStrategy.CONCISENESS:
            return self._optimize_conciseness(prompt)
        elif strategy == OptimizationStrategy.SPECIFICITY:
            return self._optimize_specificity(prompt)
        elif strategy == OptimizationStrategy.STRUCTURE:
            return self._optimize_structure(prompt)
        elif strategy == OptimizationStrategy.CHAIN_OF_THOUGHT:
            return self._add_chain_of_thought(prompt)
        elif strategy == OptimizationStrategy.FEW_SHOT:
            return self._add_few_shot_placeholder(prompt)
        else:
            return OptimizationResult(
                original=prompt,
                optimized=prompt,
                strategy=strategy,
                changes=[],
                token_reduction=0,
                confidence=0.0,
            )

    def _optimize_clarity(self, prompt: str) -> OptimizationResult:
        """Improve prompt clarity."""
        changes = []
        optimized = prompt

        # Remove vague phrases
        for phrase in self.VAGUE_PHRASES:
            if phrase.lower() in optimized.lower():
                optimized = optimized.replace(phrase, "")
                optimized = optimized.replace(phrase.capitalize(), "")
                changes.append(f"Removed vague phrase: '{phrase}'")

        # Clean up extra whitespace
        import re
        optimized = re.sub(r"\s+", " ", optimized).strip()

        # Add explicit task framing if missing
        task_words = ["analyze", "explain", "generate", "write", "create", "summarize", "review", "compare"]
        has_task_word = any(word in optimized.lower() for word in task_words)

        if not has_task_word and len(optimized) < 200:
            # Prompt might be missing clear task definition
            changes.append("Consider adding explicit task verb (analyze, explain, generate, etc.)")

        # Estimate token reduction
        token_reduction = (len(prompt.split()) - len(optimized.split()))

        return OptimizationResult(
            original=prompt,
            optimized=optimized,
            strategy=OptimizationStrategy.CLARITY,
            changes=changes if changes else ["No clarity issues found"],
            token_reduction=token_reduction,
            confidence=0.7 if changes else 0.9,
        )

    def _optimize_conciseness(self, prompt: str) -> OptimizationResult:
        """Reduce prompt verbosity."""
        changes = []
        optimized = prompt

        # Remove filler phrases
        for phrase in self.FILLER_PHRASES:
            if phrase.lower() in optimized.lower():
                # Find and remove while preserving case
                import re
                pattern = re.compile(re.escape(phrase), re.IGNORECASE)
                optimized = pattern.sub("", optimized)
                changes.append(f"Removed filler: '{phrase}'")

        # Remove redundant whitespace
        import re
        optimized = re.sub(r"\s+", " ", optimized).strip()

        # Remove trailing "thanks" or "please help"
        courtesy_endings = [
            "thanks",
            "thank you",
            "thanks in advance",
            "please help",
            "i appreciate your help",
            "let me know if you need more information",
        ]
        for ending in courtesy_endings:
            if optimized.lower().rstrip("!.").endswith(ending):
                optimized = optimized[:optimized.lower().rfind(ending)].rstrip(" .,!")
                changes.append(f"Removed courtesy ending: '{ending}'")

        # Capitalize first letter if needed
        if optimized and optimized[0].islower():
            optimized = optimized[0].upper() + optimized[1:]

        token_reduction = len(prompt.split()) - len(optimized.split())

        return OptimizationResult(
            original=prompt,
            optimized=optimized,
            strategy=OptimizationStrategy.CONCISENESS,
            changes=changes if changes else ["Prompt is already concise"],
            token_reduction=token_reduction,
            confidence=0.8 if changes else 0.95,
        )

    def _optimize_specificity(self, prompt: str) -> OptimizationResult:
        """Add specificity to prompt."""
        changes = []
        suggestions = []

        # Check for missing specificity indicators
        has_format = any(f in prompt.lower() for f in ["format", "structure", "as a", "in the form of"])
        has_length = any(l in prompt.lower() for l in ["words", "sentences", "paragraphs", "lines", "characters"])
        has_audience = any(a in prompt.lower() for a in ["for", "audience", "reader", "user", "beginner", "expert"])
        has_example = any(e in prompt.lower() for e in ["example", "such as", "like", "e.g.", "for instance"])

        if not has_format:
            suggestions.append("Consider specifying output format (JSON, bullet points, paragraph, etc.)")
            changes.append("Missing format specification")

        if not has_length:
            suggestions.append("Consider specifying desired length/detail level")
            changes.append("Missing length specification")

        if not has_audience:
            suggestions.append("Consider specifying target audience")
            changes.append("Missing audience specification")

        if not has_example:
            suggestions.append("Consider adding an example of desired output")
            changes.append("Missing example")

        # Build optimized version with suggestions as comments
        optimized = prompt
        if suggestions:
            optimized = prompt + "\n\n[Suggestions for improvement:\n- " + "\n- ".join(suggestions) + "]"

        return OptimizationResult(
            original=prompt,
            optimized=optimized,
            strategy=OptimizationStrategy.SPECIFICITY,
            changes=changes if changes else ["Prompt has good specificity"],
            token_reduction=None,  # This strategy adds tokens
            confidence=0.6,
        )

    def _optimize_structure(self, prompt: str) -> OptimizationResult:
        """Improve prompt structure."""
        changes = []
        lines = prompt.split("\n")

        # Check if already structured
        has_structure = any(
            any(marker in prompt for marker in markers)
            for markers in self.STRUCTURE_MARKERS.values()
        )

        if has_structure:
            return OptimizationResult(
                original=prompt,
                optimized=prompt,
                strategy=OptimizationStrategy.STRUCTURE,
                changes=["Prompt already has structure"],
                token_reduction=0,
                confidence=0.9,
            )

        # Try to add structure
        optimized_lines = []

        # Split into logical sections
        sentences = []
        current = ""
        for char in prompt:
            current += char
            if char in ".!?" and len(current.strip()) > 10:
                sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())

        if len(sentences) >= 3:
            # Format as numbered list
            optimized_lines.append("Please complete the following:\n")
            for i, sentence in enumerate(sentences, 1):
                optimized_lines.append(f"{i}. {sentence}")
            changes.append("Added numbered structure")
        else:
            # Just clean up
            optimized_lines = [prompt.strip()]

        optimized = "\n".join(optimized_lines)

        return OptimizationResult(
            original=prompt,
            optimized=optimized,
            strategy=OptimizationStrategy.STRUCTURE,
            changes=changes if changes else ["Could not improve structure"],
            token_reduction=len(prompt.split()) - len(optimized.split()),
            confidence=0.6 if changes else 0.4,
        )

    def _add_chain_of_thought(self, prompt: str) -> OptimizationResult:
        """Add chain-of-thought prompting."""
        cot_suffix = "\n\nThink through this step-by-step:\n1. First, understand what is being asked\n2. Break down the problem into parts\n3. Address each part systematically\n4. Synthesize your findings into a final answer"

        optimized = prompt.rstrip() + cot_suffix

        return OptimizationResult(
            original=prompt,
            optimized=optimized,
            strategy=OptimizationStrategy.CHAIN_OF_THOUGHT,
            changes=["Added chain-of-thought prompting"],
            token_reduction=None,  # Adds tokens
            confidence=0.85,
        )

    def _add_few_shot_placeholder(self, prompt: str) -> OptimizationResult:
        """Add few-shot example placeholder."""
        few_shot_template = """
Here are some examples:

Example 1:
Input: [Your example input here]
Output: [Your example output here]

Example 2:
Input: [Your example input here]
Output: [Your example output here]

Now, for the actual task:
"""
        optimized = few_shot_template + prompt

        return OptimizationResult(
            original=prompt,
            optimized=optimized,
            strategy=OptimizationStrategy.FEW_SHOT,
            changes=["Added few-shot example template (fill in examples)"],
            token_reduction=None,
            confidence=0.7,
        )

    def optimize_for_tokens(
        self,
        prompt: str,
        max_tokens: int,
        preserve_meaning: bool = True,
    ) -> OptimizationResult:
        """
        Optimize prompt to fit within token budget.

        Args:
            prompt: The prompt to optimize
            max_tokens: Maximum token count
            preserve_meaning: If True, be more conservative with cuts

        Returns:
            OptimizationResult
        """
        # Rough token estimation (1 token ≈ 4 chars)
        current_tokens = len(prompt) // 4
        changes = []

        if current_tokens <= max_tokens:
            return OptimizationResult(
                original=prompt,
                optimized=prompt,
                strategy=OptimizationStrategy.CONCISENESS,
                changes=["Prompt already within token budget"],
                token_reduction=0,
                confidence=1.0,
            )

        # First, apply conciseness optimization
        result = self._optimize_conciseness(prompt)
        optimized = result.optimized
        changes.extend(result.changes)

        # If still over budget, truncate intelligently
        new_tokens = len(optimized) // 4
        if new_tokens > max_tokens:
            # Keep first part (usually most important)
            target_chars = max_tokens * 4
            if preserve_meaning:
                # Try to break at sentence boundary
                truncated = optimized[:target_chars]
                last_period = truncated.rfind(".")
                if last_period > target_chars * 0.5:
                    truncated = truncated[: last_period + 1]
                optimized = truncated + " [truncated]"
            else:
                optimized = optimized[:target_chars] + "..."

            changes.append(f"Truncated to fit {max_tokens} token budget")

        token_reduction = current_tokens - (len(optimized) // 4)

        return OptimizationResult(
            original=prompt,
            optimized=optimized,
            strategy=OptimizationStrategy.CONCISENESS,
            changes=changes,
            token_reduction=token_reduction,
            confidence=0.7 if "truncated" in changes[-1].lower() else 0.85,
        )

    def analyze(self, prompt: str) -> dict[str, Any]:
        """
        Analyze a prompt and provide improvement suggestions.

        Returns comprehensive analysis without modifying the prompt.
        """
        word_count = len(prompt.split())
        char_count = len(prompt)
        estimated_tokens = char_count // 4

        # Check various quality indicators
        has_clear_task = any(
            word in prompt.lower()
            for word in ["analyze", "explain", "generate", "write", "create", "summarize", "compare", "review", "list"]
        )

        has_context = len(prompt) > 100
        has_specificity = any(
            indicator in prompt.lower()
            for indicator in ["format", "length", "style", "tone", "audience", "example"]
        )
        has_structure = any(
            marker in prompt
            for markers in self.STRUCTURE_MARKERS.values()
            for marker in markers
        )

        vague_count = sum(1 for phrase in self.VAGUE_PHRASES if phrase.lower() in prompt.lower())
        filler_count = sum(1 for phrase in self.FILLER_PHRASES if phrase.lower() in prompt.lower())

        # Calculate quality score
        quality_score = 0.5
        if has_clear_task:
            quality_score += 0.15
        if has_context:
            quality_score += 0.1
        if has_specificity:
            quality_score += 0.15
        if has_structure:
            quality_score += 0.1
        quality_score -= vague_count * 0.05
        quality_score -= filler_count * 0.05
        quality_score = max(0.0, min(1.0, quality_score))

        # Generate suggestions
        suggestions = []
        if not has_clear_task:
            suggestions.append("Add a clear task verb (analyze, explain, generate, etc.)")
        if not has_specificity:
            suggestions.append("Add specificity (format, length, audience)")
        if not has_structure and word_count > 50:
            suggestions.append("Consider adding structure (numbered steps, sections)")
        if vague_count > 0:
            suggestions.append(f"Remove {vague_count} vague phrase(s)")
        if filler_count > 0:
            suggestions.append(f"Remove {filler_count} filler phrase(s)")

        return {
            "word_count": word_count,
            "char_count": char_count,
            "estimated_tokens": estimated_tokens,
            "quality_score": round(quality_score, 2),
            "has_clear_task": has_clear_task,
            "has_context": has_context,
            "has_specificity": has_specificity,
            "has_structure": has_structure,
            "vague_phrase_count": vague_count,
            "filler_phrase_count": filler_count,
            "suggestions": suggestions,
            "recommended_strategies": self._recommend_strategies(
                has_clear_task, has_specificity, has_structure, vague_count, filler_count
            ),
        }

    def _recommend_strategies(
        self,
        has_clear_task: bool,
        has_specificity: bool,
        has_structure: bool,
        vague_count: int,
        filler_count: int,
    ) -> list[str]:
        """Recommend optimization strategies based on analysis."""
        strategies = []

        if vague_count > 0 or not has_clear_task:
            strategies.append(OptimizationStrategy.CLARITY.value)

        if filler_count > 0:
            strategies.append(OptimizationStrategy.CONCISENESS.value)

        if not has_specificity:
            strategies.append(OptimizationStrategy.SPECIFICITY.value)

        if not has_structure:
            strategies.append(OptimizationStrategy.STRUCTURE.value)

        return strategies if strategies else ["No optimization needed"]
