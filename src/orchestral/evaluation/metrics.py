"""
Response quality metrics computation.

Provides heuristic-based metrics for evaluating response quality
without requiring additional LLM calls.
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass
from typing import Any


@dataclass
class ResponseMetrics:
    """Computed metrics for a response."""

    # Basic metrics
    word_count: int
    sentence_count: int
    avg_sentence_length: float
    unique_word_ratio: float

    # Readability
    flesch_reading_ease: float
    flesch_kincaid_grade: float

    # Structure
    has_formatting: bool
    code_block_count: int
    list_item_count: int
    header_count: int

    # Quality signals
    coherence_score: float
    repetition_score: float
    specificity_score: float

    # Safety signals
    uncertainty_phrases: int
    hedging_phrases: int
    potential_hallucination_signals: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "avg_sentence_length": self.avg_sentence_length,
            "unique_word_ratio": round(self.unique_word_ratio, 3),
            "flesch_reading_ease": round(self.flesch_reading_ease, 1),
            "flesch_kincaid_grade": round(self.flesch_kincaid_grade, 1),
            "has_formatting": self.has_formatting,
            "code_block_count": self.code_block_count,
            "list_item_count": self.list_item_count,
            "header_count": self.header_count,
            "coherence_score": round(self.coherence_score, 3),
            "repetition_score": round(self.repetition_score, 3),
            "specificity_score": round(self.specificity_score, 3),
            "uncertainty_phrases": self.uncertainty_phrases,
            "hedging_phrases": self.hedging_phrases,
            "potential_hallucination_signals": self.potential_hallucination_signals,
        }


def count_syllables(word: str) -> int:
    """Estimate syllable count for a word."""
    word = word.lower().strip()
    if len(word) <= 3:
        return 1

    # Remove trailing e
    if word.endswith("e"):
        word = word[:-1]

    # Count vowel groups
    vowels = "aeiouy"
    count = 0
    prev_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel

    return max(1, count)


def compute_readability(text: str) -> tuple[float, float]:
    """
    Compute Flesch readability scores.

    Returns:
        Tuple of (Flesch Reading Ease, Flesch-Kincaid Grade Level)
    """
    # Clean and split
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    if not sentences or not words:
        return 0.0, 0.0

    total_sentences = len(sentences)
    total_words = len(words)
    total_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease
    # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    fre = (
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words)
    )
    fre = max(0, min(100, fre))

    # Flesch-Kincaid Grade Level
    # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
    fkgl = (
        0.39 * (total_words / total_sentences)
        + 11.8 * (total_syllables / total_words)
        - 15.59
    )
    fkgl = max(0, fkgl)

    return fre, fkgl


def compute_coherence(text: str) -> float:
    """
    Compute coherence score based on linguistic features.

    Higher scores indicate more coherent text.
    Returns value between 0 and 1.
    """
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) < 2:
        return 0.5  # Neutral for single sentence

    # Transition words indicate coherence
    transition_words = {
        "however",
        "therefore",
        "furthermore",
        "moreover",
        "additionally",
        "consequently",
        "thus",
        "hence",
        "meanwhile",
        "nevertheless",
        "nonetheless",
        "similarly",
        "likewise",
        "conversely",
        "alternatively",
        "specifically",
        "particularly",
        "notably",
        "importantly",
        "finally",
        "firstly",
        "secondly",
        "lastly",
        "next",
        "then",
        "also",
        "besides",
        "indeed",
        "certainly",
    }

    # Reference words that connect sentences
    reference_words = {
        "this",
        "that",
        "these",
        "those",
        "it",
        "they",
        "them",
        "such",
        "the",
    }

    words = text.lower().split()
    word_set = set(words)

    transition_count = len(word_set & transition_words)
    reference_count = sum(1 for w in words if w in reference_words)

    # Score components
    transition_score = min(1.0, transition_count / (len(sentences) * 0.5))
    reference_score = min(1.0, reference_count / (len(sentences) * 2))

    # Sentence length variance (too much variance = less coherent)
    lengths = [len(s.split()) for s in sentences]
    if len(lengths) > 1:
        mean_len = sum(lengths) / len(lengths)
        variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        cv = (variance**0.5) / mean_len if mean_len > 0 else 1
        length_consistency = max(0, 1 - cv * 0.5)
    else:
        length_consistency = 0.5

    return (transition_score * 0.3 + reference_score * 0.3 + length_consistency * 0.4)


def compute_repetition(text: str) -> float:
    """
    Detect repetition in text.

    Returns score from 0 (no repetition) to 1 (highly repetitive).
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(words) < 10:
        return 0.0

    # N-gram repetition
    trigrams = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0

    unique_trigrams = len(set(trigrams))
    trigram_repetition = 1 - (unique_trigrams / len(trigrams))

    # Word repetition beyond expected
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    # Exclude common words
    common_words = {"the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "in", "for", "on", "with"}
    content_words = [w for w in words if w not in common_words and len(w) > 2]

    if content_words:
        unique_content = len(set(content_words))
        content_repetition = 1 - (unique_content / len(content_words))
    else:
        content_repetition = 0.0

    return (trigram_repetition * 0.6 + content_repetition * 0.4)


def compute_specificity(text: str) -> float:
    """
    Measure specificity/concreteness of text.

    Higher scores indicate more specific, detailed responses.
    """
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if not words:
        return 0.0

    # Vague/generic words
    vague_words = {
        "thing",
        "things",
        "stuff",
        "something",
        "anything",
        "everything",
        "someone",
        "anyone",
        "everyone",
        "somehow",
        "somewhat",
        "somewhere",
        "very",
        "really",
        "quite",
        "rather",
        "pretty",
        "kind",
        "sort",
        "basically",
        "generally",
        "usually",
        "often",
        "sometimes",
        "maybe",
        "perhaps",
        "probably",
        "possibly",
        "might",
        "could",
        "would",
        "should",
        "etc",
        "etcetera",
    }

    # Specific indicators
    has_numbers = bool(re.search(r"\d+", text))
    has_quotes = bool(re.search(r'["\'].*?["\']', text))
    has_proper_nouns = bool(re.search(r"\b[A-Z][a-z]+\b", text))

    vague_count = sum(1 for w in words if w in vague_words)
    vague_ratio = vague_count / len(words)

    # Average word length (longer words tend to be more specific)
    avg_word_len = sum(len(w) for w in words) / len(words)
    length_score = min(1.0, (avg_word_len - 3) / 4)

    specificity = (
        (1 - vague_ratio) * 0.4
        + length_score * 0.2
        + (0.15 if has_numbers else 0)
        + (0.1 if has_quotes else 0)
        + (0.15 if has_proper_nouns else 0)
    )

    return min(1.0, max(0.0, specificity))


def detect_hallucination_signals(text: str) -> tuple[int, int, int]:
    """
    Detect signals that might indicate hallucination or uncertainty.

    Returns:
        Tuple of (uncertainty_phrases, hedging_phrases, hallucination_signals)
    """
    text_lower = text.lower()

    # Uncertainty phrases
    uncertainty_patterns = [
        r"\bi think\b",
        r"\bi believe\b",
        r"\bi'm not sure\b",
        r"\bi don't know\b",
        r"\bpossibly\b",
        r"\bperhaps\b",
        r"\bmaybe\b",
        r"\bmight be\b",
        r"\bcould be\b",
        r"\bprobably\b",
        r"\blikely\b",
        r"\bunlikely\b",
        r"\buncertain\b",
        r"\bunclear\b",
    ]

    # Hedging phrases
    hedging_patterns = [
        r"\bto some extent\b",
        r"\bin a way\b",
        r"\bkind of\b",
        r"\bsort of\b",
        r"\bmore or less\b",
        r"\bup to a point\b",
        r"\bas far as i know\b",
        r"\bfrom what i understand\b",
        r"\bit seems\b",
        r"\bit appears\b",
        r"\bgenerally speaking\b",
        r"\bin general\b",
        r"\btypically\b",
        r"\busually\b",
    ]

    # Potential hallucination signals
    hallucination_patterns = [
        r"\bas of my (last |knowledge )?(?:update|training|cutoff)\b",
        r"\bi don't have (access to |real-?time |current )\b",
        r"\bi cannot (browse|search|access)\b",
        r"\bmy training data\b",
        r"\bi was trained\b",
        r"\bas an ai\b",
        r"\bas a language model\b",
        r"\baccording to my knowledge\b",
        r"\bto the best of my knowledge\b",
        r"\bi apologize.{0,30}(cannot|can't|unable)\b",
    ]

    uncertainty = sum(1 for p in uncertainty_patterns if re.search(p, text_lower))
    hedging = sum(1 for p in hedging_patterns if re.search(p, text_lower))
    hallucination = sum(1 for p in hallucination_patterns if re.search(p, text_lower))

    return uncertainty, hedging, hallucination


def compute_response_metrics(text: str) -> ResponseMetrics:
    """Compute all metrics for a response."""
    # Basic stats
    words = re.findall(r"\b[a-zA-Z]+\b", text)
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    word_count = len(words)
    sentence_count = max(1, len(sentences))
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0

    unique_words = set(w.lower() for w in words)
    unique_word_ratio = len(unique_words) / word_count if word_count > 0 else 0

    # Readability
    fre, fkgl = compute_readability(text)

    # Structure
    has_formatting = bool(
        re.search(r"```|\*\*|##|__|`[^`]+`", text)
    )
    code_block_count = len(re.findall(r"```[\s\S]*?```", text))
    list_item_count = len(re.findall(r"^\s*[-*•]\s|\d+\.\s", text, re.MULTILINE))
    header_count = len(re.findall(r"^#{1,6}\s", text, re.MULTILINE))

    # Quality signals
    coherence = compute_coherence(text)
    repetition = compute_repetition(text)
    specificity = compute_specificity(text)

    # Safety signals
    uncertainty, hedging, hallucination = detect_hallucination_signals(text)

    return ResponseMetrics(
        word_count=word_count,
        sentence_count=sentence_count,
        avg_sentence_length=avg_sentence_length,
        unique_word_ratio=unique_word_ratio,
        flesch_reading_ease=fre,
        flesch_kincaid_grade=fkgl,
        has_formatting=has_formatting,
        code_block_count=code_block_count,
        list_item_count=list_item_count,
        header_count=header_count,
        coherence_score=coherence,
        repetition_score=repetition,
        specificity_score=specificity,
        uncertainty_phrases=uncertainty,
        hedging_phrases=hedging,
        potential_hallucination_signals=hallucination,
    )
