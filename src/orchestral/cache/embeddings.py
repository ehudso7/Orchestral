"""
Embedding providers for semantic similarity computation.

Supports multiple embedding backends for flexibility and cost optimization.
"""

from __future__ import annotations

import hashlib
import json
import math
from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    def dimension(self) -> int:
        """Return embedding dimension."""
        ...


class LocalEmbeddings:
    """
    Fast local embeddings using a lightweight approach.

    Uses TF-IDF style hashing for quick semantic similarity without
    external API calls. Good for development and cost-sensitive scenarios.
    """

    def __init__(self, dimension: int = 384):
        self._dimension = dimension
        self._vocab_hash_seed = 42

    @property
    def dimension(self) -> int:
        return self._dimension

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization with n-grams."""
        text = text.lower().strip()
        # Word tokens
        words = text.split()
        # Character 3-grams for robustness
        char_ngrams = [text[i : i + 3] for i in range(len(text) - 2)]
        # Word bigrams
        bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words) - 1)]
        return words + char_ngrams + bigrams

    def _hash_token(self, token: str, seed: int = 0) -> int:
        """Deterministic hash for a token."""
        h = hashlib.md5(f"{seed}:{token}".encode()).hexdigest()
        return int(h, 16)

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using feature hashing (pure Python)."""
        tokens = self._tokenize(text)
        vector = [0.0] * self._dimension

        for token in tokens:
            # Feature hashing with sign trick
            h = self._hash_token(token, self._vocab_hash_seed)
            idx = h % self._dimension
            sign = 1 if (h // self._dimension) % 2 == 0 else -1
            vector[idx] += sign

        # L2 normalize
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]

        return vector

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [await self.embed(text) for text in texts]


class OpenAIEmbeddings:
    """
    OpenAI text-embedding-3-small embeddings.

    High quality embeddings for production use.
    Cost: ~$0.02 per 1M tokens.
    """

    def __init__(self, dimension: int = 1536):
        self._dimension = dimension
        self._client = None

    @property
    def dimension(self) -> int:
        return self._dimension

    async def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI()
        return self._client

    async def embed(self, text: str) -> list[float]:
        """Generate embedding using OpenAI API."""
        client = await self._get_client()
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            dimensions=self._dimension,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        client = await self._get_client()
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            dimensions=self._dimension,
        )
        return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors (pure Python)."""
    if len(a) != len(b):
        return 0.0

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)
