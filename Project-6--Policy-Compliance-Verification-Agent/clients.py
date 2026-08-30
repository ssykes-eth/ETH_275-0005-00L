"""Convenience factory for the AI clients (a thin shim over ``rag.clients``).

The embedder and LLM live in the vendored RAG package (``rag/clients``), since RAG
already depends on them. WE7 reuses the *same* clients — the agents talk to the LLM
through the very object the RAG tool uses — so this module just builds the pair from
one API key.
"""

from __future__ import annotations

from rag.clients.embedder import TextEmbedder
from rag.clients.llm import LLMClient


def build_clients(api_key: str) -> tuple[TextEmbedder, LLMClient]:
    """Return an ``(embedder, llm)`` pair built from a single OpenRouter key."""
    return TextEmbedder(api_key=api_key), LLMClient(api_key=api_key)
