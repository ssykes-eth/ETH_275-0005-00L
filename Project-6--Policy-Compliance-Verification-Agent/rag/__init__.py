"""Vendored RAG core — a frozen copy of the WE6 reference implementation.

In WE7 the RAG pipeline is no longer the thing you build; it is a *tool* that the
agentic verifier calls. This package is a self-contained copy of the WE6 reference
implementation (ingestion, chunking, embeddings, metadata, hybrid search, the
grounded query) so WE7 runs standalone — no sibling checkout, no submodule.

If you completed WE6, you can run retrieval on *your own* code: drop your exported
``part1_ingestion.py`` / ``part2_retrieval.py`` into ``rag/solutions/`` and they are
monkey-patched onto these modules at startup (see ``rag.solutions.apply``). A fresh
checkout simply runs on the reference implementation.
"""
