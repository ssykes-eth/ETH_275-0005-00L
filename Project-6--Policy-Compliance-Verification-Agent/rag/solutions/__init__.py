"""Optional drop-zone for your WE6 RAG solutions.

If you finished WE6, you exported two files there:

    part1_ingestion.py   (chunking, ingestion pipeline, metadata)
    part2_retrieval.py   (cosine, keyword, RRF, the LLM context step)

Drop them into *this* folder and ``apply()`` monkey-patches them onto the vendored
``rag.*`` modules, so the policy-retrieval tool runs on *your* RAG code. This is a
no-op on a fresh checkout (the shipped stubs are marked ``@todo`` and skipped), so
WE7 always runs on the frozen reference implementation out of the box.

This mirrors the WE6 ``solutions`` loader; only the patch *targets* changed to the
``rag.`` package namespace.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable


def todo(fn: Callable) -> Callable:
    """Mark a shipped stub as not-yet-implemented so ``apply()`` skips it."""
    fn._is_todo = True  # type: ignore[attr-defined]
    return fn


@dataclass(frozen=True)
class PatchSpec:
    part_attr: str
    target_module: str
    target_path: str
    kind: str  # "function" | "method" | "static"


# Targets live in the vendored rag.* package (the only change vs. WE6).
PART1_SPECS: list[PatchSpec] = [
    PatchSpec("sliding_window", "rag.chunking", "sliding_window", "function"),
    PatchSpec("ingest_document", "rag.ingestion_core", "IngestionCore.ingest_document", "method"),
    PatchSpec("derive_title", "rag.loaders", "_derive_title", "function"),
]

PART2_SPECS: list[PatchSpec] = [
    PatchSpec("cosine_similarity", "rag.retrieval_core", "RetrievalCore._cosine_similarity", "static"),
    PatchSpec("keyword_search", "rag.retrieval_core", "RetrievalCore.keyword_search", "method"),
    PatchSpec("reciprocal_rank_fusion", "rag.retrieval_core", "RetrievalCore._reciprocal_rank_fusion", "static"),
    PatchSpec("retrieve_and_answer", "rag.rag_core", "RAGCore.retrieve_and_answer", "method"),
]


def _bind(spec: PatchSpec, fn: Callable) -> None:
    module = importlib.import_module(spec.target_module)
    parts = spec.target_path.split(".")
    owner = module
    for name in parts[:-1]:
        owner = getattr(owner, name)
    value = staticmethod(fn) if spec.kind == "static" else fn
    setattr(owner, parts[-1], value)


def _apply_specs(part_module_name: str, specs: list[PatchSpec]) -> list[str]:
    try:
        part = importlib.import_module(part_module_name)
    except ModuleNotFoundError:
        return []
    patched: list[str] = []
    for spec in specs:
        fn = getattr(part, spec.part_attr, None)
        if fn is None or getattr(fn, "_is_todo", False):
            continue
        _bind(spec, fn)
        patched.append(f"{spec.target_module}.{spec.target_path}")
    return patched


def apply(verbose: bool = True) -> list[str]:
    """Patch your WE6 functions onto the vendored RAG, where provided."""
    patched = _apply_specs("rag.solutions.part1_ingestion", PART1_SPECS)
    patched += _apply_specs("rag.solutions.part2_retrieval", PART2_SPECS)
    if verbose:
        if patched:
            print(f"[rag.solutions] running retrieval on your WE6 code: {', '.join(patched)}")
        else:
            print("[rag.solutions] no WE6 solutions found — using the vendored reference RAG.")
    return patched
