"""Drop your WE6 Part 1 export here (chunking / ingestion / metadata).

Until you do, these stubs stay marked ``@todo`` and are skipped — the vendored
reference RAG keeps working. Replace the file wholesale with your WE6 export.
"""

from __future__ import annotations

from rag.solutions import todo


@todo
def sliding_window(text, chunk_size=200, overlap=40):
    raise NotImplementedError


@todo
def ingest_document(self, document):
    raise NotImplementedError


@todo
def derive_title(text, stem):
    raise NotImplementedError
