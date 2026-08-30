"""Exported from the WE6 notebook. Do not edit by hand; re-export instead."""
from __future__ import annotations
from agentic.models import RetrievedChunk


def retrieve_policies(self, context):
    """Retrieve policy passages for a VerificationContext.
    Arguments:
        context (VerificationContext): The context to retrieve policies for.
    Returns:
        A list of RetrievedChunk objects.
    """

    results = self.rag.retrieve(
        # 🎯 the query string from the context
        context.query, 
        # hybrid by default
        search_type=self.search_type, 
        # 🎯 the metadata filter from the context 
        metadata_filter=context.metadata_filter, 
    )
    return [
        RetrievedChunk(
            # 🎯 the source of the chunk (or "unknown" if not present)
            source=chunk.metadata.get("source", "unknown"),
            # 🎯 the index of the chunk in the source document
            chunk_id=chunk.index, 
            # 🎯 the document title of the chunk; or its source if no title is present; or "unknown" if neither is present
            document_title=chunk.metadata.get("title", chunk.metadata.get("source", "unknown")), 
            # 🎯 the text of the chunk
            text=chunk.text, 
            # 🎯 the closeness of the chunk to the query
            score=score, 
        )
        for chunk, score in results
    ]
