from pathlib import Path

from rag.clients.embedder import TextEmbedder
from rag.clients.llm import LLMClient
from rag.db_manager import DBManager
from rag.ingestion_core import IngestionCore
from rag.models import Chunk, Document
from rag.retrieval_core import RetrievalCore, SearchType

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context does not contain the answer, say so."
)


class RAGCore:
    """Top-level orchestrator tying ingestion, storage, and retrieval together.

    In WE7 this whole class is the *retrieval tool* the agentic verifier calls —
    you do not modify it; you wrap it (see ``agentic.policy_tool``).

    Data flow:
        ingest_*  ->  IngestionCore writes embedded chunks into DBManager (Qdrant)
        _refresh  ->  RetrievalCore is (re)loaded from the store
        receive_query  ->  retrieve -> stuff context -> ask the LLM
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        llm: LLMClient,
        db: DBManager | None = None,
        db_path: str | None = None,
        collection_name: str = "documents",
        # Retrieval defaults
        search_type: SearchType = SearchType.HYBRID,
        top_k: int = 5,
        system_prompt: str = SYSTEM_PROMPT,
        # Ingestion / chunking defaults
        chunk_size: int = 200,
        overlap: int = 40,
        strategy: str = "hierarchical",
    ):
        self.embedder = embedder
        self.llm = llm
        if db is not None and db_path is not None:
            raise ValueError("Pass either `db` or `db_path`, not both.")
        self.db = db if db is not None else DBManager(path=db_path)
        self.collection_name = collection_name

        self.search_type = search_type
        self.system_prompt = system_prompt

        self.ingestion_core = IngestionCore(
            self.embedder,
            self.db,
            collection_name,
            chunk_size=chunk_size,
            overlap=overlap,
            strategy=strategy,
        )
        self.retrieval_core = RetrievalCore(self.embedder, top_k=top_k)
        # Hydrate immediately so a RAGCore over an already-populated store is queryable.
        self._refresh_retrieval()

    # ------------------------------------------------------------------ #
    # Ingestion entry points — each refreshes the retriever afterwards.
    # ------------------------------------------------------------------ #
    def ingest_document(self, document: Document) -> list[Chunk]:
        chunks = self.ingestion_core.ingest_document(document)
        self._refresh_retrieval()
        return chunks

    def ingest_path(self, path: str | Path) -> list[Chunk]:
        """Ingest a single file or a whole directory of ``.txt``/``.md``/``.pdf`` files."""
        path = Path(path)
        if path.is_dir():
            chunks = self.ingestion_core.ingest_directory(path)
        else:
            chunks = self.ingestion_core.ingest_file(path)
        self._refresh_retrieval()
        return chunks

    def _refresh_retrieval(self) -> None:
        """Load the stored chunks back into the in-memory retriever."""
        if not self.db.collection_exists(self.collection_name):
            self.retrieval_core.chunks = []
            return
        self.retrieval_core.chunks = self.db.get_all_chunks(self.collection_name)

    # ------------------------------------------------------------------ #
    # Query
    # ------------------------------------------------------------------ #
    def _retrieve(
        self,
        query: str,
        search_type: SearchType | None,
        metadata_filter: dict | None,
    ) -> list[tuple[Chunk, float]]:
        """Dispatch to the chosen retrieval strategy (default-aware)."""
        search_type = self.search_type if search_type is None else search_type
        if search_type == SearchType.KEYWORD:
            return self.retrieval_core.keyword_search(query, metadata_filter=metadata_filter)
        if search_type == SearchType.EMBEDDING:
            return self.retrieval_core.embedding_search(query, metadata_filter=metadata_filter)
        if search_type == SearchType.HYBRID:
            return self.retrieval_core.hybrid_search(query, metadata_filter=metadata_filter)
        raise ValueError(f"Unknown search type: {search_type}")

    def retrieve(
        self,
        query: str,
        search_type: SearchType | None = None,
        metadata_filter: dict | None = None,
    ) -> list[tuple[Chunk, float]]:
        """Retrieve relevant chunks without calling the LLM.

        This is the entry point the WE6 agentic layer uses: the verifier wants the
        grounding evidence (the policy chunks), not a free-text answer.
        """
        return self._retrieve(query, search_type, metadata_filter)

    def retrieve_and_answer(
        self,
        query: str,
        search_type: SearchType | None = None,
        metadata_filter: dict | None = None,
    ) -> tuple[str, list[tuple[Chunk, float]]]:
        """Retrieve, answer, and return **both** the answer and the chunks used."""
        results = self._retrieve(query, search_type, metadata_filter)
        context = "\n\n".join(chunk.text for chunk, _score in results)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        answer = self.llm.complete(prompt, system_prompt=self.system_prompt)
        return answer, results

    def receive_query(
        self,
        query: str,
        search_type: SearchType | None = None,
        metadata_filter: dict | None = None,
    ) -> str:
        answer, _results = self.retrieve_and_answer(query, search_type, metadata_filter)
        return answer
