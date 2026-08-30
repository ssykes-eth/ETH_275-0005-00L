# Defense Walkthrough — Compliance Q&A RAG Project

A 15-minute talking-points document: background, scope, what was built, and the theory behind
each design decision. For exhaustive cell-by-cell detail see `DOCUMENTATION.md` in this folder.

---

## 1. Background (≈2 min)

- Course project for **FDD-WE5** (`eth-fdd-fs26/FDD-WE5-public`). The task: build a
  **Retrieval-Augmented Generation (RAG)** system that answers questions about company compliance
  policies using only the provided document set (`data/`), not the LLM's own memory.
- **Why RAG, not just an LLM:** an LLM asked to recite policy from memory can hallucinate a
  plausible-sounding but wrong answer. RAG turns this into an "open-book exam" — retrieve the
  actual relevant passages first, then force the model to answer *only* from them.
- **Why this is graded the way it is:** the repository ships with the entire system built —
  loading, storage, orchestration, the web app — except **eight functions**, which raise
  `NotImplementedError` on a fresh checkout. There is no reference implementation anywhere in the
  public repo. The project *is* implementing those eight functions correctly enough that the
  system's own offline test suite and evaluation harness turn green.
- **Delivery mechanism:** implement each function inside the Jupyter notebook
  (`compliance_rag_project.ipynb`), which immediately **monkey-patches** it onto the live project
  classes (e.g. `chunking.sliding_window = sliding_window`), so the same code that passes the
  notebook's own tests is what powers `main.py` and the FastAPI/React web app. Export cells then
  write the finished functions to `solutions/part1_ingestion.py` and `solutions/part2_retrieval.py`.

## 2. Scope and goal (≈2 min)

The system has two halves that meet at an in-memory "simple vector database":

| Side | Stage | Responsibility |
|---|---|---|
| **Write** (ingestion) | `load → chunk → embed → store` | Turn raw `.txt`/`.md`/`.pdf` policy docs into small, labelled, embedded, persisted chunks |
| **Read** (retrieval + answer) | `search → fuse → rerank → answer` | Take a question, find the best-matching chunks, hand them to the LLM, or admit "I don't know" |

**Explicit non-goals**, by design of the course:
- Qdrant is used purely for **persistence** — all ranking math (BM25, cosine, RRF) runs on plain
  Python/NumPy over an in-memory list, precisely so the algorithms are visible and teachable rather
  than hidden inside a database's internals.
- No production-grade evaluation: the gold set is 12 hand-written questions, enough to catch a
  regression or show a trend, not to certify the system.
- Offline-first: every exercise and every test must pass **without** an API key, using
  deterministic test doubles (`MockEmbedder`, `HashingEmbedder`, `MockLLM`).

**Goal for the defense:** show that all eight functions are correctly implemented, and that each
implementation choice is backed by a specific, statable reason — not just "it passed the test."

## 3. The eight applied changes (≈7 min — the core of the defense)

Grouped by pipeline stage. For each: what was implemented, and the theoretical reason behind the
specific approach (not just what the code does — *why this and not the naive alternative*).

### Part 1 — Ingestion (write side)

**1. `sliding_window(text, chunk_size, overlap)`**
Splits text into overlapping word windows (`step = chunk_size - overlap`), rejecting
`overlap >= chunk_size` since that would make the window never advance (infinite loop).
- *Theory:* embedding an entire document as one vector blends every topic in it into a single
  blurry point in vector space, so a query about one specific clause competes with noise from the
  whole document. Chunking gives each idea its own embedding. **Overlap** exists because a sentence
  that straddles a hard chunk boundary would otherwise be split and lost to both chunks; a small
  overlap guarantees any answer-bearing sentence sits *whole* inside at least one chunk.

**2. `ingest_document(document)`**
Runs chunk → wrap in `Chunk` objects (with a **copied**, not shared, metadata dict) → batch-embed
→ persist, sizing the Qdrant collection from `len(embeddings[0])`.
- *Theory:* batching the embedding call (`embed_batch`, one request for all chunks) instead of one
  request per chunk is a throughput/cost argument — network round-trips dominate latency, not
  compute. Copying `document.metadata` per chunk avoids aliasing: mutating one chunk's metadata
  later must never silently mutate every sibling chunk's metadata (shared-reference bug class).

**3. `apply_metadata_filter(chunks, metadata_filter)`**
AND-matches every key in the filter dict against each chunk's metadata (`.get`, not `[]`, so a
missing key excludes rather than crashes); returns the *same* list object, unchanged, when there is
no filter.
- *Theory:* this is a **hard boolean gate that runs before scoring**, not a ranking step — it can
  narrow the candidate pool but can never rescue a badly-phrased query. Returning the identical
  list object (not a copy) when unfiltered matters operationally: `RetrievalCore`'s BM25 index is
  cached keyed on object identity of `self.chunks`, so returning a "look-alike" copy would silently
  defeat that cache on every unfiltered call.

### Part 2 — Retrieval and answering (read side)

**4. `cosine_similarity(query_vec, matrix)`**
Vectorised: `matrix @ query_vec` for the dot products, `np.linalg.norm(matrix, axis=1)` for the
per-row norms, safe division via `np.divide(..., where=denom != 0)` to avoid `NaN` on a zero
vector.
- *Theory:* cosine similarity measures the **angle** between two vectors, not their magnitude —
  which is the correct notion of "similar meaning" for embeddings, since embedding length is not a
  semantic signal. Vectorising is not a style preference: a Python-level loop over every chunk
  would be the dominant cost of a query at any real corpus size, and NumPy already provides this as
  a single BLAS call.

**5. `keyword_search(query, top_k, metadata_filter)`**
Tokenizes the query, filters candidates, builds/reuses a cached BM25 index (`self._bm25_index`),
scores, and returns the top-k.
- *Theory:* BM25 is a decades-old, well-understood lexical scoring function that rewards *exact*,
  *rare* shared words over common ones — it has no notion of meaning, so "time off" will not match
  "annual leave." The index is cached (not rebuilt per call) because BM25 needs whole-corpus term
  statistics (e.g. inverse document frequency) up front, and rebuilding that on every single query —
  twice per hybrid-search call — would be wasteful; the cache is only valid for the *unfiltered*
  corpus since a metadata filter changes the term statistics.

**6. `reciprocal_rank_fusion(ranked_lists, top_k, k=60)`**
Sums `1 / (k + rank)` per chunk across every input ranked list, sorts by fused score, truncates to
`top_k`.
- *Theory:* BM25 scores and cosine similarities live on **incompatible numeric scales** (unbounded
  vs. `[-1, 1]`) — averaging or summing them directly would be numerically meaningless. RRF
  sidesteps this entirely by discarding scores and using only **rank position**, which is
  comparable across any two ranking methods. The constant `k=60` dampens the gap between
  neighbouring ranks so one list's single top pick cannot unilaterally dominate a chunk that both
  lists rank respectably.

**7. `hybrid_search(query, top_k, metadata_filter, overretrieve)`**
Runs keyword and embedding search each with `top_k=pool` where `pool = top_k * overretrieve`
(same filter on both branches), then fuses and cuts back to `top_k`.
- *Theory:* fusion can only **reorder what it is handed** — a chunk ranked 8th in one list and 6th
  in another can never surface if each branch is only asked for its top 5. Over-retrieving widens
  the pool so fusion has real material to promote from. Both branches must receive the identical
  filter, or an unfiltered branch becomes a "side door" that leaks excluded documents back into the
  results. `overretrieve=1` intentionally reproduces the old narrow behaviour for comparison.
  *(Measured in §2.4/§2.5 of the notebook: recall rises from 0.75 at `overretrieve=1` to 0.83,
  plateauing beyond ~5×, which is what justifies the default of 20 rather than an arbitrarily large
  number.)*

**8. `retrieve_and_answer(query, search_type, metadata_filter)`**
Retrieves; if nothing survives (`_retrieve` applies `min_score`), returns a fixed
`NO_CONTEXT_ANSWER` **without calling the LLM**; otherwise concatenates chunk texts into a context
block, prompts the LLM with `Context:… Question:…`, and returns the answer plus the source chunks
used.
- *Theory:* this is the **abstention** design, arguably the single highest-stakes decision in the
  whole system. Retrieval always returns *some* nearest chunks, even when nothing in the corpus is
  actually relevant. A capable, "helpful" LLM handed marginally-related leftover context will often
  still produce a fluent, confident-sounding, and **wrong** answer — which is strictly worse than an
  honest "I don't know," especially in a compliance setting where a wrong answer has real
  consequences. Checking `min_score` and short-circuiting before the LLM call is what prevents this
  failure mode. Returning the source chunks alongside the answer is what lets the web frontend show
  citations, so a user can verify the answer against the actual policy text.

## 4. How correctness was verified (≈2 min)

- Every exercise has its own `TestSuite` cell directly below it in the notebook (project's own
  lightweight harness, not `pytest`); all eight pass with the final implementations.
- `uv run python -m tests` runs the same eight functions' checks plus `test_db_manager` and
  `test_evaluation` (which are implementation-independent) fully offline — no API key, no network,
  an in-memory Qdrant.
- Beyond unit correctness, §2.5 of the notebook **measures** design decisions rather than asserting
  them: a 12-question gold set (6 lexical, 6 semantic) scored on document recall@k, answer-passage
  recall@k, and context cost. This is what surfaces non-obvious results worth mentioning in a
  defense, e.g.:
  - hybrid search wins on document recall, but keyword search alone briefly wins on *answer*
    recall — traced to the offline `HashingEmbedder` (a word-overlap stand-in, not a real semantic
    model) weakening the embedding branch, confirmed by re-running with a real embedding model
    (semantic score 0.33 → 1.00, hybrid answer recall 0.42 → 0.92).
  - an **un-chunked** ("whole document") baseline actually wins on both recall metrics — but at
    ~10× the context cost — which is precisely why context cost is tracked as a third metric
    alongside recall; recall alone cannot see this tradeoff.

## 5. One incidental fix, not an exercise (≈1 min, if time allows)

Both export cells originally called `_path.write_text(_body)` with no explicit encoding. On
Windows this defaults to `cp1252`, which cannot encode the em dashes (`—`) present in the exported
functions' own source comments, causing a `UnicodeEncodeError` on export. Fixed by passing
`encoding="utf-8"` explicitly — a portability fix, not a change to any of the eight graded
functions. Flagged separately since the notebook is generated upstream and this fix does not
survive a regeneration.

## 6. Anticipated questions

- **"Why RRF instead of just averaging normalized scores?"** — normalization (e.g. min-max) is
  fragile to outliers and still assumes the two distributions are comparable after rescaling; RRF
  needs no distributional assumptions at all, only rank order.
- **"Why not always over-retrieve maximally (very high `overretrieve`)?"** — measured plateau past
  ~5×; beyond that it only adds compute cost (scoring more candidates) for no recall gain, and once
  the pool exceeds the corpus size it degrades to "everything," which is measured explicitly in the
  hybrid_search tests.
- **"Why is abstention a fixed threshold (`min_score`) instead of asking the LLM to judge
  relevance itself?"** — a threshold is cheap, deterministic, and auditable; asking the LLM adds
  another call (cost/latency) and another place a confident-sounding wrong judgment could occur.
  The notebook also demonstrates the threshold's own fragility (a nonsense question phrased as a
  full sentence can still score above threshold on BM25 with no stopword removal) — worth
  mentioning as a known limitation, not something the implementation glosses over.
