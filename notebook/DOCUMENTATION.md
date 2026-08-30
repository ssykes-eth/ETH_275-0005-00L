# Documentation: `compliance_rag_project.ipynb`

This file explains, in plain language, what every cell in the notebook
`notebook/compliance_rag_project.ipynb` does, and — in full, step-by-step detail — exactly what was
written to fill in each of the eight `TODO` exercises (plus one small bug fix that was needed to run
the notebook on Windows).

It's written so that someone with very little programming background can follow along: every
technical term is explained the first time it shows up, usually with an everyday comparison.

---

## Part A — What is this notebook, really?

The notebook builds a **RAG system** — "RAG" stands for **R**etrieval-**A**ugmented **G**eneration.
In plain words: it's a program that answers questions about a pile of documents (here, company
policies) by first *finding* the paragraphs that are relevant, and then handing those paragraphs to
an AI language model (an **LLM** — "Large Language Model", like the engine behind ChatGPT or
Claude) and asking it to answer *using only that information*. Think of it like an open-book exam:
instead of asking the AI to recite company policy from memory (which it might get wrong or make up),
you first hand it the exact pages of the policy book that are relevant, then ask the question.

The notebook doesn't ask you to build this whole system from scratch. Almost everything —
loading files, storing data, talking to the AI — is already written for you in the surrounding
project (files like `chunking.py`, `retrieval_core.py`, `rag_core.py`, etc.). Your job is to fill in
**8 specific functions**, each one a small but essential piece of the puzzle. Every exercise gives you
the function already "shaped" — the loops and structure are there — with the important pieces of
logic blanked out as a marker called `TODO`. Filling in a `TODO` is like filling in the blank in a
recipe card that already tells you the steps and just needs you to say *how much sugar* or *how long
to bake*.

Once you write a function in the notebook, a special trick called **monkey-patching** immediately
makes it "the real" version everywhere in the project. Monkey-patching means: at run time, you take an
existing piece of software and swap in your own version of one of its parts, without editing the
original file. In code it looks like `chunking.sliding_window = sliding_window` — this line says
"from now on, whenever any part of the program calls `chunking.sliding_window`, use *my*
function instead of the empty placeholder that shipped in the file." That's how your notebook code
ends up powering the whole pipeline (and even a real website, in Part 3) without you ever opening
another file.

### The two halves of a RAG system

1. **Write side (ingestion, built in Part 1):** turn raw documents into small, searchable pieces
   stored in a database.
2. **Read side (retrieval + answering, built in Part 2):** take a question, find the best-matching
   pieces, and ask the LLM to answer using them.

---

## Part B — A glossary of terms used throughout

Read this once; the walkthrough below will make more sense with these in hand.

- **Document** — one whole source file (e.g. one policy `.md` file), loaded into memory along with
  some **metadata** (see below).
- **Chunk** — a smaller piece cut out of a document (e.g. one paragraph-sized passage). RAG systems
  work on chunks, not whole documents, because whole documents are too big and vague to search well
  (more on this in §1.1 below).
- **Metadata** — extra facts *about* a document or chunk, stored alongside its text — like a label on
  a filing folder. Here that's things like which file it came from (`source`) and its title
  (`document_title`).
- **Embedding** — a way of turning a piece of text into a list of numbers (called a **vector**) such
  that texts with *similar meaning* end up as *numerically similar* lists. Imagine every sentence
  becomes an arrow pointing in some direction in space; sentences about similar topics point in
  similar directions. A model trained for this job produces the embedding; two arrows pointing the
  same way mean "these two texts mean roughly the same thing," even if they don't share a single
  word.
- **Vector** — just a list of numbers, e.g. `[0.12, -0.87, 0.33, ...]`. Here, an embedding vector.
- **Vector database / vector store** — a place to store chunks along with their embedding vectors, so
  you can later ask "which stored vectors are closest to this new vector?" This project uses
  **Qdrant** as that storage. Crucially, per this project's design, Qdrant is *only* used to save the
  data — all the actual searching/ranking logic you write runs on data loaded into plain Python lists
  in memory (deliberately, so the ranking math is visible and teachable, not hidden inside the
  database).
- **Cosine similarity** — a way to measure how similar two vectors (arrows) are, by comparing the
  *angle* between them and ignoring how long each arrow is. Two arrows pointing in exactly the same
  direction score `1.0` (identical meaning); two pointing in totally unrelated directions score `0`;
  two pointing in opposite directions score `-1`.
- **Tokenize** — to break a piece of text down into individual words (and lowercase them), e.g.
  `"Annual Leave!"` becomes `["annual", "leave"]`. This is the first step keyword search needs.
- **BM25** — a decades-old, very well-tested formula for scoring "how relevant is this chunk of text
  to this search query," based on *exact* shared words, rewarding rare/unusual shared words more than
  common ones (the word "MFA" matching is a much bigger signal than the word "the" matching, for
  example). This is "keyword search" or "lexical search" in the notebook.
- **Embedding search / semantic search** — searching by comparing embedding vectors (cosine
  similarity) instead of matching exact words. Its superpower is understanding that "time off" and
  "annual leave" mean similar things even though they share no words.
- **Hybrid search** — running *both* keyword search and embedding search, then intelligently merging
  ("fusing") their two rankings into one combined ranking, so a question only needs to be answerable
  by *one* of the two approaches to succeed.
- **Reciprocal Rank Fusion (RRF)** — the specific merging trick used here. Instead of trying to
  compare BM25 scores and cosine scores directly (which live on completely different numeric scales
  and can't be added meaningfully), RRF only looks at each chunk's *position* (1st place, 2nd place,
  …) in each ranked list, and rewards chunks that rank well in either or both lists.
- **top_k** — "give me the best `k` results," e.g. `top_k=3` means "give me the 3 best-scoring
  chunks."
- **Over-retrieval** — asking a search method for *more* candidates than you ultimately want to keep,
  so that a later merging/fusing step has a wide enough pool of options to pick the true best answers
  from.
- **Context / context block** — the block of retrieved chunk text that gets pasted into the prompt
  sent to the LLM, so it has the information it needs to answer. Think of it as "the pages from the
  policy book you photocopied and handed to the AI before asking your question."
- **Abstention** — the RAG system deliberately saying "I don't know" instead of forcing an answer, when
  nothing relevant was found. This matters because an LLM handed irrelevant information will often
  still produce a confident-sounding — but wrong — answer, which is worse than admitting ignorance.
- **min_score** — a numeric threshold: if the best chunk found scores below this number, the system
  decides "nothing relevant enough was found" and abstains instead of answering.
- **Recall@k** — an evaluation metric: out of a set of test questions, for what fraction of them does
  the *correct* source document show up somewhere in the top `k` search results?
- **Answer recall@k** — a stricter evaluation metric: for what fraction of test questions does the
  *exact sentence* that answers the question literally appear in the retrieved text (not just "the
  right document showed up somewhere")?
- **Context cost** — how many words of text get handed to the LLM to produce an answer. More words
  means more money, more time, and (for LLMs) often worse focus — so a good system doesn't just want
  high recall, it wants high recall *cheaply*.

---

## Part C — How the notebook is organized

The notebook has **119 cells** in total (numbered 0–118 below, 0-indexed, matching their order top
to bottom), split into four parts:

- **Part 0 (cells 0–16): Setup and scaffolding.** Intro, roadmap, cloning the project, imports, and
  a handful of "run and forget" helper cells (display helpers, the `TODO` marker, evaluation
  helpers). No exercises here.
- **Part 1 (cells 17–46): Ingestion — the write side.** Turning documents into stored, searchable,
  embedded chunks. Contains exercises 1–3 (`sliding_window`, `ingest_document`,
  `apply_metadata_filter`).
- **Part 2 (cells 47–115): The RAG core — the read side.** Building and measuring the search/answer
  pipeline. Contains exercises 4–8 (`cosine_similarity`, `keyword_search`, `reciprocal_rank_fusion`,
  `hybrid_search`, `retrieve_and_answer`), plus a substantial measurement/evaluation section (§2.5)
  with no exercises — only analysis.
- **Part 3 (cells 116–118): The web app (optional) and a scratch appendix.** Instructions for
  running the project as an actual website, plus a free space to write your own experiments.

Every exercise in Parts 1 and 2 follows the *same four-cell pattern*:

1. A **"✏️ Your task"** cell — a nicely formatted instruction card explaining exactly what the
   function must do, what tools are already provided to use, and what it must return.
2. The **exercise cell itself** — the function definition, with `TODO` blanks to fill in.
3. A **patch cell** — one line that monkey-patches your function onto the real project code (e.g.
   `chunking.sliding_window = sliding_window`).
4. A **test cell** — a small `TestSuite` (the project's own lightweight test runner, not `pytest`)
   that calls your function with a handful of example inputs and checks the outputs are correct.

---

## Part D — Cell-by-cell walkthrough

### Part 0 — Setup and scaffolding (cells 0–16)

| # | Type | What it does |
|---|------|--------------|
| 0 | markdown | Title and welcome. Explains the monkey-patching model, that the 8 functions are empty in the repo on purpose, lays out the Part 1 / Part 2 / Part 3 roadmap, and previews the `TODO` blank mechanism. |
| 1 | code (form) | Draws an SVG diagram of the *whole* system: the write path (Documents → Chunk → Embed → Vector store) on top, the read path (Question → Hybrid search → LLM → Answer) below. |
| 2 | markdown | Explains the diagram: the two paths meet at the vector store — ingestion writes into it, queries read from it. |
| 3 | markdown | §0.1 Setup — explains what the next cell does, in Colab and locally, and that the project's working directory is `project/`. |
| 4 | code (form) | §0.1 Setup cell. In Google Colab, clones the public course repo and installs dependencies. Run locally, it instead walks *up* the folder tree from wherever you launched Jupyter until it finds `chunking.py` (the project root), adds that folder to Python's import path, and changes into it — so it works whether you opened the notebook from `project/` or `project/notebook/`. |
| 5 | markdown | §0.2 — explains that an OpenRouter API key is *optional*; everything in the notebook works fully offline with stand-in ("mock") AI clients. |
| 6 | code | Loads a local `.env` file (if one exists) into environment variables — this is how a local run can supply an API key without pasting it into the notebook. |
| 7 | code | Pulls the key from Colab's secret-storage system if running in Colab; sets a `HAS_KEY` flag used later to decide whether "live" demo cells run. |
| 8 | markdown | §0.3 Imports — notes that the notebook reuses the project's *own* test harness (`TestSuite`) and offline stand-in AI clients, the same tools the course staff use. |
| 9 | code | The big import cell: brings in `chunking`, `loaders`, `IngestionCore`, `DBManager`, `RAGCore`, `RetrievalCore`, the `Chunk`/`Document` data models, the `evaluation` module, and the offline test doubles `HashingEmbedder` / `MockEmbedder` / `MockLLM`. |
| 10 | markdown | Explains the `TODO` blank mechanism — a blank is a *live* placeholder object, not just a comment, so touching an unfilled blank crashes immediately and loudly rather than silently giving a wrong answer. |
| 11 | code (form) | Defines the `_Blank` class (whose single instance is called `TODO`) and a helper `has_blanks(fn)` that checks whether a function's source code still contains the text `TODO` (used later by the export cells). |
| 12 | markdown | §0.4 — introduces the display helpers as "run once, don't read." |
| 13 | code (form) | Defines the HTML-rendering helper functions used throughout the notebook to show pretty result cards: `show_documents`, `show_chunks`, `show_chunking`, `show_results`, `show_answer`. |
| 14 | markdown | §0.5 — introduces the evaluation-specific display helpers. |
| 15 | code (form) | Defines more display helpers used only in the measurement section: `explain` (a callout box), `show_compare` (side-by-side ranked result panels), `show_eval_grid`, `show_scoreboard`, `show_kinds`, `show_sweep`. |
| 16 | markdown | Section header for **Part 1**, laying out the ingestion pipeline stages: `load → CHUNK → EMBED → store`. |

### Part 1 — Document ingestion pipeline (cells 17–46)

| # | Type | What it does |
|---|------|--------------|
| 17 | code (form) | SVG diagram of the ingestion pipeline, marking which stages ("CHUNK", "EMBED") are the ones you implement. |
| 18 | markdown | One-line transition: "let's look at the documents we'll work with." |
| 19 | code (form) | Loads every file in `data/` into `Document` objects (`loaders.load_directory`) and displays them in a table (title, source file, word count). |
| 20 | markdown | **§1.1 Chunking.** Explains *why* chunking is necessary: embedding an entire 800-word policy as one vector blurs every idea in it together, so a specific question competes with noise from the whole document. The fix is a **sliding window** with a little **overlap** so sentences that straddle a boundary aren't lost. |
| 21 | code (form) | SVG diagram of a concrete sliding-window example: 9 words (`w0`–`w8`), `chunk_size=4`, `overlap=1`, showing exactly how three overlapping chunks are produced. |
| 22 | markdown | Points out that `w3` and `w6` each sit in two chunks (that's the overlap in action), and poses a reflection question: what happens if `overlap` is large relative to `chunk_size`, or zero? |
| 23 | code (form) | The **"✏️ Your task · 1.1"** instruction card for `sliding_window`. |
| 24 | code | **Exercise 1** — the `sliding_window` function body. See [Part E, §1](#1-sliding_window--11-splitting-text-into-overlapping-pieces) below for the full explanation of what was filled in. |
| 25 | code | Monkey-patches: `chunking.sliding_window = sliding_window`. |
| 26 | code | `TestSuite("Part 1.1 — sliding_window")` — 4 checks (overlap shares words across chunks, a short text becomes one chunk, empty text gives no chunks, `overlap >= chunk_size` is rejected). |
| 27 | markdown | Sets up a "see why chunking matters" comparison. |
| 28 | code (form) | Runs `chunk_text` on a real policy document with `strategy="whole_document"` (no chunking) vs `strategy="sliding_window"`, and shows both side by side. |
| 29 | markdown | **§1.2 Embeddings & the simple vector database.** Explains what an embedding is, and lays out the four things `ingest_document` must do (chunk, wrap in `Chunk` objects, embed in one batched call, store). |
| 30 | code (form) | The **"✏️ Your task · 1.2"** instruction card for `ingest_document`. |
| 31 | code | **Exercise 2** — the `ingest_document` function body. See [Part E, §2](#2-ingest_document--12-turning-one-document-into-stored-searchable-chunks) below. |
| 32 | code | Monkey-patches: `IngestionCore.ingest_document = ingest_document`. |
| 33 | code | `TestSuite("Part 1.2 — ingest_document")` — 4 checks (every chunk comes back embedded, chunk indices are sequential, chunks are persisted and re-readable, document metadata rides onto every chunk). |
| 34 | markdown | **§1.3 Metadata.** Explains what metadata is (`source`, `document_title`) and that it's what lets a later search be *narrowed* to specific documents. |
| 35 | code (form) | Shows the source code of `loaders._derive_title` (how a document gets its title) and prints `metadata_schema.REQUIRED_FIELDS`; explains why the project validates every chunk's metadata *at write time* rather than letting a missing field degrade search results silently later. |
| 36 | markdown | Introduces `apply_metadata_filter`: every search strategy in Part 2 calls it *first*, before scoring, to narrow the field of candidates. A filter dict's key/value pairs must **all** match (AND, not OR); `None` means "no filter." |
| 37 | code (form) | The **"✏️ Your task · 1.3"** instruction card for `apply_metadata_filter`. |
| 38 | code | **Exercise 3** — the `apply_metadata_filter` function body. See [Part E, §3](#3-apply_metadata_filter--13-keeping-only-the-chunks-that-match-a-filter) below. |
| 39 | code | Monkey-patches (as a `staticmethod`): `RetrievalCore._apply_metadata_filter = staticmethod(apply_metadata_filter)`. |
| 40 | code | `TestSuite("Part 1.3 — apply_metadata_filter")` — 5 checks (keeps matching chunks, `None` means everything passes, empty dict also means everything passes, all keys must match, an unknown key matches nothing rather than crashing). |
| 41 | markdown | Sets up "the payoff" demo. |
| 42 | code (form) | Ingests all nine real policy documents, then applies a metadata filter for one specific document and shows how many chunks survive; explains that filtering is a hard yes/no gate that runs *before* scoring — it can't rank anything, and it can't rescue a badly-worded query. |
| 43 | markdown | **§1.4 — Export your Part 1 functions.** Explains that running the next cell writes your three functions into `solutions/part1_ingestion.py`, which is what the full app actually runs on. |
| 44 | code | **Export cell** for Part 1 — writes `solutions/part1_ingestion.py`. This is one of the two cells that needed the bug fix described in [Part E, §9](#9-a-bug-fix-not-an-exercise--the-unicodeencodeerror-on-export) below. |
| 45 | markdown | Optional bonus aside about smarter vector stores (approximate-nearest-neighbour indexes like HNSW, which Qdrant actually uses internally) — not an exercise, just further reading. |
| 46 | markdown | Section header for **Part 2**. |

### Part 2 — The RAG core (cells 47–115)

| # | Type | What it does |
|---|------|--------------|
| 47 | code (form) | SVG diagram of the full retrieval path: Question → (Keyword search *and* Embedding search) → Fuse (RRF) → Rerank → LLM → Answer. |
| 48 | markdown | **§2.0 — First, watch retrieval fail.** Sets up a demonstration with two phrasings of the same underlying question. |
| 49 | code (form) | Runs a **throwaway** inline BM25 ranker (spelled out here because the real `keyword_search` isn't built until §2.2) on two versions of the same question — one using the policy's own vocabulary, one phrased the way a person actually talks — and shows the results side by side. |
| 50 | markdown | Explains the failure: BM25 only scores *exact shared words*; "time off" and "annual leave" share no tokens at all, so the correct document is invisible to it even though it answers the question perfectly. |
| 51 | code (form) | "What you are about to build" callout: previews embedding search (fixes the vocabulary-mismatch problem, but is fuzzy), hybrid search (combines both so a question only needs one method to work), and measurement (§2.5, how you *prove* any of this instead of just arguing it). |
| 52 | markdown | **§2.1 — Embedding (semantic) search.** Introduces the cosine similarity formula: `cos(q, d) = (q · d) / (‖q‖ * ‖d‖)`. |
| 53 | code (form) | SVG diagram of two vectors (arrows) with the angle θ between them, illustrating what cosine similarity actually measures. |
| 54 | markdown | Explains the shape of the function to write: `query_vec` is one vector, `matrix` holds one chunk vector per row, and the function must return one score per row — and must be *vectorized* (done with NumPy array operations across all rows at once, rather than a slow Python loop that processes rows one at a time). |
| 55 | code (form) | The **"✏️ Your task · 2.1"** instruction card for `cosine_similarity`. |
| 56 | code | **Exercise 4** — the `cosine_similarity` function body. See [Part E, §4](#4-cosine_similarity--21-scoring-how-similar-every-chunk-is-to-the-question) below. |
| 57 | code | Monkey-patches (as a `staticmethod`): `RetrievalCore._cosine_similarity = staticmethod(cosine_similarity)`. |
| 58 | code | `TestSuite("Part 2.1 — cosine_similarity")` — 3 checks (identical direction scores 1.0, perpendicular vectors score 0.0, a zero vector is handled safely with no `NaN`). |
| 59 | markdown | **§2.2 — Keyword (lexical) search.** Explains BM25 and, importantly, explains *why* there's a special `self._bm25_index(candidates)` helper instead of just rebuilding a `BM25Okapi` index inline: rebuilding it is expensive (it has to scan the whole corpus), and a naive implementation would pay that cost on *every single query* — twice per hybrid-search query — so the provided helper caches the built index for the normal (unfiltered) case. |
| 60 | code (form) | The **"✏️ Your task · 2.2"** instruction card for `keyword_search`. |
| 61 | code | **Exercise 5** — the `keyword_search` function body. See [Part E, §5](#5-keyword_search--22-ranking-chunks-by-exact-word-overlap) below. |
| 62 | code | Monkey-patches: `RetrievalCore.keyword_search = keyword_search`. |
| 63 | code | `TestSuite("Part 2.2 — keyword_search")` — 4 checks (the chunk with the query's words ranks first, `top_k` caps the results, a metadata filter restricts the candidates, an empty query returns no results). |
| 64 | markdown | **§2.3 — Combining the two: Reciprocal Rank Fusion.** Explains why BM25 scores and cosine scores can't simply be added together (incompatible numeric scales), and introduces the RRF formula: `fused_score(chunk) = Σ over lists  1 / (k + rank_in_that_list)`, with `k=60` as a conventional dampening constant. |
| 65 | code (form) | Visual diagram: a "keyword" ranked list and an "embedding" ranked list, showing how chunk B (which isn't #1 in either) ends up winning the fused ranking because it appears well-placed in *both*. |
| 66 | markdown | One-line explanation of why that's the desired behaviour. |
| 67 | code (form) | The **"✏️ Your task · 2.3"** instruction card for `reciprocal_rank_fusion`. |
| 68 | code | **Exercise 6** — the `reciprocal_rank_fusion` function body. See [Part E, §6](#6-reciprocal_rank_fusion--23-fairly-merging-two-different-rankings) below. |
| 69 | code | Monkey-patches (as a `staticmethod`): `RetrievalCore._reciprocal_rank_fusion = staticmethod(reciprocal_rank_fusion)`. |
| 70 | code | `TestSuite("Part 2.3 — reciprocal_rank_fusion")` — 3 checks (a chunk ranked highly in both lists wins, results are deduplicated by chunk id, `top_k` limits the output). |
| 71 | markdown | **§2.4 intro — Hybrid search: over-retrieve, then fuse.** Explains the over-retrieval problem: if each branch only returns `top_k` results, the fuser can never promote a chunk that ranked, say, 8th in one list and 6th in the other, because neither list was long enough to even contain it. Poses a reflection question about what happens once the pool size exceeds the corpus size. |
| 72 | code (form) | The **"✏️ Your task · 2.4"** instruction card for `hybrid_search`. |
| 73 | code | **Exercise 7** — the `hybrid_search` function body. See [Part E, §7](#7-hybrid_search--24-running-both-searches-wide-then-fusing) below. |
| 74 | code | Monkey-patches: `RetrievalCore.hybrid_search = hybrid_search`. |
| 75 | code | `TestSuite("Part 2.4 — hybrid_search")` — 6 checks (over-retrieval genuinely widens the candidate pool, still returns exactly `top_k` results, `overretrieve=1` reproduces the old narrow behaviour, a pool larger than the corpus degrades gracefully, the metadata filter reaches both branches). |
| 76 | markdown | Sets up a real-corpus demonstration of over-retrieval's effect. |
| 77 | code (form) | Runs `hybrid_search` on a real query with `overretrieve=1` vs `overretrieve=20` and shows the two result sets side by side. |
| 78 | markdown | Explains the comparison: a narrow pool can let an irrelevant document occupy a top-3 slot by default; a wide pool lets the chunks both retrievers actually agree on rise to the top. Sets up §2.5 to put a number on the improvement. |
| 79 | markdown | **§2.5 — Measure it, or you are guessing.** Introduces the 12-question gold evaluation set built into `evaluation.py` (6 "lexical" questions using the document's own rare terms, 6 "semantic" questions phrased the way a person actually asks), each labelled with which document should answer it. |
| 80 | code | Prints a handful of example evaluation questions along with their expected ("gold") source document and a note. |
| 81 | markdown | Explains the three measurement metrics used throughout this section: **recall@k**, **answer recall@k**, and **context cost** (see the glossary above), and why document recall alone is not enough (a whole un-chunked document "hits" every time while costing far more context). |
| 82 | code (form) | An honest caveat callout: the notebook runs offline using `HashingEmbedder`, a stand-in that scores *word overlap*, not real meaning — good enough to make rankings meaningful (rather than everything tying), but the semantic-question numbers below should be read as a floor, not a true estimate. |
| 83 | markdown | Sets up the head-to-head strategy comparison. |
| 84 | code | Runs keyword search, embedding search, and hybrid search through the same 12 questions at the same `top_k`, and shows a scoreboard table of document recall / answer recall / context cost for each. |
| 85 | code (form) | Breaks the same comparison down by question kind (lexical vs semantic) for each strategy. |
| 86 | markdown | A detailed analysis of the results: **document recall says hybrid wins** (it has no catastrophic weakness across question kinds), but **answer recall says keyword search wins**, which is presented as a genuine anomaly worth investigating rather than explaining away. Two competing explanations are proposed (fusion itself is the problem, vs. one input branch is simply weak), and the offline embedding branch's poor semantic score (0.33) is flagged as the more likely culprit — to be confirmed later in cell 97. |
| 87 | code (form) | Shows a per-question hit/miss grid for hybrid search, so you can see *which* specific questions still fail rather than just an aggregate number. |
| 88 | markdown | **§2.5 continued — Does over-retrieval actually help?** Sets up an empirical test of the claim made in §2.4. |
| 89 | code | Sweeps the `overretrieve` factor across `1, 2, 5, 20, 100` and plots document recall against it. |
| 90 | markdown | Explains the result: recall genuinely improves with a wider pool, then plateaus — which is why a moderate default (like 20) is a reasonable choice, and why that number is a measured decision rather than a guess. Introduces the chunk-size/overlap experiment next. |
| 91 | code | Defines `evaluate_config()` (re-ingests the whole corpus with given chunking settings, then scores retrieval over it), then runs five configurations (no chunking, and four chunk-size/overlap combinations) and shows a scoreboard. |
| 92 | markdown | Analyzes the (perhaps surprising) result: "no chunking" wins on *both* recall columns — but costs roughly 10× the context, because it always hands over an entire document. This is the concrete payoff of tracking three metrics instead of one: document recall alone cannot see this problem. Notes that very small chunks are cheap but score badly because an answer spanning a chunk boundary gets split and lost — motivating `overlap`. |
| 93 | code | Sweeps `overlap` at a fixed `chunk_size=120` across `0, 20, 40, 60` and shows a scoreboard. |
| 94 | markdown | Explains that overlap buys a real gain in answer recall at almost no added context cost, since answers that used to straddle two chunks now sit whole inside one. Invites the reader to change `K` (the `top_k` used throughout this section) and re-run to see the tradeoff curve for themselves. |
| 95 | code (form) | A callout on the limits of this evaluation set: 12 hand-written questions are enough to spot a trend or catch a regression, but not enough to *certify* a system — real evaluation sets are larger and continually refreshed. |
| 96 | markdown | Sets up the "settle it" experiment: if explanation (2) from cell 86 is right (the embedding branch was simply weak, not fusion itself), swapping in a real trained embedding model should lift both the embedding branch *and* hybrid's answer recall, with zero changes to the fusion code. |
| 97 | code | If an API key is available (`HAS_KEY`), re-runs the entire strategy comparison using a real `TextEmbedder` instead of the offline `HashingEmbedder`, and prints the results next to the earlier offline ones for direct comparison. |
| 98 | markdown | Analyzes the real-embedding results: the embedding branch's semantic-question score jumps from 0.33 to **1.00**, and hybrid's answer recall rises from 0.42 to **0.92** — confirming that fusion itself was never the problem. Draws three lessons: fusion only inherits the quality of what you feed it; a measurement is only as trustworthy as its setup (and you should report what was mocked); and a 12/12 perfect score means this evaluation set is now "saturated" and needs harder questions to keep being useful. |
| 99 | markdown | **§2.6 — Putting it together: answer with the LLM.** Explains the final RAG step (retrieve, build a context block, ask the LLM) and, critically, the rule for **abstention**: since the nearest chunks always exist even when nothing in the corpus is actually relevant, the function must check for "no results" and return the fixed `NO_CONTEXT_ANSWER` *without* calling the LLM at all. |
| 100 | code (form) | The **"✏️ Your task · 2.6"** instruction card for `retrieve_and_answer`. |
| 101 | code | **Exercise 8 (the last one)** — the `retrieve_and_answer` function body. See [Part E, §8](#8-retrieve_and_answer--26-the-final-step-asking-the-llm) below. |
| 102 | code | Monkey-patches: `RAGCore.retrieve_and_answer = retrieve_and_answer`. |
| 103 | code | `TestSuite("Part 2.6 — retrieve_and_answer")` — 4 checks (returns an answer plus source chunks, the retrieved context actually reaches the LLM prompt, abstains with an empty store, and abstaining never calls the LLM at all). |
| 104 | markdown | Sets up an "abstention in the wild" demonstration: the interesting case isn't an empty store, but a *full* store asked something it genuinely has no answer to. |
| 105 | code (form) | Ingests the whole real policy corpus (offline), then runs keyword search side by side for a genuinely answerable question and a completely unrelated one ("sourdough levain hydration autolyse"), showing the raw scores. |
| 106 | markdown | Explains that, by default, both cases return "three chunks and a confident reply" — `min_score` is the missing threshold that would tell them apart. |
| 107 | code | Sets `_qa.min_score = 3.0`, then re-asks the off-topic question and shows it now correctly abstains with `NO_CONTEXT_ANSWER`. |
| 108 | markdown | Sets up a demonstration of the threshold's fragility. |
| 109 | code (form) | Shows a *different* nonsense question — but phrased as a grammatically complete sentence — that scores *above* the same threshold anyway, purely because of shared common words like "what," "is," "the." Explains why (no stopword removal is done), and draws three lessons: the right threshold is corpus- and strategy-specific; it should be *set by measuring*, exactly as in §2.5; and a real embedding model separates on-topic from off-topic questions far more cleanly than raw BM25. |
| 110 | markdown | Sets up a demonstration of the exact prompt sent to the LLM. |
| 111 | code (form) | Answers a real question through the offline mock pipeline and displays it — since the mock LLM just echoes back whatever prompt it received, this view doubles as an "X-ray" of exactly what text gets sent to the real model. |
| 112 | markdown | Introduces an optional live demo that needs a real API key. |
| 113 | code | If a key is available, ingests the real policy folder and answers a real question using the actual embedding model and LLM (not mocks); otherwise prints a message and is skipped. |
| 114 | markdown | **§2.7 — Export your Part 2 functions.** Explains that running the next cell writes your five Part 2 functions into `solutions/part2_retrieval.py`. |
| 115 | code | **Export cell** for Part 2 — writes `solutions/part2_retrieval.py`. This is the second of the two cells that needed the bug fix described in [Part E, §9](#9-a-bug-fix-not-an-exercise--the-unicodeencodeerror-on-export) below. |

### Part 3 — The web app and appendix (cells 116–118)

| # | Type | What it does |
|---|------|--------------|
| 116 | markdown | **Part 3 (optional).** Explains how to run the actual website version of this project: a FastAPI **backend** (`server.py`) that wraps the pipeline in an HTTP API and holds one shared, persistent database while building a fresh `RAGCore` per request, and a React/Vite **frontend** with a key field, document upload, a question box with a search-strategy selector, and expandable source citations. Explains that on startup the backend runs `solutions.apply()`, which monkey-patches the same two files this notebook exports onto the real project — meaning until both `solutions/part1_ingestion.py` and `solutions/part2_retrieval.py` exist with all eight functions filled in, the website can boot but cannot actually answer a question. Gives the step-by-step commands to clone the repo, install dependencies (`uv sync`), and run both the backend (`uv run uvicorn server:app --reload`) and frontend (`npm install && npm run dev`). |
| 117 | markdown | **Appendix** — introduces a free scratch space using the same `TestSuite` test harness the graders use, for trying out edge cases or debugging anything you're unsure about. |
| 118 | code | A starter scratch cell: one example `TestSuite("My experiments")` case testing `sliding_window`, plus a commented-out line showing how to just print intermediate values instead. |

---

## Part E — Changes applied (the ✏️ TODOs)

This is the detailed part: exactly what was written into each blank, and why, for all eight
exercises — plus one small but important bug fix that had nothing to do with the exercises
themselves.

### 1. `sliding_window` — §1.1: splitting text into overlapping pieces

**The goal:** take a long piece of text and cut it into smaller, overlapping "windows" of words, so
that each idea in a long document gets its own chunk instead of being blended into one giant,
blurry vector.

Final code:

```python
def sliding_window(text, chunk_size=200, overlap=40):
    """Split `text` into overlapping windows of words."""
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})")

    words = text.split()
    if not words:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(words), step):
        window = words[start:start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
    return chunks
```

Explaining it piece by piece, imagine your text is a row of numbered wooden blocks, one per word,
lined up on a table. A "window" is just a fixed number of blocks you're looking at right now — say,
4 blocks wide.

- **`overlap >= chunk_size` guard.** Imagine sliding that 4-block window to the right by some step
  size each time. The "step" is how far you slide it. If the window is 4 blocks wide and the overlap
  (how many blocks the *next* window shares with the current one) is 4 or more, the window would
  slide forward by zero blocks or even backward — it would never actually move on to new material.
  That's an infinite loop, not just a wrong answer, so the code checks for this and refuses to run
  with such settings, raising a clear error instead of hanging forever.
- **`step = chunk_size - overlap`.** This is the actual distance the window slides each time. If
  `chunk_size=4` and `overlap=1`, `step = 3` — each new window starts 3 words after the previous one
  started, so 1 word (the last word of the previous window) is shared between them. If `overlap=0`,
  windows sit end-to-end with no sharing at all.
- **`words[start:start + chunk_size]`.** This is Python's way of "grab a slice of a list starting at
  position `start` and going for `chunk_size` items." If there aren't enough words left to fill the
  full window, Python's slicing automatically just returns however many are left — it doesn't crash
  or need special-case handling.
- **`" ".join(window)`.** The window is a list of separate word-strings, like
  `["annual", "leave", "policy"]`. `" ".join(...)` glues them back together with spaces between them
  into one single string: `"annual leave policy"`. That string is what gets stored as one chunk.
- **The final `if start + chunk_size >= len(words): break`.** Once a window reaches the very last
  word of the text, there's no more new material to slide onto — sliding forward again would just
  produce a shorter version of the *same* ending. So the loop stops right there rather than emitting
  a duplicate, overlapping "tail" chunk.

### 2. `ingest_document` — §1.2: turning one document into stored, searchable chunks

**The goal:** run the entire "write side" of the pipeline for one document — cut it into chunks
(using the function above), wrap each chunk with the right labels, turn each chunk's text into an
embedding vector, and save everything into the database.

Final code:

```python
def ingest_document(self, document):
    """Run load->CHUNK->EMBED->store for one Document. Return the list of stored Chunks."""
    texts = chunking.chunk_text(
        document.text,
        chunk_size=self.chunk_size,
        overlap=self.overlap,
        strategy=self.strategy,
    )
    if not texts:
        return []

    chunks = [
        Chunk(document_id=document.id, index=i, text=text,
              metadata=dict(document.metadata))
        for i, text in enumerate(texts)
    ]

    embeddings = self.embedder.embed_batch(texts)
    for chunk, embedding in zip(chunks, embeddings):
        chunk.embedding = embedding

    self.db.ensure_collection(self.collection_name, vector_size=len(embeddings[0]))
    self.db.insert_document(self.collection_name, document, chunks)
    return chunks
```

- **`metadata=dict(document.metadata))`.** Every document has a metadata dictionary — think of it
  as a little card of labels stuck to the document, like `{"source": "leave_policy.md"}`. Every
  chunk cut from that document should carry a copy of those same labels. The important word is
  *copy*. `dict(document.metadata)` creates a brand-new dictionary with the same contents, rather
  than reusing the exact same dictionary object for every single chunk. Why does that matter? Imagine
  ten kids sharing one single recipe card instead of each getting a photocopy: if one kid scribbles a
  note on it, all ten "see" that scribble, because it's really the same physical card. In code, if
  every chunk pointed at the literal same dictionary object and something later changed one chunk's
  metadata, it would silently change *every* chunk's metadata too. Making a fresh copy for each chunk
  avoids that trap entirely.
- **`embeddings = self.embedder.embed_batch(texts)`.** Rather than asking the embedding model to
  convert one chunk of text into a vector at a time (which would mean one slow network request per
  chunk), this sends *all* the chunk texts in a single request — a "batch" — and gets back a list of
  vectors in the same order the texts were given. This is much faster. Because the order is
  preserved, `zip(chunks, embeddings)` (which pairs up the first chunk with the first embedding, the
  second chunk with the second embedding, and so on) correctly attaches each vector to the chunk it
  actually belongs to.
- **`chunk.embedding = embedding`** inside that loop simply stores the matched vector onto its
  chunk object, so later code can find `chunk.embedding` whenever it needs it.
- **`vector_size=len(embeddings[0])`.** Every embedding vector produced by the same model has
  exactly the same length (the same number of numbers in the list) — e.g. always 384 numbers, or
  always 1536 numbers, depending on the model. Before the database can store anything, it needs to
  know how wide each vector slot should be, so it can set up storage of the right shape. Since all
  vectors are the same length, just checking the length of the very first one (`embeddings[0]`) tells
  you everything you need to know.

### 3. `apply_metadata_filter` — §1.3: keeping only the chunks that match a filter

**The goal:** given a list of chunks and a "wish-list" of labels they must match (or no wish-list at
all), return only the chunks that satisfy every item on the list.

Final code:

```python
def apply_metadata_filter(chunks, metadata_filter):
    """Keep only chunks whose metadata matches every key/value in the filter."""
    if not metadata_filter:
        return chunks
    return [
        chunk for chunk in chunks
        if all(chunk.metadata.get(key) == value for key, value in metadata_filter.items())
    ]
```

- **The no-filter case: `return chunks`.** If there's no filter at all (`metadata_filter` is `None`
  or an empty dictionary), every chunk should pass through untouched. The subtle but important detail
  here is returning the *exact same list object* that was passed in, rather than making a new list
  that merely *contains the same chunks*. Elsewhere in the project, a caching mechanism for the BM25
  search index checks "is this literally the same list I built my cache for?" — like recognising a
  friend by their actual face rather than a photograph that merely looks similar. If a copy were
  returned instead, that recognition would fail and the cache would rebuild itself needlessly on
  every single search.
- **The filtering case.** `metadata_filter.items()` walks through every `(key, value)` pair in the
  wish-list — e.g. `("source", "leave_policy.md")`. For a chunk to survive, `all(...)` requires
  *every single pair* to be satisfied (this is what "AND, not OR" means — matching just one label
  isn't enough if the filter asked for two). `chunk.metadata.get(key)` looks up that label on the
  chunk. Using `.get(key)` instead of `chunk.metadata[key]` matters: if the chunk simply doesn't have
  that label at all, `.get(key)` calmly returns `None` (which then just fails to equal `value`,
  correctly excluding the chunk) instead of crashing the whole search with an error, the way
  `chunk.metadata[key]` would if the key were missing.

### 4. `cosine_similarity` — §2.1: scoring how similar every chunk is to the question

**The goal:** given the question's embedding vector and a big table of every chunk's embedding
vector, compute one similarity score per chunk — all at once, without a slow one-at-a-time loop.

Final code:

```python
def cosine_similarity(query_vec, matrix):
    """Cosine similarity of `query_vec` against every row of `matrix`."""
    denom = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
    scores = matrix @ query_vec
    return np.divide(scores, denom, out=np.zeros_like(scores), where=denom != 0)
```

Picture every chunk's embedding as an arrow pointing somewhere in space, and the question's
embedding as one more arrow. Cosine similarity measures how closely two arrows point in the *same
direction*, ignoring how long each arrow happens to be. `matrix` here is literally a table with one
row per chunk and one column per number in the vector — so if there are, say, 500 chunks and each
embedding has 384 numbers, `matrix` is a table of 500 rows and 384 columns.

- **`np.linalg.norm(matrix, axis=1)`.** The "norm" of a vector is its length (how long the arrow
  is). `axis=1` tells NumPy (Python's numeric-computing library) "collapse each *row* down to a
  single number" — i.e. compute the length of each chunk's arrow separately, giving back one number
  per chunk (500 numbers, in the example above). Using `axis=0` instead would mistakenly collapse
  each *column*, mixing together numbers from different chunks, which isn't what's wanted here.
  Multiplying that by `np.linalg.norm(query_vec)` (the length of the question's single arrow) gives
  the full denominator of the cosine-similarity formula for every chunk in one shot.
- **`scores = matrix @ query_vec`.** The `@` symbol here means "matrix multiplication." Computing
  the dot product between two vectors (which is the numerator of the cosine-similarity formula)
  normally means multiplying matching numbers together and adding them all up. Doing that separately
  for 500 chunks with a Python `for` loop would be slow. `matrix @ query_vec` does the dot product of
  *every single row* against the query vector simultaneously, in one very fast operation — this is
  what "vectorized" means, and it's exactly what the exercise card asked for.
- **`np.divide(scores, denom, out=np.zeros_like(scores), where=denom != 0)`.** Normally you'd just
  write `scores / denom`. The problem is that a chunk whose embedding happens to be a zero vector
  (all zeros — length zero) would make the denominator zero, and dividing by zero in this context
  produces `NaN` ("Not a Number") — a poison value that can silently wreck every later calculation
  that touches it. `np.divide`'s `where=denom != 0` argument says "only actually perform the division
  in the positions where the denominator isn't zero." For every other position, `out=np.zeros_like(scores)`
  pre-fills the answer with `0.0`, so a zero-length chunk simply scores `0` (meaning "not similar at
  all") instead of corrupting the results with `NaN`.

### 5. `keyword_search` — §2.2: ranking chunks by exact word overlap

**The goal:** given a question, rank all the (optionally filtered) chunks by how well their exact
words overlap with the question's words, using the BM25 formula, and return the best few.

Final code:

```python
def keyword_search(self, query, top_k=None, metadata_filter=None):
    """Rank self.chunks by BM25 overlap with `query`. Return list[(Chunk, score)]."""
    top_k = self.top_k if top_k is None else top_k
    query_tokens = _tokenize(query)

    candidates = self._apply_metadata_filter(self.chunks, metadata_filter)
    if not candidates or not query_tokens:
        return []

    bm25 = self._bm25_index(candidates)
    scores = bm25.get_scores(query_tokens)
    return self._top_k(candidates, scores, top_k)
```

- **`candidates = self._apply_metadata_filter(self.chunks, metadata_filter)`.** This calls the very
  function written in exercise 3 above, narrowing the pool of chunks *before* any scoring happens —
  e.g. "only consider chunks from the leave policy," if that filter was requested.
- **`bm25 = self._bm25_index(candidates)`.** This is a provided helper (not one you write) that
  builds — or reuses a cached — BM25 search index over exactly the candidate chunks. A BM25 index
  needs to look at the whole set of documents it's searching to know things like "how rare is this
  word across the whole collection," which is why it can't just be built for a single question in
  isolation; it needs the full candidate pool up front.
- **`scores = bm25.get_scores(query_tokens)`.** This does the actual scoring: it takes the
  tokenized (lowercased, split-into-words) question and returns one relevance number per candidate
  chunk, in the exact same order as `candidates` — a higher number means "shares more, and rarer,
  words with the question."
- **`self._top_k(candidates, scores, top_k)`** is another provided helper that sorts the candidates
  by their score and hands back just the requested number of best `(chunk, score)` pairs.
- The early `return []` when there are no candidates or no query tokens exists because BM25's
  underlying library actually raises an error if you try to build an index over zero documents — so
  this check avoids ever reaching that crash.

### 6. `reciprocal_rank_fusion` — §2.3: fairly merging two different rankings

**The goal:** given several separately-ranked lists of chunks (e.g. one from keyword search, one
from embedding search), merge them into a single fair ranking — without ever trying to directly
compare a BM25 score to a cosine score, since those numbers live on completely different scales and
simply adding them together would be meaningless.

Final code:

```python
def reciprocal_rank_fusion(ranked_lists, top_k, k=60):
    """Fuse several ranked [(Chunk, score)] lists into one. Return list[(Chunk, fused_score)]."""
    fused_scores = {}
    chunks_by_id = {}
    for ranked in ranked_lists:
        for rank, (chunk, _score) in enumerate(ranked):
            fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1 / (k + rank)
            chunks_by_id[chunk.id] = chunk

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [(chunks_by_id[cid], score) for cid, score in ordered[:top_k]]
```

The trick behind Reciprocal Rank Fusion is to completely ignore the actual scores and only look at
each chunk's **position** — first place, second place, third place, and so on — within each list.

- **`enumerate(ranked)`** walks through a ranked list and, for each item, also gives you its
  position, starting from `0` for first place. So `rank=0` means "the very best result in this
  particular list."
- **`1 / (k + rank)`** is the RRF formula's contribution for one appearance in one list. Because
  `rank` starts small (0, 1, 2, …) for the best results, `1 / (k + rank)` gives a *bigger* number for
  results near the top of a list, and a *smaller* number the further down the list you go — exactly
  what you want, since being ranked #1 should count for more than being ranked #50. The constant
  `k=60` softens the difference between neighbouring ranks (the difference between rank 0 and rank 1
  is small relative to 60), which stops one list's single top pick from completely dominating
  everything else.
- **`fused_scores[chunk.id] = fused_scores.get(chunk.id, 0.0) + 1 / (k + rank)`.** This adds up a
  chunk's contribution *across every list it appears in*. If a chunk was 3rd in the keyword list and
  5th in the embedding list, its total fused score is `1/(60+2) + 1/(60+4)` — two separate bonuses
  added together. A chunk that both search methods "agree" is relevant, even moderately, can end up
  beating a chunk that only one method loved.
- **`chunks_by_id[chunk.id] = chunk`** just remembers which actual `Chunk` object goes with which id,
  so the final answer can hand back real chunk objects, not just id numbers.
- **`sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)`.** `fused_scores.items()`
  gives back `(chunk_id, total_score)` pairs. Sorting needs to be done by the *score* — that's
  `item[1]`, the second element of each pair — not by the chunk id itself (`item[0]`), which would
  produce a meaningless, essentially random ordering. `reverse=True` means "biggest score first,"
  since a higher fused score means a more relevant chunk.
- Finally, `ordered[:top_k]` keeps only the requested number of best results, and the last line
  converts the `(chunk_id, score)` pairs back into `(chunk, score)` pairs using the `chunks_by_id`
  dictionary built earlier.

### 7. `hybrid_search` — §2.4: running both searches wide, then fusing

**The goal:** run keyword search and embedding search generously (wider than the final answer
needs), fuse their results with the function above, then trim back down to exactly the number of
results the caller asked for.

Final code:

```python
def hybrid_search(self, query, top_k=None, metadata_filter=None, overretrieve=None):
    """Run both retrievers wide, fuse their rankings, then cut to top_k."""
    top_k = self.top_k if top_k is None else top_k
    overretrieve = self.overretrieve if overretrieve is None else overretrieve

    pool = top_k * overretrieve

    keyword_results = self.keyword_search(query, top_k=pool, metadata_filter=metadata_filter)
    embedding_results = self.embedding_search(query, top_k=pool, metadata_filter=metadata_filter)

    return self._reciprocal_rank_fusion(
        [keyword_results, embedding_results],
        top_k=top_k,
        k=self.rrf_k,
    )
```

- **Why "over-retrieve" at all?** Fusion (exercise 6) can only ever reorder the chunks it's actually
  handed — it can't invent new candidates out of thin air. If you only ever gave it, say, the top 5
  from each list, a chunk that both retrievers considered decent-but-not-amazing (say, 8th place in
  one list and 6th in the other) would never even be *in* either list, so fusion could never promote
  it, no matter how good the fusion math is. The fix is to ask each retriever for a much wider pool
  than you actually want to keep at the end, so fusion has real material to work with.
- **`pool = top_k * overretrieve`.** If the caller ultimately wants the top 5 results
  (`top_k=5`) and `overretrieve=20` (the project's chosen default), then `pool = 100` — each of the
  two search branches is asked to return its top 100 candidates, a much wider net than 5, giving
  fusion plenty to choose from before the final trim.
- **Both branches get the exact same `top_k=pool` and the exact same `metadata_filter`.** This
  matters for fairness and correctness: if, for example, only the keyword branch respected a filter
  narrowing the search to one document while the embedding branch searched everything, chunks from
  documents that should have been excluded could sneak back into the final results through the
  unfiltered branch — a "side door" around the filter.
- **`top_k=top_k` in the final fusion call.** The two branches were deliberately asked for a *wide*
  pool (`pool`, e.g. 100 results each), but the caller of `hybrid_search` only actually wants
  `top_k` (e.g. 5) results back. This line tells the fusion function "after you've merged everything,
  cut back down to just the best `top_k`" — the wide pool was only ever meant to give fusion good raw
  material to choose from, not to be the final answer size.

### 8. `retrieve_and_answer` — §2.6: the final step, asking the LLM

**The goal:** retrieve the relevant chunks for a question, and either honestly admit that nothing
relevant was found, or paste the retrieved text into a prompt and ask the LLM to answer using only
that information.

Final code:

```python
def retrieve_and_answer(self, query, search_type=None, metadata_filter=None):
    """Retrieve, then answer with the LLM. Return (answer_str, list[(Chunk, score)])."""
    results = self._retrieve(query, search_type, metadata_filter)
    if not results:
        return NO_CONTEXT_ANSWER, []

    context = "\n\n".join(chunk.text for chunk, _score in results)
    prompt = f"Context:\n{context}\n\nQuestion: {query}"
    answer = self.llm.complete(prompt, system_prompt=self.system_prompt)
    return answer, results
```

- **The abstention check: `if not results: return NO_CONTEXT_ANSWER, []`.** This is arguably the
  single most important line in the whole notebook, conceptually. Retrieval (via `self._retrieve`,
  which applies the `min_score` threshold discussed in cells 105–109) can come back completely
  empty — meaning nothing in the corpus was relevant enough to the question. If that happens, the
  correct behaviour is to say "I don't know" (the fixed `NO_CONTEXT_ANSWER` message) and stop right
  there, *without* calling the LLM at all. Why not just let the LLM try anyway? Because a capable,
  "helpful" language model handed some vaguely-related leftover text and a question it can't actually
  answer from that text will often still produce something that *sounds* confident and well-sourced —
  and is completely wrong. That's considered the single most damaging failure mode a RAG system can
  have (a wrong-but-confident answer is worse than an honest "I don't know"), so it's caught here,
  before the LLM ever gets involved.
- **`context = "\n\n".join(chunk.text for chunk, _score in results)`.** `results` is a list of
  `(chunk, score)` pairs. This line pulls out just the `.text` of each chunk (ignoring the score,
  which is why that part of the pair is named `_score` — the underscore prefix is a common Python
  convention meaning "I'm not using this value") and glues all those chunk texts together into one
  big block of text, with a blank line (`"\n\n"`) between each chunk so they stay visually separated
  when the LLM reads them. This combined block is the "context" — literally the pages photocopied out
  of the policy book and handed over before asking the question.
- **`prompt = f"Context:\n{context}\n\nQuestion: {query}"`.** This builds the actual message sent to
  the AI model: the retrieved context first, then the user's original question, clearly labelled so
  the model knows which part is background information and which part is the actual question to
  answer.
- **`answer = self.llm.complete(prompt, system_prompt=self.system_prompt)`.** This is the actual
  call to the language model. `self.system_prompt` is a separate, fixed set of instructions
  (configured elsewhere on the pipeline) telling the model *how* to behave — in particular, to answer
  only using the provided context and not from its own general knowledge. Sending it as a distinct
  `system_prompt` (rather than baking it into the same text as the question) is a standard way of
  giving an LLM standing behavioural instructions separate from the specific request.
- Finally, the function returns both the `answer` text *and* the `results` (the chunks that were
  actually used), so that whatever calls this function — including the real web app — can show the
  user exactly which source passages the answer came from, as citations.

### 9. A bug fix, not an exercise — the `UnicodeEncodeError` on export

While running the export cells (§1.4, cell 44, and §2.7, cell 115), a real (non-exercise) bug
surfaced on Windows:

```
UnicodeEncodeError: 'charmap' codec can't encode characters in position 614-615: character maps to <undefined>
```

**What was going wrong:** both export cells end with a line like `_path.write_text(_body)`, which
saves the exported Python code to a file. When you don't explicitly tell Python which "character
encoding" to use for writing text, it silently falls back to whatever the operating system's default
happens to be. Think of a character encoding as a translation table between the letters/symbols a
computer wants to write and the raw bytes actually saved to disk — different tables can represent
different sets of symbols. On Windows, that silent default is typically an old encoding called
`cp1252`, which can represent ordinary English letters and numbers just fine, but *cannot* represent
certain special characters — including the em dash (`—`) used throughout this very notebook's own
code comments, for example in a line like `# 2. WRAP — one Chunk per passage...`. Since the export
cells copy a function's source code (comments and all) directly into the file being written, and
several of the exercise functions' comments contain em dashes, writing the file out on Windows
crashed the moment it hit one of those characters.

**The fix**, applied to both export cells:

```python
# Before:
_path.write_text(_body)

# After:
_path.write_text(_body, encoding="utf-8")
```

`UTF-8` is a much more modern, universal encoding that can represent essentially any character from
any language (plus symbols, emoji, and so on) without trouble. Explicitly requesting it removes the
dependency on whatever the operating system happens to default to, so the same notebook now behaves
identically on Windows, macOS, and Linux.

One thing worth knowing: this notebook file is itself generated from a private instructor repository
(per this project's own `CLAUDE.md` documentation) — this fix was applied directly to the local
`.ipynb` copy in this project, so it will not automatically appear if the notebook is regenerated
upstream. It's worth flagging to the course instructor as a bug affecting any Windows-based student.

---

## Part F — A note on the tests

Every one of the eight exercises has its own `TestSuite` cell directly below it (using the project's
own lightweight test runner, not the popular `pytest` library). These aren't optional — they're the
fastest way to know whether an exercise's logic is actually correct, independent of whether it merely
"runs without crashing." All eight exercises' test suites pass with the final code shown above.
