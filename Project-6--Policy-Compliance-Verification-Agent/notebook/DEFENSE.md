# Defense Walkthrough — Policy-Compliance Verification Agent

*A 15-minute talking-point script. Pair with `DOCUMENTATION.md` for full detail on
every cell and every fix; this file is the condensed version to actually present.*

Suggested timing: ~3 min background/scope, ~9 min walking the seven parts and
defending the design/implementation choices, ~3 min known limitations + what's next.

---

## 1. Background, scope, goal (~3 min)

**Background.** The previous project (WE5) built a RAG system: search a pile of policy
documents, answer a question, cite the source. That pipeline is reused here, unchanged,
as one *tool* inside a larger system — this project is not about retrieval anymore,
it's about **wiring several small, specialized steps into one auditable agentic
workflow**.

**Scope.** A company dashboard lets employees submit four kinds of requests (expense
report, procurement request, access/config change, time-off request). The agent watches
one submitted request end to end and must:
- reject structurally broken input before spending any AI reasoning on it,
- decide compliance strictly by citing retrieved policy text (never invent a rule),
- propose one concrete fix per violation, each also grounded in policy text,
- reflect all of that on the UI through a fixed, auditable set of tool calls — never
  free-form text into the DOM.

**Goal.** Not "an LLM that answers policy questions" — a **controlled pipeline** where
every step passes typed, structured objects (`Action` → `VerificationContext` →
`RetrievedChunk` → `Verdict`/`Problem` → `Solution` → `UIAction`) to the next, so each
step is independently testable and every decision is traceable back to a specific
policy sentence.

## 2. The seven-part architecture, and why it's built this way (~1 min)

```
Action → 1.Validate → 2.Context → 3.Retrieve → 4.Verifier → 5.Solution → 6.Display
                                                                              ↓
                                                                        UI feedback
7. Pipeline wires 1–6 into one verify(action) call.
```

Key design defense: **why not one big prompt that does everything?** Because a single
free-text answer can't be independently unit-tested, can't be forced to cite its
source, and any parsing failure anywhere silently corrupts everything downstream. By
forcing every hop to be a typed object with a defined schema (see `agentic/models.py`),
each of the seven steps can be tested in complete isolation with a scripted mock LLM —
which is exactly what the notebook's test suites do — without ever calling a real
model.

## 3. Part-by-part: what it does and how I'd defend the implementation (~9 min, ~75s per part)

### Part 1 — `validate_action`
**What:** Deterministic, non-AI gatekeeping: reject a request with no `action_type`,
an unrecognized `action_type`, empty `fields`, or a missing/blank required field —
returning a list of human-readable error strings (empty list = valid).

**Defend it:** This step exists specifically to avoid *wasting LLM calls (and money) on
requests that are trivially wrong*, and to fail with a precise, addressable error
message rather than an opaque downstream exception. Implementation detail worth
explaining: `required_fields` is a *function* looked up per-`action_type`
(`required_fields(action.action_type)`), not a static dict — so the required-field list
is derived from the single source of truth in `ACTION_TYPES`, and adding a new action
type doesn't require touching this function at all.

### Part 2 — `build_context`
**What:** Converts the validated `Action` into a natural-language RAG query plus a
`metadata_filter` restricted to exactly the one policy document that governs this
action type (via `policy_source_for`).

**Defend it:** The `metadata_filter` is a precision lever — it stops the retriever from
pulling in irrelevant excerpts from unrelated policies (e.g. leave policy noise on an
expense-report verification), which both improves retrieval accuracy and keeps the
downstream prompt small and focused.

### Part 3 — `retrieve_policies`
**What:** Wraps the WE5 RAG core's `.retrieve(query, search_type, metadata_filter)`
call and reshapes each `(Chunk, score)` pair into a `RetrievedChunk` DTO exposing only
what the agents/UI actually need: `source`, `chunk_id`, `document_title`, `text`,
`score`.

**Defend it:** This is a deliberate **anti-corruption layer** — the rest of the
pipeline never touches the RAG library's internal `Chunk` type directly, so the RAG
implementation could be swapped out later without touching any agent code. Also worth
mentioning proactively: the hierarchical chunking strategy (splitting on markdown
headers, prepending parent headings) keeps every chunk traceable back to *exactly*
which section of which document it came from — that traceability is what lets Part 4
cite a `chunk_id`, not just a filename.

### Part 4 — `verify_action`
**What:** The first point an LLM reasons. Builds a prompt embedding the action + the
retrieved policy text, sends it with a system prompt that fixes the exact JSON schema
of the reply, and parses that JSON (via the provided `extract_json` helper) into a
typed `Verdict` with a list of `Problem`s, each carrying `field`, `policy_source`,
`chunk_id`, `explanation`, `severity`.

**Defend it:** Two guarantees make this trustworthy rather than "an LLM vibing": (1)
**grounding** — the system prompt explicitly forbids inventing rules not present in the
retrieved excerpts; (2) **structured output with defaults** — every `Problem` field is
read defensively (`item.get(key, default)`), so a slightly malformed model reply
degrades gracefully into a well-typed object with sane defaults instead of crashing the
whole pipeline.

### Part 5 — `propose_solution`
**What:** One focused LLM subagent call *per detected problem* (not one call for the
whole verdict) — proposes a concrete `proposed_fix` and `corrected_value`, grounded in
the retrieved policy text, citing its own `policy_source`/`chunk_id`.

**Defend it:** This is the "detection vs. repair separation" principle in practice: the
verifier's job is only to *detect*, never to *fix* — repair is delegated to a
narrowly-scoped subagent whose entire context is one problem, which keeps its prompt
small, its output easy to validate, and its failure blast-radius limited to a single
field rather than the whole verdict.

### Part 6 — `run_display_agent`
**What:** The only step with **zero LLM calls** — a deterministic translator from
`VerificationResult` into a fixed set of UI tool calls: `mark_ok`, `warn`,
`highlight_field`, `suggest_correction`, `attach_citation`.

**Defend it:** This is the clearest illustration of "controlled tool use" in the whole
project — the agent is architecturally *incapable* of doing anything to the UI besides
those five calls (`UITools` is an `ABC` with exactly those five abstract methods), so
its behavior is fully predictable and reviewable, unlike letting an LLM emit arbitrary
HTML/JS. Determinism here is a deliberate reliability choice, called out explicitly in
the assignment brief, not a shortcut.

### Part 7 — `VerifierPipeline.verify`
**What:** The orchestrator — calls Parts 1–6 in order, short-circuiting with an
`ActionValidationError` if Part 1 fails, and spawning one Part-5 call per problem Part
4 found.

**Defend it:** Nothing here does its own reasoning — it's the one function whose job is
purely to *sequence and pass data*, which is exactly what makes the whole system
composable: because every step is resolved by attribute/module lookup at call-time
(`self.verifier.verify_action(...)`, `action_validation.validate_action(...)`), a
monkey-patched student implementation flows through unchanged, both inside the
notebook's own tests and inside the real deployed web app.

## 4. Debugging note worth mentioning if asked "what went wrong along the way?"

Every failure hit during implementation was a **wiring mistake, not a design
mistake** — calling `required_fields` like a dict instead of a function, reading a
retrieval score off the wrong half of a `(chunk, score)` pair, swapping two positional
arguments to `suggest_correction`, and calling a differently-named
attribute/method (`self.verifier_agent.reason_over_action_and_policies` instead of
`self.verifier.verify_action`). All four were caught immediately by the provided
automated test suites (`TestSuite`) before ever reaching the live app — which is itself
a point worth making in the defense: **the test-first structure of this notebook is
what makes these mistakes cheap to find**, instead of surfacing as a silent wrong
verdict in production.

## 5. Known limitations to volunteer, not hide (~2 min)

Be upfront about these — they show you understand the system's edges, not just its
happy path:

1. **Fixed retrieval budget per verdict, not per problem.** Only `top_k` policy chunks
   (5 by default) are retrieved once per action. A request with several unrelated
   violations may not get enough distinct evidence to ground every single one.
2. **Solution subagent is "blind" to valid values.** It never sees what a *compliant*
   value for a field would look like, only the policy text — so a proposed correction
   isn't guaranteed to be sensible, only policy-grounded in spirit.
3. **A small metadata-key mismatch in Part 3**: `retrieve_policies` currently reads a
   chunk's display title via `chunk.metadata.get("title", ...)`, but the RAG loader
   stores it under `"document_title"`. It silently falls back to the source filename —
   nothing crashes and no test catches it (none asserts the exact title), but the UI's
   displayed document title isn't as human-readable as intended. Good example of a bug
   that passes every test yet still isn't "correct."

## 6. One-line closing statement

"This project isn't an LLM that answers policy questions — it's a controlled pipeline
of small, typed, independently-testable steps, where the only free-form reasoning
happens in two tightly grounded places (the verifier and the solution subagent), and
every other step — including everything that touches the UI — is deterministic and
auditable by design."
