# Project 6 — Policy-Compliance Verification Agent

*Documentation written for ETH course 275-0005-00L, "From Data to Solutions" — Weekend 6 project.*

---

## 1. What is this project, really? (the simple version)

Imagine a company where employees can click buttons on a dashboard to ask for things:
"I want to be reimbursed for a hotel," "I want to buy a new laptop," "I want access to
a server," "I want a week off." Every one of those requests has to follow the
**company's rulebook** (its policies) — e.g. "hotels above a certain price need a
receipt," "you can't buy something over CHF 5000 without three quotes," etc.

Today, a human has to read every request and check it against the rulebook by hand.
That's slow and boring. This project builds a **robot helper (an "agent")** that:

1. Watches what the employee typed in,
2. Checks it's not obviously broken (e.g. did they forget to fill something in?),
3. Goes and finds the *exact pages* of the rulebook that matter for this request,
4. Reads those pages and decides: **is this OK, or is it a problem?**
5. If it's a problem, a second little helper suggests **exactly how to fix it**,
6. A third helper puts little red flags, notes, and "click here to fix it"
   buttons on the screen so the employee understands what to change.

Nobody in this pipeline is allowed to "just make things up." Every verdict must point
at the *actual sentence* in the policy document that justifies it — that's what makes
the robot **trustworthy** instead of just guessing.

## 2. Background: this project builds on the previous one

In the previous weekend project (WE5), you built a **RAG system** (Retrieval-Augmented
Generation): a tool that can search a pile of documents and answer questions about them,
citing where the answer came from. That RAG system lives in this repo's `rag/` folder
and is reused here **unchanged** — it is imported as a *tool*, not rebuilt.

This project (WE6/WE7) wraps that RAG tool inside a bigger **agentic workflow** — a
chain of small, specialized steps, each with a clearly defined job, each passing
**structured data** (Python objects, not loose text) to the next one. That's the whole
point of the exercise: a reliable AI system is not one giant prompt that does
everything — it's several small, testable, auditable steps wired together.

## 3. Scope and goal

**Goal (from the official project brief):** Build a robust agentic workflow that makes
sure every dashboard action is compliant with company policy — flags problems, cites
the rules that were broken, and suggests concrete fixes.

**The four request types** the dashboard supports:

| Action Type | What it means |
|---|---|
| Expense Report | Submit a business expense for reimbursement |
| Procurement Request | Ask approval to buy something from a vendor |
| Access / Config Change | Ask for access to a system, or change a security setting |
| Time-off Request | Ask for leave / vacation |

**In scope:** the seven-step pipeline described below, fully offline-testable with
mocked AI, plus an optional "live" run using a real OpenRouter API key, plus a small
web dashboard (FastAPI backend + React frontend) that exercises the whole pipeline
end to end.

**Out of scope / explicitly flagged as "not production-ready"** by the assignment
itself: the retrieval only fetches a fixed number of policy chunks per request rather
than per problem (so a request with many violations might not get enough evidence for
all of them), and the "fix suggester" doesn't see what a *valid* value would look like,
so its suggested corrections aren't guaranteed to make sense. These are intentionally
left as "what's next" exercises.

## 4. The seven-step assembly line

Think of it like a school where your homework passes through five different people,
one after another, each doing one job and then handing it to the next person:

```
   Action (📥 what the        →  1. Validate   →  2. Context   →  3. Retrieve
   employee typed in)             (bouncer)         (translator)    (librarian)
                                                                          ↓
   6. Display  ←  5. Solution  ←            4. Verifier
   (messenger)      (tutor)                    (teacher)
        ↓
   Feedback (📤 red flags, notes, "click to fix" buttons on the dashboard)
```

7. **Pipeline** ties all six of the above into one single function call, so that
   submitting one action automatically walks it through every step above, in order.

Every box above is a Python function **you (the student) had to write** — marked with
a 🎯 in the notebook. Everything else (the data models, the RAG search tool, the
prompts, the JSON parsing helper, the test harness) was given to you as scaffolding.

## 5. Repository layout — what lives where, and why

```
Project-6--Policy-Compliance-Verification-Agent/
├── .env                      # your OpenRouter key — never committed (see .gitignore)
├── notebook/
│   ├── WE6_project_student.ipynb   # the exercise notebook you work in
│   └── FDD26-W6-Project.pdf        # the official assignment brief (slides)
├── agentic/                  # the 7 pipeline steps — models, prompts, and the
│                              #   6 "raise NotImplementedError" stubs you patch
├── rag/                      # the WE5 RAG search tool (reused unchanged) +
│                              #   the actual policy documents in rag/data/
├── examples/                 # sample requests (JSON) to test your code against
├── solutions/                # where your notebook EXPORTS your finished functions,
│                              #   so the real web app can pick them up
├── tests/                    # offline "mock AI" doubles + the little test runner
│                              #   used inside the notebook (MockLLM, TestSuite, ...)
├── clients.py                 # builds the *real* embedder + LLM clients (needs a key)
├── server.py / frontend/     # the actual web dashboard (FastAPI + React)
└── qdrant_data/               # the local vector database file storage
```

`agentic/`, `rag/`, `clients.py`, `server.py`, `frontend/`, `examples/`, `solutions/`,
`tests/`, and `qdrant_data/` were **not originally part of this repo** — they only exist
in the course's separate starter repository (`eth-fdd-fs26/FDD-WE6-public`), which the
Colab version of the notebook clones automatically. Since we're running the notebook
**locally** instead of on Colab, we manually pulled those folders in (see §7, "Local
setup work"), so this folder is now a fully self-contained copy of the project.

## 6. How the notebook is organized — cell by cell

The notebook alternates: **a markdown cell explaining a concept → a code cell where a
function has `???` blanks for you to fill in → an "Apply patching" cell → a "Test
cases" cell → a "See how it works on an example" cell.** That four-step rhythm repeats
for every one of the seven parts. Below is what each group of cells does.

### Cells 0–4 — Introduction
Explain the big picture (§1–§4 above): what the agent does, the four action types, the
seven-step diagram, and the "monkey-patching" trick this notebook uses everywhere:

```python
from agentic import action_validation
def validate_action(action):
    ...                                          # you write this in the notebook
action_validation.validate_action = validate_action   # then you overwrite the
                                                        # placeholder in the real module
```

Because the real pipeline always looks up `action_validation.validate_action` *by name,
at the moment it's called* (instead of importing the function directly by value), the
moment you overwrite it, your version runs everywhere — inside the notebook's own
tests, and inside the actual web app once you export your code.

### Cells 5–12 — Setup
- **Cell 6 (Setup):** figures out where the notebook's own folder is (`NOTEBOOK_DIR`),
  detects whether it's running on Google Colab or locally, and either clones the
  starter repo (Colab) or just adds the project folder to Python's import path
  (local). This is the cell we had to patch twice for our local setup — more in §7.
- **Cell 8 (API key):** loads your `OPENROUTER_API_KEY` — from a Colab Secret on
  Colab, or from your local `.env` file otherwise — and reports whether a live key is
  available. Everything else in the notebook works without one, using mock AI.
- **Cell 10 (Imports):** imports every provided building block once: the data models
  (`Action`, `Verdict`, `Problem`, ...), the action catalogue, the prompt templates,
  the JSON-extraction helper, the agent classes, the RAG tool, and the test harness.
- **Cell 12 (Display helpers):** defines pretty HTML "cards" (`show_action`,
  `show_policies`, `show_verdict`, `show_solutions`, `show_ui_actions`) used
  throughout the notebook purely for nicer output — no logic to implement here.

### Cells 13–18 — Part 1: Validate
Introduces the `Action` object (a request: `action_type` + `fields` + `context`), and
asks you to implement `validate_action` — cheap, deterministic sanity checks that run
*before* any AI is involved (no point spending tokens reasoning about a request that's
missing its amount field). Cell 14 just prints the action catalogue for reference; cell
17 runs 4 automated test cases against your implementation; cell 18 runs it once on a
real example file.

### Cells 19–23 — Part 2: Context
Turns the validated `Action` into a `VerificationContext`: a natural-language search
query plus a filter that narrows the policy search to just the one document that
actually governs this action type (e.g. only search the expense policy for an expense
report, not the leave policy too).

### Cells 24–29 — Part 3: Retrieve
Explains how policy documents get **chunked** (split into small, traceable pieces
following the markdown headings) and wraps the WE5 RAG search tool in
`retrieve_policies`: given the query + filter from Part 2, fetch the most relevant
policy passages as evidence. Cell 28 actually loads the real policy documents from
`rag/data/` and runs a real (offline, keyword-only) search.

### Cells 30–35 — Part 4: Verifier
The **first place an LLM actually reasons**. `verify_action` builds a prompt containing
the action plus the retrieved policy text, sends it to the LLM with strict instructions
to answer only in a fixed JSON schema, and parses that JSON into a typed `Verdict`
(with a list of `Problem`s, each citing a policy source and chunk). Tested offline with
a `MockLLM` that returns a canned, pre-written JSON reply.

### Cells 36–41 — Part 5: Solution
For every `Problem` the verifier found, `propose_solution` spawns one small, focused
"subagent" call whose only job is to fix that *one* problem, again grounded in the
retrieved policy text and returned as structured JSON.

### Cells 42–46 — Part 6: Display
`run_display_agent` is the one step that does **not** call an LLM at all — it's a
plain, deterministic translator from the verdict + solutions into a fixed set of UI
tool calls (`mark_ok`, `warn`, `highlight_field`, `attach_citation`,
`suggest_correction`). It's restricted to exactly those five tools so its behaviour is
predictable and reviewable, unlike free-form text generation.

### Cells 47–55 — Part 7: Pipeline + demos
`verify` glues all six previous steps into one method call. Cell 52 runs the *entire*
pipeline offline using a `ScriptedLLM`. Cells 53–55 are the **optional live demo**:
if `HAS_KEY` is true, it builds real embedder/LLM clients and runs the whole thing
against the real OpenRouter API.

### Cells 56–60 — Export, run the app, wrap-up
- **Cell 57 (Export):** writes each of your six implemented functions into
  `solutions/partN_*.py`, so the standalone web app (`server.py`) can pick them up —
  this is the cell that had the Windows encoding bug (see §7).
- **Cell 58:** instructions for running the real dashboard (`uv run uvicorn
  server:app --reload` for the backend, `npm run dev` for the frontend).
- **Cells 59–60:** a summary of what the exercise teaches, plus optional "what's next"
  ideas for making the workflow more robust (see §9 below).

## 7. Local setup work we did outside the notebook (the "plumbing")

The notebook was designed to run on **Google Colab**, where it clones its own
supporting code automatically and Colab Secrets hand it the API key. Running it
**locally in VS Code** instead needed some extra plumbing, done in this session:

1. **Pulled the missing project files** (`agentic/`, `rag/`, `clients.py`, `server.py`,
   `frontend/`, `examples/`, `solutions/`, `tests/`, `qdrant_data/`, `pyproject.toml`,
   `uv.lock`) from the course's `eth-fdd-fs26/FDD-WE6-public` repo into this folder,
   since only the notebook + PDF existed here before.
2. **Created `.env`** to hold the `OPENROUTER_API_KEY` locally, and a root
   `.gitignore` entry so it's never committed to GitHub.
3. **Patched cell 6 and cell 8** so the notebook can actually find both the code and
   the `.env` file when run from VS Code:
   - VS Code's Jupyter kernel sometimes starts with its working directory at the
     *workspace root*, not the notebook's own folder. We capture the notebook's real
     folder via `__vsc_ipynb_file__` (a variable VS Code injects for exactly this
     reason) into `NOTEBOOK_DIR`.
   - The existing "walk upward looking for an `agentic/` folder" logic now starts
     from `NOTEBOOK_DIR` instead of the raw working directory, so it reliably finds
     the project root regardless of where the kernel started.
   - `.env` is loaded from that same resolved project root (`root`), not from
     `NOTEBOOK_DIR` — this mattered once we moved the notebook itself into a
     `notebook/` subfolder (see next point), since `.env` stayed at the project root
     while the notebook is now one level deeper.
4. **Moved `WE6_project_student.ipynb` and `FDD26-W6-Project.pdf`** into a new
   `notebook/` subfolder for tidiness, and fixed the `.env`/`agentic` path resolution
   above so nothing broke.
5. **Fixed a Windows-only bug** in the export cell (cell 57): `Path.write_text(body)`
   with no explicit encoding uses Windows' default `cp1252` encoding, which can't
   represent the 🎯 emoji embedded in your solved functions' comments. Fixed by
   passing `encoding="utf-8"` explicitly.
6. **Diagnosed a Qdrant file-lock error** when starting the FastAPI server
   (`uv run uvicorn server:app --reload`): Qdrant's embedded/local mode only allows
   **one process at a time** to open its storage folder (`qdrant_data`). The error
   showed up because a Jupyter kernel from an earlier notebook run still had that
   folder open. Fix: don't run the notebook's live-demo cell and the FastAPI server
   against the same `qdrant_data` folder at the same time — restart whichever one you
   aren't actively using.

## 8. The seven parts you implemented — explained like you're five

Here's the same seven steps, but explained the way you'd explain them to a kid, plus
what actually went wrong (and how it got fixed) while writing each one.

### Part 1 — Validate: the bouncer at the door
Imagine a bouncer at a club checking your ticket before letting you in. He doesn't
care yet whether your night will be *fun* — he just checks: "Do you have a ticket at
all? Is it a real ticket for *this* club? Did you fill in your name on it?" That's
`validate_action`: before any 'thinking' happens, it makes sure the request even has
an action type, that the type is one of the four allowed ones, and that all the boxes
that must be filled in are actually filled in.

*What went wrong:* the very first draft wrote `required_fields.get(action.action_type, [])`
— treating `required_fields` like a dictionary you can `.get()` a value out of. But
`required_fields` is actually a little machine (a function) that you have to *call*
with parentheses to get an answer out of, like a vending machine: you press the button
(`required_fields(action.action_type)`), and it gives you the list of required fields.
Trying to `.get()` a machine doesn't work — you get "a function has no attribute get."

### Part 2 — Context: turning your problem into a question for the librarian
You don't walk into a library and hand the librarian a random object — you *ask a
question*, like "do you have any books about dinosaurs?" `build_context` does exactly
that: it turns the raw request (e.g. "I spent CHF 420 on a hotel") into a plain-English
question ("Is this Expense Report action allowed under company policy? Details:
category: hotel, amount: 420, ...") plus a note telling the librarian which one shelf
to search (the expense policy document, not all of them).

### Part 3 — Retrieve: the librarian fetches the exact pages
The librarian (the RAG search tool) takes your question and comes back not with a
whole book, but with the **exact paragraphs** that are relevant, each one labeled with
which document it came from and how good a match it is. `retrieve_policies` just
unwraps those results into a clean list your other helpers can read.

*What went wrong:* the search tool hands back a list of *pairs* — `(chunk, score)` —
like a matching card game where each card has two halves: the policy text on one side,
and a matching-score on the other. The first draft tried to read the score off the
*chunk* itself (`chunk.score`), but the chunk doesn't know its own score — only the
pair does. Fixing it meant using the `score` half of the pair directly, the one you
already get for free from `for chunk, score in results`.

### Part 4 — Verifier: the teacher grades the homework against the rulebook
Now, for the first time, we let an AI actually *think*. Imagine handing a teacher your
homework plus the exact rulebook pages the librarian found, and asking: "does this
follow the rules, yes or no, and if not, which rule did it break and why?" The teacher
isn't allowed to write a free essay — she must fill in a strict form (JSON) with fields
like `field`, `policy_source`, `explanation`, `severity`. `verify_action` builds that
prompt, sends it to the AI, and reads the filled-in form back into tidy Python objects.

### Part 5 — Solution: the tutor who fixes one mistake at a time
For every mistake the teacher found, we send in a **specialist tutor** who only cares
about *that one mistake* — not the whole homework — and whose only job is to say
exactly what to change (`proposed_fix`) and what the corrected value should be
(`corrected_value`), again pointing at the exact rulebook page that justifies the fix.

*What went wrong:* very similar mix-up to Part 6 below — mapping the wrong piece of
information into the wrong "slot" of the tool call. Caught by the automated tests
comparing against the exact expected corrected value.

### Part 6 — Display: the messenger who is only allowed to say five things
The teacher's and tutor's notes now need to appear on the actual dashboard screen —
red circles around bad fields, a little quote from the rulebook, a "click here to
apply this fix" button. But we don't let the AI directly draw on the screen (that
would be unpredictable!). Instead, the messenger (`run_display_agent`) is handed
exactly **five allowed actions** it's allowed to perform (`mark_ok`, `warn`,
`highlight_field`, `attach_citation`, `suggest_correction`) — like a waiter who can
only say five fixed sentences to the kitchen, never improvise.

*What went wrong (twice):*
- First mix-up: `ui.suggest_correction(field, corrected_value, proposed_fix)` takes
  its three arguments in a *fixed order*, like a form with three labeled boxes. The
  draft filled box 2 (`corrected_value`) with the *fix text* instead of the actual
  corrected number, and box 3 with something else — so the "correct value" the
  dashboard would have shown was the wrong kind of data entirely (text instead of the
  number `250`). Fixed by matching each argument to its correct labeled box.

### Part 7 — Pipeline: the school secretary who calls everyone in order
The secretary's whole job is to know, for every homework that comes in, exactly which
order to send it through: bouncer → librarian's translator → librarian → teacher →
tutor(s) → messenger. `VerifierPipeline.verify` is that secretary — it doesn't do any
of the actual work itself, it just calls each of your six functions, one after another,
in the right order, and hands the result of one to the next.

*What went wrong:* the secretary tried to call a person by the wrong name —
`self.verifier_agent.reason_over_action_and_policies(...)` — but the actual person
sitting in that office is named `self.verifier` (see the class's `__init__`), and her
actual job title/method is `verify_action`, not `reason_over_action_and_policies`.
Python doesn't guess what you meant by a similar-sounding name — it just says "there's
nobody here called that" (`AttributeError`). Fixed by calling the right name for the
right person: `self.verifier.verify_action(action, policies)`.

## 9. A known loose end worth checking

While tracing through Part 3 for this document, one small mismatch stood out that
hasn't broken any test (because no test checks this exact value) but is worth knowing
about: `retrieve_policies` currently reads the document's display title as
`chunk.metadata.get("title", ...)`, but the RAG loader (`rag/loaders.py`) actually
stores that value under the key **`"document_title"`**, not `"title"`. In practice
this means `document_title` on every `RetrievedChunk` currently falls back silently to
the `source` filename instead of showing the nicer, human-readable document title (e.g.
it shows `expense_reimbursement_policy.md` instead of something like "Expense
Reimbursement Policy"). Nothing crashes, and it doesn't affect verdicts, citations, or
the tests, but the *displayed* document title in the UI is not quite what the exercise
brief describes. Worth a one-line fix (`"title"` → `"document_title"`) whenever you're
back in that cell — just say the word and I can make that change.

## 10. What the assignment says to explore next (optional, bonus)

Straight from the notebook's own closing section — things this simple version doesn't
handle well yet, and that a more serious version of this agent would need to address:

- **Retrieval budget:** only a fixed number of policy chunks (5, by default) are
  fetched *per verdict*, not per problem. A request with many separate violations
  might not get enough evidence to properly ground every single one of them —
  retrieving once *per problem* instead of once *per action* might help.
- **Blind fix suggestions:** the solution subagent never sees what a *valid* value
  actually looks like for a field, so it can propose a "fix" that doesn't really make
  sense in context.
- An idea floated in the brief: build yet another agent whose whole job is to
  automatically try lots of different actions against the pipeline and collect
  feedback on where it fails — a simple, DIY way of stress-testing your own agent.
