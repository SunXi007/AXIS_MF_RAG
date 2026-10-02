# Implementation Guide — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Project:** `nextleap_M4_RAGchatbot`
**Derived from:** [`PRD.md`](PRD.md) · [`architecture.md`](architecture.md)
**Purpose:** Phase-by-phase build guide for an AI coding agent (Cursor / OpenCode / Claude Code)
**Status:** Draft for review
**Last updated:** 2026-10-02

---

## How to use this guide

This is the **execution layer**. `PRD.md` says *what*, `architecture.md` says *how*, and this file says
*what to type, in what order, and how to know each step is finished*.

**Working method — one phase per session:**

1. Open the project root in Cursor.
2. Paste the **Agent Prompt** block from the phase you are starting (Appendix B has them ready).
3. Let the agent implement that phase **only**.
4. Run the phase's **Verification** commands yourself. Do not trust the agent's claim that it works.
5. Record the result in the **Phase Log** table (Appendix A).
6. Commit only if the phase is green.
7. Move to the next phase.

**Non-negotiable rule:** a phase is not done because code exists. It is done when every row of its
**Acceptance gate** passes. Gates are deliberately objective and runnable so there is no judgement call
about whether a phase is complete.

**If a phase's gate fails, stop.** Do not start the next phase on a red gate. The phases are ordered so
that each one's output is the next one's input; building on a broken gate compounds the error.

---

## Phase dependency graph

```
  P0  Scaffold
  │    gate: healthcheck prints GREEN, .env ignored
  ▼
  P1  Corpus fetch + extract
  │    gate: ingest_report.md shows 18/19 fetched, extraction sane
  ▼
  P2  Normalize + chunk          ◄── HARD GATE (mandatory human sign-off)
  │    gate: §9.5 of PRD, 5 criteria, human-reviewed chunks.txt
  ▼
  P3  Embed + store
  │    gate: collection populated, dim 384, re-run = 0 duplicates
  ▼
  P4  Retrieve
  │    gate: 10-fact probe locates each fact w/ correct scheme+plan
  ▼
  P5  Generate
  │    gate: answers <=3 sentences, cited, no unsupported numerics
  ▼
  P6  Guardrails + UI
  │    gate: all 5 refusal/PII paths demonstrated in the UI
  ▼
  P7  Eval + deliverables
       gate: all PRD §14 Must metrics met, 8 deliverables present
```

### Gate summary

| Phase | Blocking gate | Why it blocks the next phase |
|---|---|---|
| P0 | healthcheck GREEN | Every later phase imports `config.py` |
| P1 | extraction report complete | Chunking has no input without extracted text |
| **P2** | **human reviews `artifacts/chunks.txt`** | **Brief requires it. Bad chunks are invisible downstream — retrieval just quietly gets worse** |
| P3 | idempotency proven | Duplicate chunks silently distort top-k |
| P4 | probe set passes | A bad index produces plausible wrong answers |
| P5 | verifier passes | Guardrails assume a trustworthy answer path |
| P6 | all refusal paths work | Cannot demo safely without them |
| P7 | metrics + deliverables | Submission requirement |

---

## Global conventions (apply to every phase)

State these once, at the start of every session, so the agent does not drift.

| Rule | Detail |
|---|---|
| **Python** | 3.11+. Type hints on every public function. No bare `Any`. |
| **Comments** | **Do not write comments in code.** Docstrings on public functions only. Names and types carry the meaning. |
| **Config** | Every tunable comes from `config.py`. No magic numbers in any other module. |
| **Secrets** | `GROQ_API_KEY` only via `python-dotenv` from `.env`. Never log, echo, or commit it. `.env` must be in `.gitignore` on day one. |
| **Dependencies** | Only the libraries in Appendix C. If the agent wants a new one, it must stop and ask. |
| **Errors** | Never `except: pass`. Log and continue, or raise. A silent failure in ingestion produces a silently empty corpus. |
| **PII** | Never write user input to disk or logs. `TraceRecord` stores `question_len` + `question_hash` only. |
| **Imports** | `config.py` imports nothing from the project. Nothing in the request path imports from the ingestion path (except `config`, `templates`). |
| **Scope** | Axis MF only. 4 schemes. No advice features. No multi-AMC. No conversation history. |
| **Commits** | One commit per phase, message `P<n>: <summary>`. Never commit `.env`, `data/`, `chroma/`. |

### Fixed values (do not re-derive)

These come from `PRD.md` §9.2 / §14.2 and `architecture.md` §4.5.2. The agent must use them verbatim.

| Key | Value |
|---|---|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `EMBED_DIM` | `384` |
| `COLLECTION_NAME` | `axis_mf_faq` |
| `MAX_CHUNK_CHARS` | `900` |
| `MIN_CHUNK_CHARS` | `120` |
| `CHUNK_OVERLAP` | `150` |
| `TABLE_OVERLAP` | `0` |
| `MIN_EXTRACTED_CHARS` | `2000` |
| `TOP_K` | `5` |
| `SIMILARITY_THRESHOLD` | `0.35` (recalibrate in P4, record the value) |
| `CITATION_SCORE_FLOOR` | `0.45` |
| `MAX_CONTEXT_CHARS` | `6000` |
| `TEMPERATURE` | `0.0` |
| `MAX_TOKENS` | `220` |
| `GROQ_MODEL` | `llama-3.1-8b-instant` |

### Dependency budget (reject anything else)

`sentence-transformers` · `chromadb` · `pymupdf` · `trafilatura` · `beautifulsoup4` · `lxml` ·
`httpx` · `groq` · `python-dotenv` · `tiktoken` · `streamlit` · `pytest` · `tqdm`

Explicitly forbidden: LangChain, LlamaIndex, openai, pinecone, weaviate, qdrant-client, any agent
framework. They are rejected for reasons in `architecture.md` §1.

---

# P0 — Scaffold

**Goal:** a repository that boots, loads config, validates secrets, and asserts the index is present —
before any ML code exists.

**Reference:** `architecture.md` §3.2, §6.1, §6.2, §10

### Tasks

1. Create the directory skeleton exactly as `PRD.md` §18.4, plus `src/__init__.py` and `tests/__init__.py`.
2. Write `requirements.txt` from the dependency budget above, **with pinned versions**.
3. Write `.gitignore` containing at minimum: `.env`, `data/`, `chroma/`, `artifacts/`, `__pycache__/`,
   `*.pyc`, `.venv/`, `models/`.
4. Write `.env.example` with `GROQ_API_KEY=` (empty) and `AXIS_TOP_K=5`.
5. Write `config.py` — every key from `architecture.md` §6.1, `AXIS_`-prefixed env overrides, and the
   import-time assertion `PII_LOG_RAW is False`.
6. Write `src/sources.py` — dataclass `Source`, loader for `sources.csv` with validation
   (unique `source_id`, url non-empty, enums valid), and the `enabled` filter.
7. Write `sources.csv` — all 19 rows from `PRD.md` §5.1. `s15` (indmoney.com) must be present with
   `enabled=false` per decision D5.
8. Write `src/healthcheck.py` — prints dependency versions, config values, `key_present: bool`
   (**never the key**), source count by scheme, and whether the Chroma collection exists.
9. Write `src/trace.py` — the `TraceRecord` dataclass and JSON-line emitter from `architecture.md` §6.3.
10. Write `tests/test_config.py` and `tests/test_sources.py`.

### Acceptance gate

```bash
python -m src.healthcheck            # exits 0, prints GREEN
python -c "from src.config import PII_LOG_RAW; assert PII_LOG_RAW is False"
python -m pytest tests/ -q           # all pass
git check-ignore .env data/ chroma/  # all three ignored
```

| # | Criterion |
|---|---|
| 1 | `healthcheck` exits 0 and reports 18 enabled / 1 disabled source |
| 2 | `PII_LOG_RAW is False` assertion holds |
| 3 | `git check-ignore` confirms `.env`, `data/`, `chroma/` are ignored |
| 4 | No key value appears anywhere in healthcheck output |
| 5 | `pytest` green |
| 6 | `requirements.txt` versions are pinned (no bare names) |

**Do not create yet:** `extract.py`, `chunk.py`, `embed.py`, anything importing
`sentence_transformers` or `chromadb`.

---

# P1 — Corpus fetch + extract

**Goal:** every enabled URL is fetched, stored raw, and converted to structured text blocks.

**Reference:** `architecture.md` §4.1, §4.2, §4.3

### Tasks

1. `src/models.py` — shared dataclasses: `Source`, `FetchResult`, `Block`, `Chunk`, `Hit`,
   `ScreenResult`, `VerifyResult`, `AnswerEnvelope`, `IngestReport`. Keep them in one module so
   `guardrails.py` stays importable without pulling in ML dependencies.
2. `src/fetch.py` — `fetch_source`, `fetch_all` per `architecture.md` §7.
   - `httpx` with browser UA, `timeout=30`, 3 retries, exponential backoff, `1.5 s` delay between requests.
   - Save to `data/raw/<source_id>/<original_filename>` (from the URL path; synthesise if absent).
   - Record `fetched_at`, `http_status`, `content_hash` (sha256 of bytes), `byte_size`.
   - On failure: append to the report, **continue** (NFR-6).
3. `src/extract.py` — `extract_blocks(path, doc_type) -> list[Block]`.
   - HTML → `trafilatura.extract(output_format="xml")`; parse `<head>` into `heading_path`,
     `<table>` into a table block, `<p>`/`<li>` into text blocks. Fallback to `bs4`+`lxml`.
   - PDF → `pymupdf`, page by page. Heading detection: font size > body median, or a
     `^\d+(\.\d+)*\s+\S` numbering pattern. Table detection via ruling lines / column clustering.
   - `Block` = `{kind, text, heading_path, page_num, table_rows}` where
     `kind ∈ {heading, text, list, table}`.
   - Save `data/processed/<source_id>.blocks.json`.
4. Write `artifacts/ingest_report.md` — a table: `source_id | scheme | status | http | bytes |
   extracted_chars | flag`. `flag` ∈ `ok`, `fetch_failed`, `suspected_client_rendered`, `extract_failed`.
5. `tests/test_extract.py` using small saved fixtures in `tests/fixtures/`.

### Acceptance gate

```bash
python -m src.fetch
python -m src.extract
python -m pytest tests/ -q
```

| # | Criterion |
|---|---|
| 1 | Report shows **18 enabled** sources attempted, each with a status |
| 2 | `data/raw/` contains one directory per enabled `source_id` |
| 3 | Every row with `flag=ok` has `extracted_chars >= 2000` |
| 4 | Every `fetch_failed` row has the URL and HTTP status recorded (R4 — record, never silently substitute) |
| 5 | Any `suspected_client_rendered` source is listed — these are the R1 risk |
| 6 | `pytest` green |

### Agent instruction (important)

After this phase, **stop and report to the user.** The list of `suspected_client_rendered` and
`fetch_failed` sources determines whether P2 can cover all four schemes. If Large Cap or Flexi Cap
scheme pages come back as shells, the same facts must be sourced from the SID PDFs instead — which
changes what P2 needs to extract. Do not proceed to P2 on assumption.

---

# P2 — Normalize + chunk  ·  HARD GATE

**Goal:** structure-aware chunks on disk, reviewed by a human, **before any embedding code exists**.

**Reference:** `PRD.md` §9 (authoritative) · `architecture.md` §4.4, §4.5

> This phase is deliberately separated from P3. The brief requires the chunking strategy to be proposed,
> justified, and inspected before indexing. Bad chunks degrade retrieval *silently* — the system still
> returns answers, they are just wrong. This gate is the only place that failure is visible.

### Tasks

1. `src/normalize.py` — `normalize_blocks`, `drop_repeated_pdf_lines`.
   Apply in the exact order of `architecture.md` §4.4: NFC unicode → NBSP/soft-hyphen →
   de-hyphenate `(\w)-\n(\w)` → collapse blank lines and spaces → drop PDF lines recurring on
   > 50% of pages matching `page \d+|axis mutual fund|www\.|^\d+$` → drop HTML nav/footer/cookie/
   breadcrumb/CTA regions → drop orphan numeric-only lines.
2. `src/chunk.py` — `chunk_document`, `flatten_table`, `write_chunks_txt`.
   Implement the three tiers of `architecture.md` §4.5 **exactly**:
   - **Tier 1** tables → atomic, `flatten_table`, overlap 0. Emit `is_table=True`.
   - **Tier 2** heading-bounded sections ≤ 900 chars → one chunk each, with `heading_path` prepended.
   - **Tier 3** sections > 900 chars → `recursive_split` with separators
     `"\n\n"` → `"\n"` → sentence `(?<=[.!?])\s+` → `" "`, overlap 150.
   - `merge_short_chunks` folds anything < 120 chars into its neighbour.
3. `flatten_table` must expand empty first-column cells into their column header, so

   | | Direct Growth | Regular Growth |
   |---|---|---|
   | Expense ratio | 0.95% | 1.95% |

   becomes `Fees > Expense ratio (Direct Growth): 0.95%` and
   `Fees > Expense ratio (Regular Growth): 1.95%`.
   **This is the mechanism that satisfies goal G2** — plan identity is inside the embedded text.
4. Emit `artifacts/chunks.json` (schema `architecture.md` §4.6) and `artifacts/chunks.txt`
   (format `architecture.md` §4.7), **including the appended quality summary block** (A-2):
   per-scheme and per-plan counts, length histogram vs the 400–900 median target, count of chunks
   failing criterion 1.
5. `tests/test_chunk.py` — Tier 1 atomicity, Tier 2 heading prefix, Tier 3 overlap, short-chunk merge,
   `flatten_table` output for the matrix above.

### Do not create yet

`embed.py`, `ingest.py`, or anything importing `sentence_transformers` / `chromadb`. The gate comes first.

### Acceptance gate — automatic portion

```bash
python -m src.normalize
python -m src.chunk
python -m pytest tests/ -q
```

| # | Criterion | Threshold |
|---|---|---|
| A1 | Every chunk has `chunk_id`, `source_id`, `source_url`, `doc_type`, `scheme`, `plan`, `section_path`, `content_hash` | 100% |
| A2 | Chunks identify scheme **and** plan, or are explicitly `scheme=amc_wide` | ≥ 90% |
| A3 | No chunk contains nav, breadcrumb, cookie text, or repeated PDF headers | 0 |
| A4 | Any chunk containing a fee/exit-load/SIP figure also contains its label | 100% |
| A5 | Median chunk length in 400–900 chars | ≥ 95% within `[120, 900]` |
| A6 | Tier-1 table chunks are never split | 100% |
| A7 | `pytest` green | — |

### Acceptance gate — HUMAN REVIEW (blocking)

Open `artifacts/chunks.txt` and read it. This cannot be automated, and it is the entire point of the phase.

| # | Manual check | Pass condition |
|---|---|
| H1 | Spot-check 10 chunks | Each is readable standalone — a chunk retrieved alone still makes sense |
| H2 | Locate 10 known facts | Each appears in **at most one** chunk |
| H3 | Read 10 randomly chosen chunks | No boilerplate, no orphaned table fragments |
| H4 | Check Direct vs. Regular entries | Both variants present, correctly labelled, not merged into one row |
| H5 | Check a `downloads`-type chunk | Procedure reads as ordered steps, not shredded bullets |
| H6 | Confirm scheme coverage | All 4 schemes have chunks |

**If any check fails: fix the chunker and re-run. Do not proceed to P3.**

Record the outcome in Appendix A. This is a milestone deliverable (`PRD.md` §12.6) and goal G7.

---

# P3 — Embed + store

**Goal:** a persistent ChromaDB collection, built once, idempotent on re-run.

**Reference:** `architecture.md` §4.8, §4.9

### Tasks

1. `src/embed.py` — `Embedder` class. `all-MiniLM-L6-v2`, `normalize_embeddings=True`, batch size 32,
   `show_progress_bar` during ingest. `fingerprint` = `f"{model_name}@{__version__}:{dim}:{normalize}"`.
2. `src/ingest.py` — `ingest(only_fetch, only_embed, force, prune)` plus a CLI.
   - Idempotency exactly as `architecture.md` §4.8: compare file hash **and** per-chunk content hashes
     against `artifacts/ingest_state.json`; skip unchanged; on change, `collection.delete(where={"source_id": ...})`
     then re-embed and upsert only that source.
   - `--prune` removes chunks whose `source_id` is no longer enabled.
   - Store the embedder fingerprint in `ingest_state.json` and **warn loudly on mismatch**.
   - Embedding input is `section_path + "\n" + text` — the breadcrumb is part of the vector (AD-2).
   - Append embedding stats to `artifacts/ingest_report.md`.
3. `tests/test_ingest.py` — run `ingest` twice in a temp dir with a fake embedder; assert identical
   chunk-id sets and zero duplicates.

### Acceptance gate

```bash
python -m src.ingest
python -m src.ingest                 # run twice on purpose
python -m pytest tests/ -q
python -m src.healthcheck            # collection now reported present
```

| # | Criterion |
|---|---|
| 1 | `chroma/` exists on disk; collection `axis_mf_faq` populated |
| 2 | Second run logs `unchanged` for every source and performs **zero** upserts |
| 3 | Chunk-id set identical between runs |
| 4 | Stored embedding dimension = 384 |
| 5 | Every chunk id in `artifacts/chunks.json` exists in the collection (and vice versa) |
| 6 | Restart reloads from disk without re-embedding (NFR-2) |
| 7 | `ingest --prune` removes orphans |
| 8 | `pytest` green |

---

# P4 — Retrieve

**Goal:** questions locate the right chunk, with the right scheme **and plan**.

**Reference:** `architecture.md` §5.2, §5.3

### Tasks

1. `src/retrieve.py` — `retrieve`, `infer_metadata_filter`.
   - Embed the question with the **same** `Embedder` instance; assert the fingerprint matches
     `ingest_state.json` and raise if not (F3).
   - `collection.query(n_results=TOP_K, where=filter, include=["documents","metadatas","distances"])`.
   - `Hit.score = 1 - distance`; drop hits below `SIMILARITY_THRESHOLD`.
   - `infer_metadata_filter`: scheme aliases (`"flexi cap"→flexi_cap`, `"elss"`/`"tax saver"→elss`,
     `"mid cap"→mid_cap`, `"large cap"→large_cap`); plan aliases (`"direct"→direct`,
     `"regular"→regular`).
     **Plan filter is one-directional (A-6):** a named plan yields
     `{"$and":[{"scheme":...},{"$or":[{"plan":matched},{"plan":"n_a"}]}]}`; **no plan named ⇒ no plan
     filter**. `n_a` must always be included so SID/KIM fee tables survive.
   - No alias match ⇒ no filter. Recall is never reduced by a failed guess.
2. `src/templates.py` — `render(template_id, **kwargs)` for the 6 template ids from
   `architecture.md` §5.7, plus `DISCLAIMER` copied **verbatim** from `PRD.md` §11.5.
3. A trace/debug entry point: `python -m src.retrieve "<question>"` prints each hit with score,
   scheme, plan, doc_type, and `section_path`.
4. **Threshold calibration** — required, not optional. Run the 10-fact probe; for each, note the score
   of the correct chunk and the score of the top-scoring irrelevant chunk. Set `SIMILARITY_THRESHOLD`
   to separate them. **Write the chosen value and the supporting scores into
   `eval/eval_report.md`.** If 0.35 already separates them cleanly, keep 0.35 and say so.
5. `tests/test_retrieve.py` — alias resolution, the one-directional plan rule, `n_a` inclusion,
   threshold filtering.

### Probe set (must be in the repo as `eval/probe_facts.jsonl`)

10 facts, each with the expected `chunk_id`, `scheme`, and `plan`:

| # | Fact | Expected plan |
|---|---|---|
| 1 | Large Cap — expense ratio | direct |
| 2 | Large Cap — expense ratio | regular |
| 3 | Large Cap — exit load | regular |
| 4 | Flexi Cap — expense ratio | direct |
| 5 | Flexi Cap — exit load | regular |
| 6 | Flexi Cap — minimum SIP | direct |
| 7 | ELSS — expense ratio | regular |
| 8 | ELSS — lock-in period | n_a |
| 9 | ELSS — benchmark | direct |
| 10 | Midcap — minimum lump sum | direct |

### Acceptance gate

```bash
python -m src.retrieve "What is the expense ratio of the Axis Flexi Cap Direct Growth plan?"
python -m pytest tests/ -q
```

| # | Criterion |
|---|---|
| 1 | All 10 probe facts retrieve their expected chunk in the **top 5** |
| 2 | For probe 1 vs. 2 (same scheme, different plans), the retrieved chunks differ in `plan` |
| 3 | No probe returns a chunk whose `plan` contradicts the question |
| 4 | An out-of-corpus query ("Axis Contra Fund") returns **0 hits** at the calibrated threshold |
| 5 | Threshold calibration value + scores written to `eval/eval_report.md` |
| 6 | `pytest` green |

---

# P5 — Generate

**Goal:** a grounded, ≤ 3-sentence answer with a server-injected citation.

**Reference:** `architecture.md` §5.4, §5.5, §5.6, §5.8

### Tasks

1. `src/prompts.py` (or the `PROMPT` constant in `templates.py`) — the template from
   `architecture.md` §5.5, **including the two additions**: rule 5 (*do not write any URL — citations
   are added automatically*) and rule 6 (*copy every number exactly; never round or reformat*).
2. `src/generate.py` — `answer`, `build_context`, `cap_sentences`.
   - `build_context`: number chunks, attach `(scheme, plan, doc_type, page)` labels, truncate at
     `MAX_CONTEXT_CHARS` = 6000, always keeping chunk 1.
   - Groq call: `temperature=0.0`, `max_tokens=220`, `stream=True`, model `llama-3.1-8b-instant`.
     The client must be **injectable** so tests and the eval harness can replay fixtures.
   - `cap_sentences`: split on `(?<=[.!?])\s+`, keep ≤ 3, ensure trailing punctuation, return
     `(text, was_truncated)` so truncation is measurable (G4).
   - `assemble`: handle `NOT_FOUND` / `NO_PERF` sentinels; **inject the citation URL from
     `hits[0].metadata["source_url"]`, never from LLM output** (AD-3); pick the citation using
     `CITATION_SCORE_FLOOR`; compute `last_updated` as the **max** `source_date` among cited chunks;
     when null, render `Last updated from sources: date not stated on page`.
   - Emits one `TraceRecord` per call.
3. `src/guardrails.py` — `verify_output` only (input screening lands in P6).
   Checks per `architecture.md` §5.8: sentence count · advice phrasing · performance claim ·
   **every numeric token in the answer body appears in the retrieved context** · citation URL is in
   `hits`. Numeric comparison must normalize currency symbols, whitespace, and unicode so
   `Rs. 500` matches `Rs 500`. **Apply to the body only**, never the injected citation or date line (A-7).
4. `tests/test_generate.py` — `cap_sentences`, context building, sentinel handling, citation injection,
   `numeric_tokens` normalization. All with a fake LLM client — **no network, no key**.

### Acceptance gate

Requires `GROQ_API_KEY` in `.env`.

```bash
python -m src.generate "What is the exit load on the Axis Large Cap Regular Growth plan?"
python -m pytest tests/ -q
```

| # | Criterion |
|---|---|
| 1 | Answer is ≤ 3 sentences |
| 2 | Exactly one citation, and its URL appears in `hits` metadata |
| 3 | `Last updated from sources:` line present (or the explicit `date not stated on page` variant) |
| 4 | Every number in the body appears verbatim in the retrieved context |
| 5 | `Last updated` uses the **max** `source_date` among cited chunks |
| 6 | LLM output contains no URL (rule 5) — assert in a test |
| 7 | Trace emitted with all `stage_ms` populated |
| 8 | p95 latency budget from `architecture.md` §6.4 measured |
| 9 | `pytest` green |

---

# P6 — Guardrails + UI

**Goal:** every out-of-bounds path refuses safely, and the "tiny UI" the brief asks for exists.

**Reference:** `architecture.md` §5.1, §5.2 · `PRD.md` §6 (FR-U*, FR-G*), §11

### Tasks

1. `src/guardrails.py` — `screen_input`.
   - **Order matters:** PII first and unconditional, so no identifier ever reaches a model (FR-G4, R12).
   - PII regexes: PAN `\b[A-Z]{5}\d{4}[A-Z]\b` · Aadhaar `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b` ·
     account/folio `(?i)(a/?c|account|folio)\s*(no|number)?\s*[:.#]?\s*\d{9,18}` ·
     OTP `(?i)(otp|verification code|pin)\s*[:.#]?\s*\d{4,6}` · email · Indian mobile
     `(?:\+?91[\s-]?)?[6-9]\d{9}`.
     **Account and OTP patterns are context-gated, not bare digit runs (A-4)** — a bare `\d{9,18}`
     would refuse "minimum investment Rs 5000" and blow the ≤ 10% false-refusal budget.
   - **Advice checked before performance (A-3).**
   - Trigger lists verbatim from `architecture.md` §5.2.2.
   - On a PII hit: log `{"pii_blocked": true, "pattern": "<name>"}` — **never the question**.
   - Ambiguous trigger → optional LLM classifier (`GROQ_CLASSIFIER_MODEL`, cheap model,
     `FACTUAL | ADVICE | PERFORMANCE`). Retain only if P4/P7 shows it is needed.
2. `src/app.py` — Streamlit.
   - Welcome line naming Axis MF + the 4 schemes (FR-U1).
   - **Exactly 3** clickable example questions from `PRD.md` §18.1 (FR-U2) — assert the count is 3.
   - `DISCLAIMER` rendered from `templates.py`, never retyped (FR-O4, FR-U3).
   - `handle_question` → `AnswerEnvelope` → render. The UI **never** inspects raw LLM output.
   - Clear distinct visual states for `NOT_FOUND`, `NO_RETRIEVAL`, `REFUSE_ADVICE`,
     `REFUSE_PERFORMANCE`, `PII_BLOCKED` (FR-U5).
   - **No session history, no cookies, no logging of question text** (D9, FR-G5).
   - Startup: load config → assert key present → open `PersistentClient` → assert collection exists →
     **assert embedder fingerprint matches `ingest_state.json`** → cache `Embedder` and collection as
     module-level singletons (Streamlit reruns the script on every interaction; without this, every
     keystroke reloads the model).
3. `tests/test_guardrails.py` — parametrized: every PII regex with a positive and a **negative** case;
   every trigger word; `verify_output` against advice, performance, and unsupported-numeric answers.
   All pure, no network.

### Acceptance gate

```bash
streamlit run src/app.py
python -m pytest tests/ -q
```

Manual — run each and confirm the exact behaviour:

| # | Input | Expected |
|---|---|---|
| 1 | "What is the exit load on the Axis Large Cap Regular Growth plan?" | Cited answer, ≤ 3 sentences |
| 2 | "Should I invest in the Axis ELSS Tax Saver Fund?" | `REFUSE_ADVICE` + educational link; **no Groq call** |
| 3 | "Which Axis fund performed best last year?" | `REFUSE_PERFORMANCE` + factsheet link |
| 4 | "My PAN is ABCDE1234F — what is the exit load?" | `PII_BLOCKED`; **no Groq call**; PAN not in logs |
| 5 | "What is the expense ratio of the Axis Contra Fund?" | `NOT_FOUND` / `NO_RETRIEVAL`, no fabricated number |
| 6 | "Minimum investment of Rs 5000?" (digits, no PII) | **Not blocked** — proves A-4 works |
| 7 | UI shows exactly 3 examples | count = 3 |
| 8 | Disclaimer visible | matches `PRD.md` §11.5 verbatim |
| 9 | `pytest` green | — |

---

# P7 — Eval + deliverables

**Goal:** prove the PRD §14 metrics, and assemble all 8 deliverables.

**Reference:** `PRD.md` §12, §14 · `architecture.md` §9

### Tasks

1. `eval/gold_set.jsonl` — 3 partitions: **40 factual** (all 4 schemes, both plans, all fact types from
   `PRD.md` §5.2, ≥ 6 near-miss traps), **12 refusal**, **8 safety**. Each factual item carries the
   expected `chunk_id`.
2. `eval/run_eval.py` — run the gold set; per item record retrieved chunk ids + scores, raw reply,
   final envelope, `stage_ms`, and per-metric pass/fail. **Retrieval metrics must compute without a
   Groq key** so chunking/retrieval quality is measurable in isolation.
   Emit `eval/eval_report.md` and a machine-readable `eval/results.json`.
3. Compute every `PRD.md` §14.2 metric:
   Recall@5 ≥ 0.85 · MRR@5 ≥ 0.70 · citation present & resolvable 100% · citation **correctness** 100% ·
   ≤ 3 sentences 100% · refusal rate 100% · false-refusal ≤ 10% · PII blocked with 0 LLM calls 100% ·
   unsupported-claim rate 0 (audit 50) · cross-plan contamination 0 · p95 ≤ 5 s.
4. `README.md` — setup steps, scope (AMC + schemes), architecture summary, **chunking rationale**,
   source-conflict precedence rule (`architecture.md` §13 A-8), and **known limits** including every
   fetch failure from P1 and every `suspected_client_rendered` source.
5. `sample_qa.md` — 5–10 real queries with the assistant's actual answers and links.
6. `sources.md` — human-readable mirror of `sources.csv`, with `s15` marked excluded and why.
7. `disclaimer.md` — the single disclaimer text, matching `templates.DISCLAIMER` and `PRD.md` §11.5.
8. A ≤ 3-minute demo script or video, if hosting is not possible (`PRD.md` §12.1).

### Acceptance gate

| # | Criterion |
|---|---|
| 1 | Every **Must** metric in `PRD.md` §14.2 met |
| 2 | Cross-plan contamination = **0** |
| 3 | All 8 deliverables in `PRD.md` §12 present |
| 4 | A clean clone runs per the README steps alone |
| 5 | `pytest` green, no key required for `tests/` |
| 6 | README lists known limits truthfully, including P1 failures |
| 7 | Phase Log (Appendix A) complete with every gate result |

---

## Appendix A — Phase log

Fill this in as you go. This is the audit trail for the milestone.

| Phase | Gate result | Date | Verified by | Notes |
|---|---|---|---|---|
| P0 | ☐ pass ☐ fail | | | |
| P1 | ☐ pass ☐ fail | | | sources flagged: |
| **P2** | ☐ pass ☐ fail | | | **H1–H6 reviewed by:** |
| P3 | ☐ pass ☐ fail | | | threshold/fingerprint: |
| P4 | ☐ pass ☐ fail | | | calibrated threshold: |
| P5 | ☐ pass ☐ fail | | | p95 latency: |
| P6 | ☐ pass ☐ fail | | | false-refusal: |
| P7 | ☐ pass ☐ fail | | | Recall@5: |

---

## Appendix B — Ready-to-paste agent prompts

Each prompt is self-contained. Paste one per session.

### P0

```
Read PRD.md §18.4, architecture.md §3.2/§6.1/§6.2/§10, and implementation.md "Global conventions".

Implement P0 (Scaffold) only. Create:
- full directory skeleton from PRD.md §18.4, plus src/__init__.py and tests/__init__.py
- requirements.txt: only the libraries in implementation.md "Dependency budget", all versions pinned
- .gitignore: .env, data/, chroma/, artifacts/, __pycache__/, *.pyc, .venv/, models/
- .env.example: GROQ_API_KEY= (empty) and AXIS_TOP_K=5
- config.py: every key from architecture.md §6.1, AXIS_ env overrides, and an import-time
  assertion that PII_LOG_RAW is False
- src/models.py: Source, FetchResult, Block, Chunk, Hit, ScreenResult, VerifyResult,
  AnswerEnvelope, IngestReport dataclasses (no ML imports in this module)
- src/sources.py: Source loader for sources.csv with validation and the enabled filter
- sources.csv: all 19 rows transcribed verbatim from PRD.md §5.1, with s15 (indmoney.com)
  set enabled=false per decision D5
- src/trace.py: TraceRecord dataclass and JSON-line emitter per architecture.md §6.3
- src/healthcheck.py: prints dependency versions, config values, key_present as a boolean
  only (never the key value), source counts by scheme, and whether the Chroma collection exists
- tests/test_config.py and tests/test_sources.py

Rules: no comments in code (docstrings on public functions only); every tunable from config.py;
type hints everywhere; never read or log the key value.

Do NOT create extract.py, chunk.py, embed.py, ingest.py, retrieve.py, generate.py, guardrails.py,
app.py, or anything importing sentence_transformers or chromadb.

When done, run: python -m src.healthcheck, python -m pytest tests/ -q,
git check-ignore .env data/ chroma/, and report the actual output of each.
```

### P1

```
Read implementation.md P1, architecture.md §4.1/§4.2/§4.3/§7.

Implement P1 (Corpus fetch + extract) only, assuming P0 exists.

Create:
- src/fetch.py: fetch_source(source, force) and fetch_all(sources, force). httpx with a browser
  User-Agent, timeout 30s, 3 retries with exponential backoff, 1.5s delay between requests.
  Save bytes to data/raw/<source_id>/<original_filename> (synthesise a filename from the URL
  path if absent). Return FetchResult with fetched_at, http_status, content_hash (sha256 of
  bytes), byte_size. On failure record the failure and CONTINUE — never raise and never abort
  the run.
- src/extract.py: extract_blocks(path, doc_type) -> list[Block].
  HTML: trafilatura.extract(output_format="xml"); parse <head> into heading_path, <table> into a
  table block, <p>/<li> into text blocks. Fallback to beautifulsoup4 + lxml.
  PDF: pymupdf page by page; heading detection via font size above body median or a
  ^\d+(\.\d+)*\s+\S numbering pattern; table detection via ruling lines or column clustering.
  Block = {kind, text, heading_path, page_num, table_rows}, kind in
  {heading, text, list, table}.
  Write data/processed/<source_id>.blocks.json.
- artifacts/ingest_report.md: a markdown table with columns
  source_id | scheme | status | http | bytes | extracted_chars | flag, where flag is one of
  ok, fetch_failed, suspected_client_rendered, extract_failed. Flag a source as
  suspected_client_rendered when extracted_chars < config.MIN_EXTRACTED_CHARS (2000).
- tests/test_extract.py using small fixtures in tests/fixtures/

Rules: config.py values only (MIN_EXTRACTED_CHARS=2000, HTTP_TIMEOUT=30, HTTP_RETRIES=3,
REQUEST_DELAY_S=1.5). No silent exception swallowing. No PII handling here.

Run: python -m src.fetch, python -m src.extract, python -m pytest tests/ -q
Then STOP and report the full ingest_report.md contents, listing every fetch_failed and
suspected_client_rendered source. Do not proceed to P2.
```

### P2  ·  the gate phase

```
Read PRD.md §9 in full (it is authoritative for this phase) and implementation.md P2,
architecture.md §4.4/§4.5/§4.6/§4.7.

Implement P2 (Normalize + chunk) only.

Create:
- src/normalize.py: normalize_blocks(blocks) and drop_repeated_pdf_lines(pages, threshold=0.5),
  applying in this exact order: (1) Unicode NFC; (2) NBSP to space, remove soft hyphens;
  (3) de-hyphenate the pattern (\w)-\n(\w) -> \1\2; (4) collapse 3+ blank lines to 2 and
  collapse runs of spaces/tabs to one; (5) drop lines recurring on >50% of pages that match
  page \d+|axis mutual fund|www\.|^\d+$; (6) drop HTML regions matching nav, footer,
  [class*=cookie], [class*=breadcrumb], [class*=cta]; (7) drop orphan numeric-only lines.
- src/chunk.py: chunk_document(blocks, source, cfg), flatten_table(rows, heading_path),
  write_chunks_txt(chunks, path). Implement the three tiers from architecture.md §4.5 exactly:
  TIER 1 - every table block is atomic (never split), flattened by flatten_table, overlap 0,
  emitted with is_table=True. TIER 2 - each heading-bounded section whose body is <=
  MAX_CHUNK_CHARS (900) becomes one chunk with heading_path prepended to the text.
  TIER 3 - a section longer than 900 chars goes through recursive_split with separators
  ["\n\n", "\n", sentence boundary (?<=[.!?])\s+, " "] and CHUNK_OVERLAP of 150.
  Finally merge_short_chunks folds any chunk shorter than MIN_CHUNK_CHARS (120) into
  its neighbour.
- flatten_table must expand an empty first-column cell into its column header, so the matrix
  (blank | Direct Growth | Regular Growth / Expense ratio | 0.95% | 1.95%) produces
  "Fees > Expense ratio (Direct Growth): 0.95%" and
  "Fees > Expense ratio (Regular Growth): 1.95%".
- artifacts/chunks.json per architecture.md §4.6 with every metadata field listed there.
- artifacts/chunks.txt per architecture.md §4.7, PLUS an appended quality summary block
  giving per-scheme counts, per-plan counts, the chunk-length histogram against the 400-900
  median target, and the count of chunks that lack a recognisable scheme or plan.
- tests/test_chunk.py covering: a table is never split; heading_path is prepended; Tier 3
  produces 150-char overlap; short chunks are merged; flatten_table output for the matrix above.

Rules: config.py values only (MAX_CHUNK_CHARS=900, MIN_CHUNK_CHARS=120, CHUNK_OVERLAP=150,
TABLE_OVERLAP=0). No comments in code.

Do NOT create embed.py, ingest.py, or anything importing sentence_transformers or chromadb.
The manual review gate in implementation.md P2 comes first.

Run: python -m src.normalize, python -m src.chunk, python -m pytest tests/ -q
Then STOP and report the chunk count, the length histogram, the per-scheme and per-plan counts,
and the count of chunks failing the scheme/plan criterion. Tell me to review
artifacts/chunks.txt before proceeding to P3.
```

### P3

```
Read implementation.md P3 and architecture.md §4.8/§4.9.

Implement P3 (Embed + store) only, assuming the P2 gate passed.

Create:
- src/embed.py: Embedder class wrapping sentence-transformers/all-MiniLM-L6-v2 with
  normalize_embeddings=True, batch size 32, and a fingerprint property returning
  "<model>@<version>:<dim>:<normalize_flag>". EMBED_DIM is 384.
- src/ingest.py: ingest(only_fetch=False, only_embed=False, force=False, prune=False) with a
  CLI exposing --only-fetch, --only-embed, --force and --prune.
  Idempotency exactly as architecture.md §4.8: read artifacts/ingest_state.json; for each enabled
  source compare the raw file sha256 AND the per-chunk content hashes; if unchanged, log
  "unchanged" and perform zero upserts; if changed, collection.delete(where={"source_id": ...})
  then re-chunk, re-embed and upsert only that source. --prune deletes chunks whose source_id is
  no longer enabled. Store the embedder fingerprint in ingest_state.json and warn loudly if it
  differs from the stored one.
  The string that gets embedded is section_path + "\n" + text, so the heading breadcrumb is
  part of the vector.
  Append embedding counts and duration to artifacts/ingest_report.md.
- tests/test_ingest.py: run ingest twice in a tmp_path with a fake embedder and assert the
  chunk-id set is identical and there are no duplicates.

Vector store: chromadb.PersistentClient(path="chroma/"), collection name "axis_mf_faq",
cosine space (pass configuration={"hnsw": {"space": "cosine"}} and also set
metadata={"hnsw:space": "cosine"} for older chromadb compatibility).

Rules: config.py values only. No comments in code. Do not modify already-reviewed chunk logic.

Run: python -m src.ingest, then python -m src.ingest again, then python -m pytest tests/ -q
Report: collection count after run 1, count after run 2, upsert count in run 2 (must be 0),
and the stored fingerprint.
```

### P4

```
Read implementation.md P4 and architecture.md §5.2/§5.3.

Implement P4 (Retrieve) only, assuming P3 exists.

Create:
- src/retrieve.py:
  - retrieve(question, top_k, threshold): embed the question with the same Embedder, then assert
    its fingerprint matches artifacts/ingest_state.json and raise a clear error if not.
    collection.query(n_results=TOP_K, where=filter, include=["documents","metadatas","distances"]).
    Convert with score = 1 - distance (cosine) and drop hits below SIMILARITY_THRESHOLD.
    Return list[Hit].
  - infer_metadata_filter(question): map scheme aliases ("large cap"->large_cap,
    "flexi cap"->flexi_cap, "elss"/"tax saver"->elss, "mid cap"->mid_cap) and plan aliases
    ("direct"->direct, "regular"->regular). If a plan is named, return
    {"$and":[{"scheme":...},{"$or":[{"plan":named},{"plan":"n_a"}]}]}. If NO plan is named,
    apply NO plan filter. If no alias matches at all, return None (no filter).
    plan "n_a" must always be included so SID/KIM fee tables survive the filter.
- src/templates.py: render(template_id, **kwargs) for template ids FACTUAL, NOT_FOUND,
  NO_RETRIEVAL, REFUSE_ADVICE, REFUSE_PERFORMANCE, PII_BLOCKED, and the DISCLAIMER constant
  copied VERBATIM from PRD.md §11.5.
- A CLI entry: python -m src.retrieve "<question>" that prints each hit with score, scheme, plan,
  doc_type, and section_path.
- eval/probe_facts.jsonl: the 10 facts listed in implementation.md P4, each with the expected
  chunk_id, scheme, and plan.
- tests/test_retrieve.py: alias resolution, the one-directional plan rule (no plan named means no
  plan filter), n_a inclusion, and threshold filtering.

Then perform threshold calibration: for each of the 10 probe facts record the score of the
correct chunk and the score of the top-scoring irrelevant chunk, choose a SIMILARITY_THRESHOLD
that separates them, update config.py, and write the chosen value with the supporting score
table into eval/eval_report.md.

Rules: config.py values only (TOP_K=5, SIMILARITY_THRESHOLD, CITATION_SCORE_FLOOR=0.45).
No comments in code.

Run: python -m src.retrieve "What is the expense ratio of the Axis Flexi Cap Direct Growth plan?"
and python -m pytest tests/ -q
Report: the calibration table, the final threshold, and whether all 10 probe facts appear in
the top 5 with the correct scheme and plan.
```

### P5

```
Read implementation.md P5 and architecture.md §5.4/§5.5/§5.6/§5.8/§6.3/§6.4.

Implement P5 (Generate) only, assuming P4 exists. Requires GROQ_API_KEY in .env.

Create or extend:
- templates.py: add the PROMPT constant exactly as architecture.md §5.5, INCLUDING rule 5
  ("Do NOT write any URL or link. Citations are added automatically after your reply.")
  and rule 6 ("Copy every number, rate, and date exactly as written in the CONTEXT. Never round,
  convert, reformat, or infer a figure.").
- src/generate.py:
  - build_context(hits, max_chars): number the chunks, label each with
    (scheme, plan, doc_type, page), truncate at MAX_CONTEXT_CHARS=6000 but always keep chunk 1.
  - answer(question, hits, cfg): build the prompt, call Groq with temperature=0.0,
    max_tokens=220, stream=True, model GROQ_MODEL. The Groq client must be injectable so tests
    can replay fixtures. Handle the NOT_FOUND and NO_PERF sentinels. Then assemble the
    AnswerEnvelope: cap the body to MAX 3 sentences via cap_sentences (returning
    (text, was_truncated)); inject citation_url from hits[0].metadata["source_url"] —
    NEVER from LLM output; select the cited hit using CITATION_SCORE_FLOOR; set last_updated to
    the MAX source_date among cited chunks, or None. Emit one TraceRecord per call.
- src/guardrails.py: verify_output only for now. Check, in order: sentence count <= 3; no advice
  phrasing ("you should", "I recommend", "suitable for you", "consider investing",
  "I would suggest"); no return/performance claim; every numeric token in the ANSWER BODY
  (never the injected citation or the Last-updated line) appears in the retrieved context,
  comparing on normalised forms so "Rs. 500" matches "Rs 500"; and citation_url is present in
  hits. Provide numeric_tokens(text) -> set[str] for this normalisation.
- tests/test_generate.py using a fake LLM client (no network, no key): cap_sentences boundaries,
  context building and truncation, sentinel handling, citation injection from metadata, and
  numeric_tokens normalisation cases.

Rules: config.py values only. No comments in code.

Run: python -m src.generate "What is the exit load on the Axis Large Cap Regular Growth plan?"
and python -m pytest tests/ -q
Report: the full answer envelope (answer, citation_url, chunk_id, last_updated, sentence_count,
guardrail_trace, stage_ms).
```

### P6

```
Read implementation.md P6, architecture.md §5.1/§5.2/§6.3/§7.1, and PRD.md §11.

Implement P6 (Guardrails + UI) only, assuming P5 exists.

Extend src/guardrails.py:
- screen_input(question) -> ScreenResult. The ORDER IS CRITICAL.
  1. PII first and unconditional. Regexes:
     PAN           \b[A-Z]{5}\d{4}[A-Z]\b
     Aadhaar       \b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b
     account/folio (?i)(a/?c|account|folio)\s*(no|number)?\s*[:.#]?\s*\d{9,18}
     OTP           (?i)(otp|verification code|pin)\s*[:.#]?\s*\d{4,6}
     email         standard email regex
     Indian mobile (?:\+?91[\s-]?)?[6-9]\d{9}
     The account and OTP patterns MUST be context-gated, never bare digit runs.
  2. Then ADVICE_TRIGGERS.
  3. Then PERF_TRIGGERS.
  4. Otherwise FACTUAL.
  Trigger lists are exactly those in architecture.md §5.2.2.
  On a PII hit, log ONLY {"pii_blocked": true, "pattern": "<name>"} — never the question text.
- Create src/app.py (Streamlit):
  - startup: load config, assert GROQ_API_KEY is present, open PersistentClient, assert the
    collection exists, assert the embedder fingerprint matches ingest_state.json, then cache
    Embedder and collection as module-level singletons (Streamlit reruns the script on every
    interaction, so without caching the model reloads each time).
  - welcome line naming Axis Mutual Fund and the four schemes
  - EXACTLY 3 clickable example questions from PRD.md §18.1
  - DISCLAIMER rendered from templates.DISCLAIMER, never retyped
  - handle_question -> AnswerEnvelope -> render answer, citation link, and Last-updated line
  - distinct visual states for NOT_FOUND, NO_RETRIEVAL, REFUSE_ADVICE, REFUSE_PERFORMANCE,
    PII_BLOCKED
  - NO session history, NO cookies, NO logging of question text
- tests/test_guardrails.py: parametrized positive AND negative cases for every PII regex (the
  negatives must include "minimum investment of Rs 5000" and "exit load 1.00%"), every trigger
  word, and verify_output against advice, performance and unsupported-numeric answers.

Run: python -m pytest tests/ -q, then streamlit run src/app.py
Report: pytest output, then the result of each of the 6 manual acceptance cases in
implementation.md P6, stating explicitly which ones made no Groq call.
```

### P7

```
Read implementation.md P7, PRD.md §12 and §14, and architecture.md §9.

Implement P7 (Eval + deliverables) only, assuming P0-P6 are green.

Create:
- eval/gold_set.jsonl with three partitions: 40 factual (all 4 schemes, both plan variants, every
  fact type in PRD.md §5.2, and at least 6 near-miss traps such as "What is the exit load on the
  Flexi Cap Direct Growth plan?"), 12 refusal, 8 safety. Each factual item carries its expected
  chunk_id.
- eval/run_eval.py: runs the gold set and records, per item, the retrieved chunk ids and scores,
  the raw reply, the final AnswerEnvelope, stage_ms, and pass/fail per metric. Retrieval metrics
  (Recall@5, MRR@5) MUST be computable without a Groq key. Emit eval/eval_report.md and
  eval/results.json.
- Compute every metric in PRD.md §14.2 and report each against its target.
- README.md: setup steps, scope (AMC + schemes), an architecture summary, the chunking rationale
  from PRD.md §9, the source-conflict precedence rule from architecture.md §13 A-8, and known
  limits — including every fetch_failed and suspected_client_rendered source from the P1 report.
- sample_qa.md: 5 to 10 real queries with the assistant's actual answers and links.
- sources.md: human-readable mirror of sources.csv, marking s15 as excluded and why.
- disclaimer.md: matching templates.DISCLAIMER and PRD.md §11.5 verbatim.

Also append the completed phase-gate results table to implementation.md Appendix A.

Report: the full metric table with target vs actual, every metric that missed its target, and a
list of the 8 PRD.md §12 deliverables with a pass/fail for each.
```

---

## Appendix C — Reference

| Document | Contains |
|---|---|
| `PRD.md` | Goals, corpus, FR/NFR, chunking strategy rationale, guardrails, eval plan, risks |
| `architecture.md` | Module design, data contracts, pipeline internals, interfaces, failure modes |
| `docs/Problem_statement.txt` | The original brief — authoritative for scope |
| **this file** | Build order, per-phase tasks, acceptance gates, agent prompts |

**Authority order when documents disagree:** `docs/Problem_statement.txt` → `PRD.md` → `architecture.md`
→ this file. Stop and ask rather than silently resolving a conflict.