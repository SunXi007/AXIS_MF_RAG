# Architecture — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Project:** `nextleap_M4_RAGchatbot`
**Derived from:** [`PRD.md`](PRD.md) (v1, 2026-10-02)
**Status:** Draft for review
**Last updated:** 2026-10-02

> This document specifies **how** the system is built. **What** it must do is defined in `PRD.md`.
> Every design decision here traces to a PRD requirement ID (`FR-*`, `NFR-*`, `G*`, `D*`). Section 12 is
> the full traceability matrix. Where this document proposes something the PRD does not mandate, it is
> flagged **[ADDITION]** and carries its own decision ID (`A*`) in §13.

---

## Table of Contents

1. [Architecture drivers](#1-architecture-drivers)
2. [System context](#2-system-context)
3. [Module map](#3-module-map)
4. [Ingestion pipeline](#4-ingestion-pipeline)
5. [Query pipeline](#5-query-pipeline)
6. [Cross-cutting concerns](#6-cross-cutting-concerns)
7. [Module interfaces](#7-module-interfaces)
8. [Worked traces](#8-worked-traces)
9. [Testing strategy](#9-testing-strategy)
10. [Run modes & operations](#10-run-modes--operations)
11. [Failure modes & degradation](#11-failure-modes--degradation)
12. [Traceability matrix](#12-traceability-matrix)
13. [Architecture decision log](#13-architecture-decision-log)

---

## 1. Architecture drivers

Five constraints do almost all the shaping work. Everything below follows from them.

| # | Driver | Architectural consequence |
|---|---|---|
| **AD-1** | The corpus is **label-dense and tabular** — fee/exit-load/SIP facts live in small two-column tables in HTML and in dense multi-page PDFs | Chunking must be **structure-aware**, not fixed-window. Tables are atomic. (§4.5) |
| **AD-2** | **Direct vs. Regular plans carry different numbers**, so plan identity is part of a fact's identity | `plan` is a first-class metadata field, never inferred at query time. Metadata filters + a zero-tolerance eval metric. (§4.6, §5.3) |
| **AD-3** | Every answer must be **citeable** | The citation URL is a *retrieval artifact*, not LLM output. It is injected server-side from chunk metadata after generation, so the model can never invent a link. (§5.6) |
| **AD-4** | Zero-tolerance guardrails (advice, performance, PII) but **p95 ≤ 5 s** | Two-stage guardrails: cheap regex first, LLM classifier only on ambiguity. Citations and the `Last updated` line are computed, not generated. (§5.2, §6.7) |
| **AD-5** | **Ingestion is expensive, queries are cheap** | Strict offline/online split. Nothing in the online path touches the network except the Groq call. Embeddings run once, on disk. (NFR-2, NFR-5) |

**Explicitly rejected alternatives:**

| Rejected | Why |
|---|---|
| Framework-managed ingestion (LangChain/LlamaIndex default splitters) | Their default chunkers are fixed-window; they cannot keep a fee table intact (AD-1). We need custom extractors anyway for page-aware PDF text. |
| LLM-generated citations | Directly violates AD-3 — a hallucinated URL is an uncitable, untraceable answer. |
| Single-stage LLM guardrail | Costs a full LLM round-trip on every query against the 5 s budget (AD-4). |
| Agentic / multi-hop retrieval | The fact types in PRD §5.2 are single-hop lookups. Multi-hop adds latency, cost, and non-determinism for no gain. |
| Persisting conversation history | Expands the PII surface for zero benefit in a facts-only, stateless assistant (NFR-3, D9). |

---

## 2. System context

```
┌───────────────────────────────────────────────────────────────────────────────┐
│                          AUTHORING TIME (build machine)                       │
│                                                                               │
│   sources.csv  ──►  fetch ──► extract ──► normalize ──► chunk ──► EMBED      │
│                     │                                     │                    │
│                     ▼                                     ▼                    │
│              data/raw/<id>/                        artifacts/chunks.txt      │
│                                                    artifacts/chunks.json     │
│                                                       ▲ MANUAL GATE (PRD9.5) │
│                                                                               │
│   Operator reviews chunks.txt ──sign-off──►  (embedding code may now be built)│
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                            RUN TIME (any machine)                             │
│                                                                               │
│   ┌─────────┐   question    ┌──────────────────────────────────┐              │
│   │ Browser │──────────────►│  Streamlit UI (FR-U1..U5)        │              │
│   └─────────┘◄──────────────│  stateless, no session storage   │              │
│                             └──────────────┬───────────────────┘              │
│                                            │ AnswerEnvelope                  │
│                                            ▼                                  │
│                                    ┌───────────────────┐                     │
│                                    │  RAG Service      │                     │
│                                    │  guardrail→retrieve│                     │
│                                    │  →generate→verify  │                     │
│                                    └──┬─────────┬──────┘                     │
│                                       │         │                             │
│                        reads only ────┘         └──── HTTPS ────┐            │
│                              │                                     │            │
│                              ▼                                     ▼            │
│                     ┌────────────────┐                   ┌──────────────┐      │
│                     │ chroma/ (disk) │                   │ Groq API     │      │
│                     │ + all-MiniLM-  │                   │ key from .env│      │
│                     │   L6-v2 (local)│                   └──────────────┘      │
│                     └────────────────┘                                          │
└───────────────────────────────────────────────────────────────────────────────┘

     External, read-only, offline:   www.axismf.com  ·  transact.axismf.com
                                     (ingestion only — never touched at query time)
```

**Trust boundaries:**

| Boundary | Data crossing | Rule |
|---|---|---|
| B1 User → Service | Question text | Untrusted. PII-screened before any model call (FR-G4). Never persisted (FR-G5). |
| B2 Service → Groq | Prompt containing retrieved chunk text only | The **only** outbound data path at query time. No user identifiers — they are stripped or the call is aborted at B1. |
| B3 Service → Chroma | Embedding vectors | Local process, on-disk. |
| B4 Fetcher → Axis | HTTP GET | Outbound only during ingestion. Read-only; no auth, no cookies, rate-limited. |

---

## 3. Module map

### 3.1 Runtime dependency graph

```
                       config.py
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    fetch.py          extract.py       guardrails.py
        │                 │              │      │
        └──► normalize.py ◄┘              │      │
                 │                       │      │
                 ▼                       │      │
              chunk.py                   │      │
                 │                       │      │
                 ▼                       │      │
              embed.py                   │      │
                 │                       │      │
                 ▼                       │      │
            ingest.py ──► chroma/        │      │
                                         │      │
   ┌─────────────────────────────────────┘      │
   │                                 retrieve.py │
   │                                     │      │
   │                                 templates.py
   │                                     │      │
   │                                 generate.py
   │                                     │      │
   │                                     ▼      │
   │                                 app.py ◄────┘
   │
   └──► eval/run_eval.py  (consumes artifacts/ + the same service path)
```

**Acyclic by design.** `fetch → extract → normalize → chunk → embed → ingest` is a strict pipeline;
`guardrails → retrieve → generate → app` is the request path. Nothing in the request path imports
anything from the ingestion path except `config` and `templates`.

### 3.2 File responsibilities

| File | Owns | Must not |
|---|---|---|
| `config.py` | All tunables: paths, chunk params, top-k, threshold, model IDs, collection name | Import any other project module (else it can't be imported first) |
| `sources.py` | `sources.csv` load/validate, source metadata, doc-type + plan inference from URL | Perform I/O beyond reading `sources.csv` |
| `fetch.py` | HTTP GET, retry/backoff, UA rotation, raw persistence, `fetched_at`, `content_hash` | Parse or interpret document content |
| `extract.py` | HTML→text (`trafilatura`), PDF→page-tagged text (`pymupdf`), block+heading+table model | Normalize or chunk |
| `normalize.py` | Whitespace, unicode, de-hyphenation, boilerplate removal, line-frequency PDF header detection | Split semantically |
| `chunk.py` | Three-tier structure-aware splitting, table flattening, metadata assignment, `chunks.txt`/`chunks.json` emission | Embed or store |
| `embed.py` | `all-MiniLM-L6-v2` load, mean-pool, L2-normalize, batch encode, model fingerprint | Choose what to embed |
| `ingest.py` | Orchestrator, idempotency, Chroma upsert, ingestion report | Contain chunking or embedding logic |
| `retrieve.py` | Query embed, top-k cosine search, threshold, metadata filters, trace record | Call the LLM |
| `guardrails.py` | PII regex screen, advice/performance classifier, output verifier, redaction | Render UI |
| `templates.py` | Response templates + disclaimer — **single source of truth** (FR-O4) | Import anything but `config` |
| `generate.py` | Prompt construction, Groq call, citation injection, `Last updated` line, sentence cap | Decide policy (that is `guardrails`) |
| `app.py` | Streamlit surface, welcome, 3 examples, disclaimer, answer rendering | Contain any generation or retrieval logic |
| `eval/run_eval.py` | Gold-set runner, metric computation, report emission | Share mutable state with `app.py` |

### 3.3 Where each PRD requirement lives

| Concern | Module |
|---|---|
| FR-I1 | `fetch.py` |
| FR-I2, FR-I3 | `extract.py`, `normalize.py` |
| FR-I4 | `chunk.py` → `artifacts/chunks.txt` |
| FR-I5, FR-I6 | `embed.py`, `ingest.py` |
| FR-I7, FR-I8 | `sources.py`, `fetch.py`, `ingest.py` (report) |
| FR-I9 | `ingest.py` CLI flags |
| FR-R1–R4 | `retrieve.py` |
| FR-R5, FR-R6 | `generate.py` |
| FR-R7, FR-R8 | `generate.py` (injects from chunk metadata) |
| FR-R9 | `generate.py` + `templates.py` |
| FR-G1–G4, FR-G6 | `guardrails.py` |
| FR-G5 | `guardrails.py` (redaction) + `app.py` (no session state) |
| FR-G7 | `generate.py` prompt |
| FR-U1–U5, FR-O4 | `app.py`, `templates.py` |
| FR-O1 | `.env`, `.gitignore`, `config.py` |

---

## 4. Ingestion pipeline

Runs offline, once per corpus snapshot. Never on the query path.

### 4.1 Stage sequence

```
sources.csv
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. LOAD                                    sources.py           │
│    validate: unique source_id, url present, doc_type in enum,  │
│    scheme in enum, plan in enum                                │
│    apply exclusion policy (D5: drop indmoney.com)              │
└───────────────────────────────────┬─────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. FETCH                                    fetch.py           │
│    httpx, browser UA, timeout 30 s, 3 retries, exp backoff     │
│    → data/raw/<source_id>/<original_filename>                  │
│    → record fetched_at, http_status, content_hash, byte_size   │
│    ON FAILURE: log to ingestion_report, continue (NFR-6)       │
└───────────────────────────────────┬─────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. EXTRACT                                  extract.py         │
│    HTML → trafilatura (fallback bs4+lxml) → Block[]           │
│           Block = {kind, text, heading_path, page_num|null}    │
│    PDF  → pymupdf per page; font-size heuristic for headings;  │
│           table detection via ruling lines / column clustering  │
│    VERIFY: extracted chars > MIN_EXTRACTED_CHARS (2000)        │
│           else flag "suspected client-rendered" (risk R1)      │
└───────────────────────────────────┬─────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. NORMALIZE                              normalize.py         │
│    → data/processed/<source_id>.txt (+ .blocks.json)           │
│    strip nav/footer/cookie/CTA; de-hyphenate; NFC unicode;      │
│    drop PDF lines recurring on > 50% of pages                   │
└───────────────────────────────────┬─────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. CHUNK                                    chunk.py           │
│    three-tier structure-aware split (PRD §9.2)                 │
│    → artifacts/chunks.json                                      │
│    → artifacts/chunks.txt   ◄── M2 SIGN-OFF GATE (PRD §9.5)    │
└───────────────────────────────────┬─────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. EMBED + STORE                      embed.py + ingest.py    │
│    all-MiniLM-L6-v2, 384-dim, L2-normalized                    │
│    → chroma/  (PersistentClient, cosine, axis_mf_faq)           │
│    → artifacts/ingest_report.md                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 `sources.csv` schema

```csv
source_id,scheme,plan,doc_type,url,enabled,notes
s01,large_cap,direct,scheme_page,https://www.axismf.com/mutual-funds/equity-funds/axis-large-cap-fund/ef-dg/direct,true,Direct Growth
s02,large_cap,regular,scheme_page,https://www.axismf.com/mutual-funds/equity-funds/axis-large-cap-fund/ef-gp/regular,true,Regular Growth
s03,large_cap,n_a,sid,https://www.axismf.com/cms/sites/default/files/Statutory/Axis%20Bluechip%20Fund%20-%20SID.pdf,true,Statutory; plan-agnostic
...
s15,midcap,n_a,third_party,https://www.indmoney.com/mutual-funds/axis-midcap-fund-direct-plan-growth,false,EXCLUDED per D5
```

| Field | Type | Rules |
|---|---|---|
| `source_id` | `sNN` | Primary key, stable, never reused |
| `scheme` | enum | `large_cap \| flexi_cap \| elss \| midcap \| amc_wide` |
| `plan` | enum | `direct \| regular \| n_a` — **AD-2**. Statutory docs are `n_a` because they cover all plans |
| `doc_type` | enum | `scheme_page \| sid \| kim \| factsheet \| efactsheet \| downloads \| homepage \| third_party` |
| `url` | str | Verbatim from PRD §5.1. Must match exactly. |
| `enabled` | bool | `false` = loaded into the manifest, never fetched. Used for D5. |
| `notes` | str | Free text; surfaced in the README |

**[ADDITION A-1]** The `enabled` flag is a cleaner expression of D5 than deleting the row: keeping
`s15` visible in `sources.csv` with `enabled=false` *documents the exclusion in the deliverable itself*
(PRD §12.2 requires exclusions to be justified), instead of silently making it disappear.

### 4.3 Extraction strategies per doc type

| doc_type | Tool | Structure signal | Known failure |
|---|---|---|---|
| `scheme_page` | `trafilatura.extract(output_format="xml")` | `<head>` elements → `heading_path`; `<table>` → table block | Fee tables may be lazily hydrated (risk R1). Mitigated by the `MIN_EXTRACTED_CHARS` check + SID fallback. |
| `sid`, `kim` | `pymupdf`, page by page | Font size > body size, or `^\d+(\.\d+)*\s` numbering | Multi-column body text interleaves on extraction; SID fee tables are usually single-column, so impact is low |
| `factsheet` | `pymupdf` | Same as SID; front page has the performance table | **Performance table is Tier-1 atomic and will retrieve for return questions** — that is correct: we answer by *redirecting* to the factsheet (FR-G3), so the table's presence is useful |
| `efactsheet` | `trafilatura` | `<head>`/`<table>` | Dated HTML snapshot; `source_date` comes from the URL path (`January-2025`) |
| `downloads`, `homepage` | `trafilatura` | Link-list extraction | **Procedural content**, not numeric. Must survive chunking as ordered steps (§4.5 Tier 2) |

### 4.4 Normalization rules

Applied in order, each idempotent:

1. Unicode → NFC; normalize NBSP → space, soft hyphen → removed.
2. De-hyphenate: `(\w)-\n(\w)` → `\1\2` (PDF line-wrapping artifact).
3. Collapse 3+ blank lines → 2; collapse runs of spaces/tabs → single space.
4. Strip repeated PDF furniture: build a line-frequency map across pages; drop any line occurring on
   > 50% of pages and matching `page \d+|axis mutual fund|www\.|sid$|^\d+$`.
5. Drop HTML boilerplate regions before text conversion where the DOM exposes them
   (`nav`, `footer`, `[class*=cookie]`, `[class*=breadcrumb]`, `[class*=cta]`).
6. Drop pure-numeric or pure-punctuation orphan lines produced by table extraction.

### 4.5 Three-tier chunking algorithm

Implements PRD §9.2. Operates on the `Block[]` model, not on raw text — this is what makes it
structure-aware.

```
def chunk_document(blocks, source, params):
    chunks = []

    # Tier 1 - atomic tables, never split
    for b in blocks where b.kind == "table":
        chunks.append(make_chunk(
            text        = flatten_table(b.table),        # "Field: Value" lines, §4.5.1
            heading     = b.heading_path,
            atomic      = True,                          # overlap disabled
        ))

    # Tier 2 - heading-bounded sections
    for section in group_by_heading_path(blocks):
        body = join(section.blocks)

        if len(body) <= max_chunk_chars:
            chunks.append(make_chunk(text = prefix_heading(b.heading_path, body)))
            continue

        # Tier 3 - recursive character window inside one over-long section
        for window in recursive_split(body, max_chunk_chars, overlap, separators):
            chunks.append(make_chunk(text = prefix_heading(b.heading_path, window)))

    return merge_short_chunks(chunks, min_chunk_chars)
```

**`recursive_split` separators**, in order: `"\n\n"` → `"\n"` → sentence boundary `(?<=[.!?])\s+` → `" "`.

**`prefix_heading`** prepends the breadcrumb to every chunk — a chunk is retrieved without its siblings,
so it must be self-describing (PRD §9.2).

#### 4.5.1 Table flattening

```python
def flatten_table(rows, heading_path):
    lines = [f"{heading_path}"]
    header = rows[0]
    for row in rows[1:]:
        if len(row) == 2:
            lines.append(f"{normalise_label(row[0], header)}: {clean_cell(row[1])}")
        else:
            label = " | ".join(normalise_label(c, header) for c in row[:-1])
            lines.append(f"{label}: {clean_cell(row[-1])}")
    return "\n".join(lines)
```

`normalise_label` expands an empty first-column cell into its column header, so a matrix like

| | Direct Growth | Regular Growth |
|---|---|---|
| Expense ratio | 0.95% | 1.95% |

becomes two unambiguous lines:

```
Fees > Expense ratio (Direct Growth): 0.95%
Fees > Expense ratio (Regular Growth): 1.95%
```

This is the concrete mechanism by which AD-2 is satisfied: **plan identity is baked into the embedded
text**, so a query for "Direct Growth expense ratio" matches the Direct row and not the Regular one.

#### 4.5.2 Parameters (identical to PRD §9.2 — single source of truth is `config.py`)

| Parameter | Value | Where enforced |
|---|---|---|
| `MAX_CHUNK_CHARS` | 900 | Tier 3 window cap; Tier 1 exempt (atomic) |
| `MIN_CHUNK_CHARS` | 120 | Post-pass merges undersized chunks into their neighbour |
| `CHUNK_OVERLAP` | 150 | Tier 3 only; never applied to Tier 1 |
| `MIN_EXTRACTED_CHARS` | 2000 | Ingest-time sanity check (risk R1) |
| `TABLE_OVERLAP` | 0 | Atomic |

### 4.6 Chunk record schema

Emitted to `artifacts/chunks.json`; one-to-one with the ChromaDB record.

```jsonc
{
  "chunk_id": "s04-c017",
  "text": "Axis Flexi Cap Fund > Direct Growth > Fees\nExpense ratio (Direct Growth): 0.95%",
  "metadata": {
    "source_id": "s04",
    "source_url": "https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/ml-dg/direct",
    "source_title": "Axis Flexi Cap Fund - Direct Growth",
    "doc_type": "scheme_page",
    "scheme": "flexi_cap",
    "plan": "direct",
    "page_num": null,
    "section_path": "Axis Flexi Cap Fund > Direct Growth > Fees",
    "source_date": null,
    "fetched_at": "2026-10-02T09:14:22Z",
    "content_hash": "9f2c1a...e7",
    "chunk_index": 17,
    "is_table": true,
    "char_len": 118
  }
}
```

`chunk_id` format is `<source_id>-c<NNN>` — stable across re-runs, and it is the join key for the
eval harness (PRD §14.3).

### 4.7 `artifacts/chunks.txt` format

The M2 sign-off artifact (FR-I4, G7). Must be readable by a human with no tooling.

```
================================================================================
CHUNK s04-c017  |  index 17  |  118 chars  |  table=true
SECTION   Axis Flexi Cap Fund > Direct Growth > Fees
SCHEME    flexi_cap        PLAN  direct        DOCTYPE  scheme_page
SOURCE    https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/ml-dg/direct
DATE      source_date=<unset>   fetched_at=2026-10-02T09:14:22Z   hash=9f2c1a...e7
--------------------------------------------------------------------------------
Axis Flexi Cap Fund > Direct Growth > Fees
Expense ratio (Direct Growth): 0.95%
================================================================================

<next chunk>
```

**[ADDITION A-2]** `chunks.txt` doubles as the chunking quality report. `ingest.py` appends a summary
block — per-`scheme` and per-`plan` chunk counts, the length histogram against
PRD §9.5 criterion 4, and the count of chunks failing criterion 1 (no identifiable scheme/plan). The
gate can then be evaluated without writing a script.

### 4.8 Idempotency & incremental re-ingest (FR-I6)

```
for source in enabled_sources:
    h_file = sha256(data/raw/<source_id>/<file>)
    h_chunks = {chunk_id: content_hash for chunk in existing_chunks if source_id == source.id}

    if h_file in state and h_chunks match the new run:
        SKIP  (log "unchanged")
    else:
        DELETE  where source_id == source.id        # Chroma: collection.delete(where=...)
        re-chunk → re-embed → upsert
```

State lives in `artifacts/ingest_state.json`. Consequences:

- Re-running `ingest.py` on an unchanged corpus produces **zero** writes and zero duplicates.
- Editing one source's raw file triggers re-embedding of that source **only**.
- Deleting a source from the corpus leaves orphans unless explicitly purged — `ingest.py --prune`
  removes chunks whose `source_id` is no longer `enabled`.

### 4.9 Vector store schema

```python
client   = chromadb.PersistentClient(path="chroma/")
coll     = client.get_or_create_collection(
    name         = "axis_mf_faq",
    configuration = {"hnsw": {"space": "cosine"}},
    metadata     = {"hnsw:space": "cosine"},   # compat with older chromadb
)
coll.upsert(ids=chunk_ids, embeddings=vectors, documents=texts, metadatas=metadatas)
```

| Property | Value | Reason |
|---|---|---|
| Distance | cosine | Matches L2-normalized embeddings; monotonic in dot product (D1) |
| Embedding dim | 384 | `all-MiniLM-L6-v2` native output |
| Embedding model fingerprint | stored in `ingest_state.json` | Changing the model invalidates the index — detected and reported, not silently mixed (NFR-4) |
| Metadata filterable | `scheme`, `plan`, `doc_type`, `source_id`, `page_num`, `source_date` | Powers FR-R4 and the eval harness |

**Embedding input is `section_path + "\n" + text`**, not `text` alone. The breadcrumb is prepended by
`chunk.py` and lands in the embedded string, so the scheme/plan context is part of the vector, not just
the metadata (AD-2).

---

## 5. Query pipeline

Per user turn. Stateless — no memory across turns (D9).

### 5.1 Sequence

```
question
    │
    ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 1 — INPUT GUARDRAIL                             guardrails │
│   1a PII regex screen            (FR-G4) ── HIT ──► PII template   │  ~2 ms
│   1b advice/performance          (FR-G1,G3)                        │
│      regex pre-filter ── HIT ──► refusal template                 │
│      ambiguous ──► LLM classifier (cheap model) ──► verdict       │
└──────────────┬─────────────────────────────────────────────────┘
               │ FACTUAL
               ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 2 — RETRIEVE                                     retrieve  │
│   metadata filter inference (scheme / plan from the question)    │
│   embed question with all-MiniLM-L6-v2            (FR-R1)        │  ~40 ms
│   collection.query(n_results=5, where=filter)      (FR-R2,R4)    │
│   apply SIMILARITY_THRESHOLD                       (FR-R3)       │
└──────────────┬─────────────────────────────────────────────────┘
               │ 0 hits above threshold
               ├──► "not in sources" template  (FR-R9)
               ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 3 — CONTEXT ASSEMBLY                          generate    │
│   number chunks, attach labels, budget to MAX_CONTEXT_CHARS     │  ~3 ms
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 4 — GENERATE                                      generate  │
│   build prompt (templates.PROMPT + numbered context)  (FR-G7)   │
│   groq.chat.completions.create(stream=True)          (FR-R6)    │  ~4 s
└──────────────┬─────────────────────────────────────────────────┘
               ▼
┌────────────────────────────────────────────────────────────────┐
│ STAGE 5 — OUTPUT ASSEMBLY + VERIFICATION              generate  │
│   parse body + sentinel     NOT_FOUND / NO_PERF                 │
│   INJECT citation URL from chunk metadata          (FR-R7)  ◄── AD-3
│   INJECT "Last updated from sources: <max date>"    (FR-R8)
│   enforce ≤ 3 sentences                                    (G4) │
│   verify: advice phrasing / unsourced numerics / URL not in ctx│
│                                            guardrails (FR-G6)  │  ~8 ms
└──────────────┬─────────────────────────────────────────────────┘
               ▼
        AnswerEnvelope  ──►  app.py renders  (FR-U4)
```

### 5.2 Stage 1 — input guardrail decision table

Order matters: PII is checked **first** and unconditionally, so no question containing an identifier
reaches a model (FR-G4, R12).

| # | Condition | Verdict | LLM call? | Response template |
|---|---|---|---|---|
| 1 | Matches any PII regex (§5.2.1) | `PII` | **No** | `PII_BLOCKED` |
| 2 | Matches `ADVICE_TRIGGERS` | `ADVICE` | No | `REFUSE_ADVICE` |
| 3 | Matches `PERF_TRIGGERS` | `PERFORMANCE` | No | `REFUSE_PERFORMANCE` |
| 4 | Pre-filter fires but is ambiguous | → classifier | Yes (cheap) | per verdict |
| 5 | Otherwise | `FACTUAL` | — | proceed to Stage 2 |

**[ADDITION A-3]** Advice is checked **before** performance. A question like *"Should I buy the fund that
performed best?"* is advice-shaped first — refusing it as a performance question would imply we would
otherwise answer it.

#### 5.2.1 PII patterns

| Pattern | Regex | Note |
|---|---|---|
| PAN | `\b[A-Z]{5}\d{4}[A-Z]\b` | Word-boundary anchored; avoids matching ordinary words |
| Aadhaar | `\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b` | |
| Account / folio | `(?i)(a/?c|account|folio)\s*(no|number)?\s*[:.#]?\s*\d{9,18}` | Context-gated, not bare digits |
| OTP | `(?i)(otp|verification code|pin)\s*[:.#]?\s*\d{4,6}` | Context-gated |
| Email | standard email regex | |
| Phone | `(?:\+?91[\s-]?)?[6-9]\d{9}` | 10-digit Indian mobile |

**[ADDITION A-4] Account-number and OTP patterns are context-gated rather than bare digit runs.**
A bare `\d{9,18}` rule would refuse legitimate factual questions — *"What is the minimum investment of
Rs 5000?"* has digits, and a table of exit-load slabs is full of them. Context gating keeps the
false-refusal rate inside the ≤ 10% budget (PRD §14.2).

On a PII hit, the log record is `{"pii_blocked": true, "pattern": "<name>", "at": "<iso>"}` —
**never the question text** (FR-G5, NFR-3).

#### 5.2.2 Trigger lists

```python
ADVICE_TRIGGERS = [
    r"\bshould (i|we)\b", r"\bis it (safe|worth|good|bad)\b", r"\bworth investing\b",
    r"\bwhich (is|one is) better\b", r"\brecommend\b", r"\bsuitable for\b",
    r"\bgood time to\b", r"\b(allocate|allocation)\b", r"\bportfolio should\b",
    r"\bbest (fund|scheme|option)\b", r"\bshould i (buy|sell|switch|exit|redeem)\b",
    r"\bcan i (buy|sell|invest)\b", r"\bopinion\b", r"\btips?\b", r"\bpros and cons\b",
]

PERF_TRIGGERS = [
    r"\breturns?\b", r"\bCAGR\b", r"\bXIRR\b", r"\bNAV\b", r"\bAUM\b",
    r"\bperformance\b", r"\bbest performing\b", r"\boutperform\w*\b", r"\bprofit\b",
    r"\bhow much (did|has|will) .* (grow|gained|earned)\b", r"\bcompare returns?\b",
    r"\bexpect(ed)? (return|growth)\b",
]
```

**[ADDITION A-5]** `\bprofit\b` and `\btips?\b` are included even though they can appear in factual
questions. The escape valve is Stage 1 rule 4: an ambiguous trigger is escalated to the LLM
classifier rather than refused outright, so the false-refusal budget is protected by design rather
than by trimming the trigger list.

### 5.3 Stage 2 — retrieval

```python
def retrieve(question, top_k=5, threshold=SIMILARITY_THRESHOLD):
    where = infer_metadata_filter(question)      # e.g. {"$and": [{"scheme": "flexi_cap"},
                                     #             {"$or": [{"plan": "direct"}, {"plan": "n_a"}]}]}
    qvec  = embedder.encode([question], normalize_embeddings=True)[0]

    res = collection.query(
        query_embeddings=[qvec],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = [Hit(chunk_id, text, metadata, score=1.0 - distance)
            for ... if (1.0 - distance) >= threshold]
    return hits
```

| Design point | Value | Reason |
|---|---|---|
| `top_k` | 5 | PRD FR-R2. Small because the corpus is ~19 documents; more context invites more unsupported numerics. |
| `SIMILARITY_THRESHOLD` | **0.35**, calibrated in M4 | Treated as a *tunable*, not a constant. Calibration: run the 40-question gold set, take the score of the correct chunk (positive) and of the top-scoring irrelevant chunk (negative), and set the threshold at the midpoint band. Recorded in `eval/eval_report.md`. |
| Metadata filter | `$and` of scheme + `plan IN {matched, n_a}` | Statutory docs are `plan = n_a` and must survive a Direct-specific filter — excluding them would throw away the authoritative fee table. |
| Distance → score | `score = 1 - distance` (cosine) | Enables a single, interpretable threshold. |

**`infer_metadata_filter`** is a small rule-based resolver over scheme aliases
(`"flexi cap" → flexi_cap`, `"elss"/"tax saver" → elss`, `"mid cap" → mid_cap`) and plan aliases
(`"direct" → direct`, `"regular" → regular`). It **narrows** retrieval only. If no alias matches, no
filter is applied — recall is never reduced by a failed guess.

**[ADDITION A-6]** `plan` filtering is deliberately one-directional. A question naming "Direct" narrows
to `{direct, n_a}`; a question naming **no** plan applies **no** plan filter at all. Applying a
default plan when the user did not specify one would manufacture cross-plan answers — the exact
failure the zero-tolerance metric targets.

### 5.4 Stage 3 — context assembly

```
[1] SOURCE  Axis Flexi Cap Fund - Direct Growth  (scheme=flexi_cap, plan=direct, scheme_page)
    <chunk 1 text>
[2] SOURCE  Axis Flexi Cap Fund - SID  (scheme=flexi_cap, plan=n_a, sid, p.12)
    <chunk 2 text>
```

Chunks are ordered by descending score and truncated at `MAX_CONTEXT_CHARS` (default 6000 chars,
≈ 1500 tokens), always keeping at least chunk 1 so the answer path never has empty context.

`MAX_CONTEXT_CHARS` is sized so that prompt + context stays comfortably inside the model's context
window at `temperature=0`, with headroom for the rule block.

### 5.5 Stage 4 — prompt template

Lives in `templates.py` as `PROMPT`. Reproduced from PRD §10.2 with two additions.

```
You are a mutual fund FAQ assistant for Axis Mutual Fund.

STRICT RULES
1. Answer ONLY from the CONTEXT below. If the answer is not in the context, reply exactly:
   NOT_FOUND
2. Maximum 3 sentences. No bullet lists. No tables.
3. Do NOT give investment advice, recommendations, opinions, or suitability guidance.
4. Do NOT state, compute, or compare returns, NAV, or performance. If asked, reply exactly: NO_PERF
5. Do NOT write any URL or link. Citations are added automatically after your reply.
6. Copy every number, rate, and date exactly as written in the CONTEXT. Never round, convert,
   reformat, or infer a figure. Omit rather than guess.

CONTEXT
{context}
```

| Addition | Reason |
|---|---|
| Rule 5 — *do not write any URL* | AD-3. The model should not even attempt the citation; it is injected server-side. Removing the `Source: <url>` line from the original draft means the model cannot emit a malformed or hallucinated link that a naive parser would then try to render. |
| Rule 6 — *copy numbers exactly* | PRD R5 (LLM invents a fee). The failure mode is subtle rounding or unit drift ("0.95%" → "about 1%"). |

Generation call:

```python
client.chat.completions.create(
    model          = GROQ_MODEL,          # default "llama-3.1-8b-instant"
    messages       = [{"role": "system", "content": PROMPT},
                     {"role": "user",   "content": question}],
    temperature    = 0.0,
    max_tokens     = 220,
    stream         = True,
)
```

`temperature=0` for reproducibility (NFR-4). `max_tokens=220` comfortably fits 3 sentences plus the
sentinel, and bounds the worst-case latency tail.

### 5.6 Stage 5 — output assembly (AD-3 core)

```python
def assemble(body, hits, question):
    if body.strip() == "NOT_FOUND":
        return templates.NOT_IN_CORPUS
    if body.strip() == "NO_PERF":
        return templates.REFUSE_PERFORMANCE

    body = cap_sentences(clean(body), max_sentences=3)

    cited = [h for h in hits if h.score >= CITATION_SCORE_FLOOR]
    citation = cited[0] if cited else hits[0]
    source_date = max((h.metadata["source_date"] for h in cited if h.metadata["source_date"]),
                      default=None)

    return AnswerEnvelope(
        answer         = body,
        citation_url   = citation.metadata["source_url"],   # verbatim, never LLM-authored
        source_title   = citation.metadata["source_title"],
        chunk_id       = citation.chunk_id,                 # NFR-7 traceability
        last_updated   = source_date,                       # FR-R8
        hits           = hits,
        latency_ms     = ...,
    )
```

**`CITATION_SCORE_FLOOR`** is a second, lower threshold (default 0.45) used only to decide *which*
retrieved chunk is the single best citation. The top hit by score is cited.

`cap_sentences` splits on `(?<=[.!?])\s+`, truncates to 3, and appends a trailing period if the
truncation landed mid-sentence. Every invocation is counted into the trace so G4's 100% target is
measurable rather than assumed.

### 5.7 `AnswerEnvelope` — the UI boundary

```python
@dataclass
class AnswerEnvelope:
    answer: str
    citation_url: str | None
    source_title: str | None
    chunk_id: str | None
    last_updated: str | None
    hits: list[Hit]
    template_id: str          # "FACTUAL" | "NOT_FOUND" | "REFUSE_ADVICE" |
                              # "REFUSE_PERFORMANCE" | "PII_BLOCKED" | "NO_RETRIEVAL"
    latency_ms: dict[str, int]
    guardrail_trace: dict
```

`template_id` lets `eval/run_eval.py` assert refusal behaviour from the envelope alone, without parsing
English. `app.py` renders the envelope and **never** inspects raw LLM output.

### 5.8 Output verifier (FR-G6)

Runs last, on the assembled string. Any failure → discard the answer, emit the matching template.

| # | Check | On failure |
|---|---|---|
| 1 | Sentence count ≤ 3 | Already enforced by `cap_sentences`; re-asserted |
| 2 | Contains no advice phrasing (`you should`, `I recommend`, `suitable for you`, `consider investing`, `I would suggest`) | `REFUSE_ADVICE` |
| 3 | Contains no return/performance claim outside a redirect | `REFUSE_PERFORMANCE` |
| 4 | **Every numeric token in the answer appears in the retrieved context** | `NOT_FOUND` |
| 5 | `citation_url` is present in `hits` metadata | `NOT_FOUND` |
| 6 | `template_id` is not `REFUSE_*` while advice phrasing is present | forced `REFUSE_ADVICE` |

Check 4 is the highest-value one for a financial facts-only system and the strictest to implement
correctly: numeric extraction must be normalization-aware, or `Rs. 500` in the answer will not match
`Rs 500` in the context. The verifier normalizes both sides — strip currency symbols, unify
whitespace, normalize Unicode, and compare on a canonical digit-sequence set.

**[ADDITION A-7]** Check 4 is applied to the **answer body only**, never to the injected citation or
the `Last updated` line — those are server-generated and legitimately contain digits not present in a
single chunk.

---

## 6. Cross-cutting concerns

### 6.1 Configuration

`config.py` is the single place any tunable is defined. No magic numbers in other modules.

| Group | Keys | Default |
|---|---|---|
| Paths | `RAW_DIR`, `PROCESSED_DIR`, `ARTIFACTS_DIR`, `CHROMA_DIR`, `CHUNKS_TXT`, `CHUNKS_JSON` | as PRD §18.4 |
| Corpus | `SOURCES_CSV`, `EMBED_MODEL`, `COLLECTION_NAME`, `EMBED_DIM` | `all-MiniLM-L6-v2`, `axis_mf_faq`, `384` |
| Chunking | `MAX_CHUNK_CHARS`, `MIN_CHUNK_CHARS`, `CHUNK_OVERLAP`, `TABLE_OVERLAP` | `900`, `120`, `150`, `0` |
| Retrieval | `TOP_K`, `SIMILARITY_THRESHOLD`, `CITATION_SCORE_FLOOR`, `MAX_CONTEXT_CHARS` | `5`, `0.35`, `0.45`, `6000` |
| Generation | `GROQ_MODEL`, `GROQ_CLASSIFIER_MODEL`, `TEMPERATURE`, `MAX_TOKENS` | `llama-3.1-8b-instant`, same, `0.0`, `220` |
| Fetch | `HTTP_TIMEOUT`, `HTTP_RETRIES`, `USER_AGENT`, `REQUEST_DELAY_S` | `30`, `3`, browser UA, `1.5` |
| Guardrails | `ENABLE_PII_SCREEN`, `ENABLE_OUTPUT_VERIFIER`, `PII_LOG_RAW` (must be `False`) | `True`, `True`, `False` |

Environment overrides use the `AXIS_` prefix (`AXIS_TOP_K=8`). `config.py` asserts
`PII_LOG_RAW is False` at import — the privacy invariant is enforced by the module, not by reviewer
discipline (NFR-3).

### 6.2 Secrets (FR-O1, NFR-9)

| Rule | Enforcement |
|---|---|
| `GROQ_API_KEY` read only from `.env` via `python-dotenv` | No `os.environ.get` fallback elsewhere |
| `.env` gitignored; `.env.example` committed with an empty value | `.gitignore` entry |
| Key never logged, echoed, or included in trace records | `TraceRecord` has no key field; health-check prints only `key_present: bool` |
| Health check fails loudly if the key is absent | `python -m src.healthcheck` |

### 6.3 Observability

A `TraceRecord` is emitted per turn to stdout as one JSON line. Fields are chosen so that every
PRD §14 metric is computable without re-running anything.

```python
{
  "trace_id": "a1b2c3",
  "template_id": "FACTUAL",
  "question_len": 52,               # not the question itself (FR-G5)
  "question_hash": "7f3e...",       # enables repeat-detection without storing content
  "stage_ms": {"guardrail": 3, "embed": 41, "search": 27,
               "generate": 3980, "verify": 8, "total": 4059},
  "n_hits": 5,
  "hits": [{"chunk_id": "s04-c017", "score": 0.71, "scheme": "flexi_cap",
            "plan": "direct", "doc_type": "scheme_page"}],
  "cited_chunk_id": "s04-c017",
  "citation_url": "https://www.axismf.com/...",
  "last_updated": null,
  "sentence_count": 2,
  "guardrails": {"pii": "clear", "input_class": "FACTUAL",
                 "output": {"advice": "clear", "performance": "clear",
                            "unsupported_numerics": "clear", "url_trusted": true}},
  "truncated_by_cap": false,
  "model": "llama-3.1-8b-instant"
}
```

Only `question_len` and `question_hash` are recorded — never the text (FR-G5, NFR-3). The trace makes
latency percentiles (NFR-1), refusal rate, and citation selection directly measurable.

### 6.4 Latency budget (NFR-1)

| Stage | p95 budget | Notes |
|---|---|---|
| Input guardrail | 5 ms | Regex only on the fast path |
| Classifier (only when ambiguous) | 600 ms | Cheapest Groq model |
| Query embed | 60 ms | MiniLM on CPU, single sentence |
| Chroma search | 250 ms | ~1–3k chunks; comfortably inside |
| Context assembly | 5 ms | String ops |
| **Groq generation** | **4000 ms** | Dominant cost. Streaming keeps perceived latency lower. |
| Output verify | 10 ms | Regex + numeric extraction |
| **Total** | **4330 ms** | Headroom to the 5 s NFR-1 target |

### 6.5 PII minimisation

| Control | Where |
|---|---|
| Question screened before any model call | `guardrails.screen_input` |
| Raw question never persisted | no question field in `TraceRecord`; `app.py` keeps no session history |
| PII hit logs pattern name only | `guardrails.py` |
| No cookies, no auth, no user accounts | fetcher and UI |
| Corpus is public scheme data — contains no PII by construction | `sources.csv` |

---

## 7. Module interfaces

Signatures only; bodies live in the modules. These are the seams that keep the project testable.

```python
# src/fetch.py
def fetch_source(source: Source, *, force: bool = False) -> FetchResult: ...
def fetch_all(sources: list[Source], *, force: bool = False) -> list[FetchResult]: ...

# src/extract.py
def extract_blocks(path: Path, doc_type: DocType) -> list[Block]: ...

# src/normalize.py
def normalize_blocks(blocks: list[Block]) -> list[Block]: ...
def drop_repeated_pdf_lines(pages: dict[int, str], threshold: float = 0.5) -> dict[int, str]: ...

# src/chunk.py
def chunk_document(blocks: list[Block], source: Source, cfg: Config) -> list[Chunk]: ...
def flatten_table(rows: list[list[str]], heading_path: str) -> str: ...
def write_chunks_txt(chunks: list[Chunk], path: Path) -> None: ...

# src/embed.py
class Embedder:
    def encode(self, texts: list[str]) -> list[list[float]]: ...
    @property
    def fingerprint(self) -> str: ...

# src/ingest.py
def ingest(*, only_fetch: bool = False, only_embed: bool = False,
           force: bool = False, prune: bool = False) -> IngestReport: ...

# src/retrieve.py
def retrieve(question: str, *, top_k: int, threshold: float) -> list[Hit]: ...
def infer_metadata_filter(question: str) -> dict | None: ...

# src/guardrails.py
def screen_input(question: str) -> ScreenResult: ...
def verify_output(answer: str, hits: list[Hit]) -> VerifyResult: ...
def numeric_tokens(text: str) -> set[str]: ...

# src/generate.py
def answer(question: str, hits: list[Hit], cfg: Config) -> AnswerEnvelope: ...
def build_context(hits: list[Hit], max_chars: int) -> str: ...
def cap_sentences(text: str, max_sentences: int) -> tuple[str, bool]: ...

# src/templates.py
def render(template_id: str, **kwargs) -> str: ...
DISCLAIMER: str

# src/app.py
def handle_question(question: str) -> AnswerEnvelope: ...
```

### 7.1 Why these seams

| Seam | Enables |
|---|---|
| `Embedder.fingerprint` | Index invalidation detection (NFR-4); the eval harness can prove the same model produced the index and the queries |
| `screen_input` / `verify_output` as pure functions | Unit-testing all guardrails with **no** network and **no** Groq key |
| `injectable` LLM client in `generate.answer` | Recording/replaying raw replies offline; the eval harness never needs a live key for retrieval metrics |
| `render(template_id)` | Guarantees the UI and README show identical text (FR-O4) |

---

## 8. Worked traces

### 8.1 Factual — exit load, Regular plan

```
Q  "What is the exit load on the Axis Large Cap Fund - Regular Growth plan?"
```

| # | Stage | Result |
|---|---|---|
| 1 | `screen_input` | no PII; `regular` alias hit (not a trigger) → `FACTUAL`, 3 ms |
| 2 | `infer_metadata_filter` | `{"$and":[{"scheme":"large_cap"},{"$or":[{"plan":"regular"},{"plan":"n_a"}]}]}` |
| 3 | embed | 384-dim normalized vector, 41 ms |
| 4 | Chroma query | 5 hits; top 2 are `s02-c031` (score 0.79) and `s03-c118` (SID, `n_a`, 0.71); all ≥ 0.35 |
| 5 | context | 1,840 chars, under 6000 |
| 6 | Groq | `"For the Regular Growth plan, exit load is 1.00% if units are redeemed within 365 days, 0.50% between 366 and 730 days, and 0% after 730 days."` — 2 sentences, 0 URLs |
| 7 | assemble | cite `s02-c031` → the Regular scheme page (score 0.79 ≥ floor 0.45); `last_updated = null` → line renders `Last updated from sources: date not stated on page` |
| 8 | verify | sentences 2 ✓ · no advice ✓ · no performance ✓ · numerics `{1.00%, 0.50%, 365, 730}` all present in context ✓ · URL trusted ✓ |
| 9 | render | answer + `Source:` link + date line |

**Why `plan = n_a` is included in the filter:** the SID (`s03`) is authoritative and was retrieved as
hit 2. Filtering to `regular` alone would have discarded it. This is why the filter is
`{matched, n_a}` and never just `{matched}`.

### 8.2 Refusal — advice question

```
Q  "Should I invest in the Axis ELSS Tax Saver Fund?"
```

| # | Stage | Result |
|---|---|---|
| 1 | `screen_input` | no PII; `ADVICE_TRIGGERS` matches `\bshould i\b` → `ADVICE` |
| 2–5 | — | **skipped entirely** — no embedding, no search, no Groq call |
| 6 | `render("REFUSE_ADVICE")` | facts-only message + SEBI adviser pointer + AMC downloads link |
| — | trace | `template_id=REFUSE_ADVICE`, `stages_skipped=[embed,search,generate]`, ~3 ms |

### 8.3 Safety — PII

```
Q  "My PAN is ABCDE1234F - what is the exit load?"
```

| # | Stage | Result |
|---|---|---|
| 1 | PII regex | PAN matches `\b[A-Z]{5}\d{4}[A-Z]\b` → `PII_BLOCKED` |
| 2–5 | — | **skipped** — the question never reaches Groq (B2 never crossed) |
| 6 | `render("PII_BLOCKED")` | security message |
| — | log | `{"pii_blocked": true, "pattern": "pan"}` — **the PAN is not logged** |

### 8.4 Performance — return question

```
Q  "What's the 3-year return of the Axis Large Cap Fund?"
```

| # | Stage | Result |
|---|---|---|
| 1 | `screen_input` | `\breturns?\b` and `\breturn\b` → `PERFORMANCE` |
| 2–5 | — | skipped |
| 6 | `render("REFUSE_PERFORMANCE")` | "I don't calculate or compare returns" + link to the official factsheet |

### 8.5 No retrieval hit

```
Q  "What is the expense ratio of the Axis Contra Fund?"
```

| # | Stage | Result |
|---|---|---|
| 1 | `screen_input` | `FACTUAL` |
| 2 | filter | `scheme` alias fails → **no filter applied** (A-6) |
| 3 | embed + search | top score 0.19, below `SIMILARITY_THRESHOLD` = 0.35 → **0 hits** |
| 4 | `render("NO_RETRIEVAL")` | "I don't have that in the sources I use" — **no Groq call** |

This is the FR-R3 outcome that matters: the correct behaviour for an out-of-corpus scheme is an honest
gap, not a confident number from the nearest available fund.

---

## 9. Testing strategy

Unit tests need neither network nor a Groq key — possible because of the seams in §7.

| Layer | Scope | Fixtures / method |
|---|---|---|
| `sources` | CSV validation, `enabled=false` exclusion | In-memory rows |
| `extract` | HTML/PDF → `Block[]`; heading paths; table detection | Saved raw samples in `tests/fixtures/` (a few KB each) |
| `normalize` | De-hyphenation, repeated-line removal, unicode | Inline strings |
| `chunk` | Tier 1 atomicity, Tier 2 heading prefix, Tier 3 overlap, short-chunk merge, **table flattening output** | Inline blocks |
| `embed` | dim = 384, vectors L2-normalized (‖v‖≈1) | 3 sentences |
| `guardrails` | **Every PII regex** (positive + negative), advice/perf triggers, `numeric_tokens`, `verify_output` | Parametrized tables. This is the highest-coverage module — it is pure and cheap |
| `generate` | `cap_sentences`, context building, citation injection, sentinel handling | Fake LLM client |
| `retrieve` | `infer_metadata_filter` alias resolution; threshold behaviour | In-memory Chroma or mock collection |
| `templates` | Every `template_id` renders non-empty; disclaimer matches PRD §11.5 verbatim | — |
| `ingest` | **Idempotency**: run twice → identical chunk set, zero duplicates | Temp dir, fake embedder |
| `eval` | End-to-end gold set, metric computation, report emission | Gold set; retrieval metrics run without a Groq key |

**Highest-risk invariants, tested explicitly:**

1. Tier-1 tables are never split.
2. Re-running `ingest.py` produces zero duplicate chunk ids.
3. The citation URL is always one that appeared in `hits` — enforced by `verify_output` check 5, and
   independently asserted by the eval harness.
4. No numeric token in a factual answer is absent from its retrieved context.
5. `PII_LOG_RAW is False` at import.

---

## 10. Run modes & operations

| Command | Stage | Network | Needs key |
|---|---|---|---|
| `python -m src.healthcheck` | env + dependency + index presence | no | no |
| `python -m src.fetch` | ingestion 1–2 | yes | no |
| `python -m src.extract` | ingestion 3–4 | no | no |
| `python -m src.chunk` | ingestion 5 → `chunks.txt` | no | no |
| `python -m src.ingest --only-embed` | ingestion 6 | no | no |
| `python -m src.ingest` | ingestion 6 | no | no |
| `python -m src.ingest --prune` | purge orphans | no | no |
| `streamlit run src/app.py` | query serving | yes (Groq only) | yes |
| `python -m eval.run_eval` | evaluation | yes | retrieval-only mode: no |

**Startup sequence for `app.py`:** load config → assert `GROQ_API_KEY` present → open
`PersistentClient` → assert collection exists **and** the embedder fingerprint in `ingest_state.json`
matches → cache the `Embedder` and `collection` as module-level singletons (a Streamlit rerun reloads
the script, so this is the difference between a 1 s and an 8 s cold start) → render UI.

The fingerprint check is what prevents the worst silent failure in a RAG system: querying an index
built with a different embedding model, which returns plausible-looking nonsense with no error raised.

---

## 11. Failure modes & degradation

| # | Failure | Detection | Behaviour | User sees |
|---|---|---|---|---|
| F1 | `GROQ_API_KEY` missing or invalid | Health check; Groq 401 | Fail fast at startup; no silent fallback to a different provider | Clear setup error |
| F2 | Chroma collection missing | Startup assert | Instruct to run `ingest.py`; the app refuses to start | "Index not built — run `python -m src.ingest`" |
| F3 | Embedder fingerprint mismatch | Startup assert | Refuse to serve; instruct to re-embed | Clear rebuild message |
| F4 | Source URL 404s / times out (R4) | `FetchResult.status` | Log to `ingest_report.md`; continue with the rest of the corpus (NFR-6) | Corpus gap, documented in README |
| F5 | Scheme page returns a client-rendered shell (R1) | extracted chars < `MIN_EXTRACTED_CHARS` | Flag `suspected_client_rendered`; rely on the SID/KIM PDF for the same facts; record as a known limit | Possibly thinner answers for that scheme |
| F6 | PDF extraction produces interleaved columns | sanity check on text ordering | Reduce reliance on that doc; note in README | — |
| F7 | Groq timeout / rate limit (R7) | HTTP error | Retry 2× with backoff, then return a graceful message | "Temporarily unavailable — please retry" |
| F8 | Answer has no citation because all hits are weak | `cited` empty | Never emit an uncited factual answer (G1) — emit `NO_RETRIEVAL` instead | "I don't have that in the sources I use" |
| F9 | Output verifier rejects a legitimate answer | verifier fires | Log + emit the conservative template | A refusal where an answer was possible — visible in the eval false-refusal metric |
| F10 | Corpus contradicts itself across documents (R9) | not auto-detected | Precedence: statutory (SID/KIM) > scheme page > factsheet; `source_date` always shown | Answer with the most authoritative source and its date |
| F10 | **A 2024 factsheet and a 2026 scheme page disagree on a fee** | not auto-detected in v1 | Document the precedence rule in the README; flag as a **known limit**. A conflict *detector* (same `section_path` + same label, different value → surface both with dates) is a natural post-milestone improvement | Honest two-source answer |

**[ADDITION A-8]** F10 is called out explicitly because PRD §17 Q4 (conflict precedence) is unresolved
and this is the one place where the architecture commits to a rule ahead of the decision: *prefer the
statutory document, always display the document date.* If the reviewer prefers scheme-page-first, only
the precedence constant changes — no component is affected.

---

## 12. Traceability matrix

Every PRD requirement maps to at least one module and one verification method.

| PRD ID | Requirement (abbrev.) | Module | Verified by |
|---|---|---|---|
| FR-I1 | Fetch all URLs, save raw | `fetch.py` | `ingest_report.md`, FR-I8 |
| FR-I2 | HTML/PDF clean extraction | `extract.py` | unit tests + `MIN_EXTRACTED_CHARS` |
| FR-I3 | Normalize / de-boilerplate | `normalize.py` | unit tests, PRD §9.5 gate #2 |
| FR-I4 | Chunk + write `chunks.txt` | `chunk.py` | file exists; gate §9.5 |
| FR-I5 | Embed + store persistent | `embed.py`, `ingest.py` | collection non-empty, dim 384 |
| FR-I6 | Idempotent ingest | `ingest.py` | run twice → 0 duplicates |
| FR-I7 | `source_date` + `fetched_at` | `sources.py`, `fetch.py` | metadata inspection |
| FR-I8 | Report failures, never silent | `ingest.py` | `ingest_report.md` |
| FR-I9 | Resumable CLI flags | `ingest.py` | `--only-fetch`, `--only-embed` |
| FR-R1 | Same embedder for question | `retrieve.py` | fingerprint equality assert |
| FR-R2 | top-k = 5 cosine | `retrieve.py` | `n_results=TOP_K`, hnsw cosine |
| FR-R3 | Similarity threshold | `retrieve.py` | §8.5 trace; threshold calibration in eval report |
| FR-R4 | Metadata filters | `retrieve.py` | `infer_metadata_filter` unit tests |
| FR-R5 | Only retrieved chunks in prompt | `generate.py` | `build_context` test |
| FR-R6 | Groq, ≤ 3 sentences, grounded | `generate.py` | `cap_sentences`; 100% metric |
| FR-R7 | ≥ 1 citation from chunk metadata | `generate.py` | verifier check 5; 100% metric |
| FR-R8 | `Last updated from sources:` | `generate.py` | snapshot test |
| FR-R9 | Honest "not in sources" | `generate.py`, `templates.py` | §8.5 trace |
| FR-G1 | Refuse advice | `guardrails.py` | 12-question refusal partition |
| FR-G2 | Polite refusal + educational link | `templates.py` | template test |
| FR-G3 | Refuse performance, link factsheet | `guardrails.py`, `templates.py` | §8.4 trace |
| FR-G4 | PII blocked pre-LLM | `guardrails.py` | 8-question safety partition + 0-call assert |
| FR-G5 | Never persist raw input | `guardrails.py`, `app.py` | trace has no question field |
| FR-G6 | Post-generation check | `guardrails.py` `verify_output` | unit tests + eval |
| FR-G7 | Hard prompt instructions | `templates.py` | prompt snapshot test |
| FR-U1 | Welcome line naming scope | `app.py` | manual + screenshot |
| FR-U2 | Exactly 3 clickable examples | `app.py`, `templates.py` | count asserted = 3 |
| FR-U3 | Persistent disclaimer note | `app.py`, `templates.py` | `DISCLAIMER` imported verbatim |
| FR-U4 | Render answer + link + date | `app.py` (renders `AnswerEnvelope`) | manual + screenshot |
| FR-U5 | Clear not-found / refusal states | `app.py` | §8.2–8.5 traces |
| FR-O1 | `.env` + `.gitignore` | repo root, `config.py` | healthcheck, no key in git |
| FR-O2 | README | repo root | §10 commands run from a clean clone |
| FR-O3 | Sample Q&A file | `sample_qa.md` | 5–10 entries, all cited |
| FR-O4 | Disclaimer single source of truth | `templates.py` | test asserts equality with PRD §11.5 |
| FR-O5 | Source list CSV/MD | `sources.csv`, `sources.md` | matches PRD §5.1 |
| NFR-1 | p95 ≤ 5 s | `generate.py` | `stage_ms` percentiles |
| NFR-2 | No re-embed on restart | `app.py` startup | startup time |
| NFR-3 | Zero PII persisted | `guardrails.py` | trace schema review |
| NFR-4 | Reproducible | `embed.py` fingerprint, `temperature=0` | fingerprint assert |
| NFR-5 | Offline embeddings | `embed.py` | ingest with network disabled |
| NFR-6 | One failure ≠ broken build | `fetch.py`, `ingest.py` | `ingest_report.md` |
| NFR-7 | Traceable to chunk → URL → date | `AnswerEnvelope` | every envelope has `chunk_id` |
| NFR-8 | CPU-only, < 2 GB | whole stack | MiniLM is 90 MB |
| NFR-9 | No key leakage | `config.py` | healthcheck prints presence only |
| NFR-10 | Swappable components | §7 interfaces | fake LLM/embedder in tests |
| D4 | Chunking strategy | `chunk.py`, `config.py` | §9.5 gate sign-off |
| D5 | Exclude indmoney.com | `sources.py` (`enabled=false`) | `sources.csv` row `s15` |
| D9 | Stateless | `app.py` | no session-history code |

---

## 13. Architecture decision log

| ID | Decision | Rationale | Status |
|---|---|---|---|
| **A-1** | Model D5 with an `enabled` boolean in `sources.csv` rather than deleting the row | Keeps the exclusion visible in the PRD §12.2 deliverable | Proposed — needs review |
| **A-2** | `chunks.txt` carries an appended quality-summary block | Makes the PRD §9.5 gate checkable without writing a script | Proposed — needs review |
| **A-3** | Advice checked before performance | An advice-shaped performance question should refuse as advice, not imply it would otherwise be answered | Accepted |
| **A-4** | Account/OTP PII patterns are context-gated | Bare digit runs would refuse legitimate factual questions; protects the ≤ 10% false-refusal budget | Proposed — needs review |
| **A-5** | Ambiguous triggers escalate to an LLM classifier | Guards the false-refusal budget without trimming the trigger lists | Accepted |
| **A-6** | `plan` filtering is one-directional; no plan named ⇒ no plan filter | A defaulted plan would manufacture cross-plan answers — the zero-tolerance failure mode | Proposed — needs review |
| **A-7** | Output verifier's numeric check applies to the answer body only | The injected citation and date line legitimately contain digits absent from any single chunk | Accepted |
| **A-8** | Source-conflict precedence fixed at statutory > scheme page > factsheet, with date always shown | Unblocks implementation ahead of PRD §17 Q4; affects only a precedence constant | **Blocked on PRD §17 Q4** |
| **A-9** | The LLM is instructed **not** to write URLs; citations are injected server-side | Removes the hallucinated-citation failure mode entirely rather than validating it after the fact | Proposed — needs review |
| **A-10** | Rule 6 added to the prompt: copy numbers exactly | Guards against rounding/unit drift on financial figures | Proposed — needs review |

---

## 14. Open architecture questions

1. **Conflict detection (F10).** Should the system *detect* same-label/different-value conflicts across
   documents and surface both with dates, or only apply a static precedence rule? Detection is a
   post-milestone feature; the static rule is enough for v1.
2. **Hybrid retrieval.** PRD D8 defers BM25 + dense with RRF fusion. This architecture assumes dense-only.
   Adding BM25 is contained: it affects `retrieve.py` and requires a second index — no other module changes.
3. **Classifier cost.** The advice/performance LLM classifier adds a round-trip on ambiguous questions.
   If the eval false-refusal rate is comfortably under budget with regex-only, drop it (simpler, faster).
4. **Multi-turn.** PRD §17 Q5 assumes stateless. If follow-ups are wanted, the change is an explicit
   `history` parameter through `answer()` plus a follow-up-aware filter resolver — deliberately *not*
   designed here, since it reopens the PII surface.
5. **Corpus refresh.** PRD §17 Q6 assumes a one-shot snapshot. If scheduled refresh is wanted, `ingest.py`
   becomes cron-invocable and the §4.8 idempotency path already handles incremental updates.

---

*End of architecture. Implements `PRD.md`; see §12 for traceability. Decisions A-1, A-2, A-4, A-6, A-9,
A-10 require sign-off; A-8 is blocked on PRD §17 Q4.*