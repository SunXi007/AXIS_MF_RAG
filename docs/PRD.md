# PRD — Mutual Fund FAQ Assistant (Facts-Only RAG Chatbot)

**Project:** `nextleap_M4_RAGchatbot`
**Milestone:** M4 — Retrieval-Augmented Generation
**Status:** Draft for review
**Owner:** _TBD_
**Last updated:** 2026-10-02
**Source of truth:** [`docs/Problem_statement.txt`](docs/Problem_statement.txt)

---

## 1. Summary

Build a Retrieval-Augmented Generation (RAG) chatbot that answers **factual** questions about a
scoped set of Axis Mutual Fund schemes using **only** official public sources (AMC pages, SID/KIM PDFs,
factsheets, SEBI/AMFI material).

Every answer must carry **at least one source link**, be **≤ 3 sentences**, and must **refuse**
opinionated or advisory questions. The system must never compute or compare returns, never accept PII,
and never persist user data that could identify a user.

**Tech constraints (non-negotiable, from brief):**

| Concern | Mandated choice |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, 384-dim, same model for docs + queries) |
| Vector store | `ChromaDB`, persisted to disk (ingest once, reuse across restarts) |
| LLM | Groq API, key in `.env`, never committed |
| Chunking | Agent-proposed strategy — must be justified before code is written, and chunks dumped to a readable `.txt` |

---

## 2. Problem Statement

Retail investors comparing Axis Mutual Fund schemes repeatedly ask the same factual questions —
expense ratio, exit load, minimum SIP, ELSS lock-in period, benchmark, riskometer category, and how to
download a capital-gains statement. Support and content teams answer these manually, from scattered
pages across an AMC website and multiple PDF statutory documents.

Three concrete failures today:

1. **Scattered sources.** The fact for a single scheme is spread across a scheme page, a Direct/Regular
   pair, an SID PDF, a KIM PDF, and a monthly factsheet. A human must know which document to open.
2. **Stale or wrong figures.** Direct vs. Regular plans have *different* expense ratios and exit loads.
   Answering from the wrong plan variant is a factual error, not a rounding difference.
3. **No safe boundary for automation.** Generic LLM chat on this domain drifts into advice
   ("should I buy this?"), returns-comparison claims, and unciteable answers — all of which are
   out of bounds for this milestone.

**The gap:** a scoped, citeable, facts-only retrieval system over a fixed corpus of official pages —
with hard guardrails that make it safe to demo.

---

## 3. Goals & Non-Goals

### 3.1 Goals

| # | Goal | Measurable outcome |
|---|---|---|
| G1 | Answer factual MF questions from official sources only | 100% of answers contain ≥ 1 resolvable source link |
| G2 | Distinguish Direct vs. Regular plan facts correctly | Zero cross-plan contamination in the eval set |
| G3 | Refuse advice / opinion / performance-comparison questions | 100% refusal on the opinionated test set |
| G4 | Answer concisely | 100% of factual answers ≤ 3 sentences |
| G5 | Never store or process PII | 0 PII fields persisted; PII-pattern questions blocked pre-LLM |
| G6 | Surface data freshness | Every answer ends with `Last updated from sources: <date>` |
| G7 | Inspectable retrieval | All chunks written to a human-readable `.txt` artifact before indexing |

### 3.2 Non-Goals (Out of Scope)

- **No investment advice** — no buy/sell/hold/suitability recommendations, no portfolio allocation.
- **No performance claims** — no return computation, no fund ranking, no "best performing" claims.
  Redirect to the official factsheet instead.
- **No live/real-time NAV, AUM, or fund-return data** — corpus is static public pages.
- **No login, accounts, user profiles, or conversation history persistence.**
- **No multi-AMC scope** — Axis only, four schemes (v1).
- **No third-party blogs, aggregator blogs, or forums as sources.**
- **No fine-tuning / no model training.**
- **No voice or multimodal input.**

---

## 4. Users & Primary Use Cases

### Persona A — Retail investor (comparing schemes)
> *"I'm comparing Axis Large Cap vs Flexi Cap. What's the exit load on the flexi cap regular plan?"*

Needs a fast, factual, sourced answer to make their own decision. Does **not** want to be told what to do.

### Persona B — Support / content team
> *"What's the minimum SIP for the ELSS Tax Saver, and where do I download the capital-gains statement?"*

Needs consistent, repeatable answers with citations they can hand back to a customer, and a confidence
that nothing was hallucinated.

### Persona C — Reviewer / evaluator
> *"Show me that the answer came from an official source and reflects the current document."*

Needs a visible citation, a source date, and a citable audit trail.

---

## 5. Scope — Corpus Definition

**AMC:** Axis Mutual Fund
**Schemes (4, incl. optional):** Large Cap, Flexi Cap, ELSS Tax Saver, Midcap *(optional 4th)*

### 5.1 Source Inventory

URLs are transcribed verbatim from the brief. The **Source list deliverable** (§12.2) must contain
exactly these; any addition or substitution must be justified in the README.

| # | Scheme | URL | Doc type | Notes |
|---|---|---|---|---|
| 1 | Large Cap | `https://www.axismf.com/mutual-funds/equity-funds/axis-large-cap-fund/ef-dg/direct` | Scheme page — Direct Growth | |
| 2 | Large Cap | `https://www.axismf.com/mutual-funds/equity-funds/axis-large-cap-fund/ef-gp/regular` | Scheme page — Regular Growth | Plan variant |
| 3 | Large Cap | `https://www.axismf.com/cms/sites/default/files/Statutory/Axis%20Bluechip%20Fund%20-%20SID.pdf` | SID (PDF) | Statutory |
| 4 | Flexi Cap | `https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/ml-dg/direct` | Scheme page — Direct Growth | |
| 5 | Flexi Cap | `https://www.axismf.com/mutual-funds/equity-funds/axis-flexi-cap-fund/ml-gp/regular` | Scheme page — Regular Growth | Plan variant |
| 6 | Flexi Cap | `https://transact.axismf.com/cms/sites/default/files/Statutory/Axis%20Flexi%20Cap%20Fund%20-%20SID.pdf` | SID (PDF) | Statutory |
| 7 | Flexi Cap | `https://www.axismf.com/cms/sites/default/files/pdf-factsheets/20190204018-Flexi%20Cap%20Fund%20(November%202024)%20DP-Leaflet.pdf` | Factsheet (PDF) | Dated leafet |
| 8 | Flexi Cap | `https://transact.axismf.com/cms/sites/default/files/pdf-factsheets/Axis%20Flexi%20Cap.pdf` | Factsheet (PDF) | |
| 9 | ELSS | `https://www.axismf.com/mutual-funds/equity-funds/axis-elss-tax-saver-fund/ts-dg/direct` | Scheme page — Direct Growth | |
| 10 | ELSS | `https://www.axismf.com/mutual-funds/equity-funds/axis-elss-tax-saver-fund/ts-gp/regular` | Scheme page — Regular Growth | Plan variant |
| 11 | ELSS | `https://www.axismf.com/1/5/464/2258/4302/kim_and_application_form_axis_elss_tax_saver_fund.pdf` | KIM + application form (PDF) | |
| 12 | ELSS | `https://www.axismf.com/cms/sites/default/files/Statutory/Axis%20ELSS%20Tax%20Saver%20Fund%20-%20SID.pdf` | SID (PDF) | Statutory |
| 13 | ELSS | `https://www.axismf.com/efactsheet/January-2025/Innerpage/ELSS-TAX-SAVER-FUND.html` | E-factsheet (HTML) | |
| 14 | Midcap *(opt.)* | `https://www.axismf.com/mutual-funds/equity-funds/axis-mid-cap-fund/mc-dg/direct` | Scheme page — Direct Growth | Optional scheme |
| 15 | Midcap *(opt.)* | `https://www.indmoney.com/mutual-funds/axis-midcap-fund-direct-plan-growth` | **Third-party aggregator** | ⚠ See §5.3 |
| 16 | AMC-wide | `https://www.axismf.com/` | AMC homepage | |
| 17 | AMC-wide | `https://www.axismf.com/downloads` | Downloads index | Statement / tax docs |
| 18 | AMC-wide | `https://www.axismf.com/1/5/1423/1484/1487/2872/4561/Axis_Fund_Factsheet_July_2026_2412c4ee93.pdf` | AMC factsheet (PDF) | |
| 19 | AMC-wide | `https://transact.axismf.com/cms/sites/default/files/pdf-factsheets/Axis%20Fund%20Factsheet%20March%202026.pdf` | AMC factsheet (PDF) | |

### 5.2 Fact Coverage Matrix (must be answerable)

| Fact | Primary source | Challenge |
|---|---|---|
| Expense ratio | Scheme pages (#1, #2, #4, #5, #9, #10) | Direct vs. Regular differ; usually in a small table |
| Exit load | Scheme pages | Tiered table (duration slab → %); tabular structure |
| Minimum SIP / lump sum | Scheme pages | Small labelled field; easy to lose in a naive split |
| ELSS lock-in period | ELSS pages + KIM (#9–#13) | 3 years; multiple docs mention it |
| Riskometer category | Scheme pages / factsheets | Category + periodic updates; volatile |
| Benchmark | Scheme pages, SID, factsheets | Full benchmark index name is long |
| Minimum investment | Scheme pages | ₹500 / ₹1,000 typical |
| Statement / tax-doc download | AMC downloads (#17), homepage (#16) | **Procedural, not numeric** — step-by-step guidance |
| AUM / NAV (current) | Factsheets (#7, #8, #13, #18, #19) | Dated; must carry source date |

### 5.3 Source Caveats (must be documented in README)

1. **URL #15 is indmoney.com, not Axis.** The brief's "Public sources only / no third-party blogs"
   constraint takes precedence. **Decision: exclude #15 from the ingested corpus**, or ingest it only
   as a cross-check that is never cited. Recommended: exclude. Record this decision explicitly.
2. **URL #18 filename says `July_2026` and #19 says `March 2026`** — verify these resolve; if a URL
   404s, record it in the README as a known limit rather than silently substituting.
3. **Two hostnames** (`www.axismf.com`, `transact.axismf.com`) serve the same content. Treat as distinct
   source URLs; citations must reproduce the URL actually retrieved.
4. **Dates in filenames are the document date, not the fetch date.** Both are captured (§7.4).

---

## 6. Functional Requirements

### Ingestion

| ID | Requirement | Priority |
|---|---|---|
| FR-I1 | Fetch all in-scope URLs and store the raw response (HTML/PDF) under `data/raw/<source_id>/` for reproducibility. | Must |
| FR-I2 | Extract clean text from HTML (strip nav, footer, cookie banners, scripts, breadcrumbs) and from PDF (preserve page numbers). | Must |
| FR-I3 | Normalize whitespace, de-hyphenate line-broken words, unify unicode, strip repeated headers/footers from PDFs. | Must |
| FR-I4 | Chunk using the strategy in §9 and write **every chunk** to `artifacts/chunks.txt` (human-readable, with metadata header per chunk). | Must |
| FR-I5 | Embed each chunk with `all-MiniLM-L6-v2` and store in a **persistent** ChromaDB collection. | Must |
| FR-I6 | Ingestion must be idempotent and content-hashed — re-running must not duplicate chunks. | Must |
| FR-I7 | Persist `source_date` (from doc/filename) and `fetched_at` per chunk. | Must |
| FR-I8 | Record any fetch/parse failure to an ingestion report; never silently skip a source. | Must |
| FR-I9 | Extraction/embedding steps must be resumable via CLI flags (`--only-fetch`, `--only-embed`). | Should |

### Retrieval & Answering

| ID | Requirement | Priority |
|---|---|---|
| FR-R1 | Embed the user question with the **same** `all-MiniLM-L6-v2` model. | Must |
| FR-R2 | Retrieve **top-k = 5** chunks by cosine similarity from ChromaDB. | Must |
| FR-R3 | Apply a **minimum similarity threshold**; if nothing clears it, say the corpus doesn't cover it (never guess). | Must |
| FR-R4 | Support optional metadata filters (e.g. `scheme`, `plan`, `doc_type`) derived from the question. | Should |
| FR-R5 | Send only the retrieved chunks (numbered, with their source labels) in the prompt. | Must |
| FR-R6 | Generate via Groq, constrained to **≤ 3 sentences**, grounded only in provided context. | Must |
| FR-R7 | Append **at least one** citation link taken verbatim from the retrieved chunk metadata. | Must |
| FR-R8 | Append `Last updated from sources: <source_date>`. | Must |
| FR-R9 | If the answer cannot be grounded in context, return the "not in sources" message + a relevant educational link. | Must |

### Guardrails

| ID | Requirement | Priority |
|---|---|---|
| FR-G1 | Detect and refuse **advice/opinion** questions ("should I buy/sell", "which is better", "is it safe?"). | Must |
| FR-G2 | Refusal message is polite, facts-only, and includes a relevant **educational** link (e.g. AMC downloads / SEBI investor-education page). | Must |
| FR-G3 | Detect **performance/return** questions and redirect to the official factsheet — never compute or compare returns. | Must |
| FR-G4 | Block questions containing **PII patterns** (PAN, Aadhaar, account number, OTP, email, phone) **before** the LLM call; log a redacted marker only. | Must |
| FR-G5 | Never persist raw user input to disk. No conversation history store. | Must |
| FR-G6 | Post-generation check: if the answer contains advice-like or return-comparison phrasing despite passing the input check, discard it and emit the refusal template. | Must |
| FR-G7 | Prompt must hard-instruct: no advice, no performance claims, no facts absent from context, always cite. | Must |

### UI

| ID | Requirement | Priority |
|---|---|---|
| FR-U1 | Welcome line naming the assistant and its scope (Axis MF, 4 schemes). | Must |
| FR-U2 | Display **exactly 3** example questions, clickable to submit. | Must |
| FR-U3 | Persistent, visible note: `Facts-only. No investment advice.` | Must |
| FR-U4 | Each answer renders: answer text, source link(s), and the `Last updated from sources:` line. | Must |
| FR-U5 | Clear state for "not found in sources" and for refusal. | Must |

### Deliverables / Ops

| ID | Requirement | Priority |
|---|---|---|
| FR-O1 | `.env` for `GROQ_API_KEY`; `.gitignore` must exclude `.env`, `data/`, `chroma/`, caches. | Must |
| FR-O2 | README: setup steps, scope (AMC + schemes), known limits. | Must |
| FR-O3 | Sample Q&A file: 5–10 queries with answers + links. | Must |
| FR-O4 | Disclaimer snippet stored in one place and reused by the UI and README. | Must |
| FR-O5 | Source list as CSV/MD. | Must |

---

## 7. Non-Functional Requirements

| ID | Category | Requirement |
|---|---|---|
| NFR-1 | Latency | p95 end-to-end answer ≤ 5 s (retrieval ≤ 300 ms; LLM ≤ 4 s). Streaming response. |
| NFR-2 | Startup | App must not re-embed on restart — ChromaDB load from disk. |
| NFR-3 | Privacy | Zero PII in logs, DB, or artifacts. No user identifiers collected. |
| NFR-4 | Reproducibility | Given the same corpus snapshot + model versions, retrieval results are identical. Pin all model versions. |
| NFR-5 | Offline embeddings | Embedding step must run with **no** network/API key. |
| NFR-6 | Robustness | A single failed source must not break ingestion; failures reported, not fatal. |
| NFR-7 | Transparency | Every answer traceable to a chunk id → source URL → document date. |
| NFR-8 | Portability | Runs on CPU-only machines. Total local footprint < 2 GB. |
| NFR-9 | Secrets | No key ever logged, echoed, or committed. |
| NFR-10 | Maintainability | Swappable components behind thin interfaces (embedder, vector store, LLM client). |

---

## 8. System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              OFFLINE / INGESTION                             │
│                                                                              │
│  source_urls.csv                                                             │
│        │                                                                     │
│        ▼                                                                     │
│  ┌────────────┐   raw HTML/PDF saved        ┌──────────────────────────┐     │
│  │  Fetcher   │──────────────────────────►  │ data/raw/<source_id>/    │     │
│  │(httpx +    │                            └──────────────────────────┘     │
│  │ retry+UA)  │                                                                │
│  └─────┬──────┘                                                                │
│        ▼                                                                     │
│  ┌────────────────────────┐   ┌──────────────────────────────┐               │
│  │ Extractor               │   │ HTML: trafilatura / bs4      │               │
│  │ (per doc_type)          │   │ PDF: pymupdf (page-aware)    │               │
│  └────────────┬───────────┘   └──────────────────────────────┘               │
│               ▼                                                                │
│  ┌────────────────────────┐                                                   │
│  │ Normalizer             │  whitespace, unicode, de-hyphenate, de-boilerplate│
│  └────────────┬───────────┘                                                   │
│               ▼                                                                │
│  ┌────────────────────────┐   dump all chunks for inspection                  │
│  │ STRUCTURE-AWARE        │──────────────────────────────► artifacts/        │
│  │ CHUNKER  (§9)          │                                  chunks.txt      │
│  └────────────┬───────────┘                                  chunks.json      │
│               ▼                                                                │
│  ┌────────────────────────┐                                                   │
│  │ Embedder               │  all-MiniLM-L6-v2 · 384-dim · mean-pool · L2 norm │
│  │ (local, no API key)    │                                                   │
│  └────────────┬───────────┘                                                   │
│               ▼                                                                │
│  ┌────────────────────────┐                                                   │
│  │ ChromaDB (Persistent)  │  cosine · collection: axis_mf_faq · on disk      │
│  └────────────────────────┘                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                            ONLINE / QUERY (per turn)                          │
│                                                                              │
│  user question                                                                │
│       │                                                                       │
│       ▼                                                                       │
│  ┌──────────────────┐   BLOCKED  ──► refusal / PII message                    │
│  │ Guardrail: PII   │──────────────┤                                        │
│  │ + advice + perf  │              │                                        │
│  └────────┬─────────┘              │                                        │
│           │ passed                 │                                        │
│           ▼                         │                                        │
│  ┌──────────────────┐               │                                        │
│  │ Query embedder   │  same model   │                                        │
│  └────────┬─────────┘               │                                        │
│           ▼                         │                                        │
│  ┌──────────────────────────────────────────────────────┐                     │
│  │ ChromaDB search  → top-k=5 cosine                    │                     │
│  │   + similarity threshold  + optional metadata filter  │                     │
│  └────────┬─────────────────────────────────────────────┘                     │
│           │ no chunk above threshold                                        │
│           ▼                    │                        │                     │
│  ┌──────────────────┐         │                        │                     │
│  │ Context builder  │         │                        │                     │
│  │ numbered chunks  │         │                        │                     │
│  │ + source labels  │         │                        │                     │
│  └────────┬─────────┘         │                        │                     │
│           ▼                   │                        │                     │
│  ┌──────────────────┐         │                        │                     │
│  │ Groq LLM         │         │                        │                     │
│  │ ≤3 sentences     │         │                        │                     │
│  │ + 1 citation     │         │                        │                     │
│  └────────┬─────────┘         │                        │                     │
│           ▼                   │                        │                     │
│  ┌──────────────────┐   FAILS  │                        │                     │
│  │ Output guardrail │──────────┤                        │                     │
│  │ advice/return/   │          │                        │
│  │ grounding check  │          │                        │                     │
│  └────────┬─────────┘          ▼                        ▼                     │
│           ▼            ┌───────────────────────────────────────────┐          │
│  ┌──────────────────┐   │ Refusal / redirect / not-found template  │          │
│  │ Render answer    │◄──┤ (with educational or factsheet link)       │          │
│  │ + link + date    │   └───────────────────────────────────────────┘          │
│  └──────────────────┘                                                       │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 8.1 Component Responsibilities

| Component | Responsibility | Not responsible for |
|---|---|---|
| Fetcher | HTTP retrieval, retry, UA, raw persistence, capture `fetched_at` | Parsing, chunking |
| Extractor | HTML→text, PDF→page-tagged text | Normalization, chunking |
| Normalizer | Text hygiene, boilerplate removal | Semantic splitting |
| Chunker | Structure-aware splitting, metadata assignment | Embedding, storage |
| Embedder | `all-MiniLM-L6-v2`, 384-dim, normalized | Storage, retrieval |
| Vector Store | ChromaDB persistence + similarity search | Ranking policy, thresholding |
| Query Guardrail | PII / advice / performance classification **pre-LLM** | Output validation |
| Context Builder | Assemble numbered, labelled chunks | Generation |
| LLM Client | Groq call, prompt, token budget | Guardrails, citation validity |
| Output Guardrail | Post-generation compliance check | Retrieval |
| UI | Welcome, 3 examples, disclaimer, answer rendering, source links | Any generation logic |

---

## 9. Chunking Strategy (required proposal)

> The brief requires the strategy to be **proposed and justified before code is written**, with chunk
> size, overlap, and retained metadata specified, and all chunks saved to a readable `.txt`.

### 9.1 Problem shape

The corpus is **heterogeneous and label-dense**:

- Scheme pages are short, section-headed, and put answers inside **small two-column tables**
  (`Expense ratio (Direct) | 0.95%`).
- SID / KIM / factsheet PDFs are **long, tabular, multi-page**, with repeating headers/footers.
- One fact (e.g. minimum SIP) can appear on the scheme page *and* the factsheet with **different values**
  across dates.

Naive fixed-window character splitting fails this domain in a specific, predictable way: **it severs
a label from its value.** A 1,000-character window landing mid-table produces a chunk containing
`Exit load` but not the percentages, or a chunk of bare percentages with no idea which scheme or plan
they belong to. For a facts-only assistant, that is the single most damaging failure — it produces
confidently wrong, uncitable answers.

### 9.2 Proposed strategy — Structure-aware, heading-bounded, table-preserving

**Principle:** split on the document's own semantic boundaries first (headings, table blocks),
and only fall back to overlapping character windows when a single section is too long.

**Three-tier splitting:**

| Tier | Applies to | Rule |
|---|---|---|
| 1. **Atomic table block** | Any table in HTML or PDF | A table is **never split**. Serialize to `Field: Value` lines (see 9.4) and emit as one chunk. |
| 2. **Heading-bounded section** | HTML `h2`/`h3`; PDF headings detected by font-size / numbering pattern | One chunk per section, plus the section's heading breadcrumb prepended to the text. |
| 3. **Recursive character window** | Any Tier-2 section exceeding max size | Recursive split on `\n\n` → `\n` → sentence → char, merging up to the size budget with overlap. |

**Also stripped before chunking:** site nav, footer, cookie/consent banners, breadcrumbs, "REQUEST CALL
BACK" CTAs, share widgets, and repeated PDF page headers/footers (detected as lines recurring on > 50%
of pages).

**Parameters:**

| Parameter | Value | Rationale |
|---|---|---|
| `max_chunk_chars` | **900** | ≈ 220–250 MiniLM tokens. Below the 256-token window truncation limit while leaving room for the heading breadcrumb + metadata in the embedding input. Large enough to hold a full fee table or a short section. |
| `min_chunk_chars` | **120** | Below this, chunks carry too little signal; merge into the neighbour. |
| `chunk_overlap` | **150 chars** (~17%) | Enough to preserve a table header row or a sentence split across the boundary. Overlap is only applied in Tier 3, **never** between table blocks. |
| `chunk_unit` | character (not token) | Tokenizer-free chunking keeps the strategy independent of the embedder's window; 900 chars is reliably under 256 tokens for English + numbers. |
| Tables | atomic, no overlap | Splitting a table is the primary failure mode (§9.1). |
| Headings | prepended to every chunk | A chunk must be self-describing — it is retrieved without its siblings. |

### 9.3 Metadata retained per chunk

| Field | Type | Purpose |
|---|---|---|
| `chunk_id` | str | Stable traceability (`<source_id>-c<NNN>`) |
| `source_id` | str | FK to `source_urls.csv` |
| `source_url` | str | **Citation string, verbatim** |
| `source_title` | str | Human-readable label |
| `doc_type` | enum | `scheme_page \| sid \| kim \| factsheet \| efactsheet \| downloads \| homepage` |
| `scheme` | enum | `large_cap \| flexi_cap \| elss \| midcap \| amc_wide` |
| `plan` | enum | `direct \| regular \| n_a` — **critical** for FR-I7/G2 |
| `page_num` | int \| null | PDF page (null for HTML) |
| `section_path` | str | Heading breadcrumb, e.g. `Axis Flexi Cap Fund > Direct Growth > Fees` |
| `source_date` | str \| null | Document date from filename/content (drives `Last updated from sources:`) |
| `fetched_at` | iso8601 | Ingestion timestamp |
| `content_hash` | str | Idempotency (FR-I6) |
| `chunk_index` | int | Ordinal within source |

### 9.4 Table serialization

Tables are flattened to text before embedding so a fee table is retrievable as *text*, not as an opaque
blob:

```
Expense ratio (Direct Growth): 0.95%
Expense ratio (Regular Growth): 1.95%
Exit load (up to 365 days): 1.00%
Exit load (366 days to 730 days): 0.50%
Exit load (above 730 days): 0.00%
Minimum SIP: Rs. 500
Minimum lump sum: Rs. 5,000
```

This also makes the chunk directly quotable by the LLM, which supports the ≤ 3-sentence + citation
requirement.

### 9.5 Validation before indexing (acceptance gate)

`artifacts/chunks.txt` must be **manually reviewed** before FR-I5 proceeds. Gate criteria:

1. ≥ 90% of chunks contain a recognisable scheme **and** plan (or are explicitly `amc_wide`).
2. No chunk contains site nav, breadcrumbs, or repeated PDF headers.
3. Every chunk that mentions a fee/exit-load/SIP figure also contains its label.
4. Chunk length distribution: median in 400–900 chars; < 5% outside `[min, max]` after merging.
5. A manual probe set of 10 known facts can each be located in **at most one** chunk.

---

## 10. Technical Approach

### 10.1 Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` | 384-dim, mean pooling, L2-normalized; **same model** for docs & queries |
| Vector DB | `chromadb` `PersistentClient` | Cosine space; collection `axis_mf_faq`; on-disk at `chroma/` |
| Chunking | Structure-aware per §9 + `tiktoken` for token counting | |
| HTML extract | `trafilatura` (fallback `beautifulsoup4` + `lxml`) | |
| PDF extract | `pymupdf` (fitz) | Page-aware, font-size for heading detection |
| Fetch | `httpx` | Retry, backoff, browser UA, timeout |
| LLM | `groq` SDK | Default `llama-3.1-8b-instant` (latency/quality for short factual answers); config to `llama-3.3-70b-versatile` for harder tables |
| UI | `streamlit` | Fastest path to the "tiny UI" in the brief; `gradio` acceptable |
| Config | `python-dotenv` + a single `config.py` | No magic numbers inline |

### 10.2 Prompt contract (draft)

```
You are a mutual fund FAQ assistant for Axis Mutual Fund.

STRICT RULES
1. Answer ONLY from the CONTEXT below. If the answer is not in the context, reply exactly:
   NOT_FOUND
2. Maximum 3 sentences. No bullet lists. No tables.
3. Do NOT give investment advice, recommendations, opinions, or suitability guidance.
4. Do NOT state, compute, or compare returns, NAV, or performance. If asked, reply exactly: NO_PERF
5. Always end with the citation line:
   Source: <url>
6. Do not invent fees, ratios, dates, or lock-in periods. Omit rather than guess.

CONTEXT
[1] (scheme=Axis Flexi Cap Fund | plan=Regular | source=scheme_page)
    <chunk text>
[2] ...
```

### 10.3 Output parsing & assembly

- Parse the LLM reply into `body` + `Source:` line.
- If body is `NOT_FOUND` / `NO_PERF`, swap in the appropriate template (§11.4).
- Enforce sentence count ≤ 3 programmatically (split on `[.!?]`, hard-truncate with a log entry).
- Append `Last updated from sources: <max source_date across cited chunks>`.
- Output guardrail (§11.3) runs on the final string.

---

## 11. Safety & Compliance Behaviour

### 11.1 PII patterns (blocked pre-LLM, FR-G4)

| Pattern | Regex sketch |
|---|---|
| PAN | `[A-Z]{5}\d{4}[A-Z]` |
| Aadhaar | `\b\d{4}\s?\d{4}\s?\d{4}\b` |
| Account no. | 9–18 consecutive digits with context words (`a/c`, `account`, `folio`) |
| OTP | 4–6 digits near `otp\|code\|verification` |
| Email | standard email regex |
| Phone | `(?:\+91[\-\s]?)?[6-9]\d{9}` |

On match → do **not** call the LLM. Reply:
> "For your security I can't process personal identifiers. Please remove account numbers, PAN,
> Aadhaar, OTPs, email addresses, or phone numbers. — Facts-only. No investment advice."

Log only: `{"pii_blocked": true, "pattern": "<name>"}` — never the raw text.

### 11.2 Advice / performance classification (FR-G1, FR-G3)

Two-stage: a **regex/keyword pre-filter** (cheap, catches the obvious majority) followed by an
**LLM classifier** (Groq, cheap model, single-token JSON verdict) only when the pre-filter is
ambiguous. Verdicts: `FACTUAL | ADVICE | PERFORMANCE`.

- `ADVICE` triggers (`should`, `which is better`, `is it safe`, `worth investing`, `good time to`,
  `recommend`, `allocate`, `portfolio should`, `best fund`).
- `PERFORMANCE` triggers (`return`, `CAGR`, `XIRR`, `NAV`, `AUM`, `performance`, `best performing`,
  `outperform`, `growth`, `profit`, `compare returns`).

### 11.3 Output guardrail (FR-G6)

Post-generation, reject and replace with the refusal template if the answer contains:
- advice phrasing (`you should`, `I recommend`, `suitable for you`, `consider investing`);
- unsourced numeric claims — a figure in the answer that does not appear in the retrieved context;
- > 3 sentences;
- a `Source:` line whose URL is not in the retrieved chunk metadata.

### 11.4 Response templates

| Case | Response |
|---|---|
| Refused (advice) | "I can share facts about these schemes, but I don't give investment advice. For guidance on choosing a scheme, please speak with a SEBI-registered investment adviser. You can review scheme facts here: <AMC downloads link>" |
| Refused (performance) | "I don't calculate or compare returns. The official monthly factsheet has the official performance figures: <factsheet link>" |
| Not in corpus | "I couldn't find that in the official sources I'm using (Axis Mutual Fund — Large Cap, Flexi Cap, ELSS Tax Saver, Midcap). Try asking about expense ratio, exit load, minimum SIP, ELSS lock-in, benchmark, riskometer, or how to download a statement: <AMC downloads link>" |
| No retrieval hit | "I don't have that in the sources I use. — Facts-only. No investment advice." |
| PII | §11.1 |

### 11.5 UI disclaimer (FR-O4, verbatim, single source of truth)

```
Facts-only. No investment advice.
Answers are generated only from official Axis Mutual Fund public pages and statutory documents.
Mutual fund investments are subject to market risks. Read all scheme related documents carefully.
```

---

## 12. Deliverables

| # | Deliverable | Format | Acceptance |
|---|---|---|---|
| 12.1 | Working prototype | Running app (Streamlit/Gradio) + repo, **or** ≤ 3-min demo video if hosting is impossible | Starts from README steps alone |
| 12.2 | Source list | `sources.csv` + mirrored `sources.md` — the 19 URLs, doc type, scheme, plan, source date, fetch status | Matches §5.1; exclusions justified |
| 12.3 | README | `README.md` — setup, scope (AMC + schemes), architecture summary, chunking rationale, known limits | A new user runs it in < 15 min |
| 12.4 | Sample Q&A | `sample_qa.md` — 5–10 queries with the assistant's actual answers + links | All answers cited, ≤ 3 sentences |
| 12.5 | Disclaimer snippet | `disclaimer.md`, imported by the UI | Matches §11.5 exactly |
| 12.6 | Chunk artifact | `artifacts/chunks.txt` + `artifacts/chunks.json` | Passes the §9.5 gate |
| 12.7 | Eval report | `eval/eval_report.md` — metrics from §14 | All Must thresholds met |
| 12.8 | PRD | `PRD.md` (this document) | Reviewed |

---

## 13. Milestones

| Phase | Work | Exit criteria |
|---|---|---|
| **M0 — Scaffold** | Repo layout, `config.py`, `.env` + `.gitignore`, dependency pinning, health-check script | App boots; `GROQ_API_KEY` loads from env only |
| **M1 — Corpus** | `sources.csv`; fetcher; HTML/PDF extractors; extraction report | All in-scope URLs fetched or explicitly logged as failed; §5.3 decisions recorded |
| **M2 — Chunking** | Normalizer + structure-aware chunker; **write `artifacts/chunks.txt`** | §9.5 validation gate passes **before** any embedding code is written |
| **M3 — Index** | Embedder + persistent ChromaDB; ingestion CLI; idempotency check | Re-running ingest produces zero duplicates; restart reloads from disk |
| **M4 — Retrieval** | Query embedder, top-k search, threshold, metadata filters, trace/debug view | Known-fact probe set locates each fact with correct scheme **and plan** |
| **M5 — Answering** | Prompt contract, Groq client, citation assembly, `Last updated` line, output guardrail | Answers ≤ 3 sentences, cited, non-fabricated on the probe set |
| **M6 — Guardrails + UI** | PII block, advice/perf classifier, refusal templates, Streamlit UI, disclaimer | Every §11.2/§11.4 case demonstrated |
| **M7 — Eval + Handoff** | Gold test set, eval harness, report, README, sample Q&A | All §14 Must metrics met; deliverables complete |

**Hard dependency:** M2's chunk-inspection gate must pass before M3 begins. This is the brief's explicit
requirement, not a scheduling preference.

---

## 14. Evaluation Plan

### 14.1 Gold test set

Build `eval/gold_set.jsonl` with three partitions:

| Partition | Size | Purpose |
|---|---|---|
| **Factual** | 40 questions | Grounded answers; each maps to a known source chunk. Must cover all 4 schemes, both plan variants, all fact types in §5.2. |
| **Refusal** | 12 questions | Advice + performance. Expected: refusal. |
| **Safety** | 8 questions | PII-shaped inputs. Expected: PII block, no LLM call. |

Include ≥ 6 **near-miss trap** questions, e.g. *"What's the exit load on the Flexi Cap Direct Growth?"*
(the corpus's factsheets may only cover Regular — the correct behaviour may be *partial answer + honest
gap*, not a confident wrong number).

### 14.2 Metrics

| Metric | Target | Type |
|---|---|---|
| Retrieval Recall@5 | ≥ 0.85 | Must |
| Retrieval MRR@5 | ≥ 0.70 | Should |
| Citation present & resolvable | 100% | Must |
| Citation **correctness** (link contains the cited fact) | 100% | Must |
| Answer ≤ 3 sentences | 100% | Must |
| Refusal rate on refusal partition | 100% | Must |
| False-refusal rate on factual partition | ≤ 10% | Must |
| PII blocked, 0 LLM calls | 100% | Must |
| Unsupported-claim rate (manual audit of 50 answers) | 0 | Must |
| Direct/Regular cross-contamination | 0 | Must |
| p95 latency | ≤ 5 s | Must |

### 14.3 Harness

`eval/run_eval.py` — runs the gold set, records for each item: retrieved chunk ids + scores, raw LLM
reply, final answer, latency, and pass/fail per metric. Emits `eval/eval_report.md`. Ground-truth
chunk labels come from §9.5 probe work. Retrieval metrics are computed independently of the LLM so
chunking/retrieval quality is measurable on its own.

---

## 15. Risks & Mitigations

| # | Risk | Impact | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | Scheme pages render facts client-side; HTTP fetch returns an empty shell | Corpus silently empty | High | Fetch with a browser UA, then verify non-trivial extracted length; log short extractions; fall back to the SID PDF for the same facts |
| R2 | Direct vs. Regular figures confused | **Factual errors** — worst outcome | High | `plan` metadata on every chunk; §14 metric with 0 tolerance; near-miss traps in the gold set |
| R3 | Table-heavy PDFs chunk into unusable fragments | Poor retrieval | Medium | Tier-1 atomic tables + `Field: Value` serialization (§9.4) |
| R4 | Some listed URLs 404 (dated factsheets, `transact.` host) | Corpus gaps | Medium | Record in README known-limits; never silently substitute a source |
| R5 | LLM invents a fee absent from context | Fabricated financial fact | Medium | Grounding-only prompt + `NOT_FOUND` sentinel + output guardrail on unsupported numerics (§11.3) |
| R6 | Refusal over-triggers on legitimate facts ("Is exit load high?") | Bad UX | Medium | Pre-filter + LLM classifier; ≤ 10% false-refusal budget; phrase test questions in the gold set |
| R7 | Groq rate limits / latency | Demo failure | Medium | Streaming, retry with backoff, cache identical questions in-session, `llama-3.1-8b-instant` default |
| R8 | `all-MiniLM-L6-v2` 256-token window truncates | Lost content | Medium | `max_chunk_chars=900` keeps chunks under the window (NFR-4: verify empirically in M2) |
| R9 | Document dates differ across sources for the same fact | Conflicting answers | Medium | Prefer statutory docs (SID/KIM) for fees; surface `source_date`; note precedence in README |
| R10 | Scope creep into advice / multi-AMC | Milestone slip | Medium | §3.2 non-goals enforced; refusal path is a first-class feature, not an afterthought |
| R11 | Source #15 (indmoney.com) pulled in | Violates "public sources only" | Medium | Excluded by decision (§5.3); documented in README |
| R12 | PII reaches the LLM provider | Privacy incident | Low | Pre-LLM regex block (§11.1) + no raw-input logging (NFR-3) |

---

## 16. Technical Decisions (log)

| # | Decision | Rationale | Status |
|---|---|---|---|
| D1 | Embedding = `all-MiniLM-L6-v2`, 384-dim, L2-normalized, cosine | Mandated by brief; runs locally, no API key | Accepted |
| D2 | ChromaDB `PersistentClient`, cosine space | Mandated; ingest once, reuse across restarts (NFR-2) | Accepted |
| D3 | Groq as LLM, key in `.env` only | Mandated; §10.1 default `llama-3.1-8b-instant` | Accepted |
| D4 | Structure-aware chunking, 900 chars / 150 overlap, tables atomic | §9 — protects label→value pairing, the key domain failure mode | **Proposed — needs review** |
| D5 | Drop indmoney.com (#15) | Brief forbids third-party sources; #15 is an aggregator | **Proposed — needs review** |
| D6 | Guardrails are pre-LLM (regex) + post-LLM (verify) | Cheap first pass, correctness second pass | Accepted |
| D7 | Streamlit for the UI | "Tiny UI" requirement; fastest to a presentable demo | Accepted |
| D8 | HyDE / query expansion / hybrid BM25 | Deferred to a post-milestone iteration; baseline is dense-only so retrieval quality is attributable | Deferred |
| D9 | No conversation history persisted | Minimizes PII surface (NFR-3) | Accepted |

---

## 17. Open Questions

1. **Hosting vs. notebook.** The brief allows a working app *or* a ≤ 3-min video. Which?
2. **LLM model tier.** `llama-3.1-8b-instant` (fast, default) or `llama-3.3-70b-versatile`
   (better on dense tables, slower)? Affects the latency budget.
3. **Is Midcap required or optional?** Brief marks it "optional 4th scheme". Recommend **in**, since §5.1
   already lists two URLs for it and it improves corpus breadth.
4. **Which fact takes precedence when sources conflict** (e.g. a 2024 factsheet vs. a 2026 scheme page)?
   Proposal: statutory docs (SID/KIM) > live scheme page > dated factsheet, with the date always shown.
5. **Multi-turn follow-ups** — e.g. "what about the direct plan?" Assumed **out of scope** for v1
   (stateless). Confirm.
6. **Should the corpus be refreshed on a schedule?** Factsheets are monthly. v1 is one-shot snapshot.

---

## 18. Appendix

### 18.1 Fact-checked example questions (for the UI's 3 examples + gold set)

1. What is the expense ratio of the Axis Flexi Cap Fund – Direct Growth plan?
2. What is the exit load on the Axis Large Cap Fund – Regular Growth plan?
3. What is the minimum SIP for the Axis ELSS Tax Saver Fund, and what is the lock-in period?

Chosen because they span two schemes, two plan variants, two fact types, and (Q3) the procedural/lock-in
case — the three examples alone exercise a meaningful slice of the corpus.

### 18.2 Refusal set (examples)

- "Should I invest in the Axis ELSS Tax Saver Fund?"
- "Which Axis fund performed best last year?"
- "Is the Axis Flexi Cap a safe choice for me?"
- "What's the 3-year return of the Large Cap Fund?"

### 18.3 PII set (examples)

- "My PAN is ABCDE1234F — what is the exit load?"
- "Download the statement for a/c no. 123456789012"
- "Contact me at sunil@example.com or 9876543210"

### 18.4 Reference layout

```
nextleap_M4_RAGchatbot/
├── PRD.md
├── README.md
├── disclaimer.md
├── requirements.txt
├── .env.example                  # GROQ_API_KEY= (never the real key)
├── .gitignore
├── config.py
├── sources.csv
├── data/
│   ├── raw/<source_id>/          # fetched HTML/PDF, for reproducibility
│   └── processed/                # extracted + normalized text
├── artifacts/
│   ├── chunks.txt                # REQUIRED inspection artifact (all chunks)
│   └── chunks.json
├── chroma/                       # persisted vector store (gitignored)
├── src/
│   ├── fetch.py
│   ├── extract.py
│   ├── normalize.py
│   ├── chunk.py
│   ├── embed.py
│   ├── ingest.py
│   ├── retrieve.py
│   ├── guardrails.py
│   ├── generate.py
│   ├── templates.py
│   └── app.py
├── eval/
│   ├── gold_set.jsonl
│   ├── run_eval.py
│   └── eval_report.md
└── sample_qa.md
```

---

*End of PRD. Prepared from `docs/Problem_statement.txt`. Decisions D4 and D5 require sign-off before M2
proceeds.*