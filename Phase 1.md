## Phase 1 — Semantic Search over GitHub Stars (Step‑by‑Step Plan)

- [x] Establish single source of truth for the plan (this file) and keep it updated as tasks complete.

### 1) Scope, Constraints, and Success Criteria
- [ ] Confirm goals: natural‑language search over starred repos based on README semantics.
- [ ] Confirm constraints: static hosting on GitHub Pages; zero server costs; CI does heavy work; browser does lightweight scoring only.
- [ ] Define success metrics:
  - [ ] End‑to‑end search working on `dufferzafar.com/starmap` (or project Pages URL).
  - [ ] < 3.5–15 MB vectors download (int8 vs float32) for ~9k repos; first search < 2s cold; subsequent queries < 100ms.
  - [ ] No private tokens exposed on Pages. Nightly updates via CI.

### 2) Repository and Environments
- [ ] Decide Pages deployment mode (e.g., `gh-pages` branch or `docs/` folder in `main`).
- [x] Set up basic project structure:
  - [x] `scripts/` for Python CI pipeline (collect → preprocess → embed → aggregate → quantize → build DB → publish).
  - [x] `starmap/` (or `public/`) for published DuckDB file and a tiny `version.json`.
  - [x] `ui/` for Vue + Tailwind static site (Vite recommended).
- [x] Configure `.gitignore` and large file strategy (DuckDB committed on Pages branch only).

### 3) Data Contract (DuckDB) — Minimal Schema for Phase 1
- [x] Define schema (DDL kept in `scripts/sql/schema.sql`):
  - [x] `repos(repo_id, full_name, html_url, description, language, stars, starred_at, last_updated)`
  - [x] `readmes(repo_id, cleaned_text, hash)`
  - [x] `chunks(repo_id, chunk_id, text)` (optional but recommended for aggregation quality)
  - [x] `chunk_vectors(repo_id, chunk_id, embedding FLOAT[])`
  - [x] `repo_vectors(repo_id, embedding FLOAT[384])`
  - [x] `repo_vectors_q8(repo_id, embedding INT8[384], scale FLOAT)` (optional; for small download)
  - [x] `neighbors(repo_id, neighbor_id, sim)` (optional; for related repos)
- [x] Create SQL to build tables and indexes (on `repo_id`, `language`, `stars`).

### 4) GitHub Integration (CI)
- [ ] Create GitHub Actions secret `GH_TOKEN` (read‑only).
- [x] Python script to list starred repos (paged API); includes `starred_at` via `application/vnd.github.star+json`.
- [x] Python script to fetch primary README for each repo (`/readme` API), with basic concurrency.
- [x] Persist raw metadata and README bodies to a `.cache/` directory.

### 5) Preprocessing and Chunking
- [x] Clean README using `markdown-it-py` + `BeautifulSoup`/`lxml`: remove badges/shields (incl. tables), images-only sections, normalize whitespace; extract visible text.
- [x] Chunking: default 512 token/units windows with 50 overlap using pluggable chunkers (`hf:<model>` default, `tiktoken`, `words`, `chars`).
- [x] Add one metadata chunk per repo (title/language/description) to boost recall.
- [x] Persist `chunks` and (optionally) top snippet(s) for display.

### 6) Embeddings (Batch in CI)
- [x] Pick model (open‑source, CPU‑friendly): `BAAI/bge-small-en-v1.5` (384‑d) (default).
- [x] Implement batch embedding with `sentence-transformers` (CPU) and on‑disk cache keyed by content hash.
- [x] Write chunk embeddings to JSONL.
- [x] Aggregate to repo vectors (medoid after mean) and write `repo_vectors.jsonl` (float32) and `repo_vectors_q8.jsonl` (int8 + scale).

### 7) Build DuckDB
- [x] Materialize `repos`, `stars`, `readmes`, `chunks`, `chunk_vectors` into `starscope.duckdb` (scripted `scripts/db.py`).
- [x] Add secondary indexes for filters (language, stars, date starred) and joins.
- [x] Build DuckDB VSS HNSW index on fixed-size `FLOAT[dim]` `chunk_vectors_arr.embedding` (cosine) for fast search ([DuckDB VSS](https://duckdb.org/2024/05/03/vector-similarity-search-vss.html)).
- [x] Sanity check: basic counts and search runs.

### 8) Publish Artifacts (CI)
- [ ] Prepare `version.json` with timestamp, counts, model name, dims, and quantization flag.
- [ ] Commit `starmap.duckdb` (+ `version.json`) to the Pages branch/path.
- [ ] Schedule nightly build (cron) and manual dispatch; skip work when ETags indicate no changes.

### 9) UI (Vue 3 + Tailwind)
- [ ] Scaffold Vite + Vue + Tailwind in `ui/`.
- [ ] Minimal single‑page app at `/starmap` with search bar, filters, and results.
- [ ] Integrate `duckdb-wasm` to open `starmap.duckdb` from Pages.
- [ ] Load vectors:
  - [ ] Prefer `repo_vectors_q8` for small download; fall back to float32.
  - [ ] Cache vectors in IndexedDB and gate reloads on `version.json`.
- [ ] Query embedding in browser via `transformers.js`:
  - [ ] Load a small quantized MiniLM/BGE for queries on demand.
  - [ ] Cache model after first load.
- [ ] Scoring:
  - [ ] Compute cosine similarity (Web Worker) between query vector and repo vectors.
  - [ ] Optional hybrid: simple BM25 over title/description/top snippet to boost keyword matches.
- [ ] Results list: name, description, language, stars, top README snippet, "Open on GitHub".
- [ ] Filters: language (multi‑select), stars ≥ N, date starred window.

### 9.5) CLI & Local Search Utilities (added)
- [x] `scripts/search.py`: Chunk-level search with FAISS option, per-repo dedupe, and numpy caches.
- [x] `scripts/search_db.py`: In-DB search via DuckDB VSS HNSW (`array_cosine_distance`) using `chunk_vectors_arr`.

### 10) Performance and UX Hardening
- [ ] Measure first‑load size and search latency on desktop and a modest laptop.
- [ ] Ensure Web Worker keeps UI responsive; show quick skeletons/spinners.
- [ ] Implement error handling: offline, DB not found, model load failure → clear guidance and retry.
- [ ] Add simple analytics (privacy‑friendly) for query counts and load timings (optional, Phase 1 can skip).

### 11) Documentation
- [ ] Update `starscope/Readme.md` with architecture summary, model choices, and how to rebuild locally.
- [ ] Add `starscope/Search.md` references and note defaults (384‑d, medoid aggregation, nightly cadence).
- [x] CLI examples available via `scripts/search.py` and `scripts/search_db.py`.

### 12) Acceptance Tests (Definition of Done)
- [ ] CI run completes on a clean fork with no manual steps.
- [ ] Pages site loads, fetches vectors, embeds query, returns sensible results for at least 10 diverse prompts.
- [ ] Filters work and affect candidate set before scoring.
- [ ] IndexedDB cache prevents re‑download until `version.json` changes.
- [ ] No private tokens visible in network panel or source.

### 13) Nice‑to‑Haves (Optional for Phase 1)
- [ ] Precompute `neighbors` for “Related repos”.
- [ ] Simple synonyms map for common tech aliases (e.g., `pgsql` → `postgres`).
- [ ] Toggle between int8 and float vectors to compare quality/size.
- [ ] Export results as a shareable link with query + filters.

### 14) Milestones
- [ ] Milestone A: Data contract + CI pipeline builds `starmap.duckdb` nightly.
- [ ] Milestone B: UI loads vectors, embeds query, returns ranked repos.
- [ ] Milestone C: Filters + caching + perf targets met; docs finalized. 