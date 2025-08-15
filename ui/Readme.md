## Starscope UI (Vue 3 + Vite + Tailwind)

A minimal, Google‑style semantic search UI for your starred GitHub repos. The page loads a DuckDB database in the browser via `duckdb-wasm`, embeds queries in‑browser with `@xenova/transformers` (BGE‑small), and ranks repos by cosine similarity.

### Features
- **Clean single‑box UI**: centered search bar, results list below.
- **Tailwind v4**: styling via the Vite plugin; no custom CSS beyond basics.
- **DuckDB in the browser**: opens `starscope.duckdb` from `public/data/` using `duckdb-wasm` (local worker/wasm URLs; same‑origin safe).
- **Real embeddings**: `Xenova/bge-small-en-v1.5` (quantized) loaded on demand with `@xenova/transformers`.
- **Search strategy**:
  - Prefer DuckDB VSS (`chunk_vectors_arr` + `array_cosine_distance`) if present.
  - Fallback: JS cosine over one vector per repo (derived from `chunk_vectors`).

### Directory layout
- `src/composables/useDuckDB.ts`: loads/instantiates DuckDB WASM, opens DB from buffer.
- `src/composables/useEmbedder.ts`: loads `Xenova/bge-small-en-v1.5` and returns normalized query embeddings.
- `src/composables/useSearch.ts`: runs VSS query or JS fallback and joins repo metadata.
- `src/components/`: `SearchBar`, `ResultsList`, `RepoCard`, `EmptyState`, `ErrorState`.
- `src/App.vue`: minimal, centered layout; shows DB load state/errors.

### Local setup
Prereqs: Node, pnpm.

```bash
cd /Users/szafar/Downloads/dev/starscope/ui
pnpm install
pnpm dev
```

Open the URL printed by Vite (e.g., `http://localhost:5173/starmap/`). The first query will fetch the model once; subsequent queries are fast.

### Provide the DuckDB file
The UI expects the database at `public/data/starscope.duckdb`. Copy from the pipeline cache:

```bash
mkdir -p /Users/szafar/Downloads/dev/starscope/ui/public/data
cp -f /Users/szafar/Downloads/dev/starscope/scripts/.cache/starscope.duckdb \
  /Users/szafar/Downloads/dev/starscope/ui/public/data/starscope.duckdb
```

Notes:
- The file is large (~192 MB). This is fine for local dev; Pages will host a published artifact in production.
- The app fetches this file as binary and loads it via `registerFileBuffer` in readonly mode.

### How search works
1. On submit, `useEmbedder` embeds the query (`Xenova/bge-small-en-v1.5`, quantized; mean‑pooled, L2‑normalized).
2. `useSearch` checks for `chunk_vectors_arr`:
   - If present: uses `array_cosine_distance` over `chunk_vectors_arr.embedding`, aggregates `MIN` per `repo_id` and sorts.
   - Else: selects one vector per repo from `chunk_vectors` and computes JS cosine.
3. Joins results with `repos` for name/desc/lang/stars and builds the GitHub URL.

### Deployment
- Vite `base` is set to `/starmap/` in `vite.config.ts` for GitHub Pages.
- For production, publish the DuckDB artifact to the Pages path (e.g., `/starmap/data/starscope.duckdb`).
