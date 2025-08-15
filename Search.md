# 🎯 Goals & Constraints

* **Goal:** Natural-language search over your *starred* repos using README semantics.
* **Hosting:** GitHub Pages only (static), GitHub Actions for build.
* **Data store:** A single `starmap.duckdb` file (portable to CLI + browser).
* **Zero cost, open source.**
* **Scale:** \~9k repos today, growing; must remain fast on desktop browsers.

---

# 🏗️ System Overview (Phase 1 only)

```
GitHub Actions (CI, batch) ──▶ build starmap.duckdb ──▶ GitHub Pages
                                             ▲
                                   embeddings + metadata
```

**At build time (CI):**

* Fetch your stars + READMEs, clean, chunk, embed, aggregate to **one vector per repo**.
* Save vectors + metadata to **DuckDB**.
* (Optional) also store **quantized** vectors for smaller downloads.

**At runtime (browser on GitHub Pages):**

* Load `duckdb-wasm` and open `starmap.duckdb` directly from `/starmap/data/…`.
* Load a **small, open-source embedding model in-browser** to embed the user’s query.
* Pull the **repo-level embedding matrix** from DuckDB (once), compute cosine similarity in JS (or via DuckDB UDF), rank, show results.
* Optional hybrid: combine semantic score with lightweight BM25 from README snippets for extra precision.

---

# 🧭 User Experience (Phase 1)

* Simple search bar (“Search your stars by topic, task or stack…”).
* Filters: language, stars ≥ N, date starred.
* Results: repo name, description, top README snippet, badges (language, stars), “Open on GitHub”.
* “Related repos” (optional): top-k nearest neighbors (precomputed in CI).
* All on one page at `dufferzafar.com/starmap`.

---

# 🧱 Data & Index Design (DuckDB)

*(Conceptual—no SQL yet)*

* **`repos`**
  `repo_id, full_name, html_url, description, language, stars, starred_at, last_updated`

* **`readmes`**
  `repo_id, cleaned_text, size_tokens, hash`

* **`chunks`** *(optional for Phase 1; keep minimal)*
  `repo_id, chunk_id, text, size_tokens`

* **`repo_vectors`** *(one vector per repo for fast Phase-1 search)*
  `repo_id, embedding FLOAT[dim]`
  – **Aggregation**: average/median of chunk vectors, or the **medoid** chunk (closest to mean) to keep a “real” vector.

* **`repo_vectors_q8`** *(optional)*
  `repo_id, embedding INT8[dim], scale FLOAT`
  – 4× smaller payload; reconstruct approx float for cosine.

* **`neighbors`** *(optional)*
  `repo_id, neighbor_id, sim`
  – Precomputed top-k nearest neighbors to power “related repos” instantly.

---

# 🔄 CI Build Pipeline (GitHub Actions)

1. **Collect**

   * List all starred repos (paged API).
   * Fetch each README via `/readme` endpoint.
   * Cache ETags/SHA to avoid re-downloading unchanged files.

2. **Preprocess**

   * Strip badges/boilerplate, normalize whitespace.
   * Split into chunks (\~400–800 tokens) if long.

3. **Embed (batch)**

   * **Model (open-source, CPU-friendly):**

     * `BAAI/bge-small-en-v1.5` *(384-d, high quality for search)*, or
     * `all-MiniLM-L6-v2` *(384-d, very fast, tiny)*
   * Store **chunk vectors** temporarily; **aggregate** → **one vector per repo**.
   * (Optional) **quantize** repo vectors to int8 for a tiny on-page footprint.

4. **Index & Extras**

   * (Optional) Build a **top-k neighbor list** per repo (cosine in Python).
   * Persist everything into **`starmap.duckdb`**.

5. **Publish**

   * Commit `starmap.duckdb` (and a tiny `version.json`) to the Pages branch.
   * Nightly schedule + manual dispatch. (Stars change—no webhooks needed.)

**Why this works:** CI does the heavy lifting once; the browser only downloads a compact matrix and computes cosine similarity very fast for 9k items.

---

# ⚙️ Runtime Search Plan (Browser)

* **Model for query embedding:**
  Load a small SBERT in the browser *on demand* using **`transformers.js`**.

  * Recommended: a quantized MiniLM/BGE variant (keeps bundle small).
  * Zero server calls, zero cost.

* **Fetching the vectors:**

  * Use **`duckdb-wasm`** to open `starmap.duckdb` from GitHub Pages.
  * Query `repo_vectors` and get a typed array (Float32 or int8+scale).
  * Cache in **IndexedDB** to avoid re-downloading.

* **Scoring:**

  * Compute cosine similarity between the query vector and all repo vectors.
  * If needed, pre-filter candidates (language, stars) via DuckDB, then score fewer vectors.
  * Return top-N with metadata from `repos`.
  * (Optional) **Hybrid score** = `0.8 * cosine + 0.2 * BM25(title+desc+top_snip)` for better exact term handling.

* **Performance targets:**

  * **Vector size (384-d):** \~13–14 MB float32 for 9k repos; **\~3.5 MB** if int8-quantized.
  * **Cosine over 9k**: <10 ms in a Web Worker on modern desktop; well under 30 ms even on modest laptops.
  * First-load model fetch happens once; subsequent searches are instant.

---

# 📈 Quality Levers (Phase 1)

* **Aggregation strategy:** prefer **medoid** per repo (often beats naive averaging).
* **Stopwords/badges removal** from README before embedding—cuts noise.
* **Query hygiene:** lowercase, trim code-only queries, optional synonyms (DIY list) for common techs.
* **Hybrid re-rank:** a simple BM25 pass over title/description can fix corner-cases (“react table csv export” etc.).

---

# 🧪 Size & Performance Budget (today, \~9k repos)

* **DuckDB file** (repos + vectors + minimal text): ≈ 20–40 MB compressed (very reasonable).
* **Download strategy:**

  * Lazy-load only what’s needed (vectors first; chunks not needed in Phase 1).
  * Cache with `Cache-Control` + `ETag`; re-download only on version change.

---

# 🔐 Privacy, Tokens, Licenses

* Build uses a **GitHub token** with read scope (Actions secret); nothing exposed on Pages.
* All inference **local** (CI + browser). No user data leaves the page.
* Models: **Apache 2.0 / MIT** open-source options ensure clean licensing.

---

# 🛣️ Roadmap Hooks into Phase 2 (Later)

* Add `coords` + `clusters` tables to the same DuckDB file when you start visualization.
* Keep the **repo-level vector** as the canonical representation; it powers both search and layout.

---

# ✅ Phase-1 Deliverables

1. **Data contract**

   * `starmap.duckdb` with: `repos`, `repo_vectors` (± `repo_vectors_q8`), `neighbors` (optional).

2. **CI workflows**

   * Collect → Preprocess → Embed → Aggregate → Quantize → Build DB → Publish.

3. **Static UI (Pages)**

   * Search bar + filters + results list.
   * `duckdb-wasm` loader, `transformers.js` query embedding, JS cosine scorer.
   * IndexedDB caching & Web Worker for scoring.

4. **Docs**

   * README describing architecture, model choice, how to rebuild, how to query via DuckDB CLI.

---

## Opinionated Defaults (to move fast)

* **Embedding model:** `BAAI/bge-small-en-v1.5` (CI) + **quantized** MiniLM/BGE (browser for query).
* **Vector dim:** 384.
* **Aggregation:** medoid of README chunks.
* **Optional hybrid:** small BM25 over title/description.
* **Quantization:** int8 for on-page vectors (keep float32 in DuckDB for accuracy if you like).
* **Update cadence:** nightly CI.

This plan gives you **excellent semantic search** now, keeps **everything on GitHub**, and scales cleanly as your stars grow—while laying perfect foundations for the Phase-2 visualization.
