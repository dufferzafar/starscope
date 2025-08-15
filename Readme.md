# GitHub Star Scope

Experiments with large language models and my starred github repositories.

## Phase 1 - Semantic Search

* Allow searching for stars not just via metadata (rust) 
    - But also by their semantic content in plain english

* Options for vector embeddings
    - First try something like `BAAI/bge-small-en-v1.5`
        - List other freely available alternatives
    - OpenAI's API
        - Need to estimate cost

* The frontend should be in vue.js & use tailwind css
    - It should be minimal and only use 3rd party libs when absolutely necessary

* Could use d3.js or something else based on webgl to display clusters of repositories
    - How will the clustering happen? via embeddings?

* Single .duckdb file for storage
    - metdata, text chunks, embeddings etc.
    - can be loaded from the web (via wasm) or from cli

## Phase 2 - Clustering visulisation

## Prior Art

* https://github.com/LuisReinoso/github-stars-semantic-search
    - Live demo: https://luisreinoso.dev/github-stars-semantic-search/
    - OpenAI Embeddings, stored in PgLite
    - Processes Readme files
    - Requires auth tokens!

* https://github.com/SushantDaga/github-stars-search
    - "intelligent chunking strategies"
    - BAAI/bge-small-en-v1.5 via txtai
    - hybrid search combining neural embeddings and BM25 keyword search
    - incremental updates for newly starred repositories

* https://github.com/JaosnHsieh/github-star-search
    - No embeddings, just fuzzy search

* https://github.com/prabirshrestha/gh-stars
    - Rust
    - Keyword + vector search using embeddings

* https://github.com/BjornMelin/stardex
    - d3.js viz of repo clusters
    - scikit-learn Clustering
    - No embeddings?