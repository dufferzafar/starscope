list:
    just --list

sync:
    uv --native-tls sync --active

collect what:
    uv run --active scripts/collect.py {{what}}

preprocess:
    uv run --active scripts/preprocess.py

embed:
    uv run --active scripts/embed.py --rep-method medoid --reps-only

search term:
    uv run --active scripts/search.py '{{term}}'

search_db term:
    uv run --active scripts/search_db.py '{{term}}'

db:
    rm -f scripts/.cache/starscope.duckdb
    uv run --active scripts/build_db.py

build-ui:
    cd ui && pnpm build

gh-pages: build-ui
    cd ../starscope-pages && git pull && cp -r ../starscope/ui/dist/* . && git add . && git commit -m "Deploy UI $(date -u +"%Y-%m-%dT%H:%M:%SZ")" && git push