list:
    just --list

sync:
    uv --native-tls sync --active

collect:
    uv run --active scripts/collect.py stars

preprocess:
    uv run --active scripts/preprocess.py

embed:
    uv run --active scripts/embed.py

# --use-faiss
search term:
    uv run --active scripts/search.py '{{term}}'

search_db term:
    uv run --active scripts/search_db.py '{{term}}'

db:
    uv run --active scripts/build_db.py