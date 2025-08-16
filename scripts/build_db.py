#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import duckdb

# Default locations
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_DB = CACHE_DIR / "starscope.duckdb"

# Cache files
STARRED_JSON = CACHE_DIR / "starred.json"
REPOS_JSONL = CACHE_DIR / "repos.jsonl"
EMB_META_JSON = CACHE_DIR / "embeddings_meta.json"
REPO_REPS_JSONL = CACHE_DIR / "repo_vectors.jsonl"


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
    # Minimal schemas (no raw text tables)
    conn.execute(
        """
		CREATE TABLE IF NOT EXISTS repos (
		  repo_id BIGINT PRIMARY KEY,
		  full_name TEXT NOT NULL,
		  description TEXT,
		  language TEXT,
		  stars INTEGER,
		  updated_at TIMESTAMP,
		  pushed_at TIMESTAMP
		);
		"""
    )
    conn.execute(
        """
		CREATE TABLE IF NOT EXISTS stars (
		  repo_id BIGINT PRIMARY KEY,
		  full_name TEXT,
		  html_url TEXT,
		  description TEXT,
		  language TEXT,
		  stargazers_count INTEGER,
		  pushed_at TIMESTAMP,
		  updated_at TIMESTAMP,
		  starred_at TIMESTAMP
		);
		"""
    )
    # Repo-level representatives
    conn.execute(
        """
		CREATE TABLE IF NOT EXISTS repo_reps (
		  repo_id BIGINT NOT NULL,
		  rep_id INTEGER NOT NULL,
		  size INTEGER,
		  embedding FLOAT[],
		  PRIMARY KEY (repo_id, rep_id)
		);
		"""
    )
    print("Ensured DuckDB tables exist")


def load_jsonl_iter(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def insert_repos(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    if not path.exists():
        print(f"[repos] File not found: {path}")
        return 0
    try:
        print("[repos] Fast load via DuckDB JSON reader…")
        count = conn.execute(
            """
			SELECT COUNT(*)
			FROM read_json_auto(?)
			WHERE try_cast(id AS BIGINT) IS NOT NULL
			  AND full_name IS NOT NULL
			""",
            [str(path)],
        ).fetchone()[0]
        if count == 0:
            print("[repos] No rows to insert")
            return 0
        conn.execute(
            """
			INSERT OR REPLACE INTO repos (
				repo_id, full_name, description, language, stars, updated_at, pushed_at
			)
			SELECT
				CAST(id AS BIGINT) AS repo_id,
				CAST(full_name AS TEXT) AS full_name,
				CAST(description AS TEXT) AS description,
				CAST(language AS TEXT) AS language,
				try_cast(stargazers_count AS INTEGER) AS stars,
				try_cast(updated_at AS TIMESTAMP) AS updated_at,
				try_cast(pushed_at AS TIMESTAMP) AS pushed_at
			FROM read_json_auto(?)
			WHERE try_cast(id AS BIGINT) IS NOT NULL
			  AND full_name IS NOT NULL
			""",
            [str(path)],
        )
        print(f"[repos] Inserted {int(count)} rows via fast path")
        return int(count)
    except Exception as e:
        print(f"[repos] Fast load failed ({e}); no fallback implemented")
        return 0


def insert_stars(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    if not path.exists():
        print(f"[stars] File not found: {path}")
        return 0
    try:
        print("[stars] Fast load via DuckDB JSON reader…")
        count = conn.execute(
            """
			SELECT COUNT(*)
			FROM read_json_auto(?)
			WHERE try_cast(id AS BIGINT) IS NOT NULL
			""",
            [str(path)],
        ).fetchone()[0]
        if count == 0:
            print("[stars] No rows to insert")
            return 0
        conn.execute(
            """
			INSERT OR REPLACE INTO stars (
				repo_id, full_name, html_url, description, language,
				stargazers_count, pushed_at, updated_at, starred_at
			)
			SELECT
				CAST(id AS BIGINT) AS repo_id,
				CAST(full_name AS TEXT) AS full_name,
				CAST(html_url AS TEXT) AS html_url,
				CAST(description AS TEXT) AS description,
				CAST(language AS TEXT) AS language,
				try_cast(stargazers_count AS INTEGER) AS stargazers_count,
				try_cast(pushed_at AS TIMESTAMP) AS pushed_at,
				try_cast(updated_at AS TIMESTAMP) AS updated_at,
				try_cast(starred_at AS TIMESTAMP) AS starred_at
			FROM read_json_auto(?)
			WHERE try_cast(id AS BIGINT) IS NOT NULL
			""",
            [str(path)],
        )
        print(f"[stars] Inserted {int(count)} rows via fast path")
        return int(count)
    except Exception as e:
        print(f"[stars] Fast load failed ({e}); no fallback implemented")
        return 0


def insert_repo_reps(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
    if not path.exists():
        print(f"[repo_reps] File not found: {path}")
        return 0
    try:
        print("[repo_reps] Fast load via DuckDB JSON reader…")
        count = conn.execute(
            """
			SELECT COUNT(*)
			FROM read_json_auto(?)
			WHERE try_cast(repo_id AS BIGINT) IS NOT NULL
			  AND try_cast(rep_id AS INTEGER) IS NOT NULL
			  AND embedding IS NOT NULL
			""",
            [str(path)],
        ).fetchone()[0]
        if count == 0:
            print("[repo_reps] No rows to insert")
            return 0
        conn.execute(
            """
			INSERT OR REPLACE INTO repo_reps (repo_id, rep_id, size, embedding)
			SELECT CAST(repo_id AS BIGINT) AS repo_id,
			       CAST(rep_id AS INTEGER) AS rep_id,
			       try_cast(size AS INTEGER) AS size,
			       CAST(embedding AS FLOAT[]) AS embedding
			FROM read_json_auto(?)
			WHERE try_cast(repo_id AS BIGINT) IS NOT NULL
			  AND try_cast(rep_id AS INTEGER) IS NOT NULL
			  AND embedding IS NOT NULL
			""",
            [str(path)],
        )
        print(f"[repo_reps] Inserted {int(count)} rows via fast path")
        return int(count)
    except Exception as e:
        print(f"[repo_reps] Fast load failed ({e}); no fallback implemented")
        return 0


def create_secondary_indexes(conn: duckdb.DuckDBPyConnection) -> None:
    print("Creating secondary indexes (btree)…")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repos_language ON repos(language)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repos_stars ON repos(stars)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_stars_starred_at ON stars(starred_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_repo_reps_repo ON repo_reps(repo_id)")
    print("Secondary indexes created")


def create_vss_index(conn: duckdb.DuckDBPyConnection, cache_dir: Path) -> None:
    try:
        meta_path = cache_dir / EMB_META_JSON.name
        if not meta_path.exists():
            print("[vss] embeddings_meta.json not found; skipping VSS index")
            return
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dim = int(meta.get("dim", 0))
        if dim <= 0:
            print("[vss] Invalid embedding dim; skipping VSS index")
            return
        print(f"[vss] Installing/loading VSS extension; embedding dim={dim}")
        conn.execute("INSTALL vss; LOAD vss;")
        try:
            conn.execute("SET hnsw_enable_experimental_persistence = true;")
        except Exception:
            pass
        conn.execute(
            f"""
			CREATE TABLE IF NOT EXISTS repo_reps_arr AS
			SELECT repo_id, rep_id, size, CAST(embedding AS FLOAT[{dim}]) AS embedding
			FROM repo_reps
			WHERE embedding IS NOT NULL
			"""
        )
        print("[vss] Building HNSW index on repo_reps_arr.embedding (cosine)…")
        conn.execute(
            """
			CREATE INDEX IF NOT EXISTS idx_repo_reps_hnsw
			ON repo_reps_arr USING HNSW (embedding)
			WITH (metric = 'cosine')
			"""
        )
        print("[vss] HNSW index ready")
    except Exception as e:
        print(f"[vss] Skipped VSS index: {e}")


def run(db_path: Path, cache_dir: Path, truncate: bool) -> None:
    ensure_parent(db_path)
    print(f"Opening DB at {db_path}")
    conn = duckdb.connect(str(db_path))
    try:
        create_tables(conn)

        # Optionally truncate before load
        if truncate:
            print("Truncating existing tables…")
            conn.execute("DELETE FROM repo_reps")
            conn.execute("DELETE FROM stars")
            conn.execute("DELETE FROM repos")
            conn.execute("DROP INDEX IF EXISTS idx_repo_reps_hnsw")
            conn.execute("DROP TABLE IF EXISTS repo_reps_arr")

        # Resolve files
        starred = cache_dir / STARRED_JSON.name
        repos = cache_dir / REPOS_JSONL.name
        repo_reps = cache_dir / REPO_REPS_JSONL.name

        print("Loading repos…")
        n_repos = insert_repos(conn, repos)
        print("Loading stars…")
        n_stars = insert_stars(conn, starred)
        print("Loading repo_reps…")
        n_reps = insert_repo_reps(conn, repo_reps)

        create_secondary_indexes(conn)
        create_vss_index(conn, cache_dir)

        conn.commit()
        print("Commit complete.")

        print(
            f"Loaded into {db_path}: repos={n_repos}, stars={n_stars}, repo_reps={n_reps}"
        )
    finally:
        conn.close()
        print("Closed DB connection.")


def parse_args() -> Tuple[Path, Path, bool]:
    p = argparse.ArgumentParser(
        description="Create DuckDB and load Starscope cache data (repo representatives only)"
    )
    p.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="Path to DuckDB file to create/update",
    )
    p.add_argument(
        "--cache-dir", type=Path, default=CACHE_DIR, help="Path to .cache directory"
    )
    p.add_argument(
        "--truncate",
        action="store_true",
        help="Delete existing table contents before load",
    )
    a = p.parse_args()
    return a.db, a.cache_dir, a.truncate


if __name__ == "__main__":
    db_path, cache_dir, truncate = parse_args()
    run(db_path, cache_dir, truncate)
