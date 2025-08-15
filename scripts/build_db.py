#!/usr/bin/env python3

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import duckdb
from tqdm import tqdm

# Default locations
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_DB = CACHE_DIR / "starscope.duckdb"

# Cache files
STARRED_JSON = CACHE_DIR / "starred.json"
REPOS_JSONL = CACHE_DIR / "repos.jsonl"
READMES_CLEAN_JSONL = CACHE_DIR / "readmes_clean.jsonl"
READMES_RAW_JSONL = CACHE_DIR / "readmes.jsonl"
CHUNKS_JSONL = CACHE_DIR / "chunks.jsonl"
CHUNK_EMB_JSONL = CACHE_DIR / "chunk_embeddings.jsonl"
EMB_META_JSON = CACHE_DIR / "embeddings_meta.json"


def ensure_parent(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def create_tables(conn: duckdb.DuckDBPyConnection) -> None:
	# Minimal, URL-free schemas per instructions
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
		  starred_at TIMESTAMP
		);
		"""
	)
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS readmes (
		  repo_id BIGINT PRIMARY KEY,
		  cleaned_text TEXT NOT NULL,
		  hash TEXT
		);
		"""
	)
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS chunks (
		  repo_id BIGINT NOT NULL,
		  chunk_id INTEGER NOT NULL,
		  text TEXT NOT NULL,
		  PRIMARY KEY (repo_id, chunk_id)
		);
		"""
	)
	# Chunk-level embeddings (separate from repo_vectors aggregation)
	conn.execute(
		"""
		CREATE TABLE IF NOT EXISTS chunk_vectors (
		  repo_id BIGINT NOT NULL,
		  chunk_id INTEGER NOT NULL,
		  embedding FLOAT[],
		  PRIMARY KEY (repo_id, chunk_id)
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
	rows: List[Tuple[int, str, Optional[str], Optional[str], Optional[int], Optional[str], Optional[str]]] = []
	pbar = tqdm(desc="Loading repos.jsonl", unit="rec")
	for rec in load_jsonl_iter(path):
		# Keep only useful columns; skip URLs
		repo_id = rec.get("id")
		full_name = rec.get("full_name")
		description = rec.get("description")
		language = rec.get("language")
		stars = rec.get("stargazers_count")
		updated_at = rec.get("updated_at")
		pushed_at = rec.get("pushed_at")
		if repo_id is None or full_name is None:
			pbar.update(1)
			continue
		rows.append(
			(
				int(repo_id),
				str(full_name),
				str(description) if description is not None else None,
				str(language) if language is not None else None,
				int(stars) if isinstance(stars, int) else None,
				str(updated_at) if updated_at is not None else None,
				str(pushed_at) if pushed_at is not None else None,
			)
		)
		pbar.update(1)
	pbar.close()
	if not rows:
		print("[repos] No rows to insert")
		return 0
	print(f"[repos] Inserting {len(rows)} rows…")
	conn.executemany(
		"""
		INSERT OR REPLACE INTO repos (repo_id, full_name, description, language, stars, updated_at, pushed_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
		""",
		rows,
	)
	return len(rows)


def insert_stars(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
	if not path.exists():
		print(f"[stars] File not found: {path}")
		return 0
	rows: List[Tuple[int, Optional[str]]] = []
	data = json.loads(path.read_text(encoding="utf-8"))
	pbar = tqdm(total=len(data) if isinstance(data, list) else None, desc="Loading starred.json", unit="rec")
	for rec in data:
		repo_id = rec.get("id")
		starred_at = rec.get("starred_at")
		if repo_id is None:
			pbar.update(1)
			continue
		rows.append((int(repo_id), str(starred_at) if starred_at is not None else None))
		pbar.update(1)
	pbar.close()
	if not rows:
		print("[stars] No rows to insert")
		return 0
	print(f"[stars] Inserting {len(rows)} rows…")
	conn.executemany(
		"""
		INSERT OR REPLACE INTO stars (repo_id, starred_at) VALUES (?, ?)
		""",
		rows,
	)
	return len(rows)


def insert_readmes(conn: duckdb.DuckDBPyConnection, clean_path: Path, raw_path: Path) -> int:
	rows: List[Tuple[int, str, Optional[str]]] = []
	if clean_path.exists():
		pbar = tqdm(desc="Loading readmes_clean.jsonl", unit="rec")
		for rec in load_jsonl_iter(clean_path):
			repo_id = rec.get("repo_id")
			cleaned_text = rec.get("cleaned_text")
			hash_hex = rec.get("hash")
			if repo_id is None or not isinstance(cleaned_text, str):
				pbar.update(1)
				continue
			rows.append((int(repo_id), cleaned_text, str(hash_hex) if hash_hex is not None else None))
			pbar.update(1)
		pbar.close()
	else:
		# Fallback: store raw README text if cleaned not available
		if raw_path.exists():
			pbar = tqdm(desc="Loading readmes.jsonl (raw)", unit="rec")
			for rec in load_jsonl_iter(raw_path):
				repo_full = rec.get("full_name")
				text = rec.get("text")
				_ = (repo_full, text)
				pbar.update(1)
			pbar.close()
	if not rows:
		print("[readmes] No rows to insert")
		return 0
	print(f"[readmes] Inserting {len(rows)} rows…")
	conn.executemany(
		"""
		INSERT OR REPLACE INTO readmes (repo_id, cleaned_text, hash)
		VALUES (?, ?, ?)
		""",
		rows,
	)
	return len(rows)


def insert_chunks(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
	if not path.exists():
		print(f"[chunks] File not found: {path}")
		return 0
	rows: List[Tuple[int, int, str]] = []
	pbar = tqdm(desc="Loading chunks.jsonl", unit="rec")
	for rec in load_jsonl_iter(path):
		repo_id = rec.get("repo_id")
		chunk_id = rec.get("chunk_id")
		text = rec.get("text")
		if repo_id is None or chunk_id is None or not isinstance(text, str):
			pbar.update(1)
			continue
		rows.append((int(repo_id), int(chunk_id), text))
		pbar.update(1)
	pbar.close()
	if not rows:
		print("[chunks] No rows to insert")
		return 0
	print(f"[chunks] Inserting {len(rows)} rows…")
	conn.executemany(
		"""
		INSERT OR REPLACE INTO chunks (repo_id, chunk_id, text)
		VALUES (?, ?, ?)
		""",
		rows,
	)
	return len(rows)


def insert_chunk_vectors(conn: duckdb.DuckDBPyConnection, path: Path) -> int:
	if not path.exists():
		print(f"[chunk_vectors] File not found: {path}")
		return 0
	rows: List[Tuple[int, int, List[float]]] = []
	pbar = tqdm(desc="Loading chunk_embeddings.jsonl", unit="rec")
	for rec in load_jsonl_iter(path):
		repo_id = rec.get("repo_id")
		chunk_id = rec.get("chunk_id")
		emb = rec.get("embedding")
		if repo_id is None or chunk_id is None or not isinstance(emb, list):
			pbar.update(1)
			continue
		# Ensure float list
		try:
			vec = [float(x) for x in emb]
		except Exception:
			pbar.update(1)
			continue
		rows.append((int(repo_id), int(chunk_id), vec))
		pbar.update(1)
	pbar.close()
	if not rows:
		print("[chunk_vectors] No rows to insert")
		return 0
	print(f"[chunk_vectors] Inserting {len(rows)} rows…")
	conn.executemany(
		"""
		INSERT OR REPLACE INTO chunk_vectors (repo_id, chunk_id, embedding)
		VALUES (?, ?, ?)
		""",
		rows,
	)
	return len(rows)


def create_secondary_indexes(conn: duckdb.DuckDBPyConnection) -> None:
	print("Creating secondary indexes (btree)…")
	conn.execute("CREATE INDEX IF NOT EXISTS idx_repos_language ON repos(language)")
	conn.execute("CREATE INDEX IF NOT EXISTS idx_repos_stars ON repos(stars)")
	conn.execute("CREATE INDEX IF NOT EXISTS idx_stars_starred_at ON stars(starred_at)")
	conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_repo ON chunks(repo_id)")
	conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_vectors_repo ON chunk_vectors(repo_id)")
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
		# Opt-in to experimental persistence if available
		try:
			conn.execute("SET hnsw_enable_experimental_persistence = true;")
		except Exception:
			pass
		# Materialize fixed-size ARRAY table for VSS (FLOAT[dim])
		conn.execute(
			f"""
			CREATE TABLE IF NOT EXISTS chunk_vectors_arr AS
			SELECT repo_id, chunk_id, CAST(embedding AS FLOAT[{dim}]) AS embedding
			FROM chunk_vectors
			WHERE embedding IS NOT NULL
			"""
		)
		print("[vss] Building HNSW index on chunk_vectors_arr.embedding (cosine)…")
		conn.execute(
			"""
			CREATE INDEX IF NOT EXISTS idx_chunk_vectors_hnsw
			ON chunk_vectors_arr USING HNSW (embedding)
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
			conn.execute("DELETE FROM chunk_vectors")
			conn.execute("DELETE FROM chunks")
			conn.execute("DELETE FROM readmes")
			conn.execute("DELETE FROM stars")
			conn.execute("DELETE FROM repos")
			conn.execute("DROP INDEX IF EXISTS idx_chunk_vectors_hnsw")
			conn.execute("DROP TABLE IF EXISTS chunk_vectors_arr")

		# Resolve files
		starred = cache_dir / STARRED_JSON.name
		repos = cache_dir / REPOS_JSONL.name
		readmes_clean = cache_dir / READMES_CLEAN_JSONL.name
		readmes_raw = cache_dir / READMES_RAW_JSONL.name
		chunks = cache_dir / CHUNKS_JSONL.name
		chunk_emb = cache_dir / CHUNK_EMB_JSONL.name

		print("Loading repos…")
		n_repos = insert_repos(conn, repos)
		print("Loading stars…")
		n_stars = insert_stars(conn, starred)
		print("Loading readmes…")
		n_readmes = insert_readmes(conn, readmes_clean, readmes_raw)
		print("Loading chunks…")
		n_chunks = insert_chunks(conn, chunks)
		print("Loading chunk_vectors…")
		n_vecs = insert_chunk_vectors(conn, chunk_emb)

		create_secondary_indexes(conn)
		create_vss_index(conn, cache_dir)

		conn.commit()
		print("Commit complete.")

		print(
			f"Loaded into {db_path}: repos={n_repos}, stars={n_stars}, readmes={n_readmes}, chunks={n_chunks}, chunk_vectors={n_vecs}"
		)
	finally:
		conn.close()
		print("Closed DB connection.")


def parse_args() -> Tuple[Path, Path, bool]:
	p = argparse.ArgumentParser(description="Create DuckDB and load Starscope cache data")
	p.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to DuckDB file to create/update")
	p.add_argument("--cache-dir", type=Path, default=CACHE_DIR, help="Path to .cache directory")
	p.add_argument("--truncate", action="store_true", help="Delete existing table contents before load")
	a = p.parse_args()
	return a.db, a.cache_dir, a.truncate


if __name__ == "__main__":
	db_path, cache_dir, truncate = parse_args()
	run(db_path, cache_dir, truncate) 