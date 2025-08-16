#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

# Ensure repo root on sys.path for module imports
if str(Path(__file__).resolve().parent.parent) not in sys.path:
	sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import existing script modules
from scripts import collect  # type: ignore
from scripts import build_db  # type: ignore
from scripts import preprocess  # type: ignore
from scripts import embed  # type: ignore

import duckdb
import httpx


@dataclass
class Args:
	cache_dir: Path
	db_path: Path
	model_name: str
	batch_size: int
	device: Optional[str]
	reps_method: str
	reps_per_repo: int


def _iso_to_epoch(ts: Optional[str]) -> Optional[int]:
	if not ts:
		return None
	try:
		dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
		return int(dt.timestamp())
	except Exception:
		return None


def get_db_max_starred_at(conn: duckdb.DuckDBPyConnection) -> Optional[int]:
	try:
		row = conn.execute("SELECT max(starred_at) FROM stars").fetchone()
		if not row or row[0] is None:
			return None
		# DuckDB returns datetime; convert to epoch
		val = row[0]
		if hasattr(val, "timestamp"):
			return int(val.timestamp())
		return None
	except Exception:
		return None


def filter_new_starreds(starred_list: List[Dict[str, object]], cutoff_epoch: Optional[int]) -> List[Dict[str, object]]:
	if cutoff_epoch is None:
		return starred_list
	out: List[Dict[str, object]] = []
	for rec in starred_list:
		ts = _iso_to_epoch(rec.get("starred_at") if isinstance(rec, dict) else None)
		if ts is None or ts > cutoff_epoch:
			out.append(rec)
	return out


def write_subset_readmes(all_readmes_path: Path, target_names: Set[str], out_path: Path) -> int:
	count = 0
	with all_readmes_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
		for line in fin:
			if not line.strip():
				continue
			try:
				rec = json.loads(line)
			except Exception:
				continue
			fn = rec.get("full_name")
			tx = rec.get("text")
			if isinstance(fn, str) and fn in target_names and isinstance(tx, str):
				fout.write(json.dumps({"full_name": fn, "text": tx}, ensure_ascii=False) + "\n")
				count += 1
	return count


def run(args: Args) -> None:
	cache_dir = args.cache_dir
	cache_dir.mkdir(parents=True, exist_ok=True)

	# Paths (align with existing scripts)
	STARRED_JSON = collect.STARRED_JSON
	READMES_JSONL = collect.READMES_JSONL
	REPOS_JSONL = collect.REPOS_JSONL

	# 1) Ensure DB and tables exist
	build_db.ensure_parent(args.db_path)
	conn = duckdb.connect(str(args.db_path))
	try:
		build_db.create_tables(conn)
		conn.commit()
	finally:
		conn.close()

	# Row counts at start
	initial_stars = 0
	initial_reps = 0
	conn = duckdb.connect(str(args.db_path))
	try:
		initial_stars = int(conn.execute("SELECT COUNT(*) FROM stars").fetchone()[0])
		initial_reps = int(conn.execute("SELECT COUNT(*) FROM repo_reps").fetchone()[0])
	finally:
		conn.close()
	print(f"Initial counts: stars={initial_stars}, repo_reps={initial_reps}")

	# 2) Get DB cutoff time
	conn = duckdb.connect(str(args.db_path))
	try:
		cutoff_epoch = get_db_max_starred_at(conn)
	finally:
		conn.close()
	print(f"DB cutoff starred_at epoch: {cutoff_epoch}")

	# Seed starred.json from DB if missing/empty to prevent full historical fetch
	seed_needed = (not STARRED_JSON.exists()) or (STARRED_JSON.exists() and len(json.loads(STARRED_JSON.read_text(encoding="utf-8") or "[]")) == 0)
	if seed_needed:
		print("Seeding starred.json from DB stars table…")
		conn = duckdb.connect(str(args.db_path))
		try:
			rows = conn.execute(
				"""
				SELECT repo_id, full_name, html_url, description, language,
				       stargazers_count, pushed_at, updated_at, starred_at
				FROM stars
				"""
			).fetchall()
		finally:
			conn.close()
		seed_data: List[Dict[str, object]] = []
		def to_iso(ts) -> Optional[str]:
			if ts is None:
				return None
			if isinstance(ts, str):
				return ts
			try:
				return datetime.utcfromtimestamp(ts.timestamp()).strftime("%Y-%m-%dT%H:%M:%SZ")
			except Exception:
				return None
		for (repo_id, full_name, html_url, description, language, stargazers_count, pushed_at, updated_at, starred_at) in rows:
			seed_data.append(
				{
					"id": int(repo_id) if repo_id is not None else None,
					"full_name": full_name,
					"html_url": html_url,
					"description": description,
					"language": language,
					"stargazers_count": int(stargazers_count) if isinstance(stargazers_count, int) else stargazers_count,
					"pushed_at": to_iso(pushed_at),
					"updated_at": to_iso(updated_at),
					"starred_at": to_iso(starred_at),
				}
			)
		STARRED_JSON.write_text(json.dumps(seed_data, indent=2), encoding="utf-8")
		print(f"Seeded {len(seed_data)} records into {STARRED_JSON}")

	# 3) Update starred.json from GitHub (incremental handled inside)
	# Reuse collector 'stars' mode
	import asyncio
	asyncio.run(collect.main_async(mode="stars"))

	# Load starred.json and identify new repos after DB cutoff
	starred_list: List[Dict[str, object]] = []
	if STARRED_JSON.exists():
		starred_list = json.loads(STARRED_JSON.read_text(encoding="utf-8"))
	new_starreds = filter_new_starreds(starred_list, cutoff_epoch)
	new_repo_full_names: Set[str] = set(
		str(rec.get("full_name")) for rec in new_starreds if isinstance(rec.get("full_name"), str)
	)
	print(f"New starred repos since DB cutoff: {len(new_repo_full_names)}")

	# 4) Insert all stars (idempotent)
	conn = duckdb.connect(str(args.db_path))
	try:
		build_db.insert_stars(conn, STARRED_JSON)
		conn.commit()
	finally:
		conn.close()

	# 5) Fetch repo metadata for new repos (merging into repos.jsonl)
	async def fetch_repos_for(full_names: Set[str]) -> None:
		existing = collect.load_repos_jsonl(REPOS_JSONL)
		existing_map = {rec.get("full_name"): rec for rec in existing if rec.get("full_name")}
		existing_names = set(existing_map.keys())
		target_list = [{"full_name": fn} for fn in full_names if fn not in existing_names]
		if not target_list:
			return
		cfg = collect.GitHubConfig(token=collect.settings.GITHUB_TOKEN)
		timeout = httpx.Timeout(cfg.timeout_seconds, connect=cfg.timeout_seconds)
		limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
		async with httpx.AsyncClient(headers=collect.auth_headers(cfg.token), timeout=timeout, limits=limits):
			pass
		# Use collector helper directly (reusing client from main_async is more complex); fall back to repos mode
		await collect.main_async(mode="repos")

	asyncio.run(fetch_repos_for(new_repo_full_names))

	# 6) Fetch READMEs for new repos (append as they arrive)
	async def fetch_readmes_for(full_names: Set[str]) -> None:
		cfg = collect.GitHubConfig(token=collect.settings.GITHUB_TOKEN)
		timeout = httpx.Timeout(cfg.timeout_seconds, connect=cfg.timeout_seconds)
		limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
		existing_map = collect.load_readmes_jsonl(READMES_JSONL)
		existing_names = set(existing_map.keys())
		targets = [
			{"full_name": fn} for fn in full_names if isinstance(fn, str) and fn not in existing_names
		]
		if not targets:
			return
		append_lock = asyncio.Lock()
		async def on_result(full_name: str, text: str) -> None:
			line = json.dumps({"full_name": full_name, "text": text}, ensure_ascii=False) + "\n"
			async with append_lock:
				with READMES_JSONL.open("a", encoding="utf-8") as f:
					f.write(line)
		async with httpx.AsyncClient(headers=collect.auth_headers(cfg.token), timeout=timeout, limits=limits) as client:
			await collect.fetch_readmes_parallel(
				client=client,
				repos=targets,
				concurrency=collect.settings.MAX_CONCURRENCY,
				existing_full_names=existing_names,
				on_result=on_result,
			)

	asyncio.run(fetch_readmes_for(new_repo_full_names))

	# 7) Preprocess only new READMEs to temp files
	temp_readmes = args.cache_dir / "readmes_new.jsonl"
	written = write_subset_readmes(READMES_JSONL, new_repo_full_names, temp_readmes)
	print(f"Prepared {written} new READMEs for preprocessing")
	if written == 0:
		print("No new READMEs to process. Exiting.")
		return

	out_clean = args.cache_dir / "readmes_clean_new.jsonl"
	out_chunks = args.cache_dir / "chunks_new.jsonl"
	pp_args = preprocess.Args(
		readmes_jsonl=temp_readmes,
		starred_json=collect.STARRED_JSON,
		repos_jsonl=collect.REPOS_JSONL,
		out_clean=out_clean,
		out_chunks=out_chunks,
		chunk_size=512,
		overlap=50,
		chunker="hf:BAAI/bge-small-en-v1.5",
		encoding="cl100k_base",
		add_meta_chunk=True,
	)
	preprocess.run(pp_args)

	# 8) Generate embeddings for new chunks
	out_emb = args.cache_dir / "chunk_embeddings_new.jsonl"
	emb_args = embed.Args(
		chunks_jsonl=out_chunks,
		out_jsonl=out_emb,
		model_name=args.model_name,
		batch_size=args.batch_size,
		device=args.device,
		normalize=True,
		limit=None,
		make_repo_reps=False,
		reps_per_repo=args.reps_per_repo,
		reps_out_jsonl=args.cache_dir / "repo_vectors_new.jsonl",
		rep_method=args.reps_method,
		reps_only=False,
		reps_src_jsonl=out_emb,
	)
	embed.run(emb_args)

	# 9) Generate repo representatives for these embeddings
	reps_out = args.cache_dir / "repo_vectors_new.jsonl"
	embed.compute_and_save_repo_reps(out_emb, reps_out, args.reps_per_repo, args.reps_method)

	# 10) Insert new reps into DB
	conn = duckdb.connect(str(args.db_path))
	try:
		build_db.insert_repo_reps(conn, reps_out)
		conn.commit()
	finally:
		conn.close()

	# Row counts at end
	final_stars = 0
	final_reps = 0
	conn = duckdb.connect(str(args.db_path))
	try:
		final_stars = int(conn.execute("SELECT COUNT(*) FROM stars").fetchone()[0])
		final_reps = int(conn.execute("SELECT COUNT(*) FROM repo_reps").fetchone()[0])
	finally:
		conn.close()
	delta_stars = final_stars - initial_stars
	delta_reps = final_reps - initial_reps
	print(f"Final counts: stars={final_stars} (Δ{delta_stars}), repo_reps={final_reps} (Δ{delta_reps})")

	print("CI run complete.")


def parse_args() -> Args:
	p = argparse.ArgumentParser(description="Run CI pipeline to incrementally update DB and cache")
	p.add_argument("--cache-dir", type=Path, default=collect.CACHE_DIR, help="Path to .cache directory")
	p.add_argument("--db", dest="db_path", type=Path, default=build_db.DEFAULT_DB, help="Path to DuckDB file")
	p.add_argument("--model", dest="model_name", type=str, default=embed.DEFAULT_MODEL, help="Embedding model name")
	p.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
	p.add_argument("--device", type=str, default=None, help="Force device: cuda|mps|cpu (auto if omitted)")
	p.add_argument("--rep-method", type=str, choices=["medoid", "centroid"], default="medoid", help="Representative method")
	p.add_argument("--reps-per-repo", type=int, default=1, help="Number of representatives per repo")
	a = p.parse_args()
	return Args(
		cache_dir=a.cache_dir,
		db_path=a.db_path,
		model_name=a.model_name,
		batch_size=a.batch_size,
		device=a.device,
		reps_method=a.rep_method,
		reps_per_repo=a.reps_per_repo,
	)


if __name__ == "__main__":
	run(parse_args()) 