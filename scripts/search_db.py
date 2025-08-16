#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_DB = CACHE_DIR / "starscope.duckdb"
EMB_META_JSON = CACHE_DIR / "embeddings_meta.json"


@dataclass
class Args:
    query: str
    db_path: Path
    top_k: int
    oversample: int
    unique: bool
    device: Optional[str]
    model_name: Optional[str]
    snippet_chars: int


# --- SSL disable for HF downloads (temporary patch) --------------------------


def _patch_requests_ssl_disable() -> Optional[tuple]:
    try:
        import requests  # type: ignore
        try:
            import urllib3  # type: ignore
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        original_request = requests.sessions.Session.request  # type: ignore[attr-defined]

        def insecure_request(self, method, url, **kwargs):  # type: ignore[override]
            kwargs.setdefault("verify", False)
            return original_request(self, method, url, **kwargs)

        requests.sessions.Session.request = insecure_request  # type: ignore[attr-defined]
        return (requests, original_request)
    except Exception:
        return None


def _restore_requests_ssl(patch_state: Optional[tuple]) -> None:
    if not patch_state:
        return
    try:
        requests, original_request = patch_state
        requests.sessions.Session.request = original_request  # type: ignore[attr-defined]
    except Exception:
        pass


# --- Helpers -----------------------------------------------------------------


def load_meta(path: Path) -> Dict[str, object]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def ensure_model(model_name: str, device: str) -> SentenceTransformer:
    patch_state = _patch_requests_ssl_disable()
    try:
        model = SentenceTransformer(model_name, device=device)
    finally:
        _restore_requests_ssl(patch_state)
    return model


def detect_device(preferred: Optional[str] = None) -> str:
    try:
        import torch  # type: ignore
        if preferred:
            pref = preferred.lower()
            if pref == "cuda" and torch.cuda.is_available():
                return "cuda"
            if pref == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
            if pref == "cpu":
                return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    except Exception:
        return "cpu"


def get_embedding_dim(conn: duckdb.DuckDBPyConnection) -> int:
    # Prefer fixed-size array table for VSS
    try:
        dim = conn.execute("SELECT array_length(embedding) FROM repo_reps_arr LIMIT 1").fetchone()
        if dim and isinstance(dim[0], int) and dim[0] > 0:
            return dim[0]
    except Exception:
        pass
    # Fallback to list length from base table
    try:
        dim = conn.execute("SELECT list_length(embedding) FROM repo_reps LIMIT 1").fetchone()
        if dim and isinstance(dim[0], int) and dim[0] > 0:
            return dim[0]
    except Exception:
        pass
    # Final fallback to metadata file
    try:
        meta = load_meta(EMB_META_JSON)
        return int(meta.get("dim", 0))
    except Exception:
        return 0


def build_array_literal(vec: np.ndarray, dim: int) -> str:
    # Build a DuckDB FLOAT[dim] literal like [0.1, 0.2, ...]::FLOAT[dim]
    vals = ", ".join(f"{float(x):.8f}" for x in vec.tolist())
    return f"[{vals}]::FLOAT[{dim}]"


def dedupe_hits_by_repo(rows: List[Tuple[int, int, float]], top_k: int) -> List[Tuple[int, int, float]]:
    # rows: (repo_id, rep_id, distance). Keep smallest distance per repo
    best: Dict[int, Tuple[int, int, float]] = {}
    for rid, rep_id, dist in rows:
        curr = best.get(rid)
        if curr is None or dist < curr[2]:
            best[rid] = (rid, rep_id, dist)
    deduped = sorted(best.values(), key=lambda r: r[2])
    return deduped[:top_k]


def format_and_print(rows: List[Tuple[int, int, float]], conn: duckdb.DuckDBPyConnection, snippet_chars: int) -> None:
    if not rows:
        return
    repo_ids = [r for (r, _, _) in rows]
    placeholders = ",".join(["?"] * len(repo_ids))
    # stars holds repo metadata now
    meta = conn.execute(
        f"SELECT repo_id, full_name, description, language, stargazers_count FROM stars WHERE repo_id IN ({placeholders})",
        repo_ids,
    ).fetchall()
    meta_idx = {int(rid): (full_name, description, language, stargazers_count) for (rid, full_name, description, language, stargazers_count) in meta}

    for rid, _rep_id, dist in rows:
        full_name, description, language, stars = meta_idx.get(int(rid), (f"repo:{rid}", "", None, None))
        url = f"https://github.com/{full_name}"
        desc = (description or "").strip()
        if len(desc) > snippet_chars:
            desc = desc[: snippet_chars - 1] + "…"
        row = [url + " ", str(language) if language is not None else "", f"⭐ {stars}" if isinstance(stars, int) else "", desc]
        print(" · ".join(x for x in row if x))


# --- Main --------------------------------------------------------------------


def run(args: Args) -> None:
    device = detect_device(args.device)
    meta = load_meta(EMB_META_JSON)
    model_name = args.model_name or str(meta.get("model", "BAAI/bge-small-en-v1.5"))
    model = ensure_model(model_name, device)

    # Connect DB and ensure VSS functions are available (even without index)
    conn = duckdb.connect(str(args.db_path))
    try:
        conn.execute("INSTALL vss; LOAD vss;")
    except Exception:
        pass

    dim = get_embedding_dim(conn)
    if dim <= 0:
        print("No embeddings in DB (dim=0)", file=sys.stderr)
        return

    # Create query embedding
    q: np.ndarray = model.encode([args.query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=False)[0]
    if q.shape[0] != dim:
        print(f"Query vector dim {q.shape[0]} != DB dim {dim}", file=sys.stderr)
        return

    # Determine available rows and source
    try:
        total_n = int(conn.execute("SELECT COUNT(*) FROM repo_reps_arr").fetchone()[0])
        source_table = "repo_reps_arr"
        embed_expr = "embedding"  # already FLOAT[dim]
    except Exception:
        total_n = int(conn.execute("SELECT COUNT(*) FROM repo_reps").fetchone()[0])
        source_table = "repo_reps"
        # Cast LIST(FLOAT) to typed array so array_cosine_distance binds
        embed_expr = f"CAST(embedding AS FLOAT[{dim}])"

    pre_k = min(args.top_k * (args.oversample if args.unique else 1), int(total_n))

    # Build SQL using cosine distance; full scan is fine without index at this scale
    q_lit = build_array_literal(q.astype(np.float32, copy=False), dim)
    sql = f"""
    SELECT repo_id, rep_id, array_cosine_distance({embed_expr}, {q_lit}) AS dist
    FROM {source_table}
    ORDER BY dist ASC
    LIMIT {pre_k}
    """
    rows = conn.execute(sql).fetchall()  # (repo_id, rep_id, dist)

    if args.unique:
        rows = dedupe_hits_by_repo(rows, args.top_k)

    format_and_print(rows, conn, args.snippet_chars)
    conn.close()


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Semantic search in DuckDB (VSS) over repo representatives")
    p.add_argument("query", type=str, help="Search query text")
    p.add_argument("--db", dest="db_path", type=Path, default=DEFAULT_DB, help="Path to starscope.duckdb")
    p.add_argument("--top-k", type=int, default=20, help="Number of results to return")
    p.add_argument("--oversample", type=int, default=5, help="Factor to oversample before deduping")
    p.add_argument("--no-unique", dest="unique", action="store_false", help="Allow multiple hits per repo")
    p.add_argument("--device", type=str, default=None, help="Force device: cuda|mps|cpu (auto if omitted)")
    p.add_argument("--model", dest="model_name", type=str, default=None, help="Override model name (defaults to meta)")
    p.add_argument("--snippet-chars", type=int, default=50, help="Max characters to show from repo description")
    p.set_defaults(unique=True)
    a = p.parse_args()
    return Args(
        query=a.query,
        db_path=a.db_path,
        top_k=a.top_k,
        oversample=a.oversample,
        unique=a.unique,
        device=a.device,
        model_name=a.model_name,
        snippet_chars=a.snippet_chars,
    )


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) 