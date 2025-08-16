#!/usr/bin/env python3

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, Dict

import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CHUNKS_JSONL = CACHE_DIR / "chunks.jsonl"
OUT_EMB_JSONL = CACHE_DIR / "chunk_embeddings.jsonl"
OUT_META_JSON = CACHE_DIR / "embeddings_meta.json"
REPO_VEC_JSONL = CACHE_DIR / "repo_vectors.jsonl"
REPO_VEC_Q8_JSONL = CACHE_DIR / "repo_vectors_q8.jsonl"

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


@dataclass
class Args:
    chunks_jsonl: Path
    out_jsonl: Path
    model_name: str
    batch_size: int
    device: Optional[str]
    normalize: bool
    limit: Optional[int]
    # New options for repo representatives
    make_repo_reps: bool
    reps_per_repo: int
    reps_out_jsonl: Path
    rep_method: str
    # New: compute reps without re-embedding
    reps_only: bool
    reps_src_jsonl: Path


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


def iter_chunks(path: Path) -> Iterator[Tuple[int, int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            repo_id = rec.get("repo_id")
            chunk_id = rec.get("chunk_id")
            text = rec.get("text")
            if repo_id is None or chunk_id is None or not isinstance(text, str):
                continue
            yield int(repo_id), int(chunk_id), text


def count_chunks(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("repo_id") is None or rec.get("chunk_id") is None:
                continue
            text = rec.get("text")
            if isinstance(text, str):
                total += 1
    return total


def batch_iterator(iterable: Iterable[Tuple[int, int, str]], batch_size: int) -> Iterator[List[Tuple[int, int, str]]]:
    batch: List[Tuple[int, int, str]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def save_meta(path: Path, model_name: str, dim: int) -> None:
    meta = {"model": model_name, "dim": int(dim), "created_at": int(time.time())}
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


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


# ---------------- Repo representatives (centroids) ----------------

def iter_chunk_embeddings(path: Path) -> Iterator[Tuple[int, int, np.ndarray]]:
    """Yield (repo_id, chunk_id, embedding) from a JSONL of chunk embeddings."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rid = rec.get("repo_id")
            cid = rec.get("chunk_id")
            emb = rec.get("embedding")
            if rid is None or cid is None or not isinstance(emb, list):
                continue
            vec = np.asarray(emb, dtype=np.float32)
            yield int(rid), int(cid), vec


def kmeans_plus_plus_init(vectors: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Initialize k centers using k-means++ on (n, d) vectors."""
    n, d = vectors.shape
    centers = np.empty((k, d), dtype=np.float32)
    # Pick first center uniformly
    idx0 = int(rng.integers(0, n))
    centers[0] = vectors[idx0]
    # Distances squared to nearest center
    dists = np.full(n, np.inf, dtype=np.float32)
    for i in range(1, k):
        # Update distances-to-nearest-center
        diff = vectors - centers[i - 1]
        cur = np.einsum("nd,nd->n", diff, diff, dtype=np.float32)
        # Ensure non-negative due to numeric error
        cur = np.maximum(cur, 0.0)
        dists = np.minimum(dists, cur)
        # Choose next center proportional to distance^2; if degenerate, sample uniformly
        total = float(dists.sum())
        if not np.isfinite(total) or total <= 1e-12:
            idx = int(rng.integers(0, n))
        else:
            probs = (dists / total).astype(np.float64, copy=False)
            # Guard against rounding issues
            rem = 1.0 - probs.sum()
            if abs(rem) > 1e-12:
                probs = probs + rem / probs.size
                probs = np.clip(probs, 0.0, None)
                probs = probs / probs.sum()
            idx = int(rng.choice(n, p=probs))
        centers[i] = vectors[idx]
    return centers


def compute_centroids(vectors: np.ndarray, k: int, max_iters: int = 10, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Run a lightweight k-means to produce k centroids and cluster sizes.
    Returns (centroids[k, d], sizes[k]). If n < k, returns n centroids.
    """
    n, d = vectors.shape
    k_eff = min(k, n)
    if k_eff <= 0:
        return np.empty((0, d), dtype=np.float32), np.empty((0,), dtype=np.int32)
    if k_eff == 1:
        return vectors.mean(axis=0, dtype=np.float32, keepdims=True), np.array([n], dtype=np.int32)

    rng = np.random.default_rng(seed)
    centers = kmeans_plus_plus_init(vectors, k_eff, rng)

    for _ in range(max_iters):
        # Assign
        # Compute squared distances to centers: (n,k)
        # Using (x - c)^2 = x^2 - 2 x·c + c^2, but einsum is fine here for sizes ~1e4
        # Expand centers: compute dot then pick argmin
        dots = vectors @ centers.T  # (n, k)
        x2 = np.einsum("nd,nd->n", vectors, vectors)
        c2 = np.einsum("kd,kd->k", centers, centers)
        d2 = (x2[:, None] - 2 * dots + c2[None, :]).astype(np.float32)
        assign = np.argmin(d2, axis=1)

        # Update
        new_centers = np.zeros_like(centers)
        sizes = np.zeros(k_eff, dtype=np.int32)
        for i in range(k_eff):
            mask = assign == i
            cnt = int(mask.sum())
            if cnt == 0:
                # Re-seed an empty center to a random point
                j = int(rng.integers(0, n))
                new_centers[i] = vectors[j]
                sizes[i] = 1
            else:
                new_centers[i] = vectors[mask].mean(axis=0, dtype=np.float32)
                sizes[i] = cnt
        if np.allclose(new_centers, centers, atol=1e-4):
            centers = new_centers
            break
        centers = new_centers

    return centers.astype(np.float32, copy=False), sizes


def compute_and_save_repo_reps(embeddings_jsonl: Path, out_jsonl: Path, reps_per_repo: int, method: str = "centroid") -> None:
    if not embeddings_jsonl.exists():
        print(f"[repo_reps] Embeddings file not found: {embeddings_jsonl}")
        return
    if method not in ("centroid", "medoid"):
        print(f"[repo_reps] Unsupported method '{method}', defaulting to 'centroid'")
        method = "centroid"

    # Load all embeddings grouped by repo
    repo_to_vecs: Dict[int, List[np.ndarray]] = {}
    repo_to_meta: Dict[int, np.ndarray] = {}
    total = 0
    for rid, cid, vec in tqdm(iter_chunk_embeddings(embeddings_jsonl), desc="Loading embeddings for reps"):
        if cid == -1:
            # Keep metadata vector as dedicated representative
            repo_to_meta[rid] = vec
        else:
            lst = repo_to_vecs.get(rid)
            if lst is None:
                repo_to_vecs[rid] = [vec]
            else:
                lst.append(vec)
        total += 1

    if not repo_to_vecs and not repo_to_meta:
        print("[repo_reps] No embeddings found; skipping reps")
        return

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with out_jsonl.open("w", encoding="utf-8") as fout:
        # Iterate over union of repos that have meta or non-meta vecs
        all_repo_ids = set(repo_to_vecs.keys()) | set(repo_to_meta.keys())
        for rid in tqdm(sorted(all_repo_ids), desc="Computing repo reps"):
            # 1) Write metadata representative first if available
            meta_vec = repo_to_meta.get(rid)
            if meta_vec is not None:
                rec_meta = {
                    "repo_id": int(rid),
                    "rep_id": -1,
                    "size": 1,
                    "embedding": meta_vec.tolist(),
                }
                fout.write(json.dumps(rec_meta, ensure_ascii=False) + "\n")

            # 2) Representatives from non-metadata chunks
            vecs_list = repo_to_vecs.get(rid, [])
            if len(vecs_list) == 0:
                continue
            mat = np.asarray(vecs_list, dtype=np.float32)

            if method == "centroid":
                centers, sizes = compute_centroids(mat, reps_per_repo)
                for rep_id, (center, size) in enumerate(zip(centers, sizes)):
                    rec = {
                        "repo_id": int(rid),
                        "rep_id": int(rep_id),
                        "size": int(size),
                        "embedding": center.tolist(),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            else:  # medoid
                # Compute k centroids, then for each cluster pick the nearest actual vector (medoid proxy)
                centers, sizes = compute_centroids(mat, reps_per_repo)
                if centers.shape[0] == 0:
                    continue
                # Assign points to centers
                dots = mat @ centers.T  # (n, k)
                x2 = np.einsum("nd,nd->n", mat, mat)
                c2 = np.einsum("kd,kd->k", centers, centers)
                d2 = (x2[:, None] - 2 * dots + c2[None, :]).astype(np.float32)
                assign = np.argmin(d2, axis=1)
                for rep_id in range(centers.shape[0]):
                    mask = assign == rep_id
                    if not np.any(mask):
                        continue
                    cluster_vecs = mat[mask]
                    # nearest to centroid
                    cv = centers[rep_id]
                    diff = cluster_vecs - cv
                    dist = np.einsum("nd,nd->n", diff, diff, dtype=np.float32)
                    local_idx = int(np.argmin(dist))
                    # Map back to original mat index to take the exact vector
                    global_idx = np.nonzero(mask)[0][local_idx]
                    medoid_vec = mat[global_idx]
                    rec = {
                        "repo_id": int(rid),
                        "rep_id": int(rep_id),
                        "size": int(sizes[rep_id]) if rep_id < len(sizes) else int(mask.sum()),
                        "embedding": medoid_vec.tolist(),
                    }
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[repo_reps] Wrote representatives for {len(all_repo_ids)} repos to {out_jsonl}")
# ---------------- End repo representatives ----------------


def run(args: Args) -> None:
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # If only computing reps, skip embedding
    if args.reps_only:
        src = args.reps_src_jsonl if args.reps_src_jsonl else args.out_jsonl
        compute_and_save_repo_reps(src, args.reps_out_jsonl, args.reps_per_repo, args.rep_method)
        return

    device = detect_device(args.device)

    # Temporarily disable SSL verification for model download via huggingface_hub/requests
    patch_state = _patch_requests_ssl_disable()
    try:
        model = SentenceTransformer(args.model_name, device=device)
    finally:
        _restore_requests_ssl(patch_state)

    embedding_dim = int(model.get_sentence_embedding_dimension())

    # Persist simple metadata for downstream steps
    save_meta(OUT_META_JSON, args.model_name, embedding_dim)

    total_items = 0
    start_time = time.time()

    with args.out_jsonl.open("w", encoding="utf-8") as fout:
        total_available = count_chunks(args.chunks_jsonl)
        total_target = min(args.limit, total_available) if args.limit is not None else total_available
        pbar = tqdm(total=total_target, desc="Embedding chunks")
        for batch in batch_iterator(iter_chunks(args.chunks_jsonl), args.batch_size):
            if args.limit is not None and total_items >= args.limit:
                break

            # Trim batch if exceeding remaining quota
            if args.limit is not None:
                remaining = max(0, args.limit - total_items)
                if len(batch) > remaining:
                    batch = batch[:remaining]
            if not batch:
                break

            repo_ids = [r for (r, _, _) in batch]
            chunk_ids = [c for (_, c, _) in batch]
            texts = [t for (_, _, t) in batch]

            embeddings: np.ndarray = model.encode(
                texts,
                batch_size=args.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=args.normalize,
            )
            # Ensure float32 for size
            if embeddings.dtype != np.float32:
                embeddings = embeddings.astype(np.float32, copy=False)

            for rid, cid, emb in zip(repo_ids, chunk_ids, embeddings):
                rec = {
                    "repo_id": int(rid),
                    "chunk_id": int(cid),
                    "embedding": emb.tolist(),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_items += 1
            pbar.update(len(batch))
        pbar.close()

    elapsed = time.time() - start_time
    print(
        f"Wrote {total_items} chunk embeddings to {args.out_jsonl} | dim={embedding_dim} | model={args.model_name} | device={device} | {elapsed:.1f}s"
    )

    # Optionally compute repo representatives
    if args.make_repo_reps:
        compute_and_save_repo_reps(args.out_jsonl, args.reps_out_jsonl, args.reps_per_repo, args.rep_method)


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Embed README chunks and aggregate to repo vectors (medoid)")
    p.add_argument("--chunks-jsonl", type=Path, default=CHUNKS_JSONL, help="Input chunks JSONL path")
    p.add_argument("--out-jsonl", type=Path, default=OUT_EMB_JSONL, help="Output embeddings JSONL path")
    p.add_argument("--model", dest="model_name", type=str, default=DEFAULT_MODEL, help="SentenceTransformers model name")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")
    p.add_argument("--device", type=str, default=None, help="Force device: cuda|mps|cpu (auto if omitted)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable L2 normalization")
    p.add_argument("--limit", type=int, default=None, help="Limit number of chunks to embed (debug)")
    p.add_argument("--make-repo-reps", action="store_true", default=True, help="Compute k representatives per repo from chunk embeddings and persist JSONL")
    # Default for reps-per-repo will be decided after parsing based on --rep-method
    p.add_argument("--reps-per-repo", type=int, default=None, help="Number of representatives per repo (default: 1 for medoid, 4 for centroid)")
    p.add_argument("--rep-method", type=str, choices=["centroid", "medoid"], default="medoid", help="Representative method")
    p.add_argument("--reps-out-jsonl", type=Path, default=REPO_VEC_JSONL, help="Output JSONL for repo representatives")
    p.add_argument("--reps-only", action="store_true", help="Only compute representatives from an existing embeddings JSONL (skip embedding)")
    p.add_argument("--reps-src-jsonl", type=Path, default=OUT_EMB_JSONL, help="Source embeddings JSONL for representatives (when --reps-only)")
    p.set_defaults(normalize=True)
    a = p.parse_args()

    # Decide reps-per-repo default based on method if not provided
    reps_default = 1 if a.rep_method == "medoid" else 4
    reps_value = a.reps_per_repo if a.reps_per_repo is not None else reps_default

    print(f"Embedding chunks with {a.rep_method} representatives per repo ({reps_value})")

    return Args(
        chunks_jsonl=a.chunks_jsonl,
        out_jsonl=a.out_jsonl,
        model_name=a.model_name,
        batch_size=a.batch_size,
        device=a.device,
        normalize=a.normalize,
        limit=a.limit,
        make_repo_reps=a.make_repo_reps,
        reps_per_repo=reps_value,
        reps_out_jsonl=a.reps_out_jsonl,
        rep_method=a.rep_method,
        reps_only=a.reps_only,
        reps_src_jsonl=a.reps_src_jsonl,
    )


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) 