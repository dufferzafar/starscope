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


def _aggregate_repo_vectors_from_file(path: Path, embeddings_dim: int, normalized: bool) -> Tuple[int, int]:
    """
    Read chunk embeddings from JSONL and compute one canonical vector per repo:
    - mean over chunks, then medoid chunk (closest to mean direction) as canonical
    Writes results to REPO_VEC_JSONL and REPO_VEC_Q8_JSONL.
    Returns: (num_repos, num_chunks_seen)
    """
    repo_to_vecs: Dict[int, List[np.ndarray]] = {}
    repo_to_chunk_ids: Dict[int, List[int]] = {}
    num_chunks = 0

    with path.open("r", encoding="utf-8") as f, tqdm(desc="Aggregating: reading chunk embeddings") as pbar:
        for line in f:
            if not line.strip():
                pbar.update(1)
                continue
            rec = json.loads(line)
            rid = rec.get("repo_id")
            emb = rec.get("embedding")
            cid = rec.get("chunk_id")
            if rid is None or cid is None or not isinstance(emb, list):
                pbar.update(1)
                continue
            vec = np.asarray(emb, dtype=np.float32)
            if vec.shape[0] != embeddings_dim:
                pbar.update(1)
                continue
            repo_to_vecs.setdefault(int(rid), []).append(vec)
            repo_to_chunk_ids.setdefault(int(rid), []).append(int(cid))
            num_chunks += 1
            pbar.update(1)

    # Helper: quantize to int8 with per-vector scale
    def quantize_int8(vec: np.ndarray) -> Tuple[np.ndarray, float]:
        max_abs = float(np.max(np.abs(vec)))
        scale = max_abs / 127.0 if max_abs > 0 else 1.0
        q = np.clip(np.round(vec / scale), -128, 127).astype(np.int8)
        return q, float(scale)

    num_repos = 0
    with REPO_VEC_JSONL.open("w", encoding="utf-8") as f_repo, REPO_VEC_Q8_JSONL.open(
        "w", encoding="utf-8"
    ) as f_q8:
        for rid, vecs in tqdm(repo_to_vecs.items(), desc="Aggregating: computing repo vectors"):
            if not vecs:
                continue
            embs = np.vstack(vecs)
            mean = embs.mean(axis=0)
            # Select medoid (closest to mean direction). If embeddings are normalized, compute dot with mean normalized.
            if normalized:
                denom = np.linalg.norm(mean) + 1e-12
                mdir = mean / denom
                sims = embs @ mdir
                medoid_idx = int(np.argmax(sims))
            else:
                # Fall back to Euclidean distance to the mean
                diffs = embs - mean
                d2 = np.sum(diffs * diffs, axis=1)
                medoid_idx = int(np.argmin(d2))
            medoid_vec = embs[medoid_idx]

            rec_repo = {"repo_id": int(rid), "embedding": medoid_vec.astype(np.float32).tolist()}
            f_repo.write(json.dumps(rec_repo, ensure_ascii=False) + "\n")

            q, scale = quantize_int8(medoid_vec.astype(np.float32))
            rec_q8 = {"repo_id": int(rid), "embedding": q.astype(int).tolist(), "scale": scale}
            f_q8.write(json.dumps(rec_q8, ensure_ascii=False) + "\n")
            num_repos += 1

    return num_repos, num_chunks


def run(args: Args) -> None:
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)

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
        for batch in tqdm(batch_iterator(iter_chunks(args.chunks_jsonl), args.batch_size), desc="Embedding chunks"):
            if args.limit is not None and total_items >= args.limit:
                break

            # If limit cuts through the batch, trim
            if args.limit is not None and total_items + len(batch) > args.limit:
                batch = batch[: max(0, args.limit - total_items)]

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

    elapsed = time.time() - start_time
    print(
        f"Wrote {total_items} chunk embeddings to {args.out_jsonl} | dim={embedding_dim} | model={args.model_name} | device={device} | {elapsed:.1f}s"
    )

    # Aggregate to repo-level vectors (medoid after mean) and write JSONL outputs
    n_repos, n_chunks = _aggregate_repo_vectors_from_file(args.out_jsonl, embedding_dim, args.normalize)
    print(
        f"Wrote {n_repos} repo vectors to {REPO_VEC_JSONL} and int8 to {REPO_VEC_Q8_JSONL} (from {n_chunks} chunks)"
    )


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Embed README chunks and aggregate to repo vectors (medoid)")
    p.add_argument("--chunks-jsonl", type=Path, default=CHUNKS_JSONL, help="Input chunks JSONL path")
    p.add_argument("--out-jsonl", type=Path, default=OUT_EMB_JSONL, help="Output embeddings JSONL path")
    p.add_argument("--model", dest="model_name", type=str, default=DEFAULT_MODEL, help="SentenceTransformers model name")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding")
    p.add_argument("--device", type=str, default=None, help="Force device: cuda|mps|cpu (auto if omitted)")
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="Disable L2 normalization")
    p.add_argument("--limit", type=int, default=None, help="Limit number of chunks to embed (debug)")
    p.set_defaults(normalize=True)
    a = p.parse_args()
    return Args(
        chunks_jsonl=a.chunks_jsonl,
        out_jsonl=a.out_jsonl,
        model_name=a.model_name,
        batch_size=a.batch_size,
        device=a.device,
        normalize=a.normalize,
        limit=a.limit,
    )


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130) 