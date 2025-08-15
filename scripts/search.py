#!/usr/bin/env python3

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from sentence_transformers import SentenceTransformer, util


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
EMB_JSONL = CACHE_DIR / "chunk_embeddings.jsonl"
META_JSON = CACHE_DIR / "embeddings_meta.json"
STARRED_JSON = CACHE_DIR / "starred.json"

# Fast-load caches (chunk-level)
EMB_NPY = CACHE_DIR / "corpus_embeddings.npy"
IDMAP_NPY = CACHE_DIR / "id_map.npy"
FAISS_INDEX = CACHE_DIR / "corpus_flatip.faiss"


@dataclass
class Args:
    query: str
    top_k: int
    emb_jsonl: Path
    meta_json: Path
    starred_json: Path
    device: Optional[str]
    model_name: Optional[str]
    normalize: bool
    snippet_chars: int
    use_faiss: bool
    build_faiss: bool
    faiss_index: Path
    unique: bool
    oversample: int


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


# --- Common helpers ----------------------------------------------------------


def detect_device(preferred: Optional[str] = None) -> str:
    try:
        import torch  # type: ignore

        if preferred:
            pref = preferred.lower()
            if pref == "cuda" and torch.cuda.is_available():
                return "cuda"
            if (
                pref == "mps"
                and getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
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


def ensure_model(model_name: str, device: str) -> SentenceTransformer:
    patch_state = _patch_requests_ssl_disable()
    try:
        model = SentenceTransformer(model_name, device=device)
    finally:
        _restore_requests_ssl(patch_state)
    return model


def load_meta(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_repo_index(path: Path) -> Dict[int, Dict[str, object]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    idx: Dict[int, Dict[str, object]] = {}
    for rec in data:
        try:
            idx[int(rec["id"])] = rec
        except Exception:
            continue
    return idx


def load_corpus_from_jsonl(path: Path) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    vectors: List[List[float]] = []
    mapping: List[Tuple[int, int]] = []
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
            vectors.append(emb)
            mapping.append((int(rid), int(cid)))
    if not vectors:
        raise RuntimeError(f"No embeddings found in {path}")
    mat = np.asarray(vectors, dtype=np.float32)
    return mat, mapping


def ensure_npy_cache(
    emb_jsonl: Path, emb_npy: Path, idmap_npy: Path
) -> Tuple[np.ndarray, np.ndarray]:
    if emb_npy.exists() and idmap_npy.exists():
        embs = np.load(emb_npy, mmap_mode="r")
        idmap = np.load(idmap_npy)
        return embs, idmap
    embs, mapping = load_corpus_from_jsonl(emb_jsonl)
    np.save(emb_npy, embs)
    idmap_arr = np.asarray(mapping, dtype=np.int64)
    np.save(idmap_npy, idmap_arr)
    return np.load(emb_npy, mmap_mode="r"), idmap_arr


def try_import_faiss():
    try:
        import faiss  # type: ignore

        return faiss
    except Exception:
        return None


def build_faiss_index(
    faiss, embs: np.ndarray, index_path: Path, normalized: bool
) -> None:
    dim = embs.shape[1]
    if not normalized:
        n = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-12
        embs = embs / n
    index = faiss.IndexFlatIP(dim)
    index.add(embs.astype(np.float32, copy=False))
    faiss.write_index(index, str(index_path))


def ensure_faiss_index(
    faiss, embs: np.ndarray, index_path: Path, normalized: bool, rebuild: bool
):
    if rebuild or not index_path.exists():
        build_faiss_index(faiss, embs, index_path, normalized)
    return faiss.read_index(str(index_path))


def format_and_print_results(
    hits: List[Tuple[float, int]],
    id_map_np: np.ndarray,
    starred_json: Path,
    snippet_chars: int,
) -> None:
    repo_idx = load_repo_index(starred_json) if starred_json.exists() else {}

    for rank, (score, j) in enumerate(hits, start=1):
        rid, _cid = tuple(map(int, id_map_np[j]))
        repo = repo_idx.get(int(rid), {})
        url = repo.get("html_url", "")
        language = repo.get("language")
        stars = repo.get("stargazers_count")
        description = repo.get("description") or ""
        if len(description) > snippet_chars:
            description = description[: snippet_chars - 1] + "…"

        row = [
            f"{url} ",
            str(language),
            f"⭐ {stars}",
            description,
        ]

        print(" · ".join(row))


def dedupe_hits_by_repo(hits: List[Tuple[float, int]], id_map_np: np.ndarray, top_k: int) -> List[Tuple[float, int]]:
    # Keep best score per repo_id
    best: Dict[int, Tuple[float, int]] = {}
    for score, j in hits:
        rid, _ = tuple(map(int, id_map_np[j]))
        prev = best.get(rid)
        if prev is None or score > prev[0]:
            best[rid] = (score, j)
    deduped = sorted(best.values(), key=lambda x: x[0], reverse=True)
    return deduped[:top_k]


# --- Entry point -------------------------------------------------------------


def run(args: Args) -> None:
    import torch  # local import

    meta = load_meta(args.meta_json)
    model_name = args.model_name or str(meta.get("model", "BAAI/bge-small-en-v1.5"))

    device = detect_device(args.device)

    # If FAISS index exists and user didn't opt out, prefer FAISS
    if not args.use_faiss and args.faiss_index.exists():
        args.use_faiss = True

    model = ensure_model(model_name, device)

    # Load or build fast-load corpus cache (chunk-level)
    corpus_np, id_map_np = ensure_npy_cache(args.emb_jsonl, EMB_NPY, IDMAP_NPY)

    # Determine how many to retrieve pre-dedupe
    total_n = corpus_np.shape[0]
    pre_k = args.top_k * (args.oversample if args.unique else 1)
    pre_k = min(pre_k, total_n)

    if args.use_faiss:
        faiss = try_import_faiss()
        if faiss is None:
            print("FAISS not installed. Falling back to Torch path.", file=sys.stderr)
        else:
            index = ensure_faiss_index(
                faiss=faiss,
                embs=corpus_np,
                index_path=args.faiss_index,
                normalized=args.normalize,
                rebuild=args.build_faiss,
            )
            q = model.encode(
                [args.query],
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=args.normalize,
            )
            if not args.normalize:
                q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
            sims, idxs = index.search(q.astype(np.float32), pre_k)
            hits = [
                (float(sims[0][i]), int(idxs[0][i]))
                for i in range(len(idxs[0]))
                if idxs[0][i] >= 0
            ]
            if args.unique:
                hits = dedupe_hits_by_repo(hits, id_map_np, args.top_k)
            format_and_print_results(
                hits=hits,
                id_map_np=id_map_np,
                starred_json=args.starred_json,
                snippet_chars=args.snippet_chars,
            )
            return

    # Torch fast path: move to device and do matmul + topk
    target_device = torch.device(device)
    corpus = torch.from_numpy(corpus_np).to(target_device)
    if args.normalize:
        corpus = util.normalize_embeddings(corpus)
    q = model.encode(
        [args.query],
        convert_to_tensor=True,
        show_progress_bar=False,
        normalize_embeddings=args.normalize,
    ).to(target_device)

    scores = torch.matmul(q, corpus.T)  # [1, N]
    vals, idxs = torch.topk(scores, k=pre_k, dim=1)
    hits = [
        (float(vals[0, i].item()), int(idxs[0, i].item()))
        for i in range(min(pre_k, scores.shape[1]))
    ]
    if args.unique:
        hits = dedupe_hits_by_repo(hits, id_map_np, args.top_k)

    format_and_print_results(
        hits=hits,
        id_map_np=id_map_np,
        starred_json=args.starred_json,
        snippet_chars=args.snippet_chars,
    )


def parse_args() -> Args:
    p = argparse.ArgumentParser(
        description="Semantic search over chunk embeddings from CLI"
    )
    p.add_argument("query", type=str, help="Search query text")
    p.add_argument("--top-k", type=int, default=20, help="Number of results to return")
    p.add_argument(
        "--embeddings",
        dest="emb_jsonl",
        type=Path,
        default=EMB_JSONL,
        help="Path to chunk_embeddings.jsonl",
    )
    p.add_argument(
        "--meta",
        dest="meta_json",
        type=Path,
        default=META_JSON,
        help="Path to embeddings_meta.json",
    )
    p.add_argument(
        "--starred",
        dest="starred_json",
        type=Path,
        default=STARRED_JSON,
        help="Path to starred.json for repo metadata",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force device: cuda|mps|cpu (auto if omitted)",
    )
    p.add_argument(
        "--model",
        dest="model_name",
        type=str,
        default=None,
        help="Override model name (defaults to meta)",
    )
    p.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        help="Disable normalization (use cosine)",
    )
    p.add_argument(
        "--snippet-chars",
        type=int,
        default=50,
        help="Max characters to show from repo description",
    )
    p.add_argument(
        "--use-faiss",
        action="store_true",
        help="Use FAISS ANN index if available (fast)",
    )
    p.add_argument(
        "--build-faiss",
        action="store_true",
        help="Force rebuild FAISS index before search",
    )
    p.add_argument(
        "--faiss-index", type=Path, default=FAISS_INDEX, help="Path to FAISS index file"
    )
    p.add_argument(
        "--no-unique",
        dest="unique",
        action="store_false",
        help="Allow multiple hits from the same repo (default dedupes)",
    )
    p.add_argument(
        "--oversample",
        type=int,
        default=5,
        help="Factor to oversample before deduping (default: 5)",
    )
    p.set_defaults(normalize=True, unique=True)
    a = p.parse_args()
    return Args(
        query=a.query,
        top_k=a.top_k,
        emb_jsonl=a.emb_jsonl,
        meta_json=a.meta_json,
        starred_json=a.starred_json,
        device=a.device,
        model_name=a.model_name,
        normalize=a.normalize,
        snippet_chars=a.snippet_chars,
        use_faiss=a.use_faiss,
        build_faiss=a.build_faiss,
        faiss_index=a.faiss_index,
        unique=a.unique,
        oversample=a.oversample,
    )


if __name__ == "__main__":
    try:
        run(parse_args())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        raise SystemExit(130)
