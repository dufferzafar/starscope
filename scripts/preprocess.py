#!/usr/bin/env python3

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from tqdm import tqdm


CACHE_DIR = Path(__file__).resolve().parent / ".cache"
READMES_JSONL = CACHE_DIR / "readmes.jsonl"
STARRED_JSON = CACHE_DIR / "starred.json"
OUT_READMES_CLEAN = CACHE_DIR / "readmes_clean.jsonl"
OUT_CHUNKS = CACHE_DIR / "chunks.jsonl"


@dataclass
class Args:
    readmes_jsonl: Path
    starred_json: Path
    out_clean: Path
    out_chunks: Path
    chunk_size: int
    overlap: int
    chunker: str
    encoding: str
    add_meta_chunk: bool


# === SSL disable helpers (temporary, to avoid local cert issues) =============

def _patch_requests_ssl_disable():
    try:
        import requests  # type: ignore
        try:
            import urllib3  # type: ignore
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        original_request = requests.sessions.Session.request  # type: ignore[attr-defined]
        original_get = requests.get

        def insecure_request(self, method, url, **kwargs):  # type: ignore[override]
            kwargs.setdefault("verify", False)
            return original_request(self, method, url, **kwargs)

        def insecure_get(url, *args, **kwargs):  # type: ignore[override]
            kwargs.setdefault("verify", False)
            return original_get(url, *args, **kwargs)

        requests.sessions.Session.request = insecure_request  # type: ignore[attr-defined]
        requests.get = insecure_get  # type: ignore[attr-defined]
        return (requests, original_request, original_get)
    except Exception:
        return None


def _restore_requests_ssl(patch_state):
    if not patch_state:
        return
    try:
        requests, original_request, original_get = patch_state
        requests.sessions.Session.request = original_request  # type: ignore[attr-defined]
        requests.get = original_get  # type: ignore[attr-defined]
    except Exception:
        pass


# === Cleaning helpers ========================================================

_HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.DOTALL)
_IMG_MD = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_IMG_MD_WRAPPED = re.compile(r"\[\s*!\[[^\]]*\]\([^)]*\)\s*\]\([^)]*\)")
_IMG_HTML = re.compile(r"<img\b[^>]*>", flags=re.IGNORECASE)
_TABLE_BADGE_HINT = re.compile(r"\|.+(shields\.io|badgen\.net|badge\.fury\.io).+\|", re.IGNORECASE)
_BADGE_HOST_HINT = re.compile(r"(shields\.io|badgen\.net|badge\.fury\.io)", re.IGNORECASE)
_WHITESPACE_LINES = re.compile(r"\n{3,}")


def remove_badges_and_images(md_text: str) -> str:
    md_text = _IMG_MD_WRAPPED.sub("", md_text)
    md_text = _IMG_MD.sub("", md_text)
    md_text = _HTML_COMMENT.sub("", md_text)
    md_text = _IMG_HTML.sub("", md_text)

    cleaned_lines: List[str] = []
    for line in md_text.splitlines():
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue
        if _TABLE_BADGE_HINT.search(stripped):
            continue
        if _BADGE_HOST_HINT.search(stripped):
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def markdown_to_text(md_text: str) -> str:
    md = MarkdownIt("commonmark").disable(["html_inline", "html_block"])
    html = md.render(md_text)
    try:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n")
    except Exception:
        text = md_text
    return text


def normalize_whitespace(text: str) -> str:
    text = _WHITESPACE_LINES.sub("\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln is not None).strip()


def clean_readme(raw_markdown: str) -> str:
    text = raw_markdown.replace("\r\n", "\n").replace("\r", "\n")
    text = remove_badges_and_images(text)
    text = markdown_to_text(text)
    text = normalize_whitespace(text)
    return text


# === Chunkers ================================================================

class BaseChunker:
    def __init__(self, chunk_size: int, overlap: int) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if self.overlap < 0 or self.overlap >= self.chunk_size:
            raise ValueError("overlap must be >= 0 and < chunk_size")

    def iter_chunks(self, text: str) -> Iterator[str]:  # override
        raise NotImplementedError


class HFTokenizerChunker(BaseChunker):
    def __init__(self, model_name: str, chunk_size: int, overlap: int) -> None:
        super().__init__(chunk_size, overlap)
        from transformers import AutoTokenizer  # lazy import

        _ps = _patch_requests_ssl_disable()
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Avoid long-sequence warnings during pure tokenization by raising the soft limit
            self.tokenizer.model_max_length = 10**9
        finally:
            _restore_requests_ssl(_ps)

    def iter_chunks(self, text: str) -> Iterator[str]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) == 0:
            return iter(())
        step = self.chunk_size - self.overlap
        for start in range(0, len(ids), step):
            end = min(start + self.chunk_size, len(ids))
            piece_ids = ids[start:end]
            if not piece_ids:
                break
            yield self.tokenizer.decode(piece_ids, skip_special_tokens=True)
            if end == len(ids):
                break


class TikTokenChunker(BaseChunker):
    def __init__(self, encoding_name: str, chunk_size: int, overlap: int) -> None:
        super().__init__(chunk_size, overlap)
        self.encoding_name = encoding_name

    def _get_encoding(self):
        import tiktoken

        _ps = _patch_requests_ssl_disable()
        try:
            try:
                return tiktoken.get_encoding(self.encoding_name)
            except Exception:
                return tiktoken.get_encoding("cl100k_base")
        finally:
            _restore_requests_ssl(_ps)

    def iter_chunks(self, text: str) -> Iterator[str]:
        enc = self._get_encoding()
        token_ids = enc.encode(text)
        if len(token_ids) == 0:
            return iter(())
        step = self.chunk_size - self.overlap
        for start in range(0, len(token_ids), step):
            end = min(start + self.chunk_size, len(token_ids))
            piece = token_ids[start:end]
            if not piece:
                break
            yield enc.decode(piece)
            if end == len(token_ids):
                break


class WordChunker(BaseChunker):
    def iter_chunks(self, text: str) -> Iterator[str]:
        words = text.split()
        if not words:
            return iter(())
        step = self.chunk_size - self.overlap
        for start in range(0, len(words), step):
            end = min(start + self.chunk_size, len(words))
            piece = words[start:end]
            if not piece:
                break
            yield " ".join(piece)
            if end == len(words):
                break


class CharChunker(BaseChunker):
    def iter_chunks(self, text: str) -> Iterator[str]:
        if not text:
            return iter(())
        step = self.chunk_size - self.overlap
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            piece = text[start:end]
            if not piece:
                break
            yield piece
            if end == len(text):
                break


def build_chunker(spec: str, default_model: str, chunk_size: int, overlap: int, encoding_name: str) -> BaseChunker:
    # spec examples: "hf:BAAI/bge-small-en-v1.5", "tiktoken", "words", "chars"
    if spec.startswith("hf:"):
        model = spec.split(":", 1)[1] or default_model
        return HFTokenizerChunker(model, chunk_size, overlap)
    if spec == "tiktoken":
        return TikTokenChunker(encoding_name, chunk_size, overlap)
    if spec == "words":
        return WordChunker(chunk_size, overlap)
    if spec == "chars":
        return CharChunker(chunk_size, overlap)
    # default to HF
    return HFTokenizerChunker(default_model, chunk_size, overlap)


# === IO helpers ==============================================================

def load_starred_map(path: Path) -> Dict[str, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["full_name"]: int(item["id"]) for item in data if "full_name" in item and "id" in item}


def load_starred_index(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item.get("full_name"): item for item in data if item.get("full_name")}


def iter_readmes(path: Path) -> Iterable[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            full_name = rec.get("full_name")
            text = rec.get("text", "")
            if full_name and isinstance(text, str):
                yield full_name, text


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# === Main ===================================================================

def run(args: Args) -> None:
    args.out_clean.parent.mkdir(parents=True, exist_ok=True)
    args.out_chunks.parent.mkdir(parents=True, exist_ok=True)

    name_to_id = load_starred_map(args.starred_json)
    starred_index = load_starred_index(args.starred_json)

    chunker = build_chunker(
        spec=args.chunker,
        default_model="BAAI/bge-small-en-v1.5",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        encoding_name=args.encoding,
    )

    cleaned_written = 0
    chunks_written = 0

    with args.out_clean.open("w", encoding="utf-8") as f_clean, args.out_chunks.open(
        "w", encoding="utf-8"
    ) as f_chunks:
        for full_name, raw_md in tqdm(iter_readmes(args.readmes_jsonl), desc="Preprocessing READMEs"):
            repo_id = name_to_id.get(full_name)
            if repo_id is None:
                continue

            # Optional metadata chunk
            if args.add_meta_chunk:
                meta = starred_index.get(full_name) or {}
                language = meta.get("language") or ""
                description = meta.get("description") or ""
                if description or language:
                    meta_text = normalize_whitespace(
                        f"Title: {full_name}\nLanguage: {language}\nDescription: {description}"
                    )
                    f_chunks.write(
                        json.dumps(
                            {
                                "repo_id": repo_id,
                                "chunk_id": -1,
                                "text": meta_text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    chunks_written += 1

            cleaned = clean_readme(raw_md)
            clean_rec = {
                "repo_id": repo_id,
                "cleaned_text": cleaned,
                "hash": sha256_hex(cleaned),
            }
            f_clean.write(json.dumps(clean_rec, ensure_ascii=False) + "\n")
            cleaned_written += 1

            idx = 0
            for piece in chunker.iter_chunks(cleaned):
                f_chunks.write(
                    json.dumps(
                        {
                            "repo_id": repo_id,
                            "chunk_id": idx,
                            "text": piece,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                chunks_written += 1
                idx += 1

    print(
        f"Wrote {cleaned_written} cleaned READMEs to {args.out_clean} and {chunks_written} chunks to {args.out_chunks}"
    )


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Clean README markdown and chunk (HF/tiktoken/words/chars)")
    p.add_argument("--readmes-jsonl", type=Path, default=READMES_JSONL, help="Path to readmes.jsonl input")
    p.add_argument("--starred-json", type=Path, default=STARRED_JSON, help="Path to starred.json for repo_id map")
    p.add_argument("--out-clean", type=Path, default=OUT_READMES_CLEAN, help="Output JSONL for cleaned readmes")
    p.add_argument("--out-chunks", type=Path, default=OUT_CHUNKS, help="Output JSONL for chunks")
    p.add_argument("--chunk-size", type=int, default=512, help="Target tokens/units per chunk (default: 512)")
    p.add_argument("--overlap", type=int, default=50, help="Overlap between chunks (default: 50)")
    p.add_argument(
        "--chunker",
        type=str,
        default="hf:BAAI/bge-small-en-v1.5",
        help="Chunker: hf:<model>, tiktoken, words, chars (default: hf:BAAI/bge-small-en-v1.5)",
    )
    p.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="tiktoken encoding name (used when --chunker=tiktoken)",
    )
    p.add_argument(
        "--no-meta-chunk",
        dest="add_meta_chunk",
        action="store_false",
        help="Do not add a small metadata chunk per repo (title/language/description)",
    )
    p.set_defaults(add_meta_chunk=True)
    a = p.parse_args()
    return Args(
        readmes_jsonl=a.readmes_jsonl,
        starred_json=a.starred_json,
        out_clean=a.out_clean,
        out_chunks=a.out_chunks,
        chunk_size=a.chunk_size,
        overlap=a.overlap,
        chunker=a.chunker,
        encoding=a.encoding,
        add_meta_chunk=a.add_meta_chunk,
    )


if __name__ == "__main__":
    run(parse_args()) 