#!/usr/bin/env python3

import asyncio
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from tqdm import tqdm

# Hacks all the way up to the root of the project
if str(Path(__file__).resolve().parent.parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.settings as settings

API_ROOT = "https://api.github.com"
CACHE_DIR = Path(__file__).resolve().parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

STARRED_JSON = CACHE_DIR / "starred.json"
READMES_JSONL = CACHE_DIR / "readmes.jsonl"
REPOS_JSONL = CACHE_DIR / "repos.jsonl"


@dataclass
class GitHubConfig:
    token: str
    user_agent: str = "starscope-collector"
    per_page: int = 100
    timeout_seconds: float = 30.0


def auth_headers(token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "User-Agent": "starscope-collector",
        "X-GitHub-Api-Version": "2022-11-28",
    }


async def fetch_starred(client: httpx.AsyncClient, cfg: GitHubConfig) -> List[Dict[str, Any]]:
    starred: List[Dict[str, Any]] = []
    page = 1
    with tqdm(desc="Fetching starred repos", unit="page") as pbar:
        while True:
            url = f"{API_ROOT}/user/starred"
            params = {"per_page": cfg.per_page, "page": page}
            # Ask for star+json so we get starred_at
            headers = {"Accept": "application/vnd.github.star+json"}
            resp = await client.get(url, params=params, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                break
            for item in data:
                # Handle both shapes: star+json (item has repo+starred_at) or repo directly
                if isinstance(item, dict) and "repo" in item:
                    repo = item.get("repo", {})
                    starred_at = item.get("starred_at")
                else:
                    repo = item
                    starred_at = None
                starred.append(
                    {
                        "id": repo.get("id"),
                        "full_name": repo.get("full_name"),
                        "html_url": repo.get("html_url"),
                        "description": repo.get("description"),
                        "language": repo.get("language"),
                        "stargazers_count": repo.get("stargazers_count"),
                        "pushed_at": repo.get("pushed_at"),
                        "updated_at": repo.get("updated_at"),
                        "starred_at": starred_at,
                    }
                )
            page += 1
            pbar.update(1)
    return starred


async def fetch_readme(client: httpx.AsyncClient, full_name: str) -> Optional[Tuple[str, str]]:
    url = f"{API_ROOT}/repos/{full_name}/readme"
    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        content_b64 = data.get("content")
        encoding = data.get("encoding")
        if not content_b64 or encoding != "base64":
            return None
        try:
            text = base64.b64decode(content_b64).decode("utf-8", errors="replace")
        except Exception:
            return None
        return (full_name, text)
    except httpx.HTTPStatusError as e:
        if e.response.status_code in {403, 429}:
            # Rate-limited; caller may decide to backoff if needed
            return None
        return None
    except Exception:
        return None


async def fetch_readmes_parallel(
    client: httpx.AsyncClient,
    repos: List[Dict[str, Any]],
    max_count: int,
    concurrency: int = 20,
) -> List[Tuple[str, str]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Tuple[str, str]] = []

    async def worker(full_name: str):
        async with sem:
            res = await fetch_readme(client, full_name)
            if res:
                results.append(res)

    targets = [r["full_name"] for r in repos[:max_count] if r.get("full_name")]
    tasks = [asyncio.create_task(worker(name)) for name in targets]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching READMEs"):
        await f

    return results


def save_starred(path: Path, data: List[Dict[str, Any]]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def save_readmes_jsonl(path: Path, items: List[Tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for full_name, text in items:
            rec = {"full_name": full_name, "text": text}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_starred(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Cached starred file not found at {path}. Run in 'stars' mode first to create it."
        )
    return json.loads(path.read_text(encoding="utf-8"))


async def fetch_repo_metadata(client: httpx.AsyncClient, full_name: str) -> Optional[Dict[str, Any]]:
    url = f"{API_ROOT}/repos/{full_name}"
    try:
        resp = await client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return data
    except httpx.HTTPStatusError as e:
        if e.response.status_code in {403, 429}:
            return None
        return None
    except Exception:
        return None


async def fetch_repos_parallel(
    client: httpx.AsyncClient,
    repos: List[Dict[str, Any]],
    concurrency: int = 20,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    async def worker(full_name: str):
        async with sem:
            res = await fetch_repo_metadata(client, full_name)
            if res:
                results.append(res)

    targets = [r["full_name"] for r in repos if r.get("full_name")]
    tasks = [asyncio.create_task(worker(name)) for name in targets]

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching repo metadata"):
        await f

    return results


def save_repos_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


async def main_async(mode: str) -> None:
    cfg = GitHubConfig(token=settings.GITHUB_TOKEN)

    timeout = httpx.Timeout(cfg.timeout_seconds, connect=cfg.timeout_seconds)
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    async with httpx.AsyncClient(headers=auth_headers(cfg.token), timeout=timeout, limits=limits) as client:
        if mode == "stars":
            starred = await fetch_starred(client, cfg)
            save_starred(STARRED_JSON, starred)
            print(f"Saved starred repos to {STARRED_JSON} ({len(starred)} repos)")
            return

        if mode == "readmes":
            starred = load_starred(STARRED_JSON)
            readmes = await fetch_readmes_parallel(
                client,
                starred,
                max_count=settings.MAX_READMES,
                concurrency=settings.MAX_CONCURRENCY,
            )
            save_readmes_jsonl(READMES_JSONL, readmes)
            print(f"Saved READMEs to {READMES_JSONL} ({len(readmes)} items)")
            return

        if mode == "repos":
            starred = load_starred(STARRED_JSON)
            repos = await fetch_repos_parallel(
                client,
                starred,
                concurrency=settings.MAX_CONCURRENCY,
            )
            save_repos_jsonl(REPOS_JSONL, repos)
            print(f"Saved repo metadata to {REPOS_JSONL} ({len(repos)} items)")
            return

        raise ValueError(f"Unknown mode: {mode}")


def parse_args(argv: List[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch starred repos, READMEs, or repo metadata")
    parser.add_argument(
        "mode",
        choices=["stars", "readmes", "repos"],
        help="Which dataset to fetch: 'stars' (starred repos list), 'readmes' (repo READMEs), or 'repos' (repo metadata)",
    )
    args = parser.parse_args(argv)

    try:
        asyncio.run(main_async(mode=args.mode))
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(parse_args(sys.argv[1:])) 