#!/usr/bin/env python3

import asyncio
import base64
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Awaitable

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


def _parse_ts_iso8601_z(ts: Optional[str]) -> Optional[int]:
    if not ts or not isinstance(ts, str):
        return None
    # Expected like 2025-08-14T23:30:06Z
    try:
        from datetime import datetime, timezone
        dt = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except Exception:
        return None


async def fetch_starred(client: httpx.AsyncClient, cfg: GitHubConfig) -> List[Dict[str, Any]]:
    # Determine cutoff from existing starred.json (most recent starred_at we've already saved)
    cutoff_ts: Optional[int] = None
    if STARRED_JSON.exists():
        try:
            existing = json.loads(STARRED_JSON.read_text(encoding="utf-8"))
            for rec in existing:
                ts = _parse_ts_iso8601_z(rec.get("starred_at"))
                if ts is not None:
                    cutoff_ts = max(cutoff_ts or ts, ts)
        except Exception:
            cutoff_ts = None

    starred: List[Dict[str, Any]] = []
    page = 1
    reached_cutoff = False
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
                ts = _parse_ts_iso8601_z(starred_at)
                # Stop if we've reached already-known stars
                if cutoff_ts is not None and ts is not None and ts <= cutoff_ts:
                    reached_cutoff = True
                    break
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
            if reached_cutoff:
                break
            page += 1
            pbar.update(1)
    return starred


async def fetch_readme(client: httpx.AsyncClient, full_name: str, *, max_retries: int = 8) -> Optional[Tuple[str, str]]:
    url = f"{API_ROOT}/repos/{full_name}/readme"
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = await client.get(url)
            if resp.status_code in {403, 429}:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else delay
                await asyncio.sleep(sleep_s)
                delay = min(delay * 2.0, 60.0)
                continue
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
        except httpx.TimeoutException:
            print(f"[readme] Timeout for {full_name}; retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 60.0)
            continue
        except httpx.HTTPStatusError as e:
            if e.response.status_code in {500, 502, 503, 504}:
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)
                continue
            if e.response.status_code in {403, 429}:
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)
                continue
            return None
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 60.0)
            continue
    return None


async def fetch_readmes_parallel(
    client: httpx.AsyncClient,
    repos: List[Dict[str, Any]],
    concurrency: int = 20,
    existing_full_names: Optional[set] = None,
    on_result: Optional[Callable[[str, str], Awaitable[None]]] = None,
) -> List[Tuple[str, str]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Tuple[str, str]] = []

    # Prepare targets (dedupe, limit, skip existing)
    all_targets = [r.get("full_name") for r in repos if r.get("full_name")]
    unique_targets = sorted(set(all_targets))
    if existing_full_names:
        targets = [t for t in unique_targets if t not in existing_full_names]
    else:
        targets = unique_targets

    print(
        f"READMEs to fetch: {len(targets)} (skipping {len(unique_targets) - len(targets)} already present, total {len(unique_targets)})"
    )

    async def worker(full_name: str):
        async with sem:
            res = await fetch_readme(client, full_name)
            if res:
                fn, text = res
                if on_result is not None:
                    await on_result(fn, text)
                else:
                    results.append((fn, text))

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


def load_readmes_jsonl(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    mp: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                fn = rec.get("full_name")
                tx = rec.get("text")
                if isinstance(fn, str) and isinstance(tx, str):
                    mp[fn] = tx
            except Exception:
                continue
    return mp


async def fetch_repo_metadata(client: httpx.AsyncClient, full_name: str, *, max_retries: int = 8) -> Optional[Dict[str, Any]]:
    import time

    url = f"{API_ROOT}/repos/{full_name}"
    delay = 1.0
    for attempt in range(max_retries):
        try:
            resp = await client.get(url)
            # Handle explicit rate limit cases with headers
            if resp.status_code in {403, 429}:
                # Check remaining and reset
                remaining = resp.headers.get("X-RateLimit-Remaining")
                reset = resp.headers.get("X-RateLimit-Reset")
                retry_after = resp.headers.get("Retry-After")
                sleep_s: float = 0.0
                if retry_after:
                    try:
                        sleep_s = float(retry_after)
                    except Exception:
                        sleep_s = delay
                elif remaining == "0" and reset:
                    try:
                        reset_epoch = int(reset)
                        now = int(time.time())
                        sleep_s = max(0, reset_epoch - now) + 1
                    except Exception:
                        sleep_s = delay
                else:
                    sleep_s = delay
                await asyncio.sleep(sleep_s)
                delay = min(delay * 2.0, 60.0)
                continue
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            print(f"[repos] Timeout for {full_name}; retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 60.0)
            continue
        except httpx.HTTPStatusError as e:
            if e.response.status_code in {500, 502, 503, 504}:
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)
                continue
            if e.response.status_code in {403, 429}:
                await asyncio.sleep(delay)
                delay = min(delay * 2.0, 60.0)
                continue
            return None
        except Exception:
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 60.0)
            continue
    return None


async def fetch_repos_parallel(
    client: httpx.AsyncClient,
    repos: List[Dict[str, Any]],
    concurrency: int = 12,
    existing_full_names: Optional[set] = None,
) -> List[Dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    results: List[Dict[str, Any]] = []

    # Deduplicate targets and resume from existing JSON/JSONL
    all_targets = [r.get("full_name") for r in repos if r.get("full_name")]
    unique_targets = sorted(set(all_targets))
    if existing_full_names:
        targets = [t for t in unique_targets if t not in existing_full_names]
    else:
        targets = unique_targets

    print(
        f"Repos to fetch: {len(targets)} (skipping {len(unique_targets) - len(targets)} already present, total {len(unique_targets)})"
    )

    async def worker(full_name: str):
        async with sem:
            res = await fetch_repo_metadata(client, full_name)
            if res:
                results.append(res)

    tasks = [asyncio.create_task(worker(name)) for name in targets]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Fetching repo metadata"):
        await f

    return results


def save_repos_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in items:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def load_repos_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
    return items


async def main_async(mode: str) -> None:
    cfg = GitHubConfig(token=settings.GITHUB_TOKEN)

    timeout = httpx.Timeout(cfg.timeout_seconds, connect=cfg.timeout_seconds)
    limits = httpx.Limits(max_keepalive_connections=50, max_connections=100)
    async with httpx.AsyncClient(headers=auth_headers(cfg.token), timeout=timeout, limits=limits) as client:
        if mode == "stars":
            # Load existing to support incremental fetch
            existing: List[Dict[str, Any]] = []
            if STARRED_JSON.exists():
                try:
                    existing = json.loads(STARRED_JSON.read_text(encoding="utf-8"))
                except Exception:
                    existing = []

            new_items = await fetch_starred(client, cfg)

            if not existing:
                merged = new_items
            else:
                # Merge new (most recent first) + existing, dedupe by repo id
                seen: set = set()
                merged: List[Dict[str, Any]] = []
                for rec in new_items + existing:
                    rid = rec.get("id")
                    if rid is None or rid in seen:
                        continue
                    seen.add(rid)
                    merged.append(rec)

            save_starred(STARRED_JSON, merged)
            print(f"Saved starred repos to {STARRED_JSON} (added {len(new_items)} new, total {len(merged)})")
            return

        if mode == "readmes":
            import os

            starred = load_starred(STARRED_JSON)
            # Load existing READMEs (resume)
            existing_map = load_readmes_jsonl(READMES_JSONL)
            existing_names = set(existing_map.keys())

            # Append new items as they arrive
            append_lock = asyncio.Lock()
            appended = 0

            async def on_result(full_name: str, text: str) -> None:
                nonlocal appended
                # Double-check skip within run
                if full_name in existing_names:
                    return
                line = json.dumps({"full_name": full_name, "text": text}, ensure_ascii=False) + "\n"
                async with append_lock:
                    with READMES_JSONL.open("a", encoding="utf-8") as f:
                        f.write(line)
                        f.flush()
                        try:
                            os.fsync(f.fileno())
                        except Exception:
                            pass
                    existing_names.add(full_name)
                    appended += 1

            await fetch_readmes_parallel(
                client,
                starred,
                concurrency=settings.MAX_CONCURRENCY,
                existing_full_names=existing_names,
                on_result=on_result,
            )

            total = len(existing_names)
            print(f"Appended {appended} READMEs to {READMES_JSONL} (total now {total})")
            return

        if mode == "repos":
            starred = load_starred(STARRED_JSON)
            existing_jsonl = load_repos_jsonl(REPOS_JSONL)
            existing_map = {rec.get("full_name"): rec for rec in existing_jsonl if rec.get("full_name")}
            existing_names = set(existing_map.keys())

            fetched = await fetch_repos_parallel(
                client,
                starred,
                concurrency=min(settings.MAX_CONCURRENCY, 12),
                existing_full_names=existing_names,
            )

            # Merge, prefer freshly fetched data
            for rec in fetched:
                fn = rec.get("full_name")
                if fn:
                    existing_map[fn] = rec

            merged = list(existing_map.values())
            save_repos_jsonl(REPOS_JSONL, merged)
            print(
                f"Saved repo metadata to {REPOS_JSONL} (fetched {len(fetched)} new, total {len(merged)})"
            )
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