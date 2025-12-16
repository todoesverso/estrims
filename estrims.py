#!/usr/bin/env python3
"""
Async YouTube channel monitor.

Features:
- Load channels from streams.yaml
- Concurrent, rate-limited fetching with retries
- Robust parsing of ytInitialData from channel HTML
- Stores streams and latest status in JSON using PysonDB
- Can run once or loop every N seconds
"""

from __future__ import annotations
import asyncio
import aiohttp
import async_timeout
import re
import json
import logging
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Optional, Dict, Any
import yaml
import sys
import time
from pathlib import Path
from pysondb import getDb  # PysonDB import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("stream-monitor")


# -------------------------
# Configuration / Defaults
# -------------------------
DEFAULT_STATUSES_FILE = "data.json"
DEFAULT_YAML = "estrims.yaml"
CONCURRENT_REQUESTS = 6
REQUEST_TIMEOUT = 15  # seconds
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.5  # seconds (exponential)
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36"
)


# -------------------------
# PysonDB wrapper
# -------------------------
class JsonDB:
    def __init__(
        self,
        statuses_file: str = DEFAULT_STATUSES_FILE,
    ):
        self.statuses_db = getDb(statuses_file)

    def upsert_status(self, stream_key: str, status: Dict[str, Any]):
        status_record = {
            "stream_key": stream_key,
            "datetime": status.get("datetime"),
            "thumbnail": status.get("thumbnail"),
            "live_id": status.get("live_id"),
            "live_title": status.get("live_title"),
            "viewing": status.get("viewing"),
            "stream": status.get("stream", {}),
        }
        existing = self.statuses_db.getByQuery({"stream_key": stream_key})
        if existing:
            rec_id = existing[0]["id"]
            self.statuses_db.updateById(rec_id, status_record)
        else:
            self.statuses_db.add(status_record)

    def get_streams(self) -> List[Dict[str, str]]:
        return self.streams_db.getAll()

    def close(self):
        pass  # PysonDB doesn't require closing


# -------------------------
# Dataclasses
# -------------------------
@dataclass
class Stream:
    title: str
    channel_url: str

    def to_dict(self):
        return {"title": self.title, "channel_url": self.channel_url}


@dataclass
class StreamStatus:
    datetime: str
    stream: Stream
    stream_key: str
    thumbnail: str = ""
    live_id: Optional[str] = None
    live_title: Optional[str] = None
    viewing: int = 0

    def to_dict(self):
        d = asdict(self)
        d["stream"] = self.stream.to_dict()
        return d


# -------------------------
# Utilities
# -------------------------
def access_path(data: Any, path: List[Any]) -> Any:
    for key in path:
        if isinstance(data, dict):
            data = data[key]
        elif isinstance(data, list):
            data = data[key]
        else:
            raise TypeError(f"Expected dict or list, got {type(data).__name__}")
    return data


def get_nested_value(data: dict, candidate_paths: List[List[Any]]) -> Optional[Any]:
    for path in candidate_paths:
        try:
            return access_path(data, path)
        except (KeyError, IndexError, TypeError):
            continue
    return None


def get_title(base_path: dict) -> Optional[str]:
    paths = [["title", "runs", 0, "text"], ["title", "simpleText"]]
    return get_nested_value(base_path, paths)


# -------------------------
# Parsers
# -------------------------
YT_INITIAL_RE = re.compile(r"ytInitialData\s*=\s*(\{.*?\});", re.MULTILINE | re.DOTALL)


def parse_yt_initial_data(html: str) -> Optional[dict]:
    match = YT_INITIAL_RE.search(html)
    if not match:
        return None
    payload = match.group(1)
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        try:
            trimmed = payload.strip().rstrip(";")
            return json.loads(trimmed)
        except Exception:
            logger.exception("Failed to JSON-decode ytInitialData")
            return None


def parse_thumbnail(script_dict: dict) -> Optional[str]:
    try:
        return script_dict["metadata"]["channelMetadataRenderer"]["avatar"][
            "thumbnails"
        ][0]["url"]
    except Exception:
        return None


def parse_live_stream(script_dict: dict) -> dict:
    try:
        base_path = [
            "contents",
            "twoColumnBrowseResultsRenderer",
            "tabs",
            0,
            "tabRenderer",
            "content",
            "sectionListRenderer",
            "contents",
            0,
            "itemSectionRenderer",
            "contents",
            0,
            "channelFeaturedContentRenderer",
            "items",
            0,
            "videoRenderer",
        ]
        vr = access_path(script_dict, base_path)
        return {"live_title": get_title(vr), "live_id": vr.get("videoId")}
    except Exception:
        return {"live_title": None, "live_id": None}


def parse_curr_view(script_dict: dict) -> int:
    try:
        path = [
            "contents",
            "twoColumnBrowseResultsRenderer",
            "tabs",
            0,
            "tabRenderer",
            "content",
            "sectionListRenderer",
            "contents",
            0,
            "itemSectionRenderer",
            "contents",
            0,
            "channelFeaturedContentRenderer",
            "items",
            0,
            "videoRenderer",
            "viewCountText",
            "runs",
            0,
            "text",
        ]
        cv = access_path(script_dict, path)
        digits = re.sub(r"[^\d]", "", str(cv))
        return int(digits) if digits else 0
    except Exception:
        return 0


# -------------------------
# HTTP Fetch
# -------------------------
async def fetch_html(
    session: aiohttp.ClientSession, url: str, timeout: int = REQUEST_TIMEOUT
) -> Optional[str]:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with async_timeout.timeout(timeout):
                headers = {
                    "User-Agent": USER_AGENT,
                    "Accept-Language": "en-US,en;q=0.9",
                }
                async with session.get(url, headers=headers) as resp:
                    text = await resp.text(errors="replace")
                    if resp.status >= 400:
                        logger.warning("Bad status %s for %s", resp.status, url)
                        last_exc = RuntimeError(f"Status {resp.status}")
                    else:
                        return text
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            last_exc = e
            logger.debug("Fetch attempt %d failed for %s: %s", attempt, url, e)
        backoff = (RETRY_BACKOFF_BASE**attempt) + random.random()
        await asyncio.sleep(min(backoff, 30))
    logger.error("All fetch attempts failed for %s: %s", url, last_exc)
    return None


# -------------------------
# Monitor
# -------------------------
async def process_stream(
    db: JsonDB,
    session: aiohttp.ClientSession,
    stream: Stream,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        logger.info("Fetching %s", stream.channel_url)
        html = await fetch_html(session, stream.channel_url)
        if not html:
            logger.warning("No HTML for %s", stream.title)
            return None

        script_dict = parse_yt_initial_data(html)
        if not script_dict:
            logger.info("No ytInitialData found for %s", stream.title)
            return None

        thumbnail = parse_thumbnail(script_dict) or ""
        live = parse_live_stream(script_dict)
        viewing = parse_curr_view(script_dict)

        status = StreamStatus(
            datetime=str(datetime.now()),
            stream=stream,
            stream_key=stream.title,
            thumbnail=thumbnail,
            viewing=viewing,
            live_id=live.get("live_id"),
            live_title=live.get("live_title"),
        )
        db.upsert_status(stream.title, status.to_dict())
        logger.info(
            "Updated status for %s (live=%s views=%s)",
            stream.title,
            status.live_id,
            status.viewing,
        )
        return status


async def monitor_once(
    db: JsonDB, streams: List[Stream], concurrency: int = CONCURRENT_REQUESTS
):
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT + 5)
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    semaphore = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        tasks = [process_stream(db, session, s, semaphore) for s in streams]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    for s, r in zip(streams, results):
        if isinstance(r, Exception):
            logger.exception("Worker failed for stream %s: %s", s.title, r)
    return results


# -------------------------
# CLI
# -------------------------
def load_streams_from_yaml(path: str = DEFAULT_YAML) -> List[Stream]:
    p = Path(path)
    if not p.exists():
        logger.error("Streams YAML %s not found.", path)
        return []
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    streams: List[Stream] = []
    for item in raw:
        title = item.get("title")
        channel = item.get("channel_url")
        if title and channel:
            streams.append(Stream(title=title, channel_url=channel))
        else:
            logger.warning("Invalid stream entry in YAML: %r", item)
    return streams


async def run_loop(
    interval_seconds: int, db: JsonDB, streams: List[Stream], concurrency: int
):
    while True:
        start = time.time()
        logger.info("Starting cycle for %d streams", len(streams))
        await monitor_once(db, streams, concurrency=concurrency)
        elapsed = time.time() - start
        sleep_for = max(0, interval_seconds - elapsed)
        logger.info("Cycle completed in %.2f s; sleeping %.2f s", elapsed, sleep_for)
        await asyncio.sleep(sleep_for)


def print_usage():
    print(
        "Usage:\n"
        "  python monitor.py once [streams.yaml]        # run one cycle\n"
        "  python monitor.py loop <seconds> [streams.yaml]  # run forever every N seconds\n"
    )


def main(argv):
    if len(argv) < 2:
        print_usage()
        return 1

    cmd = argv[1]
    yaml_path = argv[2] if len(argv) > 2 and not cmd.isdigit() else DEFAULT_YAML

    db = JsonDB(DEFAULT_STATUSES_FILE)
    streams = load_streams_from_yaml(yaml_path)
    if not streams:
        logger.error("No streams loaded from %s", yaml_path)
        return 2

    if cmd == "once":
        asyncio.run(monitor_once(db, streams))
    elif cmd == "loop":
        if len(argv) < 3:
            print_usage()
            return 1
        try:
            interval = int(argv[2])
            yaml_path = argv[3] if len(argv) > 3 else DEFAULT_YAML
            streams = load_streams_from_yaml(yaml_path)
            asyncio.run(
                run_loop(interval, db, streams, concurrency=CONCURRENT_REQUESTS)
            )
        except ValueError:
            print_usage()
            return 1
    else:
        print_usage()
        return 1

    db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
