"""
Automatic tag list updates for Autocomplete++.

Fresh Danbooru / e621 tag lists are downloaded from the public-domain archive
https://github.com/DraconicDragon/dbr-e621-lists-archive (Unlicense) and stored
in ``data/<site>_latest.csv``. ``api.find_file_path`` serves that file in place of
the bundled ``tags/<site>.csv`` whenever it exists, so nothing under ``tags/``
(which is tracked by git) is ever modified.

Safety rules:
  * a download is validated (row count, column shape) before it replaces anything;
  * the replacement is atomic (``os.replace``) - a failed update keeps the old list;
  * only one update runs at a time.

The archive is committed roughly once per quarter, so "latest" means the newest
file in that archive, not a live dump of Danbooru.
"""
import asyncio
import csv
import json
import os
import re
import threading
import time

import aiohttp
from aiohttp import web
import server

from . import logger
from . import cooccurrence
from .api import DATA_DIR, TAGS_DIR, _read_user_data

ARCHIVE_REPO = "DraconicDragon/dbr-e621-lists-archive"
LISTING_URL = "https://api.github.com/repos/" + ARCHIVE_REPO + "/contents/tag-lists/{site}"

STATE_FILE = os.path.join(DATA_DIR, "tag_update_state.json")
USER_AGENT = "ComfyUI-Autocomplete-Plus-Plus (tag list updater)"

# Only lists with a post-count threshold in this range are considered: the
# archive also holds huge pt2 dumps (too heavy for an in-browser index).
PT_MIN, PT_MAX = 10, 100
MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024

SITES = {
    "danbooru": {"bundled": "danbooru.csv", "latest": "danbooru_latest.csv", "min_rows": 80_000},
    "e621": {"bundled": "e621.csv", "latest": "e621_latest.csv", "min_rows": 40_000},
}

DEFAULT_AUTO = True
DEFAULT_INTERVAL_DAYS = 7
STARTUP_DELAY_SECONDS = 20

_run_lock = threading.Lock()
_state_lock = threading.Lock()


# --------------------------------------------------------------------------- #
# State
# --------------------------------------------------------------------------- #
def _read_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data.setdefault("sites", {})
                return data
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning(f"Could not read tag update state: {e}")
    return {"last_check": 0, "last_attempt": 0, "last_error": "", "sites": {}}


def _write_state(state):
    with _state_lock:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            os.replace(tmp, STATE_FILE)
        except Exception as e:
            logger.warning(f"Could not write tag update state: {e}")


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def pick_latest(listing, site):
    """
    Pick the newest list for ``site`` from a GitHub contents listing.
    Newest date wins; for the same date the lower post threshold (more tags) wins.
    Returns the matching listing entry or None.
    """
    pattern = re.compile(r"^" + re.escape(site) + r"_(\d{4}-\d{2}-\d{2})_pt(\d+)[-\w]*\.csv$")
    best = None
    best_key = None
    for entry in listing or []:
        if not isinstance(entry, dict) or entry.get("type", "file") != "file":
            continue
        m = pattern.match(str(entry.get("name", "")))
        if not m or not entry.get("download_url"):
            continue
        date, pt = m.group(1), int(m.group(2))
        if not (PT_MIN <= pt <= PT_MAX):
            continue
        key = (date, -pt)
        if best_key is None or key > best_key:
            best_key, best = key, entry
    if best is None:
        return None
    m = pattern.match(best["name"])
    return {
        "name": best["name"],
        "version": m.group(1),
        "pt": int(m.group(2)),
        "url": best["download_url"],
        "size": best.get("size"),
    }


def validate_csv(path, min_rows):
    """
    Validate a downloaded tag CSV (``tag,category,count[,aliases]`` without header).
    Returns (ok, rows, message).
    """
    rows = bad = 0
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if not row:
                    continue
                if len(row) >= 3 and row[0].strip() and row[1].strip().isdigit() and row[2].strip().isdigit():
                    rows += 1
                else:
                    bad += 1
    except Exception as e:
        return False, rows, f"unreadable file: {e}"
    if rows < min_rows:
        return False, rows, f"only {rows} valid rows (expected at least {min_rows})"
    if bad > max(5, rows * 0.001):
        return False, rows, f"{bad} malformed rows"
    return True, rows, "ok"


def count_rows(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def should_auto_update(settings, state, now=None):
    """Decide whether the startup check should run, from persisted settings + state."""
    now = time.time() if now is None else now
    enabled = settings.get("AutoUpdateTags", DEFAULT_AUTO)
    if not enabled:
        return False
    try:
        interval = float(settings.get("AutoUpdateIntervalDays", DEFAULT_INTERVAL_DAYS))
    except (TypeError, ValueError):
        interval = DEFAULT_INTERVAL_DAYS
    interval = max(1.0, min(interval, 90.0))
    return (now - float(state.get("last_check", 0) or 0)) >= interval * 86400


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
async def _get_json(session, url):
    async with session.get(url, headers={"Accept": "application/vnd.github+json"}) as resp:
        if resp.status == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
            raise RuntimeError("GitHub API rate limit reached, try again later")
        if resp.status != 200:
            raise RuntimeError(f"GitHub listing returned HTTP {resp.status}")
        return await resp.json(content_type=None)


async def _download(session, url, dest):
    total = 0
    async with session.get(url) as resp:
        if resp.status != 200:
            raise RuntimeError(f"download returned HTTP {resp.status}")
        with open(dest, "wb") as f:
            async for chunk in resp.content.iter_chunked(256 * 1024):
                total += len(chunk)
                if total > MAX_DOWNLOAD_BYTES:
                    raise RuntimeError("download is larger than the allowed maximum")
                f.write(chunk)
    return total


def _current_rows(site, state):
    cfg = SITES[site]
    info = state.get("sites", {}).get(site) or {}
    latest = os.path.join(DATA_DIR, cfg["latest"])
    if info.get("rows") and os.path.isfile(latest):
        return int(info["rows"])
    return count_rows(os.path.join(TAGS_DIR, cfg["bundled"]))


async def _update_site(session, site, force, state):
    cfg = SITES[site]
    listing = await _get_json(session, LISTING_URL.format(site=site))
    choice = pick_latest(listing, site)
    if not choice:
        return {"status": "error", "message": f"no suitable {site} list found in the archive"}

    info = state.get("sites", {}).get(site) or {}
    dest = os.path.join(DATA_DIR, cfg["latest"])
    if not force and info.get("source_file") == choice["name"] and os.path.isfile(dest):
        return {"status": "up_to_date", "version": choice["version"], "rows": info.get("rows"),
                "source_file": choice["name"]}

    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = dest + ".tmp"
    try:
        await _download(session, choice["url"], tmp)
        ok, rows, message = validate_csv(tmp, cfg["min_rows"])
        if not ok:
            return {"status": "error", "message": f"downloaded list rejected: {message}"}
        previous = _current_rows(site, state)
        if previous and rows < previous * 0.7:
            return {"status": "error",
                    "message": f"downloaded list rejected: {rows} rows vs {previous} currently (suspiciously small)"}
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass

    state.setdefault("sites", {})[site] = {
        "version": choice["version"],
        "source_file": choice["name"],
        "rows": rows,
        "downloaded_at": int(time.time()),
    }
    logger.info(f"Updated {site} tag list to {choice['version']} ({rows} tags, {choice['name']}).")
    return {"status": "updated", "version": choice["version"], "rows": rows, "source_file": choice["name"]}


async def update_tag_lists(sites=("danbooru",), force=False):
    """
    Check the archive and update the requested sites. Safe to call concurrently:
    a second call while an update is running returns status "busy".
    """
    if not _run_lock.acquire(blocking=False):
        return {"success": False, "busy": True, "results": {}, "message": "an update is already running"}
    results = {}
    reached_archive = False
    state = {"last_check": 0, "last_attempt": 0, "last_error": "", "sites": {}}
    try:
        state = _read_state()
        state["last_attempt"] = int(time.time())
        timeout = aiohttp.ClientTimeout(total=300, connect=20)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}, trust_env=True) as session:
            for site in sites:
                if site not in SITES:
                    continue
                try:
                    results[site] = await _update_site(session, site, force, state)
                    if results[site]["status"] in ("updated", "up_to_date"):
                        reached_archive = True
                except Exception as e:
                    logger.warning(f"Tag list update for {site} failed: {e}")
                    results[site] = {"status": "error", "message": str(e) or e.__class__.__name__}
    finally:
        if reached_archive:
            state["last_check"] = int(time.time())
        errors = [r.get("message", "") for r in results.values() if r.get("status") == "error"]
        state["last_error"] = "; ".join(m for m in errors if m)
        _write_state(state)
        _run_lock.release()

    success = bool(results) and all(r["status"] != "error" for r in results.values())
    return {"success": success, "busy": False, "results": results}


def get_status():
    state = _read_state()
    sites = {}
    for site, cfg in SITES.items():
        info = state.get("sites", {}).get(site) or {}
        latest = os.path.join(DATA_DIR, cfg["latest"])
        using_latest = os.path.isfile(latest)
        sites[site] = {
            "using_latest": using_latest,
            "version": info.get("version") if using_latest else None,
            "rows": info.get("rows") if using_latest else None,
            "bundled_file": cfg["bundled"],
        }
    return {
        "success": True,
        "running": _run_lock.locked(),
        "last_check": state.get("last_check", 0),
        "last_error": state.get("last_error", ""),
        "sites": sites,
        "related_data": cooccurrence.status(),
    }


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@server.PromptServer.instance.routes.get("/autocomplete-plus-plus/tags/update-status")
async def route_update_status(_request):
    return web.json_response(get_status())


@server.PromptServer.instance.routes.post("/autocomplete-plus-plus/tags/update")
async def route_update(request):
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    sites = ["danbooru"]
    if payload.get("e621"):
        sites.append("e621")
    result = await update_tag_lists(sites=sites, force=bool(payload.get("force")))
    result["related_data"] = await cooccurrence.update_cooccurrence(force=bool(payload.get("force")))
    result["status"] = get_status()
    return web.json_response(result)


@server.PromptServer.instance.routes.post("/autocomplete-plus-plus/tags/revert")
async def route_revert(_request):
    """Delete the auto-updated lists so the bundled ones are served again."""
    removed = []
    state = _read_state()
    for site, cfg in SITES.items():
        path = os.path.join(DATA_DIR, cfg["latest"])
        if os.path.isfile(path):
            try:
                os.remove(path)
                removed.append(site)
                state.get("sites", {}).pop(site, None)
            except OSError as e:
                logger.warning(f"Could not remove {path}: {e}")
    state["last_check"] = 0
    _write_state(state)
    return web.json_response({"success": True, "removed": removed, "status": get_status()})


# --------------------------------------------------------------------------- #
# Startup check
# --------------------------------------------------------------------------- #
def _background_check():
    try:
        time.sleep(STARTUP_DELAY_SECONDS)
        settings = _read_user_data().get("settings", {})
        state = _read_state()
        if not should_auto_update(settings, state):
            return
        sites = ["danbooru"]
        if settings.get("AutoUpdateE621", False):
            sites.append("e621")
        result = asyncio.run(update_tag_lists(sites=sites))
        asyncio.run(cooccurrence.update_cooccurrence())
        if not result.get("success") and not result.get("busy"):
            problems = "; ".join(r.get("message", "") for r in result.get("results", {}).values() if r.get("message"))
            logger.warning(f"Automatic tag list update did not complete: {problems or 'unknown error'}")
    except Exception as e:
        logger.warning(f"Automatic tag list update crashed: {e}")


def start_background_check():
    """Run the startup check in its own daemon thread so ComfyUI startup is never delayed."""
    threading.Thread(target=_background_check, name="ac-tag-updater", daemon=True).start()
