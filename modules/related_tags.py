"""
Related tags for Autocomplete++.

Related tags are fetched from the Danbooru "related tags" endpoint (the same data
the Danbooru website shows) by the ComfyUI server, not by the browser, so there
are no CORS problems and results can be cached on disk:

    data/related-tags/<order>/<xx>/<sha1(tag)>.json

Entries are valid for ``ttl_days`` (default 7); when Danbooru cannot be reached a
stale cache entry is returned instead of an error.
"""
import asyncio
import base64
import hashlib
import json
import os
import shutil
import time

import aiohttp
from aiohttp import web
import server

from . import logger
from . import cooccurrence
from .api import DATA_DIR, _read_user_data

CACHE_DIR = os.path.join(DATA_DIR, "related-tags")
DANBOORU_URL = "https://danbooru.donmai.us/related_tag.json"
USER_AGENT = "ComfyUI-Autocomplete-Plus-Plus (related tags)"
ORDERS = ("Cosine", "Jaccard", "Overlap", "Frequency")
SOURCES = ("Auto", "Local", "Danbooru")
MAX_ITEMS = 300
MAX_TAG_LENGTH = 200
MEMORY_CACHE_LIMIT = 400

_memory = {}          # (order, tag) -> payload
_inflight = {}        # (order, tag) -> asyncio.Future
_semaphore = None


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def clean_tag(raw):
    """Danbooru tag names are lowercase with underscores instead of spaces."""
    if not isinstance(raw, str):
        return ""
    tag = raw.strip().lower().replace(" ", "_")
    while "__" in tag:
        tag = tag.replace("__", "_")
    return tag


def _num(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0  # drop NaN


def normalize_related(payload, query_tag=""):
    """
    Turn a Danbooru related-tag response into
    ``[{name, category, post_count, cosine, jaccard, overlap, frequency}]``.

    Accepts the current shape (``related_tags`` entries carrying a nested ``tag``
    object), a flat variant (name/category directly on the entry) and the legacy
    ``tags: [[name, category], ...]`` shape.
    """
    entries = []
    if isinstance(payload, dict):
        if isinstance(payload.get("related_tags"), list):
            entries = payload["related_tags"]
        elif isinstance(payload.get("tags"), list):
            entries = payload["tags"]
    elif isinstance(payload, list):
        entries = payload

    items = []
    seen = set()
    query_tag = clean_tag(query_tag)
    for entry in entries:
        name, category, post_count, metrics = "", 0, 0, {}
        if isinstance(entry, (list, tuple)) and entry:
            name = entry[0] if isinstance(entry[0], str) else ""
            if len(entry) > 1:
                category = entry[1]
        elif isinstance(entry, dict):
            tag = entry.get("tag") if isinstance(entry.get("tag"), dict) else entry
            name = tag.get("name") or tag.get("tag_name") or ""
            category = tag.get("category", 0)
            post_count = tag.get("post_count", 0)
            metrics = entry
        name = clean_tag(name)
        if not name or name == query_tag or name in seen:
            continue
        seen.add(name)
        try:
            category = int(category)
        except (TypeError, ValueError):
            category = 0
        items.append({
            "name": name,
            "category": category,
            "post_count": int(_num(post_count)),
            "cosine": _num(metrics.get("cosine_similarity")),
            "jaccard": _num(metrics.get("jaccard_similarity")),
            "overlap": _num(metrics.get("overlap_coefficient")),
            "frequency": _num(metrics.get("frequency")),
        })
        if len(items) >= MAX_ITEMS:
            break
    return items


def cache_path(order, tag):
    digest = hashlib.sha1(tag.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, order.lower(), digest[:2], digest + ".json")


def read_cache(order, tag):
    key = (order, tag)
    hit = _memory.get(key)
    if hit is not None:
        return hit
    try:
        with open(cache_path(order, tag), "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("items"), list) and data.get("tag") == tag:
            _remember(key, data)
            return data
    except (OSError, ValueError):
        pass
    return None


def _remember(key, payload):
    if len(_memory) >= MEMORY_CACHE_LIMIT:
        _memory.pop(next(iter(_memory)))
    _memory[key] = payload


def write_cache(order, tag, items):
    payload = {"tag": tag, "order": order, "fetched_at": int(time.time()), "items": items}
    _remember((order, tag), payload)
    path = cache_path(order, tag)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as e:
        logger.warning(f"Could not write related tags cache: {e}")
    return payload


def is_fresh(payload, ttl_days, now=None):
    if ttl_days <= 0:
        return False
    now = time.time() if now is None else now
    return (now - float(payload.get("fetched_at", 0))) < ttl_days * 86400


# --------------------------------------------------------------------------- #
# Network
# --------------------------------------------------------------------------- #
def _credentials():
    """Optional Danbooru login (username + API key) saved in the extension settings."""
    try:
        settings = _read_user_data().get("settings", {})
    except Exception:
        return None
    user = str(settings.get("DanbooruUsername", "") or "").strip()
    key = str(settings.get("DanbooruApiKey", "") or "").strip()
    if user and key:
        return user, key
    return None


def describe_http_error(status, body, authenticated):
    """Turn a failed Danbooru response into a message that says what to do about it."""
    text = (body or "")[:2000]
    low = text.lower()
    cloudflare = "cloudflare" in low or "just a moment" in low or "cf-chl" in low or "attention required" in low
    if status == 429:
        return "Danbooru rate limit reached, try again in a moment"
    if status in (401, 403):
        if cloudflare:
            return (f"Danbooru refused the request (HTTP {status}, Cloudflare protection). "
                    "Set the related tags source to 'Auto' or 'Local data' to compute them offline.")
        if authenticated:
            return (f"Danbooru refused the request (HTTP {status}) even with your login - "
                    "check the username and API key in the settings.")
        return (f"Danbooru refused the request (HTTP {status}). It may require a login for this lookup: "
                "add your Danbooru username and API key in Settings > Autocomplete++ > Related Tags.")
    return f"Danbooru returned HTTP {status}"


async def _fetch_danbooru(tag, order):
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(2)
    timeout = aiohttp.ClientTimeout(total=20, connect=8)
    params = {"search[query]": tag, "search[order]": order}
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    creds = _credentials()
    if creds:
        token = base64.b64encode(f"{creds[0]}:{creds[1]}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = "Basic " + token
    async with _semaphore:
        async with aiohttp.ClientSession(timeout=timeout, headers=headers, trust_env=True) as session:
            async with session.get(DANBOORU_URL, params=params) as resp:
                if resp.status in (404, 410):
                    return []
                if resp.status != 200:
                    try:
                        raw = await resp.read()
                        body = raw.decode("utf-8", errors="replace")
                    except Exception:
                        body = ""
                    logger.warning(f"Danbooru related tags request for '{tag}' failed: HTTP {resp.status} "
                                   f"(login {'used' if creds else 'not used'}); response starts with: "
                                   f"{body[:200].strip()!r}")
                    raise RuntimeError(describe_http_error(resp.status, body, bool(creds)))
                try:
                    payload = await resp.json(content_type=None)
                except ValueError:
                    raise RuntimeError("Danbooru returned an unexpected (non-JSON) response; "
                                       "it may be temporarily blocking automated requests")
        await asyncio.sleep(0.1)  # gentle pacing between requests
    return normalize_related(payload, tag)


async def get_related(tag, order, ttl_days):
    """Returns (payload, flags) where flags = {"cached": bool, "stale": bool}."""
    key = (order, tag)
    cached = read_cache(order, tag)
    if cached is not None and is_fresh(cached, ttl_days):
        return cached, {"cached": True, "stale": False}

    pending = _inflight.get(key)
    if pending is not None:
        return await asyncio.shield(pending), {"cached": False, "stale": False}

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    _inflight[key] = future
    try:
        items = await _fetch_danbooru(tag, order)
        payload = write_cache(order, tag, items)
        future.set_result(payload)
        return payload, {"cached": False, "stale": False}
    except Exception as exc:
        if cached is not None:
            logger.warning(f"Related tags for '{tag}' could not be refreshed ({exc}); serving cached copy.")
            future.set_result(cached)
            return cached, {"cached": True, "stale": True}
        future.set_exception(exc)
        future.exception()  # mark retrieved so asyncio does not log it a second time
        raise
    finally:
        _inflight.pop(key, None)


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@server.PromptServer.instance.routes.get("/autocomplete-plus-plus/related-tags")
async def route_related_tags(request):
    tag = clean_tag(request.query.get("tag", ""))
    if not tag or len(tag) > MAX_TAG_LENGTH:
        return web.json_response({"success": False, "error": "Missing or invalid 'tag'"}, status=400)

    order = request.query.get("order", "Cosine")
    if order not in ORDERS:
        order = "Cosine"
    try:
        ttl_days = max(0.0, min(float(request.query.get("ttl_days", 7)), 365.0))
    except ValueError:
        ttl_days = 7.0

    source = request.query.get("source", "Auto")
    if source not in SOURCES:
        source = "Auto"

    if source in ("Auto", "Local"):
        state, index = cooccurrence.ensure_index()
        if state == "missing":
            st = cooccurrence.read_state()
            if source == "Auto" and st.get("last_error") and not cooccurrence.status()["downloading"]:
                state = "fallback"
            else:
                cooccurrence.kick_download()
                if cooccurrence.status()["downloading"] or not st.get("last_error"):
                    state = "downloading"
                else:
                    return web.json_response({"success": False, "tag": tag, "order": order,
                                              "error": "Related tags data could not be downloaded: "
                                                       + st["last_error"]})
        if state == "error":
            return web.json_response({"success": False, "tag": tag, "order": order,
                                      "error": "Related tags data could not be read"})
        if state in ("building", "downloading"):
            return web.json_response({"success": True, "pending": state, "tag": tag, "order": order, "items": []})
        if state == "ready":
            items = await asyncio.get_running_loop().run_in_executor(None, index.lookup, tag, order)
            return web.json_response({"success": True, "tag": tag, "order": order, "source": "local",
                                      "cached": False, "stale": False, "items": items})

    try:
        payload, flags = await get_related(tag, order, ttl_days)
    except Exception as e:
        message = str(e) or e.__class__.__name__
        logger.warning(f"Related tags lookup for '{tag}' failed: {message}")
        return web.json_response({"success": False, "error": message, "tag": tag, "order": order})

    return web.json_response({
        "success": True,
        "tag": tag,
        "order": order,
        "source": "danbooru",
        "fetched_at": payload.get("fetched_at"),
        "cached": flags["cached"],
        "stale": flags["stale"],
        "items": payload["items"],
    })


@server.PromptServer.instance.routes.post("/autocomplete-plus-plus/related-tags/clear-cache")
async def route_clear_cache(_request):
    _memory.clear()
    removed = 0
    if os.path.isdir(CACHE_DIR):
        for _root, _dirs, files in os.walk(CACHE_DIR):
            removed += sum(1 for f in files if f.endswith(".json"))
        shutil.rmtree(CACHE_DIR, ignore_errors=True)
    return web.json_response({"success": True, "removed": removed})
