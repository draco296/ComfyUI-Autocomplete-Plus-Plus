"""
Local related tags (no Danbooru API needed).

Danbooru sits behind Cloudflare, which blocks automated requests, so related tags
are computed locally from a tag co-occurrence file instead:

    tag_a,tag_b,count        (optional header; one row per pair, count >= 100)

The file is the public dataset used by ComfyUI-Autocomplete-Plus
(https://huggingface.co/datasets/newtextdoc1111/danbooru-tag-csv, MIT). It is
downloaded to ``data/danbooru_tags_cooccurrence.csv`` in the background and loaded
into a compact in-memory index (numpy arrays, built in a worker thread).

For a query tag A and a neighbour B with pair count c, and post counts nA / nB
taken from the active tag list:

    cosine    = c / sqrt(nA * nB)        jaccard   = c / (nA + nB - c)
    overlap   = c / min(nA, nB)          frequency = c / nA

Post counts of the tag list and of the co-occurrence snapshot can differ slightly,
so every score is clamped to [0, 1].
"""
import asyncio
import csv
import json
import math
import os
import threading
import time
from array import array

import aiohttp

from . import logger
from . import api as _api
from .api import DATA_DIR

HF_DATASET = "newtextdoc1111/danbooru-tag-csv"
FILE_NAME = "danbooru_tags_cooccurrence.csv"
URL = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{FILE_NAME}"
STATE_NAME = "cooccurrence_state.json"
USER_AGENT = "ComfyUI-Autocomplete-Plus-Plus (related tags data)"

MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
MIN_ROWS = 1000
MAX_ITEMS = 300

_download_lock = threading.Lock()
_build_lock = threading.Lock()
_index = None
_index_key = None
_index_error = ""
_build_thread = None


def local_path():
    return os.path.join(DATA_DIR, FILE_NAME)


def _state_path():
    return os.path.join(DATA_DIR, STATE_NAME)


def read_state():
    try:
        with open(_state_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except (OSError, ValueError):
        pass
    return {}


def write_state(state):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = _state_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, _state_path())
    except OSError as e:
        logger.warning(f"Could not write related tags data state: {e}")


# --------------------------------------------------------------------------- #
# Parsing / scoring (pure, unit-tested)
# --------------------------------------------------------------------------- #
def load_tag_meta(path):
    """{name: (category, post_count)} from a tag CSV (``tag,category,count[,aliases]``)."""
    meta = {}
    if not path:
        return meta
    try:
        with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 3 or not row[0] or row[0].startswith("#"):
                    # '#'-prefixed tags are real Danbooru tags, but the main CSV parser skips them too
                    if len(row) >= 3 and row[0].startswith("#") and row[1].strip().isdigit():
                        pass
                    else:
                        continue
                try:
                    meta[row[0].strip()] = (int(row[1]), int(row[2]))
                except ValueError:
                    continue
    except OSError:
        pass
    return meta


def parse_pairs(path):
    """
    Read a co-occurrence CSV into (names, a_ids, b_ids, counts). Malformed rows
    and the optional ``tag_a,tag_b,count`` header are skipped.
    """
    ids = {}
    names = []
    a_ids, b_ids, counts = array("i"), array("i"), array("i")

    def intern(tag):
        i = ids.get(tag)
        if i is None:
            i = len(names)
            ids[tag] = i
            names.append(tag)
        return i

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip("\r\n").split(",")
            if len(parts) != 3:
                continue
            a, b, c = parts[0].strip(), parts[1].strip(), parts[2].strip()
            if not a or not b or a == b or not c.isdigit():
                continue
            a_ids.append(intern(a))
            b_ids.append(intern(b))
            counts.append(int(c))
    return names, a_ids, b_ids, counts


class Index:
    """Compressed-sparse-row adjacency of the (symmetric) co-occurrence graph."""

    def __init__(self, names, a_ids, b_ids, counts, tag_meta):
        import numpy as np
        self.np = np
        self.ids = {n: i for i, n in enumerate(names)}
        n = len(names)
        a = np.frombuffer(a_ids, dtype=np.int32) if len(a_ids) else np.zeros(0, dtype=np.int32)
        b = np.frombuffer(b_ids, dtype=np.int32) if len(b_ids) else np.zeros(0, dtype=np.int32)
        c = np.frombuffer(counts, dtype=np.int32) if len(counts) else np.zeros(0, dtype=np.int32)
        src = np.concatenate([a, b])
        dst = np.concatenate([b, a])
        cnt = np.concatenate([c, c])
        order = np.argsort(src, kind="stable")
        self.dst = dst[order]
        self.cnt = cnt[order]
        self.offsets = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(src, minlength=n), out=self.offsets[1:])
        self.names = names
        self.post = np.array([float(tag_meta.get(nm, (0, 0))[1]) for nm in names], dtype=np.float64)
        self.cat = np.array([int(tag_meta.get(nm, (0, 0))[0]) for nm in names], dtype=np.int16)
        self.rows = int(len(a))

    def lookup(self, tag, order="Cosine", limit=MAX_ITEMS):
        np = self.np
        i = self.ids.get(tag)
        if i is None:
            return []
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        if hi <= lo:
            return []
        nb = self.dst[lo:hi]
        c = self.cnt[lo:hi].astype(np.float64)
        na = self.post[i]
        nbp = self.post[nb]
        valid = (nbp > 0) & (na > 0)
        if not valid.any():
            return []
        nb, c, nbp = nb[valid], c[valid], nbp[valid]

        def clamp(x):
            return np.clip(x, 0.0, 1.0)

        cosine = clamp(c / np.sqrt(na * nbp))
        jaccard = clamp(c / np.maximum(na + nbp - c, 1.0))
        overlap = clamp(c / np.minimum(na, nbp))
        frequency = clamp(c / na)
        key = {"Cosine": cosine, "Jaccard": jaccard, "Overlap": overlap, "Frequency": frequency}.get(order, cosine)
        # primary: chosen score (desc); tie-break: pair count (desc)
        top = np.lexsort((-c, -key))[:limit]
        out = []
        for j in top:
            out.append({
                "name": self.names[int(nb[j])],
                "category": int(self.cat[nb[j]]),
                "post_count": int(nbp[j]),
                "cosine": round(float(cosine[j]), 6),
                "jaccard": round(float(jaccard[j]), 6),
                "overlap": round(float(overlap[j]), 6),
                "frequency": round(float(frequency[j]), 6),
            })
        return out


def build_index(pairs_path, tag_list_path):
    names, a, b, c = parse_pairs(pairs_path)
    return Index(names, a, b, c, load_tag_meta(tag_list_path))


# --------------------------------------------------------------------------- #
# Index lifecycle (built lazily in a worker thread)
# --------------------------------------------------------------------------- #
def _file_key(path):
    try:
        st = os.stat(path)
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _current_key():
    return (_file_key(local_path()), _file_key(_api.find_file_path("danbooru.csv")))


def _build_worker(key):
    global _index, _index_key, _index_error
    started = time.time()
    try:
        index = build_index(local_path(), _api.find_file_path("danbooru.csv"))
        _index, _index_key, _index_error = index, key, ""
        logger.info(f"Related tags index ready: {len(index.names)} tags, {index.rows} pairs "
                    f"({time.time() - started:.1f}s).")
    except Exception as e:
        _index_error = str(e) or e.__class__.__name__
        logger.warning(f"Could not build the related tags index: {_index_error}")
        _index, _index_key = None, key
    finally:
        _build_lock.release()


def ensure_index():
    """
    Returns ``(state, index)``; state is "ready", "building", "missing" or "error".
    Starts a background build when needed.
    """
    global _build_thread
    key = _current_key()
    if key[0] is None:
        return "missing", None
    if _index is not None and _index_key == key:
        return "ready", _index
    if _index is None and _index_key == key and _index_error:
        return "error", None
    if _build_lock.acquire(blocking=False):
        _build_thread = threading.Thread(target=_build_worker, args=(key,), name="ac-related-index", daemon=True)
        _build_thread.start()
    return "building", None


def invalidate():
    global _index, _index_key, _index_error
    _index, _index_key, _index_error = None, None, ""


def status():
    state = read_state()
    p = local_path()
    return {
        "present": os.path.isfile(p),
        "rows": state.get("rows"),
        "downloaded_at": state.get("downloaded_at"),
        "last_modified": state.get("last_modified"),
        "last_error": state.get("last_error", ""),
        "downloading": _download_lock.locked(),
        "index_ready": _index is not None and _index_key == _current_key(),
    }


# --------------------------------------------------------------------------- #
# Download / update
# --------------------------------------------------------------------------- #
def validate_pairs_file(path):
    """Cheap sanity check of a downloaded file: returns (ok, rows, message)."""
    rows = bad = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) == 3 and parts[0] and parts[1] and parts[2].strip().isdigit():
                    rows += 1
                elif not (rows == 0 and line.lower().startswith("tag_a,")):
                    bad += 1
                    if bad > 50 and bad > rows * 0.01:
                        return False, rows, "file does not look like a tag_a,tag_b,count CSV"
    except OSError as e:
        return False, rows, f"unreadable file: {e}"
    if rows < MIN_ROWS:
        return False, rows, f"only {rows} valid rows"
    return True, rows, "ok"


async def _remote_stamp(session):
    """(last_modified, etag) of the remote file, or (None, None) when unknown."""
    async with session.head(URL, allow_redirects=True) as resp:
        if resp.status != 200:
            raise RuntimeError(f"Hugging Face returned HTTP {resp.status}")
        h = resp.headers
        return h.get("Last-Modified"), h.get("ETag") or h.get("X-Linked-Etag")


async def update_cooccurrence(force=False):
    """
    Download the co-occurrence file if it is missing or changed upstream.
    Returns {"status": "updated"|"up_to_date"|"busy"|"error", ...}; never raises.
    """
    if not _download_lock.acquire(blocking=False):
        return {"status": "busy"}
    state = read_state()
    dest = local_path()
    tmp = dest + ".tmp"
    try:
        timeout = aiohttp.ClientTimeout(total=1800, connect=20)
        async with aiohttp.ClientSession(timeout=timeout, headers={"User-Agent": USER_AGENT}, trust_env=True) as session:
            stamp = (None, None)
            try:
                stamp = await _remote_stamp(session)
            except Exception as e:
                if not os.path.isfile(dest):
                    raise
                logger.warning(f"Could not check for newer related tags data: {e}")
                return {"status": "up_to_date", "rows": state.get("rows"), "note": "remote check failed"}

            known = (state.get("last_modified"), state.get("etag"))
            if not force and os.path.isfile(dest) and stamp != (None, None) and (stamp[0], stamp[1]) == known:
                return {"status": "up_to_date", "rows": state.get("rows")}

            os.makedirs(DATA_DIR, exist_ok=True)
            total = 0
            async with session.get(URL, allow_redirects=True) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Hugging Face returned HTTP {resp.status}")
                with open(tmp, "wb") as f:
                    async for chunk in resp.content.iter_chunked(1024 * 1024):
                        total += len(chunk)
                        if total > MAX_DOWNLOAD_BYTES:
                            raise RuntimeError("download is larger than the allowed maximum")
                        f.write(chunk)
        ok, rows, message = await asyncio.get_running_loop().run_in_executor(None, validate_pairs_file, tmp)
        if not ok:
            raise RuntimeError(f"downloaded file rejected: {message}")
        os.replace(tmp, dest)
        invalidate()
        state.update({"rows": rows, "downloaded_at": int(time.time()), "last_modified": stamp[0],
                      "etag": stamp[1], "last_error": ""})
        write_state(state)
        logger.info(f"Related tags data downloaded ({rows} pairs, {total // (1024 * 1024)} MB).")
        return {"status": "updated", "rows": rows}
    except Exception as e:
        message = str(e) or e.__class__.__name__
        logger.warning(f"Related tags data download failed: {message}")
        state["last_error"] = message
        write_state(state)
        return {"status": "error", "message": message}
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        _download_lock.release()


_bg_task = None


def kick_download():
    """Start a background download unless one is running / the file exists. Safe to call often."""
    global _bg_task
    if os.path.isfile(local_path()) or _download_lock.locked():
        return
    st = read_state()
    # do not hammer a failing source on every keystroke: retry at most every 10 minutes
    if st.get("last_error") and time.time() - float(st.get("last_attempt", 0) or 0) < 600:
        return
    st["last_attempt"] = int(time.time())
    write_state(st)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    _bg_task = loop.create_task(update_cooccurrence())
