"""Two-layer semantic cache to avoid re-scoring near-duplicate posts.

Layer 1 (exact)    : SHA-256 of normalized text + model tag -> O(1) Redis GET.
Layer 2 (semantic) : MiniLM embedding cosine-similarity over recently cached
                     vectors; a hit (>= threshold) returns the cached signal.

On a miss we call the sentiment ensemble and write back to both layers.
If Redis is unavailable the cache degrades to a no-op (every call scores fresh)
instead of breaking the pipeline. Groq has no embeddings endpoint, so the
embedding model runs locally.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re

import numpy as np

from app.config import settings
from app.metrics import metrics

logger = logging.getLogger(__name__)

_EXACT_PREFIX = "sent:exact:"
_VECS_KEY = "sent:sem:vecs"  # hash: id -> float32 bytes
_META_PREFIX = "sent:sem:meta:"  # string (TTL): id -> signal JSON

_redis = None
_redis_failed = False
_embedder = None

# in-process mirror of cached vectors for the cosine scan
_ids: list[str] = []
_mat: np.ndarray | None = None
_mirror_loaded = False
_counter = 0


# ── lazy singletons ──────────────────────────────────────────────────────────
def _get_redis():
    global _redis, _redis_failed
    if _redis is None and not _redis_failed:
        try:
            import redis.asyncio as aioredis

            _redis = aioredis.from_url(settings.redis_url, decode_responses=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Redis unavailable (%s); semantic cache disabled", exc)
            _redis_failed = True
    return _redis


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        logger.info("loading embedding model: %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


def embed(text: str) -> np.ndarray:
    vec = _get_embedder().encode(text, normalize_embeddings=True)
    return np.asarray(vec, dtype=np.float32)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _exact_key(text: str, tag: str) -> str:
    digest = hashlib.sha256(f"{_normalize(text)}|{tag}".encode()).hexdigest()
    return _EXACT_PREFIX + digest


# ── mirror management ────────────────────────────────────────────────────────
async def _ensure_mirror(r) -> None:
    global _mirror_loaded, _ids, _mat
    if _mirror_loaded:
        return
    _mirror_loaded = True
    try:
        raw = await r.hgetall(_VECS_KEY)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not load cache mirror: %s", exc)
        return
    items = list(raw.items())[-settings.cache_max_vectors :]
    if not items:
        return
    _ids = [k.decode() for k, _ in items]
    _mat = np.vstack([np.frombuffer(v, dtype=np.float32) for _, v in items])


def _append_mirror(entry_id: str, vec: np.ndarray) -> None:
    global _ids, _mat
    if _mat is None:
        _mat = vec.reshape(1, -1)
        _ids = [entry_id]
    else:
        _mat = np.vstack([_mat, vec])
        _ids.append(entry_id)
    if len(_ids) > settings.cache_max_vectors:
        _ids = _ids[-settings.cache_max_vectors :]
        _mat = _mat[-settings.cache_max_vectors :]


# ── public API ───────────────────────────────────────────────────────────────
async def get(text: str, tag: str) -> dict | None:
    r = _get_redis()
    if r is None:
        return None
    # Layer 1: exact
    try:
        hit = await r.get(_exact_key(text, tag))
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis get failed (%s); bypassing cache", exc)
        return None
    if hit is not None:
        metrics.incr("cache_hits_exact")
        return json.loads(hit)

    # Layer 2: semantic
    await _ensure_mirror(r)
    if _mat is not None and len(_ids):
        q = embed(text)
        sims = _mat @ q
        idx = int(np.argmax(sims))
        if float(sims[idx]) >= settings.cache_similarity_threshold:
            meta = await r.get(_META_PREFIX + _ids[idx])
            if meta is not None:  # still alive (not TTL-expired)
                metrics.incr("cache_hits_semantic")
                return json.loads(meta)
    metrics.incr("cache_misses")
    return None


async def put(text: str, tag: str, signal: dict) -> None:
    global _counter
    r = _get_redis()
    if r is None:
        return
    payload = json.dumps(signal).encode()
    ttl = settings.cache_ttl_seconds
    try:
        await r.setex(_exact_key(text, tag), ttl, payload)
        _counter += 1
        entry_id = f"{tag}:{_counter}"
        vec = embed(text)
        await r.hset(_VECS_KEY, entry_id, vec.tobytes())
        await r.setex(_META_PREFIX + entry_id, ttl, payload)
        _append_mirror(entry_id, vec)
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis put failed (%s); continuing", exc)


async def get_or_score(text: str, ticker: str) -> dict:
    """Cache-aware sentiment: exact -> semantic -> ensemble-on-miss."""
    from app.sentiment import ensemble  # local import avoids import cycle

    tag = settings.groq_model
    cached = await get(text, tag)
    if cached is not None:
        return {**cached, "cached": True}
    signal = await ensemble.score(text, ticker)
    await put(text, tag, signal)
    return {**signal, "cached": False}
