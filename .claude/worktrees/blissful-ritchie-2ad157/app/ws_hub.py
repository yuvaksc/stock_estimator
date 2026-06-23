"""WebSocket fan-out hub + Redis pub/sub bridge.

Design:
- Each connected client gets an ``asyncio.Queue(maxsize=100)``; the hub fans a
  message into every queue with **drop-oldest** backpressure (a slow client
  never blocks the others).
- Producers (the worker, REST ingest endpoints) call ``publish_event`` which
  publishes to a Redis channel. The API process runs ``start_bridge`` which
  subscribes to that channel and feeds the local hub — so events cross process
  boundaries (worker -> API). If Redis is unavailable, ``publish_event`` falls
  back to a direct local fan-out so an in-process worker still drives the UI.
"""

from __future__ import annotations

import asyncio
import json
import logging

from app.config import settings

logger = logging.getLogger(__name__)

_CHANNEL = "ws:events"


class Hub:
    def __init__(self) -> None:
        self._subs: set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)

    @property
    def n_clients(self) -> int:
        return len(self._subs)

    def fanout(self, msg: dict) -> None:
        for q in list(self._subs):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                try:  # drop-oldest, then enqueue the newest
                    q.get_nowait()
                    q.put_nowait(msg)
                except Exception:  # noqa: BLE001
                    pass


hub = Hub()

_pub_redis = None
_bridge_task: asyncio.Task | None = None


async def _get_pub():
    global _pub_redis
    if _pub_redis is None:
        import redis.asyncio as aioredis

        _pub_redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    return _pub_redis


async def publish_event(channel: str, ticker: str, payload: dict) -> None:
    """Publish a dashboard event. channel ∈ {ticks, sentiment, post, prediction}."""
    msg = {"ch": channel, "ticker": (ticker or "").upper(), "v": payload}
    try:
        r = await _get_pub()
        await r.publish(_CHANNEL, json.dumps(msg, default=str))
    except Exception as exc:  # noqa: BLE001
        logger.warning("redis publish failed (%s); local fan-out fallback", exc)
        hub.fanout(msg)


async def _bridge_loop() -> None:
    import redis.asyncio as aioredis

    while True:
        try:
            r = aioredis.from_url(settings.redis_url, decode_responses=True)
            pubsub = r.pubsub()
            await pubsub.subscribe(_CHANNEL)
            logger.info("WS Redis bridge subscribed to %s", _CHANNEL)
            async for message in pubsub.listen():
                if message.get("type") != "message":
                    continue
                try:
                    hub.fanout(json.loads(message["data"]))
                except Exception:  # noqa: BLE001
                    pass
        except asyncio.CancelledError:
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("WS Redis bridge error (%s); retrying in 3s", exc)
            await asyncio.sleep(3)


async def start_bridge() -> None:
    global _bridge_task
    if _bridge_task is None:
        _bridge_task = asyncio.create_task(_bridge_loop())


async def stop_bridge() -> None:
    global _bridge_task
    if _bridge_task is not None:
        _bridge_task.cancel()
        _bridge_task = None
