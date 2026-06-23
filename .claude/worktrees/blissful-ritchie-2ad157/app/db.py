"""Async database engine, session dependency, and idempotent schema bootstrap."""

from __future__ import annotations

import logging
import ssl as ssl_module
from pathlib import Path
from typing import AsyncIterator
from urllib.parse import parse_qsl, urlsplit, urlunsplit

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

logger = logging.getLogger(__name__)

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "postgres", "db"}


def _normalize_url(url: str) -> tuple[str, dict]:
    """Coerce to the asyncpg driver and build asyncpg-safe connect_args.

    asyncpg rejects libpq query params like ``sslmode``; instead SSL is passed
    as a connect arg. Remote hosts (e.g. Supabase) get TLS automatically.
    """
    if url.startswith("postgres://"):
        url = "postgresql+asyncpg://" + url[len("postgres://") :]
    elif url.startswith("postgresql://") and "+asyncpg" not in url:
        url = "postgresql+asyncpg://" + url[len("postgresql://") :]

    parts = urlsplit(url)
    raw_q = dict(parse_qsl(parts.query))
    sslmode = raw_q.pop("sslmode", None)
    raw_q.pop("channel_binding", None)
    clean = urlunsplit(
        (parts.scheme, parts.netloc, parts.path, "&".join(f"{k}={v}" for k, v in raw_q.items()), "")
    )

    connect_args: dict = {"statement_cache_size": 0}  # pgbouncer-safe
    host = (parts.hostname or "").lower()
    is_local = host in _LOCAL_HOSTS
    if (not is_local and sslmode != "disable") or (sslmode and sslmode not in {"disable"}):
        ctx = ssl_module.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl_module.CERT_NONE
        connect_args["ssl"] = ctx
    return clean, connect_args


_url, _connect_args = _normalize_url(settings.database_url)
engine = create_async_engine(_url, connect_args=_connect_args, pool_pre_ping=True, echo=False)
SessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_session() -> AsyncIterator[AsyncSession]:
    async with SessionLocal() as session:
        yield session


async def init_db() -> None:
    """Apply db/schema.sql (idempotent — uses CREATE ... IF NOT EXISTS)."""
    schema_path = Path(__file__).resolve().parent.parent / "db" / "schema.sql"
    if not schema_path.exists():
        logger.warning("schema.sql not found at %s; skipping init_db()", schema_path)
        return
    sql = schema_path.read_text(encoding="utf-8")
    async with engine.begin() as conn:
        for raw_stmt in sql.split(";"):
            lines = [ln for ln in raw_stmt.splitlines() if not ln.strip().startswith("--")]
            cleaned = "\n".join(lines).strip()
            if cleaned:
                await conn.exec_driver_sql(cleaned)
