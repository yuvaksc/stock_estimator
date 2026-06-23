"""Health and metrics endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.metrics import metrics

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@router.get("/metrics")
async def get_metrics(session: AsyncSession = Depends(get_session)) -> dict:
    db_ok = True
    try:
        await session.execute(text("select 1"))
    except Exception:  # noqa: BLE001 - report, don't crash
        db_ok = False
    return {"db": "ok" if db_ok else "down", **metrics.snapshot()}
