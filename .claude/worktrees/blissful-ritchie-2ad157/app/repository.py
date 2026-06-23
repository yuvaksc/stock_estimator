"""Thin data-access helpers shared by the routers and the consumer worker."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models_db import Post, Prediction, SentimentSignalRow, Stock, StockTick


async def ensure_stock(session: AsyncSession, ticker: str) -> str:
    ticker = ticker.upper()
    if await session.get(Stock, ticker) is None:
        session.add(Stock(ticker=ticker))
        await session.flush()  # satisfy FK within the same transaction
    return ticker


async def insert_tick(
    session: AsyncSession,
    *,
    ticker: str,
    price: float,
    ts: datetime,
    volume: int | None = None,
    source: str = "finnhub",
) -> StockTick:
    await ensure_stock(session, ticker)
    row = StockTick(ticker=ticker.upper(), price=price, volume=volume, ts=ts, source=source)
    session.add(row)
    await session.flush()
    return row


async def insert_post(
    session: AsyncSession,
    *,
    ticker: str,
    body: str,
    platform: str = "reddit",
    external_id: str | None = None,
    author: str | None = None,
    created_at: datetime | None = None,
) -> Post | None:
    """Insert a post, skipping duplicates by external_id. Returns None if skipped."""
    await ensure_stock(session, ticker)
    if external_id:
        exists = await session.scalar(select(Post.id).where(Post.external_id == external_id))
        if exists:
            return None
    row = Post(
        ticker=ticker.upper(),
        body=body,
        platform=platform,
        external_id=external_id,
        author=author,
        created_at=created_at,
    )
    session.add(row)
    await session.flush()
    return row


async def insert_sentiment(
    session: AsyncSession, *, ticker: str, signal: dict, post_id: int | None = None
) -> SentimentSignalRow:
    await ensure_stock(session, ticker)
    row = SentimentSignalRow(
        ticker=ticker.upper(),
        post_id=post_id,
        sentiment=signal["sentiment"],
        confidence=signal["confidence"],
        impact_horizon_hours=signal.get("impact_horizon_hours"),
        model=signal.get("model"),
        finbert_score=signal.get("finbert_score"),
        rationale=signal.get("rationale"),
    )
    session.add(row)
    await session.flush()
    return row


async def insert_prediction(
    session: AsyncSession,
    *,
    ticker: str,
    predicted_price: float,
    horizon: str | None = None,
    model_version: str | None = None,
    features_hash: str | None = None,
) -> Prediction:
    await ensure_stock(session, ticker)
    row = Prediction(
        ticker=ticker.upper(),
        predicted_price=predicted_price,
        horizon=horizon,
        model_version=model_version,
        features_hash=features_hash,
    )
    session.add(row)
    await session.flush()
    return row


async def latest_ticks(
    session: AsyncSession, ticker: str, limit: int = 100
) -> Sequence[StockTick]:
    res = await session.execute(
        select(StockTick)
        .where(StockTick.ticker == ticker.upper())
        .order_by(StockTick.ts.desc())
        .limit(limit)
    )
    return list(reversed(res.scalars().all()))  # chronological for charts


async def latest_sentiment(
    session: AsyncSession,
    ticker: str | None = None,
    limit: int = 50,
    since: datetime | None = None,
) -> Sequence[SentimentSignalRow]:
    stmt = select(SentimentSignalRow)
    if ticker:
        stmt = stmt.where(SentimentSignalRow.ticker == ticker.upper())
    if since:
        stmt = stmt.where(SentimentSignalRow.created_at >= since)
    stmt = stmt.order_by(SentimentSignalRow.created_at.desc()).limit(limit)
    res = await session.execute(stmt)
    return res.scalars().all()
