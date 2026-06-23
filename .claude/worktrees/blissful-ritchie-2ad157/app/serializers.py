"""ORM-row -> WebSocket/JSON event dicts (shared by routers and the worker)."""

from __future__ import annotations

from app.models_db import Post, Prediction, SentimentSignalRow, StockTick


def _f(v) -> float | None:
    return float(v) if v is not None else None


def tick_event(row: StockTick) -> dict:
    return {
        "id": row.id,
        "ticker": row.ticker,
        "price": _f(row.price),
        "volume": row.volume,
        "ts": row.ts.isoformat() if row.ts else None,
        "source": row.source,
    }


def sentiment_event(row: SentimentSignalRow) -> dict:
    return {
        "id": row.id,
        "ticker": row.ticker,
        "sentiment": row.sentiment,
        "confidence": _f(row.confidence),
        "impact_horizon_hours": row.impact_horizon_hours,
        "finbert_score": _f(row.finbert_score),
        "model": row.model,
        "rationale": row.rationale,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def post_event(row: Post) -> dict:
    return {
        "id": row.id,
        "ticker": row.ticker,
        "author": row.author,
        "platform": row.platform,
        "body": (row.body or "")[:280],
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def prediction_event(row: Prediction) -> dict:
    return {
        "id": row.id,
        "ticker": row.ticker,
        "predicted_price": _f(row.predicted_price),
        "horizon": row.horizon,
        "model_version": row.model_version,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
