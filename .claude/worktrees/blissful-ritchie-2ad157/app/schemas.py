"""Pydantic I/O models shared across the API and the sentiment/ML layers."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Sentiment = Literal["bullish", "bearish", "neutral"]


# ── LLM structured output (Groq + Instructor) ───────────────────────────────
class SentimentSignal(BaseModel):
    """Schema the LLM is forced to return; also the unit the ensemble produces."""

    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)
    impact_horizon_hours: int = Field(ge=0, le=168)
    rationale: str = Field(default="", max_length=200)


# ── Stocks ──────────────────────────────────────────────────────────────────
class StockIn(BaseModel):
    ticker: str
    company_name: Optional[str] = None


class StockUpdate(BaseModel):
    company_name: Optional[str] = None


class StockOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    ticker: str
    company_name: Optional[str] = None
    created_at: Optional[datetime] = None


# ── Ticks ───────────────────────────────────────────────────────────────────
class TickIn(BaseModel):
    ticker: str
    price: float
    volume: Optional[int] = None
    ts: datetime
    source: str = "finnhub"


class TickOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ticker: str
    price: float
    volume: Optional[int] = None
    ts: datetime
    source: Optional[str] = None


# ── Posts ───────────────────────────────────────────────────────────────────
class PostIn(BaseModel):
    ticker: str
    platform: str = "reddit"
    external_id: Optional[str] = None
    author: Optional[str] = None
    body: str
    created_at: Optional[datetime] = None


class PostOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    ticker: str
    platform: Optional[str] = None
    external_id: Optional[str] = None
    author: Optional[str] = None
    body: str
    created_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None


# ── Sentiment signals (DB rows) ─────────────────────────────────────────────
class SentimentSignalOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: int
    post_id: Optional[int] = None
    ticker: str
    sentiment: Sentiment
    confidence: float
    impact_horizon_hours: Optional[int] = None
    model: Optional[str] = None
    finbert_score: Optional[float] = None
    rationale: Optional[str] = None
    created_at: Optional[datetime] = None


class ScoreRequest(BaseModel):
    """Manual scoring request for POST /sentiment."""

    ticker: str
    text: str
    persist: bool = True
    post_id: Optional[int] = None


# ── Predictions ─────────────────────────────────────────────────────────────
class PredictionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())
    id: int
    ticker: str
    predicted_price: float
    horizon: Optional[str] = None
    model_version: Optional[str] = None
    features_hash: Optional[str] = None
    created_at: Optional[datetime] = None


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    ticker: str
    model_version: str
    latest_predicted_price: float
    dates: list[str]
    predicted: list[float]
    actual: list[float]
