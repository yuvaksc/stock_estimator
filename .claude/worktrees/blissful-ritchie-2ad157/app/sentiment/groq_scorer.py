"""Structured financial sentiment via Groq + Instructor (typed, validated JSON).

Instructor wraps the Groq client and enforces the `SentimentSignal` Pydantic
schema with automatic re-ask retries. Works for the Llama models (JSON mode);
point GROQ_MODEL at `openai/gpt-oss-20b` for native strict-schema decoding.
"""

from __future__ import annotations

import logging
import os

import instructor
from groq import AsyncGroq

from app.config import settings
from app.schemas import SentimentSignal

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a financial sentiment analyst. Classify the market sentiment of the "
    "text toward the given ticker. 'bullish' = expects the price to go up, "
    "'bearish' = expects it to go down, 'neutral' = no clear directional signal. "
    "confidence is your certainty from 0 to 1. impact_horizon_hours is how long "
    "(0-168) you expect the signal to stay relevant. Keep rationale under 200 chars."
)

_client: instructor.AsyncInstructor | None = None


def _get_client() -> instructor.AsyncInstructor:
    global _client
    if _client is None:
        api_key = settings.groq_api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is not set; cannot call Groq.")
        # Mode.JSON works across Groq models; gpt-oss also supports strict modes.
        _client = instructor.from_groq(AsyncGroq(api_key=api_key), mode=instructor.Mode.JSON)
    return _client


async def score_text(text: str, ticker: str) -> SentimentSignal:
    """Return a validated SentimentSignal for `text` regarding `ticker`."""
    client = _get_client()
    return await client.chat.completions.create(
        model=settings.groq_model,
        response_model=SentimentSignal,
        max_retries=2,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Ticker: {ticker}\nText: {text}"},
        ],
    )
