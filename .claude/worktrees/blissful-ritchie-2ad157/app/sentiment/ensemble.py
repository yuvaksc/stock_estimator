"""Ensemble the Groq LLM signal with the local FinBERT signal.

Groq is the primary label (rich, context-aware); FinBERT is a deterministic
cross-check whose disagreement lowers confidence. If Groq is unavailable
(no key / rate-limited), we degrade gracefully to FinBERT-only — which is local
and unlimited.
"""

from __future__ import annotations

import asyncio
import logging

from app.config import settings
from app.metrics import metrics
from app.sentiment import finbert, groq_scorer

logger = logging.getLogger(__name__)


async def score(text: str, ticker: str) -> dict:
    """Return a dict ready to persist into `sentiment_signals`."""
    # FinBERT (local) — run in a thread so we don't block the event loop.
    fb: dict | None = None
    try:
        fb = await asyncio.to_thread(finbert.score_finbert, text)
    except Exception as exc:  # noqa: BLE001
        logger.warning("FinBERT scoring failed: %s", exc)

    # Groq (LLM) — optional; degrade to FinBERT on any failure.
    groq_sig = None
    try:
        groq_sig = await groq_scorer.score_text(text, ticker)
        metrics.incr("llm_calls")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Groq scoring unavailable (%s); using FinBERT only", exc)

    if groq_sig is not None:
        sentiment = groq_sig.sentiment
        confidence = float(groq_sig.confidence)
        horizon = int(groq_sig.impact_horizon_hours)
        rationale = groq_sig.rationale or ""
        model_name = f"ensemble:groq:{settings.groq_model}+finbert"
        if (
            fb
            and fb["sentiment"] != sentiment
            and "neutral" not in (fb["sentiment"], sentiment)
        ):
            confidence = round(confidence * 0.6, 4)  # opposing signals → less certain
            rationale = (rationale + " [finbert disagrees]")[:200]
    elif fb is not None:
        sentiment = fb["sentiment"]
        confidence = round(float(fb["confidence"]), 4)
        horizon = 24
        rationale = "FinBERT-only (LLM unavailable)"
        model_name = "finbert"
    else:
        sentiment, confidence, horizon = "neutral", 0.0, 0
        rationale, model_name = "scoring unavailable", "none"

    metrics.incr("signals_scored")
    metrics.incr(f"sentiment_{sentiment}")
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "impact_horizon_hours": horizon,
        "rationale": rationale,
        "model": model_name,
        "finbert_score": round(float(fb["finbert_score"]), 6) if fb else None,
    }
