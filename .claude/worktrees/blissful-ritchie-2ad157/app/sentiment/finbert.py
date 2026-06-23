"""Local FinBERT (ProsusAI/finbert) scorer — a deterministic second opinion.

Runs on CPU, no API/network. Returns a bullish/bearish/neutral label plus a
signed `finbert_score = P(positive) - P(negative)`.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import torch

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    logger.info("loading FinBERT model: %s", settings.finbert_model)
    tokenizer = AutoTokenizer.from_pretrained(settings.finbert_model)
    model = AutoModelForSequenceClassification.from_pretrained(settings.finbert_model)
    model.eval()
    return tokenizer, model


def score_finbert(text: str) -> dict:
    tokenizer, model = _load()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()

    # Use the model's own label map (ProsusAI/finbert: positive/negative/neutral).
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    p = {id2label[i]: float(probs[i]) for i in range(len(probs))}
    pos, neg, neu = p.get("positive", 0.0), p.get("negative", 0.0), p.get("neutral", 0.0)

    if pos >= neg and pos >= neu:
        label = "bullish"
    elif neg >= pos and neg >= neu:
        label = "bearish"
    else:
        label = "neutral"

    return {
        "sentiment": label,
        "confidence": max(pos, neg, neu),
        "finbert_score": pos - neg,
        "probs": p,
    }
