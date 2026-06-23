"""FastAPI application entrypoint (app factory + lifespan)."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.config import settings
from app.db import engine, init_db
from app.routers import health, stocks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── startup ──
    try:
        await init_db()
        logger.info("database schema ensured")
    except Exception as exc:  # noqa: BLE001
        logger.warning("init_db failed (apply db/schema.sql manually?): %s", exc)
    yield
    # ── shutdown ──
    await engine.dispose()


def create_app() -> FastAPI:
    app = FastAPI(
        title="Real-Time Stock-Sentiment AI System",
        version=__version__,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers (more are added as the system grows: ticks, posts, sentiment,
    # predictions, ws).
    app.include_router(health.router)
    app.include_router(stocks.router)
    return app


app = create_app()
