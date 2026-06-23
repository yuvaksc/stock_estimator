"""Seed the `stocks` table from the tickers present in the dataset.

Usage:  python scripts/seed_stocks.py
Reads ticker symbols from the filenames in RAW_PRICE_DIR (stock_data/price/raw),
falling back to the TICKERS env list if the directory is absent.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import SessionLocal, init_db  # noqa: E402
from app.models_db import Stock  # noqa: E402


async def main() -> None:
    await init_db()

    raw_dir = Path(settings.raw_price_dir)
    if raw_dir.exists():
        tickers = sorted(p.stem.upper() for p in raw_dir.glob("*.csv"))
    else:
        tickers = settings.tickers_list
    if not tickers:
        print("No tickers found to seed.")
        return

    async with SessionLocal() as session:
        existing = set((await session.execute(select(Stock.ticker))).scalars().all())
        added = 0
        for ticker in tickers:
            if ticker not in existing:
                session.add(Stock(ticker=ticker))
                added += 1
        await session.commit()

    print(f"✅ Seeded {added} new stock(s); {len(tickers)} ticker(s) discovered.")


if __name__ == "__main__":
    asyncio.run(main())
