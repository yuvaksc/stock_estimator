"""CRUD endpoints for the stocks reference table."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_session
from app.deps import require_token
from app.models_db import Stock
from app.schemas import StockIn, StockOut, StockUpdate

router = APIRouter(prefix="/stocks", tags=["stocks"])


@router.get("", response_model=list[StockOut])
async def list_stocks(session: AsyncSession = Depends(get_session)):
    res = await session.execute(select(Stock).order_by(Stock.ticker))
    return res.scalars().all()


@router.get("/{ticker}", response_model=StockOut)
async def get_stock(ticker: str, session: AsyncSession = Depends(get_session)):
    obj = await session.get(Stock, ticker.upper())
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "stock not found")
    return obj


@router.post("", response_model=StockOut, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_token)])
async def create_stock(payload: StockIn, session: AsyncSession = Depends(get_session)):
    ticker = payload.ticker.upper()
    if await session.get(Stock, ticker) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, "stock already exists")
    obj = Stock(ticker=ticker, company_name=payload.company_name)
    session.add(obj)
    await session.commit()
    await session.refresh(obj)
    return obj


@router.put("/{ticker}", response_model=StockOut, dependencies=[Depends(require_token)])
async def update_stock(
    ticker: str, payload: StockUpdate, session: AsyncSession = Depends(get_session)
):
    obj = await session.get(Stock, ticker.upper())
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "stock not found")
    if payload.company_name is not None:
        obj.company_name = payload.company_name
    await session.commit()
    await session.refresh(obj)
    return obj


@router.delete("/{ticker}", status_code=status.HTTP_204_NO_CONTENT,
               dependencies=[Depends(require_token)])
async def delete_stock(ticker: str, session: AsyncSession = Depends(get_session)):
    obj = await session.get(Stock, ticker.upper())
    if obj is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "stock not found")
    await session.delete(obj)
    await session.commit()
