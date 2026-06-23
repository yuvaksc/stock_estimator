"""Shared FastAPI dependencies (auth)."""

from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import settings


async def require_token(
    authorization: str | None = Header(default=None),
    x_api_token: str | None = Header(default=None),
) -> None:
    """Guard write/ingest endpoints. No-op when API_AUTH_TOKEN is unset (dev)."""
    if not settings.api_auth_token:
        return
    supplied = None
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
    supplied = supplied or x_api_token
    if supplied != settings.api_auth_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid or missing API token"
        )


def token_is_valid(token: str | None) -> bool:
    """Used by the WebSocket handshake (token passed as a query param)."""
    if not settings.api_auth_token:
        return True
    return token == settings.api_auth_token
