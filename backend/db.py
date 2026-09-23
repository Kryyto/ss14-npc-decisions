"""Supabase prediction logging.

The `supabase` package is imported lazily inside `get_client` so this module can
be imported (and the app tested) without the dependency installed.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

_TRUE = {"1", "true", "yes", "on"}

_client = None


def logging_enabled() -> bool:
    """LOGGING_REQUIRED defaults to true; set it to false only for local dev/tests."""
    return os.environ.get("LOGGING_REQUIRED", "true").strip().lower() in _TRUE


def get_client():
    """One process-wide Supabase client, built on first use."""
    global _client
    if _client is None:
        from supabase import create_client

        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set when "
                "LOGGING_REQUIRED is true"
            )
        _client = create_client(url, key)
    return _client


def log_prediction(
    *,
    id: str,
    job: str,
    lang: str,
    message: str,
    answers: Dict[str, Any],
    latency_ms: Optional[int],
    error: Optional[str],
) -> None:
    """Insert one row into `predictions`; the UUID is supplied by the caller."""
    get_client().table("predictions").insert(
        {
            "id": id,
            "job": job,
            "lang": lang,
            "message": message,
            "answers": answers,
            "latency_ms": latency_ms,
            "error": error,
        }
    ).execute()
