"""Environment loading. Single source of truth for the OpenRouter secret.

The key lives in `.env` under the var name `OPEN_ROUTER_KEY` (valid identifier;
see docs/implementation.md A.7). `load_env()` is idempotent and safe to call at
the top of any entrypoint.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

OPENROUTER_KEY_VAR = "OPEN_ROUTER_KEY"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def load_env() -> None:
    """Load `.env` into the process environment (no-op if already loaded)."""
    load_dotenv()


def openrouter_api_key() -> str:
    """Return the OpenRouter API key, raising a clear error if it is absent."""
    load_env()
    key = os.environ.get(OPENROUTER_KEY_VAR, "").strip()
    if not key:
        raise RuntimeError(
            f"{OPENROUTER_KEY_VAR} is not set. Add it to .env, e.g. "
            f"`{OPENROUTER_KEY_VAR}=sk-or-...`."
        )
    return key
