"""T0 acceptance: project scaffold, dependencies importable, env loading works."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "hamilton",
        "mlflow",
        "langchain_openai",
        "langchain_huggingface",
        "hyperextract",
        "ontomem",
        "numpy",
        "dotenv",
    ],
)
def test_dependency_importable(module: str) -> None:
    """Every pipeline dependency imports — proves `uv sync` resolved the env."""
    assert importlib.import_module(module) is not None


def test_package_imports() -> None:
    """Our own package is installed and importable."""
    import eval_hyper_extract

    assert eval_hyper_extract.__version__


def test_env_loads_and_reports_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """`openrouter_api_key()` raises a clear error when the var is absent."""
    from eval_hyper_extract.env import OPENROUTER_KEY_VAR, openrouter_api_key

    monkeypatch.delenv(OPENROUTER_KEY_VAR, raising=False)
    # Prevent .env on disk from satisfying the key during this negative test.
    monkeypatch.setattr("eval_hyper_extract.env.load_env", lambda: None)
    with pytest.raises(RuntimeError, match=OPENROUTER_KEY_VAR):
        openrouter_api_key()


def test_env_returns_key_when_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the var is set, the key is returned (stripped)."""
    from eval_hyper_extract.env import OPENROUTER_KEY_VAR, openrouter_api_key

    monkeypatch.setenv(OPENROUTER_KEY_VAR, "  sk-or-test  ")
    assert openrouter_api_key() == "sk-or-test"
