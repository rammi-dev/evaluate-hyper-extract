"""Clients — the Hamilton node layer (design §9).

Builds prebuilt LangChain objects and passes them downstream; never calls ontomem's
`create_client`. Pure construction/probe logic lives in `clients.py` (re-exported for
tests, excluded from the DAG).
"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from eval_hyper_extract import clients
from eval_hyper_extract.clients import build_embedder, build_llm, check_embedder, check_llm  # re-exports
from eval_hyper_extract.config_module import Config

__all__ = ["build_embedder", "build_llm", "check_embedder", "check_llm", "checked_llm", "checked_embedder"]


def checked_llm(config: Config) -> BaseChatModel:
    return clients.check_llm(clients.build_llm(config))


def checked_embedder(config: Config) -> Embeddings:
    return clients.check_embedder(clients.build_embedder(config))
