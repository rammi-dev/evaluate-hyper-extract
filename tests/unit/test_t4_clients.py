"""T4 acceptance: client fail-fast gates (mocked) + real ping (integration)."""

from __future__ import annotations

import pytest
from langchain_core.embeddings import Embeddings

from eval_hyper_extract.clients_module import check_embedder, check_llm
from tests.fakes import DeterministicEmbeddings, FakeChat


class _EmptyEmbedder(Embeddings):
    def embed_documents(self, texts):
        return [[] for _ in texts]

    def embed_query(self, text):
        return []


class _NaNEmbedder(Embeddings):
    def embed_documents(self, texts):
        return [[float("nan")] for _ in texts]

    def embed_query(self, text):
        return [float("nan")]


def test_check_llm_passes_on_content() -> None:
    llm = FakeChat(content="pong")
    assert check_llm(llm) is llm


def test_check_llm_raises_on_empty_content() -> None:
    with pytest.raises(AssertionError, match="empty content"):
        check_llm(FakeChat(content=""))


def test_check_llm_raises_on_unreachable() -> None:
    with pytest.raises(RuntimeError, match="simulated LLM failure"):
        check_llm(FakeChat(raise_on_invoke=True))


def test_check_embedder_passes() -> None:
    emb = DeterministicEmbeddings()
    assert check_embedder(emb) is emb


def test_check_embedder_raises_on_empty() -> None:
    with pytest.raises(AssertionError, match="empty vector"):
        check_embedder(_EmptyEmbedder())


def test_check_embedder_raises_on_nan() -> None:
    with pytest.raises(AssertionError, match="NaN"):
        check_embedder(_NaNEmbedder())


@pytest.mark.integration
def test_real_clients_ping() -> None:
    from eval_hyper_extract.clients_module import build_embedder, build_llm, checked_embedder, checked_llm
    from eval_hyper_extract.config_module import Config

    cfg = Config(
        llm_model="google/gemini-2.5-flash",
        embed_model="BAAI/bge-small-en-v1.5",  # smaller download for CI
    )
    assert checked_llm(cfg) is not None
    vec = build_embedder(cfg).embed_query("probe")
    assert vec and len(vec) > 10
    assert checked_embedder(cfg) is not None
    _ = build_llm  # referenced for completeness
