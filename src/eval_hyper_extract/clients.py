"""Pure client construction + probe logic (NOT a Hamilton module).

Kept out of `clients_module` so the helpers don't become DAG nodes. The `check_*`
probes take a prebuilt client so the gate is unit-tested with fakes (no network).
"""

from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from eval_hyper_extract.config_module import Config
from eval_hyper_extract.env import OPENROUTER_BASE_URL, openrouter_api_key


def build_llm(config: Config) -> BaseChatModel:
    """Construct the OpenRouter chat client (no network yet)."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=config.llm_model,
        base_url=OPENROUTER_BASE_URL,
        api_key=openrouter_api_key(),
        temperature=0,
    )


def build_embedder(config: Config) -> Embeddings:
    """Construct the local embedder (downloads the model on first use)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.embed_model)


def check_llm(llm: BaseChatModel) -> BaseChatModel:
    """Probe the chat client; raise if it can't return content (the gate)."""
    resp = llm.invoke("ping")
    content = getattr(resp, "content", resp)
    if not content:
        raise AssertionError("LLM ping returned empty content — check model id / key")
    return llm


def check_embedder(embedder: Embeddings) -> Embeddings:
    """Probe the embedder; raise on empty / NaN vector (the gate)."""
    vec = embedder.embed_query("probe")
    if not vec:
        raise AssertionError("embedder returned an empty vector")
    if any(math.isnan(x) for x in vec):
        raise AssertionError("embedder returned NaN")
    return embedder
