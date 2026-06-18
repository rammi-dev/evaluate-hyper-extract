"""Deterministic test doubles — no network, no model download, reproducible.

`DeterministicEmbeddings` returns a feature-weighted unit vector per text, tuned so
the mini_graph fixture has the intended candidate structure at tau=0.55:
  - the four P-101 variants are mutually near (cosine ~0.95-1.0),
  - P-101 vs P-102 are a *candidate* pair (~0.73) — the lookalike the verifier must
    reject (embeddings propose, the LLM disposes),
  - pumps vs the tank fall below threshold (~0.48), so they are never proposed.

`FakeChat` is a minimal LangChain-ish chat stub for the client ping (T4). Pair
verification (T8/T12/T13) is injected at the resolver-helper level via a stub
`verify_fn`, not by emulating structured output here.
"""

from __future__ import annotations

import math

from langchain_core.embeddings import Embeddings

# (name, predicate over lowercased text, weight)
_FEATURES: list[tuple[str, object, float]] = [
    ("equipment", lambda t: True, 2.0),
    ("pump", lambda t: any(s in t for s in ("pump", "p-10", "p10", "p 10", "feedwater", "feed water")), 2.0),
    ("tank", lambda t: any(s in t for s in ("tank", "t-200", "t200", "storage")), 2.0),
    ("feedwater", lambda t: "feed" in t, 1.0),
    ("cooling", lambda t: "cool" in t, 1.0),
    ("tag101", lambda t: "101" in t, 1.0),
    ("tag102", lambda t: "102" in t, 1.0),
    ("tag200", lambda t: "200" in t, 1.0),
    ("unit2", lambda t: "unit 2" in t, 1.0),
    ("unit1", lambda t: "unit 1" in t, 1.0),
]


def _vec(text: str) -> list[float]:
    t = text.lower()
    raw = [w if pred(t) else 0.0 for _, pred, w in _FEATURES]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


class DeterministicEmbeddings(Embeddings):
    """Feature-based, normalized, fully deterministic embedder."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [_vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return _vec(text)


class FakeChat:
    """Minimal chat stub. `.invoke()` returns an object with `.content`.

    Set `raise_on_invoke=True` to simulate an unreachable endpoint / bad key, so the
    `checked_llm` fail-fast gate can be exercised without a network call.
    """

    def __init__(self, content: str = "ok", *, raise_on_invoke: bool = False) -> None:
        self._content = content
        self._raise = raise_on_invoke

    def invoke(self, _input: object) -> object:
        if self._raise:
            raise RuntimeError("simulated LLM failure")

        class _Resp:
            content = self._content

        return _Resp()
