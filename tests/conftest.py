"""Shared test fixtures: the synthetic graph + manifest, and deterministic doubles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_hyper_extract.schema import Graph, GroundTruth
from tests.fakes import DeterministicEmbeddings, FakeChat

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def mini_graph() -> Graph:
    """Hand-built fragmented graph: 7 raw nodes for 3 real entities (A=4, B=1, C=2)."""
    data = json.loads((FIXTURES / "mini_graph.json").read_text())
    return Graph.model_validate(data)


@pytest.fixture
def ground_truth() -> GroundTruth:
    """Manifest matching mini_graph; lookalike pair (A=P-101, B=P-102)."""
    data = json.loads((FIXTURES / "mini_entities.json").read_text())
    return GroundTruth.model_validate(data)


@pytest.fixture
def fake_embedder() -> DeterministicEmbeddings:
    return DeterministicEmbeddings()


@pytest.fixture
def fake_llm() -> FakeChat:
    return FakeChat()
