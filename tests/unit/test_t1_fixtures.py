"""T1 acceptance: synthetic graph + manifest are valid, fragmented, and usable.

These fixtures are the ground for the deterministic core (T7/T8). If they aren't
genuinely fragmented and don't carry a lookalike pair, later metric tests are vacuous.
"""

from __future__ import annotations

import math

from eval_hyper_extract.schema import Graph, GroundTruth
from tests.fakes import DeterministicEmbeddings


def _cos(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (
        (math.sqrt(sum(x * x for x in a)) or 1.0) * (math.sqrt(sum(y * y for y in b)) or 1.0)
    )


def test_fixtures_load_and_validate(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    assert isinstance(mini_graph, Graph)
    assert isinstance(ground_truth, GroundTruth)


def test_manifest_has_variants_and_lookalike(ground_truth: GroundTruth) -> None:
    """Non-vacuous: >=1 canonical with >=2 variants, AND >=1 lookalike pair."""
    assert any(len(e.variants) >= 2 for e in ground_truth.canonical_entities)
    assert len(ground_truth.lookalike_pairs) >= 1


def test_graph_is_actually_fragmented(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    """Distinct raw nodes must exceed canonical entities — i.e. it *is* fragmented."""
    n_nodes = len(mini_graph.nodes)
    n_entities = len(ground_truth.canonical_entities)
    assert n_nodes > n_entities, f"{n_nodes} nodes but {n_entities} entities — not fragmented"
    assert n_nodes == 7 and n_entities == 3


def test_node_ids_unique_and_edges_reference_existing_nodes(mini_graph: Graph) -> None:
    """Raw graph: ids unique (key==name) and every edge endpoint resolves to a node."""
    ids = [n.id for n in mini_graph.nodes]
    assert len(ids) == len(set(ids))
    node_ids = mini_graph.node_ids()
    for e in mini_graph.edges:
        assert e.source in node_ids and e.target in node_ids


def test_every_variant_maps_to_a_node(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    """Each manifest variant corresponds to a raw node name (so recall is measurable)."""
    names = {n.name for n in mini_graph.nodes}
    for entity in ground_truth.canonical_entities:
        for variant in entity.variants:
            assert variant in names, f"variant {variant!r} missing from graph"


def _text(node) -> str:
    return f"{node.name} | {node.type} | {node.scope} | {node.description}"


def test_embedder_candidate_structure(mini_graph: Graph) -> None:
    """The deterministic embedder yields the intended candidate structure at tau=0.55."""
    emb = DeterministicEmbeddings()
    vecs = {n.id: emb.embed_query(_text(n)) for n in mini_graph.nodes}

    # P-101 variants cluster tightly.
    assert _cos(vecs["P-101"], vecs["P 101"]) >= 0.9
    assert _cos(vecs["P-101"], vecs["the feedwater pump"]) >= 0.55

    # Lookalike P-101 / P-102: a CANDIDATE (>=0.55) the verifier must later reject.
    assert _cos(vecs["P-101"], vecs["P-102"]) >= 0.55

    # Pump vs tank stay below threshold — never proposed.
    assert _cos(vecs["P-101"], vecs["T-200"]) < 0.55

    # Tank variants cluster.
    assert _cos(vecs["T-200"], vecs["the storage tank"]) >= 0.9
