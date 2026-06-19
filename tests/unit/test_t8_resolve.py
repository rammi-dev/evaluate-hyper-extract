"""T8 acceptance: the offline resolver — deterministic, stub-verified, exact.

Uses an *oracle* verifier (ground-truth-backed) so the LLM is never called: it
confirms true variants and rejects the P-101/P-102 lookalike.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval_hyper_extract.metrics_module import score
from eval_hyper_extract.resolve_module import (
    Verdict,
    connected_components,
    build_clusters,
    candidate_pairs,
    clusters as cluster_offline,
    confirm_pairs,
    embed_nodes,
    node_embeddings,
    rewrite_graph,
)
from eval_hyper_extract.report_module import resolved_graph
from eval_hyper_extract.schema import Cluster, Graph, GroundTruth


def _oracle(ground_truth: GroundTruth):
    name_to_entity = {v: e.id for e in ground_truth.canonical_entities for v in e.variants}

    def verify(a, b) -> Verdict:
        ea, eb = name_to_entity.get(a.name), name_to_entity.get(b.name)
        return Verdict(same=(ea is not None and ea == eb), reason="oracle")

    return verify


def _pairset(pairs) -> set[frozenset[str]]:
    return {frozenset((p.a, p.b)) for p in pairs}


# ------------------------------------------------------------------- shared signal

def test_embed_nodes_normalized(mini_graph: Graph, fake_embedder) -> None:
    ne = embed_nodes(mini_graph.nodes, fake_embedder)
    assert ne.vectors.shape[0] == len(mini_graph.nodes)
    assert not np.isnan(ne.vectors).any()
    assert np.allclose(np.linalg.norm(ne.vectors, axis=1), 1.0)


def test_candidate_pairs_threshold(mini_graph: Graph, fake_embedder) -> None:
    ne = embed_nodes(mini_graph.nodes, fake_embedder)
    c55 = candidate_pairs(ne, 0.55)
    ids = ne.ids
    assert all(p.a in ids and p.b in ids for p in c55)
    assert all(-1.0 <= p.sim <= 1.0 for p in c55)

    pairs = _pairset(c55)
    assert frozenset(("P-101", "P-102")) in pairs  # lookalike IS a candidate
    assert frozenset(("P-101", "T-200")) not in pairs  # pump/tank below threshold

    # raising tau strictly shrinks the candidate set
    assert len(candidate_pairs(ne, 0.99)) < len(c55)


def test_confirm_pairs_keeps_same_only(mini_graph: Graph, fake_embedder, ground_truth: GroundTruth) -> None:
    ne = embed_nodes(mini_graph.nodes, fake_embedder)
    cands = candidate_pairs(ne, 0.55)
    by_id = {n.id: n for n in mini_graph.nodes}
    confirmed = confirm_pairs(cands, by_id, _oracle(ground_truth))

    name_to_entity = {v: e.id for e in ground_truth.canonical_entities for v in e.variants}
    for cp in confirmed:
        assert name_to_entity[by_id[cp.a].name] == name_to_entity[by_id[cp.b].name]
    assert frozenset(("P-101", "P-102")) not in _pairset(confirmed)  # lookalike rejected
    assert _pairset(confirmed) <= _pairset(cands)  # subset of candidates


# ----------------------------------------------------------------------- clustering

def test_connected_components_transitive() -> None:
    comps = connected_components(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    sets = sorted((sorted(g) for g in comps), key=len, reverse=True)
    assert sets[0] == ["a", "b", "c"]  # transitive chain merged
    assert ["d"] in [sorted(g) for g in comps]  # singleton kept


def test_build_clusters_canonical(mini_graph: Graph, fake_embedder, ground_truth: GroundTruth) -> None:
    ne = embed_nodes(mini_graph.nodes, fake_embedder)
    confirmed = confirm_pairs(candidate_pairs(ne, 0.55), {n.id: n for n in mini_graph.nodes}, _oracle(ground_truth))
    clusters = build_clusters(mini_graph.nodes, confirmed)
    assert len(clusters) == 3
    assert {c.canonical_name for c in clusters} == {"P-101", "P-102", "T-200"}
    covered = sorted(nid for c in clusters for nid in c.node_ids)
    assert covered == sorted(n.id for n in mini_graph.nodes)  # partition


def test_rewrite_graph_redirect_dedupe_selfdrop(mini_graph: Graph) -> None:
    clusters = [
        Cluster(id="P-101", canonical_name="P-101", node_ids=["P-101", "P 101", "P101", "the feedwater pump"]),
        Cluster(id="P-102", canonical_name="P-102", node_ids=["P-102"]),
        Cluster(id="T-200", canonical_name="T-200", node_ids=["T-200", "the storage tank"]),
    ]
    g = rewrite_graph(mini_graph, clusters)
    assert g.node_ids() == {"P-101", "P-102", "T-200"}
    # e1 (P-101->T-200) and e2 (variants->variants) collapse to ONE; e3 self-edge dropped; e4 kept
    edge_sigs = {(e.source, e.target, e.relation) for e in g.edges}
    assert edge_sigs == {("P-101", "T-200", "feeds"), ("P-102", "T-200", "feeds")}
    assert all(e.source != e.target for e in g.edges)


# ------------------------------------------------------------------- end-to-end + gates

def test_offline_end_to_end(mini_graph: Graph, fake_embedder, ground_truth: GroundTruth) -> None:
    ne = node_embeddings(mini_graph, fake_embedder)
    cands = candidate_pairs(ne, 0.55)
    confirmed = confirm_pairs(cands, {n.id: n for n in mini_graph.nodes}, _oracle(ground_truth))
    clusters = cluster_offline(mini_graph, confirmed)
    g = resolved_graph(mini_graph, clusters)

    assert len(g.nodes) == 3  # the collapse: 7 -> 3
    m = score(mini_graph.nodes, clusters, ground_truth)
    assert m.recall == 1.0 and m.precision == 1.0 and m.lookalike_preserved is True


def test_clusters_offline_partition_gate(mini_graph: Graph) -> None:
    # no confirmed pairs → all singletons, still a valid partition
    clusters = cluster_offline(mini_graph, [])
    assert len(clusters) == len(mini_graph.nodes)


# ----------------------------------------------------------------------- integration

@pytest.mark.integration
def test_real_verifier_distinguishes_tags() -> None:
    """Real LLM: P-101 vs P-102 → different; P-101 vs P 101 → same."""
    import os

    from langchain_openai import ChatOpenAI

    from eval_hyper_extract.env import OPENROUTER_BASE_URL, openrouter_api_key
    from eval_hyper_extract.resolve_module import verify_pair
    from eval_hyper_extract.schema import Node

    llm = ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "google/gemini-2.5-flash"),
        base_url=OPENROUTER_BASE_URL,
        api_key=openrouter_api_key(),
        temperature=0,
    )
    p101 = Node(id="1", name="P-101", type="pump", scope="Unit 2", description="feedwater pump")
    p101b = Node(id="2", name="P 101", type="pump", scope="Unit 2", description="feed water pump")
    p102 = Node(id="3", name="P-102", type="pump", scope="Unit 1", description="cooling pump")

    assert verify_pair(llm, p101, p101b).same is True
    assert verify_pair(llm, p101, p102).same is False
