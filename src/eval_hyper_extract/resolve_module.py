"""External entity resolution — the Hamilton node layer (offline mode).

Thin wrappers over `resolve.py` (pure logic). Online/hybrid clusterers join in
T12/T13 under the same `clusters` node via `@config.when`. Pure helpers are
re-exported for tests; Hamilton ignores them (defined in `resolve`, not here).
"""

from __future__ import annotations

import numpy as np
from hamilton.function_modifiers import config
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from eval_hyper_extract import resolve
from eval_hyper_extract.resolve import (  # re-exports for tests (excluded from the DAG)
    ConfirmedPair,
    NodeEmbeddings,
    Pair,
    PairVerdict,
    Verdict,
    build_clusters,
    confirm_pairs,
    connected_components,
    embed_nodes,
    rewrite_graph,
    verify_all_pairs,
    verify_pair,
)
from eval_hyper_extract.schema import Cluster, Graph

__all__ = [
    "ConfirmedPair", "NodeEmbeddings", "Pair", "PairVerdict", "Verdict",
    "build_clusters", "confirm_pairs", "connected_components", "embed_nodes",
    "rewrite_graph", "verify_all_pairs", "verify_pair",
    "node_embeddings", "candidate_pairs", "pair_verdicts", "verified_pairs",
    "clusters__offline", "resolved_graph",
]


def node_embeddings(raw_graph: Graph, checked_embedder: Embeddings) -> NodeEmbeddings:
    ne = resolve.embed_nodes(raw_graph.nodes, checked_embedder)
    assert ne.vectors.shape[0] == len(raw_graph.nodes), "embedding row count mismatch"
    assert not np.isnan(ne.vectors).any(), "NaN in embeddings"
    return ne


def candidate_pairs(node_embeddings: NodeEmbeddings, tau_candidate: float) -> list[Pair]:
    return resolve.candidate_pairs(node_embeddings, tau_candidate)


def pair_verdicts(candidate_pairs: list[Pair], raw_graph: Graph, checked_llm: BaseChatModel) -> list[PairVerdict]:
    """LLM verdict for EVERY candidate pair (same and different) — the audit trail.

    This is where the verifier LLM is called (once per pair); `verified_pairs` and the
    validation table both derive from it without re-calling the model.
    """
    nodes_by_id = {n.id: n for n in raw_graph.nodes}
    return resolve.verify_all_pairs(candidate_pairs, nodes_by_id, lambda a, b: resolve.verify_pair(checked_llm, a, b))


def verified_pairs(pair_verdicts: list[PairVerdict]) -> list[ConfirmedPair]:
    """Confirmed-same pairs only (pure filter over the verdicts — no LLM call)."""
    return [ConfirmedPair(v.a, v.b, v.reason) for v in pair_verdicts if v.same]


@config.when(resolution_mode="offline")
def clusters__offline(raw_graph: Graph, verified_pairs: list[ConfirmedPair]) -> list[Cluster]:
    """Offline (batch): connected components over all confirmed-same pairs.

    `@config.when(resolution_mode="offline")` makes this the `clusters` node when the
    driver is built with `resolution_mode="offline"`. Online/hybrid join it in T12/T13.
    """
    clusters = resolve.build_clusters(raw_graph.nodes, verified_pairs)
    covered = [nid for c in clusters for nid in c.node_ids]
    assert sorted(covered) == sorted(n.id for n in raw_graph.nodes), "clusters must partition all nodes once"
    return clusters


def resolved_graph(raw_graph: Graph, clusters: list[Cluster]) -> Graph:
    g = resolve.rewrite_graph(raw_graph, clusters)
    ids = g.node_ids()
    assert all(e.source != e.target for e in g.edges), "self-edge survived"
    assert all(e.source in ids and e.target in ids for e in g.edges), "dangling edge endpoint"
    return g
