"""Metrics — the Hamilton node layer (design §5).

Thin scalar nodes (auto-logged as MLflow metrics) over the pure logic in `metrics.py`.
`raw_metrics` scores the library's key-grouping; `resolved_metrics` carries the
lookalike hard gate. Logic helpers are re-exported for tests (excluded from the DAG).
"""

from __future__ import annotations

from eval_hyper_extract import metrics
from eval_hyper_extract.metrics import Metrics, library_clusters, score  # re-exports for tests
from eval_hyper_extract.schema import Cluster, Graph, GroundTruth

__all__ = [
    "Metrics", "library_clusters", "score",
    "raw_metrics", "resolved_metrics",
    "raw_recall", "raw_precision", "raw_f1",
    "recall", "precision", "f1", "raw_node_count", "resolved_node_count", "lookalike_preserved",
]


def raw_metrics(raw_graph: Graph, library_key: str, ground_truth: GroundTruth) -> Metrics:
    """Baseline: score the library's key-grouping (design A.2), not an assumed `name`."""
    return metrics.score(raw_graph.nodes, metrics.library_clusters(raw_graph, library_key), ground_truth)


def resolved_metrics(raw_graph: Graph, clusters: list[Cluster], ground_truth: GroundTruth) -> Metrics:
    """Score the resolved clustering; HARD GATE: no lookalike pair co-clustered."""
    m = metrics.score(raw_graph.nodes, clusters, ground_truth)
    assert m.lookalike_preserved, (
        "HARD GATE failed: a lookalike pair was co-clustered "
        f"(lookalike_pairs={ground_truth.lookalike_pairs})"
    )
    return m


def raw_recall(raw_metrics: Metrics) -> float:
    return raw_metrics.recall


def raw_precision(raw_metrics: Metrics) -> float:
    return raw_metrics.precision


def raw_f1(raw_metrics: Metrics) -> float:
    return raw_metrics.f1


def recall(resolved_metrics: Metrics) -> float:
    return resolved_metrics.recall


def precision(resolved_metrics: Metrics) -> float:
    return resolved_metrics.precision


def f1(resolved_metrics: Metrics) -> float:
    return resolved_metrics.f1


def raw_node_count(raw_graph: Graph) -> int:
    return len(raw_graph.nodes)


def resolved_node_count(clusters: list[Cluster]) -> int:
    return len(clusters)


def lookalike_preserved(resolved_metrics: Metrics) -> int:
    return int(resolved_metrics.lookalike_preserved)
