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
    "raw_metrics", "resolved_metrics", "is_disqualified",
    "raw_recall", "raw_precision", "raw_f1",
    "recall", "precision", "f1", "raw_node_count", "resolved_node_count", "lookalike_preserved",
    "bcubed_scores", "b3_precision", "b3_recall", "b3_f1", "llm_calls",
]


def raw_metrics(raw_graph: Graph, library_key: str, ground_truth: GroundTruth) -> Metrics:
    """Baseline: score the library's key-grouping (design A.2), not an assumed `name`."""
    return metrics.score(raw_graph.nodes, metrics.library_clusters(raw_graph, library_key), ground_truth)


def resolved_metrics(raw_graph: Graph, clusters: list[Cluster], ground_truth: GroundTruth) -> Metrics:
    """Score the resolved clustering. The look-alike check is the `lookalike_preserved`
    metric (0/1), used to **disqualify** a run at assessment time (`is_disqualified`) —
    not a crash, so a resolver that over-merges (e.g. Splink on tiny data) still produces
    a comparable MLflow row that shows it failed."""
    return metrics.score(raw_graph.nodes, clusters, ground_truth)


def is_disqualified(resolved_metrics: Metrics) -> bool:
    """Assessment gate (design §8): a run that co-clustered a look-alike is disqualified
    regardless of F1. Use to filter the MLflow runs table; logged as a metric below."""
    return not resolved_metrics.lookalike_preserved


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


def bcubed_scores(raw_graph: Graph, clusters: list[Cluster], ground_truth: GroundTruth) -> tuple[float, float, float]:
    return metrics.bcubed(raw_graph.nodes, clusters, ground_truth)


def b3_precision(bcubed_scores: tuple) -> float:
    return bcubed_scores[0]


def b3_recall(bcubed_scores: tuple) -> float:
    return bcubed_scores[1]


def b3_f1(bcubed_scores: tuple) -> float:
    return bcubed_scores[2]


def llm_calls(pair_verdicts: list) -> int:
    """Cost metric: LLM verifier calls in the resolver — 0 for Splink, N for offline."""
    return len(pair_verdicts)
