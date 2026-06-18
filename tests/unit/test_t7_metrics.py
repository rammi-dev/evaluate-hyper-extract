"""T7 acceptance: metrics are exact and the lookalike hard gate fires.

All numbers are computed by hand against the mini_graph fixture:
  A = {P-101, P 101, P101, the feedwater pump}  (4 nodes)
  B = {P-102}                                    (1 node, lookalike of A)
  C = {T-200, the storage tank}                  (2 nodes)
  should-link pairs = C(4,2)+C(2,2)+C(1,2) = 6+1+0 = 7
"""

from __future__ import annotations

import pytest

from eval_hyper_extract.metrics_module import (
    library_clusters,
    raw_metrics,
    raw_node_count,
    resolved_metrics,
    resolved_node_count,
    score,
)
from eval_hyper_extract.schema import Cluster, Graph, GroundTruth

A = ["P-101", "P 101", "P101", "the feedwater pump"]
B = ["P-102"]
C = ["T-200", "the storage tank"]


def _clusters(*groups: list[str]) -> list[Cluster]:
    return [Cluster(id=f"c{i}", node_ids=list(g)) for i, g in enumerate(groups)]


def test_perfect_clustering(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    m = score(mini_graph.nodes, _clusters(A, B, C), ground_truth)
    assert m.recall == 1.0
    assert m.precision == 1.0
    assert m.f1 == 1.0
    assert m.lookalike_preserved is True
    assert m.n_clusters == 3


def test_total_fragmentation(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    singletons = _clusters(*([nid] for nid in (A + B + C)))
    m = score(mini_graph.nodes, singletons, ground_truth)
    assert m.recall == 0.0  # no variant reunified
    assert m.precision == 1.0  # but no false merges either
    assert m.lookalike_preserved is True
    assert m.n_clusters == 7


def test_over_merged_lookalike(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    m = score(mini_graph.nodes, _clusters(A + B, C), ground_truth)
    assert m.recall == 1.0  # all true links still present
    assert m.precision == pytest.approx(7 / 11)  # 7 true of 11 predicted pairs
    assert m.f1 == pytest.approx(2 * (7 / 11) * 1.0 / ((7 / 11) + 1.0))
    assert m.lookalike_preserved is False  # A and B share a cluster


def test_under_merge(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    # one A variant split off into its own cluster
    m = score(mini_graph.nodes, _clusters(["P-101", "P 101", "P101"], ["the feedwater pump"], B, C), ground_truth)
    assert m.recall == pytest.approx(4 / 7)  # 3 A-pairs + 1 C-pair of 7 should-links
    assert m.precision == 1.0  # no false merges
    assert m.lookalike_preserved is True


def test_raw_metrics_keyed_by_library_key(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    by_name = raw_metrics(mini_graph, "name", ground_truth)
    by_type = raw_metrics(mini_graph, "type", ground_truth)

    # key=name: every node unique → fully fragmented baseline
    assert by_name.n_clusters == 7
    assert by_name.recall == 0.0
    assert by_name.precision == 1.0
    assert by_name.lookalike_preserved is True

    # key=type: pumps collapse together → over-merged baseline (incl. the lookalike)
    assert by_type.n_clusters == 2
    assert by_type.recall == 1.0
    assert by_type.precision == pytest.approx(7 / 11)
    assert by_type.lookalike_preserved is False

    # swapping the key changes the baseline fragmentation
    assert by_name.n_clusters != by_type.n_clusters


def test_library_clusters_groups_by_key(mini_graph: Graph) -> None:
    assert len(library_clusters(mini_graph, "name")) == 7
    assert len(library_clusters(mini_graph, "type")) == 2  # pump, tank
    assert len(library_clusters(mini_graph, "scope")) == 2  # Unit 2, Unit 1
    assert len(library_clusters(mini_graph, "{type}|{scope}")) == 3  # pump|U2, pump|U1, tank|U2


def test_resolved_metrics_passes_clean(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    m = resolved_metrics(mini_graph, _clusters(A, B, C), ground_truth)
    assert m.recall == 1.0 and m.lookalike_preserved is True


def test_resolved_metrics_hard_gate_raises(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    with pytest.raises(AssertionError, match="HARD GATE"):
        resolved_metrics(mini_graph, _clusters(A + B, C), ground_truth)


def test_scalar_counts(mini_graph: Graph) -> None:
    assert raw_node_count(mini_graph) == 7
    assert resolved_node_count(_clusters(A, B, C)) == 3
