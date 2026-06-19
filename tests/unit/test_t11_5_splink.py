"""T11.5 acceptance: the Splink resolver flow — records, cluster mapping, partition.

Splink runs locally (DuckDB) with no LLM/network, so the end-to-end test is a plain
unit test. EM on 13 nodes is unstable, so it asserts *structural* properties (a valid
partition that the shared tail can consume), not exact membership.
"""

from __future__ import annotations

import pandas as pd

from eval_hyper_extract.report_module import resolved_graph
from eval_hyper_extract.resolve_splink import clusters_from_df, to_records
from eval_hyper_extract.resolve_splink_module import clusters, pair_verdicts, splink_records
from eval_hyper_extract.schema import Graph

ALL_IDS = ["P-101", "P 101", "P101", "the feedwater pump", "P-102", "T-200", "the storage tank"]


def test_splink_records_shape_and_uniqueness(mini_graph: Graph) -> None:
    df = splink_records(mini_graph)
    assert len(df) == len(mini_graph.nodes)
    assert df["unique_id"].is_unique
    assert {"unique_id", "name", "name_norm", "type", "description"} <= set(df.columns)
    assert to_records(mini_graph).iloc[0]["name_norm"] == "p-101"  # normalized


def test_clusters_from_df_groups_and_canonical(mini_graph: Graph) -> None:
    # cluster_id grouping → Cluster with the same canonical pick as offline
    df = pd.DataFrame(
        {
            "unique_id": ALL_IDS,
            "cluster_id": [1, 1, 1, 1, 2, 3, 3],  # A's 4, P-102 alone, T-200 + variant
        }
    )
    cl = clusters_from_df(df, mini_graph)
    assert len(cl) == 3
    assert {c.canonical_name for c in cl} == {"P-101", "P-102", "T-200"}
    covered = sorted(nid for c in cl for nid in c.node_ids)
    assert covered == sorted(n.id for n in mini_graph.nodes)  # total + disjoint


def test_clusters_node_partition_gate_and_shared_tail(mini_graph: Graph) -> None:
    df = pd.DataFrame({"unique_id": ALL_IDS, "cluster_id": [1, 1, 1, 1, 2, 3, 3]})
    cl = clusters(df, mini_graph)
    # the shared tail consumes splink clusters unchanged
    g = resolved_graph(mini_graph, cl)
    assert g.node_ids() == {"P-101", "P-102", "T-200"}
    assert all(e.source != e.target for e in g.edges)


def test_pair_verdicts_empty() -> None:
    assert pair_verdicts() == []  # no LLM verifier → llm_calls == 0


def test_splink_end_to_end_partitions(mini_graph: Graph) -> None:
    """Train Splink on the fixture and assert it yields a valid partition (structural)."""
    from eval_hyper_extract.resolve_splink import train_linker

    df = to_records(mini_graph)
    linker = train_linker(df)
    pred = linker.inference.predict(threshold_match_probability=0.9)
    cdf = linker.clustering.cluster_pairwise_predictions_at_threshold(
        pred, threshold_match_probability=0.9
    ).as_pandas_dataframe()
    cl = clusters_from_df(cdf, mini_graph)
    covered = sorted(nid for c in cl for nid in c.node_ids)
    assert covered == sorted(n.id for n in mini_graph.nodes)  # every node placed exactly once
