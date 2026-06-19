"""Splink resolver — the Hamilton node layer (a peer flow to resolve_module).

Provides the `clusters` node from Fellegi-Sunter linkage, plus an empty `pair_verdicts`
(Splink has no LLM verifier) so the shared tail (final_report, verifier_agreement,
llm_calls=0) works unchanged. Run as its own flow:
    run(resolver_module=resolve_splink_module, resolution_mode="splink", match_probability_threshold=0.9)
"""

from __future__ import annotations

import pandas as pd

from eval_hyper_extract import resolve_splink
from eval_hyper_extract.resolve import PairVerdict  # re-export for tests
from eval_hyper_extract.schema import Cluster, Graph

__all__ = [
    "PairVerdict",
    "splink_records", "splink_linker", "splink_pairwise", "splink_clusters",
    "clusters", "pair_verdicts",
]


def splink_records(raw_graph: Graph) -> pd.DataFrame:
    df = resolve_splink.to_records(raw_graph)
    assert len(df) == len(raw_graph.nodes), "one Splink record per node"
    assert df["unique_id"].is_unique, "unique_id must be unique"
    return df


def splink_linker(splink_records: pd.DataFrame) -> object:
    """EM-trained dedupe Linker (weights illustrative on a tiny corpus — see module doc)."""
    return resolve_splink.train_linker(splink_records)


def splink_pairwise(splink_linker: object, match_probability_threshold: float) -> object:
    pred = splink_linker.inference.predict(threshold_match_probability=match_probability_threshold)
    assert "match_probability" in pred.as_pandas_dataframe().columns
    return pred


def splink_clusters(splink_linker: object, splink_pairwise: object, match_probability_threshold: float) -> pd.DataFrame:
    res = splink_linker.clustering.cluster_pairwise_predictions_at_threshold(
        splink_pairwise, threshold_match_probability=match_probability_threshold
    )
    df = res.as_pandas_dataframe()
    assert {"unique_id", "cluster_id"} <= set(df.columns)
    return df


def clusters(splink_clusters: pd.DataFrame, raw_graph: Graph) -> list[Cluster]:
    """Splink cluster_id grouping → `list[Cluster]` (consumed by the shared `resolved_graph`)."""
    cl = resolve_splink.clusters_from_df(splink_clusters, raw_graph)
    covered = sorted(nid for c in cl for nid in c.node_ids)
    assert covered == sorted(n.id for n in raw_graph.nodes), "splink clusters must partition all nodes once"
    return cl


def pair_verdicts() -> list[PairVerdict]:
    """No LLM verifier in the Splink flow → empty (so llm_calls=0 and the shared tail runs)."""
    return []
