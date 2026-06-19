"""Pure Splink (Fellegi-Sunter) resolution logic (NOT a Hamilton module).

Classical, unsupervised, local (DuckDB) entity resolution as a comparison baseline
for the LLM resolver. Emits the same `list[Cluster]` shape so the shared tail
(resolved_graph → metrics → report) is reused. Imported by `resolve_splink_module`.

NOTE: EM estimates m/u from pairwise comparisons; on a tiny corpus the weights are
illustrative, not trustworthy — validate on a larger corpus (see docs/implementation
T11.5). The flow still runs and is comparable to the offline run in MLflow.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from eval_hyper_extract.resolve import canonical_name
from eval_hyper_extract.schema import Cluster, Graph


def to_records(graph: Graph) -> pd.DataFrame:
    """One row per node: the comparison columns Splink links on."""
    return pd.DataFrame(
        [
            {
                "unique_id": n.id,
                "name": n.name,
                "name_norm": (n.name or "").lower().strip(),
                "type": n.type or "",
                "description": n.description or "",
            }
            for n in graph.nodes
        ]
    )


def train_linker(df: pd.DataFrame):
    """Build + EM-train a dedupe Linker (DuckDB backend, in-process)."""
    import splink.comparison_library as cl
    from splink import DuckDBAPI, Linker, SettingsCreator, block_on

    # Two blocking rules so EM can learn m for both fields (a field blocked on in a
    # pass is held fixed that pass — see the Splink docs).
    rules = [block_on("type"), block_on("substr(name_norm, 1, 1)")]
    settings = SettingsCreator(
        link_type="dedupe_only",
        comparisons=[
            # TF adjustment up-weights agreement on RARE tokens — the principled way to
            # keep P-101 / P-102 apart (shared part common, distinguishing digit rare).
            cl.JaroWinklerAtThresholds("name", [0.9, 0.7]).configure(term_frequency_adjustments=True),
            cl.ExactMatch("type").configure(term_frequency_adjustments=True),
        ],
        blocking_rules_to_generate_predictions=rules,
        retain_intermediate_calculation_columns=True,  # for the waterfall
    )
    linker = Linker(df, settings, DuckDBAPI())
    linker.training.estimate_probability_two_random_records_match([block_on("type")], recall=0.7)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e5)
    for rule in rules:
        linker.training.estimate_parameters_using_expectation_maximisation(rule)
    return linker


def clusters_from_df(clusters_df: pd.DataFrame, graph: Graph) -> list[Cluster]:
    """Group `unique_id` by `cluster_id` → `Cluster` (same canonical pick as offline)."""
    by_id = {n.id: n for n in graph.nodes}
    groups: dict[object, list[str]] = defaultdict(list)
    for _, row in clusters_df.iterrows():
        groups[row["cluster_id"]].append(str(row["unique_id"]))

    clusters: list[Cluster] = []
    for member_ids in groups.values():
        canonical = canonical_name([by_id[i].name for i in member_ids])
        clusters.append(Cluster(id=canonical, node_ids=member_ids, canonical_name=canonical))
    return clusters
