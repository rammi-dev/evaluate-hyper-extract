"""Report — the Hamilton node layer (design §7).

`final_report` writes artifacts (markdown, raw/resolved viz, fragmentation table) and
logs them to the active MLflow run. Pure helpers live in `report.py` (re-exported for
tests, excluded from the DAG).
"""

from __future__ import annotations

import json
from pathlib import Path

import mlflow

from eval_hyper_extract import report, resolve
from eval_hyper_extract.config_module import Config
from eval_hyper_extract.metrics import Metrics
from eval_hyper_extract.report import fragmentation_rows, verdict_validation  # re-exports for tests
from eval_hyper_extract.resolve import PairVerdict
from eval_hyper_extract.schema import Cluster, Graph, GroundTruth
from eval_hyper_extract.viz_module import render_graph

__all__ = ["resolved_graph", "fragmentation_rows", "verdict_validation", "verifier_agreement", "final_report"]


def resolved_graph(raw_graph: Graph, clusters: list[Cluster]) -> Graph:
    """Collapse the raw graph by the resolver's clusters (shared tail — every flow).

    Lives here (a shared module) so offline / Splink / online flows all reuse it; the
    resolver modules only differ in how they produce `clusters`.
    """
    g = resolve.rewrite_graph(raw_graph, clusters)
    ids = g.node_ids()
    assert all(e.source != e.target for e in g.edges), "self-edge survived"
    assert all(e.source in ids and e.target in ids for e in g.edges), "dangling edge endpoint"
    return g


def verifier_agreement(pair_verdicts: list[PairVerdict], raw_graph: Graph, ground_truth: GroundTruth) -> float:
    """Scalar metric: fraction of ground-truth-labeled pairs the LLM verifier got right."""
    return report.agreement_score(report.verdict_validation(pair_verdicts, raw_graph, ground_truth))


def final_report(
    raw_graph: Graph,
    resolved_graph: Graph,
    clusters: list[Cluster],
    pair_verdicts: list[PairVerdict],
    raw_metrics: Metrics,
    resolved_metrics: Metrics,
    ground_truth: GroundTruth,
    config: Config,
) -> str:
    """Write report.md + raw/resolved HTML + fragmentation + verdict-validation; log to MLflow."""
    out = Path(config.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    raw_html = render_graph(raw_graph.nodes, raw_graph.edges, str(out / "raw_graph.html"), "Raw (fragmented)")
    res_html = render_graph(resolved_graph.nodes, resolved_graph.edges, str(out / "resolved_graph.html"), "Resolved")

    # graph dumps (resolved nodes carry `aliases` = the surface forms they absorbed)
    (out / "raw_graph.json").write_text(raw_graph.model_dump_json(indent=2), encoding="utf-8")
    (out / "resolved_graph.json").write_text(resolved_graph.model_dump_json(indent=2), encoding="utf-8")

    rows = report.fragmentation_rows(raw_graph, clusters, ground_truth)
    frag_table = report.table(rows)
    (out / "fragmentation_table.json").write_text(json.dumps(frag_table, indent=2), encoding="utf-8")

    # validation: every verifier verdict vs the ground-truth expected answer
    validation = report.verdict_validation(pair_verdicts, raw_graph, ground_truth)
    (out / "verdict_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    report_path = out / "report.md"
    report_path.write_text(report.markdown(raw_metrics, resolved_metrics, rows), encoding="utf-8")

    if mlflow.active_run() is not None:
        for artifact in (raw_html, res_html, str(report_path)):
            mlflow.log_artifact(artifact)
        mlflow.log_table(data=frag_table, artifact_file="fragmentation_table.json")
        if validation:
            mlflow.log_table(
                data={k: [r[k] for r in validation] for k in validation[0]},
                artifact_file="verdict_validation.json",
            )

    return str(report_path)
