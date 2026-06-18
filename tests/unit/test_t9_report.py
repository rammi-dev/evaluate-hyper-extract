"""T9 acceptance: report writes all artifacts; fragmentation exact; MLflow logging fires."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_hyper_extract.config_module import Config
from eval_hyper_extract.metrics_module import score
from eval_hyper_extract.resolve_module import build_clusters, rewrite_graph
from eval_hyper_extract.resolve_module import ConfirmedPair, PairVerdict
from eval_hyper_extract.schema import Graph, GroundTruth
from eval_hyper_extract import report_module
from eval_hyper_extract.report_module import final_report, fragmentation_rows, verifier_agreement
from eval_hyper_extract.report import verdict_validation

A = ["P-101", "P 101", "P101", "the feedwater pump"]
C = ["T-200", "the storage tank"]

# verdicts: two correct merges + one correct split (all agree with ground truth)
VERDICTS = [
    PairVerdict("P-101", "P 101", True, "variant"),
    PairVerdict("T-200", "the storage tank", True, "coreference"),
    PairVerdict("P-101", "P-102", False, "different tags"),
]


def _perfect_clusters(mini_graph: Graph):
    # confirmed pairs that fully connect A and C (B stays singleton)
    confirmed = [ConfirmedPair(A[0], x) for x in A[1:]] + [ConfirmedPair(C[0], C[1])]
    return build_clusters(mini_graph.nodes, confirmed)


def test_verdict_validation_and_agreement(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    rows = verdict_validation(VERDICTS, mini_graph, ground_truth)
    by_pair = {(r["node_a"], r["node_b"]): r for r in rows}
    assert by_pair[("P-101", "P 101")]["expected_same"] is True and by_pair[("P-101", "P 101")]["agree"] is True
    assert by_pair[("P-101", "P-102")]["expected_same"] is False and by_pair[("P-101", "P-102")]["agree"] is True
    assert verifier_agreement(VERDICTS, mini_graph, ground_truth) == 1.0


def test_verifier_agreement_flags_disagreement(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    wrong = [PairVerdict("P-101", "P-102", True, "WRONG merge")]  # expected False, llm True
    assert verifier_agreement(wrong, mini_graph, ground_truth) == 0.0


def test_fragmentation_rows_exact(mini_graph: Graph, ground_truth: GroundTruth) -> None:
    clusters = _perfect_clusters(mini_graph)
    rows = {r["entity"]: r for r in fragmentation_rows(mini_graph, clusters, ground_truth)}
    assert rows["P-101"]["raw_fragments"] == 4 and rows["P-101"]["resolved_fragments"] == 1
    assert rows["P-102"]["raw_fragments"] == 1 and rows["P-102"]["resolved_fragments"] == 1
    assert rows["T-200"]["raw_fragments"] == 2 and rows["T-200"]["resolved_fragments"] == 1


def test_final_report_writes_artifacts(mini_graph: Graph, ground_truth: GroundTruth, tmp_path: Path) -> None:
    clusters = _perfect_clusters(mini_graph)
    resolved = rewrite_graph(mini_graph, clusters)
    raw_m = score(mini_graph.nodes, [type(clusters[0])(id=n.id, node_ids=[n.id]) for n in mini_graph.nodes], ground_truth)
    res_m = score(mini_graph.nodes, clusters, ground_truth)
    cfg = Config(llm_model="x", out_dir=str(tmp_path))

    path = final_report(mini_graph, resolved, clusters, VERDICTS, raw_m, res_m, ground_truth, cfg)

    assert Path(path).name == "report.md"
    for f in ("report.md", "raw_graph.html", "resolved_graph.html", "fragmentation_table.json", "verdict_validation.json"):
        assert (tmp_path / f).exists(), f"missing {f}"

    table = json.loads((tmp_path / "fragmentation_table.json").read_text())
    assert table["raw_fragments"] == [4, 1, 2]
    assert table["resolved_fragments"] == [1, 1, 1]
    assert "recall" in (tmp_path / "report.md").read_text()


def test_final_report_logs_to_mlflow_when_run_active(
    mini_graph: Graph, ground_truth: GroundTruth, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clusters = _perfect_clusters(mini_graph)
    resolved = rewrite_graph(mini_graph, clusters)
    m = score(mini_graph.nodes, clusters, ground_truth)

    logged_artifacts: list[str] = []
    logged_tables: list[str] = []
    monkeypatch.setattr(report_module.mlflow, "active_run", lambda: object())
    monkeypatch.setattr(report_module.mlflow, "log_artifact", lambda p: logged_artifacts.append(p))
    monkeypatch.setattr(
        report_module.mlflow, "log_table", lambda data, artifact_file: logged_tables.append(artifact_file)
    )

    final_report(mini_graph, resolved, clusters, VERDICTS, m, m, ground_truth, Config(llm_model="x", out_dir=str(tmp_path)))

    assert len(logged_artifacts) == 3  # raw html, resolved html, report.md
    assert logged_tables == ["fragmentation_table.json", "verdict_validation.json"]
