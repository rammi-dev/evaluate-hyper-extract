"""Pure report-building logic (NOT a Hamilton module).

Fragmentation accounting + markdown rendering, kept out of `report_module` so they
don't become DAG nodes.
"""

from __future__ import annotations

from eval_hyper_extract.metrics import Metrics, node_label_map
from eval_hyper_extract.resolve import PairVerdict
from eval_hyper_extract.schema import Cluster, Graph, GroundTruth


def fragmentation_rows(raw_graph: Graph, clusters: list[Cluster], ground_truth: GroundTruth) -> list[dict]:
    """Per entity: how many raw nodes it fragmented into vs how many resolved clusters hold it."""
    label = node_label_map(raw_graph.nodes, ground_truth)  # node id -> entity id
    rows: dict[str, dict] = {
        e.id: {"entity": e.canonical_name, "raw_fragments": 0, "resolved_fragments": 0}
        for e in ground_truth.canonical_entities
    }
    for nid, eid in label.items():
        if eid is not None:
            rows[eid]["raw_fragments"] += 1
    for e in ground_truth.canonical_entities:
        rows[e.id]["resolved_fragments"] = sum(
            1 for c in clusters if any(label.get(nid) == e.id for nid in c.node_ids)
        )
    return list(rows.values())


def verdict_validation(
    pair_verdicts: list[PairVerdict], raw_graph: Graph, ground_truth: GroundTruth
) -> list[dict]:
    """For each verifier verdict, the ground-truth expected answer and whether they agree.

    `expected_same` = do the two node names belong to the same canonical entity? It is
    None when either node's surface form isn't in the manifest (then `agree` is None and
    the pair is excluded from the agreement score).
    """
    label = node_label_map(raw_graph.nodes, ground_truth)
    rows: list[dict] = []
    for v in pair_verdicts:
        la, lb = label.get(v.a), label.get(v.b)
        expected = None if (la is None or lb is None) else (la == lb)
        rows.append(
            {
                "node_a": v.a,
                "node_b": v.b,
                "llm_same": v.same,
                "expected_same": expected,
                "agree": None if expected is None else (v.same == expected),
                "reason": v.reason,
            }
        )
    return rows


def agreement_score(rows: list[dict]) -> float:
    """Fraction of ground-truth-labeled pairs where the LLM matched the expected answer."""
    judged = [r for r in rows if r["agree"] is not None]
    return sum(r["agree"] for r in judged) / len(judged) if judged else 1.0


def table(rows: list[dict]) -> dict:
    return {
        "entity": [r["entity"] for r in rows],
        "raw_fragments": [r["raw_fragments"] for r in rows],
        "resolved_fragments": [r["resolved_fragments"] for r in rows],
    }


def markdown(raw: Metrics, resolved: Metrics, rows: list[dict]) -> str:
    lines = [
        "# evaluate-hyper-extract — run report",
        "",
        "| metric | raw (library `==`) | resolved |",
        "|---|---|---|",
        f"| nodes/clusters | {raw.n_clusters} | {resolved.n_clusters} |",
        f"| recall | {raw.recall:.3f} | {resolved.recall:.3f} |",
        f"| precision | {raw.precision:.3f} | {resolved.precision:.3f} |",
        f"| f1 | {raw.f1:.3f} | {resolved.f1:.3f} |",
        f"| lookalike_preserved | {int(raw.lookalike_preserved)} | {int(resolved.lookalike_preserved)} |",
        "",
        "## Fragmentation (variants reunified)",
        "",
        "| entity | raw fragments | resolved fragments |",
        "|---|---|---|",
    ]
    lines += [f"| {r['entity']} | {r['raw_fragments']} | {r['resolved_fragments']} |" for r in rows]
    return "\n".join(lines) + "\n"
