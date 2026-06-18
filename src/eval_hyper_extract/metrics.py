"""Pure scoring logic (NOT a Hamilton module) — design §5.

Pairwise precision/recall/F1 of a clustering vs the ground-truth manifest, plus the
`lookalike_preserved` gate and the library-key baseline clustering. Imported by
`metrics_module` (the node layer) and tests.
"""

from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

from pydantic import BaseModel

from eval_hyper_extract.schema import Cluster, Graph, GroundTruth, Node


class Metrics(BaseModel):
    recall: float
    precision: float
    f1: float
    lookalike_preserved: bool
    n_clusters: int
    n_nodes: int


def node_label_map(nodes: list[Node], gt: GroundTruth) -> dict[str, str | None]:
    """node id -> canonical entity id (or None if the node name is unknown)."""
    name_to_entity: dict[str, str] = {}
    for entity in gt.canonical_entities:
        for variant in entity.variants:
            name_to_entity[variant] = entity.id
    return {n.id: name_to_entity.get(n.name) for n in nodes}


def _assignment(clusters: list[Cluster]) -> dict[str, int]:
    assign: dict[str, int] = {}
    for i, c in enumerate(clusters):
        for nid in c.node_ids:
            assign[nid] = i
    return assign


def _lookalike_preserved(clusters: list[Cluster], label: dict[str, str | None], gt: GroundTruth) -> bool:
    cluster_labels = [{label.get(nid) for nid in c.node_ids if label.get(nid)} for c in clusters]
    for x, y in gt.lookalike_pairs:
        for labels in cluster_labels:
            if x in labels and y in labels:
                return False
    return True


def score(nodes: list[Node], clusters: list[Cluster], gt: GroundTruth) -> Metrics:
    """Pairwise precision/recall/F1 + lookalike gate for a clustering of `nodes`."""
    label = node_label_map(nodes, gt)
    assign = _assignment(clusters)
    labeled = [nid for nid, lab in label.items() if lab is not None]

    should = predicted = both = 0
    for a, b in combinations(labeled, 2):
        s = label[a] == label[b]
        ca, cb = assign.get(a), assign.get(b)
        p = ca is not None and ca == cb
        should += s
        predicted += p
        both += s and p

    recall = both / should if should else 1.0
    precision = both / predicted if predicted else 1.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

    return Metrics(
        recall=recall,
        precision=precision,
        f1=f1,
        lookalike_preserved=_lookalike_preserved(clusters, label, gt),
        n_clusters=len(clusters),
        n_nodes=len(nodes),
    )


def _eval_key(expr: str, node: Node) -> str:
    """Mirror of hyperextract `_extractor`: simple field or `{a}|{b}` composite."""
    if "{" in expr:
        fields = re.findall(r"\{(\w+)\}", expr)
        return expr.format(**{f: getattr(node, f, "") for f in fields})
    return str(getattr(node, expr))


def library_clusters(graph: Graph, library_key: str) -> list[Cluster]:
    """Group raw nodes by the library's `==` key — the fragmented baseline clustering."""
    groups: dict[str, list[str]] = defaultdict(list)
    for n in graph.nodes:
        groups[_eval_key(library_key, n)].append(n.id)
    return [Cluster(id=k, node_ids=v) for k, v in groups.items()]
