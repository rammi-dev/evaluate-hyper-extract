"""Pure extraction-mapping + template logic (NOT a Hamilton module).

Kept out of `extract_module` so helpers don't become DAG nodes. `_to_graph` maps the
library's stored items into our schema; `validate_template_cfg` is the template gate.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_hyper_extract.schema import Edge, Graph, Node


def _attr(item: object, name: str) -> str:
    v = getattr(item, name, "")
    return str(v) if v is not None else ""


def to_graph(lib_nodes: list, lib_edges: list) -> Graph:
    """Map library node/edge items → our schema. Node id = name (key == name)."""
    nodes = [
        Node(
            id=_attr(n, "name"),
            name=_attr(n, "name"),
            type=_attr(n, "type"),
            scope=_attr(n, "scope"),
            description=_attr(n, "description"),
            evidence=_attr(n, "evidence"),
        )
        for n in lib_nodes
    ]
    edges = [
        Edge(source=_attr(e, "source"), target=_attr(e, "target"), relation=_attr(e, "type"))
        for e in lib_edges
    ]
    return Graph(nodes=nodes, edges=edges)


def assert_nonempty(graph: Graph) -> Graph:
    """Extraction-sanity gate."""
    assert graph.nodes, "extraction produced no nodes (broken template/client?)"
    assert graph.edges, "extraction produced no edges"
    return graph


def validate_template_cfg(cfg: object) -> None:
    """Template gate: a graph template with entities, relations, and an entity key."""
    assert getattr(cfg, "type", None) == "graph", "template type must be 'graph'"
    out = getattr(cfg, "output", None)
    assert getattr(out, "entities", None) is not None, "template output has no entities"
    assert getattr(out, "relations", None) is not None, "template output has no relations"
    ids = getattr(cfg, "identifiers", None)
    assert ids is not None and getattr(ids, "entity_id", None), "identifiers.entity_id missing"


def write_graph_json(graph: Graph, path: str) -> str:
    """Dump `{nodes, edges}` to disk (design A.3 shape)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {"nodes": [n.model_dump() for n in graph.nodes], "edges": [e.model_dump() for e in graph.edges]},
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(p)
