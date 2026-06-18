"""T6 acceptance: the extraction mapping + gates (unit), real extraction (integration)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval_hyper_extract.extract_module import (
    to_graph,
    assert_nonempty,
    library_key,
    validate_template_cfg,
    write_graph_json,
)
from eval_hyper_extract.schema import Graph


def test_to_graph_maps_fields() -> None:
    lib_nodes = [SimpleNamespace(name="P-101", type="pump", scope="Unit 2", description="feedwater pump", evidence="ev")]
    lib_edges = [SimpleNamespace(source="P-101", target="T-200", type="feeds")]
    g = to_graph(lib_nodes, lib_edges)
    n = g.nodes[0]
    assert (n.id, n.name, n.type, n.scope, n.description, n.evidence) == (
        "P-101", "P-101", "pump", "Unit 2", "feedwater pump", "ev",
    )
    e = g.edges[0]
    assert (e.source, e.target, e.relation) == ("P-101", "T-200", "feeds")  # type -> relation


def test_to_graph_handles_missing_fields() -> None:
    g = to_graph([SimpleNamespace(name="X")], [])  # no type/scope/etc.
    assert g.nodes[0].type == "" and g.nodes[0].scope == ""


def test_assert_nonempty_gate() -> None:
    with pytest.raises(AssertionError, match="no nodes"):
        assert_nonempty(Graph(nodes=[], edges=[]))


def test_library_key_from_cfg() -> None:
    cfg = SimpleNamespace(identifiers=SimpleNamespace(entity_id="name"))
    assert library_key(cfg) == "name"


def test_validate_template_rejects_missing_relations() -> None:
    bad = SimpleNamespace(
        type="graph",
        output=SimpleNamespace(entities=object(), relations=None),
        identifiers=SimpleNamespace(entity_id="name"),
    )
    with pytest.raises(AssertionError, match="relations"):
        validate_template_cfg(bad)


def test_write_graph_json_shape(mini_graph: Graph, tmp_path: Path) -> None:
    p = write_graph_json(mini_graph, str(tmp_path / "graph" / "data.json"))
    data = json.loads(Path(p).read_text())
    assert set(data) == {"nodes", "edges"}
    assert len(data["nodes"]) == 7 and len(data["edges"]) == 4


@pytest.mark.integration
def test_real_extraction_fragments() -> None:
    """Real run: corpus → non-empty graph whose nodes carry the enriched fields."""
    from eval_hyper_extract.clients_module import checked_embedder, checked_llm
    from eval_hyper_extract.config_module import Config
    from eval_hyper_extract.corpus_module import corpus_docs
    from eval_hyper_extract.extract_module import raw_graph, template

    cfg = Config(
        llm_model=os.environ.get("LLM_MODEL", "google/gemini-2.5-flash"),
        embed_model="BAAI/bge-small-en-v1.5",
        template_path="data/template.yaml",
        corpus_dir="data/corpus",
        entities_path="data/entities.json",
    )
    g = raw_graph(checked_llm(cfg), checked_embedder(cfg), corpus_docs(cfg), cfg, template(cfg))
    assert g.nodes and g.edges
    assert any(n.name for n in g.nodes)
    assert any(n.type for n in g.nodes)  # enriched schema populated
