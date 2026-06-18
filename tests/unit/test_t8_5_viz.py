"""T8.5 acceptance: the KG visualizer writes valid standalone HTML."""

from __future__ import annotations

from pathlib import Path

from eval_hyper_extract.schema import Graph
from eval_hyper_extract.viz_module import render_graph


def test_render_writes_html_with_labels_and_edges(mini_graph: Graph, tmp_path: Path) -> None:
    out = render_graph(mini_graph.nodes, mini_graph.edges, str(tmp_path / "g.html"), title="Raw")
    html = Path(out).read_text(encoding="utf-8")
    assert html.strip()
    for node in mini_graph.nodes:
        assert node.name in html  # every node label rendered
    assert "feeds" in html  # an edge relation label rendered


def test_render_empty_graph_does_not_crash(tmp_path: Path) -> None:
    out = render_graph([], [], str(tmp_path / "empty.html"))
    assert Path(out).exists() and Path(out).read_text(encoding="utf-8").strip()
