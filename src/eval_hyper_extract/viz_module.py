"""Simple knowledge-graph visualization — reusable for raw/resolved/online/hybrid.

`render_graph` writes a self-contained HTML (pyvis, inline assets — no server, no
network) so any `{nodes, edges}` is visually inspectable. Node label = name; tooltip
carries type/scope/description.
"""

from __future__ import annotations

from pathlib import Path

from eval_hyper_extract.schema import Edge, Node


def render_graph(nodes: list[Node], edges: list[Edge], out_path: str, title: str = "Knowledge graph") -> str:
    """Render a graph to a standalone HTML file; return its path."""
    from pyvis.network import Network

    net = Network(height="750px", width="100%", directed=True, notebook=False, cdn_resources="in_line")
    net.heading = title

    present: set[str] = set()
    for n in nodes:
        merged = [a for a in n.aliases if a != n.name]
        tooltip = f"type: {n.type} | scope: {n.scope}\n{n.description}".strip()
        label = n.name or n.id
        if merged:  # resolved node that absorbed variants — make the merge visible
            tooltip += "\nmerged from: " + ", ".join(n.aliases)
            label = f"{label}  (+{len(merged)})"
        net.add_node(n.id, label=label, title=tooltip)
        present.add(n.id)

    for e in edges:
        for endpoint in (e.source, e.target):
            if endpoint not in present:  # tolerate dangling endpoints in raw graphs
                net.add_node(endpoint, label=endpoint)
                present.add(endpoint)
        net.add_edge(e.source, e.target, label=e.relation, title=e.relation)

    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(p))
    return str(p)
