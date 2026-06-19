"""Regenerate docs visuals from the LAST logged MLflow run — no LLM re-execution.

Reads metrics from out/mlflow.db and out/fragmentation_table.json, renders the DAG
(static, no node execution), and writes out/before_after.png. Run after:

    uv run python -m eval_hyper_extract.run
    uv run python scripts/make_visuals.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np

from eval_hyper_extract import resolve_module
from eval_hyper_extract.run import TRACKING_URI, build_driver

# --- DAG (static render; does not execute nodes, so no LLM/embedder calls) ----------
build_driver(resolve_module, with_mlflow=False).display_all_functions("out/dag.png", orient="TB")

# --- latest OFFLINE run (the before/after visual is the offline story) ---------------
mlflow.set_tracking_uri(TRACKING_URI)
runs = mlflow.search_runs(search_all_experiments=True, order_by=["start_time DESC"])
runs = runs[runs["metrics.b3_f1"].notna()]
if "tags.resolution_mode" in runs.columns:
    offline = runs[runs["tags.resolution_mode"] == "offline"]
    runs = offline if len(offline) else runs
m = runs.iloc[0]
raw_n, res_n = int(m["metrics.raw_node_count"]), int(m["metrics.resolved_node_count"])
raw = [m["metrics.raw_recall"], m["metrics.raw_precision"], m["metrics.raw_f1"]]
res = [m["metrics.recall"], m["metrics.precision"], m["metrics.f1"]]

RAW_C, RES_C = "#c0392b", "#27ae60"
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

ax = axes[0]
bars = ax.bar(["raw (library ==)", "resolved (external ER)"], [raw_n, res_n], color=[RAW_C, RES_C])
ax.set_title("Graph size: fragmentation collapsed")
ax.set_ylabel("distinct nodes")
for b, v in zip(bars, [raw_n, res_n]):
    ax.text(b.get_x() + b.get_width() / 2, v, str(v), ha="center", va="bottom", fontweight="bold")

ax = axes[1]
labels = ["recall", "precision", "f1"]
x = np.arange(len(labels))
w = 0.36
ax.bar(x - w / 2, raw, w, label="raw (library ==)", color=RAW_C)
ax.bar(x + w / 2, res, w, label="resolved (external ER)", color=RES_C)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.15)
ax.set_title("Resolution quality vs ground truth")
ax.legend(loc="lower right")

fig.suptitle("evaluate-hyper-extract — before vs after the external resolution pass", fontweight="bold")
fig.tight_layout()
fig.savefig("out/before_after.png", dpi=120)

table = json.loads(Path("out/fragmentation_table.json").read_text())
print(f"nodes {raw_n} -> {res_n} | raw recall {raw[0]:.2f} -> resolved {res[0]:.2f}")
print("fragmentation:", table)

# --- the two knowledge graphs, raw vs resolved (static, embeddable) -----------------
import networkx as nx  # noqa: E402


def _draw(ax, graph: dict, title: str, resolved: bool) -> None:
    g = nx.DiGraph()
    labels, colors = {}, []
    for n in graph["nodes"]:
        merged = [a for a in n.get("aliases", []) if a != n["name"]]
        g.add_node(n["name"])
        labels[n["name"]] = f"{n['name']}\n(+{len(merged)})" if merged else n["name"]
    for e in graph["edges"]:
        if e["source"] in g and e["target"] in g:
            g.add_edge(e["source"], e["target"])
    for n in graph["nodes"]:
        merged = [a for a in n.get("aliases", []) if a != n["name"]]
        colors.append("#27ae60" if (resolved and merged) else ("#9fd9b0" if resolved else "#e08a82"))
    pos = nx.spring_layout(g, seed=7, k=1.1)
    nx.draw_networkx_edges(ax=ax, G=g, pos=pos, edge_color="#bbbbbb", arrowsize=8, width=0.8)
    nx.draw_networkx_nodes(ax=ax, G=g, pos=pos, node_color=colors, node_size=900, edgecolors="#555")
    nx.draw_networkx_labels(ax=ax, G=g, pos=pos, labels=labels, font_size=6)
    ax.set_title(title, fontweight="bold")
    ax.axis("off")


raw_g = json.loads(Path("out/raw_graph.json").read_text())
res_g = json.loads(Path("out/resolved_graph.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
_draw(axes[0], raw_g, f"RAW — Hyper-Extract == matching ({raw_n} nodes, fragmented)", resolved=False)
_draw(axes[1], res_g, f"RESOLVED — after external ER ({res_n} nodes; green = merged variants)", resolved=True)
fig.suptitle("Same extraction, different matching — variant nodes collapse, look-alikes stay split", fontweight="bold")
fig.tight_layout()
fig.savefig("out/raw_vs_resolved_graph.png", dpi=130)

print("wrote out/dag.png, out/before_after.png, out/raw_vs_resolved_graph.png")
