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

from eval_hyper_extract.run import EXPERIMENT, TRACKING_URI, build_driver

# --- DAG (static render; does not execute nodes, so no LLM/embedder calls) ----------
build_driver("offline", with_mlflow=False).display_all_functions("out/dag.png", orient="TB")

# --- pull the latest run's metrics --------------------------------------------------
mlflow.set_tracking_uri(TRACKING_URI)
runs = mlflow.search_runs(experiment_names=[EXPERIMENT], order_by=["start_time DESC"])
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
print("wrote out/dag.png, out/before_after.png")
