"""Driver — assemble the Hamilton DAG, attach local MLflow, run one config.

One `dr.execute` per configuration = one MLflow run (design §7). `resolution_mode`
selects the `clusters` impl via `@config.when` and is also logged as a param/tag.
"""

from __future__ import annotations

import mlflow
import mlflow.langchain
from hamilton import driver
from hamilton.plugins.h_mlflow import MLFlowTracker

from eval_hyper_extract import (
    clients_module,
    config_module,
    corpus_module,
    extract_module,
    metrics_module,
    report_module,
    resolve_module,
)

MODULES = [
    config_module,
    clients_module,
    corpus_module,
    extract_module,
    resolve_module,
    metrics_module,
    report_module,
]

# Scalars auto-log as MLflow metrics; final_report writes + logs artifacts.
TERMINAL = [
    "raw_recall",
    "raw_precision",
    "raw_f1",
    "recall",
    "precision",
    "f1",
    "raw_node_count",
    "resolved_node_count",
    "lookalike_preserved",
    "verifier_agreement",
    "final_report",
]

TRACKING_URI = "sqlite:///out/mlflow.db"
ARTIFACT_LOCATION = "out/mlartifacts"
EXPERIMENT = "evaluate-hyper-extract"


def default_inputs(**overrides) -> dict:
    # resolution_mode is supplied via with_config (@config.when), NOT inputs.
    base = dict(
        llm_model="google/gemini-2.5-flash",
        tau_candidate=0.55,
        embed_model="BAAI/bge-small-en-v1.5",
        ingest_order=0,
        template_path="data/template.yaml",
        corpus_dir="data/corpus",
        entities_path="data/entities.json",
        out_dir="out",
    )
    base.update(overrides)
    return base


def build_driver(resolution_mode: str = "offline", *, with_mlflow: bool = True, run_name: str | None = None):
    if with_mlflow:
        # Align the GLOBAL mlflow store with the adapter's client, so start_run and
        # the report node's mlflow.log_* attach to the same sqlite DB.
        mlflow.set_tracking_uri(TRACKING_URI)
        # Capture every ChatOpenAI call (extraction + each pair verification) as a
        # trace — prompt, response, tokens, latency — attached to the run, so the
        # model's actual outputs are inspectable for validation in the MLflow UI.
        mlflow.langchain.autolog()
    builder = (
        driver.Builder()
        .with_modules(*MODULES)
        .with_config({"resolution_mode": resolution_mode})
    )
    # NOTE: `.with_cache()` is deferred. The LLM/embedder client objects hold thread
    # locks (unpicklable), so Hamilton can neither store them nor derive a stable
    # cache key for the LLM-output nodes that depend on them. The proper fix is to key
    # the cached nodes on the config (model id) rather than the client object; until
    # then we run without the token cache (the eval corpus is tiny).
    if with_mlflow:
        builder = builder.with_adapters(
            MLFlowTracker(
                tracking_uri=TRACKING_URI,
                artifact_location=ARTIFACT_LOCATION,
                experiment_name=EXPERIMENT,
                run_name=run_name or f"mode={resolution_mode}",
                run_tags={"resolution_mode": resolution_mode},
            )
        )
    return builder.build()


def run(resolution_mode: str = "offline", *, with_mlflow: bool = True, **overrides) -> dict:
    """Execute one configuration end-to-end; returns the requested terminal values.

    Each call appends a NEW run to out/mlflow.db (history is never deleted) — that is
    the point: runs accumulate so configs can be compared in the MLflow UI.
    """
    inputs = default_inputs(**overrides)
    run_name = f"{resolution_mode} | tau={inputs['tau_candidate']} | {inputs['llm_model']}"
    dr = build_driver(resolution_mode, with_mlflow=with_mlflow, run_name=run_name)
    return dr.execute(TERMINAL, inputs=inputs)


def sweep(configs: list[dict]) -> None:
    """Run several configurations; each appends one comparable MLflow run."""
    for cfg in configs:
        mode = cfg.pop("resolution_mode", "offline")
        result = run(mode, **cfg)
        print(f"[{mode} {cfg}] -> recall {result['recall']:.2f} | "
              f"nodes {result['raw_node_count']}->{result['resolved_node_count']} | "
              f"lookalike_preserved {result['lookalike_preserved']}")


if __name__ == "__main__":
    result = run()
    print({k: v for k, v in result.items() if k != "final_report"})
    print("report:", result.get("final_report"))
