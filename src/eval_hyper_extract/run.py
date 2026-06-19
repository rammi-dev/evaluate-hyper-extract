"""Driver — assemble one resolver flow, attach local MLflow, run one config.

**Separate flows, compared in MLflow.** Each resolver (offline today; Splink, online,
hybrid later) is its own flow: the shared modules (config, clients, corpus, extract,
metrics, report) plus exactly ONE resolver module that provides the `clusters` node.
You run each flow separately — one `dr.execute` = one MLflow run — and compare the
runs in the MLflow UI. There is no in-DAG mode dispatch.

**Different inputs.** The dataset is parameterized: pass `corpus_dir=`, `entities_path=`,
`template_path=` (and model/threshold knobs) to `run(...)` to point any flow at a
different or bigger corpus without code changes.
"""

from __future__ import annotations

from pathlib import Path

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

# Shared across every flow; the resolver module (providing `clusters`) is added per flow.
SHARED_MODULES = [config_module, clients_module, corpus_module, extract_module, metrics_module, report_module]

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
    "b3_precision",
    "b3_recall",
    "b3_f1",
    "llm_calls",
    "verifier_agreement",
    "final_report",
]

TRACKING_URI = "sqlite:///out/mlflow.db"
ARTIFACT_LOCATION = "out/mlartifacts"
EXPERIMENT = "evaluate-hyper-extract"
LLM_CACHE_PATH = "out/llm_cache.db"


def enable_llm_cache(path: str = LLM_CACHE_PATH) -> None:
    """Make LLM calls reproducible + cheap: cache every chat response by (prompt, model).

    The only non-determinism in the pipeline is the LLM (the local embedder is
    deterministic). Caching at the call level keys on the exact prompt — so identical
    inputs (same corpus, template, model, candidate pairs) reuse identical responses,
    and re-running a config yields identical results. Delete `out/llm_cache.db` to
    force fresh sampling. (On a cache hit the call is short-circuited, so cached calls
    are not re-traced in MLflow — the first, cache-miss run captures the traces.)
    """
    from langchain_community.cache import SQLiteCache
    from langchain_core.globals import set_llm_cache

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    set_llm_cache(SQLiteCache(database_path=path))


def default_inputs(**overrides) -> dict:
    """Run inputs (auto-logged as MLflow params). Override any for a different dataset/knob."""
    base = dict(
        llm_model="google/gemini-2.5-flash",
        tau_candidate=0.55,
        embed_model="BAAI/bge-small-en-v1.5",
        resolution_mode="offline",  # a logged LABEL for the flow (not an in-DAG switch)
        ingest_order=0,
        template_path="data/template.yaml",
        corpus_dir="data/corpus",
        entities_path="data/entities.json",
        out_dir="out",
    )
    base.update(overrides)
    return base


def build_driver(resolver_module=resolve_module, *, with_mlflow: bool = True, llm_cache: bool = True, run_name=None, run_tags=None):
    """Assemble a flow = SHARED_MODULES + one resolver module."""
    if llm_cache:
        enable_llm_cache()
    if with_mlflow:
        # Align the GLOBAL mlflow store with the adapter's client, so start_run and the
        # report node's mlflow.log_* attach to the same sqlite DB; trace every LLM call.
        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.langchain.autolog()
    builder = driver.Builder().with_modules(*SHARED_MODULES, resolver_module)
    # NOTE: `.with_cache()` is deferred — the LLM/embedder clients hold thread locks
    # (unpicklable), so Hamilton can't key the LLM-output nodes on them. Corpus is tiny.
    if with_mlflow:
        builder = builder.with_adapters(
            MLFlowTracker(
                tracking_uri=TRACKING_URI,
                artifact_location=ARTIFACT_LOCATION,
                experiment_name=EXPERIMENT,
                run_name=run_name,
                run_tags=run_tags or {},
            )
        )
    return builder.build()


def run(resolver_module=resolve_module, *, with_mlflow: bool = True, llm_cache: bool = True, **overrides) -> dict:
    """Run ONE flow end-to-end; appends a new MLflow run (history is never deleted).

    `resolver_module` selects the flow (default: the offline LLM resolver). `overrides`
    set inputs — e.g. `run(corpus_dir="data/big", entities_path="data/big.json")`.
    With `llm_cache=True` (default) repeated runs of the same config are reproducible.
    """
    inputs = default_inputs(**overrides)
    mode = inputs["resolution_mode"]
    run_name = f"{mode} | tau={inputs['tau_candidate']} | {inputs['llm_model']}"
    dr = build_driver(
        resolver_module, with_mlflow=with_mlflow, llm_cache=llm_cache,
        run_name=run_name, run_tags={"resolution_mode": mode},
    )
    return dr.execute(TERMINAL, inputs=inputs)


def sweep(configs: list[dict]) -> None:
    """Run several flows/configs; each appends one comparable MLflow run.

    Each dict may carry `resolver_module` (to switch flow) plus input overrides, e.g.
    `sweep([{"tau_candidate": 0.45}, {"tau_candidate": 0.65}])`.
    """
    for cfg in configs:
        resolver_module = cfg.pop("resolver_module", resolve_module)
        result = run(resolver_module, **cfg)
        print(f"[{result.get('resolution_mode', '')} {cfg}] -> recall {result['recall']:.2f} | "
              f"nodes {result['raw_node_count']}->{result['resolved_node_count']} | "
              f"lookalike_preserved {result['lookalike_preserved']}")


if __name__ == "__main__":
    result = run()
    print({k: v for k, v in result.items() if k != "final_report"})
    print("report:", result.get("final_report"))
