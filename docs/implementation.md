---

## Appendix A — Verified reference (so the implementer does NOT need to research)

These facts were established by reading source (ontomem `0.2.3`, Hyper-Extract
`main`) and current docs. Treat them as given; verify only the few items in
Appendix B against the installed versions.

### A.1 What the library is (the experiment's premise)

The Hyper-Extract / ontomem stack does three jobs: **(1) extraction** (LLM turns
text into entity/relation objects — real work, irreplaceable), **(2) field
fusion** (LLM or rules reconcile fields of items declared identical — well
engineered), and **(3) matching** (decides what is "the same" — and this is a
plain string-equality `==`). Graph topology is decided entirely by job 3.
**Do not try to "fix" this inside the library** — the harness's whole purpose is
to show the fragmentation and then unify externally.

### A.2 ontomem internals (verified, v0.2.3)

- Matching lives in `ontomem/merger/base.py` → `_group_by_key`, which buckets
  items with `defaultdict[key_extractor(item)]`. Items merge **only when their
  key string is byte-equal**. That dict lookup *is* the `==`.
- For the graph templates the node `key_extractor` is `lambda x: x.name` (e.g.
  `hyperextract/methods/typical/kg_gen.py:129`, `itext2kg.py:149`,
  `itext2kg_star.py:163`, `atom.py:356`, `methods/rag/hyper_rag.py:155`). Edge
  key is `"{source}|{type}|{target}"`-style. So node identity = the raw `name`
  string, unnormalized.
- The **7 merge strategies** (`MERGE_FIELD`, `KEEP_INCOMING`, `KEEP_EXISTING`,
  `LLM.BALANCED`, `LLM.PREFER_INCOMING`, `LLM.PREFER_EXISTING`,
  `LLM.CUSTOM_RULE`) are all **field-fusion-after-match**. None of them performs
  matching. Switching strategy changes *how* fields fuse, never *what* is
  considered the same.
- The `embedder` passed to `OMem` is used **only** for `build_index` / search
  (FAISS). It is **not** used in `add()` or in matching. Configuring an embedder
  does not make identity semantic.
- Merge runs as a tournament (log-depth, batched, concurrent field-fusion). It is
  a latency/parallelism optimization, not a token-cost reduction, and not ER.

### A.3 Hyper-Extract usage facts

- `read_input` reads UTF-8 text only (txt/md). There is **no PDF/Docling**; the
  README's `paper.pdf` example is misleading. Feed text via the Python API
  `feed_text(text)`; for a directory the CLI concatenates `*.txt`/`*.md` (loses
  per-doc provenance). The harness feeds Markdown directly.
- The graph object holds two `OMem` instances (`_node_memory`, `_edge_memory`).
  `.nodes` / `.edges` return the stored Pydantic items.
- On-disk dump = a directory with `data.json` (`{"nodes":[...],"edges":[...]}`)
  + `metadata.json` + `index/` (FAISS). The viz consumes `data.json` only; it is
  NOT a graph DB — topology is implicit in edge `source`/`target` strings.

### A.4 Apache Hamilton cheat-sheet (verified)

- Install **`apache-hamilton`** (`sf-hamilton` is a redirect). `[visualization]`
  extra additionally needs the system `graphviz` binary for PNG export.
- A function is a node; its **parameter names are its upstream dependencies**;
  return type annotation is required; `_`-prefixed functions are excluded; two
  nodes can't share a name across modules.
- Builder: `driver.Builder().with_modules(*mods).with_config(d).with_cache()
  .with_adapters(a).with_materializers(*m).build()`. Order is free; `.build()`
  last. `with_config` is for `@config.when` selection (not needed here).
- Execute: `dr.execute(final_vars: list[str], inputs: dict, overrides: dict)`.
- Visualize: `dr.display_all_functions("dag.png")`,
  `dr.visualize_execution(final_vars, "exec.png", inputs=...)`.
- Validation: `@check_output(...)` (built-ins like range/dtype) and
  `@check_output_custom(validator)`. A failing check raises `DataValidationError`
  and **downstream nodes don't run** — that is the "gate." For hard gates, a plain
  `assert`/`raise` in the node body works identically and avoids validator-API
  uncertainty.
- Caching: `.with_cache()` keys on code+inputs; cache dir `.hamilton_cache/`.

### A.5 Hamilton MLflow plugin (verified)

- `from hamilton.plugins.h_mlflow import MLFlowTracker`; attach via
  `.with_adapters(MLFlowTracker(...))`.
- Constructor params: `tracking_uri, registry_uri, artifact_location,
  experiment_name="Hamilton", experiment_tags, run_id, run_name, run_tags,
  run_description, log_system_metrics`.
- Behavior: starts an MLflow run **before** graph execution; auto-logs **inputs
  as params** and **scalar requested outputs as metrics**; figures/plots and
  models (via `to.mlflow(...)` materializer) as artifacts. No `mlflow` import is
  needed in node code for this auto-logging. One run per `dr.execute`.

### A.6 MLflow API (local, venv-only)

- `uv add mlflow` only — no server/Docker. `tracking_uri="sqlite:///out/mlflow.db"`,
  artifacts to a local folder; both auto-created on first run.
- Manual logging (only inside `final_report`, where the adapter's run is active):
  `mlflow.log_artifact(path)`, `mlflow.log_table(data=dict, artifact_file="x.json")`.
  `log_metric`/`log_param` are handled by the adapter.
- LLM tracing: `mlflow.langchain.autolog()` captures each LangChain `ChatOpenAI`
  call as a span (prompt, response, latency, tokens).
- View: `uv run mlflow ui --backend-store-uri sqlite:///out/mlflow.db`
  (http://127.0.0.1:5000). Compare via runs table, parallel-coordinates, charts.

### A.7 OpenRouter + embeddings facts

- OpenRouter is **OpenAI-compatible chat completions only** — no embeddings
  endpoint. Use `langchain_openai.ChatOpenAI(model=..., base_url=
  "https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY, temperature=0)`.
- Embeddings are **local**: `langchain_huggingface.HuggingFaceEmbeddings(
  model_name="BAAI/bge-m3")` (wraps sentence-transformers; ~2 GB first load;
  `BAAI/bge-small-en-v1.5` is the faster English fallback).

### A.8 Resolver algorithm (the external ER pass) — definitive

1. Take nodes from `out/graph/data.json`.
2. Embed `f"{name} | {type} | {description}"` per node (local embedder).
3. Pairwise cosine (corpus is tiny); candidate if `sim >= tau_candidate` (0.55).
4. LLM verify each candidate: strict yes/no + reason; **the prompt must state that
   differing equipment tag numbers (e.g. `P-101` vs `P-102`) denote different
   physical assets and must not be merged.** Keep `same == True`.
5. Connected components over confirmed-same pairs = clusters (singletons kept).
6. Per cluster: pick canonical (most specific tag-bearing name), merge
   descriptions, redirect every edge `source`/`target` to the canonical id, drop
   self-edges, dedupe edges → `out/resolved_graph.json`.
7. Metrics vs `entities.json`: recall (variants unified into one cluster),
   precision (cluster purity); **hard assert** every `lookalike_pairs` pair is in
   different clusters.

---

## Appendix B — The only things to verify locally (quick checks, not research)

Do these as 1–2 minute checks against the installed packages; do not open-ended
research them.

1. **Hyper-Extract graph construction signature.** Inspect how to build a graph
   extractor and pass an LLM + embedder: check `hyperextract.create_client` and
   the graph `Template`/`AutoGraph` constructor. If it won't accept a custom
   `base_url`/`api_key` or a prebuilt `ChatOpenAI`, fall back to env vars
   (`OPENAI_BASE_URL`, `OPENAI_API_KEY`). Confirm by printing one extracted node.
2. **Chosen graph template id + output schema.** List installed templates; pick
   one whose schema is `entities` + `relations`; print one raw node to confirm
   field names (`name`, `type`, `description`).
3. **`BaseDefaultValidator` interface** (only if using `@check_output_custom`).
   Otherwise use in-node `assert` for the hard gates — no check needed.
4. **`mlflow` autolog flavor** for the client path: `mlflow.langchain.autolog()`
   vs `mlflow.openai.autolog()`. Pick whichever traces the actual `ChatOpenAI`
   calls.
5. **Current cheap OpenRouter model id** for `config.LLM_MODEL`: confirm one is
   live on openrouter.ai/models (e.g. `google/gemini-2.0-flash-001`,
   `openai/gpt-4o-mini`). The `checked_llm` gate fails fast if the id is wrong.