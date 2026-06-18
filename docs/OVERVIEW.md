# evaluate-hyper-extract — what this is and how to look at it

A visual, plain-language tour of what was built and why. For the full design see
[design.md](design.md); for the task-by-task plan see
[implementation-plan.md](implementation-plan.md); for verified library facts see
[implementation.md](implementation.md).

## The goal (not "extract a knowledge graph")

The goal was to **expose and fix the one defect that silently decides graph quality**.
Hyper-Extract / ontomem does extraction and field-fusion with an LLM, but it does
**matching** — deciding which mentions are the *same* entity — with a string-equality
`==` on the entity key (`ontomem...BaseMerger._group_by_key`, a `defaultdict[key]`).
That `==` **fragments** variant-named entities (`P-101` / `P 101` / `the feedwater
pump` → many nodes) and **over-merges** look-alikes (`P-101` in two units → one node).

So this repo is a **harness** that:

1. **Measures** the fragmentation the library produces,
2. **Repairs** it with a proper external entity-resolution pass
   (embed → candidate → LLM-verify → cluster), and
3. **Evaluates** the before/after against a ground-truth manifest,

…all as a **gated Apache Hamilton dataflow** with **local, venv-only MLflow**
observability. Nothing here trusts the `==`; the whole point is to show what it costs
and undo it.

## Raw vs resolved — same extraction, different *matching*

This is the crux, and it's easy to misread. **Both graphs come from the same
Hyper-Extract extraction** — the LLM reading the documents into entity objects works
fine and is shared. What differs is how mentions are **matched into entities**:

- **Raw graph** = Hyper-Extract's *own* matching: a string `==` on the name. So
  `P-101`, `P 101`, and `Unit 2 feedwater pump` stay **three separate nodes** — the
  library **misses** that they are one physical pump. This is the defect under study.
- **Resolved graph** = we take *those same nodes* and re-match them with a real
  entity-resolution pass (embed → LLM-verify → cluster). It recognizes the three are
  one asset and **collapses them into a single node**.

The resolved pass does **not** re-run Hyper-Extract; it repairs Hyper-Extract's
matching. Concretely, from one run:

```
RAW  (Hyper-Extract ==):  'P 101' , 'P-101' , 'Unit 2 feedwater pump'   →  3 nodes
RESOLVED (external ER):   'P-101'  ⟵ merged: 'P 101', 'Unit 2 feedwater pump'   →  1 node
```

So if you look at the resolved graph and **don't see `P 101`, that's the point** — it
was absorbed into the `P-101` node. Each resolved node records the surface forms it
swallowed in its `aliases` field (shown as `P-101 (+2)` with a "merged from:" tooltip
in `out/resolved_graph.html`, and in `out/resolved_graph.json`). The look-alike
`P-102` is correctly **not** merged.

## The result, at a glance

One real offline run over a 3-document plant-equipment corpus
(`google/gemini-2.5-flash` + local `bge` embeddings):

![before vs after](images/before_after.png)

- **Library (`==`) baseline:** 13 fragmented nodes, **recall 0.00** — it unified
  *nothing*; every surface variant is its own node.
- **After external resolution:** **10 nodes**, **recall 1.00 / precision 1.00 / f1
  1.00**, and the **look-alike pair `P-101` vs `P-102` stayed distinct** (the hard
  gate held).

Per-entity fragmentation (`out/fragmentation_table.json`): `P-101` collapsed 2→1,
`P-102` 1→1, `T-200` 1→1.

## The pipeline (one node per step, a gate on each)

The harness is a Hamilton DAG. Every box is a function whose parameters are its
upstream dependencies; each carries a validation gate (a failing gate halts the run).

![pipeline DAG](images/dag.png)

Reading top-to-bottom: `config` → clients (`checked_llm`, `checked_embedder`) +
`corpus_docs` + `template` → **`raw_graph`** (the library's fragmented output) →
`node_embeddings` → `candidate_pairs` (embeddings = recall) → **`verified_pairs`**
(LLM = precision) → `clusters` → `resolved_graph` → metrics (`raw_*` vs resolved
`recall`/`precision`/`f1`, `lookalike_preserved`) → `final_report`. `library_key`
feeds the **baseline** so it is scored by the library's *real* key, not an assumed one.

## What was built (mapped to the goal)

| Goal step | Module(s) | What it does |
|---|---|---|
| Pin the defect | `docs/implementation.md` | verified the `==` is `BaseMerger._group_by_key`; key = template `identifiers.entity_id` |
| Reproduce fragmentation | `extract_module` / `extract` | build the library `AutoGraph` from `data/template.yaml`, feed corpus → raw graph |
| Score the damage honestly | `metrics_module` / `metrics` | pairwise recall/precision/F1 + the `lookalike_preserved` hard gate |
| Repair it | `resolve_module` / `resolve` | embed → candidate → **LLM-verify** → connected-components → rewritten graph |
| Make it inspectable | `viz_module`, `report_module` / `report` | interactive KG HTML + fragmentation table + markdown report |
| Make it observable & comparable | `run.py` + MLflow adapter | one `dr.execute` per config = one MLflow run with params + metrics + artifacts |
| Keep it honest | gates on every node | non-empty extraction, valid partition, no dangling edges, **no co-clustered look-alike** |

Correctness is locked by **68 unit tests** (deterministic core tested to exact
numbers) plus opt-in integration tests that hit the real LLM.

## How to analyze it visually

Everything below reads from the last run's artifacts in `out/` (git-ignored;
regenerate with the commands in the next section).

### 1. The knowledge graphs — before vs after
Open the standalone interactive HTML (pyvis; no server needed) in a browser:

- `out/raw_graph.html` — the **fragmented** library output (13 scattered nodes)
- `out/resolved_graph.html` — the **resolved** graph (variants collapsed)

From WSL you can launch them with `explorer.exe out\raw_graph.html` (or just open the
files from the VS Code explorer).

### 2. The MLflow run — metrics, params, artifacts
Start the local UI (SQLite-backed, no server/Docker) and browse to
http://127.0.0.1:5000:

```bash
uv run mlflow ui --backend-store-uri sqlite:///out/mlflow.db
```

In the UI: the **runs table** shows `raw_recall` vs `recall`, `raw_node_count` vs
`resolved_node_count`, `lookalike_preserved`, etc.; **params** show
`llm_model / tau_candidate / embed_model / resolution_mode`; **artifacts** include the
two graph HTMLs, `report.md`, and the fragmentation table. Sweeping configs (different
`tau_candidate`, model, mode) produces one comparable row each — sort by `f1`, filter
to `lookalike_preserved == 1`.

### 2b. Model responses (LLM traces) — validate what the model actually said
`mlflow.langchain.autolog()` is enabled, so **every `ChatOpenAI` call is captured as a
trace** — the full prompt, the model's response, token counts, and latency. Open the
**Traces** tab in the MLflow UI (or the run's Traces sub-tab) to read, for each
candidate pair, the verifier's exact decision and reasoning, e.g.:

```
prompt:   "You decide whether two extracted entities are the SAME real-world asset… A: P-101 … B: P-102 …"
response: {"same": false, "reason": "Different names and types."}
```

A typical offline run captures ~80 traces (the extraction calls plus one per verified
pair). Each trace is tagged with `mlflow.sourceRun`, so a run's own **Traces** sub-tab
shows only that run's calls (with token usage + cost); the experiment-level tab shows
the union across all runs. Query them programmatically with `mlflow.search_traces(...)`.

### 2c. Verdict validation — is the model actually right?
Reading traces tells you *what* the model said; the harness also checks it. Every
candidate verdict is captured (node `pair_verdicts`) and scored against ground truth:

- **`verifier_agreement`** — a logged MLflow **metric**: the fraction of
  ground-truth-checkable pairs where the LLM matched the expected answer (1.00 on the
  current run — the verifier agreed with the manifest on all 6 checkable pairs).
- **`out/verdict_validation.json`** (also an MLflow table artifact) — one row per
  verdict: `node_a, node_b, llm_same, expected_same, agree, reason`. Disagreements are
  the rows where `agree == false` — those are exactly the model calls to inspect.

```
OK  'P 101' vs 'P-101' : llm=True  expected=True   (Same tag and scope, surface variant.)
OK  'P 101' vs 'P-102' : llm=False expected=False  (differing equipment tag numbers and scope)
```

Only pairs whose *both* node names appear in the manifest are checkable (`expected_same`
is null otherwise — the surface-form caveat below); the rest are still captured for
manual review.

### 3. The static images
`docs/images/dag.png` (pipeline) and `docs/images/before_after.png` (this run's
delta) are committed to the repo and shown above; `out/report.md` is the same numbers
in markdown.

## How to run / regenerate

**Every run is appended to `out/mlflow.db` — history is never deleted.** That is the
point: runs accumulate so you can compare configs in the UI. Do **not** `rm` the DB
between runs.

```bash
# one offline run end-to-end (real LLM) → appends a run; writes out/*.html + report.md
uv run python -m eval_hyper_extract.run

# a sweep — each config appends one comparable run (vary tau / model / mode)
uv run python -c "from eval_hyper_extract.run import sweep; \
  sweep([{'tau_candidate':0.45},{'tau_candidate':0.55},{'tau_candidate':0.65}])"

# refresh the DAG + before/after PNG from the latest run (reads MLflow; no LLM calls)
uv run python scripts/make_visuals.py

# browse all accumulated runs + their LLM traces
uv run mlflow ui --backend-store-uri sqlite:///out/mlflow.db
```

Each run is named `"<mode> | tau=<…> | <model>"` so the runs table is self-describing.
Requires `OPEN_ROUTER_KEY` in `.env`. Tests: `uv run pytest` (unit, offline) or
`uv run pytest -m integration` (hits the real LLM).

## Honest caveats (what's not done / what to watch)

- **Manifest vs LLM surface forms.** The LLM emits names that don't always byte-match
  the hand-authored `data/entities.json` variants (e.g. `Unit 2 feedwater pump` vs
  `the feedwater pump`); unmatched nodes drop out of recall. Recall is exact for the
  *labeled* fragments, but the manifest should be reconciled with real extraction
  before reading too much into absolute recall. (Recorded in project memory.)
- **Token caching is deferred.** `.with_cache()` can't store the unpicklable LLM/
  embedder clients nor key the LLM-output nodes on them; the proper fix is to key
  those nodes on the config (model id). The corpus is tiny, so runs are cheap.
- **Online / hybrid modes (T12/T13) are designed but not yet built** — only the
  `offline` clusterer runs today. The DAG and `resolution_mode` dispatch are ready
  for them.
