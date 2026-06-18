# evaluate-hyper-extract — Design

> Companion to [implementation.md](implementation.md). That file's **Appendix A**
> is verified reference (treat as given); **Appendix B** is the short list of
> things to check against installed packages. This document is the experiment
> design those appendices support: what we are building, why, and the exact node
> graph to implement.

## 1. Premise (what this experiment proves)

Hyper-Extract / ontomem does three jobs. Two are real work done well — **extraction**
(LLM: text → entity/relation objects) and **field fusion** (LLM/rules reconcile
fields of items *already declared the same*). The third — **matching** (deciding
what is "the same entity") — is a string-equality `==` on an extracted key.
Graph topology is set entirely by job 3.

**Where the `==` lives (verified in source).** `ontomem` is a *separate installed
package* (indexed as `home-rami-Work-tmp-ontomem`), not a Hyper-Extract subdir.
Its `ontomem/merger/base.py::BaseMerger._group_by_key` buckets items with
`defaultdict[ key_extractor(item) ]` — two items merge **iff their key strings are
byte-equal**; that `defaultdict` *is* the `==`. The tournament merge wrapped around
it (`_cross_key_tournament_merge`) only batches field-fusion of items the key
*already* declared identical. The `key_extractor` is **not hardcoded** to `name`:
`AutoGraph.__init__` takes `node_key_extractor`/`edge_key_extractor` as required
params, which `TemplateFactory.create_graph` derives from the chosen template's
**`identifiers`** YAML block via `parse_identifiers` → `_extractor` (a plain field
`name` → `lambda x: str(x.name)`; a bracket template `{name}|{type}` → an f-string
composite key). So **the template's `identifiers` expression is the precise
definition of "what the library thinks is the same"** — the harness must read it,
not assume `name` (see §2.3, §4).

So the library, fed a corpus that names one asset many ways (`P-101` / `P 101` /
`P101` / `the feedwater pump`) and names different assets alike (`P-101` in two
plants), will **fragment** and **over-merge** — silently. The harness's purpose is
**not** to fix this inside the library (the seam is per-item and cannot compare
across items). It is to:

1. Run the library as-is and **measure** the fragmentation it produces.
2. Run an **external entity-resolution pass** on the dumped nodes
   (embed → candidate → LLM-verify → cluster).
3. **Score both** against a ground-truth manifest and show the delta.
4. Do all of it as a **Hamilton DAG** with a **validation gate on every node** and
   **local, venv-only MLflow** observability, so configs can be swept and compared.

The headline result is a before/after on the same corpus: *N fragments collapse to
M canonical entities, lookalike pairs stay distinct, recall ↑ without precision
collapse.*

## 2. Domain & corpus

Industrial-plant equipment — chosen because it exhibits both failure classes
cleanly and unambiguously:

- **Under-merge fuel:** tag-number formatting (`P-101` / `P 101` / `P101`), and
  description-by-role coreference (`the feedwater pump`, `the unit-2 feed pump`).
- **Over-merge fuel:** same tag in different units/plants (`P-101` @ Unit 2 vs
  `P-101` @ Plant B) — the **lookalike pairs** that a looser threshold or naked
  embedding similarity will wrongly collapse.

### 2.1 Input documents — `data/corpus/*.md`

A handful of short Markdown docs (UTF-8; `read_input` takes txt/md only — no PDF).
Each describes plant equipment in prose, deliberately using **variant surface
forms** for the same asset across and within docs, and reusing tags across
distinct units to seed lookalikes. ~3–6 docs is enough; the corpus is tiny so
pairwise resolution is fine.

### 2.2 Ground-truth manifest — `data/entities.json`

Metrics are meaningless without this. Schema:

```json
{
  "canonical_entities": [
    { "id": "P-101@U2", "canonical_name": "P-101",
      "type": "pump", "variants": ["P-101", "P 101", "P101", "the feedwater pump"] }
  ],
  "lookalike_pairs": [
    ["P-101@U2", "P-101@PB"]
  ]
}
```

- `canonical_entities[].variants` — every surface form that **must** end up in one
  cluster (drives **recall**).
- `lookalike_pairs` — pairs of distinct canonical ids that look alike and **must
  stay in different clusters** (drives the **hard gate** + precision). At least one
  pair is mandatory.

Authoring rule: the corpus and the manifest are written together — every variant
in the manifest appears somewhere in the corpus, and every reused-tag collision is
registered as a lookalike pair.

### 2.3 Extraction template — enriched entity schema (the right use of "hints")

The template's `identifiers` block is configurable, which tempts a "smarter" match
key like `{name}|{type}|{scope}`. **Do not** — a composite key is *still* string
equality; it only relocates the failure (more key fields → worse fragmentation,
fewer → worse collision), and the key fields are themselves variant LLM output. The
skill's rule stands: never try to make identity precise *inside* the library.

Instead, use the template's **`output` schema** and **`guideline`** (extraction
prompt) to make the LLM emit *disambiguating context fields*. Those fields are **not**
the match key — they feed the external resolver (§4), where an LLM can actually
*reason* about identity:

```yaml
# data/template.yaml  (or a chosen preset under hyperextract/templates/presets/)
identifiers:
  entity: name            # keep the library key SIMPLE — it only buckets
output:
  entity:
    name: ...             # surface form, exactly as written (do NOT normalize)
    type: ...             # pump / valve / tank
    scope: ...            # plant / unit / area — the namespace that splits lookalikes
    description: ...       # one line of role/context
    evidence: ...          # the source sentence (provenance)
guideline: >
  Extract each asset with its tag exactly as written, its equipment type, and the
  plant/unit it belongs to if stated. Do not normalize or merge tags.
```

`type`/`scope`/`description`/`evidence` do two jobs in the resolver, *neither* being
"be the key": (1) **disambiguation signal** for the verifier (it sees
`scope: Unit 2` vs `scope: Plant B` and answers "not same" with a reason); (2)
**separation in embedding space** (`P-101 | pump | Unit 2` and `P-101 | pump | Plant B`
embed apart → fewer false candidates; `P-101` and `the feedwater pump` embed closer
→ better recall). This is "identify them more precisely" done where intelligence
can live. `identifiers.entity` (`name` vs `{name}|{scope}`) becomes a **swept
config** (§8), not a fix.

## 3. Pipeline architecture (Hamilton DAG)

One function per step; **parameter names are dependencies**; return-type annotation
required; `_`-prefixed helpers excluded. Grouped into modules by concern. One
combined `dr.execute([...])` per configuration = one MLflow run.

### 3.1 Modules

| module | responsibility |
|---|---|
| `config_module` | runtime knobs surfaced as `inputs=` (auto-logged as MLflow params) |
| `clients_module` | LLM client (OpenRouter `ChatOpenAI`) + local embedder, each behind a fail-fast gate |
| `corpus_module` | load `data/corpus/*.md` + `data/entities.json` |
| `extract_module` | build `AutoGraph` from the template, feed corpus → dumped `{nodes, edges}`; also expose the template's resolved `identifiers` key expression |
| `resolve_module` | external ER: embed → candidate → verify → cluster → rewritten graph |
| `metrics_module` | thin scalar nodes (recall/precision/F1/counts/gate) for auto-log |
| `report_module` | artifacts: viz, fragmentation table, markdown report, `mlflow.log_*` |

### 3.2 Node table (transcribe each row to a function + in-node gate)

| node | depends on | returns | validation gate |
|---|---|---|---|
| `config` | (inputs) | `Config` | required keys present; `tau_candidate ∈ (0,1]` |
| `checked_llm` | `config` | `ChatOpenAI` | 1-token ping succeeds (bad model id / key fails here) |
| `checked_embedder` | `config` | `Embeddings` | embeds a probe string → non-empty vector |
| `corpus_docs` | `config` | `list[Doc]` | ≥1 doc, all non-empty UTF-8 |
| `ground_truth` | `config` | `GroundTruth` | ≥1 canonical entity **and** ≥1 lookalike pair |
| `template` | `config` | `TemplateCfg` | `output` has entities+relations; `identifiers.entity` present |
| `library_key` | `template` | `str` | the resolved key expr (e.g. `name`, `{name}\|{scope}`) — echoed as a param |
| `raw_graph` | `checked_llm`, `checked_embedder`, `corpus_docs`, `template` | `Graph` | **non-empty** nodes & edges (extraction sanity) |
| `raw_metrics` | `raw_graph`, `library_key`, `ground_truth` | `Metrics` | baseline fragmentation **keyed by `library_key`**, not an assumed `name` |
| `node_embeddings` | `raw_graph`, `checked_embedder` | `ndarray` | rows == node count; no NaNs |
| `candidate_pairs` | `node_embeddings`, `config` | `list[Pair]` | all sims ∈ [-1,1]; candidates ⊆ all-pairs |
| `verified_pairs` | `candidate_pairs`, `checked_llm` | `list[Pair]` | each verdict ∈ {same, different} + reason |
| `clusters` | `raw_graph`, `verified_pairs` | `list[Cluster]` | partition covers every node exactly once |
| `resolved_graph` | `raw_graph`, `clusters` | `Graph` | no self-edges; every edge endpoint ∈ node ids |
| `resolved_metrics` | `resolved_graph`, `clusters`, `ground_truth` | `Metrics` | **HARD: no lookalike pair co-clustered** |
| `recall` / `precision` / `f1` | `resolved_metrics` | `float` | scalar (auto-logs as metric) |
| `raw_node_count` / `resolved_node_count` | `*_graph` | `int` | scalar |
| `lookalike_preserved` | `resolved_metrics` | `int` (0/1) | scalar gate-as-metric |
| `final_report` | report + all metrics | `str` (path) | artifacts written; `mlflow.log_*` called |

The `resolved_metrics` lookalike assert is the **primary hard gate**: a failing
node raises and halts downstream — a run that over-merges a known-distinct pair
cannot silently report a good F1.

## 4. The external resolver (resolve_module) — definitive algorithm

Operates on `raw_graph` nodes (`name`, `type`, `scope`, `description`, `evidence`
per §2.3). This is the **blocking → candidate → verify → cluster** record-linkage
pattern: embeddings buy recall, the LLM is the precision backstop.

0. **Record the library's key** (`library_key`) so the baseline (§5) is scored
   against *how the library actually bucketed*, not an assumed `name`. The resolver
   then re-clusters those buckets semantically.
1. **Embed** `f"{name} | {type} | {scope} | {description}"` per node with the local
   embedder. The enriched context (not the match key) gives the verifier
   disambiguating signal *and* pushes lookalikes apart in vector space while pulling
   true variants together.
2. **Candidate generation** — pairwise cosine (corpus is tiny; kNN if it grows).
   Candidate iff `sim >= tau_candidate` (default **0.55**, biased to recall).
3. **LLM verify** each candidate: strict `same: yes/no` + `reason`, **with both
   nodes' `type`/`scope`/`evidence` in the prompt**. The prompt **must state** that
   differing equipment tag numbers (`P-101` vs `P-102`) denote different physical
   assets and that a differing `scope` (plant/unit) means distinct assets even when
   the tag matches. Keep only `same == True`. `temperature=0`; cacheable.
4. **Cluster** — connected components over confirmed-same pairs; singletons kept.
5. **Fuse** — per cluster pick the canonical name (most specific / tag-bearing),
   union descriptions, **redirect every edge `source`/`target` to the canonical
   id**, drop self-edges, dedupe edges → `out/resolved_graph.json`.
6. **Score** vs `entities.json` (§5).

Why not pure vectors: `P-101`/`P-102` embed as near-identical but are distinct
(vector over-merges); `P-101`/`the feedwater pump` embed far apart but are the same
(vector under-merges). The LLM resolves both; embeddings only *propose*.

## 5. Metrics (metrics_module)

Against `data/entities.json`, computed for **both** raw and resolved graphs:

- **Recall** — fraction of manifest variants that landed in a single cluster with
  their canonical (fragmentation undone). Raw graph ≈ low; resolved ≈ high.
- **Precision / cluster purity** — fraction of clusters containing exactly one
  canonical entity's members (no over-merge).
- **F1** — primary sweep metric (harmonic mean).
- **`lookalike_preserved`** — `1` iff *every* `lookalike_pairs` entry is in
  distinct clusters; else `0`. Surfaced as a scalar metric **and** enforced as the
  `resolved_metrics` hard assert.
- **Counts** — `raw_node_count`, `resolved_node_count` (the headline collapse).

Sweep selection rule: rank by F1, **but disqualify any run with
`lookalike_preserved == 0`** regardless of F1.

## 6. Validation gates (the "every step validated" requirement)

Each node carries a gate; a failure raises and stops downstream — that *is* the
gate. Prefer **in-node `assert`/`raise`** for hard gates (no validator-API risk);
`@check_output(range=...)` only for trivial numeric/type checks. Critical gates:

- `checked_llm` / `checked_embedder` — fail fast on bad model id, missing key,
  unreachable endpoint, before any expensive work.
- `raw_graph` — non-empty extraction (catches a broken template/client).
- `clusters` — partition covers every node exactly once (no lost/duplicated nodes).
- `resolved_graph` — no dangling edge endpoints, no self-edges.
- `resolved_metrics` — **lookalike pairs not co-clustered** (the headline gate).

## 7. Observability (MLflow, local, venv-only)

No server, no Docker. Tracking = `sqlite:///out/mlflow.db`; artifacts =
`out/mlartifacts/`. Two layers:

1. **Run-level** — Hamilton's `MLFlowTracker` adapter on the Builder. Auto-logs
   `inputs=` as **params** and requested **scalar** outputs as **metrics**. Hence
   the thin scalar nodes (`recall`, `precision`, `f1`, `*_node_count`,
   `lookalike_preserved`) — request them in `dr.execute([...])`.
2. **Artifacts** — inside `final_report` only (the adapter's run is active there):
   `mlflow.log_artifact(...)` for the graph HTML viz (raw vs resolved), the
   markdown report, and `mlflow.log_table(...)` for the fragmentation table
   (`entity | raw_fragments | resolved_fragments`).
3. **LLM tracing** — `mlflow.langchain.autolog()` (confirm flavor per Appendix B.4)
   captures each verifier call as a span so a mis-merge's reasoning is readable.

Caching: `.with_cache()` on the expensive LLM nodes (`raw_graph`, `verified_pairs`,
`temperature=0`); keep metric nodes cheap so they recompute and log every run.
Clear `.hamilton_cache/` to force fresh traces.

## 8. Config sweep (how we assess)

Each configuration = one `dr.execute` = one MLflow run. Swept knobs (all surfaced
via `config`/`inputs`, so auto-logged as params):

- `tau_candidate` — candidate cosine threshold (e.g. 0.45 / 0.55 / 0.65).
- `llm_model` — OpenRouter id (e.g. `google/gemini-2.0-flash-001`,
  `openai/gpt-4o-mini`).
- `embed_model` — `BAAI/bge-m3` (default) vs `BAAI/bge-small-en-v1.5` (faster).
- `library_key` — the template's `identifiers.entity`: `name` vs the composite
  `{name}|{scope}`. This is a *baseline* lever (cheap blocking against the
  cross-scope over-merge at extraction time), **not** a resolution mechanism — it
  does nothing for fragmentation and is brittle on `scope` extraction. The sweep
  shows the trade; the external pass still does the real lift.

Compare in `uv run mlflow ui`: runs table sorted by F1, parallel-coordinates for
the threshold↔recall/precision trade-off, filtered on `lookalike_preserved == 1`.

## 9. Clients (clients_module) — verified wiring

The harness **builds both clients itself and passes them straight into `AutoGraph`**
(`__init__(..., llm_client: BaseChatModel, embedder: Embeddings, ...)` accepts
prebuilt LangChain objects). We **never** call ontomem's `create_client` /
`create_embedder` — which sidesteps the "OpenRouter has no embeddings endpoint"
problem entirely: OpenRouter serves chat, a local model serves embeddings, and
nothing forces embeddings through ontomem's API-only factory. (ontomem uses the
embedder only for FAISS index/search, never for matching, so a local one is fine.)

- **LLM** via OpenRouter (OpenAI-compatible **chat only**):
  `ChatOpenAI(model=cfg.llm_model, base_url="https://openrouter.ai/api/v1",
  api_key=os.environ["OPEN_ROUTER_KEY"], temperature=0)`.
- **Embeddings local** (one model, used by *both* `AutoGraph`'s index and the §4
  resolver): `HuggingFaceEmbeddings(model_name=cfg.embed_model)` (`BAAI/bge-m3`;
  ~2 GB first load). No API embeddings anywhere in the pipeline.
- Secret loaded from `.env` as **`OPEN_ROUTER_KEY`** (valid identifier; the
  conventional `OPENROUTER_API_KEY` would also work if `ChatOpenAI` auto-reads it —
  we pass it explicitly regardless).

## 10. Directory layout

```
evaluate-hyper-extract/
├── .env                       # OPEN_ROUTER_KEY=... (gitignored)
├── data/
│   ├── corpus/*.md            # input documents (variant naming + reused tags)
│   ├── template.yaml          # enriched entity schema + simple identifiers key (§2.3)
│   └── entities.json          # ground-truth manifest
├── src/eval_hyper_extract/
│   ├── config_module.py
│   ├── clients_module.py
│   ├── corpus_module.py
│   ├── extract_module.py      # Hyper-Extract → raw_graph dump
│   ├── resolve_module.py      # external ER pass
│   ├── metrics_module.py
│   ├── report_module.py
│   └── run.py                 # builds the driver, executes, one run per config
├── out/                       # mlflow.db, mlartifacts/, graph/, resolved_graph.json, *.html
├── docs/{implementation.md, design.md}
└── pyproject.toml
```

## 11. Dependencies (`uv add`)

`apache-hamilton[visualization]` (+ system `graphviz` for PNG), `mlflow`,
`langchain-openai`, `langchain-huggingface` (sentence-transformers), `hyperextract`
/ `ontomem`, `numpy`, `python-dotenv`. Secret via `.env`: `OPEN_ROUTER_KEY`.

## 12. Build order

1. Author `data/corpus/*.md` + `data/entities.json` **together** (§2).
2. Run the **Appendix B** checks (graph template id + node schema, LLM/embedder
   wiring, mlflow autolog flavor, a live OpenRouter model id, `BaseDefaultValidator`
   only if used). 1–2 min each, not research.
3. Implement modules bottom-up: clients → corpus → extract → `raw_metrics`
   (baseline) → resolve → `resolved_metrics` → report.
4. Wire `MLFlowTracker`; confirm `out/mlflow.db` + populated run in the UI.
5. Sweep `tau_candidate` × `llm_model` × `embed_model`; compare runs; pick the
   best F1 among runs with `lookalike_preserved == 1`.

## 13. Success criteria

- Raw graph demonstrably **fragmented** (`raw` recall low, `raw_node_count` high).
- Resolved graph: recall ↑ materially, `resolved_node_count` ↓, **precision holds**.
- **`lookalike_preserved == 1`** on the chosen run (hard gate never tripped).
- Every node has a gate; every config is one comparable MLflow run with params,
  metrics, viz, fragmentation table, and verifier traces attached.
