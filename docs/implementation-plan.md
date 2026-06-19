# evaluate-hyper-extract — Implementation Plan

Derived from [design.md](design.md); facts from [implementation.md](implementation.md).
Tasks are ordered by dependency. Each has an **Objective**, **Deliverables**, and
**Tests** (the acceptance criteria). Build bottom-up so the deterministic core
(metrics, clustering) is fully tested *before* any LLM is wired.

## Progress tracker

Check a task off only when its **Tests** pass. Sub-boxes track the gating steps.

- [x] **T0 — Scaffold & dependencies** · `pytest` green · imports smoke · `.env` loads
- [x] **T1 — Test fixtures** (synthetic graph + manifest)
- [x] **T2 — Data assets** (corpus, template, manifest) · variant-coverage test
- [ ] **T2.5 — Matching characterization** (library on fake data) · `==` fragments + over-merges
- [x] **T3 — `config_module`** · range/missing-key/mode gates
- [x] **T4 — `clients_module`** · fail-fast gate (mock) · real ping (integration)
- [x] **T5 — `corpus_module`** · ground-truth gate
- [x] **T6 — `extract_module`** · `library_key` · empty-graph gate · real extraction verified (13 nodes, recall 0)
- [x] **T7 — `metrics_module`** (core) · exact recall/precision · lookalike gate raises
- [x] **T8 — `resolve_module` (offline)** · connected-components · edge redirect · stub verifier
- [x] **T8.5 — `viz_module`** · `render_graph` writes HTML · raw/resolved
- [x] **T9 — `report_module`** · artifacts · `mlflow.log_*`
- [x] **T10 — `run.py` + MLflow adapter** (offline) · real run logged · LLM traces captured · separate-flow `build_driver(resolver_module)` · DAG renders
- [ ] **T10b — driver tests** (DAG asserts, planted over-merge halts at gate) · pending
- [ ] **T11 — Sweep & assessment** · ≥2 comparable runs · disqualify rule
- [x] **T11.5 — Splink (Fellegi-Sunter) resolver flow** · runs end-to-end · B-cubed + `llm_calls` + disqualify gate · comparison logged (offline wins both; Splink over/under-merges) · _waterfall artifact still TODO_
- [ ] **T12 — Online-semantic resolver** · greedy link/mint · **order-dependence** test
- [ ] **T13 — Hybrid resolver** · online→offline reconcile · **convergence** test

## Testing strategy (read first)

The pipeline has two kinds of code, tested differently:

- **Deterministic core** — clustering, edge redirect, metrics, gates. No I/O, no
  models. **Fully unit-tested** against hand-built fixtures. This is where most
  tests live and where correctness actually matters.
- **External edges** — the OpenRouter LLM and the local embedder. **Mocked/stubbed**
  in unit tests; exercised by a small number of **opt-in integration tests**.

Conventions:

- `uv run pytest` runs **unit tests only** (fast, no network, no model download).
- Integration tests are marked `@pytest.mark.integration` and **skip unless**
  `OPEN_ROUTER_KEY` is set (and, for embedder tests, the model is downloadable).
  Run with `uv run pytest -m integration`.
- Determinism: LLM `temperature=0`, fixed `embed_model`, fixed seeds. Resolver
  logic never depends on model output in unit tests — verifier verdicts are
  injected via a stub.
- Shared fixtures in `tests/conftest.py`; synthetic graph + manifest in
  `tests/fixtures/`.

```
tests/
├── conftest.py                  # fixtures: mini_graph, ground_truth, fake_llm, fake_embedder
├── fixtures/
│   ├── mini_graph.json          # hand-built {nodes, edges} with known fragmentation
│   └── mini_entities.json       # matching ground-truth manifest
├── unit/                        # default run — no network
└── integration/                 # @pytest.mark.integration — needs OPEN_ROUTER_KEY
```

---

## T0 — Project scaffold & dependencies

**Objective.** A runnable `uv` project with deps, package layout, env loading, and a
green (empty) test run.

**Deliverables.** `pyproject.toml`; `src/eval_hyper_extract/__init__.py`;
`tests/` tree; `.gitignore` (`.env`, `out/`, `.hamilton_cache/`,
`.codebase-memory/`); `python-dotenv` loaded at startup.

**Tests.**
- `uv run pytest` exits 0 (collects, zero failures).
- Import smoke test: `import hamilton, mlflow, langchain_openai,
  langchain_huggingface, hyperextract, ontomem` all succeed.
- `.env` with `OPEN_ROUTER_KEY` loads and is readable via `os.environ`.

---

## T1 — Test fixtures (synthetic graph + manifest)

**Objective.** A tiny, hand-authored fragmented graph and its ground truth, used by
all core tests — **no LLM, fully deterministic**.

**Deliverables.** `tests/fixtures/mini_graph.json` and `mini_entities.json`, plus
`conftest.py` fixtures `mini_graph`, `ground_truth`, `fake_llm`, `fake_embedder`.

Design the fixture to contain every case the metrics must catch:
- One entity with 3 surface variants split into 3 nodes (`P-101`, `P 101`, `P101`).
- One coreference variant (`the feedwater pump`) for the same entity.
- One **lookalike pair**: a second `P-101` in a different `scope` (must stay split).
- An edge whose endpoint uses a variant name (to test redirect + dangling logic).

**Tests.**
- Fixtures load and validate against the Pydantic/JSON schemas.
- `mini_entities.json` has ≥1 canonical entity with ≥2 variants **and** ≥1
  lookalike pair (else later metric tests are vacuous).
- Sanity: the raw fixture's distinct-node count > canonical-entity count (i.e. it
  *is* fragmented), so "before" is meaningfully worse than "after".

---

## T2 — Data assets (corpus, template, manifest)

**Objective.** The real inputs: a small Markdown corpus, the enriched extraction
template, and the ground-truth manifest — corpus and manifest authored together.

**Deliverables.** `data/corpus/*.md` (3–6 docs); `data/template.yaml` (§2.3 schema:
`identifiers.entity: name`; `output` = entities+relations with
`name/type/scope/description/evidence`; `guideline`); `data/entities.json`.

**Tests.**
- `data/entities.json` validates (schema of §2.2); ≥1 lookalike pair present.
- **Coverage test:** every `variants[]` string in the manifest appears verbatim in
  at least one corpus doc (grep-style assertion) — guarantees the corpus can
  actually exhibit the claimed fragmentation.
- **Collision test:** every `lookalike_pairs` entry corresponds to a reused
  surface form that genuinely appears under ≥2 distinct scopes in the corpus.
- `data/template.yaml` parses; `output` resolves to an entity schema with the five
  fields; `identifiers.entity` is present.

---

## T2.5 — Matching characterization (run the library on fake data)

**Objective.** A focused, standalone test that **runs the real `AutoGraph` on tiny
crafted inputs and asserts how its `==` matching behaves** — the "check how matching
is working" capability, independent of the full pipeline. This is the experiment's
premise made executable.

**Deliverables.** A small helper `extract_module.build_graph(template, llm, embedder)`
+ `feed(graph, texts)` usable in isolation; `tests/integration/test_t2_5_matching.py`.

**Tests** (integration — needs `OPEN_ROUTER_KEY`; tiny corpora keep it cheap):
- **Fragmentation:** feed text mentioning the *same* asset two ways (`P-101` and
  `the feedwater pump`) → the library emits **two** nodes (keys differ ⇒ no merge).
- **Over-merge:** feed two *distinct* assets that share a tag (`P-101` in Unit 2,
  `P-101` in Plant B) → the library emits **one** node (keys equal ⇒ merged).
- **Determinism of `==` topology:** feed the same mentions in reversed order → the
  node/edge *key set* is identical (proves `==` topology is order-independent —
  motivates why a *semantic* online mode is needed for order-dependence, T12).
- Captures one verifier-free `data.json` dump for the T8.5 viz smoke test.

---

## T3 — `config_module`

**Objective.** A typed `Config` surfaced as Hamilton `inputs=` (so it auto-logs as
MLflow params), with the validation gate.

**Deliverables.** `config_module.py`: `config(...) -> Config` node; fields
`tau_candidate, llm_model, embed_model, library_key, resolution_mode, ingest_order,
template_path, corpus_dir, entities_path, out_dir`.

**Tests.**
- Valid inputs build a `Config`.
- `tau_candidate` outside `(0, 1]` → raises (gate).
- `resolution_mode` not in `{offline, online, hybrid}` → raises (gate).
- Missing required key → raises with a clear message.

---

## T4 — `clients_module`

**Objective.** Build the OpenRouter chat client and the local embedder, each behind
a fail-fast gate; pass prebuilt objects downstream (never `create_client`).

**Deliverables.** `clients_module.py`: `checked_llm(config) -> ChatOpenAI`,
`checked_embedder(config) -> Embeddings`. Each does a probe call and asserts.

**Tests.**
- *unit:* with a **mock** client whose ping succeeds → node returns it; with a mock
  whose ping raises/returns empty → gate raises (bad key/model id path).
- *unit:* `checked_embedder` with a fake embedder returning a non-empty vector
  passes; empty/NaN vector → raises.
- *integration* (`OPEN_ROUTER_KEY` required): real 1-token ping to OpenRouter
  succeeds; real `bge-m3` embeds a probe string to a non-empty float vector.

---

## T5 — `corpus_module`

**Objective.** Load corpus docs and the ground-truth manifest as typed nodes.

**Deliverables.** `corpus_module.py`: `corpus_docs(config) -> list[Doc]`,
`ground_truth(config) -> GroundTruth`.

**Tests.**
- Loads the expected number of docs from a temp corpus dir; contents non-empty UTF-8.
- Empty/missing corpus dir → `corpus_docs` raises.
- `ground_truth` gate: manifest with 0 lookalike pairs → raises; valid manifest →
  parses canonical entities + pairs.

---

## T6 — `extract_module`

**Objective.** Build `AutoGraph` from the template with harness-built clients, feed
the corpus, dump `{nodes, edges}`, and expose the library's real match key.

**Deliverables.** `extract_module.py`: `template(config) -> TemplateCfg`,
`library_key(template) -> str`, `raw_graph(checked_llm, checked_embedder,
corpus_docs, template) -> Graph`, plus a dump to `out/graph/data.json`.

**Tests.**
- *unit:* `library_key` returns the template's resolved `identifiers.entity`
  expression (e.g. `"name"`); for a composite template it returns `"{name}|{scope}"`.
- *unit:* `template` gate rejects a template whose `output` lacks entities or
  relations.
- *unit:* `raw_graph` with a **stub `AutoGraph`** returning a known graph → passes;
  stub returning empty nodes/edges → gate raises (extraction-sanity).
- *integration:* real extraction over `data/corpus/` yields ≥1 node and ≥1 edge;
  one printed node has fields `name/type/scope/description`; dump file has shape
  `{"nodes":[...],"edges":[...]}`.

---

## T7 — `metrics_module` (deterministic core — most tests here)

**Objective.** Pure scoring functions over (graph, clusters, ground_truth), keyed by
the **real** `library_key` for the baseline. No models.

**Deliverables.** `metrics_module.py`: `raw_metrics`, `resolved_metrics`, and the
thin scalar nodes `recall, precision, f1, raw_node_count, resolved_node_count,
lookalike_preserved`. Helper: `clusters_to_assignment(...)`.

**Tests** (all against `mini_graph` / `mini_entities`, exact expected numbers):
- **Perfect clustering** (all variants together, lookalikes split) → `recall==1.0`,
  `precision==1.0`, `lookalike_preserved==1`.
- **Total fragmentation** (every node its own cluster) → low recall, `precision==1.0`
  (pure but fragmented), counts equal node count.
- **Over-merge a lookalike pair** → `lookalike_preserved==0` and `precision<1.0`.
- **Under-merge** (one variant left out of its cluster) → `recall<1.0` with the
  exact fraction.
- `f1` equals the harmonic mean of the computed recall/precision.
- `raw_metrics` baseline is computed with `library_key` (test that swapping the key
  expression changes the baseline fragmentation count).
- `resolved_metrics` **hard gate**: a clusters input that co-clusters a lookalike
  pair → `resolved_metrics` raises (this is the headline gate).

---

## T8 — `resolve_module` (offline / batch mode)

**Objective.** The shared signal + the **offline** clusterer: embed → candidate →
verify → connected-components → rewrite, with the verifier mocked in unit tests.
Online/hybrid modes come in T12/T13; this task ships the default and the shared
machinery they reuse. Connected-components and edge-rewrite logic is deterministic
and exhaustively tested.

**Deliverables.** `resolve_module.py` (the offline flow's resolver): `node_embeddings`,
`candidate_pairs`, `pair_verdicts`, `verified_pairs`, `clusters`, `resolved_graph`.
Pure logic in `resolve.py` is reused by the peer resolver modules (splink/online/hybrid).

**Tests.**
- *unit:* `node_embeddings` with `fake_embedder` → array rows == node count, no NaNs
  (gate); embed text is `f"{name} | {type} | {scope} | {description}"`.
- *unit:* `candidate_pairs` keeps only pairs with `sim >= tau_candidate`; result ⊆
  all pairs; sims clamped to `[-1, 1]` (gate). Raising `tau` strictly shrinks the set.
- *unit:* `verified_pairs` with a **stub verifier** (maps pair→`same/different` from a
  table) keeps only `same==True`; each kept pair carries a `reason`.
- *unit:* `clusters` = connected components over confirmed-same pairs — test:
  transitive chain (a~b, b~c ⇒ one cluster), singletons kept, **partition covers
  every node exactly once** (gate). A lookalike pair never linked stays in two
  clusters.
- *unit:* `resolved_graph` — canonical name = most specific/tag-bearing; **edge
  endpoints redirected** to canonical ids; self-edges dropped; duplicate edges
  deduped; no dangling endpoints (gate).
- *unit:* end-to-end on `mini_graph` with a stub verifier that confirms the true
  variants and rejects the lookalike → resolved graph has exactly
  `canonical_entity_count` nodes and `lookalike_preserved==1`.
- *integration:* real verifier on a crafted `P-101` vs `P-102` pair returns
  `same=False`; on `P-101` vs `P 101` returns `same=True`.

---

## T8.5 — `viz_module` (simple KG visualization)

**Objective.** A reusable, standalone renderer for a `{nodes, edges}` graph → HTML,
so any graph (raw, resolved, online, hybrid) is visually inspectable. Built early
and small so it can render the T2.5 dump immediately.

**Deliverables.** `viz_module.py`: `render_graph(nodes, edges, out_path,
title) -> str` (returns the path). Uses a lightweight HTML/JS graph (e.g. pyvis or a
self-contained vis-network template; no server). Node label = canonical name; tooltip
= `type/scope/description`.

**Tests.**
- *unit:* `render_graph` on `mini_graph` writes a non-empty `.html` containing every
  node label and an edge count matching the input.
- *unit:* empty graph → still writes a valid (empty) HTML, no crash.
- *smoke:* renders the T2.5 fake-data dump to `out/fake_graph.html` (manual eyeball).

---

## T9 — `report_module`

**Objective.** Emit artifacts and attach them to the active MLflow run.

**Deliverables.** `report_module.py`: `final_report(...)` writing
`out/report.md`, raw/resolved graph HTML (via `viz_module.render_graph`, T8.5), and
the fragmentation table; calls `mlflow.log_artifact` / `mlflow.log_table`.

**Tests.**
- *unit:* with a temp `out_dir`, `final_report` writes `report.md`, both HTML files,
  and a fragmentation-table JSON whose rows = `[entity, raw_fragments,
  resolved_fragments]` matching the fixture counts.
- *unit:* `mlflow.log_*` calls are asserted via a patched `mlflow` (no real run).
- *integration:* under an active `MLFlowTracker` run, artifacts appear in
  `out/mlartifacts/`.

---

## T10 — `run.py` (driver wiring + MLflow adapter)

**Objective.** Assemble the Hamilton driver (modules + `.with_cache()` +
`MLFlowTracker`), execute the terminal + scalar nodes once per config.

**Deliverables.** `run.py`: builds driver, `dr.execute([...])` requesting the scalar
metric nodes + `final_report`; CLI/loop over a config list.

**Tests.**
- *unit:* the DAG assembles — `driver.Builder().with_modules(...).build()` succeeds
  and `display_all_functions` lists every node from the table (no name clashes,
  no unresolved deps).
- *unit:* a **planted over-merge** (stub verifier merges a lookalike pair) makes
  `dr.execute` raise at `resolved_metrics` and **not** write a success report
  (gate halts downstream).
- *integration:* one real `dr.execute` creates `out/mlflow.db` with exactly one run
  whose params (`tau_candidate, llm_model, embed_model, library_key`) and metrics
  (`recall, precision, f1, *_node_count, lookalike_preserved`) are populated.

---

## T11 — Config sweep & assessment

**Objective.** Run the matrix and confirm runs are comparable and the disqualify
rule holds.

**Deliverables.** A sweep entrypoint over `tau_candidate × llm_model × embed_model
× library_key × resolution_mode × ingest_order`; short README on reading the MLflow UI.

**Tests.**
- *integration:* sweeping ≥2 `tau_candidate` values yields ≥2 separate, comparable
  MLflow runs.
- *integration:* the three `resolution_mode`s each produce a run; accuracy order
  `offline ≥ hybrid ≥ online ≥ raw` holds on F1 (within tolerance).
- Assessment assertion (scripted): among runs, the selected "best" has
  `lookalike_preserved == 1`; any run with `0` is excluded regardless of F1.

---

## T11.5 — Splink (Fellegi-Sunter) resolver mode — classical comparison baseline

Full drop-in spec: [docs/splink](splink). Sequenced **before** online/hybrid because
it's a peer of the existing `offline` resolver, not a new paradigm — it answers
"does the LLM resolver justify its per-call cost and opacity vs principled classical
ER on this domain?" on the same ground truth, metrics, and MLflow.

**Objective.** Add `resolution_mode="splink"`: an unsupervised Fellegi-Sunter resolver
(Splink + DuckDB, local/venv-only) that emits the **same `clusters` shape**, so the
entire shared tail (`resolved_graph` → metrics → viz → report → MLflow) is reused
unchanged. That shared tail is what makes the comparison apples-to-apples.

**Reconciliation with [docs/splink](splink)** (the doc predates the current code):
- the doc's `entity_clusters` dispatch node = our **`clusters`** node. We use
  **separate flows, not in-DAG dispatch**: each resolver is its own module defining a
  `clusters` node; a flow loads exactly one. The shared tail already exists, so the
  doc's §1 "refactor first" is **already done**.
- `extracted_graph` → **`raw_graph`**; `run_one` → **`run`** / `sweep`.
- cluster type: emit **`list[Cluster]`** (not `list[set]`) — group `unique_id` by
  `cluster_id`, pick canonical via existing `resolve.canonical_name`, so
  `resolved_graph` is reused unchanged.

**Deliverables.**
- `uv add splink` (bundles DuckDB; no server).
- `resolve_splink.py` (pure logic) + `resolve_splink_module.py` (Hamilton node layer,
  same logic/node split as resolve): `splink_records` → `splink_linker` (EM-trained) →
  `splink_pairwise` → `splink_clusters` → a **`clusters` node** (`list[Cluster]`).
  Run it as its own flow: `run(resolver_module=resolve_splink_module,
  resolution_mode="splink", match_probability_threshold=0.9)`.
- `match_probability_threshold` config dim (peer to `tau_candidate`).
- **Shared comparison metrics** (improve BOTH modes, not just Splink):
  `bcubed_metrics` (entity-level B-cubed precision/recall/f1 — the partition, not just
  pairwise) and `llm_calls` scalar (`0` for splink; `len(pair_verdicts)` for offline).
- Explainability: `out/splink_waterfall.html` + match-weights table logged to MLflow
  (Splink's analog to the LLM traces).

**Note — validate on bigger data later.** Splink's EM estimates m/u from pairwise
comparisons; on this 13-node toy corpus the trained weights are *illustrative, not
trustworthy* (too few pairs). Build and run the flow now, but treat the Splink↔LLM
comparison as indicative until re-run on a larger corpus. The flow accepts a different
`corpus_dir` / `entities_path`, so swapping in bigger data needs no code change.

**Tests.**
- *unit:* `splink_records` one-row-per-node + unique `unique_id` gate; the splink
  clustering is a **total + disjoint** partition of node ids.
- *unit:* `bcubed_metrics` exact on `mini_graph` (hand-computed B-cubed); `llm_calls`
  is `0` on the splink path.
- *unit:* the splink `clusters` node returns `list[Cluster]` that `resolved_graph` consumes
  (reuse the T8 shared-tail assertions — no dangling edges, valid partition).
- *integration:* a `splink` run and an `offline` run both appear as comparable MLflow
  rows (pairwise + B-cubed + `llm_calls` + `lookalike_preserved`); waterfall attached.
- *integration (the gate):* the look-alike hard gate holds for splink — TF-adjusted
  Jaro-Winkler keeps `P-101`/`P-102` in different clusters.

**Residuals (1–2 min, per the doc's appendix):** confirm `predict()`/`cluster_*`
column names, `comparison_library` class names (`dir(cl)`), and the waterfall accessor.

---

## T12 — Online-semantic resolver (incremental mode)

**Objective.** The greedy incremental clusterer: process nodes one at a time in
`ingest_order`, matching each against clusters-so-far and committing link/mint
immediately. Exhibits the **order-dependence** the reference describes.

**Deliverables.** A separate `resolve_online_module` defining a `clusters` node:
greedy incremental matching that maintains cluster representatives, reuses the shared
embed/verify logic, orders by `ingest_order`. Run as its own flow
(`run(resolver_module=resolve_online_module, resolution_mode="online")`).

**Tests.**
- *unit:* with a **stub verifier**, on `mini_graph` the online pass produces a valid
  partition (every node once — gate), variants linked, lookalike minted separately.
- *unit (the headline):* **order-dependence** — run online with two `ingest_order`s
  crafted so a greedy early link differs; assert the resulting cluster sets **differ**.
- *unit:* representative update is correct (linking a node updates the cluster it
  was matched into; subsequent nodes can match the updated representative).
- *integration:* real verifier, small corpus — online recall ≤ offline recall on the
  same inputs (greedy misses some global links).

---

## T13 — Hybrid resolver (online + offline reconciliation)

**Objective.** Online ingest for freshness, then an offline reconciliation pass that
re-clusters globally — repairing online's path-dependence.

**Deliverables.** A separate `resolve_hybrid_module` defining a `clusters` node: runs
the online pass, then applies the offline connected-components reconciler over its
members. Run as its own flow (`run(resolver_module=resolve_hybrid_module,
resolution_mode="hybrid")`).

**Tests.**
- *unit (the headline):* **convergence** — for the two `ingest_order`s where online
  diverged (T12), hybrid yields **identical** cluster sets, equal to the offline
  result on the same nodes.
- *unit:* hybrid partition is valid (every node once — gate); recall ≥ online recall.
- *integration:* on the small corpus, hybrid F1 ≈ offline F1 and ≥ online F1.

---

## Done-when (acceptance, maps to design §13)

1. `uv run pytest` green (unit); `uv run pytest -m integration` green with
   `OPEN_ROUTER_KEY` set.
2. Raw graph is demonstrably fragmented (`raw` recall low, `raw_node_count` high);
   resolved graph: recall ↑, `resolved_node_count` ↓, precision holds.
3. `lookalike_preserved == 1` on the chosen run; the hard gate provably halts a
   planted over-merge (T10 test).
4. **Mode story proven:** offline order-independent, online order-dependent (T12),
   hybrid converges to offline (T13); `==` topology order-independent (T2.5).
5. A simple KG visualization renders raw + resolved graphs (T8.5).
6. Every node has a gate; each config is one MLflow run with params, metrics, viz,
   fragmentation table, and verifier traces attached.

## Task dependency order

```
T0 → T1 → T2 → T2.5                         (T2.5 = library matching probe, integration)
        ├→ T3 → T4 ─┐
        ├→ T5 ──────┼→ T6 ─┐
        └→ T7 ←─────┘      ├→ T8 ─┬→ T8.5 ─┐
                    T7 ────┘      │         ├→ T9 → T10 → T11 → T11.5 → T12 → T13
                                  ├→ T11.5 ─┤   (Splink: peer clusterer, reuses T8 shared tail)
                                  ├→ T12 ───┤
                                  └→ T13 ───┘   (T13 depends on T12; all reuse the T8 shared tail)
```

T7 (metrics) depends only on fixtures (T1) — build and test it early, before the
LLM-touching tasks, since it encodes the experiment's correctness criteria. T8 ships
the shared resolve→cluster tail that **T11.5 (Splink), T12 (online), and T13 (hybrid)
all reuse** — so build offline first. T11.5 is sequenced before the modes: it's a
classical *baseline* for the same offline resolution, not a new paradigm, and its
B-cubed + `llm_calls` metrics sharpen every later mode comparison too.
