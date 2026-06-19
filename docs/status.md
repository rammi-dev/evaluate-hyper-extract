# Project status

Snapshot of where `evaluate-hyper-extract` stands. For the *why/what* see
[OVERVIEW.md](OVERVIEW.md), the architecture in [design.md](design.md), and the
task-by-task plan in [implementation-plan.md](implementation-plan.md).

## Where it is

The **offline flow runs end-to-end for real** and is fully observable in MLflow.
Hyper-Extract extraction → external entity resolution (embed → candidate → LLM-verify
→ cluster) → metrics → report, every node gated.

- **Tests:** `70 passed` (deterministic core to exact numbers) + opt-in integration
  tests that hit the real LLM (`uv run pytest -m integration`).
- **Latest offline run:** 13 fragmented nodes → 10; raw recall `0.00` → resolved
  `1.00`; precision `1.00`; look-alike `P-101`/`P-102` kept distinct;
  `verifier_agreement 1.00`.
- **Runs accumulate** in `out/mlflow.db` (never deleted) for cross-config comparison.
- **Reproducible** — a LangChain SQLite LLM cache (`out/llm_cache.db`) keys every chat
  call on its prompt+model, so re-running a config gives identical results (the local
  embedder is already deterministic). Delete `out/llm_cache.db` to force fresh sampling.

## Architecture (one line)

**Separate flows, compared in MLflow.** Each resolver is its own module providing a
`clusters` node; a flow = shared modules + one resolver module
(`run(resolver_module=…)`). No in-DAG mode switch. Inputs (`corpus_dir`,
`entities_path`, `template_path`, model/threshold) are parameterized, so any flow runs
on a different/bigger dataset without code changes.

## Done

T0 scaffold · T1 fixtures · T2 data assets · T3 config · T4 clients · T5 corpus ·
T6 extract · T7 metrics · T8 offline resolver · T8.5 viz · T9 report · T10 driver +
MLflow + LLM traces + verdict validation.

## Next

- **T11** — sweep & assessment (compare runs).
- **T11.5** — Splink (Fellegi-Sunter) resolver as a separate flow — classical baseline
  vs the LLM resolver (see [splink](splink)).
- **T12 / T13** — online (incremental) and hybrid resolver flows.
- **T2.5 / T10b** — library matching-characterization + driver tests.

## Open caveats

- **Manifest vs LLM surface forms** — the LLM emits names that don't always match the
  hand-authored `data/entities.json` variants, so only labeled pairs are scored;
  reconcile before reading absolute recall as a benchmark.
- **Hamilton node cache deferred** — the clients are unpicklable, so we don't use
  Hamilton's `.with_cache()`. Reproducibility is instead handled at the LLM level (see
  "Reproducibility" above), which is what actually mattered.
- **Toy scale** — 3 docs / 13 nodes; illustrative, not a benchmark.
