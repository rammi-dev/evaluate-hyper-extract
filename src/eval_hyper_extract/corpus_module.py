"""Corpus + ground-truth loading, each behind a gate.

Reads UTF-8 Markdown/text docs (design A.3 — no PDF) and the evaluation manifest.
"""

from __future__ import annotations

import json
from pathlib import Path

from eval_hyper_extract.config_module import Config
from eval_hyper_extract.schema import Doc, GroundTruth


def corpus_docs(config: Config) -> list[Doc]:
    """Load every `*.md` / `*.txt` doc from `corpus_dir`."""
    root = Path(config.corpus_dir)
    paths = sorted(p for p in root.glob("*") if p.suffix.lower() in (".md", ".txt"))
    docs = [Doc(name=p.name, text=p.read_text(encoding="utf-8")) for p in paths]

    assert docs, f"no .md/.txt documents found in {root}"
    assert all(d.text.strip() for d in docs), "a corpus document is empty"
    return docs


def ground_truth(config: Config) -> GroundTruth:
    """Load + validate the manifest; require canonical entities AND a lookalike pair."""
    data = json.loads(Path(config.entities_path).read_text(encoding="utf-8"))
    gt = GroundTruth.model_validate(data)

    assert gt.canonical_entities, "manifest has no canonical_entities"
    assert gt.lookalike_pairs, "manifest needs >=1 lookalike pair (else metrics are vacuous)"
    return gt
