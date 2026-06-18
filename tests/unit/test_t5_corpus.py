"""T5 acceptance: corpus + ground-truth loading and their gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_hyper_extract.config_module import Config
from eval_hyper_extract.corpus_module import corpus_docs, ground_truth

FIXTURE_ENTITIES = Path(__file__).parent.parent / "fixtures" / "mini_entities.json"


def _cfg(**kw) -> Config:
    return Config(llm_model="x", **kw)


def test_corpus_docs_loads_md_and_txt(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("Pump P-101 supplies the boiler.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Tank T-200 stores condensate.", encoding="utf-8")
    (tmp_path / "ignore.pdf").write_bytes(b"%PDF-1.4 binary")  # not text → skipped

    docs = corpus_docs(_cfg(corpus_dir=str(tmp_path)))
    assert len(docs) == 2
    assert {d.name for d in docs} == {"a.md", "b.txt"}
    assert all(d.text.strip() for d in docs)


def test_corpus_docs_empty_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match="no .md/.txt"):
        corpus_docs(_cfg(corpus_dir=str(tmp_path)))


def test_corpus_docs_empty_file_raises(tmp_path: Path) -> None:
    (tmp_path / "blank.md").write_text("   \n", encoding="utf-8")
    with pytest.raises(AssertionError, match="empty"):
        corpus_docs(_cfg(corpus_dir=str(tmp_path)))


def test_ground_truth_loads_valid_manifest() -> None:
    gt = ground_truth(_cfg(entities_path=str(FIXTURE_ENTITIES)))
    assert len(gt.canonical_entities) == 3
    assert gt.lookalike_pairs == [("A", "B")]


def test_ground_truth_no_lookalike_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(
        json.dumps({"canonical_entities": [{"id": "A", "canonical_name": "P-101", "variants": ["P-101"]}], "lookalike_pairs": []}),
        encoding="utf-8",
    )
    with pytest.raises(AssertionError, match="lookalike"):
        ground_truth(_cfg(entities_path=str(manifest)))


def test_ground_truth_no_entities_raises(tmp_path: Path) -> None:
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"canonical_entities": [], "lookalike_pairs": [["A", "B"]]}), encoding="utf-8")
    with pytest.raises(AssertionError, match="canonical_entities"):
        ground_truth(_cfg(entities_path=str(manifest)))
