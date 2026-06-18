"""T2 acceptance: real data assets are valid, non-vacuous, and corpus-consistent."""

from __future__ import annotations

import json
from pathlib import Path

from eval_hyper_extract.config_module import Config
from eval_hyper_extract.extract_module import template, validate_template_cfg
from eval_hyper_extract.schema import GroundTruth

DATA = Path(__file__).parents[2] / "data"


def _manifest() -> GroundTruth:
    return GroundTruth.model_validate(json.loads((DATA / "entities.json").read_text()))


def test_manifest_valid_with_lookalike() -> None:
    gt = _manifest()
    assert any(len(e.variants) >= 2 for e in gt.canonical_entities)
    assert len(gt.lookalike_pairs) >= 1


def test_every_variant_appears_in_corpus() -> None:
    """Each manifest variant must appear verbatim in some doc (so fragmentation is real)."""
    gt = _manifest()
    corpus = "\n".join(p.read_text(encoding="utf-8") for p in (DATA / "corpus").glob("*.md"))
    for entity in gt.canonical_entities:
        for variant in entity.variants:
            assert variant in corpus, f"variant {variant!r} missing from corpus"


def test_lookalike_pairs_are_genuine() -> None:
    """A lookalike pair = distinct entities (different name) of the SAME type."""
    gt = _manifest()
    by_id = {e.id: e for e in gt.canonical_entities}
    for x, y in gt.lookalike_pairs:
        ex, ey = by_id[x], by_id[y]
        assert ex.canonical_name != ey.canonical_name  # distinct assets
        assert ex.type == ey.type  # but same type → genuinely look alike


def test_template_loads_and_validates() -> None:
    cfg = template(Config(llm_model="x", template_path=str(DATA / "template.yaml")))
    validate_template_cfg(cfg)
    assert cfg.type == "graph"
    assert cfg.identifiers.entity_id == "name"  # the match key under study
