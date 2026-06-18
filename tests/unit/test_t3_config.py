"""T3 acceptance: Config validates; the gates fire on bad input."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eval_hyper_extract.config_module import Config, config


def _valid_kwargs() -> dict:
    return dict(
        llm_model="openai/gpt-4o-mini",
        tau_candidate=0.55,
        embed_model="BAAI/bge-m3",
        resolution_mode="offline",
        ingest_order=0,
        template_path="data/template.yaml",
        corpus_dir="data/corpus",
        entities_path="data/entities.json",
        out_dir="out",
    )


def test_valid_config_builds() -> None:
    cfg = config(**_valid_kwargs())
    assert isinstance(cfg, Config)
    assert cfg.tau_candidate == 0.55
    assert cfg.resolution_mode == "offline"


@pytest.mark.parametrize("bad_tau", [0.0, -0.1, 1.5])
def test_tau_out_of_range_raises(bad_tau: float) -> None:
    with pytest.raises(ValidationError, match="tau_candidate"):
        Config(llm_model="x", tau_candidate=bad_tau)


def test_tau_boundary_one_ok() -> None:
    assert Config(llm_model="x", tau_candidate=1.0).tau_candidate == 1.0


def test_invalid_resolution_mode_raises() -> None:
    with pytest.raises(ValidationError):
        Config(llm_model="x", resolution_mode="batch")  # not in {offline,online,hybrid}


def test_missing_required_key_raises() -> None:
    with pytest.raises(ValidationError, match="llm_model"):
        Config(tau_candidate=0.55)  # llm_model omitted
