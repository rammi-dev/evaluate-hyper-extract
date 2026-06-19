"""Configuration — the runtime knobs, validated. Each becomes an MLflow param.

`Config` carries its own validation (tau range, mode enum), so constructing it IS the
gate. The `config` Hamilton node assembles it from the scalar `inputs=` (which the
MLflow adapter auto-logs as params).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, field_validator

ResolutionMode = Literal["offline", "splink", "online", "hybrid"]


class Config(BaseModel):
    llm_model: str  # required — no sensible default (a live OpenRouter id)
    tau_candidate: float = 0.55
    embed_model: str = "BAAI/bge-m3"
    resolution_mode: ResolutionMode = "offline"
    ingest_order: int = 0  # permutation seed for online/hybrid; ignored by offline
    template_path: str = "data/template.yaml"
    corpus_dir: str = "data/corpus"
    entities_path: str = "data/entities.json"
    out_dir: str = "out"

    @field_validator("tau_candidate")
    @classmethod
    def _tau_in_unit_interval(cls, v: float) -> float:
        if not (0.0 < v <= 1.0):
            raise ValueError(f"tau_candidate must be in (0, 1], got {v}")
        return v


def config(
    llm_model: str,
    tau_candidate: float,
    embed_model: str,
    resolution_mode: str,
    ingest_order: int,
    template_path: str,
    corpus_dir: str,
    entities_path: str,
    out_dir: str,
) -> Config:
    """Assemble + validate the run config from inputs (validation = the gate).

    Note: the *library* match key is NOT a config input — it is read from the
    template (`extract_module.library_key`), the single source of truth (design A.2).
    """
    return Config(
        llm_model=llm_model,
        tau_candidate=tau_candidate,
        embed_model=embed_model,
        resolution_mode=resolution_mode,
        ingest_order=ingest_order,
        template_path=template_path,
        corpus_dir=corpus_dir,
        entities_path=entities_path,
        out_dir=out_dir,
    )
