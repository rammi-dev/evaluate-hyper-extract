"""Extraction — the Hamilton node layer.

Builds the library `AutoGraph` from our template (design §2.3), feeds the corpus, and
maps the result into `schema.Graph`. Pure helpers live in `extract.py` (re-exported
for tests, excluded from the DAG).
"""

from __future__ import annotations

from hyperextract.utils.template_engine.parsers.loader import TemplateCfg, load_template
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from eval_hyper_extract import extract
from eval_hyper_extract.extract import assert_nonempty, to_graph, validate_template_cfg, write_graph_json  # re-exports
from eval_hyper_extract.config_module import Config
from eval_hyper_extract.schema import Doc, Graph

__all__ = [
    "assert_nonempty", "to_graph", "validate_template_cfg", "write_graph_json",
    "template", "library_key", "raw_graph",
]


def template(config: Config) -> TemplateCfg:
    """Load + validate the extraction template (validation = the gate)."""
    cfg = load_template(config.template_path)
    extract.validate_template_cfg(cfg)
    return cfg


def library_key(template: TemplateCfg) -> str:
    """The library's real match key — `identifiers.entity_id` (design A.2)."""
    key = template.identifiers.entity_id
    assert key, "template identifiers.entity_id is empty"
    return str(key)


def raw_graph(
    checked_llm: BaseChatModel,
    checked_embedder: Embeddings,
    corpus_docs: list[Doc],
    config: Config,
    template: TemplateCfg,
) -> Graph:
    """Build the AutoGraph from the template, feed the corpus, map → schema.Graph."""
    from hyperextract import Template

    graph = Template.create(
        config.template_path, "en", llm_client=checked_llm, embedder=checked_embedder
    )
    for doc in corpus_docs:
        graph.feed_text(doc.text)
    return extract.assert_nonempty(extract.to_graph(graph.nodes, graph.edges))
