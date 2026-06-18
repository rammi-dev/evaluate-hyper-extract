"""Shared data contracts for the pipeline.

These are the types that flow between modules (extract → resolve → metrics → report).
Defined once here so fixtures, the library adapter, and the resolver all agree.

A `Node.id` is the stable identifier. In the *raw* graph produced by the library the
key is the entity name (design A.2), so for raw nodes `id == name` and edges
reference nodes by that name. The external resolver assigns each cluster a canonical
id and rewrites edge endpoints to it.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Doc(BaseModel):
    """One input document (UTF-8 text; design A.3 — txt/md only)."""

    name: str
    text: str


class Node(BaseModel):
    """An extracted entity node (raw or resolved)."""

    id: str
    name: str
    type: str = ""
    scope: str = ""  # plant / unit / area — the namespace that splits lookalikes
    description: str = ""
    evidence: str = ""  # source sentence (provenance)
    aliases: list[str] = Field(default_factory=list)  # surface forms merged into this node (resolved only)


class Edge(BaseModel):
    """A relation. `source`/`target` are node ids (raw graph: node names)."""

    source: str
    target: str
    relation: str = ""


class Graph(BaseModel):
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def node_ids(self) -> set[str]:
        return {n.id for n in self.nodes}


class CanonicalEntity(BaseModel):
    """One real-world entity and every surface form it should resolve to."""

    id: str
    canonical_name: str
    type: str = ""
    scope: str = ""
    variants: list[str] = Field(default_factory=list)


class GroundTruth(BaseModel):
    """The evaluation manifest (design §2.2)."""

    canonical_entities: list[CanonicalEntity] = Field(default_factory=list)
    # pairs of canonical-entity ids that look alike but MUST stay in distinct clusters
    lookalike_pairs: list[tuple[str, str]] = Field(default_factory=list)


class Cluster(BaseModel):
    """A resolved group of raw node ids (one real entity)."""

    id: str
    node_ids: list[str] = Field(default_factory=list)
    canonical_name: str = ""
