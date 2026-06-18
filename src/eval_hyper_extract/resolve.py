"""Pure entity-resolution logic (NOT a Hamilton module).

Lives apart from `resolve_module` so the node layer stays thin and Hamilton doesn't
try to turn these helpers into nodes. Imported by `resolve_module` (the DAG layer),
the online/hybrid modes (T12/T13), and tests.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_core.exceptions import OutputParserException
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, ValidationError

from eval_hyper_extract.schema import Cluster, Edge, Graph, Node

logger = logging.getLogger(__name__)

# Only malformed/truncated structured output is recoverable here. Network/auth/
# timeout errors must propagate — the run should fail loudly, not silently mis-resolve.
_PARSE_ERRORS = (OutputParserException, ValidationError, json.JSONDecodeError)


# --------------------------------------------------------------------------- types

@dataclass
class NodeEmbeddings:
    ids: list[str]
    vectors: np.ndarray  # (n, d), L2-normalized rows


@dataclass
class Pair:
    a: str
    b: str
    sim: float


@dataclass
class ConfirmedPair:
    a: str
    b: str
    reason: str = ""


@dataclass
class PairVerdict:
    """The verifier's decision on one candidate pair (kept for ALL pairs, validation)."""

    a: str
    b: str
    same: bool
    reason: str = ""


class Verdict(BaseModel):
    same: bool
    reason: str = ""


# ------------------------------------------------------------------- shared signal

def embed_text(node: Node) -> str:
    """Enriched context for resolution (design §4.0) — NOT the match key."""
    return f"{node.name} | {node.type} | {node.scope} | {node.description}"


def embed_nodes(nodes: list[Node], embedder: Embeddings) -> NodeEmbeddings:
    """Embed each node's enriched text; return L2-normalized rows aligned to `ids`."""
    ids = [n.id for n in nodes]
    raw = np.asarray(embedder.embed_documents([embed_text(n) for n in nodes]), dtype=float)
    if raw.ndim != 2 or raw.shape[0] != len(nodes):
        raise AssertionError(f"embedding matrix shape {raw.shape} != ({len(nodes)}, d)")
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return NodeEmbeddings(ids=ids, vectors=raw / norms)


def candidate_pairs(node_embeddings: NodeEmbeddings, tau_candidate: float) -> list[Pair]:
    """All node pairs with cosine >= tau (rows are normalized, so cosine = dot)."""
    ids, v = node_embeddings.ids, node_embeddings.vectors
    if len(ids) < 2:
        return []
    sims = np.clip(v @ v.T, -1.0, 1.0)
    out: list[Pair] = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if sims[i, j] >= tau_candidate:
                out.append(Pair(ids[i], ids[j], float(sims[i, j])))
    return out


def verify_pair(llm: BaseChatModel, a: Node, b: Node, *, retries: int = 1) -> Verdict:
    """LLM judges whether two nodes are the same entity (the precision backstop).

    Only *parse* failures (truncated / malformed structured output, a known LLM
    flakiness) are handled: retry once, then fall back to "not same" — the
    precision-preserving default that never fabricates a merge from a failed call,
    and which is recorded in `reason`. Transport errors (network/auth/timeout)
    deliberately propagate so the run fails loudly.
    """
    chain = llm.with_structured_output(Verdict)
    prompt = (
        "You decide whether two extracted entities are the SAME real-world asset.\n"
        "Rules:\n"
        "- Differing equipment tag numbers (e.g. P-101 vs P-102) are DIFFERENT assets.\n"
        "- A differing scope (plant/unit/area) means DIFFERENT assets even if the tag matches.\n"
        "- Surface variants / coreference of the same tag and scope are the SAME.\n"
        "Keep `reason` to one short clause (<= 12 words).\n\n"
        f"A: name={a.name!r} type={a.type!r} scope={a.scope!r} desc={a.description!r} evidence={a.evidence!r}\n"
        f"B: name={b.name!r} type={b.type!r} scope={b.scope!r} desc={b.description!r} evidence={b.evidence!r}\n"
    )
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return chain.invoke(prompt)
        except _PARSE_ERRORS as exc:
            last_exc = exc
            logger.warning("verify_pair parse failure (attempt %d) for %r vs %r: %s", attempt + 1, a.name, b.name, exc)
    return Verdict(same=False, reason=f"unparseable verifier output after {retries + 1} attempts: {last_exc}")


def verify_all_pairs(candidates: list[Pair], nodes_by_id: dict[str, Node], verify) -> list[PairVerdict]:
    """Verdict for EVERY candidate pair (same and different) — the audit trail."""
    return [
        PairVerdict(p.a, p.b, (v := verify(nodes_by_id[p.a], nodes_by_id[p.b])).same, v.reason)
        for p in candidates
    ]


def confirm_pairs(candidates: list[Pair], nodes_by_id: dict[str, Node], verify) -> list[ConfirmedPair]:
    """Keep only candidate pairs the verifier confirms as the same entity."""
    return [
        ConfirmedPair(vr.a, vr.b, vr.reason)
        for vr in verify_all_pairs(candidates, nodes_by_id, verify)
        if vr.same
    ]


# ----------------------------------------------------------------------- clustering

def connected_components(node_ids: list[str], pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Union-find over confirmed-same pairs; singletons kept."""
    parent = {nid: nid for nid in node_ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        parent[find(a)] = find(b)

    groups: dict[str, list[str]] = {}
    for nid in node_ids:
        groups.setdefault(find(nid), []).append(nid)
    return list(groups.values())


_DIGIT = re.compile(r"\d")
# an equipment tag like P-101 / P 101 / P101 / T-200A — short, letter(s)+digits
_TAG = re.compile(r"^[A-Za-z]{1,4}[-_ ]?\d{2,4}[A-Za-z]?$")


def canonical_name(names: list[str]) -> str:
    """Pick the canonical name for a cluster, preferring a clean equipment tag.

    Order: a tag pattern (`P-101`) beats a descriptive phrase (`Unit 2 feedwater
    pump`); then any digit-bearing name; then the longer/lexically-greater form
    (so `P-101` wins over `P101`).
    """
    def key(name: str) -> tuple:
        s = name.strip()
        return (bool(_TAG.match(s)), bool(_DIGIT.search(s)), len(s), s)

    return max(names, key=key)


def build_clusters(nodes: list[Node], confirmed: list[ConfirmedPair]) -> list[Cluster]:
    """Connected components → clusters with a canonical name/id each."""
    by_id = {n.id: n for n in nodes}
    components = connected_components([n.id for n in nodes], [(c.a, c.b) for c in confirmed])
    clusters: list[Cluster] = []
    for member_ids in components:
        canonical = canonical_name([by_id[i].name for i in member_ids])
        clusters.append(Cluster(id=canonical, node_ids=member_ids, canonical_name=canonical))
    return clusters


def rewrite_graph(raw_graph: Graph, clusters: list[Cluster]) -> Graph:
    """Collapse to one node per cluster; redirect edges, drop self-edges, dedupe."""
    by_id = {n.id: n for n in raw_graph.nodes}
    redirect: dict[str, str] = {}
    nodes: list[Node] = []
    for c in clusters:
        members = [by_id[i] for i in c.node_ids]
        for i in c.node_ids:
            redirect[i] = c.id
        descriptions = sorted({m.description for m in members if m.description})
        rep = next((m for m in members if m.name == c.canonical_name), members[0])
        nodes.append(
            Node(
                id=c.id,
                name=c.canonical_name,
                type=rep.type,
                scope=rep.scope,
                description="; ".join(descriptions),
                evidence=rep.evidence,
                aliases=sorted({m.name for m in members}),  # every surface form this node absorbed
            )
        )

    seen: set[tuple[str, str, str]] = set()
    edges: list[Edge] = []
    for e in raw_graph.edges:
        s, t = redirect.get(e.source, e.source), redirect.get(e.target, e.target)
        if s == t:
            continue
        sig = (s, t, e.relation)
        if sig in seen:
            continue
        seen.add(sig)
        edges.append(Edge(source=s, target=t, relation=e.relation))

    return Graph(nodes=nodes, edges=edges)
