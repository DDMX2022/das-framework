"""
das/integrations/hierarchical_node.py
-------------------------------------
The specialty TREE as an agent node: slot a ``HierarchicalDASForest`` under a
LangGraph (or any state->state) orchestration, with TWO-LEVEL provenance and
PER-BRANCH escalation policy written into graph state.

This is the "governed multi-specialist agent" wiring: an orchestrator decomposes
a task, and each sub-task flows through this node — which branch (specialty)
claimed it, which leaf inside that branch, both confidences, and whether the
tree is confident enough to keep it LOCAL or the orchestrator should ESCALATE
that sub-task to a frontier model. Per-branch thresholds make the escalation
dial a *specialty-level* policy (react escalates at 0.9, math at 0.6) — the
tree-only capability the flat node cannot offer.

Same contract as :class:`DASExpertNode`: a plain ``callable(state) -> dict``
(LangGraph's node shape), no hard langgraph dependency, NumPy only.

HONEST NOTE: the hierarchy has no governance ControlPlane yet (that integration
is documented open in SPECIALTY_FORESTS.md), so unlike ``DASExpertNode`` this
node performs no RBAC check — put it behind the governed flat node or a gateway
if the caller's identity matters. Routing/provenance is what it adds today.
"""
import numpy as np


class HierarchicalDASNode:
    """A LangGraph-compatible node wrapping a :class:`HierarchicalDASForest`.

    Reads the query embedding from ``embedding_key``; writes under ``out_prefix``:

      ``{prefix}specialty``             — which branch (tree) claimed the input
      ``{prefix}specialty_confidence``  — the top router's weight on that branch
      ``{prefix}leaf``                  — the leaf index inside the branch
      ``{prefix}leaf_confidence``       — the branch router's weight on that leaf
      ``{prefix}prediction``            — the leaf's output (list)
      ``{prefix}decision``              — ``"local"`` or ``"escalate"``
      ``{prefix}frontier``              — the escalation target when escalating

    Escalation: the input escalates when EITHER level is unsure — a shaky top
    choice means "possibly the wrong specialty entirely", a shaky leaf choice
    means "the specialty is right but no leaf owns this". ``branch_thresholds``
    overrides the default threshold per specialty.
    """

    def __init__(self, tree, embedding_key="embedding", out_prefix="das_",
                 confidence_threshold=0.0, branch_thresholds=None,
                 frontier=None):
        self.tree = tree
        self.embedding_key = embedding_key
        self.p = out_prefix
        self.confidence_threshold = float(confidence_threshold)
        self.branch_thresholds = dict(branch_thresholds or {})
        self.frontier = frontier

    def _threshold(self, specialty):
        return float(self.branch_thresholds.get(specialty, self.confidence_threshold))

    def __call__(self, state):
        h = np.asarray(state[self.embedding_key], dtype=float).reshape(1, -1)
        row = self.tree.route_explain(h)[0]
        thr = self._threshold(row["specialty"])
        escalate = (row["specialty_confidence"] < thr
                    or row["leaf_confidence"] < thr)
        p = self.p
        return {
            f"{p}specialty": row["specialty"],
            f"{p}specialty_confidence": row["specialty_confidence"],
            f"{p}leaf": row["leaf"],
            f"{p}leaf_confidence": row["leaf_confidence"],
            f"{p}prediction": row["prediction"],
            f"{p}decision": "escalate" if escalate else "local",
            f"{p}frontier": self.frontier if escalate else None,
        }


def build_hierarchical_graph(tree, node_name="das_tree", **node_kwargs):
    """Compile a minimal LangGraph ``StateGraph`` around a
    :class:`HierarchicalDASNode` (langgraph imported lazily). Invoke with
    ``graph.invoke({"embedding": [...]})`` — the result carries the two-level
    provenance and the local/escalate decision for the orchestrator to branch on."""
    from typing import Any, TypedDict

    from langgraph.graph import END, START, StateGraph

    class TreeState(TypedDict, total=False):
        embedding: Any
        das_specialty: str
        das_specialty_confidence: float
        das_leaf: int
        das_leaf_confidence: float
        das_prediction: Any
        das_decision: str
        das_frontier: str

    node = HierarchicalDASNode(tree, **node_kwargs)
    g = StateGraph(TreeState)
    g.add_node(node_name, node)
    g.add_edge(START, node_name)
    g.add_edge(node_name, END)
    return g.compile()
