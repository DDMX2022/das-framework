"""
das/integrations/langgraph_node.py
----------------------------------
Slot a governed DAS expert fleet UNDER a LangGraph orchestration as one node.

DAS doesn't replace LangGraph; it sits beneath it. A LangGraph graph routes a
conversation across nodes — and one of those nodes can be "the governed expert
fleet": it hard-routes the query to the right isolated, auditable expert, runs
it, and (the part that matters for governance) writes the PROVENANCE — which
tenant/expert served the query, the router's confidence, the acting principal —
back into the graph state. That makes the whole orchestrated run attributable,
and RBAC denials surface as state rather than uncaught exceptions.

No hard dependency on langgraph: `DASExpertNode` is a plain
`callable(state) -> dict`, which is exactly LangGraph's node contract, so it runs
in CI with NumPy only. If langgraph IS installed, `build_graph` compiles a
minimal StateGraph around it.
"""
import numpy as np

from das.governance import AccessDenied


class DASExpertNode:
    """A LangGraph-compatible node wrapping a governed DAS ``ControlPlane``.

    Call contract (LangGraph nodes are ``state -> partial-state``): read the query
    embedding from ``embedding_key`` and the acting principal from ``actor_key``,
    route through the governed forest, and return provenance under ``out_prefix``:

      ``{prefix}eid``        — stable id of the expert that served the query
      ``{prefix}tenant``     — which tenant owns that expert
      ``{prefix}expert``     — its human name
      ``{prefix}confidence`` — the hard router's softmax weight on that choice
      ``{prefix}prediction`` — the expert's output (list)
      ``{prefix}actor``      — the principal that was checked

    If the actor lacks the ``predict`` permission, the control plane logs the
    denial (tamper-evidently) and this node writes ``{prefix}denied = True`` plus
    a reason instead of a prediction — so the graph can branch on it rather than
    crash. Set ``raise_on_denied=True`` to propagate ``AccessDenied`` instead.
    """

    def __init__(self, control_plane, embedding_key="embedding", actor_key="actor",
                 default_actor="root", out_prefix="das_", raise_on_denied=False):
        self.cp = control_plane
        self.embedding_key = embedding_key
        self.actor_key = actor_key
        self.default_actor = default_actor
        self.p = out_prefix
        self.raise_on_denied = raise_on_denied

    def __call__(self, state):
        actor = state.get(self.actor_key, self.default_actor)
        h = np.asarray(state[self.embedding_key], dtype=float).reshape(1, -1)
        p = self.p
        try:
            prov = self.cp.route_explain(actor, h)[0]
        except AccessDenied as e:
            if self.raise_on_denied:
                raise
            return {f"{p}denied": True, f"{p}denied_reason": str(e), f"{p}actor": actor}
        return {
            f"{p}eid": prov["eid"],
            f"{p}tenant": prov["tenant"],
            f"{p}expert": prov["name"],
            f"{p}confidence": prov["confidence"],
            f"{p}prediction": prov["prediction"],
            f"{p}actor": actor,
            f"{p}denied": False,
        }


def build_graph(control_plane, node_name="das_expert", **node_kwargs):
    """Compile a minimal LangGraph ``StateGraph``: ``START -> das_expert -> END``,
    wrapping ``control_plane`` in a :class:`DASExpertNode`. Requires langgraph to
    be installed (imported lazily here so the rest of the module stays NumPy-only).
    Returns the compiled graph; invoke it with ``graph.invoke({"embedding": ...,
    "actor": ...})``."""
    from typing import Any, TypedDict

    from langgraph.graph import END, START, StateGraph

    class DASState(TypedDict, total=False):
        embedding: Any
        actor: str
        das_eid: int
        das_tenant: str
        das_expert: str
        das_confidence: float
        das_prediction: Any
        das_actor: str
        das_denied: bool
        das_denied_reason: str

    node = DASExpertNode(control_plane, **node_kwargs)
    g = StateGraph(DASState)
    g.add_node(node_name, node)
    g.add_edge(START, node_name)
    g.add_edge(node_name, END)
    return g.compile()
