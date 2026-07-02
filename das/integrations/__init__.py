"""
das.integrations
----------------
Adapters that slot a governed DAS expert fleet UNDER the orchestrators people
already use. DAS is not an orchestrator — it's the governed, auditable layer the
orchestrator routes *into*. Each adapter is dependency-light: the integration
target (e.g. langgraph) is imported lazily, so importing this package never
requires it.
"""
from das.integrations.langgraph_node import DASExpertNode, build_graph
from das.integrations.hierarchical_node import HierarchicalDASNode, build_hierarchical_graph

__all__ = ["DASExpertNode", "build_graph",
           "HierarchicalDASNode", "build_hierarchical_graph"]
