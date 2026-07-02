# Specialty Forests Suggestion

## Question

Can we create a tree for a speciality, such as a tree for React.js, a math forest,
an alphabet tree, or a physics forest?

## Suggestion

Yes. DAS can represent specialty-focused expert groups, but the current codebase
implements a flat forest:

```text
Input -> Router -> One expert leaf
```

That already supports domain-specific forests such as:

```text
React.js Forest
  -> Hooks expert
  -> Components expert
  -> State management expert
  -> Next.js expert
  -> Testing expert

Math Forest
  -> Algebra expert
  -> Calculus expert
  -> Probability expert
  -> Geometry expert

Physics Forest
  -> Mechanics expert
  -> Electromagnetism expert
  -> Quantum expert
  -> Thermodynamics expert
```

Each expert can remain isolated, hot-swappable, auditable, and removable without
changing the other experts.

For a true specialty tree, DAS would need a hierarchical layer on top of the
current `DASForest`:

```text
Global Router
  -> React.js Forest
       -> Hooks expert
       -> Components expert
       -> Performance expert

  -> Math Forest
       -> Algebra expert
       -> Calculus expert
       -> Statistics expert

  -> Physics Forest
       -> Mechanics expert
       -> Optics expert
       -> Quantum expert
```

The clean implementation would be a `HierarchicalDASForest` or `ForestOfForests`
wrapper:

```text
SpecialtyRouter
  chooses: react | math | physics | alphabet

SubRouter
  chooses the specific expert inside that specialty
```

Example:

```text
Question: Why does useEffect run twice in React strict mode?

Top router: React.js
React router: Hooks expert
Answer comes from: react/hooks expert
Audit provenance:
  specialty = react
  expert = hooks
  confidence = 0.94
```

This fits DAS well because each specialty forest can preserve the existing
governance guarantees: isolation, deletion, provenance, and auditability.

## Status: built

`HierarchicalDASForest` ([das/hierarchy.py](../das/hierarchy.py)) implements the
wrapper: named `DASForest` branches under a specialty router, with two-level
provenance (`specialty -> expert -> prediction`, both confidences), branch-level
graft/prune with byte-identical survivor proofs, and — the tree-only guarantee —
**routing isolation**: grafting inside one branch provably leaves every other
branch's router (and the top router) byte-identical, where the flat forest
retrains one shared router over everyone's data on every graft.

See [examples/hierarchy_demo.py](../examples/hierarchy_demo.py) and
[tests/test_hierarchy.py](../tests/test_hierarchy.py). Measured honestly in
[benchmarks/hierarchy_bench.py](../benchmarks/hierarchy_bench.py): routing
accuracy TIES a flat router at adequate training budget (and loses when
budget-starved) — the wins are active routing compute (`d·(K+M)` vs `d·(K·M)`),
routing isolation, branch-level policy, and two-level provenance. Governance /
platform integration (a hierarchical ControlPlane) remains open.
