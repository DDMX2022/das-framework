"""
lora_growth_demo.py — the demo moment, on real adapters
-------------------------------------------------------
PLATFORM_PLAN §8: "the demo moment is the growth loop — grow a specialist
live, prove the others didn't move." This is that demo on the LoRA backend:
every expert is a LoRA adapter on the frozen MiniLM's own attention layers,
and the proof is the same byte-identity the auditor gets in the bundle.

    python examples/lora_growth_demo.py        # needs the [hf] extra, ~1 min

What it shows, in order:
  1. one spec -> a live governed transformer fleet (two tenants);
  2. real text routing to the right specialist;
  3. a NEW specialist grown live — while every existing expert's weight hash
     is streamed before/after, unchanged;
  4. the rank ladder refusing to grow a saturated seed (parsimony);
  5. the signed audit trail that recorded all of it.
"""
from das.platform import deploy

SPEC = {
    "client": "northwind-demo",
    "backend": "lora-minilm",
    "tenants": [
        {"name": "legalco", "experts": [{"name": "legal"}]},
        {"name": "medico", "experts": [{"name": "medical"}]},
    ],
}


def hashes(dep):
    return {r["name"]: dep.cp.forest.leaves[i].weight_hash()[:16]
            for i, r in enumerate(dep.cp.experts)}


def main():
    print("1) deploying a governed LoRA fleet (frozen MiniLM, adapter experts)…")
    dep = deploy(SPEC, secret="demo-secret")
    for row in dep.growth_report():
        print(f"   {row['name']:<10} tenant={row['tenant']:<8} "
              f"stage={row['stage']:<6} rank={row['rank']} params={row['params']}")

    print("\n2) real text routes to the right specialist:")
    for q in ("the arbitration clause triggered a security alert",
              "the MRI scan was blocked pending urgent review"):
        r = dep.route("root", q)
        print(f"   {q[:52]:<54} -> {r['expert']} ({r['confidence']:.2f})")

    print("\n3) growing 'finance' LIVE — watch the others not move:")
    before = hashes(dep)
    for name, h in before.items():
        print(f"   before  {name:<10} {h}…")
    dep.grow("root", "medico", "finance", stage="seed")
    after = hashes(dep)
    for name, h in after.items():
        mark = "" if name not in before else (
            "  UNCHANGED" if before[name] == h else "  ** MOVED **")
        print(f"   after   {name:<10} {h}…{mark}")
    assert all(after[n] == before[n] for n in before), "isolation violated!"

    print("\n4) the parsimony gate — a saturated seed is refused more capacity:")
    res = dep.germinate("root", dep.cp.experts[-1]["eid"], target_acc=0.85)
    print(f"   action={res['action']}  stage={res.get('stage')}  "
          f"accuracy={res.get('accuracy'):.3f}  (target 0.85)")

    print("\n5) and the audit chain saw everything:")
    v = dep.verify()
    print(f"   verified={v['ok']}  entries={v['entries']}")
    for e in dep.cp.read_audit("root", n=4):
        print(f"   [{e['seq']}] {e['event']}: {e['detail'][:76]}")


if __name__ == "__main__":
    main()
