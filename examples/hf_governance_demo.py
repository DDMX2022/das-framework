"""
hf_governance_demo.py — the governance guarantees, on a REAL encoder + REAL text
--------------------------------------------------------------------------------
This is the "it's not a toy" demo (PRODUCT_PLAN.md Phase 1, first milestone). Every
governance claim DAS makes — provable isolation, hot-swap grafting, right-to-be-
forgotten deletion, a signed tamper-evident audit trail — is shown end to end on:

  * a REAL frozen pretrained encoder (sentence-transformers MiniLM, 384-d), and
  * REAL English sentences (das_text.DEMO_CORPUS), routed live.

Pipeline:  text --[frozen MiniLM]--> 384-d embedding --[StemRouter]--top1--> one
isolated LoRA expert (adapter + head on a shared frozen backbone) --> prediction,
with a SHA-256 fingerprint per expert and an HMAC-signed, hash-chained audit log.

Run:
    pip install -e ".[hf]"
    python examples/hf_governance_demo.py

First run downloads the ~90 MB MiniLM weights (then cached). CPU is fine.
"""
import torch

from das_text import TextEncoder, embed_domains, DEMO_CORPUS
from das_torch import LoRAForest, train_leaf_isolated_lora, train_router, leaf_hash
from das.audit import AuditLog

DEVICE = "cpu"
FEAT = 128          # shared-backbone feature width (on top of the frozen encoder)
RANK = 8            # LoRA rank per expert
torch.manual_seed(7)


def sample(cache, name, n):
    """Draw n (embedding, label) rows for a domain, with replacement — the real
    text corpus is small, so we resample it into minibatches (honest: a small
    real set, augmented by resampling, not synthetic noise)."""
    emb, y = cache[name]
    idx = torch.randint(0, emb.shape[0], (n,))
    return emb[idx], y[idx]


def pretrain_backbone(forest, cache, domains, steps=300):
    """Train the small backbone-on-embeddings ONCE to separate the domains, then
    freeze it for good. The genuinely pretrained part is the frozen MiniLM beneath
    it; this just shapes its 384-d output into router-friendly features."""
    tmp = torch.nn.Linear(FEAT, len(domains))
    opt = torch.optim.Adam(list(forest.backbone.parameters()) + list(tmp.parameters()), lr=1e-3)
    Xs, ds = [], []
    for d, name in enumerate(domains):
        emb, _ = cache[name]
        Xs.append(emb)
        ds.append(torch.full((emb.shape[0],), d, dtype=torch.long))
    X, d = torch.cat(Xs), torch.cat(ds)
    for _ in range(steps):
        i = torch.randint(0, len(X), (min(128, len(X)),))
        opt.zero_grad()
        torch.nn.functional.cross_entropy(tmp(forest.backbone(X[i])), d[i]).backward()
        opt.step()
    forest.freeze_backbone()


def train_router_on(forest, cache, domains):
    feats, ys = [], []
    for slot, name in enumerate(domains):
        emb, _ = cache[name]
        feats.append(forest.features(emb).detach())
        ys.append(torch.full((emb.shape[0],), slot, dtype=torch.long))
    train_router(forest, torch.cat(feats), torch.cat(ys), steps=400, lr=0.05, device=DEVICE)


def hashes(forest, names):
    return {names[i]: leaf_hash(forest.leaves[i]) for i in range(len(names))}


def route(forest, encoder, text):
    """Embed one real sentence, route it, return (domain_slot, confidence)."""
    emb = encoder.embed([text])
    with torch.no_grad():
        idx, tau, _ = forest.router(forest.features(emb))
    return int(idx[0]), float(tau[0].max())


def rule(s):
    print("\n" + "─" * 74 + f"\n{s}\n" + "─" * 74)


def main():
    rule("1 · Load the REAL frozen encoder")
    enc = TextEncoder(device=DEVICE)
    print(f"  encoder : {enc.model_name}")
    print(f"  dim     : {enc.dim}  (frozen, pretrained — never trained here)")

    active = ["legal", "medical", "finance"]
    cache = embed_domains(enc, active, DEMO_CORPUS)
    print(f"  domains : {active}  (real sentences embedded once)")

    rule("2 · Build the forest: router + isolated LoRA experts on the embeddings")
    forest = LoRAForest(enc.dim, FEAT, out_dim=2, num_leaves=len(active), rank=RANK).to(DEVICE)
    pretrain_backbone(forest, cache, active)
    for i, name in enumerate(active):
        acc = train_leaf_isolated_lora(forest, i, *sample(cache, name, 300), steps=250, device=DEVICE)
        print(f"  trained LoRA expert '{name}'  (in-domain acc {acc:.2f})")
    train_router_on(forest, cache, active)

    log = AuditLog()
    log.append("init", f"forest with experts {active}", payload=hashes(forest, active))

    rule("3 · Route REAL, unseen sentences (provenance = which expert served it)")
    probes = [
        ("legal",   "They added an uncapped indemnity that overrides our liability cap."),
        ("medical", "The patient suddenly collapsed with no pulse and isn't breathing."),
        ("finance", "A large wire was rerouted overnight to an unknown foreign account."),
    ]
    for expected, text in probes:
        slot, conf = route(forest, enc, text)
        routed = active[slot] if slot < len(active) else "?"
        ok = "✓" if routed == expected else "✗"
        print(f"  {ok} routed→{routed:8s} conf={conf:.2f}  | \"{text[:52]}…\"")

    rule("4 · GRAFT a new expert — prove the others stay byte-identical")
    before = hashes(forest, active)
    enc_cache_new = embed_domains(enc, ["code"], DEMO_CORPUS)
    cache.update(enc_cache_new)
    nid = forest.graft_leaf()
    active.append("code")
    acc = train_leaf_isolated_lora(forest, nid, *sample(cache, "code", 300), steps=250, device=DEVICE)
    train_router_on(forest, cache, active)
    after = hashes(forest, active)
    intact = all(after[n] == before[n] for n in before)
    print(f"  grafted 'code' (in-domain acc {acc:.2f}); existing experts byte-identical: {intact}")
    for n in before:
        flag = "unchanged" if after[n] == before[n] else "CHANGED!"
        print(f"    {n:8s} {before[n]} → {after[n]}  [{flag}]")
    log.append("graft", "added expert 'code'; prior experts unchanged: %s" % intact,
               payload=after)

    rule("5 · PRUNE an expert (right-to-be-forgotten) — survivors byte-identical")
    drop = "finance"
    di = active.index(drop)
    survivors_before = {active[i]: leaf_hash(forest.leaves[i]) for i in range(len(active)) if i != di}
    keep = [i for i in range(len(active)) if i != di]
    forest.leaves = torch.nn.ModuleList([forest.leaves[i] for i in keep])
    old = forest.router.gate
    new = torch.nn.Linear(old.in_features, len(keep))
    with torch.no_grad():
        new.weight.copy_(old.weight[keep]); new.bias.copy_(old.bias[keep])
    forest.router.gate = new
    active.pop(di)
    train_router_on(forest, cache, active)
    survivors_after = {n: leaf_hash(forest.leaves[active.index(n)]) for n in survivors_before}
    intact = all(survivors_after[n] == survivors_before[n] for n in survivors_before)
    print(f"  pruned '{drop}'; surviving experts byte-identical: {intact}")
    slot, conf = route(forest, enc, "A duplicate payment to an unverified supplier was flagged.")
    print(f"  a '{drop}'-type query now routes to '{active[slot] if slot < len(active) else '?'}' "
          f"(its dedicated expert is gone)")
    log.append("prune", f"removed expert '{drop}' (right-to-be-forgotten); survivors unchanged: {intact}",
               payload=survivors_after)

    rule("6 · Verify the signed, hash-chained audit trail")
    ok, idx, reason = log.verify()
    print(f"  entries: {len(log.entries)}   chain valid: {ok}   ({reason})")
    for e in log.entries:
        print(f"    [{e['sig'][:12]}] {e['event']:6s} — {e['detail']}")

    rule("Done — every guarantee above ran on a real encoder + real English text")
    print("  Isolation, grafting, deletion, and a tamper-evident audit log:")
    print("  demonstrated on MiniLM embeddings of real sentences, not synthetic vectors.")


if __name__ == "__main__":
    main()
