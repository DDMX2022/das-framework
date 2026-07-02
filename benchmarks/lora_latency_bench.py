"""
lora_latency_bench.py
---------------------
What does the LoRA expert path cost per query? (PLATFORM_PLAN §12 step 4 —
turning the "latency SLA" honest-gap line from an absence into a number.)

Three stages of the query path, timed separately over repeated runs:

  * route   — embed through the bare frozen MiniLM + NumPy router argmax
              (what EVERY query pays, adapter or not);
  * predict rank 0 — route + head over the frozen embedding (a seed expert);
  * predict rank 8 — route + a second encoder pass with the adapter hooks
              live (the worst-case adapter expert: two full forwards).

Run of 2026-07-02 (Apple-silicon CPU, batch 1 / 16, 30 repeats): routing is
the floor at ~7 ms p50 for a single query; a SEED expert costs exactly that
(its head reuses the routing embedding — measured before that reuse it paid a
second encoder pass, ~14 ms); an ADAPTED expert doubles it (~14 ms — the
adapter genuinely needs its own encoder pass, and the hook overhead vs rank
is noise). Batch 16 amortizes to ~1 ms/text (seed) and ~2.4 ms/text
(adapted). THE honest caveat: this is MiniLM-L6 on CPU with short sentences —
a real SLA at scale (bigger encoder, GPU, concurrency) is design-partner
territory and stays open in §11.

Needs the [hf] extra + cached MiniLM. ~1 min.
"""
import time

import numpy as np

from das.platform.lora_expert import (
    MiniLMLoRABackbone,
    MiniLMLoRAForest,
    TopicRiskTextTeacher,
)

REPEATS = 30
QUERY = "the arbitration clause triggered a security alert"


def timed(fn, repeats=REPEATS):
    for _ in range(3):                      # warm-up
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return np.percentile(times, 50), np.percentile(times, 95)


def main():
    backbone = MiniLMLoRABackbone.cached()
    lessons = TopicRiskTextTeacher().generate("finance", n_train=64, n_eval=16)

    rows = []
    for batch in (1, 16):
        texts = ([QUERY] if batch == 1
                 else list(lessons.texts_train[:16]))
        f0 = MiniLMLoRAForest(backbone, num_leaves=2, stage=0)
        f8 = MiniLMLoRAForest(backbone, num_leaves=2, stage=8)
        for label, fn in [
            ("route only", lambda: f0.router.route(f0.embed(texts))),
            ("predict rank 0", lambda: f0.predict(texts)),
            ("predict rank 8", lambda: f8.predict(texts)),
        ]:
            p50, p95 = timed(fn)
            rows.append({"batch": batch, "path": label,
                         "p50_ms": p50, "p95_ms": p95,
                         "ms_per_text": p50 / batch})

    print(f"{'batch':>6}  {'path':<16}{'p50 ms':>8}{'p95 ms':>8}{'ms/text':>9}")
    for r in rows:
        print(f"{r['batch']:>6}  {r['path']:<16}{r['p50_ms']:>8.1f}"
              f"{r['p95_ms']:>8.1f}{r['ms_per_text']:>9.1f}")
    print("\nreading: every query pays the routing embed; an adapted expert "
          "adds one more encoder pass (with hooks). CPU MiniLM numbers — a "
          "real SLA at scale is design-partner territory (PLATFORM_PLAN §11).")


if __name__ == "__main__":
    main()
