# Predictive Swap Results

`python3 results/run_real.py` (Qwen3-8B, local, bf16, 20 new tokens per prompt)

| Metric | Baseline | With SwapCache pre-step |
|--------|----------|-------------------------|
| Avg time (s) | 0.656 | 0.474 |

Caveat: SwapCache is run alongside `generate` but is not yet integrated into attention, so this measures overhead only; no cache hit rate is claimed.
The baseline's first prompt includes GPU warm-up, so its average is inflated; the two are effectively equal.
