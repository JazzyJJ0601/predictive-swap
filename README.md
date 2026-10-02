# Predictive KV swapping

When a long KV cache lives in CPU memory, each layer can only attend to the pages that are already on the GPU.
Quest-style page selection picks the right pages, but it needs the layer's own query, so the fetch can't start
until the layer is already running. This repo predicts that query **one layer early**: it applies layer *l*'s
query projection (norm, `q_proj`, `q_norm`, RoPE) to the hidden state entering layer *l−1*. The page choice is
then ready while layer *l−1* computes, so the copy from CPU can overlap with useful work.

## Result (Qwen3-8B, real model)

16 WikiText-2 test documents of 4,096 tokens, perplexity scored on tokens 1,024–4,095, so most of the context
is outside the GPU budget. Each query may attend to C entries per KV head: the first page (attention sinks), the
last 128 tokens, and k pages of 16 tokens. Lower perplexity is better. Recall is the share of full attention's
softmax mass that the kept entries cover.

| Policy | C=512 ppl | recall | C=1024 ppl | recall |
|---|---:|---:|---:|---:|
| full attention (all 4,096) | 8.72 | 1.000 | 8.72 | 1.000 |
| sinks | 9.99 | 0.732 | 9.22 | 0.837 |
| random | 10.05 | 0.684 | 9.25 | 0.794 |
| prev | 9.20 | 0.811 | 8.88 | 0.901 |
| **predicted (this repo)** | **8.89** | 0.841 | **8.74** | 0.926 |
| oracle | 8.88 | 0.845 | 8.74 | 0.929 |

With a 512-entry budget (12.5% of the context), the predicted query gives perplexity 8.89 against 9.99 for StreamingLLM sinks and 9.20 for reusing the previous layer's pages, and it is within 0.02 of the oracle that uses the layer's real query (8.88). At 1,024 entries it reaches 8.74, 0.03 from full attention (8.72) and 0.002 from the oracle. Predicting one layer early costs almost nothing in page quality.

## Policies

- **full**: ordinary causal attention over all 4,096 tokens (the upper bound).
- **sinks**: StreamingLLM at the same budget: first page + the most recent C−16 tokens, no swapping.
- **random**: sinks + window + k random earlier pages.
- **prev**: sinks + window + the pages the previous layer would have chosen (reuse; no new scoring).
- **predicted** (this repo): Quest page scores (query against each page's key min/max), computed with the query
  predicted from layer *l−1*'s input. Available one layer ahead.
- **oracle**: the same scores with the layer's real query. Best choice, but only known once the layer runs.

Layers 0 and 1 use full attention under every sparse policy, as in Quest.

## Limits

- This measures **which pages to fetch**, not speed. Pages that aren't chosen are masked out of attention; there
  is no real CPU↔GPU copy kernel here, so no latency or throughput numbers are claimed.
- One model (Qwen3-8B), one dataset (WikiText-2), 16 documents of 4,096 tokens.

## Run it

```bash
python results/run_real.py      # writes results/real.json (~10 min on an RTX 3090 Ti)
python -m pytest -q tests       # page bounds, eligibility and mask tests
```

The model path is set at the top of `results/run_real.py`. Full numbers: [RESULTS.md](RESULTS.md).
