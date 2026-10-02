# Results

Qwen3-8B (bf16, RTX 3090 Ti). WikiText-2 test, 16 documents × 4,096 tokens; perplexity on tokens 1,024–4,095
(49,152 scored tokens). Page size 16, recent window 128, layers 0–1 full attention. Source: `results/real.json`,
produced by `results/run_real.py`. Every policy run is reported; nothing was dropped.

| Policy | C=512 ppl | recall | C=1024 ppl | recall |
|---|---:|---:|---:|---:|
| full attention (all 4,096) | 8.72 | 1.000 | 8.72 | 1.000 |
| sinks | 9.99 | 0.732 | 9.22 | 0.837 |
| random | 10.05 | 0.684 | 9.25 | 0.794 |
| prev | 9.20 | 0.811 | 8.88 | 0.901 |
| **predicted (this repo)** | **8.89** | 0.841 | **8.74** | 0.926 |
| oracle | 8.88 | 0.845 | 8.74 | 0.929 |

## Reading it

- **Predicted vs oracle.** The gap is 0.013 ppl at C=512 and 0.0015 at C=1024; recall differs by under 0.004.
  The query entering layer *l−1*, pushed through layer *l*'s query projection, picks almost the same pages as
  layer *l*'s real query.
- **Predicted vs prev.** Reusing the previous layer's choice (also available early) is clearly worse: 9.20 vs 8.89
  at C=512. Different layers want different pages; predicting the query recovers that.
- **Predicted vs sinks.** Sinks needs no swapping at all but loses 1.10 ppl at C=512 and 0.48 at C=1024.
- **Random pages** are worse than simply keeping a longer recent window (sinks).

## Not measured

No CPU↔GPU copy is performed: unchosen pages are masked. So these numbers say the early page choice is as good as
the late one; they do not measure the latency saved by overlapping the fetch. One model, one dataset.
