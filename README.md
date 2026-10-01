# Predictive Swap Transformer

A novel attention mechanism that intercepts KV cache entries between decoder layers to efficiently reuse computation via predictive swapping.

## Features
- Predictive cache entry management
- Swap logic for optimized memory access
- Minimal footprint for inference acceleration


## Results

**Measured status:** Overhead only: SwapCache runs alongside generation on Qwen3-8B but is not wired into attention yet, so no speedup or hit rate is claimed.

See [RESULTS.md](RESULTS.md)
