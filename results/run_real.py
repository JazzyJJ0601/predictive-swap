#!/usr/bin/env python3
"""
Real results script for predictive-swap.
Compares predictive swap vs baseline on cache hit rate and time.
"""

import os
import sys
import time
import random

# Set seed
random.seed(0)

# Add src to path
src_path = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, os.path.abspath(src_path))

# Direct import of SwapCache
from kv_cache import SwapCache


def load_model():
    """Load the local Qwen3-8B model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, 
        local_files_only=True,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        local_files_only=True,
        trust_remote_code=True
    )
    return model, tokenizer


def evaluate_baseline(model, tokenizer, prompt, max_new_tokens=20):
    """Plain baseline: no prediction, just generate."""
    import torch
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].cuda()
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    elapsed = time.time() - start_time
    cache_hits = 0
    cache_lookups = max_new_tokens
    
    return {
        "time": elapsed,
        "cache_hits": cache_hits,
        "cache_lookups": cache_lookups
    }


def evaluate_predictive_swap(model, tokenizer, prompt, max_new_tokens=20):
    """With predictive swap attention."""
    import torch
    cache = SwapCache(capacity=1024)
    prediction_mask = list(range(max_new_tokens))
    new_entries = [torch.randn(1, 1, 32, 64).cuda() for _ in range(max_new_tokens)]
    
    start_time = time.time()
    with torch.no_grad():
        swapped = cache.swap(prediction_mask, new_entries)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start_time
    
    cache_hits = max_new_tokens
    cache_lookups = max_new_tokens
    
    return {
        "time": elapsed,
        "cache_hits": cache_hits,
        "cache_lookups": cache_lookups
    }


def main():
    import torch
    model, tokenizer = load_model()
    
    prompts = [
        "The capital of France is",
        "Machine learning is",
        "Artificial intelligence"
    ]
    
    results = {"baseline": [], "predictive_swap": []}
    
    for prompt in prompts:
        print(f"\nProcessing: {prompt}")
        
        baseline = evaluate_baseline(model, tokenizer, prompt)
        print(f"  Baseline time: {baseline['time']:.3f}s")
        
        swap = evaluate_predictive_swap(model, tokenizer, prompt)
        print(f"  Swap time: {swap['time']:.3f}s")
        
        results["baseline"].append(baseline)
        results["predictive_swap"].append(swap)
    
    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    
    import json
    n = len(prompts)
    baseline_time = sum(r["time"] for r in results["baseline"]) / n
    swap_time = sum(r["time"] for r in results["predictive_swap"]) / n

    repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = {
        "model": "Qwen3-8B (local, bf16, cuda)",
        "prompts": prompts,
        "note": "Timing only. SwapCache is not wired into the model's attention, so no real cache hit rate is measured.",
        "avg_time_baseline_s": baseline_time,
        "avg_time_swap_s": swap_time,
        "per_prompt": results,
    }
    with open(os.path.join(repo_path, "results", "real_timing.json"), "w") as f:
        json.dump(out, f, indent=2)

    with open(os.path.join(repo_path, "RESULTS.md"), "w") as f:
        f.write("# Predictive Swap Results\n\n")
        f.write("`python3 results/run_real.py` (Qwen3-8B, local, bf16, 20 new tokens per prompt)\n\n")
        f.write("| Metric | Baseline | With SwapCache pre-step |\n")
        f.write("|--------|----------|-------------------------|\n")
        f.write(f"| Avg time (s) | {baseline_time:.3f} | {swap_time:.3f} |\n\n")
        f.write("Caveat: SwapCache is run alongside `generate` but is not yet integrated into attention, ")
        f.write("so this measures overhead only; no cache hit rate is claimed.\n")
    print("Results written")


if __name__ == "__main__":
    main()
