"""Real Qwen3-8B run: which KV pages must sit on the GPU when the full cache lives in CPU memory?

The whole KV cache is kept (off-GPU in a real system); each query may only attend to a GPU budget of C
entries per KV head: the first page (attention sinks) + the last W tokens + k pages of P tokens chosen by a
policy. Pages not chosen are simply not attended. Perplexity is scored on tokens 1024..4095 of 4096-token
WikiText-2 test documents (16 docs), so most of the context is outside the budget.

  full       ordinary causal attention (upper bound)
  sinks      StreamingLLM at the same budget: first page + last C-P tokens, no swapping
  random     sinks + window + k random earlier pages
  prev       sinks + window + the pages the previous layer's oracle chose (reuse, no new scoring)
  oracle     Quest-style: score every page with the layer's own query against per-page key min/max, take the
             top k. Needs the query first, so the fetch cannot start before the layer runs.
  predicted  ours: the same page score, but with the query PREDICTED from the previous layer's input
             (layer l's norm + q_proj + q_norm + RoPE applied to the hidden state entering layer l-1). It is
             available one layer early, so the page fetch can overlap layer l-1's compute.

Layers 0 and 1 use full attention under every sparse policy (as in Quest). Also records attention recall: the
share of the full-attention softmax mass (per query head, scored positions, layers >= 2) that the chosen
entries cover. Writes results/real.json.
"""
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

MODEL_PATH = "/home/jasper/eirene-projects/03-inference-lab/ai-lab/models/Qwen--Qwen3-8B"
SEQ, SCORE_FROM, N_DOCS, P, W, FULL_LAYERS, CHUNK = 4096, 1024, 16, 16, 128, 2, 256
BUDGETS = (512, 1024)
OUT = Path(__file__).resolve().parent / "real.json"
STATE = {"policy": "full"}


def page_bounds(key, page=P):
    """key (h, S, D) -> per-page centre and half-range (h, n_pages, D) for the Quest upper bound."""
    h, S, D = key.shape
    n = S // page
    k = key[:, :n * page].float().reshape(h, n, page, D)
    lo, hi = k.amin(2), k.amax(2)
    return (hi + lo) / 2, (hi - lo) / 2


def page_scores(q, mid, half):
    """Quest bound sum_d max(q_d*min_d, q_d*max_d) = q.mid + |q|.half, summed over the query heads of each
    KV head. q (kvh, g, c, D) float; mid/half (kvh, n, D) -> (kvh, c, n)."""
    return (q @ mid.transpose(1, 2)[:, None] + q.abs() @ half.transpose(1, 2)[:, None]).sum(1)


def eligible(t, n_pages, page=P, window=W):
    """(c, n) bool: page lies entirely before the window of query t and is not the sink page."""
    p = torch.arange(n_pages, device=t.device)
    return (p[None] >= 1) & ((p[None] + 1) * page <= (t[:, None] - window + 1))


def build_mask(sel, t, S, page=P, window=W):
    """sel (kvh, c, n) bool chosen pages -> (kvh, c, S) bool allowed keys (sinks + window + pages, causal)."""
    j = torch.arange(S, device=t.device)
    base = (j[None] < page) | (j[None] > t[:, None] - window)
    pages = sel.repeat_interleave(page, -1)
    pages = F.pad(pages, (0, S - pages.shape[-1]))
    return (base[None] | pages) & (j[None, None] <= t[None, :, None])


def choose(scores, ok, k):
    """Top-k pages among eligible ones. scores (kvh, c, n), ok (c, n) -> (kvh, c, n) bool."""
    s = scores.masked_fill(~ok[None], float("-inf"))
    k = min(k, s.shape[-1])
    idx = s.topk(k, -1).indices
    return torch.zeros_like(s, dtype=torch.bool).scatter_(-1, idx, True) & ok[None]


def predicted_query(module, h_prev):
    """Layer l's query computed from the hidden state that entered layer l-1."""
    x = module._ln(h_prev)
    shape = (*x.shape[:-1], -1, module.head_dim)
    q = module.q_norm(module.q_proj(x).view(shape)).transpose(1, 2)
    cos, sin = STATE["pos"]
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
    return apply_rotary_pos_emb(q, q, cos, sin)[0]


def oracle_pages(query, key):
    """Oracle page choice for every position, (kvh, S, n) bool (used by the 'prev' policy)."""
    S, kvh, D = query.shape[2], key.shape[1], query.shape[3]
    g, n = query.shape[1] // kvh, S // P
    mid, half = page_bounds(key[0])
    out = []
    for c0 in range(0, S, CHUNK):
        c1 = min(c0 + CHUNK, S)
        t = torch.arange(c0, c1, device=query.device)
        out.append(choose(page_scores(query[0, :, c0:c1].float().view(kvh, g, c1 - c0, D), mid, half),
                          eligible(t, n), STATE["k"]))
    return torch.cat(out, 1)


def swap_attention(module, query, key, value, attention_mask, scaling=None, **kw):
    pol, li = STATE["policy"], module.layer_idx
    if pol == "full" or li < FULL_LAYERS:
        out = F.scaled_dot_product_attention(query, key, value, is_causal=True, scale=scaling, enable_gqa=True)
        if pol == "prev":
            STATE["prev_sel"] = oracle_pages(query, key)
        return out.transpose(1, 2).contiguous(), None
    S, kvh, D = query.shape[2], key.shape[1], query.shape[3]
    g = query.shape[1] // kvh
    n = S // P
    mid, half = page_bounds(key[0])
    qp = predicted_query(module, STATE["hidden"][li - 1]) if pol == "predicted" else None
    kfull = key.float().repeat_interleave(g, 1)
    oracle_all = []
    outs = []
    for c0 in range(0, S, CHUNK):
        c1 = min(c0 + CHUNK, S)
        t = torch.arange(c0, c1, device=query.device)
        ok = eligible(t, n)
        qc = query[0, :, c0:c1].float().view(kvh, g, len(t), D)
        so = page_scores(qc, mid, half)
        orc = choose(so, ok, STATE["k"])
        oracle_all.append(orc)
        if pol == "oracle":
            sel = orc
        elif pol == "predicted":
            sel = choose(page_scores(qp[0, :, c0:c1].float().view(kvh, g, len(t), D), mid, half), ok, STATE["k"])
        elif pol == "prev":
            sel = STATE["prev_sel"][:, c0:c0 + len(t)] & ok[None]
        elif pol == "random":
            gen = torch.Generator(device=query.device).manual_seed(1000 * li + c0)
            sel = choose(torch.rand(so.shape, device=query.device, generator=gen), ok, STATE["k"])
        else:  # sinks: first page + last C-P tokens, no pages
            sel = torch.zeros_like(orc)
        win = STATE["C"] - P if pol == "sinks" else W
        mask = build_mask(sel, t, S, window=win).repeat_interleave(g, 0)[None]   # (1, H, c, S)
        qq = query[:, :, c0:c1]
        outs.append(F.scaled_dot_product_attention(qq, key, value, attn_mask=mask, scale=scaling, enable_gqa=True))
        if c0 + CHUNK > SCORE_FROM:
            logits = (qq.float() @ kfull.transpose(2, 3)) * scaling
            causal = torch.arange(S, device=t.device)[None] <= t[:, None]
            probs = logits.masked_fill(~causal, float("-inf")).softmax(-1)
            keep = t >= SCORE_FROM
            cov = (probs * mask)[..., keep, :].sum(-1)
            STATE["recall"][0] += cov.sum().item()
            STATE["recall"][1] += cov.numel()
    STATE["prev_sel"] = torch.cat(oracle_all, 1)
    return torch.cat(outs, 2).transpose(1, 2).contiguous(), None


@torch.no_grad()
def nll(model, docs):
    tot, cnt = 0.0, 0
    for d in docs:
        x = d.unsqueeze(0).cuda()
        h = model.model(input_ids=x).last_hidden_state[0]
        for s in range(SCORE_FROM - 1, SEQ - 1, 512):
            e = min(s + 512, SEQ - 1)
            logp = torch.log_softmax(model.lm_head(h[s:e]).float(), -1)
            tot -= logp.gather(1, x[0, s + 1:e + 1, None]).sum().item()
            cnt += e - s
    return tot, cnt


def main():
    from datasets import load_dataset
    from transformers import AttentionInterface, AutoModelForCausalLM, AutoTokenizer
    AttentionInterface.register("swap", swap_attention)
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.bfloat16, local_files_only=True,
                                                 device_map="cuda").eval()
    model.set_attn_implementation("swap")
    STATE["hidden"] = {}
    for i, layer in enumerate(model.model.layers):
        object.__setattr__(layer.self_attn, "_ln", layer.input_layernorm)   # not a new submodule

        def keep_hidden(mod, args, kwargs, i=i):
            STATE["hidden"][i] = args[0] if args else kwargs["hidden_states"]

        def keep_pos(mod, args, kwargs):
            STATE["pos"] = kwargs["position_embeddings"]

        layer.register_forward_pre_hook(keep_hidden, with_kwargs=True)
        layer.self_attn.register_forward_pre_hook(keep_pos, with_kwargs=True)
    text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    docs = [ids[i * SEQ:(i + 1) * SEQ] for i in range(N_DOCS)]
    res = {"model": "Qwen3-8B", "data": f"wikitext-2 test, {N_DOCS} x {SEQ} tokens",
           "scored": f"tokens {SCORE_FROM}..{SEQ - 1}", "page": P, "window": W, "full_layers": FULL_LAYERS,
           "rows": []}

    def run(policy, C=None):
        STATE.update(policy=policy, C=C, k=(C - P - W) // P if C else 0, recall=[0.0, 0])
        t = time.time()
        tot, cnt = nll(model, docs)
        row = {"policy": policy, "budget": C, "pages": STATE["k"] if policy not in ("full", "sinks") else 0,
               "ppl": round(math.exp(tot / cnt), 4), "tokens": cnt, "seconds": round(time.time() - t, 1)}
        if STATE["recall"][1]:
            row["recall"] = round(STATE["recall"][0] / STATE["recall"][1], 4)
        res["rows"].append(row)
        print(row, flush=True)
        OUT.write_text(json.dumps(res, indent=2))

    run("full")
    for C in BUDGETS:
        for policy in ("sinks", "random", "prev", "predicted", "oracle"):
            run(policy, C)


if __name__ == "__main__":
    main()
