import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "results"))
from run_real import P, W, build_mask, choose, eligible, page_bounds, page_scores  # noqa: E402


def test_quest_bound_is_upper_bound():
    torch.manual_seed(0)
    key = torch.randn(2, 8 * P, 16)
    q = torch.randn(2, 1, 5, 16)
    mid, half = page_bounds(key)
    bound = page_scores(q, mid, half)                              # (2, 5, 8)
    exact = (q[:, 0] @ key.transpose(1, 2)).view(2, 5, 8, P).amax(-1)
    assert (bound >= exact - 1e-4).all()


def test_eligible_pages_never_overlap_window_or_sinks():
    S = 40 * P
    t = torch.arange(S)
    ok = eligible(t, S // P)
    assert not ok[:, 0].any()
    for q in (W, W + P, S - 1):
        last = ok[q].nonzero().max().item() if ok[q].any() else -1
        assert (last + 1) * P - 1 <= q - W


def test_mask_respects_budget_and_causality():
    S, k = 40 * P, 5
    t = torch.arange(S)
    torch.manual_seed(1)
    sel = choose(torch.randn(2, S, S // P), eligible(t, S // P), k)
    mask = build_mask(sel, t, S)
    assert (mask.sum(-1) <= P + W + k * P).all()
    assert not (mask & (torch.arange(S)[None, None] > t[None, :, None])).any()
    assert mask[..., 0].all()                                      # sinks always visible
