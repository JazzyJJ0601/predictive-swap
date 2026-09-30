import pytest
from src.kv_cache import SwapCache
from src.transformer import PredictiveSwapAttention


def test_swap_cache():
    cache = SwapCache(capacity=10)
    result = cache.swap([0, 1], ["entry1", "entry2"])
    assert result[0] == "entry1"


def test_predictive_attention():
    attn = PredictiveSwapAttention(head_dim=64)
    res = attn.forward([0], [0], [0], [0])
    assert res is not None
