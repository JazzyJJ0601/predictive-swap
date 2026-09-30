# Predictive swap attention layer

from .kv_cache import SwapCache


class PredictiveSwapAttention:
    """Core predictive-swap attention layer."""
    def __init__(self, head_dim, capacity=1024):
        self.head_dim = head_dim
        self.cache = SwapCache(capacity)

    def forward(self, query, key, value, prediction_mask):
        """Intercept KV cache entries between decoder layers."""
        swapped_cache = self.cache.swap(prediction_mask, value)
        return swapped_cache
