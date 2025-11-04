from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np

# Optional sklearn for PCA
_HAVE_SKLEARN = False
try:
    from sklearn.decomposition import PCA  # type: ignore
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False


def robust_percentile_scale(arr: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    lo = np.percentile(a, low)
    hi = np.percentile(a, high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1e-6
    a = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    return a


def arcsinh_normalize(arr: np.ndarray, cofactor: float = 5.0) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    return np.arcsinh(a / cofactor)


def percentile_clip_normalize(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    a = arr.astype(np.float32, copy=False)
    vmin = np.percentile(a, p_low)
    vmax = np.percentile(a, p_high)
    clipped = np.clip(a, vmin, vmax)
    if vmax > vmin:
        normalized = (clipped - vmin) / (vmax - vmin)
    else:
        normalized = np.zeros_like(clipped)
    return normalized


def combine_channels(images: List[np.ndarray], method: str, weights: List[float] = None) -> np.ndarray:
    if not images:
        raise ValueError("No images provided")

    if method == "single":
        return images[0]

    if method == "mean":
        return np.mean(np.stack(images, axis=0), axis=0)

    if method == "max":
        return np.max(np.stack(images, axis=0), axis=0)

    if method == "weighted":
        if weights is None or len(weights) != len(images):
            raise ValueError("Weights must be provided and match number of images")
        w = np.array(weights, dtype=np.float32)
        w = w / (np.sum(w) + 1e-8)
        stack = np.stack(images, axis=0)
        return np.tensordot(w, stack, axes=(0, 0))

    if method == "pca1":
        if not _HAVE_SKLEARN:
            raise ImportError("scikit-learn required for PCA method")
        flattened = np.array([img.flatten() for img in images]).T
        pca = PCA(n_components=1)
        pca_result = pca.fit_transform(flattened)
        return pca_result.reshape(images[0].shape)

    raise ValueError(f"Unknown combination method: {method}")


class PreprocessingCache:
    """Cache for preprocessing statistics to ensure identical batch runs."""

    def __init__(self):
        self.cache: Dict[str, Dict] = {}

    def get_key(self, acq_id: str, channel: str, method: str, **params) -> str:
        param_str = "_".join(f"{k}_{v}" for k, v in sorted(params.items()))
        return f"{acq_id}_{channel}_{method}_{param_str}"

    def get_stats(self, acq_id: str, channel: str, method: str, **params) -> dict:
        key = self.get_key(acq_id, channel, method, **params)
        return self.cache.get(key, {})

    def set_stats(self, acq_id: str, channel: str, method: str, stats: dict, **params):
        key = self.get_key(acq_id, channel, method, **params)
        self.cache[key] = stats

    def clear(self):
        self.cache.clear()


def stack_to_rgb(stack: np.ndarray) -> np.ndarray:
    H, W, C = stack.shape
    if C == 1:
        g = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        return np.dstack([g, g, g])
    elif C == 2:
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        return np.dstack([r, g, g])
    else:
        r = (stack[..., 0] - np.min(stack[..., 0])) / (np.max(stack[..., 0]) - np.min(stack[..., 0]) + 1e-8)
        g = (stack[..., 1] - np.min(stack[..., 1])) / (np.max(stack[..., 1]) - np.min(stack[..., 1]) + 1e-8)
        b = (stack[..., 2] - np.min(stack[..., 2])) / (np.max(stack[..., 2]) - np.min(stack[..., 2]) + 1e-8)
        return np.dstack([r, g, b])


