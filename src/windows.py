import numpy as np


def rectangular_window(t: np.ndarray) -> np.ndarray:
    """
    Unit rectangular window over the provided sampled interval.
    """
    return np.ones_like(t, dtype=float)


def hann_window(t: np.ndarray) -> np.ndarray:
    """
    Hann window over the provided sampled interval.
    """
    n = len(t)
    if n < 2:
        return np.ones_like(t, dtype=float)
    return np.hanning(n).astype(float)


def get_window(t: np.ndarray, kind: str = "rectangular") -> np.ndarray:
    """
    Return a named window sampled on t.
    """
    kind = kind.lower()
    if kind in ("rect", "rectangular", "boxcar"):
        return rectangular_window(t)
    if kind in ("hann", "hanning"):
        return hann_window(t)
    raise ValueError(f"Unknown window kind: {kind}")