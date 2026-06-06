"""Distances between graph-statistic distributions.

Compare an original graph and its downsampled version via the distance between
their structural distributions (degree sequence, spectrum). Distribution-level
distances are the right comparison when the two graphs have different sizes.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def ks_distance(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    res = stats.ks_2samp(x, y)
    return float(res.statistic), float(res.pvalue)


def wasserstein(x: np.ndarray, y: np.ndarray) -> float:
    """1-Wasserstein (earth mover's) distance between empirical distributions."""
    return float(stats.wasserstein_distance(x, y))


def normalized_degree_distance(deg_a: np.ndarray, deg_b: np.ndarray) -> float:
    """Wasserstein distance between degree sequences normalized by graph size.

    Degrees scale with n, so for graphs of different sizes we compare
    degree / (n - 1) (i.e., per-node connection probability).
    """
    na, nb = len(deg_a), len(deg_b)
    if na < 2 or nb < 2:
        return float("nan")
    return wasserstein(deg_a / (na - 1), deg_b / (nb - 1))
