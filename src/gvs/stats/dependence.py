"""Dependence tests between paired series of graph summaries.

Setting (Fujita et al.): k paired graphs {(G1_i, G2_i)}. Each graph is mapped
to a summary (scalar lambda_max or an embedding vector); we test whether the
two summary series are dependent.

All tests use permutation null distributions (B permutations of the pairing),
so power comparisons across statistics are at the same exact alpha level.

- Scalar summaries -> |Pearson r|
- Vector summaries -> distance correlation (Szekely et al. 2007). dCor is
  invariant to orthogonal transforms of either space, which matters for learned
  embeddings whose axes are arbitrary.
"""

from __future__ import annotations

import numpy as np


def _double_center(d: np.ndarray) -> np.ndarray:
    return d - d.mean(axis=0, keepdims=True) - d.mean(axis=1, keepdims=True) + d.mean()


def _dist_matrix(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x.T).T  # (k,) -> (k, 1)
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def dcor_from_centered(a: np.ndarray, b: np.ndarray) -> float:
    dcov2 = (a * b).mean()
    dvar = (a * a).mean() * (b * b).mean()
    if dvar <= 0:
        return 0.0
    return float(np.sqrt(max(dcov2, 0.0) / np.sqrt(dvar)))


def dcor(x: np.ndarray, y: np.ndarray) -> float:
    """Distance correlation between two sample matrices (k x dx), (k x dy)."""
    return dcor_from_centered(
        _double_center(_dist_matrix(x)), _double_center(_dist_matrix(y))
    )


def dcor_perm_test(
    x: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int | None = None
) -> tuple[float, float]:
    """dCor and permutation p-value. Permuting the centered distance matrix of y
    by the same row/col permutation commutes with centering, so we center once."""
    rng = np.random.default_rng(seed)
    a = _double_center(_dist_matrix(x))
    b = _double_center(_dist_matrix(y))
    obs = dcor_from_centered(a, b)
    k = a.shape[0]
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(k)
        if dcor_from_centered(a, b[np.ix_(perm, perm)]) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def pearson_perm_test(
    x: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int | None = None
) -> tuple[float, float]:
    """|Pearson r| and two-sided permutation p-value for scalar series."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    obs = abs(np.corrcoef(x, y)[0, 1])
    count = 0
    for _ in range(n_perm):
        if abs(np.corrcoef(x, rng.permutation(y))[0, 1]) >= obs:
            count += 1
    return float(obs), (count + 1) / (n_perm + 1)
