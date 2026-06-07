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


def whiten(x: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """ZCA-whiten a sample matrix (k x d): center, then rescale so the empirical
    covariance is identity. Exposes low-variance directions that distance-based
    statistics would otherwise ignore — the direct test of the 'geometry vs
    information' mechanism for fixed spectra."""
    xc = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(xc, rowvar=False)
    cov = np.atleast_2d(cov)
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, eps)
    return xc @ v @ np.diag(w**-0.5) @ v.T


def _gaussian_gram(x: np.ndarray) -> np.ndarray:
    d = _dist_matrix(x)
    iu = np.triu_indices(d.shape[0], k=1)
    sigma = np.median(d[iu])
    if sigma <= 0:
        sigma = 1.0
    return np.exp(-(d**2) / (2 * sigma**2))


def hsic_perm_test(
    x: np.ndarray, y: np.ndarray, n_perm: int = 200, seed: int | None = None
) -> tuple[float, float]:
    """HSIC (Gaussian kernels, median-heuristic bandwidth) with permutation p-value."""
    rng = np.random.default_rng(seed)
    k = x.shape[0] if x.ndim > 1 else len(x)
    h = np.eye(k) - np.full((k, k), 1.0 / k)
    kc = h @ _gaussian_gram(x) @ h
    lc = h @ _gaussian_gram(y) @ h
    obs = float((kc * lc).mean())
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(k)
        if float((kc * lc[np.ix_(perm, perm)]).mean()) >= obs:
            count += 1
    return obs, (count + 1) / (n_perm + 1)


def _cca_first_direction(
    x: np.ndarray, y: np.ndarray, reg: float = 1e-2
) -> tuple[np.ndarray, np.ndarray]:
    """First canonical directions (w_x, w_y) maximizing corr(x w_x, y w_y),
    ridge-regularized for small samples. Solves the standard generalized
    eigenproblem C_xx^{-1} C_xy C_yy^{-1} C_yx w_x = rho^2 w_x."""
    xc = x - x.mean(axis=0, keepdims=True)
    yc = y - y.mean(axis=0, keepdims=True)
    n = xc.shape[0]
    cxx = xc.T @ xc / n + reg * np.eye(xc.shape[1])
    cyy = yc.T @ yc / n + reg * np.eye(yc.shape[1])
    cxy = xc.T @ yc / n
    m = np.linalg.solve(cxx, cxy) @ np.linalg.solve(cyy, cxy.T)
    w, v = np.linalg.eig(m)
    wx = np.real(v[:, int(np.argmax(np.real(w)))])
    wy = np.linalg.solve(cyy, cxy.T @ wx)
    return wx, wy


def cca_split_test(
    x: np.ndarray,
    y: np.ndarray,
    n_perm: int = 200,
    reg: float = 1e-2,
    seed: int | None = None,
) -> tuple[float, float]:
    """Learned-projection dependence test with sample splitting.

    Learn the maximally-dependent linear projection (CCA) on a random half of
    the pairs, then run a permutation Pearson test on the projected canonical
    variates of the held-out half. Splitting makes the learned directions
    independent of the test sample, so the permutation null is exact regardless
    of how the projection was chosen — the embedding adapts to wherever the
    dependence lives without invalidating calibration.
    """
    rng = np.random.default_rng(seed)
    k = x.shape[0]
    perm = rng.permutation(k)
    tr, te = perm[: k // 2], perm[k // 2:]
    wx, wy = _cca_first_direction(x[tr], y[tr], reg=reg)
    xt, yt = x[te] @ wx, y[te] @ wy
    return pearson_perm_test(xt, yt, n_perm=n_perm, seed=seed)


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
