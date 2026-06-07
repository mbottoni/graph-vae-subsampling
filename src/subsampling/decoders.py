"""M1 — latent-subsampling decoder.

Given the latent matrix Z (N x d) of a trained VGAE — and optionally per-node
degree biases b from the degree-corrected decoder — downsample the graph to
m < N nodes by picking m latent vectors and decoding them.

Selection strategies (operate on the joint space [Z | b] when biases exist, so
a node's degree propensity travels with its latent position):

- "random":    a uniform subset of m rows (latent analogue of node sampling)
- "kmeans":    the m k-means centroids (coverage-maximizing summary)
- "posterior": m draws from a Gaussian fit to the rows (aggregated posterior —
               the only variant that generates *new* latent points)

Decode rules (both density-matched to a target density, so the discriminating
metrics are degree shape, clustering, triangles, and the spectrum):

- "topk":      keep the k highest-scoring edges. Deterministic; E1/E2 showed this
               manufactures transitivity (clustered latents -> mutually high
               scores -> triangles).
- "bernoulli": sample each edge independently from sigmoid(logit + c), with the
               offset c calibrated by bisection so the *expected* edge count
               matches the target. Conditional independence given Z should
               reduce the manufactured transitivity.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def _edge_logits(z: torch.Tensor, bias: torch.Tensor | None) -> np.ndarray:
    """Upper-triangle edge logits: z_i . z_j (+ b_i + b_j)."""
    with torch.no_grad():
        logits = (z @ z.t()).numpy()
        if bias is not None:
            b = bias.numpy()
            logits = logits + b[:, None] + b[None, :]
    iu = np.triu_indices(z.shape[0], k=1)
    return logits[iu]


def _build(m: int, iu: tuple, mask: np.ndarray) -> nx.Graph:
    g = nx.empty_graph(m)
    g.add_edges_from(zip(iu[0][mask], iu[1][mask]))
    return g


def decode_topk(z: torch.Tensor, density: float, bias: torch.Tensor | None = None) -> nx.Graph:
    """Keep the top-k edges by logit, k matching `density`."""
    m = z.shape[0]
    k = int(round(density * m * (m - 1) / 2))
    logits = _edge_logits(z, bias)
    iu = np.triu_indices(m, k=1)
    mask = np.zeros(len(logits), dtype=bool)
    mask[np.argsort(logits)[len(logits) - k:]] = True
    return _build(m, iu, mask)


def _calibrate_offset(logits: np.ndarray, target_edges: float) -> float:
    """Bisection for c such that sum(sigmoid(logits + c)) = target_edges."""
    lo, hi = -30.0, 30.0
    for _ in range(80):
        mid = (lo + hi) / 2
        expected = (1 / (1 + np.exp(-(logits + mid)))).sum()
        if expected > target_edges:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def decode_bernoulli(
    z: torch.Tensor,
    density: float,
    bias: torch.Tensor | None = None,
    seed: int | None = None,
) -> nx.Graph:
    """Sample edges independently from density-calibrated probabilities."""
    m = z.shape[0]
    target = density * m * (m - 1) / 2
    logits = _edge_logits(z, bias)
    c = _calibrate_offset(logits, target)
    probs = 1 / (1 + np.exp(-(logits + c)))
    rng = np.random.default_rng(seed)
    mask = rng.random(len(probs)) < probs
    return _build(m, np.triu_indices(m, k=1), mask)


def select_latents(
    z: torch.Tensor,
    m: int,
    method: str,
    seed: int | None = None,
    bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Pick m rows of [Z | b]; returns (z_m, bias_m) with bias_m None if b is None."""
    rng = np.random.default_rng(seed)
    zn = z.detach().numpy()
    joint = zn if bias is None else np.column_stack([zn, bias.numpy()])

    if method == "random":
        idx = rng.choice(joint.shape[0], size=m, replace=False)
        sel = joint[idx]
    elif method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=m, n_init=5, random_state=seed).fit(joint)
        sel = km.cluster_centers_
    elif method == "posterior":
        mu = joint.mean(axis=0)
        cov = np.cov(joint, rowvar=False)
        sel = rng.multivariate_normal(mu, cov, size=m)
    else:
        raise ValueError(f"unknown selection method: {method}")

    sel = np.asarray(sel, dtype=np.float32)
    if bias is None:
        return torch.from_numpy(sel), None
    return torch.from_numpy(sel[:, :-1]), torch.from_numpy(sel[:, -1])


def latent_downsample(
    z: torch.Tensor,
    m: int,
    density: float,
    method: str = "random",
    seed: int | None = None,
    bias: torch.Tensor | None = None,
    decode: str = "topk",
) -> nx.Graph:
    """Full M1 pipeline: select m latents via `method`, decode via `decode`."""
    z_m, bias_m = select_latents(z, m, method, seed, bias)
    if decode == "topk":
        return decode_topk(z_m, density, bias_m)
    if decode == "bernoulli":
        return decode_bernoulli(z_m, density, bias_m, seed)
    raise ValueError(f"unknown decode rule: {decode}")
