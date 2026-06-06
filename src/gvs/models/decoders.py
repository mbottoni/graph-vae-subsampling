"""M1 — latent-subsampling decoder.

Given the latent matrix Z (N x d) of a trained VGAE, downsample the graph to
m < N nodes by picking m latent vectors and decoding them with the same
inner-product decoder. Three selection strategies:

- "random":    a uniform subset of m rows of Z (the latent analogue of node sampling)
- "kmeans":    the m k-means centroids of Z (coverage-maximizing summary of the latent space)
- "posterior": m draws from a Gaussian fit to Z (sampling the aggregated posterior —
               the only variant that generates *new* latent points)

Decoding: A_hat = sigmoid(Z_m Z_m^T), then keep the top-k edges where k matches
a target density (default: the original graph's density). Density matching makes
the comparison fair — every method gets the density right by construction, so the
discriminating metrics are degree shape, clustering, triangles, and the spectrum.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch


def decode_topk(z: torch.Tensor, density: float) -> nx.Graph:
    """Decode latents to a graph, keeping the top-k edges to match `density`."""
    m = z.shape[0]
    k = int(round(density * m * (m - 1) / 2))
    with torch.no_grad():
        scores_full = torch.sigmoid(z @ z.t()).numpy()
    iu = np.triu_indices(m, k=1)
    scores = scores_full[iu]
    keep = np.argsort(scores)[len(scores) - k:]
    g = nx.empty_graph(m)
    g.add_edges_from(zip(iu[0][keep], iu[1][keep]))
    return g


def select_latents(
    z: torch.Tensor, m: int, method: str, seed: int | None = None
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    zn = z.detach().numpy()

    if method == "random":
        idx = rng.choice(zn.shape[0], size=m, replace=False)
        sel = zn[idx]
    elif method == "kmeans":
        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=m, n_init=5, random_state=seed).fit(zn)
        sel = km.cluster_centers_
    elif method == "posterior":
        mu = zn.mean(axis=0)
        cov = np.cov(zn, rowvar=False)
        sel = rng.multivariate_normal(mu, cov, size=m)
    else:
        raise ValueError(f"unknown selection method: {method}")
    return torch.from_numpy(np.asarray(sel, dtype=np.float32))


def latent_downsample(
    z: torch.Tensor, m: int, density: float, method: str = "random", seed: int | None = None
) -> nx.Graph:
    """Full M1 pipeline: select m latents via `method`, decode at `density`."""
    z_m = select_latents(z, m, method, seed)
    return decode_topk(z_m, density)
