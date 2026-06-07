"""Graph coarsening — the 'smaller graph, same structure' philosophy.

The decode approach (decoders.py) generates a *new* small graph by decoding
selected latent vectors, which hallucinates edges from latent geometry. Coarsening
instead AGGREGATES the original graph's real edges: partition the N nodes into m
supernodes and let supernode edges be the (thresholded) sum of edges between the
groups. The coarse graph is built from real connectivity, not from the decoder's
inductive bias.

The research question is whether the partition should come from the VGAE latent
space (`latent_coarsen`) or whether any balanced partition does as well
(`random_coarsen`, the ablation).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _coarsen_by_labels(g: nx.Graph, labels: np.ndarray, m: int, density: float) -> nx.Graph:
    """Aggregate edges into an m-supernode graph, thresholded to `density`.

    Supernode weight C[a,b] = number of original edges between groups a and b.
    Intra-group edges are dropped (they collapse into the supernode). Keep the
    top-k inter-supernode weights so the coarse graph matches the target density,
    matching the decode pipeline's density-matching for a fair comparison.
    """
    a = nx.to_numpy_array(g)
    s = np.zeros((g.number_of_nodes(), m))
    s[np.arange(g.number_of_nodes()), labels] = 1.0
    c = s.T @ a @ s
    np.fill_diagonal(c, 0.0)
    iu = np.triu_indices(m, k=1)
    weights = c[iu]
    k = int(round(density * m * (m - 1) / 2))
    keep = np.argsort(weights)[len(weights) - k:]
    # only keep edges that actually carry weight (avoid inventing zero-weight links)
    keep = keep[weights[keep] > 0]
    out = nx.empty_graph(m)
    out.add_edges_from(zip(iu[0][keep], iu[1][keep]))
    return out


def latent_coarsen(
    g: nx.Graph, z, m: int, density: float, seed: int | None = None
) -> nx.Graph:
    """k-means the VGAE latents into m groups, then aggregate real edges."""
    from sklearn.cluster import KMeans

    zn = z.detach().numpy() if hasattr(z, "detach") else np.asarray(z)
    labels = KMeans(n_clusters=m, n_init=5, random_state=seed).fit_predict(zn)
    return _coarsen_by_labels(g, labels, m, density)


def random_coarsen(
    g: nx.Graph, m: int, density: float, seed: int | None = None
) -> nx.Graph:
    """Ablation: a uniformly random balanced partition into m groups."""
    rng = np.random.default_rng(seed)
    n = g.number_of_nodes()
    labels = np.tile(np.arange(m), n // m + 1)[:n]
    rng.shuffle(labels)
    return _coarsen_by_labels(g, labels, m, density)
