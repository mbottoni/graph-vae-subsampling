"""Graph-level summaries (embeddings) used as test statistics for dependence.

Each function maps a graph to a fixed-length vector. The benchmark is Fujita's
scalar lambda_max; the project hypothesis is that richer summaries carry more
of the dependence signal.

Note on the VGAE summary: per-graph training makes latent axes arbitrary (any
orthogonal rotation of Z is an equally good solution), so coordinates are not
comparable across graphs. We therefore use the singular values of the centered
latent matrix — rotation-invariant, and the natural analogue of using the
adjacency spectrum. ("vgae_sv" vs "spectral" is then a clean comparison:
learned spectrum vs adjacency spectrum.)
"""

from __future__ import annotations

import networkx as nx
import numpy as np

from gvs.metrics.graph_stats import adjacency_spectrum, degree_sequence


def emb_lambda_max(g: nx.Graph) -> np.ndarray:
    """Fujita baseline: spectral radius only."""
    return adjacency_spectrum(g)[:1]


def emb_spectral(g: nx.Graph, q: int = 5) -> np.ndarray:
    """Top-q adjacency eigenvalues (descending)."""
    return adjacency_spectrum(g)[:q]


def emb_features(g: nx.Graph) -> np.ndarray:
    """Cheap structural feature vector (degree moments + clustering + top spectrum)."""
    deg = degree_sequence(g)
    spec = adjacency_spectrum(g)
    n = g.number_of_nodes()
    return np.array([
        nx.density(g),
        nx.average_clustering(g),
        deg.mean() / max(n - 1, 1),
        deg.std() / max(n - 1, 1),
        spec[0] / n,
        spec[1] / n,
        spec[-1] / n,
    ])


def emb_full_spectrum(g: nx.Graph) -> np.ndarray:
    """All n adjacency eigenvalues (descending) — the decisive fixed baseline."""
    return adjacency_spectrum(g)


def emb_netlsd(g: nx.Graph, n_t: int = 16) -> np.ndarray:
    """NetLSD heat-trace signature (Tsitsulin et al. 2018), normalized by n:
    h(t) = (1/n) sum_i exp(-t mu_i), mu_i eigenvalues of the normalized Laplacian."""
    n = g.number_of_nodes()
    lap = nx.normalized_laplacian_matrix(g).toarray()
    mu = np.linalg.eigvalsh(lap)
    ts = np.logspace(-2, 2, n_t)
    return np.exp(-np.outer(ts, mu)).sum(axis=1) / n


def emb_vgae_sv(
    g: nx.Graph,
    q: int = 5,
    hidden: int = 16,
    latent_dim: int = 8,
    epochs: int = 60,
    seed: int = 0,
) -> np.ndarray:
    """Top-q singular values of the centered VGAE latent matrix (per-graph training).

    Uses full-graph reconstruction loss (no link split) — for summarization we
    want the best fit, not generalization.
    """
    import torch

    from gvs.data.synthetic import to_pyg
    from gvs.models.vgae import build_vgae

    torch.manual_seed(seed)
    data = to_pyg(g)
    model = build_vgae(data.num_node_features, hidden, latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model.encode(data.x, data.edge_index)
        loss = model.recon_loss(z, data.edge_index) + model.kl_loss() / data.num_nodes
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index).numpy()
    z = z - z.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(z, compute_uv=False)
    return sv[:q] / np.sqrt(g.number_of_nodes())
