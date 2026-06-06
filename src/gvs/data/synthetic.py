"""Synthetic graph generators for experiments.

All generators return networkx.Graph; `to_pyg` converts to a torch_geometric
Data object with identity features (no node attributes yet — Phase D4 adds them).
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def erdos_renyi(n: int, p: float, seed: int | None = None) -> nx.Graph:
    return nx.erdos_renyi_graph(n, p, seed=seed)


def barabasi_albert(n: int, m: int, seed: int | None = None) -> nx.Graph:
    return nx.barabasi_albert_graph(n, m, seed=seed)


def watts_strogatz(n: int, k: int, p: float, seed: int | None = None) -> nx.Graph:
    return nx.watts_strogatz_graph(n, k, p, seed=seed)


def stochastic_block_model(
    sizes: list[int], p_in: float, p_out: float, seed: int | None = None
) -> nx.Graph:
    k = len(sizes)
    probs = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def correlated_er_pair(
    n: int, p: float, rho: float, seed: int | None = None
) -> tuple[nx.Graph, nx.Graph]:
    """Generate a pair of ER graphs with edge-wise correlation rho.

    Construction: G1 ~ ER(n, p). For G2, each potential edge copies G1's edge
    indicator with probability rho and is resampled from Bernoulli(p) otherwise.
    Marginally G2 ~ ER(n, p); corr(A1_ij, A2_ij) = rho. This is the standard
    correlated-pair construction used to evaluate graph dependence tests.
    """
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, k=1)
    a1 = rng.random(len(iu[0])) < p
    copy = rng.random(len(iu[0])) < rho
    a2 = np.where(copy, a1, rng.random(len(iu[0])) < p)

    def build(edges_mask: np.ndarray) -> nx.Graph:
        g = nx.empty_graph(n)
        g.add_edges_from(zip(iu[0][edges_mask], iu[1][edges_mask]))
        return g

    return build(a1), build(a2)


def to_pyg(g: nx.Graph) -> Data:
    """Convert to PyG Data with identity-matrix node features (featureless setting)."""
    data = from_networkx(g)
    data.x = torch.eye(g.number_of_nodes())
    # from_networkx may attach node/graph attrs (e.g. SBM "block"); keep only structure.
    return Data(x=data.x, edge_index=data.edge_index, num_nodes=g.number_of_nodes())
