"""Synthetic graph generators (shared common library).

Single-graph random-graph generators used across both research directions, plus
`to_pyg` for the VGAE. Paired-graph generators for dependence testing live in
`deptest.pairs`.
"""

from __future__ import annotations

import networkx as nx
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


def to_pyg(g: nx.Graph) -> Data:
    """Convert to PyG Data with identity-matrix node features (featureless setting)."""
    data = from_networkx(g)
    data.x = torch.eye(g.number_of_nodes())
    # from_networkx may attach node/graph attrs (e.g. SBM "block"); keep only structure.
    return Data(x=data.x, edge_index=data.edge_index, num_nodes=g.number_of_nodes())
