"""Classical graph-sampling baselines (Leskovec & Faloutsos, KDD 2005).

All return the induced subgraph on m sampled nodes, relabeled 0..m-1.
These are the honest competitors for M1: any latent-space method has to beat
(or at least match) these to be interesting.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def _induced(g: nx.Graph, nodes: set) -> nx.Graph:
    return nx.convert_node_labels_to_integers(g.subgraph(nodes).copy())


def uniform_node(g: nx.Graph, m: int, seed: int | None = None) -> nx.Graph:
    """Induced subgraph on m uniformly sampled nodes."""
    rng = np.random.default_rng(seed)
    nodes = rng.choice(list(g.nodes), size=m, replace=False)
    return _induced(g, set(nodes.tolist()))


def random_walk(
    g: nx.Graph, m: int, restart_prob: float = 0.15, seed: int | None = None
) -> nx.Graph:
    """Random walk with restarts; collect nodes until m unique are visited."""
    rng = np.random.default_rng(seed)
    nodes_list = list(g.nodes)
    start = nodes_list[rng.integers(len(nodes_list))]
    visited = {start}
    current = start
    stall = 0
    while len(visited) < m:
        neighbors = list(g.neighbors(current))
        if not neighbors or rng.random() < restart_prob:
            current = start
            stall += 1
            # Walk trapped in a small component: jump to a fresh start node.
            if stall > 100 * m:
                start = nodes_list[rng.integers(len(nodes_list))]
                visited.add(start)
                current = start
                stall = 0
            continue
        current = neighbors[rng.integers(len(neighbors))]
        visited.add(current)
    return _induced(g, visited)


def forest_fire(
    g: nx.Graph, m: int, p_forward: float = 0.7, seed: int | None = None
) -> nx.Graph:
    """Forest-fire sampling: burn outward from a random seed, branching
    geometrically with mean p_forward / (1 - p_forward) neighbors per node."""
    rng = np.random.default_rng(seed)
    visited: set = set()
    while len(visited) < m:
        unvisited = [v for v in g.nodes if v not in visited]
        frontier = [unvisited[rng.integers(len(unvisited))]]
        visited.add(frontier[0])
        while frontier and len(visited) < m:
            v = frontier.pop(0)
            candidates = [u for u in g.neighbors(v) if u not in visited]
            if not candidates:
                continue
            n_burn = min(rng.geometric(1 - p_forward), len(candidates))
            burn = rng.choice(candidates, size=n_burn, replace=False)
            for u in burn:
                if len(visited) >= m:
                    break
                visited.add(u)
                frontier.append(u)
    return _induced(g, visited)
