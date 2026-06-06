"""Structural summary statistics of a graph.

Used everywhere: to compare an original graph against its downsampled version
(Thread 1) and as candidate summary statistics for dependence testing (Thread 2).
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def degree_sequence(g: nx.Graph) -> np.ndarray:
    return np.array([d for _, d in g.degree()], dtype=float)


def adjacency_spectrum(g: nx.Graph) -> np.ndarray:
    """Eigenvalues of the adjacency matrix, sorted descending."""
    a = nx.to_numpy_array(g)
    eig = np.linalg.eigvalsh(a)
    return eig[::-1]


def summary(g: nx.Graph) -> dict[str, float]:
    """Scalar structural summary — the metrics tracked in every experiment."""
    n = g.number_of_nodes()
    m = g.number_of_edges()
    deg = degree_sequence(g)
    spec = adjacency_spectrum(g)
    return {
        "n_nodes": n,
        "n_edges": m,
        "density": nx.density(g),
        "mean_degree": float(deg.mean()) if n else 0.0,
        "clustering": nx.average_clustering(g),
        "assortativity": _safe_assortativity(g),
        "n_triangles": int(sum(nx.triangles(g).values()) / 3),
        "lambda_max": float(spec[0]) if n else 0.0,
    }


def _safe_assortativity(g: nx.Graph) -> float:
    """Degree assortativity; NaN-safe for regular/empty graphs."""
    if g.number_of_edges() == 0:
        return float("nan")
    r = nx.degree_assortativity_coefficient(g)
    return float(r)
