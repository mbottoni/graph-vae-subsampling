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


def er_pair_series(
    k: int,
    n: int,
    rho: float,
    p_range: tuple[float, float] = (0.1, 0.3),
    seed: int | None = None,
) -> tuple[list[nx.Graph], list[nx.Graph], np.ndarray, np.ndarray]:
    """k pairs of ER graphs whose *parameters* are dependent (Fujita's setting).

    p1_i ~ U(p_range); with probability rho, p2_i = p1_i (shared parameter),
    otherwise p2_i is an independent draw. Marginals are exactly U(p_range) and
    corr(p1, p2) = rho. Given the parameters, graphs are independent — the
    dependence lives entirely at the parameter level, which is what the
    spectral/embedding statistics must detect.
    """
    rng = np.random.default_rng(seed)
    p1 = rng.uniform(*p_range, size=k)
    copy = rng.random(k) < rho
    p2 = np.where(copy, p1, rng.uniform(*p_range, size=k))
    seeds = rng.integers(0, 2**31, size=2 * k)
    gs1 = [nx.erdos_renyi_graph(n, p, seed=int(s)) for p, s in zip(p1, seeds[:k])]
    gs2 = [nx.erdos_renyi_graph(n, p, seed=int(s)) for p, s in zip(p2, seeds[k:])]
    return gs1, gs2, p1, p2


def sbm_pair_series(
    k: int,
    n: int,
    rho: float,
    correlate: str = "both",
    p_in_range: tuple[float, float] = (0.15, 0.35),
    p_out_range: tuple[float, float] = (0.02, 0.10),
    seed: int | None = None,
) -> tuple[list[nx.Graph], list[nx.Graph]]:
    """k pairs of 2-block SBMs with parameter-level dependence of strength rho.

    The parameter is 2-dimensional, (p_in, p_out). `correlate` chooses where the
    dependence lives:
      "both"  — each parameter copied with probability rho (joint dependence)
      "p_in"  — only p_in correlated; p_out independent
      "p_out" — only p_out correlated; p_in independent

    Rationale: for a 2-block SBM, lambda_1 ~ (n/2)(p_in + p_out) and
    lambda_2 ~ (n/2)(p_in - p_out). A test using only the spectral radius sees
    a 1-D projection of the parameter; with correlate="p_out" the dependence
    signal is mostly orthogonal to that projection (p_out's range is small
    relative to p_in's), so lambda_max should lose power while statistics that
    retain lambda_2 should not.
    """
    rng = np.random.default_rng(seed)

    def draw_pair(lo_hi: tuple[float, float], correlated: bool) -> tuple[np.ndarray, np.ndarray]:
        a = rng.uniform(*lo_hi, size=k)
        if correlated:
            copy = rng.random(k) < rho
            b = np.where(copy, a, rng.uniform(*lo_hi, size=k))
        else:
            b = rng.uniform(*lo_hi, size=k)
        return a, b

    if correlate not in ("both", "p_in", "p_out"):
        raise ValueError(f"unknown correlate mode: {correlate}")
    p_in1, p_in2 = draw_pair(p_in_range, correlate in ("both", "p_in"))
    p_out1, p_out2 = draw_pair(p_out_range, correlate in ("both", "p_out"))

    sizes = [n // 2, n - n // 2]
    seeds = rng.integers(0, 2**31, size=2 * k)

    def make(p_in: float, p_out: float, s: int) -> nx.Graph:
        probs = [[p_in, p_out], [p_out, p_in]]
        return nx.stochastic_block_model(sizes, probs, seed=int(s))

    gs1 = [make(pi, po, s) for pi, po, s in zip(p_in1, p_out1, seeds[:k])]
    gs2 = [make(pi, po, s) for pi, po, s in zip(p_in2, p_out2, seeds[k:])]
    return gs1, gs2


def dc_sbm_pair_series(
    k: int,
    n: int,
    rho: float,
    n_blocks: int = 2,
    degree_hetero: float = 0.0,
    p_in_range: tuple[float, float] = (0.15, 0.35),
    p_out_range: tuple[float, float] = (0.02, 0.10),
    seed: int | None = None,
) -> tuple[list[nx.Graph], list[nx.Graph]]:
    """k pairs of (degree-corrected) SBMs with dependence only in p_out.

    Generalizes sbm_pair_series toward realistic connectomes along two axes:
      n_blocks      — number of equal-size communities (2 -> multi-community)
      degree_hetero — per-node degree multipliers theta_i ~ LogNormal(0, s),
                      s = degree_hetero (0 recovers the plain SBM). Edge
                      probability is clip(theta_i theta_j B_{c_i c_j}, 0, 1).

    theta is drawn fresh per graph: degree heterogeneity is a within-graph
    nuisance, present in both members of a pair but uncorrelated across them, so
    the only dependence is still p_out. This is the adversarial setting for the
    benchmark — extra nuisance variance with no extra signal.
    """
    rng = np.random.default_rng(seed)
    p_in1 = rng.uniform(*p_in_range, size=k)
    p_in2 = rng.uniform(*p_in_range, size=k)  # p_in independent across the pair
    copy = rng.random(k) < rho
    p_out1 = rng.uniform(*p_out_range, size=k)
    p_out2 = np.where(copy, p_out1, rng.uniform(*p_out_range, size=k))

    base = n // n_blocks
    sizes = [base] * (n_blocks - 1) + [n - base * (n_blocks - 1)]
    blocks = np.concatenate([[b] * s for b, s in enumerate(sizes)])

    def make(p_in: float, p_out: float, s: int) -> nx.Graph:
        rr = np.random.default_rng(s)
        theta = (rr.lognormal(0.0, degree_hetero, size=n)
                 if degree_hetero > 0 else np.ones(n))
        bmat = np.full((n_blocks, n_blocks), p_out)
        np.fill_diagonal(bmat, p_in)
        prob = theta[:, None] * theta[None, :] * bmat[np.ix_(blocks, blocks)]
        iu = np.triu_indices(n, k=1)
        draw = rr.random(len(iu[0])) < np.clip(prob[iu], 0.0, 1.0)
        g = nx.empty_graph(n)
        g.add_edges_from(zip(iu[0][draw], iu[1][draw]))
        return g

    seeds = rng.integers(0, 2**31, size=2 * k)
    gs1 = [make(pi, po, int(s)) for pi, po, s in zip(p_in1, p_out1, seeds[:k])]
    gs2 = [make(pi, po, int(s)) for pi, po, s in zip(p_in2, p_out2, seeds[k:])]
    return gs1, gs2


def to_pyg(g: nx.Graph) -> Data:
    """Convert to PyG Data with identity-matrix node features (featureless setting)."""
    data = from_networkx(g)
    data.x = torch.eye(g.number_of_nodes())
    # from_networkx may attach node/graph attrs (e.g. SBM "block"); keep only structure.
    return Data(x=data.x, edge_index=data.edge_index, num_nodes=g.number_of_nodes())
