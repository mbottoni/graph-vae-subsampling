"""M1 mechanism check — do VGAE latent singular values disentangle (p_in, p_out)?

The D2 mechanism claim: adjacency eigenvalues entangle the SBM parameters
(lambda_{1,2} ~ (n/2)(p_in +/- p_out) — both coordinates load on both
parameters, dominated by p_in's larger range), while the VGAE latent SVs give
p_out its own coordinate (between-cluster separation) approximately independent
of p_in (within-cluster spread).

Test: sample 200 independent 2-block SBMs over the same parameter ranges as D2,
compute each summary coordinate, and report its Pearson correlation with p_in
and with p_out. Evidence for disentanglement = some vgae SV coordinate with
|corr(., p_out)| high and |corr(., p_in)| low, while every leading adjacency
eigenvalue has |corr(., p_in)| dominant.

Writes results/m1_disentanglement.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from gvs.stats.embeddings import emb_spectral, emb_vgae_sv

N_GRAPHS, N_NODES = 200, 60
P_IN_RANGE, P_OUT_RANGE = (0.15, 0.35), (0.02, 0.10)
RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    rng = np.random.default_rng(0)
    p_in = rng.uniform(*P_IN_RANGE, size=N_GRAPHS)
    p_out = rng.uniform(*P_OUT_RANGE, size=N_GRAPHS)
    sizes = [N_NODES // 2, N_NODES // 2]

    spec, vgae = [], []
    for i in range(N_GRAPHS):
        probs = [[p_in[i], p_out[i]], [p_out[i], p_in[i]]]
        g = nx.stochastic_block_model(sizes, probs, seed=int(rng.integers(2**31)))
        spec.append(emb_spectral(g, q=5))
        vgae.append(emb_vgae_sv(g, q=5, seed=i))
    spec, vgae = np.array(spec), np.array(vgae)

    def corr_table(x: np.ndarray, names: list[str]) -> dict:
        out = {}
        for j, name in enumerate(names):
            out[name] = {
                "p_in": float(np.corrcoef(x[:, j], p_in)[0, 1]),
                "p_out": float(np.corrcoef(x[:, j], p_out)[0, 1]),
            }
        return out

    tables = {
        "adjacency_spectrum": corr_table(spec, [f"lambda_{i+1}" for i in range(5)]),
        "vgae_sv": corr_table(vgae, [f"sv_{i+1}" for i in range(5)]),
        "derived": {
            "(lambda1-lambda2)/2": {
                "p_in": float(np.corrcoef((spec[:, 0] - spec[:, 1]) / 2, p_in)[0, 1]),
                "p_out": float(np.corrcoef((spec[:, 0] - spec[:, 1]) / 2, p_out)[0, 1]),
            },
        },
    }

    print(f"{'coordinate':<22}{'corr p_in':>12}{'corr p_out':>12}")
    for block, tab in tables.items():
        print(f"-- {block}")
        for name, c in tab.items():
            print(f"{name:<22}{c['p_in']:>12.3f}{c['p_out']:>12.3f}")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "m1_disentanglement.json"
    path.write_text(json.dumps({
        "experiment": "m1_disentanglement",
        "config": {"n_graphs": N_GRAPHS, "n_nodes": N_NODES,
                   "p_in_range": P_IN_RANGE, "p_out_range": P_OUT_RANGE},
        "correlations": tables,
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
