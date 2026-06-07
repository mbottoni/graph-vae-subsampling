"""D1-D3 — power curves for graph dependence tests.

Setting (Fujita et al.): k pairs of ER graphs whose parameters are correlated
with strength rho (shared-parameter construction, exact marginals). For each
rho and each summary statistic, estimate the rejection rate at alpha=0.05 over
R replicates:

  rho = 0    -> type-I error (D3): should be ~0.05 for every valid test
  rho > 0    -> power (D1/D2): Fujita's lambda_max benchmark vs richer summaries

Statistics (all tested with permutation null, B=200, same alpha):
  lambda_max : |Pearson r| of spectral radii (the benchmark)
  spectral5  : dCor of top-5 adjacency eigenvalues
  features   : dCor of structural feature vectors
  vgae_sv    : dCor of top-5 singular values of per-graph VGAE latents
               (rotation-invariant learned spectrum)

Writes results/d1_dependence_power.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from deptest.pairs import er_pair_series
from deptest.dependence import dcor_perm_test, pearson_perm_test
from deptest.embeddings import emb_features, emb_spectral, emb_vgae_sv

K, N_NODES = 40, 60
RHOS = [0.0, 0.25, 0.5, 0.75, 1.0]
R = 20
N_PERM = 200
ALPHA = 0.05
RESULTS = Path(__file__).resolve().parent.parent / "results"

METHODS = ["lambda_max", "spectral5", "features", "vgae_sv"]


def graph_summaries(graphs: list, seed: int) -> dict[str, np.ndarray]:
    spec = np.array([emb_spectral(g, q=5) for g in graphs])
    feat = np.array([emb_features(g) for g in graphs])
    vgae = np.array([
        emb_vgae_sv(g, q=5, seed=seed + j) for j, g in enumerate(graphs)
    ])
    return {
        "lambda_max": spec[:, :1],
        "spectral5": spec,
        "features": feat,
        "vgae_sv": vgae,
    }


def main() -> None:
    pvals: dict[tuple[float, str], list[float]] = {
        (rho, m): [] for rho in RHOS for m in METHODS
    }
    t0 = time.time()

    for rho in RHOS:
        for r in range(R):
            seed = int(1000 * rho) * 1000 + r
            gs1, gs2, _, _ = er_pair_series(K, N_NODES, rho, seed=seed)
            s1 = graph_summaries(gs1, seed=seed)
            s2 = graph_summaries(gs2, seed=seed + 500)
            for m in METHODS:
                if m == "lambda_max":
                    _, p = pearson_perm_test(s1[m], s2[m], N_PERM, seed=seed)
                else:
                    _, p = dcor_perm_test(s1[m], s2[m], N_PERM, seed=seed)
                pvals[(rho, m)].append(p)
        done = {m: float(np.mean(np.array(pvals[(rho, m)]) < ALPHA)) for m in METHODS}
        print(f"rho={rho:.2f}  " + "  ".join(f"{m}={v:.2f}" for m, v in done.items())
              + f"  ({time.time() - t0:.0f}s)")

    power = {
        m: {str(rho): float(np.mean(np.array(pvals[(rho, m)]) < ALPHA)) for rho in RHOS}
        for m in METHODS
    }

    print(f"\nrejection rate at alpha={ALPHA} (rho=0 row = type-I error):")
    print(f"{'rho':<8}" + "".join(f"{m:>14}" for m in METHODS))
    for rho in RHOS:
        print(f"{rho:<8}" + "".join(f"{power[m][str(rho)]:>14.2f}" for m in METHODS))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "d1_dependence_power.json"
    path.write_text(json.dumps({
        "experiment": "d1_dependence_power",
        "config": {"k_pairs": K, "n_nodes": N_NODES, "rhos": RHOS,
                   "replicates": R, "n_perm": N_PERM, "alpha": ALPHA,
                   "p_range": [0.1, 0.3]},
        "power": power,
        "pvalues": {f"{rho}|{m}": pvals[(rho, m)] for rho in RHOS for m in METHODS},
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
