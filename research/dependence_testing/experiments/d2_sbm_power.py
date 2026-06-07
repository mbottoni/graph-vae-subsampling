"""D2-SBM — where the spectral radius should genuinely lose.

2-block SBM pairs with 2-D parameter (p_in, p_out); lambda_1 ~ (n/2)(p_in+p_out)
and lambda_2 ~ (n/2)(p_in-p_out). Conditions:

  correlate="both"  — dependence in both parameters (lambda_max-friendly)
  correlate="p_out" — dependence only in p_out: mostly orthogonal to lambda_1
                      (p_out range 0.08 vs p_in range 0.20, so independent p_in
                      noise dominates the radius). Statistics retaining
                      lambda_2 should keep their power; the benchmark should not.

Same protocol as D1: permutation nulls (B=200), alpha=0.05, type-I at rho=0.

Writes results/d2_sbm_power.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from deptest.pairs import sbm_pair_series
from deptest.dependence import dcor_perm_test, pearson_perm_test
from deptest.embeddings import emb_features, emb_spectral, emb_vgae_sv

K, N_NODES = 40, 60
RHOS = [0.0, 0.5, 1.0]
CONDITIONS = ["both", "p_out"]
R = 25
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
    power: dict[str, dict] = {}
    t0 = time.time()

    for cond in CONDITIONS:
        pvals: dict[tuple[float, str], list[float]] = {
            (rho, m): [] for rho in RHOS for m in METHODS
        }
        for rho in RHOS:
            for r in range(R):
                # Deterministic seed (hash() is salted per process).
                seed = (CONDITIONS.index(cond) * 1000 + int(100 * rho)) * 1000 + r
                gs1, gs2 = sbm_pair_series(K, N_NODES, rho, correlate=cond, seed=seed)
                s1 = graph_summaries(gs1, seed=seed)
                s2 = graph_summaries(gs2, seed=seed + 500)
                for m in METHODS:
                    if m == "lambda_max":
                        _, p = pearson_perm_test(s1[m], s2[m], N_PERM, seed=seed)
                    else:
                        _, p = dcor_perm_test(s1[m], s2[m], N_PERM, seed=seed)
                    pvals[(rho, m)].append(p)
            done = {m: float(np.mean(np.array(pvals[(rho, m)]) < ALPHA)) for m in METHODS}
            print(f"[{cond}] rho={rho:.1f}  "
                  + "  ".join(f"{m}={v:.2f}" for m, v in done.items())
                  + f"  ({time.time() - t0:.0f}s)")
        power[cond] = {
            m: {str(rho): float(np.mean(np.array(pvals[(rho, m)]) < ALPHA))
                for rho in RHOS}
            for m in METHODS
        }

    for cond in CONDITIONS:
        print(f"\ncondition: correlate={cond} (alpha={ALPHA}, rho=0 = type-I)")
        print(f"{'rho':<8}" + "".join(f"{m:>14}" for m in METHODS))
        for rho in RHOS:
            print(f"{rho:<8}" + "".join(
                f"{power[cond][m][str(rho)]:>14.2f}" for m in METHODS
            ))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "d2_sbm_power.json"
    path.write_text(json.dumps({
        "experiment": "d2_sbm_power",
        "config": {"k_pairs": K, "n_nodes": N_NODES, "rhos": RHOS,
                   "conditions": CONDITIONS, "replicates": R, "n_perm": N_PERM,
                   "alpha": ALPHA, "p_in_range": [0.15, 0.35],
                   "p_out_range": [0.02, 0.10]},
        "power": power,
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
