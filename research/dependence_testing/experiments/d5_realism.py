"""D5 — does the blind spot survive realistic graph structure?

The headline used a clean balanced 2-block SBM. Real connectomes have degree
heterogeneity and many communities. D5 reruns the p_out-only blind-spot test
under four increasingly realistic generators (n=60, k=40, R=100, decoupled
seeds):

  sbm2        : plain 2-block (the original setting; reference)
  dcsbm2      : 2-block + degree heterogeneity (LogNormal theta, s=0.5)
  sbm4        : 4-block, homogeneous degree
  dcsbm4      : 4-block + degree heterogeneity (closest to a connectome)

PRE-REGISTERED PREDICTION: degree heterogeneity and extra blocks add nuisance
VARIANCE without adding p_out signal, so the benchmark stays blind (possibly
worse) and the whitening / learned-summary rescue persists. If instead whitening
collapses under heterogeneity, the method is fragile and the paper must say so.

Statistics: benchmark, raw spectrum, whitened spectrum, learned summary.
rho in {0, 1} (type-I + max power).

Writes results/d5_realism.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from deptest.pairs import dc_sbm_pair_series
from deptest.dependence import dcor_perm_test, pearson_perm_test, whiten
from deptest.embeddings import emb_spectral, emb_vgae_sv

N_NODES, K, R = 60, 40, 100
RHOS = [0.0, 1.0]
N_PERM = 200
ALPHA = 0.05
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "d5_realism.json"

GENERATORS = {
    "sbm2":   {"n_blocks": 2, "degree_hetero": 0.0},
    "dcsbm2": {"n_blocks": 2, "degree_hetero": 0.5},
    "sbm4":   {"n_blocks": 4, "degree_hetero": 0.0},
    "dcsbm4": {"n_blocks": 4, "degree_hetero": 0.5},
}
METHODS = ["lambda_max", "spectral5", "spec5_white", "vgae_sv"]


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def main() -> None:
    root = np.random.SeedSequence(505_2026)
    results: dict[str, dict] = {}
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    for gname, gkw in GENERATORS.items():
        results[gname] = {}
        for rho in RHOS:
            cell_ss = root.spawn(1)[0]
            data_ss, train_ss, perm_ss = cell_ss.spawn(3)
            data_seeds = data_ss.generate_state(R)
            perm_seeds = perm_ss.generate_state(R)
            train_seeds = train_ss.generate_state(R * 2 * K).reshape(R, 2, K)

            pv: dict[str, list[float]] = {m: [] for m in METHODS}
            for r in range(R):
                gs1, gs2 = dc_sbm_pair_series(
                    K, N_NODES, rho, seed=int(data_seeds[r]), **gkw
                )
                s1 = np.array([emb_spectral(g, q=5) for g in gs1])
                s2 = np.array([emb_spectral(g, q=5) for g in gs2])
                vg1 = np.array([emb_vgae_sv(g, seed=int(train_seeds[r, 0, j]))
                                for j, g in enumerate(gs1)])
                vg2 = np.array([emb_vgae_sv(g, seed=int(train_seeds[r, 1, j]))
                                for j, g in enumerate(gs2)])
                ps = int(perm_seeds[r])
                _, p = pearson_perm_test(s1[:, 0], s2[:, 0], N_PERM, seed=ps)
                pv["lambda_max"].append(p)
                _, p = dcor_perm_test(s1, s2, N_PERM, seed=ps + 1)
                pv["spectral5"].append(p)
                _, p = dcor_perm_test(whiten(s1), whiten(s2), N_PERM, seed=ps + 2)
                pv["spec5_white"].append(p)
                _, p = dcor_perm_test(vg1, vg2, N_PERM, seed=ps + 3)
                pv["vgae_sv"].append(p)

            cell = {m: {"power": float(np.mean(np.array(pv[m]) < ALPHA))} for m in METHODS}
            for m in METHODS:
                cell[m]["ci95"] = wilson_ci(cell[m]["power"], R)
            results[gname][str(rho)] = cell
            tag = "type-I" if rho == 0.0 else "power "
            print(f"{gname:<8} rho={rho:.0f} {tag}  "
                  + "  ".join(f"{m}={cell[m]['power']:.2f}" for m in METHODS)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

            OUT.write_text(json.dumps({
                "experiment": "d5_realism",
                "config": {"n_nodes": N_NODES, "k_pairs": K, "replicates": R,
                           "n_perm": N_PERM, "alpha": ALPHA, "rhos": RHOS,
                           "generators": GENERATORS},
                "results": results, "complete": False,
            }, indent=2))

    data = json.loads(OUT.read_text())
    data["complete"] = True
    OUT.write_text(json.dumps(data, indent=2))

    print("\n=== blind spot across realistic generators (rho=1 power) ===")
    print(f"{'generator':<10}" + "".join(f"{m:>14}" for m in METHODS))
    for gname in GENERATORS:
        c = results[gname]["1.0"]
        print(f"{gname:<10}" + "".join(f"{c[m]['power']:>14.2f}" for m in METHODS))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
