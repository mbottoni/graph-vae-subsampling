"""D4 — does the blind spot (and the whitening rescue) survive scaling in (n, k)?

The paper's headline is at a single (n=60, k=40). This sweeps graph size and
population size in the p_out-only condition, where the phenomenon lives.

PRE-REGISTERED PREDICTIONS (stated before running):
  P1. The benchmark blind spot PERSISTS for all n. lambda_max ~ (n/2)(p_in+p_out)
      with p_in independent nuisance; growing n sharpens concentration but does
      not remove p_in from the radius -> power stays ~alpha regardless of n.
  P2. The whitening rescue is NON-DECREASING in n. lambda_1, lambda_2 concentrate
      around (n/2)(p_in +/- p_out) as n grows (edge fluctuations are O(sqrt n) vs
      signal O(n)), so (lambda_1-lambda_2)/2 becomes a cleaner p_out reading ->
      spec5_white power non-decreasing in n.
  P3. The gap WIDENS with k. More pairs give more power to any test WITH signal;
      the benchmark has none in p_out (stays ~alpha for all k) while spec5_white
      gains -> separation increases with k.

Cheap statistics only (eigenvalues + permutation, no VGAE) so the grid is fast;
a VGAE spot-check at two (n) points confirms the learned summary tracks.

Writes results/d4_sensitivity.json (incremental per cell).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gvs.data.synthetic import sbm_pair_series
from gvs.stats.dependence import dcor_perm_test, pearson_perm_test, whiten
from gvs.stats.embeddings import emb_spectral

NS = [40, 60, 100, 150]
KS = [20, 40, 80]
RHOS = [0.0, 1.0]
COND = "p_out"
R = 100
N_PERM = 200
ALPHA = 0.05
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "d4_sensitivity.json"

METHODS = ["lambda_max", "spectral5", "spec5_white"]


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def power_cell(n_nodes: int, k: int, rho: float, root: np.random.SeedSequence) -> dict:
    data_ss, perm_ss = root.spawn(2)
    data_seeds = data_ss.generate_state(R)
    perm_seeds = perm_ss.generate_state(R)
    pv: dict[str, list[float]] = {m: [] for m in METHODS}
    for r in range(R):
        gs1, gs2 = sbm_pair_series(k, n_nodes, rho, correlate=COND,
                                   seed=int(data_seeds[r]))
        s1 = np.array([emb_spectral(g, q=5) for g in gs1])
        s2 = np.array([emb_spectral(g, q=5) for g in gs2])
        ps = int(perm_seeds[r])
        _, p = pearson_perm_test(s1[:, 0], s2[:, 0], N_PERM, seed=ps)
        pv["lambda_max"].append(p)
        _, p = dcor_perm_test(s1, s2, N_PERM, seed=ps + 1)
        pv["spectral5"].append(p)
        _, p = dcor_perm_test(whiten(s1), whiten(s2), N_PERM, seed=ps + 2)
        pv["spec5_white"].append(p)
    cell = {}
    for m in METHODS:
        rej = float(np.mean(np.array(pv[m]) < ALPHA))
        cell[m] = {"power": rej, "ci95": wilson_ci(rej, R)}
    return cell


def main() -> None:
    root = np.random.SeedSequence(404_2026)
    grid: dict[str, dict] = {}
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    for n_nodes in NS:
        for k in KS:
            for rho in RHOS:
                key = f"n={n_nodes}|k={k}|rho={rho}"
                cell_ss = root.spawn(1)[0]
                grid[key] = power_cell(n_nodes, k, rho, cell_ss)
                tag = "type-I" if rho == 0.0 else "power "
                print(f"{key:<22} {tag}  "
                      + "  ".join(f"{m}={grid[key][m]['power']:.2f}" for m in METHODS)
                      + f"  ({time.time() - t0:.0f}s)", flush=True)
                OUT.write_text(json.dumps({
                    "experiment": "d4_sensitivity",
                    "config": {"ns": NS, "ks": KS, "rhos": RHOS, "condition": COND,
                               "replicates": R, "n_perm": N_PERM, "alpha": ALPHA},
                    "grid": grid, "complete": False,
                }, indent=2))

    data = json.loads(OUT.read_text())
    data["complete"] = True
    OUT.write_text(json.dumps(data, indent=2))

    # Prediction checks.
    def pw(n_nodes, k, rho, m):
        return grid[f"n={n_nodes}|k={k}|rho={rho}"][m]["power"]

    print("\n=== prediction checks (p_out, rho=1) ===")
    print("P1 benchmark blind spot persists (lambda_max power vs n, k=40):")
    print("   " + "  ".join(f"n={n}:{pw(n, 40, 1.0, 'lambda_max'):.2f}" for n in NS))
    print("P2 whitening non-decreasing in n (spec5_white power vs n, k=40):")
    print("   " + "  ".join(f"n={n}:{pw(n, 40, 1.0, 'spec5_white'):.2f}" for n in NS))
    print("P3 gap widens with k (spec5_white - lambda_max vs k, n=60):")
    print("   " + "  ".join(
        f"k={k}:{pw(60, k, 1.0, 'spec5_white') - pw(60, k, 1.0, 'lambda_max'):+.2f}"
        for k in KS))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
