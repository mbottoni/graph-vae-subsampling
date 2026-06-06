"""D2b — kill-or-confirm: strong fixed baselines vs the learned summary.

The D2 headline (vgae_sv 0.96 vs spectral5 0.24 in the p_out-only condition)
must survive baselines a reviewer would demand:

  full_spec   : dCor on ALL n adjacency eigenvalues
  spec5_white : dCor on ZCA-whitened top-5 eigenvalues — the direct mechanism
                test: whitening equalizes the nuisance (p_in) and signal (p_out)
                directions, so if the failure is geometric, whitening should
                rescue the fixed spectrum
  netlsd      : dCor on NetLSD heat-trace signatures
  hsic_spec5  : HSIC (Gaussian, median heuristic) on top-5 eigenvalues —
                kernel dependence measure instead of dCor

Same generator, conditions, seeds, and protocol as D2 (experiments/d2_sbm_power.py),
so columns are directly comparable across the two result files.

Writes results/d2b_kill_or_confirm.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gvs.data.synthetic import sbm_pair_series
from gvs.stats.dependence import dcor_perm_test, hsic_perm_test, pearson_perm_test, whiten
from gvs.stats.embeddings import emb_full_spectrum, emb_netlsd, emb_vgae_sv

K, N_NODES = 40, 60
RHOS = [0.0, 0.5, 1.0]
CONDITIONS = ["both", "p_out"]
R = 25
N_PERM = 200
ALPHA = 0.05
RESULTS = Path(__file__).resolve().parent.parent / "results"

METHODS = ["lambda_max", "spectral5", "spec5_white", "full_spec",
           "netlsd", "hsic_spec5", "vgae_sv"]


def main() -> None:
    power: dict[str, dict] = {}
    t0 = time.time()

    for cond in CONDITIONS:
        pvals: dict[tuple[float, str], list[float]] = {
            (rho, m): [] for rho in RHOS for m in METHODS
        }
        for rho in RHOS:
            for r in range(R):
                # Same deterministic seeds as d2_sbm_power.py for comparability.
                seed = (CONDITIONS.index(cond) * 1000 + int(100 * rho)) * 1000 + r
                gs1, gs2 = sbm_pair_series(K, N_NODES, rho, correlate=cond, seed=seed)

                spec1 = np.array([emb_full_spectrum(g) for g in gs1])
                spec2 = np.array([emb_full_spectrum(g) for g in gs2])
                net1 = np.array([emb_netlsd(g) for g in gs1])
                net2 = np.array([emb_netlsd(g) for g in gs2])
                vg1 = np.array([emb_vgae_sv(g, seed=seed + j) for j, g in enumerate(gs1)])
                vg2 = np.array([emb_vgae_sv(g, seed=seed + 500 + j) for j, g in enumerate(gs2)])

                tests = {
                    "lambda_max": lambda: pearson_perm_test(
                        spec1[:, 0], spec2[:, 0], N_PERM, seed=seed),
                    "spectral5": lambda: dcor_perm_test(
                        spec1[:, :5], spec2[:, :5], N_PERM, seed=seed),
                    "spec5_white": lambda: dcor_perm_test(
                        whiten(spec1[:, :5]), whiten(spec2[:, :5]), N_PERM, seed=seed),
                    "full_spec": lambda: dcor_perm_test(
                        spec1, spec2, N_PERM, seed=seed),
                    "netlsd": lambda: dcor_perm_test(
                        net1, net2, N_PERM, seed=seed),
                    "hsic_spec5": lambda: hsic_perm_test(
                        spec1[:, :5], spec2[:, :5], N_PERM, seed=seed),
                    "vgae_sv": lambda: dcor_perm_test(
                        vg1, vg2, N_PERM, seed=seed),
                }
                for m in METHODS:
                    _, p = tests[m]()
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
        print(f"{'rho':<8}" + "".join(f"{m:>13}" for m in METHODS))
        for rho in RHOS:
            print(f"{rho:<8}" + "".join(
                f"{power[cond][m][str(rho)]:>13.2f}" for m in METHODS
            ))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "d2b_kill_or_confirm.json"
    path.write_text(json.dumps({
        "experiment": "d2b_kill_or_confirm",
        "config": {"k_pairs": K, "n_nodes": N_NODES, "rhos": RHOS,
                   "conditions": CONDITIONS, "replicates": R, "n_perm": N_PERM,
                   "alpha": ALPHA, "seeds": "same as d2_sbm_power"},
        "power": power,
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
