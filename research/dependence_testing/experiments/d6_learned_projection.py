"""D6 — a learned projection (CCA) as an adaptive dependence test.

The whitened spectrum fixes the blind spot but COSTS power when dependence is
spectrum-aligned (D3d: both/rho=0.5 spec5_white 0.65 vs raw lambda_max 0.90).
That tradeoff exists because whitening is a fixed, dependence-agnostic
re-metrization. The principled alternative: LEARN the maximally-dependent linear
projection of the spectral features (canonical correlation analysis), with
sample splitting so the permutation null stays exact.

PRE-REGISTERED PREDICTION: cca_split is calibrated (type-I ~0.05 via splitting)
and ADAPTIVE — it finds the radius direction when dependence is aligned and the
(lambda_1-lambda_2) direction in the blind spot, so it matches or beats both the
benchmark (aligned) and the whitened spectrum (blind spot) at rho=0.5, paying
only the sample-splitting price (half the effective pairs). If instead it
underperforms both, the splitting cost outweighs the adaptivity.

Compared: lambda_max (benchmark), spec5_white (current best fixed), cca_split.
SBM both/p_out, rho in {0, 0.5, 1}, R=100, decoupled seeds.

Writes results/d6_learned_projection.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from deptest.pairs import sbm_pair_series
from deptest.dependence import (
    cca_crossfit_test,
    cca_split_test,
    dcor_perm_test,
    pearson_perm_test,
    whiten,
)
from deptest.embeddings import emb_spectral

N_NODES, K, R = 60, 40, 100
RHOS = [0.0, 0.5, 1.0]
CONDITIONS = ["both", "p_out"]
N_PERM = 200
ALPHA = 0.05
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "d6_learned_projection.json"
METHODS = ["lambda_max", "spec5_white", "cca_split", "cca_crossfit"]


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def main() -> None:
    root = np.random.SeedSequence(606_2026)
    power: dict[str, dict] = {}
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    for cond in CONDITIONS:
        power[cond] = {}
        for rho in RHOS:
            cell_ss = root.spawn(1)[0]
            data_ss, perm_ss = cell_ss.spawn(2)
            data_seeds = data_ss.generate_state(R)
            perm_seeds = perm_ss.generate_state(R)
            pv: dict[str, list[float]] = {m: [] for m in METHODS}
            for r in range(R):
                gs1, gs2 = sbm_pair_series(K, N_NODES, rho, correlate=cond,
                                           seed=int(data_seeds[r]))
                s1 = np.array([emb_spectral(g, q=5) for g in gs1])
                s2 = np.array([emb_spectral(g, q=5) for g in gs2])
                ps = int(perm_seeds[r])
                _, p = pearson_perm_test(s1[:, 0], s2[:, 0], N_PERM, seed=ps)
                pv["lambda_max"].append(p)
                _, p = dcor_perm_test(whiten(s1), whiten(s2), N_PERM, seed=ps + 1)
                pv["spec5_white"].append(p)
                _, p = cca_split_test(s1, s2, N_PERM, seed=ps + 2)
                pv["cca_split"].append(p)
                _, p = cca_crossfit_test(s1, s2, N_PERM, seed=ps + 3)
                pv["cca_crossfit"].append(p)
            cell = {}
            for m in METHODS:
                rej = float(np.mean(np.array(pv[m]) < ALPHA))
                cell[m] = {"power": rej, "ci95": wilson_ci(rej, R)}
            power[cond][str(rho)] = cell
            tag = "type-I" if rho == 0.0 else "power "
            print(f"[{cond}] rho={rho:.1f} {tag}  "
                  + "  ".join(f"{m}={cell[m]['power']:.2f}" for m in METHODS)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            OUT.write_text(json.dumps({
                "experiment": "d6_learned_projection",
                "config": {"n_nodes": N_NODES, "k_pairs": K, "replicates": R,
                           "rhos": RHOS, "conditions": CONDITIONS,
                           "n_perm": N_PERM, "alpha": ALPHA,
                           "note": "cca_split learns projection on half, tests on half"},
                "power": power, "complete": False,
            }, indent=2))

    data = json.loads(OUT.read_text())
    data["complete"] = True
    OUT.write_text(json.dumps(data, indent=2))

    print("\n=== adaptive test: does CCA get best-of-both at rho=0.5? ===")
    for cond in CONDITIONS:
        c = power[cond]["0.5"]
        print(f"{cond:6} rho=0.5  lambda_max={c['lambda_max']['power']:.2f}  "
              f"spec5_white={c['spec5_white']['power']:.2f}  "
              f"cca_split={c['cca_split']['power']:.2f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
