"""D3-scaled — the definitive version of the headline tables: R=100 + combined statistic.

Two purposes (plan priorities 2 and 3):
1. R=100 replicates per cell -> binomial SE <= 0.05, settling (a) vgae_sv's
   type-I rate (0.08 at R=25 — must come back ~0.05) and (b) tight CIs on the
   0.96-vs-0.24 headline.
2. The combined statistic: concatenation of the separately-whitened truncated
   spectrum and VGAE singular values. If the two summaries fail in complementary
   regimes, the combination should track the upper envelope in both conditions.

Methods kept: the benchmark, the raw spectrum, the two winners, the combination.
(full_spec / netlsd / hsic were settled in D2b and are dropped.)

Fresh seed base (offset from D2/D2b) — this is an independent confirmation, not
a re-analysis of the same draws. Partial results are written to JSON after every
(condition, rho) block so a long run can be inspected/recovered mid-flight.

Writes results/d3_scaled.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gvs.data.synthetic import sbm_pair_series
from gvs.stats.dependence import dcor_perm_test, pearson_perm_test, whiten
from gvs.stats.embeddings import emb_spectral, emb_vgae_sv

K, N_NODES = 40, 60
RHOS = [0.0, 0.5, 1.0]
CONDITIONS = ["both", "p_out"]
R = 100
N_PERM = 200
ALPHA = 0.05
SEED_BASE = 7_000_000  # fresh draws, independent of D2/D2b
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "d3_scaled.json"

METHODS = ["lambda_max", "spectral5", "spec5_white", "vgae_sv", "combined"]


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def main() -> None:
    power: dict[str, dict] = {}
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    for cond in CONDITIONS:
        power[cond] = {}
        for rho in RHOS:
            pv: dict[str, list[float]] = {m: [] for m in METHODS}
            for r in range(R):
                seed = SEED_BASE + (CONDITIONS.index(cond) * 1000
                                    + int(100 * rho)) * 1000 + r
                gs1, gs2 = sbm_pair_series(K, N_NODES, rho, correlate=cond, seed=seed)

                spec1 = np.array([emb_spectral(g, q=5) for g in gs1])
                spec2 = np.array([emb_spectral(g, q=5) for g in gs2])
                vg1 = np.array([emb_vgae_sv(g, seed=seed + j)
                                for j, g in enumerate(gs1)])
                vg2 = np.array([emb_vgae_sv(g, seed=seed + 500 + j)
                                for j, g in enumerate(gs2)])
                sw1, sw2 = whiten(spec1), whiten(spec2)
                comb1 = np.hstack([sw1, whiten(vg1)])
                comb2 = np.hstack([sw2, whiten(vg2)])

                _, p = pearson_perm_test(spec1[:, 0], spec2[:, 0], N_PERM, seed=seed)
                pv["lambda_max"].append(p)
                _, p = dcor_perm_test(spec1, spec2, N_PERM, seed=seed)
                pv["spectral5"].append(p)
                _, p = dcor_perm_test(sw1, sw2, N_PERM, seed=seed)
                pv["spec5_white"].append(p)
                _, p = dcor_perm_test(vg1, vg2, N_PERM, seed=seed)
                pv["vgae_sv"].append(p)
                _, p = dcor_perm_test(comb1, comb2, N_PERM, seed=seed)
                pv["combined"].append(p)

                if (r + 1) % 20 == 0:
                    print(f"[{cond}] rho={rho:.1f}  {r + 1}/{R}  "
                          f"({time.time() - t0:.0f}s)", flush=True)

            cell = {}
            for m in METHODS:
                rej = float(np.mean(np.array(pv[m]) < ALPHA))
                lo, hi = wilson_ci(rej, R)
                cell[m] = {"power": rej, "ci95": [lo, hi]}
            power[cond][str(rho)] = cell
            print(f"[{cond}] rho={rho:.1f}  DONE  "
                  + "  ".join(f"{m}={cell[m]['power']:.2f}" for m in METHODS),
                  flush=True)

            # Incremental save so a long run is inspectable/recoverable.
            OUT.write_text(json.dumps({
                "experiment": "d3_scaled",
                "config": {"k_pairs": K, "n_nodes": N_NODES, "rhos": RHOS,
                           "conditions": CONDITIONS, "replicates": R,
                           "n_perm": N_PERM, "alpha": ALPHA,
                           "seed_base": SEED_BASE,
                           "combined": "hstack(whiten(spec5), whiten(vgae_sv))"},
                "power": power,
                "complete": False,
            }, indent=2))

    data = json.loads(OUT.read_text())
    data["complete"] = True
    OUT.write_text(json.dumps(data, indent=2))

    print(f"\nrejection rates at alpha={ALPHA}, R={R} (95% Wilson CI)")
    for cond in CONDITIONS:
        print(f"\ncondition: correlate={cond}")
        print(f"{'rho':<6}" + "".join(f"{m:>22}" for m in METHODS))
        for rho in RHOS:
            cells = power[cond][str(rho)]
            print(f"{rho:<6}" + "".join(
                f"{c['power']:.2f} [{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]".rjust(22)
                for c in (cells[m] for m in METHODS)
            ))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
