"""D3d — definitive headline tables: R=100, decoupled seed streams.

Replaces d3_scaled (demoted to preliminary after D3c showed its seed scheme
couples data / VGAE-training / permutation randomness, mildly inflating
vgae_sv's type-I to ~0.10). Here every replicate draws from three independent
SeedSequence-spawned streams, the scheme validated in D3c (type-I exactly 0.05).

Statistics: benchmark, raw spectrum, whitened spectrum, learned SVs, and TWO
combiners — concatenation (diluted in d3_scaled interim: 0.63 vs 0.77 envelope
at rho=0.5) and Bonferroni min-p (calibrated at 0.02 in D3c).

Writes results/d3d_definitive.json (incremental saves per block).
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
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "d3d_definitive.json"

METHODS = ["lambda_max", "spectral5", "spec5_white", "vgae_sv",
           "concat", "bonferroni_min"]


def wilson_ci(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return float(center - half), float(center + half)


def main() -> None:
    root = np.random.SeedSequence(126_2026)
    power: dict[str, dict] = {}
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    for cond in CONDITIONS:
        power[cond] = {}
        for rho in RHOS:
            cell_ss = root.spawn(1)[0]
            data_ss, train_ss, perm_ss = cell_ss.spawn(3)
            data_seeds = data_ss.generate_state(R)
            perm_seeds = perm_ss.generate_state(R)
            train_seeds = train_ss.generate_state(R * 2 * K).reshape(R, 2, K)

            pv: dict[str, list[float]] = {m: [] for m in METHODS}
            for r in range(R):
                gs1, gs2 = sbm_pair_series(
                    K, N_NODES, rho, correlate=cond, seed=int(data_seeds[r])
                )
                spec1 = np.array([emb_spectral(g, q=5) for g in gs1])
                spec2 = np.array([emb_spectral(g, q=5) for g in gs2])
                vg1 = np.array([emb_vgae_sv(g, seed=int(train_seeds[r, 0, j]))
                                for j, g in enumerate(gs1)])
                vg2 = np.array([emb_vgae_sv(g, seed=int(train_seeds[r, 1, j]))
                                for j, g in enumerate(gs2)])
                sw1, sw2 = whiten(spec1), whiten(spec2)

                ps = int(perm_seeds[r])
                _, p = pearson_perm_test(spec1[:, 0], spec2[:, 0], N_PERM, seed=ps)
                pv["lambda_max"].append(p)
                _, p = dcor_perm_test(spec1, spec2, N_PERM, seed=ps + 1)
                pv["spectral5"].append(p)
                _, p_sw = dcor_perm_test(sw1, sw2, N_PERM, seed=ps + 2)
                pv["spec5_white"].append(p_sw)
                _, p_vg = dcor_perm_test(vg1, vg2, N_PERM, seed=ps + 3)
                pv["vgae_sv"].append(p_vg)
                _, p = dcor_perm_test(np.hstack([sw1, whiten(vg1)]),
                                      np.hstack([sw2, whiten(vg2)]),
                                      N_PERM, seed=ps + 4)
                pv["concat"].append(p)
                pv["bonferroni_min"].append(min(p_sw, p_vg) * 2)

                if (r + 1) % 25 == 0:
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

            OUT.write_text(json.dumps({
                "experiment": "d3d_definitive",
                "config": {"k_pairs": K, "n_nodes": N_NODES, "rhos": RHOS,
                           "conditions": CONDITIONS, "replicates": R,
                           "n_perm": N_PERM, "alpha": ALPHA,
                           "seeding": "SeedSequence(1262026), spawned per cell:"
                                      " data/train/perm (D3c-validated)"},
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
