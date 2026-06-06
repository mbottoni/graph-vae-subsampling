"""D3c — diagnose vgae_sv's type-I inflation (0.10 [0.06, 0.17] at R=100).

Permutation tests are exact for iid pairs, so inflation should be impossible.
Suspect: seed coupling in the experiment harness, not the statistic itself —
in d3_scaled the SAME base seed drives (a) the data-generating RNG, (b) VGAE
training seeds (seed+j, which also collide with neighboring replicates' data
seeds), and (c) the permutation RNG. Under coupling, permutations are not
independent of the data across the ensemble and replicates are not mutually
independent.

This script reruns the rho=0 cell (pure null) with FULLY DECOUPLED streams:
independent SeedSequence-spawned generators for data / training / permutations.

  decoupled type-I ~ 0.05  -> coupling confirmed; fix harness, rerun headline
  decoupled type-I ~ 0.10  -> something deeper; investigate the statistic

Also evaluates a Bonferroni combined test, reject if
min(p_spec5w, p_vgae) < alpha/2 — candidate replacement for the diluted
concatenation statistic. Stores all p-values.

Writes results/d3c_type1_diagnostic.json.
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
R = 100
N_PERM = 200
ALPHA = 0.05
RESULTS = Path(__file__).resolve().parent.parent / "results"

METHODS = ["lambda_max", "spec5_white", "vgae_sv", "bonferroni_min"]


def main() -> None:
    # Three mutually independent seed streams.
    ss = np.random.SeedSequence(20260606)
    data_ss, train_ss, perm_ss = ss.spawn(3)
    data_seeds = data_ss.generate_state(R)
    perm_seeds = perm_ss.generate_state(R)
    train_seeds = train_ss.generate_state(R * 2 * K).reshape(R, 2, K)

    pv: dict[str, list[float]] = {m: [] for m in METHODS}
    t0 = time.time()
    for r in range(R):
        gs1, gs2 = sbm_pair_series(
            K, N_NODES, rho=0.0, correlate="both", seed=int(data_seeds[r])
        )
        spec1 = np.array([emb_spectral(g, q=5) for g in gs1])
        spec2 = np.array([emb_spectral(g, q=5) for g in gs2])
        vg1 = np.array([emb_vgae_sv(g, seed=int(train_seeds[r, 0, j]))
                        for j, g in enumerate(gs1)])
        vg2 = np.array([emb_vgae_sv(g, seed=int(train_seeds[r, 1, j]))
                        for j, g in enumerate(gs2)])

        ps = int(perm_seeds[r])
        _, p_lm = pearson_perm_test(spec1[:, 0], spec2[:, 0], N_PERM, seed=ps)
        _, p_sw = dcor_perm_test(whiten(spec1), whiten(spec2), N_PERM, seed=ps + 1)
        _, p_vg = dcor_perm_test(vg1, vg2, N_PERM, seed=ps + 2)
        pv["lambda_max"].append(p_lm)
        pv["spec5_white"].append(p_sw)
        pv["vgae_sv"].append(p_vg)
        pv["bonferroni_min"].append(min(p_sw, p_vg) * 2)  # Bonferroni-adjusted

        if (r + 1) % 20 == 0:
            rates = {m: float(np.mean(np.array(pv[m]) < ALPHA)) for m in METHODS}
            print(f"{r + 1}/{R}  " + "  ".join(f"{m}={v:.2f}" for m, v in rates.items())
                  + f"  ({time.time() - t0:.0f}s)", flush=True)

    rates = {m: float(np.mean(np.array(pv[m]) < ALPHA)) for m in METHODS}
    print("\ntype-I at alpha=0.05, decoupled seed streams, R=100:")
    for m, v in rates.items():
        print(f"  {m:<16} {v:.3f}")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "d3c_type1_diagnostic.json"
    path.write_text(json.dumps({
        "experiment": "d3c_type1_diagnostic",
        "config": {"k_pairs": K, "n_nodes": N_NODES, "rho": 0.0, "replicates": R,
                   "n_perm": N_PERM, "alpha": ALPHA,
                   "seeding": "SeedSequence(20260606).spawn(3): data/train/perm"},
        "type1": rates,
        "pvalues": pv,
    }, indent=2))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
