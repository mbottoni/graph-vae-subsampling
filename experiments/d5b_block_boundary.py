"""D5b — trace the blind-spot boundary across block count.

D5 found the benchmark blind spot at k=2 blocks but gone at k=4. The mechanism:
in a balanced b-block SBM the spectral radius is
    lambda_max ~ (n/b) p_in + n (b-1)/b p_out,
so p_out's weight relative to p_in's is (b-1):1. The benchmark is blind iff
nuisance beats signal in the radius, i.e. (b-1) Delta p_out < Delta p_in.

PRE-REGISTERED (Delta p_in=0.20, Delta p_out=0.08): blind threshold at
(b-1) = 0.20/0.08 = 2.5, i.e. b <= 3 blind, b >= 4 detects. Predicted
lambda_max power: low at b in {2,3}, rising sharply at b in {4,5,6}.

Cheap stats only (lambda_max, spec5_white), R=100, decoupled seeds.

Writes results/d5b_block_boundary.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from gvs.data.synthetic import dc_sbm_pair_series
from gvs.stats.dependence import dcor_perm_test, pearson_perm_test, whiten
from gvs.stats.embeddings import emb_spectral

N_NODES, K, R = 80, 40, 100
BLOCKS = [2, 3, 4, 5, 6]
RHOS = [0.0, 1.0]
N_PERM = 200
ALPHA = 0.05
RESULTS = Path(__file__).resolve().parent.parent / "results"
OUT = RESULTS / "d5b_block_boundary.json"
METHODS = ["lambda_max", "spec5_white"]


def main() -> None:
    root = np.random.SeedSequence(515_2026)
    results: dict[str, dict] = {}
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)

    for b in BLOCKS:
        results[str(b)] = {}
        for rho in RHOS:
            cell_ss = root.spawn(1)[0]
            data_ss, perm_ss = cell_ss.spawn(2)
            data_seeds = data_ss.generate_state(R)
            perm_seeds = perm_ss.generate_state(R)
            pv: dict[str, list[float]] = {m: [] for m in METHODS}
            for r in range(R):
                gs1, gs2 = dc_sbm_pair_series(
                    K, N_NODES, rho, n_blocks=b, degree_hetero=0.0,
                    seed=int(data_seeds[r])
                )
                s1 = np.array([emb_spectral(g, q=5) for g in gs1])
                s2 = np.array([emb_spectral(g, q=5) for g in gs2])
                ps = int(perm_seeds[r])
                _, p = pearson_perm_test(s1[:, 0], s2[:, 0], N_PERM, seed=ps)
                pv["lambda_max"].append(p)
                _, p = dcor_perm_test(whiten(s1), whiten(s2), N_PERM, seed=ps + 1)
                pv["spec5_white"].append(p)
            cell = {m: float(np.mean(np.array(pv[m]) < ALPHA)) for m in METHODS}
            results[str(b)][str(rho)] = cell
            tag = "type-I" if rho == 0.0 else "power "
            ratio = (b - 1) * 0.08
            print(f"b={b} (sig/nui ratio {ratio:.2f} vs Dp_in 0.20) rho={rho:.0f} "
                  f"{tag}  " + "  ".join(f"{m}={cell[m]:.2f}" for m in METHODS)
                  + f"  ({time.time() - t0:.0f}s)", flush=True)
            OUT.write_text(json.dumps({
                "experiment": "d5b_block_boundary",
                "config": {"n_nodes": N_NODES, "k_pairs": K, "replicates": R,
                           "blocks": BLOCKS, "rhos": RHOS, "alpha": ALPHA,
                           "dp_in": 0.20, "dp_out": 0.08,
                           "predicted_threshold_blocks": 3.5},
                "results": results, "complete": False,
            }, indent=2))

    data = json.loads(OUT.read_text())
    data["complete"] = True
    OUT.write_text(json.dumps(data, indent=2))

    print("\n=== blind-spot boundary (rho=1 lambda_max power vs blocks) ===")
    print("predicted: blind for b<=3, detects for b>=4")
    for b in BLOCKS:
        print(f"  b={b}: lambda_max={results[str(b)]['1.0']['lambda_max']:.2f}  "
              f"spec5_white={results[str(b)]['1.0']['spec5_white']:.2f}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
