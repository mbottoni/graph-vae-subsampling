"""E4 — scale-up: n=1000, sample fractions 5% and 10%.

E1-E3 were n=100 -> 50% samples, a regime where uniform node sampling is hard
to beat. The classical-sampling literature (Leskovec & Faloutsos 2005) says node
sampling degrades at small sample fractions of larger graphs — exactly the
regime the latent approach was designed for. E4 tests whether the E3 winners
(vanilla+bernoulli, dc+bernoulli) overtake the classical baselines there.

Families (mean degree ~10, matching the E1-E3 density regime):
  ER(1000, 0.01), BA(1000, 5), SBM(4 x 250, p_in=0.03, p_out=0.003)

Writes results/e4_scale.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from gvs.data.synthetic import barabasi_albert, erdos_renyi, stochastic_block_model, to_pyg
from gvs.metrics.distances import normalized_degree_distance
from gvs.metrics.graph_stats import degree_sequence, summary
from gvs.models.decoders import latent_downsample
from gvs.models.vgae import encode_latents, node_biases, train_vgae
from gvs.sampling.baselines import forest_fire, random_walk, uniform_node

N, MS, R = 1000, [50, 100], 5
RESULTS = Path(__file__).resolve().parent.parent / "results"

FAMILIES = {
    "er": lambda seed: erdos_renyi(N, 0.01, seed=seed),
    "ba": lambda seed: barabasi_albert(N, 5, seed=seed),
    "sbm": lambda seed: stochastic_block_model([N // 4] * 4, 0.03, 0.003, seed=seed),
}
LATENT_VARIANTS = [("vanilla_bern", False), ("dc_bern", True)]
BASELINES = {
    "uniform_node": uniform_node,
    "random_walk": random_walk,
    "forest_fire": forest_fire,
}
METRICS = ["clustering", "modularity", "lambda_max_norm", "degree_w1"]


def modularity(g: nx.Graph) -> float:
    if g.number_of_edges() == 0:
        return float("nan")
    comms = nx.algorithms.community.greedy_modularity_communities(g)
    return float(nx.algorithms.community.modularity(g, comms))


def evaluate(g_orig: nx.Graph, g_small: nx.Graph) -> dict[str, float]:
    s = summary(g_small)
    return {
        "clustering": s["clustering"],
        "modularity": modularity(g_small),
        "lambda_max_norm": s["lambda_max"] / s["n_nodes"],
        "degree_w1": normalized_degree_distance(
            degree_sequence(g_orig), degree_sequence(g_small)
        ),
    }


def main() -> None:
    out: dict[str, dict] = {}
    for family, make in FAMILIES.items():
        records: list[dict] = []
        aucs: dict[str, list[float]] = {"vanilla": [], "dc": []}
        originals: list[dict] = []
        t0 = time.time()

        for r in range(R):
            torch.manual_seed(r)
            np.random.seed(r)
            g = make(r)
            data = to_pyg(g)
            density = nx.density(g)
            s = summary(g)
            originals.append({
                "clustering": s["clustering"],
                "modularity": modularity(g),
                "lambda_max_norm": s["lambda_max"] / N,
            })

            latents = {}
            for kind, dc in [("vanilla", False), ("dc", True)]:
                res = train_vgae(data, epochs=300, seed=r, degree_corrected=dc)
                aucs[kind].append(res.auc)
                latents[kind] = (encode_latents(res.model, data), node_biases(res.model))

            for m in MS:
                for label, dc in LATENT_VARIANTS:
                    z, b = latents["dc" if dc else "vanilla"]
                    g_small = latent_downsample(
                        z, m, density, method="random", seed=r, bias=b, decode="bernoulli"
                    )
                    records.append({"method": label, "m": m} | evaluate(g, g_small))
                for name, fn in BASELINES.items():
                    records.append(
                        {"method": name, "m": m} | evaluate(g, fn(g, m, seed=r))
                    )
            print(f"  {family} replicate {r + 1}/{R} "
                  f"({time.time() - t0:.0f}s elapsed)")

        methods = [v[0] for v in LATENT_VARIANTS] + list(BASELINES)
        agg = {}
        for m in MS:
            for method in methods:
                rows = [x for x in records if x["method"] == method and x["m"] == m]
                agg[f"{method}|m={m}"] = {
                    k: {
                        "mean": float(np.nanmean([x[k] for x in rows])),
                        "std": float(np.nanstd([x[k] for x in rows])),
                    }
                    for k in METRICS
                }
        orig = {k: float(np.nanmean([o[k] for o in originals]))
                for k in ["clustering", "modularity", "lambda_max_norm"]}
        out[family] = {
            "original": orig,
            "auc": {k: float(np.mean(v)) for k, v in aucs.items()},
            "methods": agg,
        }

        print(f"\n=== {family.upper()} n={N} (AUC vanilla={out[family]['auc']['vanilla']:.3f} "
              f"dc={out[family]['auc']['dc']:.3f}) ===")
        print(f"{'(original)':<22}{orig['clustering']:>18.4f}{orig['modularity']:>18.4f}"
              f"{orig['lambda_max_norm']:>18.4f}{'—':>18}")
        for m in MS:
            print(f"--- m = {m} ({100 * m // N}%) ---")
            print(f"{'method':<22}" + "".join(f"{k:>18}" for k in METRICS))
            for method in methods:
                a = agg[f"{method}|m={m}"]
                print(f"{method:<22}" + "".join(
                    f"{a[k]['mean']:>11.4f}±{a[k]['std']:<6.4f}" for k in METRICS
                ))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "e4_scale.json"
    path.write_text(json.dumps({
        "experiment": "e4_scale",
        "config": {"n": N, "ms": MS, "replicates": R, "epochs": 300,
                   "families": {"er": "ER(1000, 0.01)", "ba": "BA(1000, 5)",
                                "sbm": "SBM(4x250, 0.03, 0.003)"}},
        "results": out,
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
