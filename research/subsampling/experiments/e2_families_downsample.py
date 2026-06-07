"""E2 — downsample structured graph families: BA, WS, SBM (n=100 -> m=50).

E1 showed that on ER, uniform node sampling trivially wins (induced ER is ER)
and latent methods over-cluster. E2 asks the interesting question: on graphs
with real structure — hubs (BA), local clustering (WS), communities (SBM) —
do latent methods preserve family-specific properties better than node sampling,
which is known to distort them (e.g., subsampled BA is not BA)?

Adds modularity (community structure) to the E1 metric set.

Writes results/e2_families_downsample.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from gvs.data.synthetic import barabasi_albert, stochastic_block_model, to_pyg, watts_strogatz
from gvs.metrics.distances import normalized_degree_distance
from gvs.metrics.graph_stats import degree_sequence, summary
from subsampling.decoders import latent_downsample
from gvs.models.vgae import encode_latents, train_vgae
from subsampling.baselines import forest_fire, random_walk, uniform_node

N, M_SUB = 100, 50
R = 10
RESULTS = Path(__file__).resolve().parent.parent / "results"

FAMILIES = {
    "ba": lambda seed: barabasi_albert(N, 5, seed=seed),
    "ws": lambda seed: watts_strogatz(N, 10, 0.1, seed=seed),
    "sbm": lambda seed: stochastic_block_model([N // 2, N // 2], 0.18, 0.02, seed=seed),
}
LATENT_METHODS = ["random", "kmeans", "posterior"]
BASELINES = {
    "uniform_node": uniform_node,
    "random_walk": random_walk,
    "forest_fire": forest_fire,
}
METRICS = ["density", "clustering", "modularity", "lambda_max_norm", "degree_w1"]


def modularity(g: nx.Graph) -> float:
    if g.number_of_edges() == 0:
        return float("nan")
    comms = nx.algorithms.community.greedy_modularity_communities(g)
    return float(nx.algorithms.community.modularity(g, comms))


def evaluate(g_orig: nx.Graph, g_small: nx.Graph) -> dict[str, float]:
    s = summary(g_small)
    return {
        "density": s["density"],
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
        originals: list[dict] = []
        for r in range(R):
            torch.manual_seed(r)
            np.random.seed(r)
            g = make(r)
            data = to_pyg(g)
            density = 2 * g.number_of_edges() / (N * (N - 1))
            result = train_vgae(data, epochs=300, seed=r)
            z = encode_latents(result.model, data)
            s = summary(g)
            originals.append({
                "density": s["density"],
                "clustering": s["clustering"],
                "modularity": modularity(g),
                "lambda_max_norm": s["lambda_max"] / N,
                "auc": result.auc,
            })

            for method in LATENT_METHODS:
                g_small = latent_downsample(z, M_SUB, density, method=method, seed=r)
                records.append({"method": f"latent_{method}"} | evaluate(g, g_small))
            for name, fn in BASELINES.items():
                records.append({"method": name} | evaluate(g, fn(g, M_SUB, seed=r)))

        methods = [f"latent_{m}" for m in LATENT_METHODS] + list(BASELINES)
        agg = {
            method: {
                k: {
                    "mean": float(np.nanmean(
                        [x[k] for x in records if x["method"] == method]
                    )),
                    "std": float(np.nanstd(
                        [x[k] for x in records if x["method"] == method]
                    )),
                }
                for k in METRICS
            }
            for method in methods
        }
        orig = {
            k: float(np.nanmean([o[k] for o in originals]))
            for k in ["density", "clustering", "modularity", "lambda_max_norm", "auc"]
        }
        out[family] = {"original": orig, "methods": agg}

        print(f"\n=== {family.upper()} (n={N} -> m={M_SUB}, R={R}, "
              f"VGAE AUC={orig['auc']:.3f}) ===")
        print(f"{'method':<18}" + "".join(f"{k:>18}" for k in METRICS))
        print(f"{'(original)':<18}"
              + f"{orig['density']:>18.4f}{orig['clustering']:>18.4f}"
              + f"{orig['modularity']:>18.4f}{orig['lambda_max_norm']:>18.4f}"
              + f"{'—':>18}")
        for method in methods:
            a = agg[method]
            print(f"{method:<18}" + "".join(
                f"{a[k]['mean']:>11.4f}±{a[k]['std']:<6.4f}" for k in METRICS
            ))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "e2_families_downsample.json"
    path.write_text(json.dumps({
        "experiment": "e2_families_downsample",
        "config": {"n": N, "m": M_SUB, "replicates": R, "epochs": 300,
                   "families": {"ba": "BA(m=5)", "ws": "WS(k=10, p=0.1)",
                                "sbm": "SBM([50,50], 0.18, 0.02)"}},
        "results": out,
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
