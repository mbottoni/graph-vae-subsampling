"""E3 — decoder ablation: can we fix the over-clustering bias?

E1/E2 finding: M1 with the vanilla inner-product decoder + top-k decoding
inflates clustering 2-5x across all graph families. Two candidate fixes,
ablated factorially on ER / BA / WS / SBM (n=100 -> m=50, R=10):

  decoder:  vanilla            vs  degree-corrected (per-node biases)
  decode:   topk (determ.)     vs  bernoulli (density-calibrated sampling)

Latent selection fixed to "random" (simplest, representative of E1/E2).
Reference: uniform node sampling + the original graph's statistics.

Writes results/e3_decoder_ablation.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from gvs.data.synthetic import (
    barabasi_albert,
    erdos_renyi,
    stochastic_block_model,
    to_pyg,
    watts_strogatz,
)
from gvs.metrics.distances import normalized_degree_distance
from gvs.metrics.graph_stats import degree_sequence, summary
from subsampling.decoders import latent_downsample
from gvs.models.vgae import encode_latents, node_biases, train_vgae
from subsampling.baselines import uniform_node

N, M_SUB, R = 100, 50, 10
RESULTS = Path(__file__).resolve().parent.parent / "results"

FAMILIES = {
    "er": lambda seed: erdos_renyi(N, 0.1, seed=seed),
    "ba": lambda seed: barabasi_albert(N, 5, seed=seed),
    "ws": lambda seed: watts_strogatz(N, 10, 0.1, seed=seed),
    "sbm": lambda seed: stochastic_block_model([N // 2, N // 2], 0.18, 0.02, seed=seed),
}
VARIANTS = [  # (label, degree_corrected, decode)
    ("vanilla_topk", False, "topk"),
    ("vanilla_bern", False, "bernoulli"),
    ("dc_topk", True, "topk"),
    ("dc_bern", True, "bernoulli"),
]
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

            models = {}
            for kind, dc in [("vanilla", False), ("dc", True)]:
                res = train_vgae(data, epochs=300, seed=r, degree_corrected=dc)
                aucs[kind].append(res.auc)
                models[kind] = res.model

            for label, dc, decode in VARIANTS:
                model = models["dc" if dc else "vanilla"]
                z = encode_latents(model, data)
                b = node_biases(model)
                g_small = latent_downsample(
                    z, M_SUB, density, method="random", seed=r, bias=b, decode=decode
                )
                records.append({"method": label} | evaluate(g, g_small))
            records.append(
                {"method": "uniform_node"} | evaluate(g, uniform_node(g, M_SUB, seed=r))
            )

        methods = [v[0] for v in VARIANTS] + ["uniform_node"]
        agg = {
            method: {
                k: {
                    "mean": float(np.nanmean([x[k] for x in records if x["method"] == method])),
                    "std": float(np.nanstd([x[k] for x in records if x["method"] == method])),
                }
                for k in METRICS
            }
            for method in methods
        }
        orig = {k: float(np.nanmean([o[k] for o in originals]))
                for k in ["clustering", "modularity", "lambda_max_norm"]}
        out[family] = {
            "original": orig,
            "auc": {k: float(np.mean(v)) for k, v in aucs.items()},
            "methods": agg,
        }

        print(f"\n=== {family.upper()} (AUC vanilla={out[family]['auc']['vanilla']:.3f} "
              f"dc={out[family]['auc']['dc']:.3f}) ===")
        print(f"{'method':<15}" + "".join(f"{k:>18}" for k in METRICS))
        print(f"{'(original)':<15}{orig['clustering']:>18.4f}{orig['modularity']:>18.4f}"
              f"{orig['lambda_max_norm']:>18.4f}{'—':>18}")
        for method in methods:
            a = agg[method]
            print(f"{method:<15}" + "".join(
                f"{a[k]['mean']:>11.4f}±{a[k]['std']:<6.4f}" for k in METRICS
            ))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "e3_decoder_ablation.json"
    path.write_text(json.dumps({
        "experiment": "e3_decoder_ablation",
        "config": {"n": N, "m": M_SUB, "replicates": R, "epochs": 300,
                   "selection": "random", "variants": [v[0] for v in VARIANTS]},
        "results": out,
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
