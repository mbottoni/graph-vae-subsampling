"""E5 — coarsening vs decoding vs sampling: which downsampling philosophy wins?

Three ways to make a smaller graph from a large one:
  decode   — generate a new graph by decoding selected VGAE latents (E1-E3);
             edges come from the decoder's latent geometry (can hallucinate)
  coarsen  — partition nodes into m supernodes, aggregate REAL edges (coarsening.py)
  sample   — induced subgraph on m sampled nodes (classical baseline)

And the key ablation: does the coarsening PARTITION need to come from the VGAE
latent space (latent_coarsen) or does a random balanced partition do as well
(random_coarsen)?

PRE-REGISTERED PREDICTION: coarsening preserves community/spectral structure
better than decoding or sampling because it aggregates real connectivity; and
latent_coarsen >> random_coarsen on modularity (the latent partition respects
communities, a random one mixes them). If latent ~ random, the VGAE adds nothing
over trivial partitioning and the "learned" part of the idea is not pulling weight.

n=100 -> m=50, families ER/BA/WS/SBM, R=10.
Writes results/e5_coarsening.json.
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
from gvs.models.vgae import encode_latents, train_vgae
from subsampling.coarsening import latent_coarsen, random_coarsen
from subsampling.decoders import latent_downsample
from subsampling.baselines import uniform_node

N, M_SUB, R = 100, 50, 10
RESULTS = Path(__file__).resolve().parent.parent / "results"

FAMILIES = {
    "er": lambda s: erdos_renyi(N, 0.1, seed=s),
    "ba": lambda s: barabasi_albert(N, 5, seed=s),
    "ws": lambda s: watts_strogatz(N, 10, 0.1, seed=s),
    "sbm": lambda s: stochastic_block_model([N // 2, N // 2], 0.18, 0.02, seed=s),
}
METHODS = ["latent_coarsen", "random_coarsen", "latent_decode", "uniform_node"]
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
        "lambda_max_norm": s["lambda_max"] / max(s["n_nodes"], 1),
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
            density = nx.density(g)
            res = train_vgae(data, epochs=300, seed=r)
            z = encode_latents(res.model, data)
            s = summary(g)
            originals.append({"clustering": s["clustering"], "modularity": modularity(g),
                              "lambda_max_norm": s["lambda_max"] / N})

            built = {
                "latent_coarsen": latent_coarsen(g, z, M_SUB, density, seed=r),
                "random_coarsen": random_coarsen(g, M_SUB, density, seed=r),
                "latent_decode": latent_downsample(z, M_SUB, density, method="random",
                                                   seed=r, decode="bernoulli"),
                "uniform_node": uniform_node(g, M_SUB, seed=r),
            }
            for name, gs in built.items():
                records.append({"method": name} | evaluate(g, gs))

        agg = {
            method: {
                k: {"mean": float(np.nanmean([x[k] for x in records if x["method"] == method])),
                    "std": float(np.nanstd([x[k] for x in records if x["method"] == method]))}
                for k in METRICS
            }
            for method in METHODS
        }
        orig = {k: float(np.nanmean([o[k] for o in originals]))
                for k in ["clustering", "modularity", "lambda_max_norm"]}
        out[family] = {"original": orig, "methods": agg}

        print(f"\n=== {family.upper()} (n={N} -> m={M_SUB}, R={R}) ===")
        print(f"{'method':<16}" + "".join(f"{k:>16}" for k in METRICS))
        print(f"{'(original)':<16}{orig['clustering']:>16.3f}{orig['modularity']:>16.3f}"
              f"{orig['lambda_max_norm']:>16.3f}{'—':>16}")
        for method in METHODS:
            a = agg[method]
            print(f"{method:<16}" + "".join(
                f"{a[k]['mean']:>10.3f}±{a[k]['std']:<5.3f}" for k in METRICS))

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "e5_coarsening.json"
    path.write_text(json.dumps({
        "experiment": "e5_coarsening",
        "config": {"n": N, "m": M_SUB, "replicates": R, "epochs": 300},
        "results": out,
    }, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
