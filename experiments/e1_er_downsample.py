"""E1 — downsample ER(n=100, p=0.1) to m in {25, 50, 75}.

Compare M1 latent-subsampling variants (random / kmeans / posterior) against
classical baselines (uniform node / random walk / forest fire), averaged over
R replicates. Each replicate draws a fresh ER graph and trains a fresh VGAE.

Question: is p_hat and the degree distribution preserved? Where do the latent
methods distort structure (clustering/triangles — the E0 over-clustering bias)?

Writes results/e1_er_downsample.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gvs.data.synthetic import erdos_renyi, to_pyg
from gvs.metrics.distances import normalized_degree_distance
from gvs.metrics.graph_stats import degree_sequence, summary
from gvs.models.decoders import latent_downsample
from gvs.models.vgae import encode_latents, train_vgae
from gvs.sampling.baselines import forest_fire, random_walk, uniform_node

N, P = 100, 0.1
MS = [25, 50, 75]
R = 10
RESULTS = Path(__file__).resolve().parent.parent / "results"

LATENT_METHODS = ["random", "kmeans", "posterior"]
BASELINES = {
    "uniform_node": uniform_node,
    "random_walk": random_walk,
    "forest_fire": forest_fire,
}
TRACKED = ["density", "clustering", "assortativity", "lambda_max"]


def evaluate(g_orig, g_small) -> dict[str, float]:
    s = summary(g_small)
    row = {k: s[k] for k in TRACKED}
    # lambda_max scales ~ m*p for ER; normalize by m to make sizes comparable.
    row["lambda_max_norm"] = s["lambda_max"] / s["n_nodes"]
    row["triangle_density"] = s["n_triangles"] / max(
        1, s["n_nodes"] * (s["n_nodes"] - 1) * (s["n_nodes"] - 2) / 6
    )
    row["degree_w1"] = normalized_degree_distance(
        degree_sequence(g_orig), degree_sequence(g_small)
    )
    return row


def main() -> None:
    records: list[dict] = []
    originals: list[dict] = []

    for r in range(R):
        torch.manual_seed(r)
        np.random.seed(r)
        g = erdos_renyi(N, P, seed=r)
        data = to_pyg(g)
        density = 2 * g.number_of_edges() / (N * (N - 1))
        result = train_vgae(data, epochs=300, seed=r)
        z = encode_latents(result.model, data)
        originals.append(summary(g) | {"auc": result.auc})

        for m in MS:
            for method in LATENT_METHODS:
                g_small = latent_downsample(z, m, density, method=method, seed=r)
                records.append(
                    {"method": f"latent_{method}", "m": m, "rep": r}
                    | evaluate(g, g_small)
                )
            for name, fn in BASELINES.items():
                g_small = fn(g, m, seed=r)
                records.append({"method": name, "m": m, "rep": r} | evaluate(g, g_small))
        print(f"replicate {r + 1}/{R} done (AUC={result.auc:.3f})")

    # Aggregate: mean +/- std per method x m.
    methods = [f"latent_{m}" for m in LATENT_METHODS] + list(BASELINES)
    metrics = ["density", "clustering", "triangle_density", "lambda_max_norm", "degree_w1"]
    agg: dict[str, dict] = {}
    for method in methods:
        for m in MS:
            rows = [x for x in records if x["method"] == method and x["m"] == m]
            agg[f"{method}|m={m}"] = {
                k: {
                    "mean": float(np.nanmean([x[k] for x in rows])),
                    "std": float(np.nanstd([x[k] for x in rows])),
                }
                for k in metrics
            }

    orig_mean = {
        k: float(np.nanmean([o[k] for o in originals]))
        for k in ["density", "clustering", "lambda_max"]
    }
    orig_mean["triangle_density"] = float(
        np.nanmean([o["n_triangles"] / (N * (N - 1) * (N - 2) / 6) for o in originals])
    )
    orig_mean["lambda_max_norm"] = orig_mean.pop("lambda_max") / N

    # Report.
    print(f"\noriginal ER(n={N}, p={P}) means: "
          + "  ".join(f"{k}={v:.4f}" for k, v in orig_mean.items()))
    for m in MS:
        print(f"\n--- m = {m} ---")
        header = f"{'method':<18}" + "".join(f"{k:>18}" for k in metrics)
        print(header)
        for method in methods:
            a = agg[f"{method}|m={m}"]
            print(f"{method:<18}" + "".join(
                f"{a[k]['mean']:>11.4f}±{a[k]['std']:<6.4f}" for k in metrics
            ))

    RESULTS.mkdir(exist_ok=True)
    out = {
        "experiment": "e1_er_downsample",
        "config": {"n": N, "p": P, "ms": MS, "replicates": R, "epochs": 300},
        "original_means": orig_mean,
        "aggregated": agg,
    }
    path = RESULTS / "e1_er_downsample.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
