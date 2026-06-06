"""E0 — Phase 0+1 sanity check.

Train the standard VGAE (M0) on an ER(n=100, p=0.1) graph and verify it learns
the structure: link-prediction AUC/AP on held-out edges, plus a full-graph
reconstruction comparison (threshold A_hat at the original edge density and
compare structural summaries).

Writes results/e0_vgae_er.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch

from gvs.data.synthetic import erdos_renyi, to_pyg
from gvs.metrics.distances import ks_distance, normalized_degree_distance
from gvs.metrics.graph_stats import degree_sequence, summary
from gvs.models.vgae import encode_latents, train_vgae

SEED = 0
N, P = 100, 0.1
RESULTS = Path(__file__).resolve().parent.parent / "results"


def reconstruct_graph(z: torch.Tensor, n_edges: int) -> nx.Graph:
    """Decode A_hat = sigmoid(ZZ^T) and keep the top-n_edges edges (density matching)."""
    with torch.no_grad():
        a_hat = torch.sigmoid(z @ z.t())
    n = a_hat.shape[0]
    iu = np.triu_indices(n, k=1)
    scores = a_hat.numpy()[iu]
    keep = np.argsort(scores)[-n_edges:]
    g = nx.empty_graph(n)
    g.add_edges_from(zip(iu[0][keep], iu[1][keep]))
    return g


def main() -> None:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    g = erdos_renyi(N, P, seed=SEED)
    data = to_pyg(g)
    print(f"ER(n={N}, p={P}): {g.number_of_edges()} edges")

    result = train_vgae(data, epochs=300, seed=SEED, verbose=True)
    print(f"\nlink prediction  AUC={result.auc:.4f}  AP={result.ap:.4f}")

    # Full-graph reconstruction from posterior-mean latents.
    z = encode_latents(result.model, data)
    g_rec = reconstruct_graph(z, g.number_of_edges())

    orig, rec = summary(g), summary(g_rec)
    deg_o, deg_r = degree_sequence(g), degree_sequence(g_rec)
    ks_stat, ks_p = ks_distance(deg_o, deg_r)

    print("\n{:<15} {:>12} {:>12}".format("metric", "original", "reconstructed"))
    for k in orig:
        print(f"{k:<15} {orig[k]:>12.4f} {rec[k]:>12.4f}")
    print(f"\ndegree KS={ks_stat:.4f} (p={ks_p:.3f})  "
          f"W1={normalized_degree_distance(deg_o, deg_r):.4f}")

    RESULTS.mkdir(exist_ok=True)
    out = {
        "experiment": "e0_vgae_er",
        "config": {"n": N, "p": P, "seed": SEED, "epochs": 300},
        "link_prediction": {"auc": result.auc, "ap": result.ap},
        "final_loss": result.losses[-1],
        "summary_original": orig,
        "summary_reconstructed": rec,
        "degree_ks": {"statistic": ks_stat, "pvalue": ks_p},
        "degree_wasserstein_norm": normalized_degree_distance(deg_o, deg_r),
    }
    path = RESULTS / "e0_vgae_er.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
