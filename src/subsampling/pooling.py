"""Learned graph coarsening via DMoN pooling — the end-to-end 'M2' pooling.

E5 coarsened with post-hoc k-means on frozen VGAE latents. Here the partition is
LEARNED: a pooling head produces a soft assignment S (N x m), trained with the
DMoN objective (Tsitsulin et al. 2020) — a spectral-modularity term that favours
partitions respecting graph connectivity, plus orthogonality and an explicit
collapse-regularization term. (We use DMoN rather than plain MinCutPool because
the latter collapses to a single symmetric critical point with zero gradient on
these graphs; DMoN's collapse term is designed precisely to avoid that.)

`learned_coarsen` takes the SAME VGAE latents that `latent_coarsen` clusters with
k-means, so the two differ only in how the partition is chosen — k-means by
latent proximity vs. DMoN by structural modularity. The hardened assignment is
then aggregated over the original real edges (subsampling.coarsening machinery).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from torch_geometric.nn import DMoNPooling
from torch_geometric.utils import to_dense_adj

from subsampling.coarsening import _coarsen_by_labels


@dataclass
class CoarsenResult:
    graph: nx.Graph
    n_effective: int   # non-empty supernodes actually used
    losses: list[float]


def learned_coarsen(
    g: nx.Graph,
    z: torch.Tensor,
    m: int,
    density: float,
    hidden: int = 32,
    epochs: int = 300,
    lr: float = 0.005,
    seed: int = 0,
) -> CoarsenResult:
    """Learn a structure-aware partition of g (DMoN on the VGAE latents z), harden
    it, and aggregate the original real edges into the coarse graph.

    Empty supernodes are dropped and the partition relabelled, so the coarse graph
    has `n_effective <= m` nodes; size-normalized metrics keep the comparison with
    k-means coarsening fair.
    """
    torch.manual_seed(seed)
    n = g.number_of_nodes()
    feat = (z - z.mean(0)) / (z.std(0) + 1e-6)
    adj = to_dense_adj(
        torch.tensor(np.array(nx.to_numpy_array(g)).nonzero()), max_num_nodes=n
    )
    x = feat.unsqueeze(0)

    pool = DMoNPooling([feat.shape[1], hidden], m)
    opt = torch.optim.Adam(pool.parameters(), lr=lr)

    losses = []
    pool.train()
    for _ in range(epochs):
        opt.zero_grad()
        _, _, _, sp, o, c = pool(x, adj)
        loss = sp + o + c
        loss.backward()
        opt.step()
        losses.append(loss.item())

    pool.eval()
    with torch.no_grad():
        s = pool(x, adj)[0].squeeze(0)
        labels = s.argmax(dim=1).numpy()

    used = np.unique(labels)
    remap = {old: new for new, old in enumerate(used)}
    labels = np.array([remap[c] for c in labels])
    coarse = _coarsen_by_labels(g, labels, len(used), density)
    return CoarsenResult(graph=coarse, n_effective=len(used), losses=losses)
