# Project Plan

Two main research threads sharing one codebase (the VGAE and graph-statistics code is common to both), plus a side quest.

## Repository layout

```
graph-vae-subsampling/
├── pyproject.toml              # uv-managed; torch, torch-geometric, networkx, scipy
├── src/gvs/
│   ├── data/synthetic.py       # ER, BA, WS, SBM generators; correlated graph pairs
│   ├── models/vgae.py          # standard VGAE (2-layer GCN encoder, inner-product decoder)
│   ├── models/decoders.py      # downsampling decoder variants
│   ├── sampling/baselines.py   # random node, random walk, forest fire sampling
│   ├── metrics/graph_stats.py  # degree dist, clustering, spectrum, motifs
│   ├── metrics/distances.py    # KS / Wasserstein between graph-stat distributions
│   └── stats/
│       ├── fujita.py           # eigenvalue-correlation test (baseline)
│       └── embedding_test.py   # embedding-based dependence test + bootstrap
├── experiments/                # one script per experiment, writes to results/
└── results/
```

## Models

- **M0 — Standard VGAE (baseline).** Kipf & Welling: GCN encoder → node-level latents
  `z_i ~ N(mu_i, sigma_i)` → inner-product decoder `A_hat = sigmoid(Z Z^T)`.
  Validated via link-prediction AUC/AP. Everything else builds on this.
- **M1 — Latent-subsampling decoder** (core idea). After training M0, the latent space holds N
  node embeddings. To downsample to m < N nodes: sample/select m latent vectors (random subset,
  k-means centroids, or sampling from the aggregated posterior) and decode them with the same
  inner-product decoder → an m×m graph. Directly tests the hypothesis "the latent space is a
  sampleable summary of the graph".
- **M2 — Hierarchical decoder.** Image-VAE analogy: decoder with intermediate layers representing
  progressively coarser graphs (n/4 → n/2 → n nodes), so an intermediate layer *is* the
  downsampled graph. Reconstruction loss at the final layer + structural-consistency losses at
  intermediate layers. After M1.
- **M3 (optional) — GAN variant** (NetGAN-style) if VAE blurriness becomes a problem.
- **Classical baselines:** uniform node sampling, random-walk sampling, forest fire
  (Leskovec & Faloutsos, *Sampling from Large Graphs*).

## Experiments

### Thread 1 — downsampling

| Exp | Setup | Question |
|-----|-------|----------|
| E1 | ER(n=100, p=0.1); downsample to m ∈ {25, 50, 75} via M1 vs classical baselines | Is p_hat and the degree distribution preserved? (KS, Wasserstein) |
| E2 | Repeat for BA, WS, SBM | Are family-specific properties preserved (power law, clustering, communities)? |
| E3 | Scale to n = 1k–10k | Does it hold beyond toy size? |
| E4 | One real graph (connectome or citation net) | Real-world sanity check |

Metrics throughout: degree distribution distance, clustering coefficient, eigenvalue spectrum
(incl. lambda_max), assortativity, triangle/motif counts — original vs downsampled.

### Thread 2 — dependence testing

| Exp | Setup | Question |
|-----|-------|----------|
| D1 | Reimplement Fujita eigenvalue test; correlated ER pairs with known rho (shared Bernoulli noise) | Baseline reproduces published behavior? |
| D2 | Same pairs, statistic = graph embeddings (VGAE latents aggregated, node2vec, graph2vec); bootstrap ~100× | Power curve vs rho: embeddings ≥ eigenvalues? |
| D3 | Uncorrelated pairs | Type-I error control |
| D4 | Add node features to embeddings | Does feature information increase test power? |
| D5 | Correlation between embedding space and eigenvalue spectrum | Why/when do embeddings carry spectral info? |

### Thread 3 — side quest

ER/SBM parameter estimation via VI (numpyro) vs MCMC vs MLE — bias/variance/runtime.
Independent, can slot in anytime.

## Sequencing

1. **Phase 0+1:** scaffold the package, implement synthetic generators + graph-stat metrics,
   train standard VGAE on ER(n=100), verify reconstruction (E0). Everything depends on this.
2. **Phase 2:** M1 latent-subsampling + classical baselines → run E1/E2. First real result.
3. **Phase 3:** Fujita baseline + D1–D3 (reuses generators/metrics from Phase 0).
4. **Phase 4:** M2 hierarchical decoder, E3/E4, D4/D5 — guided by what Phases 2–3 show.

The threads connect: if E1 shows VGAE latents preserve structural distributions, that directly
motivates using them as the summary statistic in D2.
