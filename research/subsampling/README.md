# Direction 1 — VAE-based graph subsampling

*The original question of this repository.* Can a variational graph autoencoder
learn to **downsample a large graph** — decode a smaller graph from the latent
space that preserves the original's structural distributions (degree, clustering,
spectrum, communities)?

## Idea

Train a VGAE on a graph; its latent matrix `Z` (N × d) is a learned node embedding.
To downsample to m < N nodes, **select m latent vectors** (random subset, k-means
centroids, or draws from the aggregated posterior) and **decode** them into an
m × m graph, density-matched to the original. Compare against classical samplers.

Library: [`subsampling`](../../src/subsampling/) — `decoders.py` (latent selection +
top-k / Bernoulli decoding) and `baselines.py` (uniform node, random walk, forest
fire). The VGAE itself is in the common library (`gvs.models.vgae`).

## Experiments

| script | what it does |
|--------|--------------|
| `e0_vgae_er.py` | train a VGAE on ER(100, 0.1); link-prediction + reconstruction sanity |
| `e1_er_downsample.py` | ER → m ∈ {25,50,75}; latent variants vs classical baselines |
| `e2_families_downsample.py` | BA / WS / SBM → 50; family-specific property preservation |
| `e3_decoder_ablation.py` | decoder × decode-rule ablation (the over-clustering fix) |
| `e4_scale.py` | n = 1000 at 5–10% sample fractions |
| `e5_coarsening.py` | coarsening (aggregate real edges) vs decoding vs sampling |

Run: `uv run python research/subsampling/experiments/<script>.py` → writes `results/<name>.json`.

## What we found

- **The inner-product decoder over-clusters.** Latent subsampling reproduced density
  and degree shape but inflated clustering/triangles 2–5× across all graph families.
- **The culprit was deterministic top-k decoding, not the inner product** (E3).
  Density-calibrated **Bernoulli** decoding recovers true clustering on ER/SBM;
  degree-correction additionally fixes BA. Watts–Strogatz still under-clusters
  (its lattice transitivity is genuine edge dependence the independence-given-Z
  decoder can't reproduce).
- **Classical node sampling is hard to beat** at the scales tested — but the latent
  method wins on **variance** in the scale-free small-fraction regime (BA at 5% of
  n=1000: degree-W1 0.0072 ± 0.0001 vs uniform's 0.0093 ± 0.0024), where uniform
  sampling plays a "hub lottery."
- **Coarsening beats decoding for structure preservation** (E5). Aggregating the
  graph's *real* edges into latent-clustered supernodes preserves community and
  local structure that the decoder hallucinates away. The standout is Watts–Strogatz
  (the case decoding failed): latent coarsening is the only method keeping both
  clustering (0.44 vs orig 0.49) and modularity (0.54 vs 0.51) close, where decoding
  collapses clustering to 0.19 and uniform sampling overshoots modularity to 0.64.
  Ablation: the VGAE partition matters only where structure exists — latent_coarsen
  ≫ random_coarsen on WS/SBM modularity, but ties on ER/BA (no communities).
  Bounded: coarsening over-clusters on sparse graphs and trades away degree-
  distribution fidelity (uniform sampling stays best on degree-W1).

## Open threads (actively explored)

- [x] **Coarsening as an alternative to decoding** (E5): wins on structured graphs,
      fixes the WS case. The learned partition (vs random) matters where communities exist.
- [ ] **Learned/soft pooling (true M2):** E5 uses post-hoc k-means on latents. A
      DiffPool-style *soft* assignment trained end-to-end could fix coarsening's
      over-clustering and degree distortion by optimizing the partition for structural
      preservation rather than latent proximity.
- [ ] **Edge-dependent decoding** for WS-style lattice transitivity (the independence
      assumption given Z cannot produce it) — though E5's coarsening already addresses WS.
- [ ] **The right invariant at small sample fractions:** density matching collapses
      mean degree and fragments the graph — preserve density `p`, or mean degree /
      degree shape? (The image-VAE analogy suggests degree preservation.)
- [ ] **Hierarchical decoder (M2):** intermediate decoder layers as progressively
      coarser graphs, with structural-consistency losses.
- [ ] Real-graph downsampling (connectome / citation network).
