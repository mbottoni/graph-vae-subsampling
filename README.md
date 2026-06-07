# Graph VAE Subsampling

A research repository with **two distinct research directions** that grew from one
starting question — *can a variational graph autoencoder learn to subsample large
graphs?* — and a **common library** of code they share.

## Repository layout

```
src/
  gvs/          COMMON library — graph generators, the VGAE, structural metrics
  subsampling/  Thread-1 library — latent-subsampling decoders, classical samplers
  deptest/      Thread-2 library — graph-pair generators, dependence tests, embeddings

research/
  subsampling/            ── Direction 1: VAE-based graph downsampling (the original idea)
    README.md  experiments/  results/
  dependence_testing/     ── Direction 2: graph summaries for dependence testing
    README.md  experiments/  results/  paper/
```

The common library holds everything both directions use (random-graph generators,
the VGAE encoder/decoder and training loop, degree/clustering/spectral metrics).
Each direction's own code lives in its own package, and its experiments, results,
and writeups live under `research/<direction>/`.

## The two directions

### 1. [Subsampling](research/subsampling/README.md) — the original idea
Use a (V)GAE to **downsample a large graph**: learn node embeddings, then decode a
*smaller* graph that preserves the original's structural distributions. Compared
against classical samplers (uniform node, random walk, forest fire). **Status:**
the over-clustering bias of the inner-product decoder was diagnosed and fixed
(density-calibrated Bernoulli decoding); classical samplers remain hard to beat
except in specific small-fraction regimes. Open threads: edge-dependent decoding
for lattice transitivity, and the right invariant to preserve at small sample
fractions. *Actively being explored.*

### 2. [Dependence testing](research/dependence_testing/README.md)
Test statistical dependence between **populations of graph pairs** (extending
Fujita et al.'s spectral-radius correlation test). **Status:** a characterized
blind spot of the benchmark (few-community SBMs with between-block dependence), a
geometric explanation (information ≠ test power), a cheap fix (whitened spectra),
and a negative result (learned projections don't beat fixed ones at small
populations). Draft paper in `research/dependence_testing/paper/`.

## Setup

```bash
uv sync
uv run python research/subsampling/experiments/e0_vgae_er.py        # Direction 1
uv run python research/dependence_testing/experiments/m1_disentanglement.py  # Direction 2
```

Each experiment writes a JSON summary next to itself under the direction's
`results/`. See [`PLAN.md`](PLAN.md) for status and next steps across both directions.
