# Graph VAE Subsampling

Research project exploring **variational graph autoencoders (VGAEs) as a tool for downsampling large graphs** and, more broadly, **graph embeddings as summary statistics for testing dependence between graphs**.

## Motivation

Graphs are rich representations used successfully across many domains. Two related problems motivate this work:

1. **Downsampling large graphs.** Many graph algorithms do not scale. If a generative model (VGAE/GAN) can learn the structure of a large graph in a low-dimensional latent space, we may be able to *decode a smaller graph* that preserves the structural and distributional properties of the original — i.e., learn how to sample a representative subgraph.

2. **Measuring dependence between graphs.** In domains such as neuroscience (e.g., dependence between brain regions in fMRI networks), one needs a statistical test of dependence among sets of graphs. The main challenge is translating graph components (e.g., the adjacency matrix) into vectors of values. The state-of-the-art approach (Fujita et al., 2016) uses the **Pearson correlation of adjacency-matrix eigenvalues**, which summarize properties such as cliques and walk counts. Graph embeddings offer a richer, more flexible alternative mapping from the adjacency matrix (and optionally node/edge features) into a latent space — but they have not yet been explored for dependence testing.

## Research Threads

### 1. Graph downsampling with VGAEs

Use a stochastic embedding to sample from a large graph. Modify the decoder architecture so that input and output graphs are structurally similar, but the **output is smaller**.

Approach:
- Train a standard VGAE and use **intermediate layers of the decoder** as the downsampling mechanism.
- Alternative: use a graph GAN (or GVAE) to learn how to sample from a large graph, treating the latent space as a low-dimensional space in which to reconstruct the graph.

Validation on synthetic graphs:
1. Build a random Erdős–Rényi graph (n = 100).
2. Sample a smaller graph from it via the model.
3. Check whether structural distributions (e.g., degree distribution) are preserved.

### 2. Dependence testing with graph embeddings

Extend the eigenvalue-correlation test of Fujita et al. by replacing eigenvalues with graph embeddings as the summary statistic:

- **Phase 1:** compute embeddings from the adjacency matrix only, and test dependence among embeddings instead of eigenvalues. Compare against the eigenvalue benchmark.
- **Phase 2:** add node features as inputs to the embedding and test whether this adds information to the hypothesis test relative to the benchmark.

Evaluation plan:
- Bootstrap (~100 repetitions) on random graphs with known ground truth.
- Compare embedding-based statistics against the max-eigenvalue (λ_max) baseline.
- Include **uncorrelated graph pairs** to verify false-positive control.
- Analyze the correlation between the embedding space and the eigenvalue spectrum.

### 3. Side question: parameter estimation for random graphs

Estimate random-graph model parameters with **variational inference**, comparing against MCMC and MLE.

## Roadmap

- [x] Understand the GraphVAE / VGAE model (M0 implemented in `src/gvs/models/vgae.py`)
- [ ] Review image downsampling with VAEs (as an analogy/starting point)
- [x] Train a standard VGAE and use latent subsampling for downsampling (E0/E1, `results/`)
- [x] Erdős–Rényi sanity-check experiment (n = 100): sample and compare distributions (E1)
- [x] Structured families BA/WS/SBM downsampling comparison (E2)
- [x] Fix the over-clustering bias (E3): culprit was deterministic top-k decoding, not the
      inner product. Density-calibrated Bernoulli decoding recovers true clustering on ER/SBM;
      degree-corrected + Bernoulli is best on BA. WS (genuine transitivity) still under-clusters.
- [ ] WS gap: edge-dependent decoding (the independence assumption given Z can't produce
      lattice-like transitivity)
- [x] E4 scale-up (n=1000, 5-10% samples): latent methods match uniform node everywhere and
      beat it on BA degree shape at 5% (W1 0.0072±0.0001 vs 0.0093±0.0024 — 20x lower variance:
      uniform sampling's hub lottery vs the latent posterior's smooth degree summary)
- [ ] Open question from E4: density matching is the wrong invariant at small sample fractions
      (mean degree collapses, graphs fragment, modularity becomes meaningless). Decide what
      downsampling should preserve: density p vs mean degree / degree shape
- [ ] Embeddings vs. eigenvalues: correlation analysis between embedding space and eigenvalue spectrum
- [x] D1-D3 on ER pairs (k=40, n=60, R=20): all four statistics have exact type-I error (0.05)
      and saturate power by rho=0.75. Hints (within noise at R=20): vgae_sv best at rho=0.25
      (0.35 vs benchmark 0.25), spectral5/features best at rho=0.5 (0.95 vs 0.90)
- [x] D2 on SBM pairs — headline Thread-2 result: with dependence only in p_out (rho=1),
      the learned embedding detects what every spectral statistic misses
      (vgae_sv power 0.96 vs lambda_max 0.24, spectral5 0.24, features 0.16).
      Surprise: spectral5 fails too, despite (lambda_1 - lambda_2)/2 isolating p_out exactly —
      information content is not enough; dCor's distance geometry is dominated by independent
      p_in noise. The VGAE latent SVs *disentangle* (p_in, p_out) into separate components
      (centroid separation tracks p_out; within-cluster tightness tracks p_in)
- [x] Kill-or-confirm (D2b): whitening RESCUES the fixed spectrum — spec5_white 0.96 = vgae_sv
      0.96 in the p_out condition, while full spectrum (0.36), NetLSD (0.32), HSIC (0.28) stay
      weak. Mechanism confirmed (the failure is geometric, a linear fix suffices); method claim
      demoted (learned embeddings are *one* way to disentangle, whitening is the cheaper way).
- [x] Mechanism exhibited directly (m1): all adjacency eigenvalues p_in-dominated (0.63-0.92),
      (lambda1-lambda2)/2 isolates p_out (corr 0.894), vgae sv_1 IS the p_out coordinate (-0.61).
      Formalized as the two-centroid Proposition in the paper (sv_1 = sqrt(n/2 (logit gap)))
- [x] D3c: vgae_sv's apparent type-I inflation (0.10) was a HARNESS BUG (seed coupling), not
      the statistic — decoupled SeedSequence streams give exactly 0.050. Cautionary methods note
- [x] D3-scaled (R=100, CIs): at p_out rho=1, spec5_white 0.96 [0.90,0.98] now STRICTLY beats
      vgae_sv 0.82 [0.73,0.88] (disjoint CIs); benchmark collapses to 0.13-0.14. Whitened
      spectrum is the practical winner; learned summary's value is purely mechanistic (Prop 1)
- [~] D3d definitive (R=100, decoupled seeds) IN FLIGHT — supersedes D3-scaled for paper tables;
      includes both combiners (concat vs Bonferroni min-p)
- [x] Positioned against affinely-invariant dCor (Dueck et al. 2014) and Xiong et al. 2019
      (vertex-aligned single-pair) in the paper's related work
- [ ] Real data: paired brain networks (fMRI) — required for a non-workshop venue
- [ ] Sensitivity sweep: n in {60,100,200}, k in {20,40,80}
- [ ] VI vs. MCMC vs. MLE for random-graph parameter estimation (Thread 3, untouched)

## Literature

- *A Comprehensive Survey of Graph Embedding: Problems, Techniques and Applications*
- Fujita et al. — *Correlation between graphs with an application to brain network analysis* (2016)
- Kipf & Welling — *Variational Graph Auto-Encoders*
- *BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis*
- *Understanding Graph Embedding Methods and their Applications*
- *Learning Deep Representations for Graph Clustering*
