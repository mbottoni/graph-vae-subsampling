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

- [ ] Understand the GraphVAE / VGAE model
- [ ] Review image downsampling with VAEs (as an analogy/starting point)
- [ ] Train a standard VGAE and use intermediate decoder layers for downsampling
- [ ] Erdős–Rényi sanity-check experiment (n = 100): sample and compare distributions
- [ ] Embeddings vs. eigenvalues: correlation analysis between embedding space and eigenvalue spectrum
- [ ] Bootstrap evaluation of the embedding-based dependence test (incl. uncorrelated graphs)
- [ ] VI vs. MCMC vs. MLE for random-graph parameter estimation

## Literature

- *A Comprehensive Survey of Graph Embedding: Problems, Techniques and Applications*
- Fujita et al. — *Correlation between graphs with an application to brain network analysis* (2016)
- Kipf & Welling — *Variational Graph Auto-Encoders*
- *BrainGNN: Interpretable Brain Graph Neural Network for fMRI Analysis*
- *Understanding Graph Embedding Methods and their Applications*
- *Learning Deep Representations for Graph Clustering*
