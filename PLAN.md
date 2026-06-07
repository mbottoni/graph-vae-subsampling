# Project Plan

## Repository structure (refactored June 2026)

Two research directions over a shared library:
- `src/gvs/` — common library (generators, VGAE, metrics)
- `src/subsampling/` + `research/subsampling/` — Direction 1 (VAE downsampling; E0–E4)
- `src/deptest/` + `research/dependence_testing/` — Direction 2 (dependence testing; D1–D6, M1)

Each direction has its own README and roadmap; this file is the cross-cutting plan.
The "Thread 1 / Thread 2" naming below predates the refactor and maps to
subsampling / dependence_testing respectively.

## Status & next steps (June 2026)

**Done:** E0–E4 (Thread 1) and D1–D3 + D2-SBM (Thread 2). See `results/*.json` and the
README roadmap. Headline so far: on 2-block SBM pairs with dependence only in p_out,
the VGAE-latent-SV statistic reaches power 0.96 where lambda_max (Fujita), top-5
eigenvalues, and structural features reach 0.16–0.24 — with all tests exactly calibrated.
Mechanism: latent singular values *disentangle* (p_in, p_out) into separate coordinates,
while adjacency eigenvalues entangle them as sums/differences that dCor cannot separate
under independent nuisance variation.

**Publication target:** Thread 2 paper — "graph embeddings as summary statistics for
testing dependence between graph populations." Thread 1 is supporting material (it
produced the Bernoulli-decoding insight and the trained encoders) — not currently a
standalone paper (classical baselines win at the scales tested; one modest win: BA@5%
degree-shape variance).

### Kill-or-confirm outcome (D2b, June 2026) — paper reframed

The pre-registered prediction held exactly: **whitening rescues the fixed spectrum**
(spec5_white 0.96 = vgae_sv 0.96 in the p_out condition) while more information does not
(full spectrum 0.36, NetLSD 0.32, HSIC 0.28). And m1 exhibits the mechanism directly
(all eigenvalues p_in-dominated; (l1-l2)/2 isolates p_out at corr 0.894; vgae sv_1 is
the p_out coordinate at -0.61).

Revised paper story (more modest, more precise, better supported):
*Test power for graph-population dependence is decided by the geometry of the summary
statistic, not its information content. The Fujita benchmark and every raw spectral
statistic fail when dependence is orthogonal to dominant nuisance variation; a whitened
truncated spectrum (= affinely-invariant dCor, cf. Dueck et al. 2014) fixes it cheaply,
and learned VGAE summaries arrive disentangled without explicit whitening.* The learned
embedding is demoted from "the method" to "one instance of the principle"; spec5_white
is the practical recommendation (best in both conditions combined).

### Type-I inflation resolved (D3c)

vgae_sv's apparent anti-conservatism (0.08-0.10 across runs) was a **harness bug,
not a statistic bug**: one base seed drove the data RNG, the VGAE training seeds
(colliding with neighboring replicates' data seeds), and the permutation RNG.
With SeedSequence-spawned independent streams: vgae_sv type-I = 0.050 exactly
(lambda_max 0.03, spec5_white 0.03, Bonferroni-min combo 0.02 — conservative as
expected). Action: all headline tables must come from the decoupled harness;
d3_scaled (coupled seeds) is demoted to preliminary and rerun. Worth a methods
paragraph — seed-coupled harnesses can manufacture mild miscalibration that looks
exactly like a broken statistic.

### Definitive headline (D3d, R=100, decoupled seeds)

The paper's Table 2 is final. p_out blind spot at rho=1: spec5_white 0.97, concat 0.99,
bonferroni_min 0.98, vgae_sv 0.89; benchmark 0.07, raw spectrum 0.19. All type-I calibrated.
Settled the combiner question: **no uniformly best statistic** — whitening trades aligned-regime
power (0.65 vs raw 0.90) for blind-spot power; Bonferroni min-p is the most robust single choice
(best-of-all at the hardest cell p_out/rho=0.5 = 0.33, near-top in the blind spot = 0.98).

### Next steps, in priority order

**Scope correction (D5/D5b, June 2026).** The realism check found the blind spot is NOT
universal: it is sharp at 2 communities, survives degree heterogeneity, but erodes smoothly
as communities multiply (benchmark power 0.07/0.70/0.98 for b=2/3/4), governed by the radius
weighting p_out at (b-1)x p_in. This narrowed the headline claim but reviewer-proofs it — the
obvious 4-block counterexample is now characterized, not missed. The paper is scoped to
few-community, p_out-dominated populations (still covers coarse-parcellation / bipartition
analyses in neuroscience). Lesson: do the realism check BEFORE believing a clean-toy result.

1. ~~Kill-or-confirm~~ ~~R=100 + combiner~~ ~~realism/boundary~~ **DONE.** Synthetic core complete
   and written up (6 tables, scoped claim, smooth-boundary mechanism). Remaining blockers:
2. **Scale the headline result.** R=100 replicates (tight CIs on the 0.96-vs-0.24 claim),
   n in {60, 100, 200}, k in {20, 40, 80}, full rho grid, 3-block SBM, degree-corrected
   SBM. Verify vgae_sv type-I (was 0.08 at R=25 — must settle to ~0.05).
3. **Combined statistic.** spectral5 (+) vgae_sv concatenated (standardized per block).
   The two fail in complementary regimes ("both": spectral 0.88 vs vgae 0.32 at rho=0.5;
   "p_out": vgae 0.96 vs spectral 0.24). If the combination tracks the upper envelope,
   the paper has a practical method, not just a diagnosis.
4. **Positioning / related-work pass.** Differentiate explicitly from: Fujita et al.
   (spectral-radius correlation — our benchmark), the JHU graph-independence-testing
   line (vertex-aligned single-pair setting, dCorr on adjacency), graph two-sample
   testing, and graph-kernel MMD literature. Our setting: population of graph *pairs*,
   dependence at the generative-parameter level, no vertex alignment across pairs.
5. **One real dataset.** Paired brain networks (fMRI, e.g. two parcellations/sessions
   per subject) — the Fujita lineage's home domain. Needed for any non-workshop venue.
6. **Theory section.** For the 2-block SBM, prove in expectation that the VGAE latent
   geometry separates p_in (within-cluster spread) from p_out (centroid separation),
   and that distances on raw spectra are dominated by the nuisance direction. Even a
   lemma lifts the mechanism from observation to contribution.
7. **D4 (node features).** Add informative node features to the embedding inputs and
   test added power — phase 2 of the original abstract.
8. **Thread 1 follow-ups (lower priority):** mean-degree-matched E4 rerun (the invariant
   question), WS edge-dependent decoding, real-graph E4.

**Venues:** workshop first (NeurIPS/ICML graph or stats-ML workshop) as forcing function;
then AISTATS (with theory) or Network Science / Comput. Stat. & Data Analysis /
Network Neuroscience (with the brain application leading).

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
