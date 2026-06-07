# Direction 2 — graph summaries for dependence testing

Test statistical dependence between **populations of graph pairs**
`{(G1_i, G2_i)}` — e.g. paired brain networks — extending the spectral-radius
correlation test of Fujita et al. (2017). The question that organizes the work:
*which summary statistic of a graph best exposes parameter-level dependence, and
why?*

Library: [`deptest`](../../src/deptest/) — `pairs.py` (correlated ER / SBM /
degree-corrected SBM pair generators), `embeddings.py` (λ_max, truncated spectrum,
NetLSD, structural features, VGAE singular values), `dependence.py` (distance
correlation, HSIC, whitening, learned-projection CCA, permutation tests).

## Experiments

| script | what it does |
|--------|--------------|
| `d1_dependence_power.py` | ER pairs: calibration + power curves, 4 statistics |
| `d2_sbm_power.py` | SBM pairs: the p_out blind spot (λ_max vs richer summaries) |
| `d2b_kill_or_confirm.py` | strong baselines (full spectrum, NetLSD, HSIC, whitening) |
| `d3*_*.py` | scaling to R=100, type-I diagnostic, definitive table, combiners |
| `d4_sensitivity.py` | blind spot across graph size n and population size k |
| `d5_realism.py` / `d5b_block_boundary.py` | DC-SBM, multi-block, block-count boundary |
| `d6_learned_projection.py` | learned (CCA) projection vs fixed whitening |
| `m1_disentanglement.py` | mechanism check: how summaries load on (p_in, p_out) |

Run: `uv run python research/dependence_testing/experiments/<script>.py` → `results/<name>.json`.

## Headline findings

- **A characterized blind spot.** On few-community SBM populations whose dependence
  is confined to the between-block density `p_out`, the benchmark and every *raw*
  spectral statistic (full spectrum, NetLSD, HSIC) have power ≤ 0.19 at R=100.
- **The failure is geometric, not informational.** The spectrum *contains* the
  signal ((λ1−λ2)/2 tracks p_out at r = 0.89), but distance-based tests are
  dominated by independent p_in nuisance variation.
- **A cheap fix:** whitening the truncated spectrum (= affinely-invariant dCor)
  restores power to 0.97; the VGAE singular values self-orient to 0.89. No free
  lunch — whitening costs power when dependence is spectrum-aligned.
- **Bounded scope.** The blind spot is sharp at 2 communities, survives degree
  heterogeneity, and erodes smoothly as communities multiply (the radius weights
  p_out by (b−1)× p_in).
- **Negative result:** a *learned* projection (CCA + cross-fitting) does **not**
  beat fixed whitening at the small populations intrinsic to the problem (k≈40
  pairs) — the right geometry should be engineered, not learned.

## Paper

Working draft in [`paper/main.tex`](paper/main.tex) (`pdflatex main` to build).

## Open threads

- [ ] **Real data:** paired brain networks (fMRI) — required for a strong venue.
- [ ] Larger graphs (n ≥ 500); learned-summary behavior across the full (n, k) grid.
- [ ] Node features as additional embedding inputs (does it add test power?).
- [ ] Novelty/positioning audit vs affinely-invariant dCor and graph-independence testing.
