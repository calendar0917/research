# FSAR-R2-AR0 node assignment residual — what it learned (mechanism note, eval-only)

Branch `exp/fsar-r2-ar0-edge-binding-zinc`, created from the durable AR0 verdict
tip `69b8fe9` (branch `exp/fsar-r2-ar0-zinc`).  Mechanism revision `6acfc45`
(module `experiments/luyin16/fsar_r2_ar0_mechanism.py`, tests
`tests/test_fsar_r2_ar0_mechanism.py`).  **Official ZINC test is never loaded.**

This note does **not** re-litigate the AR0 verdict.  It explains what the
already-trained node assignment residual actually learned, corrects one wrong
descriptive claim from the AR0 note, and adds one new diagnostic experiment.
No model, protocol or pre-registered gate is changed.

Artifacts: `results/fsar_r2_ar0/mechanism/` (`h1_seed_agreement.json`,
`h2_rank.json`, `h3_pc_explanation.json`, `h4_contributions.json`,
`h5_frozen_m0.json`, `h6_witnesses.json`, `mechanism_summary.json`).

---

## 0. Correctness triage (phase-1 stop conditions)

| condition | finding |
|---|---|
| `C` / scaler computation wrong | **no.** `C = J − P` sum-centred; permutation-zero-mean and the scalar `⟨W, C⟩=0` tests still pass. |
| permutation zero-mean false | **no.** exact toy enumeration still passes. |
| cache / model mapping wrong | **no.** `Σ W_B ⊙ C̃` reproduces the recorded soup branch to `max abs err ≈ 1.5e-5` (float32 vs float64). |
| branch output from wrong checkpoint | **no.** `MB` soup seed `s` maps to `r2ar0_mb_seed{s}_top5_soup.pt`; branch std `0.65–0.68`, matching `diagnostics_*`. |
| key mechanism explanation invalid | **partially corrected, verdict intact.** The AR0 note's "`C` is close to rank-1 (effective rank 1.40, top singular fraction 0.89)" was an **audit slicing bug**, see §1. It was a caveat, not the basis of the verdict; the verdict rests on paired `M0`/`MM` MAE deltas and the assignment-shuffle diagnostic, both of which are unaffected. |

The `audit()` stage is fixed on this branch (full 1820-D matrix primary,
legacy 64-slice kept for provenance), and the old AR0 note now carries a
correction pointer.  No stop condition blocks the edge round.

---

## 1. Correction: the node `C` tensor is **not** near rank-1

The old audit computed `_effective_rank(c_matrix_all[:, :64])` where
`c_matrix_all` is the *flattened* `[N, 65·28 = 1820]` matrix.  `[:, :64]` is
only the first 64 of 1820 coordinates, not the full statistic.  Reproduced
exactly: `effective rank 1.346`, `top singular fraction 0.912` — i.e. the
published `1.40 / 0.89` was a slice artifact.

Corrected, train-fit cross-molecule spectra of the **full** flattened matrix:

| version | effective rank | top singular fraction | PC1 var | PC1–5 cum. var |
|---|---:|---:|---:|---:|
| raw `C` | 56.6 | 0.112 | 0.352 | 0.711 |
| `C/n` | 60.6 | 0.105 | 0.341 | 0.690 |
| `C/√n` | 58.7 | 0.108 | 0.345 | 0.700 |
| `C/m` | 60.5 | 0.106 | 0.343 | 0.692 |
| `C/√m` | 58.7 | 0.108 | 0.346 | 0.701 |
| model input `C̃` | 116.3 | 0.027 | 0.060 | 0.250 |

The spectrum is **flat**, not rank-1.  RMS scaling (which is the model input)
makes it even flatter.  So `C` is not a hidden one-dimensional assignment
axis; if anything the model input is `~116`-effective-dimensional.

Size control.  The train-fit PC1 score (projected on validation) correlates
with size and target as follows (Pearson):

| version | vs `n_nodes` | vs `n_edges` | vs target | vs `M0` residual | vs `MB` branch |
|---|---:|---:|---:|---:|---:|
| raw `C` | −0.267 | −0.235 | −0.364 | +0.015 | +0.050 |
| `C/n` | −0.041 | −0.012 | −0.338 | +0.014 | +0.035 |
| `C/√n` | −0.151 | −0.119 | −0.356 | +0.014 | +0.040 |
| model input `C̃` | −0.029 | −0.017 | −0.050 | +0.001 | +0.129 |

Raw `C` PC1 carries a mild size/global-scale component (`corr ≈ −0.27` with
`n_nodes`); the RMS-scaled model input is essentially size-free.  **But in no
version does a single PC explain the trained `B`** (max `corr(PC1, b) = 0.13`,
i.e. `R² ≈ 0.017`).  Near-rank-1 was neither a real assignment axis nor a pure
size artifact — it was a bug.

---

## 2. H1 — the three seeds learn the **same function** but not the same raw coefficients

`b_s(G) = ⟨W_B^(s), C̃(G)⟩` on all 1,000 validation molecules.

| pair | branch Pearson | branch Spearman | final-pred Pearson | `W_B` scaled cosine | raw-unit cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0–1 | 0.9922 | 0.9898 | 0.9965 | 0.9542 | **−0.2535** |
| 0–2 | 0.9878 | 0.9859 | 0.9978 | 0.9586 | **−0.6292** |
| 1–2 | 0.9894 | 0.9856 | 0.9974 | 0.9861 | **0.9056** |

Branch-output std: `0.673 / 0.677 / 0.653` (seeds 0/1/2).

This is the central H1 result:

* **prediction-level agreement is near-perfect** (branch Pearson `≥0.988`,
  final-pred `≥0.9965`) — the seeds converge to the same assignment score;
* **scaled-space coefficients agree** (cosine `0.95–0.99`);
* **raw-unit coefficients disagree wildly** (cosine `−0.63 … +0.91`, even
  negative) — a seed can move mass onto a different near-zero-RMS coordinate
  and produce an almost identical branch output.  Any raw-unit reading of
  `W_B` is therefore seed-unstable and must not be interpreted.

---

## 3. H3 — no dominant low-dimensional assignment direction

Train-only PCA/SVD basis on the flattened `C̃` (model input), projected onto
validation; `b_s ~ z_1..z_k`, linear `R²`:

| seed | Pearson(`z1`,`b`) | `R²` `z1` | `z1..2` | `z1..5` | `z1..10` |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.129 | 0.0167 | 0.0741 | 0.2165 | 0.3729 |
| 1 | 0.114 | 0.0130 | 0.0782 | 0.2062 | 0.3767 |
| 2 | 0.073 | 0.0054 | 0.0746 | 0.1918 | 0.3778 |

(in-sample on validation; out-of-sample train→valid `R²` at `k=10` is only
`0.082 / 0.093 / 0.103`).  So:

* a **single PC1 explains 0.5–1.7 % of the trained `B`** — the AR0 worry that
  "the 1820-D tensor is effectively one direction" is false;
* even `k=10` explains only `~37 %` in-sample, and that does not transfer
  out-of-sample (the top train PCs are not stable directions), so the trained
  `B` is spread across many, individually-weak coordinates rather than
  compressed into a stable low-dimensional subspace.

---

## 4. H4 — where the contribution actually sits (structure × atom)

`β_raw = W_B / D_C` recovered to `[65, 28]` and mapped through the fixed
group layout (`root 0:11`, `node_mean 11:22`, `node_std 22:33`,
`edge_mean 33:48`, `edge_std 48:63`, `patch_size 63:65`).  Contribution is
`contrib_{k,j}(G) = β_raw_{k,j} · C_raw_{k,j}(G) = W_B_{k,j} · C̃_{k,j}(G)`
(verified: `max |Σ contrib − recorded branch| ≈ 1.5e-5`).

Structural-group contribution share (RMS), per seed:

| group | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| root_basis | 0.045 | 0.038 | 0.038 |
| node_mean | 0.195 | 0.191 | 0.185 |
| node_std | 0.242 | 0.248 | 0.243 |
| edge_mean | 0.234 | 0.241 | 0.240 |
| edge_std | 0.244 | 0.246 | 0.257 |
| patch_size | 0.040 | 0.036 | 0.037 |

* **~94 % of the contribution is in the pooled node/edge statistics**
  (`mean` + `std`), split ~ evenly between node and edge channels;
* the rooted `root_basis` and the raw `patch_size` coordinates carry only
  `~4 %` each, despite being part of the nominal `S`; the assignment readout
  is essentially a *centred mean/variance* alignment between a node role and
  an atom category, not a rooted-orbit-specific one.

Atom-category contribution share is also seed-stable; top categories (seed 0 /
seed 1 / seed 2):

| category | seed 0 | seed 1 | seed 2 |
|---|---:|---:|---:|
| N | 0.151 | 0.150 | 0.158 |
| C | 0.109 | 0.103 | 0.112 |
| O | 0.117 | 0.098 | 0.098 |
| N H1 + | 0.089 | 0.092 | 0.087 |
| C H1 | 0.083 | 0.091 | 0.090 |
| N + | 0.087 | 0.078 | 0.080 |
| S | 0.062 | 0.080 | 0.076 |
| F | 0.047 | 0.052 | 0.052 |

So the assignment readout is dominated by the chemically distinct heteroatom
categories (N and its protonation states, O, C, S, F) — i.e. it is *not* an
arbitrary numerical direction; it tracks real atom-type/position alignment.

**Weight ≠ contribution.**  The 8 largest `|β_raw|` coordinates (seed 0) sit
on ultra-rare charged/special categories (`S +`, `O +`, `O H1 +`,
`N H1 -`) with `|β_raw|` up to `1303` and **contribution RMS exactly 0.0**.
Because `C̃` is unit-RMS per effective coordinate, `contrib_rms(k,j) ≈
|W_B(k,j)|`, and the interpretable quantity is the scaled `W_B`, never
`β_raw`.

Cross-seed consistency of the actual contribution profile:

| pair | contribution-RMS cosine | Spearman |
| --- | ---: | ---: |
| 0–1 | 0.960 | 0.983 |
| 0–2 | 0.965 | 0.985 |
| 1–2 | 0.988 | 0.991 |

So although raw coefficients are seed-unstable (§2), the **contribution map is
highly reproducible**.

---

## 5. H5 — frozen-`M0` + linear `C` recovers ~2/3 of the full `MB` gain

For each seed: take the **official `M0` soup**, freeze it completely, and train
only `W_{B,frozen} ∈ R^{65×28}` (zero-init, no bias, no MLP, no activation) on
`ŷ = M0_soup(x) + ⟨W_{B,frozen}, C̃⟩`, with the identical protocol (Adam
`lr 1e-3`, `wd 1e-5`, batch 128, L1, clip 5, max 240, patience 40, best
official-valid checkpoint, fixed Top-5 soup).  `M0` is never retrained.  The
frozen base has no dropout/BN, so per-molecule base predictions were
precomputed exactly.  Remote revision `6acfc45`, deterministic.

| seed | `M0` | frozen `M0`+B (soup) | original `MB` (soup) | `Gain_frozen` | `Gain_MB` | recovery |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.553790 | **0.494538** | 0.469285 | +0.059252 | +0.084505 | **0.701** |
| 1 | 0.535570 | **0.481283** | 0.458746 | +0.054287 | +0.076824 | **0.707** |
| 2 | 0.546893 | **0.490486** | 0.462365 | +0.056407 | +0.084528 | **0.667** |

`RecoveryFraction = Gain_frozen / Gain_MB`.

Interpretation: **≈ 67–71 % of the entire node-assignment gain is obtained by
a plain linear readout on a frozen strong base**, with no base/B co-adaptation
at all.  The remaining `~30 %` is what the jointly-trained `MB` adds on top
(likely base re-shaping around the assignment term, since the frozen base must
absorb the whole residual on its own during its original training).  This is
strong mechanism evidence that the `MB` gain is mostly the statistic `C`
itself, not an optimization artifact of joint training.

`W_{B,frozen}` norms: `0.605 / 0.681 / 0.744` (soup `0.670 / 0.660 / 0.674`),
same order as the original `MB` branch norms (`0.747 / 1.026 / 0.937`).

---

## 6. H6 — illustrative exact-`A`-key witnesses (not a proof)

Ranked by exact `(atom-count, bond-count)` key collision, then structural
marginal distance, then `|b(G1) − b(G2)|`; target/residual inspected only
after ranking.  Among the 1,000 validation molecules there are exactly
**9 exact-`A`-key collision pairs**.

| pair | `n` | `m` | `|Δb|` | struct. marg. dist | `Δ(M0 residual)` | sign match |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| 118 \| 502 | 24 | 25 | 0.585 | 7.9 | +0.628 | yes |
| 334 \| 613 | 23 | 25 | 0.467 | 16.8 | −0.208 | yes |
| 47 \| 887 | 26 | 29 | 0.377 | 22.6 | −0.487 | yes |
| 63 \| 570 | 26 | 28 | 0.204 | 10.7 | −0.868 | yes |
| 480 \| 730 | 21 | 22 | 0.199 | 5.8 | +0.490 | yes |
| 454 \| 493 | 26 | 29 | 0.184 | 8.9 | +0.453 | no |
| 117 \| 410 | 25 | 26 | 0.176 | 17.7 | +0.236 | no |
| 575 \| 626 | 21 | 23 | 0.043 | 23.2 | −0.796 | no |

`A` is exactly equal in every row (same counts ⇒ same `a(G)`); the pairs
differ only in where the atoms/bonds sit.  For the 5 largest-`|Δb|` pairs the
`M0`-residual difference has the **same sign** as the `B`-branch difference,
i.e. the learned assignment score points the same way the frozen base's error
points.  These are **illustrative witnesses of a real, `A`-invariant
assignment signal**, not a statistical proof (small `n`, and 3/8 pairs do not
match sign).

---

## 7. Judgment

> Is the current node assignment signal a stable low-dimensional functional,
> or a seed/scale-dependent artifact?

**It is a stable, high-dimensional functional — not a low-dimensional one and
not an artifact.**

* Stable: branch outputs agree across seeds at Pearson `≥0.988`; the actual
  contribution map agrees at cosine `≥0.96`; the structural-group and
  atom-category contribution profiles are nearly identical across seeds; and
  `~2/3` of the effect survives a frozen base.
* High-dimensional: the flattened `C` / `C̃` spectra are flat (effective rank
  `57 / 116`), a single PC1 explains `<2 %` of the trained `B`, and `k=10` PCs
  explain only `~37 %` in-sample (and far less out-of-sample).  The readout is
  spread over many weak coordinates, not compressed into one direction.
* Not raw-coefficient-identifiable: raw-unit cosine between seeds ranges from
  `−0.63` to `+0.91`, while the *function* is nearly identical.  Only the
  scaled coefficient / actual contribution is interpretable.
* Mechanistically, the contribution is `~94 %` pooled node/edge mean+std
  statistics, dominated by heteroatom categories (N, O, C, S, halogens) — a
  centred global role↔atom alignment, **not** a rooted-orbit-specific or
  size-driven effect.

Retractions: the AR0 caveat "`C` is close to rank-1 and the linear readout has
roughly one effective degree of freedom" is **withdrawn**.  The correct
statement is the opposite: the statistic is high-dimensional, and its value
comes from many small, seed-stable contributions.

---

## 8. Reproduce

```
python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism h1_seed_agreement --deterministic
python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism h2_rank --deterministic
python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism h3_pc_explanation --deterministic
python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism h4_contributions --deterministic
python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism h6_witnesses --deterministic
# H5 on the deployed revision (remote A100 host, GPU 1, CPU tensors):
uv run python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism h5_frozen_m0 --deterministic
python -m tracks.ksvd.experiments.luyin16.fsar_r2_ar0_mechanism summary
```
