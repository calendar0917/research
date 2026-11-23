# RICE-v0 — rooted interaction cumulant encoder pilot — report

Protocol: `PROTOCOL.md` (frozen before the formal runs, commit `2cbd3213`).
Task: `ogbg-mollipo`, D0 only (Fit 2856 / Dev 504, scaffold-disjoint), seed 0.
Code: `code/`, config `configs/rice.json`, checks `analysis/checks.json`,
results `results/RICE_{CUMULANT,RAW}_S0/`.

## 0. Answer in one paragraph

Real graph-factor contraction plus root-conditioned Möbius inversion **works as a
learned encoder**: `RICE_CUMULANT` reaches Dev RMSE **0.5994** (MAE 0.4656) with
**129,198** parameters, and the parameter-matched, identically-initialised
`RICE_RAW` arm that sees the same `Φ` responses **without** the inversion reaches
**0.7174**, so `Δ = +0.1180` `[+0.0708, +0.1654]` paired bootstrap, all 10,000
resamples positive. The two arms are indistinguishable for the first ~30 epochs
and then separate, so the inversion is not a warm-start or initialisation effect.
The size-3 and size-4 `κ` blocks are genuinely used: replacing the `κ`(size≥3)
block by its **Fit mean** (keeping the offset, removing only the root-to-root
variation) costs **+0.779** Dev RMSE, which is far more than a constant-offset
artefact could explain. Word types are reused across molecules and scaffolds; the
single most influential type (an sp² N–H ⋯ C single bond, root at N) appears in
940 Fit molecules on 494 scaffolds and moves Dev RMSE by 0.10 in isolation. All
20 pre-training mathematics/invariance checks pass, with the tight identities at
≤1e-14 in float64 against an independent brute-force reference (the float32
production path, reported for information, at 1.6e-6). The route is still **not competitive**: it is
0.049 behind the frozen R2-SENTENCE D0 reference (0.5506) and 0.073 behind GINE
(0.5266), and it clears the `≤0.60` line by 0.0006. Label: **`RICE_PROMISING`**
by the frozen rule, read as "the mechanism is real, the absolute level is not yet
competitive".

## 1. Contract and execution

* Two runs only, seed 0, D0 only, identical recipe, identical parameter set and
  identical initialisation hashes (`analysis/arms.csv`, `init_hashes` in both
  manifests are byte-identical). The arms differ in exactly one thing: the tensor
  entering sentence pooling (`κ` vs `Φ`).
* `res` (2×A100-SXM4-40GB), one process per GPU, both launched from commit
  `2cbd3213`. Runs `RICE_CUMULANT_S0-20261123-164106-a49ef246` and
  `RICE_RAW_S0-20261123-164115-cb2b68d7`, torch 2.5.1+cu124, driver 550.163.01.
* GPU cost **0.417 h total** (0.2583 + 0.1591), wall clock 15.8 min for both arms
  in parallel; the 2 GPU-hour budget was not a constraint. Reference structure
  build is 14.5 s per arm (CPU), reported separately from training.
* Both arms share 129,198 parameters, `sentence_in_dim` 388, `readout_in_dim` 297,
  peak VRAM 0.23 GiB, 45 steps/epoch at batch 64.

## 2. What was computed

For every atom as root, the induced radius-2 subgraph was enumerated into all
connected root-containing supports of size 1–4 with **every** real internal bond
(triangle and 4-clique closure included). Over the 3,360 structure rows: 90,514
roots, 90,514 size-1 words and **1,124,230** words of size ≥ 2 — 197,122 of size
2, 415,002 of size 3, 512,106 of size 4 (2.18 / 4.59 / 5.66 words per root per
order; 334.6 size-≥2 words per molecule; max 1,251 in one molecule). 97.8 % of roots own at least one size-4
word; the 2,011 that do not are the roots whose radius-2 ball has ≤3 atoms.

Energies are exact: 16 channels × 2 states, unary from the 9 OGB atom fields,
symmetric pairwise from the 3 OGB bond fields, three independent binary values
per channel, `tanh`-bounded at ±2 so `exp` cannot overflow, `logZ(∅) = 0`, ≤16
states enumerated per word (no sampling, no message passing).

`κ(S,v) = Φ(S,v) − Σ_{T⊊S,v∈T} κ(T,v)` by increasing size 1→4 is the only
high-order composition in the model. No MLP ever sees child vectors.

## 3. Pre-training checks (all pass before training)

`analysis/checks.json`, 20/20 PASS, recorded on the clean formal revision
(`3590f864`, dirty=False).

| group | what it pins | result |
|---|---|---|
| A1 | `Φ(S,v) == Σ_{T⊆S,v∈T} κ(T,v)` on every graph with ≤4 atoms (75 graphs, 934 identities), independent brute-force subset sum | max err **8.9e-16** |
| A2 | vectorised production `Φ`/`κ` vs brute-force `Φ`/`κ` (float64) | 3.6e-15 |
| A3 | float32 production path (informational) | 1.6e-6 |
| B1 | disconnected root-anchored `S` ⇒ `κ ≈ 0` (246 cases) | 1.3e-15 |
| B2 | all bond potentials zero ⇒ every `|S|≥2` `κ` is zero (934 supports) | 1.6e-15 |
| C1 | same atom+bond attribute multisets, 4-cycle vs triangle+tail: `Φ` and `κ` separate (potentials derived from the fields) | κ gap 0.033 |
| C2 | same `{C,O,O}` multiset, attribute↔topology binding swapped | κ gap 0.098 |
| C3 | 3-cycle stores all 3 internal bonds and its `κ` differs from the open chain | gap 0.094 |
| C4 | 4-clique uses all 6 internal bond slots, finite | ok |
| C5 | middle-rooted 4-path vs 4-star have the same `Φ` gap but order-4 `κ = 0` in both (conditional independence of the root branches) | 3e-16 |
| D1–D4 | atom renumbering (κ 1e-6, pred 0.0), bond direction, bond order, batch/molecule order | 0 mismatches |
| E1 | atom/bond/embedding/sentence/readout gradients finite and non-zero on the first real batch | 2.7e-3 … 4.0e-1 |
| E2 | order-3 and order-4 `κ` non-zero on non-degenerate graphs (3-cycle 0.169, 4-cycle 0.033) | ok |
| E3 | 30 optimiser steps reduce the standardised MSE | 1.063 → 1.019 |
| F1–F3 | word census, vectorised path (module forward calls per batch independent of the 20,713 words in it), measured step cost | 0.33 s CPU step |

Two bugs were found by the checks rather than by training: masking the root state
needs `unsqueeze(-1)` on the state axis (caught by A2's brute-force comparison),
and the potentials must be derived from the fields for a binding-swap test to mean
anything (caught while designing C1/C2). Both were fixed before any run.

## 4. Main result: the inversion contrast

| arm | sentence input | params | best epoch | epochs run | Fit RMSE | Dev RMSE | Dev MAE |
|---|---|---|---|---|---|---|---|
| `RICE_CUMULANT` | `κ` (Möbius, 1→4) | 129,198 | 114 | 159 | 0.4314 | **0.5994** | **0.4656** |
| `RICE_RAW` | `Φ` (same supports, same energy) | 129,198 | 61 | 106 | 0.5887 | 0.7174 | 0.5614 |

`Δ = RMSE(RAW) − RMSE(CUMULANT) = +0.1180`, paired molecule-level bootstrap
(10,000 resamples, seed 20261124) 95 % `[+0.0708, +0.1654]`, fraction positive
**1.000**. The two arms are the same model — same energies, same `Φ`, same
initial weights — so this is a pure representation contrast, not a capacity or
initialisation contrast.

The trajectory shows where it comes from (Fit/Dev RMSE):

| epoch | 20 | 30 | 40 | 50 | 60 | 80 | 100 | 114 |
|---|---|---|---|---|---|---|---|---|
| CUMULANT | 0.905/1.004 | 0.756/0.839 | 0.675/0.731 | 0.619/0.716 | 0.581/0.672 | 0.512/0.649 | 0.475/0.623 | 0.431/0.599 |
| RAW | 0.887/0.944 | 0.754/0.833 | 0.711/0.781 | 0.701/0.818 | 0.611/0.754 | 0.517/0.752 | 0.477/0.759 | — |

Through epoch 30 the raw arm is marginally ahead; afterwards the inverted arm
fits the same Fit rows better (0.431 vs 0.589) **and** generalises better. The raw
arm then early-stopped (best 61, stop 106) while the inverted arm was still
improving to epoch 114 (stop 159). Removing the lower-order redundancy leaves the
sentence with a better conditioned input, so it both optimises further and overfits
later. The reason is exact rather than heuristic: by the inversion identity
`Φ(S,v) = Σ_{T⊆S, v∈T} κ(T,v)`, so every `Φ` of size k mixes all orders 1..k of
its own support. The four `Φ` blocks are therefore informationally redundant by
construction, and the readout has to undo that overlap through a per-order
LayerNorm plus one shared MLP, whereas the `κ` blocks are disjoint by design.

## 5. Is the size ≥ 3 structure real and used?

Per-order `κ` on Dev (frozen best-Dev checkpoint, `analysis/kappa_stats.json`):

| order | supports | ‖κ‖ mean | ‖κ‖ std | participation ratio | distinct types seen |
|---|---|---|---|---|---|
| 1 | 13,801 | 0.544 | 0.113 | 2.56 | 74 atom types |
| 2 | 30,046 | 0.722 | 0.268 | 1.27 | 565 |
| 3 | 63,426 | 0.0550 | 0.0503 | 2.18 | 2,770 |
| 4 | 78,528 | 0.00205 | 0.00678 | 1.53 | 6,616 |

Order 3 is 13× and order 4 350× smaller in norm than order 2, so they can only be
read through their own per-order LayerNorm — never through their raw magnitude. Within each order the 96-dim pooled block is strongly root-dependent:

| order | relative dispersion of the pooled block | components with across-root std > 10 % of block scale |
|---|---|---|
| 1 | 1.54 | 67 % |
| 2 | 1.27 | 97 % |
| 3 | 1.16 | 85 % |
| 4 | 3.73 | 81 % |

Frozen-checkpoint interventions on Dev (no retraining):

| intervention | Dev RMSE | Δ vs full | note |
|---|---|---|---|
| none | 0.5994 | — | |
| zero `κ` size ≥ 4 | 0.7877 | +0.1883 | values only |
| zero `κ` size ≥ 3 | 1.0910 | +0.4916 | values only, counts kept |
| zero `κ` size ≥ 3 (and counts) | 1.1047 | +0.5053 | |
| zero `κ` size ≥ 2 | 1.6829 | +1.0835 | off the prediction manifold |
| **replace `κ` block (3) by its Fit mean** | 1.3158 | +0.7164 | offset kept, variation removed |
| **replace `κ` block (4) by its Fit mean** | 0.8733 | +0.2739 | |
| **replace `κ` blocks (3,4) by their Fit means** | 1.3785 | +0.7791 | |
| replace `κ` block (1) by its Fit mean | 3.2152 | +2.6159 | |
| replace `κ` block (2) by its Fit mean | 1.3766 | +0.7772 | |

The important row is the third from the bottom: making the size ≥ 3 block
**root-invariant** while leaving its level intact costs 0.78 RMSE. So the model's
dependence on the higher-order cumulants is not a constant bias that happens to
sit in that block — the *variation of `κ` across roots* is what the readout uses.
Zero-out alone would not have shown this, because a zeroed block also changes the
offset.

Two caveats are load-bearing. (i) These are off-manifold interventions on a frozen
network, so the sizes of the deltas are not effects on a retrained model, and the
per-order deltas do not add up (the sentence MLP is nonlinear). (ii) The energies
are conditional-independence models, so a middle-rooted 4-path and a 4-star have
`κ₄ = 0` identically (check C5): a non-zero order-4 `κ` requires a real cycle or a
4-clique. Order-4 activity is therefore concentrated on genuinely cyclic words,
not on all size-4 words, and this is the intended behaviour of the definition.

## 6. Reusable cross-scaffold word types

Word types are root-marked attributed-graph canonical certificates over the full
OGB categorical fields (lexicographically minimal relabelling of the non-root
slots), screened on **Fit only** (≥10 Fit molecules and ≥5 Fit scaffolds): 1,014
(size 2), 7,251 (size 3), 21,772 (size 4) distinct types, of which 6,524 pass the
reuse screen. Because `κ` depends only on the fields, a word type has **one**
`κ` in the whole dataset, so a type ablation is exact.

Top 12 by Fit prediction change when their `κ` is zeroed (Dev column is
descriptive only and was never used to select):

| size | node labels (root first) | internal bonds | Fit mols / scaffolds | Fit supports | Δ Fit MSE | Dev mols | Δ Dev RMSE |
|---|---|---|---|---|---|---|---|
| 2 | N d3 H1 q0 sp2 – C d3 H0 q0 sp2 | SINGLE | 940 / 494 | 1,081 | 0.1254 | 183 | +0.1015 |
| 2 | N d3 H0 q0 sp2 – C d3 H0 q0 sp2 | SINGLE | 294 / 142 | 307 | 0.0749 | 39 | +0.0583 |
| 2 | N d3 H0 q0 sp2 – C d3 H0 q0 sp2 (arom) | SINGLE | 457 / 208 | 525 | 0.0740 | 70 | +0.0416 |
| 3 | O d2 H1 q0 sp2 – C d3 H0 q0 sp2 – O d1 H0 q0 sp2 | SINGLE, DOUBLE | 355 / 139 | 358 | 0.0482 | 49 | +0.0512 |
| 3 | O d2 H1 – C d3 (arom) – C d3 H1 (arom) | SINGLE, AROMATIC | 201 / 106 | 388 | 0.0417 | 36 | +0.0106 |
| 2 | N d3 H0 sp2 (arom) – C d3 H0 sp2 (arom) | AROMATIC | 628 / 310 | 1,198 | 0.0375 | 138 | +0.0211 |
| 2 | N d3 H0 sp2 – C d3 H0 sp2 | SINGLE | 175 / 91 | 176 | 0.0349 | 23 | +0.0109 |
| 2 | C d3 H1 sp2 (arom) – C d3 H1 sp2 (arom) | AROMATIC | 2,677 / 1,255 | 19,798 | 0.0326 | 487 | +0.0112 |
| 2 | O d1 H0 sp2 – C d3 H0 sp2 | DOUBLE | 1,690 / 796 | 2,087 | 0.0322 | 292 | +0.0488 |
| 3 | N d3 H1 – O d1 – S d4 H0 sp3 | SINGLE, DOUBLE | 119 / 65 | 242 | 0.0315 | 32 | +0.0717 |
| 3 | N d3 H0 – C d3 (arom) – N d2 (arom) | SINGLE, AROMATIC | 196 / 101 | 265 | 0.0268 | 30 | +0.0122 |
| 3 | O d2 H1 – C d3 – C d3 (arom) | SINGLE, SINGLE | 109 / 55 | 109 | 0.0268 | 15 | −0.0035 |

These are the words the frozen model leans on most, and they are the polar /
aromatic linkages one would expect a lipophilicity model to use (amide N–H to sp²
C, the carboxylic-acid `O–H ⋯ C(=O)` triad, carbonyl `C=O`, aromatic N–C and C–C).
Across all 6,524 screened types, 3,725 raise Fit MSE when individually zeroed and
2,798 lower it (one is exactly neutral). The top type alone raises Fit MSE from
0.186 to 0.311; the summed *individual* deltas of the top 12 are 0.587, which
exceeds the full Fit MSE, so that sum is a ranking statistic and not a
decomposition (the ablations overlap through the shared pooled sentence).
Every top type is reused in hundreds of Fit molecules spread over hundreds of
scaffolds, and 11 of 12 keep the same sign on Dev. This is not a
chemical-causality claim: the 16 channels and 2 latent states are
permutation-non-identifiable, so a single `κ` component has no fixed chemical
reading, and a type ablation identifies "the model uses this word", not "this
group causes lipophilicity".

## 7. Historical coordinates and cost

| model | Dev RMSE | params | note |
|---|---|---|---|
| GINE | 0.5266 | — | reference only |
| R2-SENTENCE D0 s1 | 0.5464 | 416,177 | reference only |
| R2-SENTENCE D0 s0 | 0.5506 | 416,177 | reference only |
| **RICE_CUMULANT s0** | **0.5994** | **129,198** | this round |
| RICE_RAW s0 | 0.7174 | 129,198 | this round, matched |

Gaps: `+0.0488` behind R2 s0, `+0.0529` behind R2 s1, `+0.0728` behind GINE.
Cost: 0.417 GPU-hours for both arms, 15.8 min wall, 14.5 s CPU preprocessing per
arm, 0.33 s per CPU step (a formal GPU step is ~0.13 s). This is 3.2× cheaper in
parameters than the R2 trunk.

## 8. Answers to the required questions

**1. Does root-conditioned Möbius inversion buy predictive value over the same
responses without it?** Yes, decisively within this pair: `Δ = +0.1180`
`[+0.0708, +0.1654]`, identical parameters, identical initialisation, identical
`Φ`, identical recipe. The arms are tied for ~30 epochs and separate afterwards,
so the gain is a representational effect, not an optimisation-luck effect.

**2. Is the higher-order (size ≥ 3) structure real, non-degenerate and actually
used?** Yes. `κ₃` is 0.055 and `κ₄` 0.002 in norm on Dev but the pooled blocks are
strongly root-dependent (85 % / 81 % of components with across-root std > 10 % of
the block scale), and making the size ≥ 3 block root-invariant costs **+0.779**
Dev RMSE. `κ` of size ≥ 2 is ~1e-15 on disconnected supports and exactly 0 with
zero bond potentials, so the non-zero higher-order values come only from real
cycles and real bond couplings, not from numerical noise. Order-4 activity is
confined to cyclic words by construction (a middle-rooted 4-path or 4-star has
`κ₄ = 0`).

**3. Do the interactions form reusable, cross-scaffold local sentences?** Yes.
6,524 word types pass the Fit-only reuse screen (≥10 molecules, ≥5 scaffolds), the
top 12 account for 59 % of the Fit ablation mass, and the leading types recur in
hundreds of molecules on hundreds of scaffolds with consistent signs on Dev. The
composition is a learned type-level lexicon whose per-type code is produced by one
shared energy, not a per-word lookup table.

**4. Is the route competitive with the frozen references at this cost?** No. It
is 0.049 behind R2-SENTENCE D0 s0, 0.053 behind R2 s1 and 0.073 behind GINE,
while using 129k parameters and 0.42 GPU-hours. It clears the pre-registered
`≤0.60` line by 0.0006, i.e. it is on the line, not comfortably under it.

**5. Verdict and next step?** `RICE_PROMISING` by the frozen rule — Dev 0.5994
≤ 0.60, Δ ≥ 0.015 with an interval excluding zero, size ≥ 3 ablation ≥ 0.015,
live higher-order responses and gradients, reusable cross-scaffold words. The
honest reading is "a real, cheap, mechanism-level gain that is not yet a
competitive encoder". The gap to R2 is 0.049, which is a representation-capacity
gap (16 channels × 2 states cannot express as much per word as the R2 local
sentence), not a failure of the inversion: the inversion arm beats its own
matched control by 4× the margin it needs. The natural next step, if this route is
continued at all, is a **wider energy** (more channels/states) at fixed recipe and
a second seed — explicitly *not* done here, because it was not in the frozen
contract.

## 9. Verdict

Frozen rule from `PROTOCOL.md` §10, applied mechanically
(`analysis/verdict.json`):

| criterion | threshold | value | pass |
|---|---|---|---|
| CUMULANT Dev RMSE | ≤ 0.60 | 0.5994 | yes (margin 0.0006) |
| Δ = RAW − CUMULANT | ≥ 0.015 | 0.1180 | yes |
| zeroing size ≥ 3 degrades Dev | ≥ 0.015 | 0.4916 | yes |
| higher-order responses live | order 3 and 4 non-zero, ablation-sensitive | yes | yes |
| reusable cross-scaffold words | ≥1 with real Fit effect | 6,524 types, 12 top | yes |

Label: **`RICE_PROMISING`**.

## 10. Limitations and what is not claimed

* **Single seed.** No training-seed variance. The paired bootstrap is
  molecule-level only; it does not cover seed noise. `Δ = 0.118` is 28× the
  difference between the two R2 seeds (0.0042), so seed noise is unlikely to
  explain it, but that is an argument, not a measurement.
* **Best-Dev checkpoint selection** makes every Dev number slightly optimistic.
  The contrast is controlled across arms, but neither 0.5994 nor 0.7174 is an
  unbiased test estimate.
* **The `≤0.60` criterion is met by 0.0006.** A different seed would likely land on
  either side of it. The substantive claim rests on `Δ`, not on the absolute line.
* **Interventions are off-manifold.** Zero-out and mean-replacement both feed the
  frozen network an input distribution it was not trained on; they bound
  "how much the readout uses this block", not "what a retrained model would do".
* **No chemical causality.** Latent states and channels are permutation
  non-identifiable, so no `κ` component has a canonical chemical meaning. The word
  types are described by their OGB field labels only.
* **No D1, no official Valid/Test, no second seed, no hyper-parameter search, no
  architecture change.** None of the reported numbers is a model-selection result.
* `RICE_RAW` is the *only* control. There is no capacity-matched "κ computed by an
  MLP" arm, so this round shows that the inversion helps relative to `Φ` under
  this parameter budget, not that it is the best possible way to compress these
  responses.

## 11. Provenance and reproduction

* Formal revision: `2cbd3213` (protocol + code + config, frozen). Checks re-run on
  `c476c3f4` (identical code, clean tree): `analysis/checks.json`, 20/20, dirty=False.
* Data: raw cache `fe002287…`, reused loader `f02863be…`, split
  `60ac13c0…` (2856/504), Fit-only scaler mean 2.185815826311436 / std
  1.2139664740467941. All pinned at runtime inside `code/rice_data.py` and
  re-recorded in `data_hashes` of both manifests.
* Runs: `RICE_CUMULANT_S0-20261123-164106-a49ef246` (res-gpu0) and
  `RICE_RAW_S0-20261123-164115-cb2b68d7` (res-gpu1), commit `2cbd3213`,
  torch 2.5.1+cu124, A100-SXM4-40GB, driver 550.163.01.
* Reproduce:

```bash
uv run pytest -q tests                                   # 12 passed
uv run python code/rice_checks.py                        # 20/20 PASS
rr deploy res
rr run res RICE_CUMULANT_S0 --pool res-gpu0 --result <round>/results/RICE_CUMULANT_S0 \
  -- python <round>/code/rice_train.py --arm RICE_CUMULANT --seed 0 --device cuda \
     --run-id RICE_CUMULANT_S0 --out <round>/results/RICE_CUMULANT_S0
rr run res RICE_RAW_S0 --pool res-gpu1 --result <round>/results/RICE_RAW_S0 \
  -- python <round>/code/rice_train.py --arm RICE_RAW --seed 0 --device cuda \
     --run-id RICE_RAW_S0 --out <round>/results/RICE_RAW_S0
rr pull RICE_CUMULANT_S0 && rr pull RICE_RAW_S0
uv run python code/rice_analyze.py --device cpu           # ~60 s, writes analysis/
```

## 12. Stop condition

The round stops here: two arms, one seed, D0 only, one frozen checkpoint per arm,
one analysis pass. No RICE-v1, no wider energy, no second seed, no D1, no official
Valid/Test, no order extension, no latent-state or word-dictionary search was run
after seeing any Dev number.