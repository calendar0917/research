# REPORT — LCRC/Lipo multi-class connection-motif selectivity, parameter dependence and internal path

Round: `lcrc-lipo-multi-motif-mechanism-public122-v1`
Protocol: `PROTOCOL.md` (frozen at commit `0273cad6`, before any prediction)
Frozen source / checkpoint: `lcrc_lipo_amide_connected_rewire_public122_v1` (`64bb1c9f`),
`A_s0` (`bdd670e8…`, state `7d32221e…`), verified training-start init (`aac1e061…`)
Compute: `rr` run `lipo-multimotif-trained-s0` (`res-gpu0`, `7e2640a6`) and
`lipo-multimotif-init-s0` (`res-gpu1`, `5b8367a2`), both at commit `0273cad6e8ab`, A100-SXM4-40GB,
driver 550.163.01, torch 2.5.1+cu124 — **0.021 GPU·h total** (plan ≤ 0.2).
Zero new training: 0 optimiser, 0 backward, 0 probe, 0 new seed; both parameter states hash-identical
before and after inference.

---

## 0. Answers to the five questions, first

**Q1 — which connection families are studiable at all.** Five of the six pre-registered cells can be
rewired on Valid/Test: `amide` (533 Test pairs), `sulfonamide` (112), `aryl_ether` (112), `ester` (46),
`arylamine` (41). **`sulfonate` is not studiable**: no Valid/Test molecule has a sulfonate `S(=O)₂–O–C`
whose target bond is a bridge with a usable carbon control (only 2 Train molecules). A dose-calipered
control `C` exists for only part of each queue — 38 % of Test amide molecules (21 % of pairs), 43 % of
sulfonamide (26 %), 24 % of arylamine (10 %), 50 % of ester (15 %), 14 % of aryl-ether (9 %) — which
caps what can be said about the others.

**Q2 — does selectivity survive matching the *statistical* dose (not just the physical bond count).**
Only **`amide`** survives in full: Test `rC_TRAINED = +0.164`, 95 % CI `[0.052, 0.278]`, Bonferroni
12-item interval `[0.005, 0.334]`. **`sulfonamide`** is *partial*: `rC = +0.327 [−0.008, 0.617]`
(point positive, descriptive CI touches zero, Bonferroni covers zero) but its `E` is
Bonferroni-clear. **`arylamine`** has `rC = +0.127 [0.106, 0.189]` on 4 molecules.
`ester` (`rC = −0.187`) and `aryl_ether` (`rC = −0.433 [−0.872, −0.158]`) are **reversed**: under an
equal-dose control the model moves *more* when the pattern is kept, not less. For `sulfonate` the
competing explanation is **not excluded** (no C at all). For 50 of the 244 calipered amide focal
pairs the only available `C` is the view's *own* `K0` (13 of 68 for sulfonamide, 0 for the other
families); restricting to focals that have a *foreign* C leaves the estimates essentially unchanged
(amide `rC = +0.146 [0.018, 0.269]`, sulfonamide `+0.331 [−0.009, 0.631]`, arylamine/ester/aryl-ether
identical because they have no own-`K0`-only focal), so the calibration result is not an artefact of
that overlap.

**Q3 — does it depend on the trained parameters.** For the two positively-selective families, yes:
`E = rC_TRAINED − rC_INIT` is `+0.321 [0.159, 0.479]` (amide, Bonferroni `[0.089, 0.556]`) and
`+0.596 [0.257, 1.006]` (sulfonamide, Bonferroni `[0.072, 1.151]`) on *identical* views, C set and
weights. The verified training start does **not** prefer the target pattern (`rC_INIT = −0.157` and
`−0.269`), so this preference is a product of learning. For `arylamine` the sign is opposite
(`E = −0.378`): the untrained model already preferred the target pattern more than the trained one.
No degenerate denominators occurred (0/1183 amide pairs; the INIT state is not constant: median
`aT_INIT = 0.094` vs `0.466` trained).

**Q4 — through which internal path.** The atom-scalar block is
~zero for every family (|γ₃| ≤ 0.04). The response flows through the **S2 (typed two-path) block and
its interaction with S1**, not through S1 alone: for the dose-calipered comparison the conditional
S2 recovery `H(123) − H(13)` is `+0.46` (amide), `+0.55` (sulfonamide), `+0.64` (arylamine),
`+0.27` (aryl-ether), `+0.16` (ester), while the conditional S1 recovery `H(123) − H(23)` is
≤ 0.11 except for arylamine. The all-pairs comparison `T − K0` gives the same ordering except that
the amide response there is S1-dominant (`γ₁ = +0.126` vs `γ₂ = +0.033`), and the two large families
also carry sizeable **negative** S1×S2 interactions (sulfonamide `−0.10`, aryl-ether `−0.67` in the
calipered comparison). These are fixed-weight decompositions of an input change, not retraining
effects.

**Q5 — how far the real-label side evidence reaches.** On the reused 823 MMP pairs the frozen model's
`Δf` is closer to `Δy` than the zero-change baseline in the support-limited groups that pass the
pre-defined filter: amide Train–Train clean (23 pairs / 13 `eval_component` clusters) MAE
`0.393` vs `0.969`, sign agreement `0.78`; aryl-ether `0.273` vs `0.486` (53/30); arylamine `0.319`
vs `0.456` (23/16); sulfonamide `0.275` vs `0.436` (16/7, `LOW_SUPPORT`). This is a
direction-and-magnitude consistency statement **in this sample** with component-level uncertainty; it
is not a claim about new-molecule accuracy, and it cannot rank families.

---

## 1. What was held fixed

Data, split, row order, feature spec, Train-only normalisation, target mean/scale and the checkpoint
are byte-identical to the executed pilot round. Stage B re-verified on the GPU: frozen-input SHA256
gate pass, O-view Test RMSE `0.57616745` (`Δ = −2.6e−8`), Valid `0.70253589` (`Δ = −4.7e−9`), maximum
per-molecule prediction difference against the previous round `1.5e−6` (tolerance `1e−5`), 0
structural invariant failures, 0 AF51 failures (max atom-feature difference exactly `0.0`), all O/T/K0
single-component.

Every T/K0/C view was re-verified from scratch in Stage C: the dose catalogue was recomputed from
rebuilt graphs for all 539 accepted caliper links and matches bit-for-bit
(`max_abs_diff_vs_catalogue = 0.0`, 0 d1n/d1b/d2k mismatches, 0 ratio re-check failures, 0 bond-count
deviations in 302 Valid + 237 Test links).

## 2. Six-cell applicability × effect matrix (Test, G_FIXED)

Main metric `rC = (aT − aC)/(aT + aC)` where `aT = |pT−p0|`, `aC = mean_C |pC−p0|`; `E` compares the
two parameter states. Raw logD. One shared scaffold-cluster bootstrap per split (5000 replicates,
seed 20261114); 6 families × 2 quantities = 12 items, Bonferroni quantiles `0.00208 / 0.99792`.

| family | n mol / pairs (calibrated) | rC_TRAINED [95 %] | rC_INIT | E [95 %] | Bonf. rC | Bonf. E | verdict |
|---|---|---|---|---|---|---|---|
| amide | 62 / 110 | **+0.164** [0.052, 0.278] | −0.157 | **+0.321** [0.159, 0.479] | [0.005, 0.334] | [0.089, 0.556] | **supported** |
| sulfonamide | 13 / 29 | +0.327 [−0.008, 0.617] | −0.269 | +0.596 [0.257, 1.006] | [−0.207, 0.725] | [0.072, 1.151] | partial (`E` clear, `rC` borderline) |
| arylamine | 4 / 4 | +0.127 [0.106, 0.189] | +0.505 | −0.378 [−0.479, −0.075] | [0.106, 0.189] | [−0.479, −0.075] | positive but **not training-related** |
| aryl_ether | 8 / 10 | −0.433 [−0.872, −0.158] | −0.198 | −0.235 [−1.035, 0.285] | [−0.998, −0.106] | covers 0 | **reversed** |
| ester | 4 / 7 | −0.187 [−0.408, 0.065] | +0.042 | −0.229 [−1.083, 0.589] | covers 0 | covers 0 | reversed, `LOW_SUPPORT` |
| sulfonate | 0 / 0 | NA | NA | NA | NA | NA | **not excluded** (no C) |

Raw companions on the same queue: `J_C` = +0.146 / +0.608 / +0.185 / −0.485 / −0.109 (amide,
sulfonamide, arylamine, aryl-ether, ester); `J_K` = +0.273 / +0.636 / −0.411 / −0.098 / −0.030;
`D_C` = +0.173 / +0.206 / −0.419 / −0.162 / +0.021; `B` = +0.492 / +0.249 / +0.168 / +0.298 /
+0.101. The `K0`-only statistic `rK_TRAINED` = +0.279 / +0.463 / +0.071 / −0.020 / −0.012 and
`rK_INIT` = +0.044 / +0.323 / +0.788 / +0.048 / −0.045 — reported, never substituted for `rC`.

Scope robustness: the FULL-scope counterparts agree (`rC_FULL` = +0.152 / +0.321 / +0.137 / −0.434 /
−0.186), so the conclusion is not an artefact of freezing global42 at O's value.

**Valid is a direction check, not padding.** It agrees with Test for amide (`rC = +0.114
[0.007, 0.214]`, `E = +0.111 [−0.020, 0.230]`) and sulfonamide (`+0.357 [0.047, 0.574]`,
`E = +0.463 [0.035, 0.789]`) and disagrees for the tiny families: ester Valid `rC = +0.576` vs Test
`−0.187`, and arylamine Valid `E = +0.526` vs Test `−0.378`. The sign of the response for
ester/arylamine/aryl-ether is therefore **not stable across splits**, and we do not claim one.

Support, `E`-influence robustness (drop the worst-|J_C| molecule, then `ceil(5 % n)`):
amide `rC` +0.130 … +0.157 and `E` +0.274 … +0.308, sulfonamide +0.293 / +0.529, aryl-ether
−0.365 / −0.051, arylamine +0.129 / −0.358, ester −0.154 / −0.240. No sign flips except
aryl-ether's `E`, which is already uninformative (CI covers 0).

Scaffold strata (structure-only): amide Test `rC` = 0.139 seen-in-Train (92 mol) vs 0.201 unseen
(72 mol) — same sign, so the effect is not a memorised-scaffold artefact. Sulfonamide
0.278 / 0.437, aryl-ether −0.386 / −0.767, arylamine 0.127 / n/a.

![applicability](fig_applicability_matrix.png)
![parameter dependence](fig_parameter_dependence.png)

## 3. Internal path (TRAINED only, global42 = O)

Three groups — `G1` first-order aggregation (`b_sum`, typed neighbour, `s1_joint`), `G2` all S2 blocks,
`G3` atom scalars — injected through a read-only `LocalMLP` pre-hook over all 8 subsets; AF51 always
from O. Endpoint identities: `S=∅` reproduces O to `1.5e−6`, `S=123` reproduces the real G_FIXED
prediction to `7.2e−7`, and the seven Möbius terms sum back to the full-input change to `1.0e−6`
(eight identities re-checked in Stage C).

Seven-term decomposition, Test, `T − C` (dose-calipered) and `T − K0` (all pairs):

| family | comparison | S1 | S2 | scalar | S1×S2 | S1×S | S2×S | S1×S2×S | Σ |
|---|---|---|---|---|---|---|---|---|---|
| amide | T−C | +0.023 | **+0.104** | −0.002 | −0.011 | −0.011 | +0.009 | +0.035 | +0.146 |
| amide | T−K0 | **+0.126** | +0.033 | −0.006 | −0.001 | −0.010 | +0.037 | +0.008 | +0.187 |
| sulfonamide | T−C | +0.166 | **+0.506** | −0.006 | −0.100 | −0.004 | +0.049 | −0.002 | +0.608 |
| sulfonamide | T−K0 | +0.161 | **+0.688** | +0.029 | −0.105 | +0.023 | +0.006 | −0.078 | +0.723 |
| aryl_ether | T−C | −0.013 | +0.280 | −0.012 | **−0.668** | −0.003 | +0.008 | −0.077 | −0.485 |
| arylamine | T−C | +0.009 | **−1.082** | +0.023 | **+1.164** | −0.018 | −0.066 | +0.155 | +0.185 |
| ester | T−C | +0.014 | −0.119 | −0.016 | −0.020 | −0.003 | +0.016 | +0.020 | −0.109 |

Reading: the scalar block is inert everywhere; when the dose is matched, the response is carried by
the S2 block, with the pair interactions (S1×S2) large and often of the opposite sign — the
arylamine and aryl-ether rows are dominated by an S2/S1×S2 cancellation, which is why their totals
are small and their signs unstable. The amide family is the only one where the *all-pairs* contrast
is S1-dominant. Loss-space (`L`) terms follow the same shape (see `mechanism_matrix.md`).

![mechanism](fig_mechanism_matrix.png)

## 4. Real-label side evidence (reused 823 MMP pairs, no new mining)

`Δf` from the frozen model on the **original** molecules, `Δy` experimental; grouped by split
combination, MAE aggregated by `eval_component` (not by pair count).

| family | group | subset | pairs / components | MAE `|Δf−Δy|` | zero-change `|Δy|` | sign agr. | support |
|---|---|---|---|---|---|---|---|
| amide | Train–Train | clean | 23 / 13 | 0.393 | 0.969 | 0.78 | SUPPORTED |
| amide | has-Test+Train endpoint | clean | 12 / 5 | 0.244 | 0.581 | 0.92 | LOW_SUPPORT |
| sulfonamide | Train–Train | clean | 16 / 7 | 0.275 | 0.436 | 0.94 | LOW_SUPPORT |
| arylamine | Train–Train | clean | 23 / 16 | 0.319 | 0.456 | 0.70 | SUPPORTED |
| aryl_ether | Train–Train | clean | 53 / 30 | 0.273 | 0.486 | 0.68 | SUPPORTED |
| aryl_ether | has-Test+Train endpoint | clean | 23 / 12 | 0.211 | 0.633 | 0.91 | SUPPORTED |
| ester | Train–Train | clean | 5 / 4 | 0.384 | 1.046 | 0.80 | LOW_SUPPORT |
| sulfonate | Train–Train | clean | 1 / 1 | n/a | n/a | n/a | LOW_SUPPORT |

The full table with the Train same-`class_id` baseline and its coverage is in `mmp_table.md`. The
evidence reaches "the frozen model's predicted change is directionally right and closer to the
measured change than assuming no change, in a few support-limited groups". No significance claim.

## 5. Cases

For every non-empty family one median-|J_C| molecule and one counterexample (strongest
opposite-direction molecule), drawn from the proven O/T/K0 raw graphs with the SMILES→raw mapping
re-verified before drawing (`analysis/cases/`, index in `analysis/cases/cases_manifest.json`).
Ranking uses `J_C` on the calibration queue, `G_FIXED`.

## 6. Limitations and honest caveats

1. **Calibration coverage is a structure-only property and is low** (9–26 % of pairs, 14–50 % of
   molecules). Ester and the two aromatic families therefore rest on 4–8 Test molecules; their
   intervals are wide by construction. They are labelled `LOW_SUPPORT`, not dropped.
2. **`sulfonate` is empty** in Valid/Test; the competing explanation is not excluded for it, and we do
   not treat the empty cell as a negative result.
3. **Aromatic coverage differs from the pre-registered counts** because the frozen donor rule now
   allows aromatic carbons (`PROTOCOL.md` §3.3, recorded in the manifest before any effect was
   observed). No definition was changed after looking at an effect.
4. Two of the four positive-cell results (`arylamine`, and `aryl_ether`/`ester` in the reversed
   direction) do not reproduce their sign on Valid. Only amide (both quantities) and sulfonamide
   (`E` only) hold their sign across splits.
5. `rC` is defined only where a calipered C exists; a "main queue" `rC` would be the same quantity on
   a smaller sample, so the tables show `—` for it.
6. The amide number here is **not** numerically comparable with the pilot's `J = 0.3206`: the pilot
   used the legacy class-2-only donor rule and FULL as primary scope. This round's amide
   `J_K` (all pairs) is +0.187 and `J_C` (calipered) +0.146.
7. `D`/`B` use the original molecule's label and are not prediction errors of the rewired graphs.
8. Per-dimension range support is not a joint in-distribution statement.
9. Everything is a fixed-weight, inference-only observation on one checkpoint and one initialisation;
   it cannot separate "this motif" from "any local pattern change" beyond what the dose caliper
   already does.

## 7. Provenance and reproducibility

| stage | where | artefact |
|---|---|---|
| A (structure, doses, caliper, freeze) | local CPU | `analysis/{freeze,inputs_manifest,prepare_checks}.json`, 6 catalogues, `PROTOCOL.md`, commit `0273cad6` |
| B (TRAINED + internal path) | `rr` res-gpu0 `lipo-multimotif-trained-s0` / `7e2640a6` | `per_graph_predictions.csv`, `per_graph_internal.csv`, `infer_checks_scopes_trained-internal_trained.json` |
| B (INIT) | `rr` res-gpu1 `lipo-multimotif-init-s0` / `5b8367a2` | `per_graph_predictions_init.csv`, `infer_checks_scopes_init.json` |
| C (statistics, pairs, figures) | local CPU | `results.csv`, `uncertainty.json`, `analysis_checks.json`, `mechanism*`, `mmp*`, `scaffold*`, `dose_table.csv`, figures, cases |

Every number above is reproducible from the committed catalogues plus the two committed per-graph
prediction files by `uv run python …/code/run.py --stages analysis,figures`. Tests:
`uv run pytest -q …/tests` (10 passed).
