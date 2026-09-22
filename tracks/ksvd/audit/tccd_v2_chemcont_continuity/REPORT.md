# TCCD-v2 CHEM-CONT — local discontinuity: objective-induced or representation-limited?

Pre-registration: `tracks/ksvd/notes/tccd_v2_chemcont_preregistration.md`.
New-arm code: `tracks/ksvd/code/tccd_v2_chemcont.py`,
`tracks/ksvd/code/run_tccd_v2_chemcont.py`, tests
`tracks/ksvd/tests/test_tccd_v2_chemcont.py`.
Raw audit: `tccd_v2_chemcont_continuity_audit.json`; comparison:
`BASE_vs_CHEMCONT.json`. Frozen BASE audit is untouched at
`tracks/ksvd/audit/tccd_v2_continuity/`.

One arm, seed 0, one λ_chem, no sweeps. Official test never loaded.

---

## 1. Executive conclusion

**The Exact→VeryNear local discontinuity is primarily objective-induced, not
representation-limited — but the specific radius-1 continuity objective does not
produce a smooth chemical manifold; it relocates and sharpens the boundary while
costing task MAE.**

Evidence for objective-induced:

* A single label-free term on `z` alone moved the VeryNear Z cosine from
  `0.428` to `0.734` (+0.307, non-overlapping molecule-cluster CIs) and halved
  the Exact→VeryNear cliff (`0.572 → 0.266`). The frozen `714 → 64` linear
  encoder is therefore **not** incapable of encoding VeryNear continuity; the
  frozen TCCD-v2 geometry simply never asked for it. **Outcome C (representation
  limit) is ruled out for this cliff.**

Evidence that the fix is not a chemical manifold:

* The gain is a **contrastive re-partition**, not smoothing. The VeryNear→Moderate
  boundary became nearly deterministic in Z (AUC `0.588 → 0.909`) and
  Moderate→HardNegative `0.645 → 0.895`, while HardNegative Z cosine collapsed
  from `+0.206` to `−0.338` — chemically different matched negatives are now
  *less* similar than random pairs (Random stays `+0.050`). The discontinuity
  moved from the exact boundary to the radius-1 boundary.
* The Z change propagates only weakly to the prototype code `C`: VeryNear C
  cosine `0.301 → 0.383` (+0.082) versus Z +0.307, and VeryNear same-top-1
  prototype only `0.090 → 0.137`. The Z/C decoupling is real (a B-type signal).
* The guardrail moved: internal-dev best MAE `0.28624 → 0.30206`
  (+0.01582, above the repo's 0.010 materiality floor), Top-5 soup
  `0.26635 → 0.27527` (+0.00892).

**Primary classification: D** — the radius-1 chemical-similarity notion is too
coarse / conflicts with the distinctions the task must preserve, so extra
continuity pressure is spent on a definition-shaped margin rather than a
chemical manifold. There is a strong **B component** (prototype coding does not
follow the Z geometry). The core success criterion of the round is met: the
causal fork is resolved toward *objective*, not *representation*.

---

## 2. Frozen baseline

**BASE NOT RERUN.** All BASE numbers are read from existing artifacts.

| item | value |
|---|---|
| checkpoint | `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt` |
| checkpoint sha256 | `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42` |
| formal Gate A commit | `69a985a` (`results/tccd_v2/gateA_seed0.json`) |
| internal-dev best MAE | `0.2862437069416046` |
| internal-dev Top-5 soup | `0.26635152101516724` |
| learned temperature | `0.094572` |
| vocabulary | 64/64 active, effective `62.5755` |
| continuity audit | `audit/tccd_v2_continuity/tccd_v2_continuity_audit.json` (commit `27937d9`) |
| claims / decisions | `claim-tccd-v2-...-20260922.yaml`, `decision-tccd-v2-stop-absolute-gap-20260922.yaml` |

The audit was not re-run for BASE. The old BASE audit JSON was not modified.

---

## 3. CHEM-CONT intervention

Run: `chemcont-s0`, GPU1, commit `1f18c7466187d7915d6ad463a9bc59959cb4e9ea`,
seed 0, wall 1338 s, early stop epoch 189, best epoch 149.
Checkpoint: `tracks/ksvd/results/tccd_v2_chemcont/prototype_chemcont_seed0_best.pt`
(sha256 `781efa0cd674aaf70ca2398eb276ec6c51cf1209661c66b3247119f6683d1fcb`).

Identical to frozen TCCD-v2 Prototype-REL: same records, split, `714→64` linear
encoder init, `K=64` cosine prototypes, temperature parameterization, `BAG + CᵀRC`
reader, Adam/batch/lr/wd/clip, stopping, seed. **Only added**:

    L_chem = mean softplus(cos(z_i, z_neg) − cos(z_i, z_pos))

λ_chem was calibrated once on the first 32 training graphs, detached, and frozen:

| quantity | value |
|---|---:|
| initial task loss | `1.5957586` (identical to frozen BASE calibration `1.5957586`) |
| initial chem loss | `0.6901239` |
| λ_chem | `0.1156139` |
| initial chem contribution | `0.0797879` = 5.00% of task |
| calibration triplets | 2473 |

The local/balance regularizers reproduce the frozen BASE values exactly.
Run `gate0` passed on GPU1 (pair-tier sanity, no dev pair, label-free pair
construction under scrambled `y`, non-zero `L_chem` gradient to `W`, no gradient
to prototypes, `λ_chem=0` forward/task identity, permutation invariance).

---

## 4. Pair construction (training split only)

Deterministic, cross-molecule, no `y`, cached once (`PAIR_SEED=20260922`, cache
pulled to `results/tccd_v2_chemcont/cache/`).

* positive `j` = **VeryNear**: identical radius-1 key, different canonical
  radius-2 key, different molecule;
* negative `k` = **HardNegative**: same root type + degree, `|Δn_atoms| ≤ 1`,
  `c1 ≥ 2` radius-1 edits, different molecule.

| statistic | value |
|---|---:|
| training graphs | 8000 |
| patches / candidate anchors | 185538 |
| anchors used | 161718 |
| triplets (pos = neg = 604683) | **604683** |
| unmatched anchors | 23820 (12.84%) |
| anchors without positive | 480 |
| anchors without negative | 23340 |
| max pairs per anchor | 4 |
| graph coverage | 1.000 |
| positive L1-equal / key-diff rate | 1.000 / 1.000 |
| negative root+degree / size-match / c1≥2 rate | 1.000 / 1.000 / 1.000 |
| negative c1 histogram | `{2: 530262, 3: 72592, 4: 1829}` |
| cross-molecule rate | 1.000 |
| target label used | **False** |

---

## 5. Geometry result

Frozen BASE vs CHEM-CONT, internal-dev patches, same audit code (means; brackets
are 95% molecule-cluster bootstrap CIs).

| metric | BASE existing | CHEM-CONT | delta |
|---|---:|---:|---:|
| VeryNear Z cosine | 0.4278 [0.423, 0.433] | **0.7344 [0.730, 0.739]** | **+0.3066** |
| Moderate Z cosine | 0.3319 [0.327, 0.338] | 0.2616 [0.256, 0.269] | −0.0703 |
| HardNegative Z cosine | 0.2059 [0.202, 0.210] | **−0.3381 [−0.345, −0.330]** | **−0.5439** |
| Random Z cosine | 0.0664 | 0.0504 | −0.0160 |
| Z graded Spearman excl. Exact | 0.4058 [0.401, 0.411] | **0.6140 [0.607, 0.620]** | **+0.2081** |
| VeryNear C cosine | 0.3012 [0.296, 0.308] | 0.3831 [0.378, 0.390] | +0.0819 |
| Moderate C cosine | 0.2177 | 0.1747 | −0.0429 |
| HardNegative C cosine | 0.0948 | 0.0383 | −0.0564 |
| C graded Spearman excl. Exact | 0.3714 [0.365, 0.377] | 0.5508 [0.546, 0.556] | +0.1794 |
| VeryNear same top-1 prototype | 0.0897 | 0.1368 | +0.0471 |

Exact→VeryNear cliff (`1 − VeryNear cosine`):

| space | BASE | CHEM-CONT | change |
|---|---:|---:|---:|
| Z | 0.5722 | 0.2656 | **−0.3066** |
| C | 0.6988 | 0.6169 | −0.0819 |

**The cliff shrank in Z much more than in C (3.7×).**

### The "improvement" is a relocated, sharper boundary

Adjacent-tier separation, AUC `P(higher tier more similar)` (0.5 = chance):

| comparison | BASE | CHEM-CONT | delta |
|---|---:|---:|---:|
| Exact > VeryNear | 1.000 | 1.000 | 0.000 |
| VeryNear > Moderate | 0.588 | **0.909** | +0.321 |
| Moderate > HardNegative | 0.645 | **0.895** | +0.250 |
| HardNegative > Random | 0.669 | **0.209** | −0.460 |

The continuous radius-1 axis became *more* step-like, not smoother:

| radius-1 edits | BASE Z | CHEM-CONT Z | BASE C | CHEM-CONT C |
|---|---:|---:|---:|---:|
| `c1=0` (L1 identical) | 0.427 | **0.734** | 0.300 | 0.382 |
| `c1=1` | 0.333 | 0.261 | 0.218 | 0.174 |
| `c1=2` | 0.204 | **−0.297** | 0.098 | 0.043 |
| `c1≥3` | 0.137 | **−0.372** | 0.075 | 0.030 |

Within-family Spearman(`c1`, Z cosine) went from `−0.298` to `−0.807`. The
regularizer learned the positive definition exactly and pushed everything
outside it apart (HardNegative and even Random-adjacent behaviour), rather than
interpolating chemistry. The element-level (heavy-atom) control shows the same
pattern (`0.666 / 0.161 / −0.250 / −0.110`), so it is not an H/charge artifact.

### Z vs C

The encoder reorganized strongly, the code barely followed: Z VeryNear +0.307
vs C VeryNear +0.082; VeryNear top-1 agreement is still only 0.137. Prototype
coherence barely moved (mean within-prototype L1-equality `0.387 → 0.444`, mean
within-prototype radius-1 edits `0.812 → 0.770`; key concentration `0.266 →
0.259`). Usage stayed healthy (64/64 active, effective `62.39`, τ `0.0906`).

---

## 6. Prediction guardrail

Internal-dev only; official test and official valid never loaded.

| metric | frozen BASE | CHEM-CONT | delta |
|---|---:|---:|---:|
| best-checkpoint MAE | 0.286244 | 0.302059 | **+0.015815** |
| Top-5 soup MAE | 0.266352 | 0.275267 | +0.008915 |

Best-checkpoint degradation is above the repo's 0.010 materiality floor; soup is
just below it. λ_chem was **not** retuned after seeing this (no sweep).

---

## 7. Outcome

**Primary: D** — Z continuity clearly improves (VeryNear Z cosine, graded Z
Spearman, Exact→VeryNear cliff all move together with non-overlapping CIs), but
task MAE degrades materially at best-checkpoint, and the mechanism is exactly the
pre-registered D mechanism: the radius-1-based similarity notion is too coarse /
conflicts with the task, so the geometry is reshaped into a sharper contrastive
partition (HardNegative pushed to negative cosine, VeryNear→Moderate AUC 0.91)
instead of a smooth chemical manifold.

**Secondary: B-like component** — even though Z moved a lot, C-space improved
only modestly and top-1 agreement stayed low, so prototype coding is an
additional downstream bottleneck.

**C is ruled out** for the Exact→VeryNear cliff: the frozen `714→64` linear
encoder is demonstrably able to encode VeryNear continuity when asked.

Not INCONCLUSIVE: the Z effect sizes are far outside the bootstrap CIs and all
pre-registered "improves" conditions are met; the only judgement call is the
moderate MAE delta (best material, soup borderline), reported above.

---

## 8. One next step

**Re-specify the local chemical similarity before touching the encoder or the
dictionary: replace the binary radius-1-identical positive / `c1≥2` negative
contrast with a continuous, graded chemical-dissimilarity target (e.g. a monotone
kernel over radius-1 and radius-2 edit counts, exact pairs excluded), and test
one arm with the same frozen architecture.** This directly targets the failure
observed here — the boundary relocated to the radius-1 definition instead of
smoothing — without opening a new encoder, reader, K/λ sweep, or the official
test.
