# RUN LEDGER — zinc_dcc_information_audit_v1

Frozen base commit: `58e6837b91507aae3feb9ad9370ef75d1959b7ec` (`main`, 2026-10-26).
All paths below are relative to `tracks/structural-sufficiency/zinc_dcc_information_audit_v1/`.
Official test was never constructed, loaded, or read. Official valid was never
loaded or used for selection/training/gating.

## Frozen inputs (written before results)

| artifact | sha256 |
|---|---|
| `AUDIT_PLAN.md` | `ddc6978121bdb91588876a16b6e5f13eee95173cce6b6b9100de0e9ab0dd7f22` |
| `configs/audit_v1.json` | `0c0560cd5d55408647c5f223b405b165570dcf9ad41945cf0eaba8c24180da8d` |
| `configs/b_phase_freeze.json` | `e5a45abdbe494ce8b391f0692fa95268579d79f248d2307f9d61f6c96e610bc9` |
| `configs/split_fit8k_probe_audit.json` | `d39f1d323a1a4ec97c402c1f0797cd8a16ce506f67eafe9ac9163a0b18073297` |

Split role hashes (row ids, int64 little-endian):

- Fit8k: `b2666522b589ab118bc52779b74df1100f169d00844ec192cd493b1a84f7ff42`
- ProbeFit1k: `67644e905d7dcbc637d19e7e6c7f7cc91ed4ae51399a267ec3a1a2a8a6bb5892`
- AuditEval1k: `f2b7f5592cb0cac901554e1a5160c69fe1b58044517077e680ff6597f6410bc4`

The split is structure-only. Three exact isomorphic duplicate pairs (6 rows)
were verified by `nx.is_isomorphic`; all were assigned to Fit8k, so no duplicate
spans roles. No non-isomorphic WL collisions.

## Access discipline

- Data loaded: official ZINC **train only** (`ad_train.pt`), train P3 objects,
  train pretrain encoders, DCC-L checkpoints. `ad_valid.pt` and `ad_test.pt`
  were only ever wrapped/refused by the load guard; they were never loaded.
- `analysis/provenance.json` and `analysis/label_access_audit.json` record the
  loaded paths. A1 never touched `y`.
- A2/A3/A4 use train labels only in B3/B4 where explicitly declared. A4 is
  label-free. B3 guards AuditEval labels; B4 reads them once after prediction
  hashes are frozen.

## Checkpoints

| id | path | sha256 | note |
|---|---|---|---|
| D0 | `tracks/.../zinc_dual_channel_composer_v1/runs/L_s0/best.pt` | `fab04ec0d5fd56f338379d7dcb223e31208338a8a89a335a25a6f0c4f5226cd5` | full-train DCC-L, epoch 489, valid 0.078537 (background only) |
| D1 | `zinc_performance_search_v1/runs/B_R0_s1/best.pt` | `c444f371fe5b5ec5eeb33f984784016b9b87c8af4648e6e733727eb47d34ef1d` | full-train B-R0 s1, epoch 424, valid 0.0738369 (background only) |
| D2 | `zinc_performance_search_v1/runs/B_R0_s2/best.pt` | `a54897620bcc33163949ce23719cadeab1be4e854bde45e9f5d3d0f08a803019` | robustness only |
| SSL s0 | `structure-contribution/.../pretrain_encoder_s0.pt` | `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97` | historical full-10k SSL, background/fallback |
| SSL s1 | `structure-contribution/.../pretrain_encoder_s1.pt` | `ff8d6474896c274407a1433658b95c79561fbc9a1deb1f9885bd862b4e8b7a1a` | historical full-10k SSL, background/fallback |

## CPU phase (A1–A4): 0 GPU·h

Commands were run with `uv run python code/<script>.py`. All CPU; no `rr` job.

### A1 forward audit

`code/run_forward_audit.py` → `analysis/forward_audit.json`,
`analysis/nondegeneracy.json`, `analysis/provenance.json`.

- 1,000 official-train molecules, D0 and D1, CPU.
- Reader parity `max |capture - model.forward| = 0.0` for both checkpoints.
- Independent grouping recomputation `Wg/Vg/Wl/Vl/n_b` max diff `0.0`;
  recomputed `c_glob`/`c_loc` max diff `0.0`.
- Slice layout: `z = [z_glob(0:96), a_sum(96:192), a_sq(192:288),
  g(288:321), zl(321:417), zl2(417:513)]`; reassembly diff `0.0`;
  `zl == sum(c_loc)` diff `0.0`; `zl2 == sum(c_loc^2)` diff `0.0`;
  `reader(z) == pred` diff `0.0`.
- Invariance on 3 molecules (17/25/33 atoms): atom relabel, bond permutation,
  endpoint reversal, batch regrouping all `<= 8.94e-08` (float32 noise).
- Non-degeneracy (pooled rows): D0 `a` PR 8.67, `b` 19.02, `w` 20.42,
  `c_glob` 4.32 (top1 0.399), `c_loc` **2.94** (top1 0.572, 16 zero
  eigenvalues), `z` 2.28 (top1 0.653, 19 zero eigenvalues). D1 `c_glob` 2.60,
  `c_loc` 3.34 (top1 0.521), `z` 2.07 (top1 0.689).
- Within-molecule `c_loc` PR median 2.94 (D0) / 3.43 (D1); pooled
  within-molecule PR 3.12/3.56. This is trained-weight low effective rank,
  not an initialization claim.

### A2 algebraic audit

`code/run_algebraic_audit.py` → `analysis/algebraic_counterexample.json`.

- A2.1 fixed 6-bond path line graph, two integer bond-state assignments
  `X=(u,v,-u,-v,w,-w)` and `Y=(u,-u,v,-v,w,-w)`.
  `sum` max diff `0.0`, `sumsq` max diff `0.0`, independent reference matches,
  reader output abs diff `0.0`. Correspondence-dependent R1 stat changes
  `0.7708`; adjacent-pair dot sum changes `370.0`.
  This is an activation-level unidentifiability certificate for the pooled
  `(sum, sumsq)` reader, not a real-molecule non-identifiability proof.
- A2.2 200 real train molecules (>=6 bonds), fixed multiset-preserving
  rebinding of bond states across positions, D0/D1. Prediction mean abs delta
  `3.05e-08 / 3.91e-08` (invariant); `zl` max diff `3.8e-06/7.6e-06`
  (multiset-preserving float noise); exact multiset reader delta `0.0`;
  `T_R1` mean abs delta `0.0101/0.0125`, changed on 100% of molecules.
  `c_loc` within-molecule PR median `3.20/3.60`.
  Conclusion: the pooled reader cannot see position correspondence, while
  J_latent changes on every tested molecule. This is not a legal molecule
  reconstruction and not a label counterfactual.

### A3 joint-stat implementation and checks

`code/run_joint_stats.py` → `analysis/joint_stats_checks.json`,
`analysis/joint_stats_distribution.json`, `analysis/joint_stats_corr.json`,
`analysis/j_raw.npy`, `analysis/j_latent_D{0,1}.npy`.

- Primary `J_raw` is **32 dims**, not the plan's 48-dim 4-bond template: the
  actual train vocabulary is `{1,2,3}` only (counts 185,060 / 63,548 / 671;
  249,279 physical bonds; no aromatic type 4). Four-slot 48-dim is diagnostic
  only and never used in models/gates.
- `J_latent` is 4 dims (`T_R1..T_R4`), within-molecule centered `c_loc`,
  `/96`, empty relation → 0.
- Independent float64 exhaustive reference over 100 rows:
  `J_raw` max diff `0.0`; `J_latent` max diff `4.86e-17`.
- Graph relabel/order/endpoint/batch invariance: max diffs `<= 1e-15`.
- Edge cases: empty, single bond, tree path5, C6 ring, fused rings, two
  components. Relation counts as expected (e.g. C6 R1=6, R2=6, R3=3, R4=15).
- Relation pair counts over 10k: R1 mean 34.64 (0 empty), R2 43.93 (0 empty),
  R3 233.62 (0 empty), R4 46.15 (0.55% empty).
- `J_raw` has 11 near-constant / zero dims (9 exact constants plus
  `R3_freq_3_3` and `R4_empty`); they are listed and excluded before A4
  fitting/scoring. Stable `J_raw` dims = 21. `J_latent` has no constant dims.
- Correlations: `J_raw` vs `global33` max |r| 0.706; vs D0 `z` 0.084; vs D1
  `z` 0.165. `J_latent` vs `z` max |r| 0.096 (D0) / 0.238 (D1); vs `global33`
  0.187/0.223. Linear correlation is only a screen; A4 ridge is the main test.

### A4 label-free conditional redundancy (gate)

`code/run_redundancy.py` → `analysis/redundancy.json`.

- Split frozen before A4: Fit8k / ProbeFit / AuditEval.
- Ridge on Fit8k (5-fold internal CV, fixed alpha grid); scored on ProbeFit
  and AuditEval. MLP sensitivity: 513→128→out SiLU, AdamW 1e-3/wd 1e-4,
  batch 256, max 300 epochs, patience 20, 1,000-row internal holdout.
- Primary gate rows (standardized block R² on ProbeFit; near-constant dims
  excluded):

| target | seed | ProbeFit block R² | residual share |
|---|---:|---:|---:|
| J_raw | D0 | 0.9410 | 0.0590 |
| J_raw | D1 | 0.9323 | 0.0677 |
| J_latent | D0 | -0.1896 | 1.1896 |
| J_latent | D1 | 0.5805 | 0.4195 |

- AuditEval block R²: J_raw 0.934/0.933; J_latent 0.774/0.762.
- The J_latent_D0 ProbeFit R² is dominated by a single ProbeFit molecule
  (row 3776; its T_R2 is far outside the Fit8k range). This is a split
  outlier, not a passed gate. AuditEval J_latent is 0.774/0.762; the MLP
  sensitivity is 0.239/0.740. The J_raw MLP sensitivity is **0.9765** on
  ProbeFit, close to the 0.98 stop line: J_raw is almost reconstructible
  nonlinearly even though the primary Ridge gate is 0.94. The gate is defined
  on the primary Ridge, so it does not stop; the nonredundant signal is mainly
  J_latent.
- **Gate outcome: `STOP_BEFORE_NEW_GPU = false`.** Not all candidate blocks
  are reconstructible at R² ≥ 0.98. J_raw is close but not sufficient
  (R4 ring-block frequencies ~0.90; R2/R3 mixed-pair frequencies 0.56–0.89).
  J_latent is clearly not reconstructible by the frozen probes. This is an
  engineering screen, not a proof of information-theoretic absence.
- First A4 implementation had a gate-key `KeyError`; it was fixed and the
  script rerun. No erroneous result was used. The rerun changed only the
  near-constant-dimension exclusion, documented above; gate outcome unchanged.

## B phase (conditional; GPU on `res`)

Frozen in `configs/b_phase_freeze.json`. Planned 1.8 A100·h, soft cap 4.0,
hard cap 6.0. All jobs via `rr run res ... --pool res-gpu{0,1}`. Occupancy
and actual GPU·h are recorded here after each job.

Planned:
1. Fit8k-only masked SSL, seeds 0/1, 7,560 updates, batch 128 (parallel on
   two `res` GPUs).
2. Native DCC-L, seeds 0/1, exactly 47,400 updates, final-update checkpoint
   (parallel on two `res` GPUs).
3. B3 freeze per seed: z/J extraction, J_hat Fit8k-only, ProbeFit Z/ZJ/ZN
   heads, AuditEval prediction hashes.
4. B4 one-time AuditEval label read + scoring.
5. Decision + report.

GPU ledger (actual):

| job | pool | run id | wall | GPU·h actual | status | notes |
|---|---|---|---:|---:|---|---|
| SSL Fit8k s0 | res-gpu0 | `ssl-fit8k-s0-20261026-095838-3c7f4b47` | 1561.6 s | 0.434 | completed, exit 0 | encoder sha `388452ce…`; 7,560 updates |
| SSL Fit8k s1 | res-gpu1 | `ssl-fit8k-s1-20261026-095853-771d0b1a` | 1607.9 s | 0.447 | completed, exit 0 | encoder sha `a6a45387…`; 7,560 updates |
| DCC-L Fit8k s0 | res-gpu0 | `dccl-fit8k-s0-20261026-102722-89c254ae` | 1195.1 s | 0.332 | completed, exit 0 | 47,400 updates / 753 epochs; sha `b7b92659…` |
| DCC-L Fit8k s1 | res-gpu1 | `dccl-fit8k-s1-20261026-102735-c419e285` | 1343.2 s | 0.373 | completed, exit 0 | 47,400 updates / 753 epochs; sha `ff660cc1…` |
| B3 heads s0 | res-gpu0 | `b3-fit8k-s0-20261026-105228-a97f6e27` | 27.13 s | 0.0075 | completed, exit 0 | freeze + audit-label guard |
| B3 heads s1 | res-gpu1 | `b3-fit8k-s1-20261026-105241-b190faf2` | 28.34 s | 0.0079 | completed, exit 0 | freeze + audit-label guard |

Actual B GPU total: **1.601 A100·h** (0.434 + 0.447 + 0.332 + 0.373
+ 0.0075 + 0.0079), under the 4.0 soft cap. B4 was CPU-only. No GPU job
failed or was cancelled; no GPU budget was spent on a failed run.

### B2: Fit8k-only SSL and native DCC-L

- SSL used the `bjoc_masked_pretraining_v1` recipe verbatim, seeds 0/1,
  7,560 updates, batch 128, Fit8k only. Both jobs completed exit 0.
- DCC-L used the native P3/DCC-L builder, 494,193 parameters, exactly 47,400
  updates, batch 128, raw-y L1, final-update checkpoint (no early stopping or
  selection). Both jobs completed exit 0; `train_fit_mae` 0.000415 / 0.000502
  is a Fit8k fit diagnostic, not a generalization metric.
- Per-seed pairing is disclosed: supervised seed 0 loads SSL seed 0; supervised
  seed 1 loads SSL seed 1. The historical full-10k SSL checkpoints were not
  used for B.
- `global33` mean/scale was recomputed from Fit8k only; root/block scalars are
  not consumed by the DCC-L/P3N collate and remain frozen-cache values.

### B3: frozen audit heads

`code/b3_audit_heads.py` → `analysis/b/s{0,1}/freeze.json` and head/prediction
arrays. Both seeds: one Fit8k J_hat ridge; ProbeFit Z/ZJ/ZN ridge with the
pre-fixed alpha grid; Huber sensitivity; radius-2 WL support strata.

- Both seeds: full support 316 / low support 392 / unseen 292 AuditEval rows.
- Both seeds: `label_guard.audit_labels_accessed = false`; 1,000 forbidden
  AuditEval rows, 1,000 unique ProbeFit rows accessed. AuditEval prediction
  hashes were written before B4.
- Main head alpha: seed 0 `100.0`; seed 1 `0.0001` (pre-fixed grid; boundary
  values, not tuned after seeing AuditEval). Huber `n_iter` hit 300 in seed 0,
  so Huber is a sensitivity check, not a converged main result.
- No AuditEval labels were present in the B3 process's label array.

### B4: one-time AuditEval scoring

`code/b4_score.py` → `analysis/b/s{0,1}/metrics.json`. Run from the repo root
because the freeze records repo-relative checkpoint paths. The first seed-0
attempt failed at the checkpoint hash check before any label read; it was fixed
by resolving repo-relative paths and rerun. The successful runs read AuditEval
labels exactly once each:

- seed 0: `2026-10-26T02:56:04Z`, 1,000 rows, rows sha `f2b7f559…`
- seed 1: `2026-10-26T02:56:20Z`, 1,000 rows, rows sha `f2b7f559…`

No official valid or test data was loaded (`official_valid_loaded=false`,
`official_test_loaded=false` in both metrics files). B4 did not retrain or
select on AuditEval.

Key B4 results (all 1,000 AuditEval molecules; MAE; paired bootstrap 10,000,
seed 20261026; fixed models, so intervals are sample uncertainty, not seed
uncertainty):

| arm | seed 0 | seed 1 | mean |
|---|---:|---:|---:|
| raw DCC-L | 0.07524 | 0.07423 | 0.07474 |
| Z (z only) | 0.26726 | 0.31983 | 0.29354 |
| ZJ (z + J) | 0.26357 | 0.31675 | 0.29016 |
| ZN (dim-matched null) | 0.26400 | 0.32027 | 0.29214 |
| Huber Z | 0.11427 | 0.12826 | 0.12126 |
| Huber ZJ | 0.11293 | 0.13336 | 0.12315 |
| Huber ZN | 0.11200 | 0.13398 | 0.12299 |

- `Gain_J = MAE(Z) - MAE(ZJ)`: **0.003688** seed 0, 95% CI
  `[0.001335, 0.005937]`; **0.003079** seed 1, 95% CI
  `[-0.005152, 0.011036]`. Both point estimates are above the frozen 0.003
  practical band, but seed 1 is not interval-positive.
- `Gain_vs_N = MAE(ZN) - MAE(ZJ)`: **0.000430** seed 0 (CI includes 0),
  **0.003521** seed 1 (CI includes 0). The dim-matched null captures most of
  the seed-0 gain.
- Top-10 gain-row removal: seed 0 `+0.002212` (still positive); seed 1
  `-0.001681` (sign reversal). Seed 1's top 10 rows contribute 0.004744,
  more than the full 0.003079 gain.
- Huber sensitivity: `Gain_huber_J` **+0.001332** seed 0, **-0.005105**
  seed 1; mean **-0.001887**. `Gain_huber_vs_N` **-0.000931 / +0.000613**.
  The robust check does not reproduce a stable gain.
- Correlation of Z vs ZJ residuals: Pearson 0.996 (seed 0) / 0.960 (seed 1);
  ZN vs ZJ residuals 0.987 / 0.959. J changes predictions only marginally.
- The Ridge heads are badly miscalibrated: they are worse than raw DCC-L on
  AuditEval. Two ProbeFit molecules (rows 2210 and 3776; labels -42.04 and
  -20.26) account for **96.96%** of ProbeFit residual SSE and 33.5% of
  ProbeFit MAE; ProbeFit residual std falls from ~1.01 to ~0.176 without them
  (seed 0; seed 1 nearly identical). The heads fit those two extremes
  (ProbeFit correction min -24.5 / -29.1) and extrapolate to AuditEval
  (correction min -6.6 / -5.8). See
  `analysis/b/probe_outlier_diagnostic.json`.
- Support strata (full 316 / low 392 / unseen 292): seed-0 `Gain_J`
  +0.0045 / +0.0037 / +0.0027; seed-1 +0.0067 / -0.0066 / +0.0121.
  Intervals mostly include 0; no monotone low-support/unseen pattern.
- Tail `y <= -3` (n=68): seed-0 +0.0154 (CI includes 0), seed-1 -0.0198
  (CI includes 0). No stable tail advantage.
- Ring-rich (>=4 rings, n=67) exploratory: +0.0076 seed 0 and +0.0294 seed 1,
  both CIs positive, but small n and the same miscalibrated heads; not used
  for the decision.

**Decision: class B `NONREDUNDANT_BUT_NOT_PREDICTIVE`**, tags
`AUDIT_HEAD_CALIBRATION_FAILURE`, `AUDIT_HEAD_OUTLIER_DOMINATED`,
`HUBER_SENSITIVITY_SIGN_DISAGREES`, `PROBEFIT_OUTLIER_SENSITIVE`,
`SINGLE_MOLECULE_DRIVEN`, `ZN_CONTROL_NEAR_GAIN`.
`decision.json` is the machine-readable record. The primary Ridge gain is
positive but not stable; the robust sensitivity and the dim-matched null do
not support an independent predictive increment. This is a conclusion under
the current representation/data/diagnostic function, not a proof that all
correspondence-aware relation modeling is useless.

### Deviations / disclosures

- Primary bond vocabulary is 3 types, not 4; `J_raw` is 32 dims. The 48-dim
  four-slot version is diagnostic only.
- A4 near-constant definition (zero fraction ≥ 0.99 or std ≤ 1e-3) was fixed
  before B and applied symmetrically to all blocks/seeds. It excludes 11
  `J_raw` dims; gate outcome is unchanged if they are included (0.923/0.916).
- The A2.2 rebinding is an information control, not a chemically valid
  molecule and not a label counterfactual.
- The full-10k historical SSL checkpoints were used only for A1–A4 reference
  backbone D0/D1. B uses new Fit8k-only SSL encoders and discloses the pairing.
