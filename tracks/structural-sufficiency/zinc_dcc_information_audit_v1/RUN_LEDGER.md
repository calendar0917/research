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
  outlier, not a passed gate. AuditEval and the MLP sensitivity (0.239/0.740)
  both remain far below 0.98.
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

GPU ledger (to be filled):

| job | pool | GPU·h actual | status | notes |
|---|---|---:|---|---|
| SSL Fit8k s0 | res-gpu0 | — | planned | |
| SSL Fit8k s1 | res-gpu1 | — | planned | |
| DCC-L Fit8k s0 | res-gpu0 | — | planned | |
| DCC-L Fit8k s1 | res-gpu1 | — | planned | |

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
