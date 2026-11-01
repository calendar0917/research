# LCRC-A on ESOL — short-cycle migration report (`lcrc_esol_quickstart_v1`)

Round id `lcrc-esol-quickstart-v1`. Everything below is produced by
`code/run_esol.py` and the artifacts it wrote into `analysis/`. No hand-edited
numbers.

---

## 1. One-paragraph verdict

The unchanged LCRC-A representation (explicit S1 neighbour statistics + explicit
S2 two-hop path statistics, per-atom local representation with additive
contribution readout, global `rho` correction) transfers from Lipophilicity to
water solubility without any structural change: mean Test RMSE **0.699 ± 0.026**
log(mol/L), MAE 0.517, R² 0.883, against a no-information (Train-mean) baseline
of 2.115. It therefore *does* learn ESOL and generalises across a scaffold
split. The information is in the local statistics and it is transferable, and it
is not the same failure mode as on Lipophilicity: the errors are not scattered,
they are **scaffold-clustered** (one 9-molecule test scaffold group carries 23%
of the Test squared error; five groups carry 50%), and the model fails exactly
where the whole-molecule skeleton is new (Test RMSE 0.80–0.91 on the 70 cyclic
molecules, of which even the 21 that contain a plain benzene ring stay at 0.91,
versus **0.34** on the 43 acyclic molecules). The single pre-registered
iteration (dropout 0.10 → 0.20) was tested and **refuted**: it does shrink the
Train/Validation gap (0.299 → 0.248) and moves the best epoch earlier in all
three seeds (172/157/159 → 149/129/147), but Validation and Test both get
slightly worse (paired ΔValidation = −0.001/+0.041/−0.005, mean **+0.012**
worse; ΔTest mean **+0.014** worse).

---

## 2. Data source, provenance, and audit

| item | value |
|---|---|
| file | `delaney-processed.csv` (DeepChem `deepchem.molnet.load_delaney`, ESOL/Delaney) |
| URL | `https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv` |
| SHA256 | `8c06a76f0c6487d29ab0f903e6a7a7139f189ab3c1178f159c8be8964602f189` |
| local copy | `analysis/source/delaney-processed.csv` (byte-identical, re-hashed on every read) |
| rows | 1,128 · 1,123 distinct SMILES · **5 duplicated SMILES kept, never dropped** |
| label | `measured log solubility in mols per litre`, unit log(mol/L) |
| label stats (all rows) | mean −3.0501, sd 2.0956, min −11.6, max 1.58, 0 missing |
| forbidden column | `ESOL predicted log solubility in mols per litre` — never parsed (asserted in P0: `forbidden_columns_parsed = []`) |
| label-free raw graph artifact | `analysis/source/esol_raw_v1.json.gz` (14,991 atoms, 15,428 bonds, 0 RDKit failures) |
| provenance manifest | `analysis/source/source_manifest.json` |

The 5 duplicate SMILES (rows 213/976, 233/276, 450/1019, 680/1069, 703/822) are
one literal duplicate label pair and four near-duplicates; they are kept because
the task fixes N = 1,128 and dropping rows would silently change the dataset.
They are handled at the split level instead (see §3).

### Vocabulary compatibility (the actual migration risk)

ESOL contains only C, O, N, Cl, S, F, Br, P, I → every atom lands inside the
frozen Lipophilicity `lipo_v1` 16-class vocabulary:

- unknown atoms **0**, unknown bonds **0**, `nbr_code_max` 68 < vocab 80,
  `path_cat_max` 343 < vocab 400;
- unused classes: `B`, `Si`, `UNK` (atom), `UNK` (bond);
- an explicit UNK probe (dimethyl selenide, tetramethyl tin → index 15; boric
  acid stays in-vocabulary as `B`) confirms the fallback path works.

So the ESOL models use a representation that is bit-identical in vocabulary and
code path to the Lipophilicity round — the only thing that changes is the data.

---

## 3. Split (fixed before any training)

Bemis–Murcko scaffold partition, `split_seed = 20261103`, fractions 0.8/0.1/0.1,
`analysis/split/split_esol_scaffold_seed_20261103.json`
(file sha256 prefix `47db06adfd4f`, payload `split_sha256 = 45d4388c874c…`,
per-split row-hash vectors recorded, `split_audit.json` PASS).

**train 902 / valid 113 / test 113**, 584 groups, 509 singletons.

Two facts that had to be decided *before* training and are recorded in the
manifest as evidence:

1. **Strict empty-scaffold grouping is unusable.** 317/1,128 molecules (28.1%)
   are acyclic, i.e. scaffold `""`. Treating `""` as one indivisible group puts
   all 317 into Validation and leaves **Test = 0**. Recorded under
   `strict_empty_scaffold_alternative` (`sizes = {train 811, valid 317, test 0}`).
2. **So an acyclic molecule is grouped by its SMILES**, not per row. A per-row
   singleton key would have let the 5 duplicated SMILES straddle splits — one of
   them (sorbitol, rows 233/276, labels 0.06 vs 1.09) would have landed in Train
   *and* Test. With the SMILES key, `duplicate_smiles.groups_crossing_splits = 0`.

Non-empty scaffolds are always indivisible; `scaffold_groups_cross_split = 0`.
Side effect worth stating: the 254-molecule benzene group sits entirely in Train
(`benzene_scaffold_rows_per_split = {train 254, valid 0, test 0}`) and the
acyclic block is spread 222/52/43.

---

## 4. P0 gates — `analysis/p0/p0_summary.json`, status **PASS**, 25/25 checks

| gate | result |
|---|---|
| S1/S2 enumeration vs the frozen independent reference (`cpu_gates.reference_enumeration`), all 1,128 molecules | 0 mismatches, 30,856 S1 codes / 42,198 S2 paths |
| cache alignment (atom/bond counts, order, ids = CSV row indices) | max abs diff 0 |
| vocabulary bounds / UNK fallback | 0 unknown atoms, probe OK |
| label isolation (`forbidden_columns_parsed`, target vector hash, finite, non-missing) | pass |
| split sizes + row hashes + no group straddles + duplicates | pass |
| Train-only normalisation (structural, target, physchem) re-derived from Train | pass, `valid_or_test_used = false` |
| single vs batch forward equality, dense backward with finite grads | pass |
| physchem enters only the `rho` branch (bitwise-identical shared init) | pass |
| physchem recomputation cross-checked against the CSV descriptor columns | max abs diff **0.0** for MolWt/TPSA/HBD/RotBonds (MolLogP, HBA have no CSV column) |

Edge cases: 7 molecules have **zero** S2 paths (e.g. `C=C`), max 55 atoms,
62 bonds, no multi-component SMILES, 59 molecules with a charged atom.

---

## 5. Frozen configuration

Frozen (never tuned) at `analysis/protocol_freeze.json` (`freeze_sha256`
`099eee4e24c5`, config `7e3dd936c8b9`, `test_labels_read_at_freeze_time = false`):
the entire recipe inherited from `lcrc_lipo_best_epoch_distance_fix_v1` —
AdamW peak 3e-4 → 1e-5 per-step cosine, betas (0.9, 0.999), eps 1e-8, wd 1e-3,
batch 128, **360 epochs**, grad clip 5.0, dropout at the final atom-level `h`,
MSE on the Train-standardised target, **single best-Validation-RMSE epoch**
checkpoint, shuffle offset 917000, seeds `[0,1,2]`, fp32.

Both arms use the same `torch.manual_seed(seed)` stream, so BASE and DROP20 share
their initialisation and batch order exactly (`batch_order_sha256` identical) —
the P2 comparison is paired within seed.

| model | what it is | dropout | parameters |
|---|---|---|---|
| `ESOL-BASE` | LCRC-A unchanged | 0.10 | 247,106 |
| `ESOL-DROP20` | identical, only `p` changed | 0.20 | 247,106 |

(`ESOL-PHYSCHEM6`, the 6-descriptor `rho` variant, is implemented and P0-gated
(247,490 params, +384 = 6×64) but **not trained** — see §7.)

---

## 6. P1 — `ESOL-BASE` (the migration question)

Per-seed (`analysis/eval/table_runs.csv`), Test read only once, after freeze:

| model | seed | best epoch | Train RMSE | Valid RMSE | Valid MAE | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ESOL-BASE | 0 | 172 | 0.4002 | 0.7205 | 0.5371 | 0.7284 | 0.5439 | 0.8733 |
| ESOL-BASE | 1 | 157 | 0.4316 | 0.7670 | 0.5685 | 0.6900 | 0.4899 | 0.8863 |
| ESOL-BASE | 2 | 159 | 0.4196 | 0.6612 | 0.4853 | 0.6786 | 0.5157 | 0.8900 |
| **mean ± sd** | | 163 ± 8 | **0.4171 ± 0.0158** | **0.7162 ± 0.0530** | 0.5303 ± 0.0420 | **0.6990 ± 0.0261** | 0.5165 ± 0.0270 | 0.8832 ± 0.0088 |

Reference points: Train-mean predictor on Test = **2.115** RMSE; Test label sd
2.055. So the model explains ≈ 88% of Test variance and cuts the error by 67%.
Runtime: 0.19–0.25 s/epoch, 70–90 s per run, single A100.

### Overtraining signature (this is what selected P2)

| seed | best epoch | Train @best | Valid @best | Train @360 | Valid @360 | Valid @best+25 | epochs after best | improving epochs after best |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 172 | 0.432 | 0.720 | 0.341 | 0.776 | 0.785 | 188 | **0** |
| 1 | 157 | 0.431 | 0.767 | 0.339 | 0.900 | 0.920 | 203 | **0** |
| 2 | 159 | 0.440 | 0.661 | 0.351 | 0.719 | 0.700 | 201 | **0** |

In 3/3 seeds the best epoch is at 43–48% of the fixed budget, **not one** of the
remaining 188–203 epochs improves Validation, and Train RMSE still falls another
0.089–0.092. Mean Train/Validation gap 0.299 on a target with Train sd 2.08.

---

## 7. P2 — one pre-registered iteration, and its refutation

Full pre-registration in `P2_HYPOTHESIS.md` (written and committed *before*
launching, commit `ea8bb0d1`): candidate **A**, dropout 0.10 → 0.20, justified by
the table above, with explicit refutation conditions. Candidate B
(`ESOL-PHYSCHEM6`) was **not** run: its trigger is "no clear overfitting
evidence", which is the opposite of what P1 showed, and the task allows exactly
one direction and at most 6 formal runs total.

| seed | BASE Valid | DROP20 Valid | ΔValid RMSE | BASE Test | DROP20 Test | ΔTest RMSE | ΔValid MAE | ΔTest MAE | best epoch shift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.7205 | 0.7194 | −0.0011 | 0.7284 | 0.7033 | −0.0251 | −0.0048 | −0.0136 | −23 |
| 1 | 0.7670 | 0.8082 | **+0.0412** | 0.6900 | 0.7302 | **+0.0401** | +0.0205 | +0.0500 | −28 |
| 2 | 0.6612 | 0.6559 | −0.0053 | 0.6786 | 0.7044 | +0.0258 | −0.0030 | +0.0216 | −12 |
| **mean** | **0.7162** | **0.7278** | **+0.0116** | **0.6990** | **0.7126** | **+0.0136** | +0.0043 | +0.0193 | −21 |

Diagnostics: Train RMSE 0.417 → **0.480**, Train/Validation gap 0.299 → **0.248**,
Train sd across seeds 0.016 → 0.058, best-Validation epoch earlier in 3/3.

**Verdict: refuted, by the criterion that was written down in advance.**
Regularisation mechanically did what it should (smaller gap, less capacity to
memorise) and the Validation/Test numbers nevertheless got worse, and the two
"wins" (−0.0011, −0.0053) are 20–50× smaller than the between-seed sd (0.053).
The secondary refutation condition — "best epoch moves earlier and Train RMSE
stays high while Validation also worsens ⇒ capacity/fitting-speed limited, not
overfitting limited" — is met exactly, in all three seeds.

Paired molecule bootstrap (10,000 resamples, seed 20261103) on ΔTest RMSE per
seed: [−0.075, +0.022], [−0.010, +0.095], [−0.003, +0.057] — all cross zero.
The only interval that excludes zero anywhere is ΔTest **MAE** at seed 1
([+0.011, +0.090]), and it says *worse*.

---

## 8. Test evaluation (one-shot, after `MANIFEST_FROZEN`)

All 6 checkpoints frozen at `analysis/manifest.json` (sha `81f6508cb4c9`) with
each Validation RMSE re-derived from the reloaded checkpoint (`recheck_abs_diff`
≤ 2.1e-8) and `curve_argmin_epoch == best_epoch`. Errors are computed in **raw
log(mol/L)** — the Train-fitted scaler is inverted, no DeepChem normalisation is
used anywhere. `test_labels_read` is `false` in every `train.json`; Test views are
constructed only inside `evaluate_test`, whose output directory is written once.

| model | Test RMSE | Test MAE | Test R² |
|---|---:|---:|---:|
| ESOL-BASE (3 seeds) | **0.6990 ± 0.0261** | 0.5165 ± 0.0270 | 0.8832 ± 0.0088 |
| ESOL-DROP20 (3 seeds) | 0.7126 ± 0.0152 | 0.5359 ± 0.0050 | 0.8787 ± 0.0052 |

Validation and Test rank the six runs only weakly (Pearson 0.51, Spearman 0.43):
seed 1 is the *worst* Validation run of BASE (0.767) and the *best* Test run
(0.690). With 113 molecules this is expected and is the reason no claim here
rests on a single seed or a single split.

---

## 9. Test error analysis (`analysis/eval/error_analysis.json`, per-seed CSVs)

**Where the error lives** (BASE seed 1, Test):

| subset | n | RMSE | mean signed error | label sd |
|---|---:|---:|---:|---:|
| acyclic molecules | 43 | **0.338** | −0.02 | 1.33 |
| cyclic, novel scaffold, contains a benzene ring | 21 | **0.911** | −0.20 | 1.93 |
| cyclic, novel scaffold, no benzene ring | 49 | **0.801** | −0.10 | 1.92 |

Local atom-environment statistics transfer across molecules regardless of ring
system; the failure mode is whole-molecule architecture. Even the presence of a
familiar benzene ring does not help when the scaffold as a whole is new — those
molecules do *worse* than the ring-free cyclic ones. That is a statement about
the **global `rho` branch**, which pools S1/S2 means and cannot identify
scaffold-level solvation differences.

**Range behaviour** (same run): bias is +0.28 in the most insoluble bin
(y ∈ [−12,−4), RMSE 0.79) and −0.29 in the most soluble bin (y ∈ [0,2)) —
classic shrinkage toward the Train mean. No seed shows a different pattern
(all |bias| ≤ 0.21).

**Extreme errors are few, and clustered:**

| | value |
|---|---|
| median \|err\| | 0.28–0.39 |
| p90 \|err\| | 1.06–1.23 |
| max \|err\| | 2.14 (BASE seed 0) |
| SSE share of worst 1% / 5% / 10% of molecules | 8% / 36% / 53% |
| SSE share of the single largest-error **scaffold group** (`c1cnc2ncncc2n1`, 9 molecules) | **23%** |
| groups needed for 50% of Test SSE (of 71 test groups) | **5** |

Worst molecules are chemically coherent rather than random: purine/pteridine-type
fused N-heterocycles under-predicted by 1.3–2.2 log units; a heavily chlorinated
cage ketone under-predicted by 1.4; a large diether over-predicted by 2.0; `C=C`
(2 atoms, **zero** S2 paths) under-predicted by 1.4. Absolute error correlates
with molecule size (Spearman 0.24–0.27 with n_atoms) and not with logP
(−0.02 … −0.11).

**Seed dependence:** the *level* is stable (Test RMSE 0.679–0.728 across seeds)
but the *ranking* of runs is not, and the Validation→Test agreement is weak.
Nothing in this report should be read as a claim about split stability: this is a
single scaffold partition and a three-seed estimate — a **first cross-task
exploration**.

---

## 10. Answers to the three questions

**Q1 — can the original S1/S2 representation learn ESOL and generalise?**
Yes, and better than expected for a representation that carries no explicit
polarity/hydrophobicity descriptor. Mean Test RMSE 0.699 (MAE 0.517, R² 0.883)
under a scaffold split, against 2.115 for a no-information predictor. For context
only — different split protocol, features, label handling and metric convention,
so not a benchmark comparison — Delaney's own ESOL equation is reported at RMSE
≈ 0.75 / R² ≈ 0.77 under 5-fold CV on these same 1,128 compounds.

**Q2 — is the bottleneck small-sample overfitting or missing solubility
information?**
The evidence points to **information / representation of the global correction**,
not to regularisation-fixable overfitting:

- the overfitting hypothesis was falsified operationally: doubling dropout
  shrank the Train/Validation gap by 17% and made both Validation (+0.012) and
  Test (+0.014) worse, with the best epoch earlier in 3/3 seeds;
- the error is *structured by scaffold novelty*, not spread over molecules
  (acyclic 0.34 vs novel cyclic 0.80–0.91; ≤5 scaffold groups carry 50% of SSE);
- the residual bias is shrinkage toward the Train mean at both ends of the
  range, which is what a global branch that cannot separate scaffolds does.

Caveat stated plainly: this is inference from error structure, not a test of the
information hypothesis — candidate B (`ESOL-PHYSCHEM6`) was deliberately not run,
so "adding explicit polarity/hydrophobicity to `rho` would help" remains an
untested hypothesis, not a result.

**Q3 — one simple, interpretable modification?**
`ESOL-DROP20` upgraded exactly one scalar and was refuted with paired evidence. Reportable
as a negative result: on ESOL, LCRC-A is **not** primarily limited by the kind of
variance that dropout removes.

---

## 11. An observation for the next round (not acted on)

Every run clips gradients on 26–31% of steps (736–904 of 2,880), with recorded
max gradient norms of 370–1,480 under a `clip = 5.0` inherited from the
Lipophilicity recipe (`nonfinite_events = 0`, `gradient_probe` passes in all
runs). On Lipophilicity the same clip was effectively inactive. Whether
`clip = 5.0` is mis-calibrated for ESOL (or simply reflects a harder early loss
surface) was **not** investigated because the formal-run budget was spent; it is
recorded here as the cheapest next diagnostic, not as a claim.

---

## 12. Execution status and provenance

| item | value |
|---|---|
| formal training runs | **6** (P1: 3 × ESOL-BASE, P2: 3 × ESOL-DROP20) — hard cap 6 |
| GPU | 6 × A100-SXM4-40GB on `res` (process backend, pools `res-gpu0` / `res-gpu1`), fp32, `CUDA_VISIBLE_DEVICES` 0/1 recorded per run |
| GPU consumption | **0.132 GPU·h** train wall (475 s total; 69.8–89.8 s/run) + 0.004 h overhead → ledger **0.1357 / 1.5 GPU·h** (`analysis/gpu_ledger.json`) |
| driver / torch | 550.163.01 / 2.5.1+cu124 |
| no Slurm | everything ran on the process backend of `res`; `res-2` unused |
| commits | `c160719f` code + data + split + P0 + freeze; `ea8bb0d1` P1 (+P2 hypothesis); `144036b2` P2 + checkpoint freeze; report commit = this file |
| runs ran at commit | `c160719f5d8a`, `git_dirty_tracked = false` in all six `train.json` |
| smoke | 3 local CPU smoke runs before deploy (`analysis/smoke/smoke.json`), never counted as formal |
| failures / retries | no failed or re-run training; two `rr deploy` refusals (stale untracked run artifacts on the remote) fixed by moving them aside — no experiment was repeated |
| artifacts | `analysis/{source,split,p0,protocol_freeze.json,smoke,runs,eval,manifest.json,MANIFEST_FROZEN,gpu_ledger.json,rr_status,p1,p2}` |
| test predictions | `analysis/eval/per_molecule_test.csv` (all 6 arms × 113 molecules), `metrics.json`, `summary.json`, `paired.json` |

### Deliberately not done

S1-only / S2-only ablation · third model candidate · extra scaffold splits ·
more seeds · Lipophilicity retraining · SSL / message passing / attention ·
external architectures (PNA, D-MPNN, GIN) · hyper-parameter search · dropout sweep
(0.15 / 0.25 / 0.30) · any Test-driven adjustment.
