# LCRC Lipophilicity v4 — local-chemistry (atom-level) pilot

**Status:** **COMPLETE** — one unconditional GPU run; the conditional broadcast control was **not run** (P1 gate not met).
**Protocol:** `lcrc-lipo-local-chem-pilot-v4`
**Protocol freeze SHA-256:** `1085d817465c6515a85e0d7e22e44d2cd43c5132e83a891c29c434dff6b7bf2e`
**GPU authorization:** `analysis/gpu_authorization_v4.json`, pinned to that freeze, host `res`, pool `res-gpu0`, cap 0.20 A100 h, mode `pre_authorized_by_researcher_in_session`.
**Verdict:** **`NO_CLEAR_LOCAL_CHEM_GAIN`**
**GPU ledger:** **0.0442 h / 0.20 h**, 1 charged job, 0.1558 h remaining.

---

## 摘要（中文）

v4 的问题是：把疏水性/极性/氢键化学信息注入冻结 LCRC-A 的**原子级**表示，是否比 v3 的"分子级描述符拼进 `rho`"更有效？

答案：**没有**。加入 Crippen 原子 logP、TPSA 原子贡献、HBD/HBA 原子标记（4 列，+512 参数，其余权重与 seed-0 逐位一致）后，Fold-1 dev RMSE 由 `0.740343` 变为 `0.747044`，**ΔRMSE = −0.006701**（方向为变差，远低于预注册的 0.03 材料性阈值）；dev MAE 反而略好 `+0.006547`。RMSE 与 MAE 方向冲突、bootstrap 区间跨零、且增益集中在"容易的分子"上（去掉历史残差最大的 2% 后 Δ 变为 +0.006229，即整体变差但尾部更差）。因此按冻结规则**不执行** P2 广播对照，判定 `NO_CLEAR_LOCAL_CHEM_GAIN`。

配合 v3 的 `NO_GLOBAL_DESCRIPTOR_GAIN`：在冻结 LCRC-A 上，无论把化学信息放在**分子级 `rho`** 还是**原子级输入**，这两类现成 RDKit 化学量都没有带来可材料化的泛化增益。本轮唯一 GPU 消耗 0.0442 A100 h。

---

## 1. Question

Does injecting hydrophobicity / polarity / hydrogen-bond chemistry into the **atom-level** representation of the frozen LCRC-A generalise better than v3's **molecule-level** descriptors appended to `rho`?

This is deliberately the atom-local counterpart of v3. v3 answered "no" for global descriptors (`NO_GLOBAL_DESCRIPTOR_GAIN`). v4 tests whether the failure was *placement* rather than *information*: the same chemistry families, but attached to the atom that owns them, so the message passing can route them through bonds.

Scope is unchanged from v2/v3: exploratory, single fold (Fold 1 of Fit2352), single seed (0), single recipe (R3). No SSL, no Monitor, no Reserved, no official valid/test labels. Diagnostics use Fold-internal Fit labels only.

Frozen comparator (`F1_A_sel_s0`, v2 historical A-R3):

| Arm | Fold | Seed | Train RMSE | Dev RMSE | Dev MAE |
|---|---:|---:|---:|---:|---:|
| A-R3 (v2 historical) | 1 | 0 | 0.2696302152948522 | 0.7403427006133139 | 0.546694119695902 |

---

## 2. What was added (and how it was proven aligned)

`AtomChem4`, in the frozen order:

| # | column | definition |
|---|---|---|
| 1 | `crippen_logp_atom` | RDKit Crippen atom logP contribution on `Chem.AddHs(mol)`, each added-H contribution folded onto its unique parent heavy atom |
| 2 | `tpsa_atom` | `rdMolDescriptors._CalcTPSAContribs` per heavy atom |
| 3 | `is_hbd` | 1 if the atom is the single atom of any `BaseFeatures.fdef` Donor feature |
| 4 | `is_hba` | 1 if the atom is the single atom of any `BaseFeatures.fdef` Acceptor feature |

Design choices that were *audited, not assumed*:

- **Crippen convention.** Heavy-atom Crippen contributions alone do **not** sum to `Crippen.MolLogP` (the H terms are missing). Computing on `AddHs(mol)` and folding each H onto its parent makes the heavy-atom sum equal `MolLogP` exactly: max |sum − MolLogP| = **8.88e-15** over all 2,352 Fit molecules. TPSA: max |sum − `Descriptors.TPSA`| = **5.68e-14**.
- **Donor/acceptor ambiguity.** All Donor and Acceptor features in the shipped `BaseFeatures.fdef` (sha256 `766f9790…d9784`) are single-atom, so `multi_atom_donor_acceptor_features = 0`; no atom had to be split or guessed.
- **Atom order.** Nothing is assumed. Every atom index is checked attribute-by-attribute against (a) the frozen raw-graph artifact produced by the v1 adapter in `GetIdx()` order, (b) the frozen v2 cache one-hots (charge/H), and (c) a fresh RDKit parse: atom class, charge/H one-hots, decoded physical bond sets, plus the two contribution sums above. **`n_mismatches = 0`** over 2,352 molecules / 63,231 atoms. The audit records the whole identity chain (`align → raw row → cache position → molecule id`).
- **Standardisation.** Fold-1 **train atoms only**: per-column mean/std, no zero-variance columns. Train means after standardisation are 8.6e-14, stds 3.5e-13. A perturbation experiment (multiplying dev atoms by 1000 and shifting) leaves mean/scale bit-identical, proving dev atoms never enter the fit.
- **No leakage of labels.** Features are computed from structure only (`labels_used = false`); thresholds and normalisation never touch dev labels.

Feature counts: 5,255 donor atoms, 8,853 acceptor atoms, 1,275 atoms that are both.

---

## 3. Model surgery (+512 parameters, nothing else)

P1 `A_AtomChem4` differs from the frozen A model **only** by the atom-encoder input width, 51 → 55.

- Original model built first with seed 0; every same-shaped tensor copied over. All **34** shared tensors are bitwise equal, and the widened first atom-encoder layer's first 51 columns are bitwise equal to the historical init.
- New 4 columns zero-initialised (`new_columns_zero_count = 0` non-zero values).
- Parameters: **247,106 → 247,618**, delta **+512** (= 4 × 128), asserted at build time and again after training.
- Zero-extra forward is **exactly** the original model: max |Δoutput| = **0.0** on 64 dev molecules. With the real features the output is also unchanged at init (zero columns), confirming the model starts on the historical trajectory.
- The new columns receive finite, non-zero gradients (512/512 non-zero, ‖g‖ = 3.2615) and the first-51-column gradient equals the original to 4.1e-08 relative (float32 reassociation of a 55-wide vs 51-wide matmul; bitwise equality is explicitly *not* claimed for that reason).
- Bond13, scalar9, global42, S1/S2, `f`, `rho`, `contrib + rho` output and the absence of iterative message passing are unchanged; the training loop is `v2_training.train_job` reused verbatim with only a builder dispatch and a collate that appends the 4 columns.
- Data identity was re-verified from the GPU run itself: the run's `data_view` (`train_positions_sha256`, `dev_positions_sha256`, `target_norm_*`, `train_norm_fingerprint`) is byte-identical to the historical `F1_A_sel_s0` run.

CPU gates: `analysis/cpu_gates_v4.json` → `READY_FOR_GPU_AUTHORIZATION`, 0 failures (atom features/alignment, fold alignment, train-only normalisation, forward equivalence + gradients, broadcast control, raw-logD de-normalisation + historical checkpoint replay, v1/v2/v3 immutability, temporary CPU training). The historical checkpoint replays to 1.68e-06 on CPU, so the reported metric is genuinely raw logD units.

Broadcast control (P2) was built and audited but **not trained**: per-molecule constant within every molecule, exactly equal to the per-molecule mean of the standardised vector, same spec hash, same parameter count, same init-state hash.

---

## 4. Results

All rows: Fold 1, seed 0, recipe R3. `ΔRMSE = RMSE(historical A-R3) − RMSE(v4)`; **positive would favour the v4 model**.

| Model | Fold | Seed | Train RMSE | Dev RMSE | Dev MAE | ΔRMSE vs A-R3 | Train−dev gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| A-R3 (v2 historical) | 1 | 0 | 0.269630 | **0.740343** | 0.546694 | 0.000000 (reference) | 0.470712 |
| P1 `A_AtomChem4` | 1 | 0 | 0.263057 | 0.747044 | **0.540148** | **−0.006701** | 0.483987 |
| P2 `A_BroadcastChem4` | — | — | — | — | — | **not run** (gate) | — |

P1 fits the training fold slightly better than A-R3 (0.263057 vs 0.269630) and is slightly worse on dev. The generalisation gap widens by 0.0133. Relative dev change: **+0.905 %** worse.

### Paired and bootstrap evidence (n = 784 dev molecules, 2,000 resamples, seed 20261102)

| Comparison | ΔRMSE | 95 % CI | ΔMAE | 95 % CI | Molecule win rate (absolute) |
|---|---:|---|---:|---|---:|
| P1 vs A-R3 | −0.006701 | [−0.031255, +0.016516] | +0.006547 | [−0.007027, +0.020309] | 53.2 % (417 improved / 367 worsened) |

The RMSE interval straddles zero and the MAE interval straddles zero: the two metrics point in **opposite** directions at a magnitude an order of magnitude below the pre-registered 0.03 materiality. Mean squared improvement is **−0.009967** (negative); median absolute improvement is +0.014514. This is a wash, not a gain.

### Tail diagnostic (pre-registered)

Removing the 16 dev molecules with the largest **historical** A-R3 residuals (2 %, selected before any v4 run):

| Quantity | Value |
|---|---:|
| ΔRMSE full dev | −0.006701 |
| ΔRMSE after dropping top 2 % historical tail | **+0.006229** |
| retained_fraction | undefined (full Δ ≤ 0) |
| tail share of net squared-error gain | 1.79 (tail contributes *negative* improvement) |

So the honest reading is: P1 is very slightly better on the bulk of dev molecules and clearly worse on the handful of hardest ones. Because the full-dev δ is negative, `not_tail_dependent = false` **by construction** — the flag is not evidence in either direction here. The decision rests on the materiality threshold, not on the tail flag.

---

## 5. Where the (tiny) differences live — post-hoc subgroups

Descriptive only; not used for the decision (`analysis/subgroup_results_v4.json`).

| Group | n | A-R3 RMSE | P1 RMSE | ΔRMSE |
|---|---:|---:|---:|---:|
| all dev | 784 | 0.7403 | 0.7470 | −0.0067 |
| TPSA ≤ 58.4 (train q25) | 213 | 0.7203 | 0.6744 | **+0.0459** |
| TPSA ≤ 81.2 (train median) | 408 | 0.7042 | 0.7010 | +0.0031 |
| TPSA (81.2, 97.5] | 170 | 0.7539 | 0.7121 | **+0.0418** |
| TPSA > 116.3 (train q90) | 63 | 1.0430 | 1.1454 | **−0.1025** |
| HBD = 1 | 277 | 0.7531 | 0.7317 | +0.0214 |
| HBD = 2 | 274 | 0.6892 | 0.6667 | +0.0225 |
| HBD = 3 | 97 | 0.7449 | 0.7925 | −0.0476 |
| HBD ≥ 4 | 24 | 1.4076 | 1.6213 | **−0.2137** |
| HBA 0–1 | 23 | 0.9686 | 0.8567 | +0.1119 |
| HBA 4–5 | 295 | 0.8011 | 0.8458 | −0.0447 |

The pattern is consistent and unhelpful: P1 helps low-to-mid polarity molecules and **degrades exactly the most polar / most hydrogen-bonded molecules** that dominate the error. Adding atom-resolved polarity features made the hardest cases harder, not easier — the mirror image of the v3 global-descriptor result.

**Formal charge could not be evaluated**: 2,351 of 2,352 Fit molecules are neutral, and the single charged molecule is in the Fold-1 *train* split, so the dev charged subgroup has n = 0. This is reported rather than silently dropped.

---

## 6. Why P2 was not run

The frozen gate required, for `A_AtomChem4` vs `F1_A_sel_s0`:

1. `ΔRMSE ≥ 0.03` — observed **−0.006701** → **FAIL**
2. `ΔMAE ≥ 0` — observed +0.006547 → pass
3. gain not tail-dependent — structurally false because (1) fails

Rule (1) alone stops the round. Per the frozen protocol, the broadcast control exists to distinguish *atom-correspondence* from *feature information* **given a material gain**; with no gain there is nothing to attribute, and running it would spend budget without a trigger. `F1_A_BroadcastChem4_R3_s0` was therefore **not executed**.

---

## 7. Interpretation

- **Local placement did not rescue the chemistry prior.** v3 (molecule-level, global channel) → −0.0096 / −0.0171. v4 (atom-level, local encoder) → −0.0067. Three chemistry injections, two placements, one direction: no material gain.
- **RMSE/MAE conflict is diagnostic.** MAE improved by 0.0065 while RMSE worsened by 0.0067, and the win rate is 53 %. P1 trades a small typical-molecule improvement for larger errors on the extreme molecules. A prior that helps the easy molecules and hurts the hard ones cannot improve a tail-dominated regression.
- **What this does and does not close.** It closes the route "give frozen LCRC-A the standard RDKit lipophilicity/polarity/hydrogen-bond quantities, at molecule level or atom level". It does **not** rule out learned chemical representations, different supervision, chemistry features outside this 4-column family, or architectural change — each of which needs a new hypothesis, protocol and freeze.
- **Why it plausibly fails.** The frozen atom51 encoding already carries element/aromaticity/charge/H-count, which is most of what Crippen and TPSA compute from; the four extra columns are deterministic functions of that same local environment. Deterministic re-encodings of information already present add capacity (and 2,807 gradient-clip events vs 2,797 historically) without adding information.

---

## 8. Limitations

- One fold, one seed, one recipe. Nothing here supports a mechanism claim; the deltas are within single-seed noise, which is exactly why the pre-registered materiality threshold (0.03) was set high.
- Four features only, all RDKit heuristics, all fold-train standardised. No 3-D, no ESP, no learned atom embeddings, no pharmacophore distances.
- Formal-charge subgroup is empty on dev; the dataset is effectively neutral-only.
- The dataset has no separate experimental logP column: the label *is* logD, so "logD − MolLogP" is computed as `label − Crippen.MolLogP`. Descriptively: mean −1.1336, median −0.8488, q05 −3.6416, q95 +0.7054 over Fit2352. Only 3 molecules sit on a clipping boundary (4.5), none on −1.5.
- P2 was designed, audited and left unrun; its code path is covered by CPU tests but has never seen a GPU epoch.

---

## 9. Provenance, GPU accounting, checkpoints

Execution regime: host `res`, pool `res-gpu0`, node `a100-2`, NVIDIA A100-SXM4-40GB (driver 550.163.01), torch 2.5.1+cu124, CUDA 12.4, python 3.12 — the same regime as the historical comparator, so the comparison is regime-matched.

| Item | Value |
|---|---|
| run key | `F1_A_AtomChem4_R3_s0` |
| git commit at launch | `0bc827f5335f6774668d94e17cf60f1d7d91eef6` |
| wall seconds | 151.09 |
| charged GPU hours | 0.044191 |
| cap / remaining | 0.20 / 0.155809 |
| `soup.pt` sha256 | `5c198ac9dcb64e5b4f02e8f89d13ed99ddde5dd07472bee7467d2478d2fac46f` |
| `predictions.npz` sha256 | `7ee81336d876b89723c302837022cf2493513a186faa23d9ba23bb79a4c408bb` |
| init state hash | `cc64a0a157ab410815761f88806f3b5a21f0274e9bb7fbce891e78d9fc8cdf9b` |
| soup state hash | `e81f5ce2d5848bb83b8204a503fdad7659e9e466538981590e71715f2ed8b62e` |
| grad-clip events (vs historical) | 2,807 (vs 2,797) |

Two operational notes, recorded for honesty:

1. `rr run v4-A1` (~13:00) **failed after ~10 s with exit 1 and no training**: the GPU-authorization JSON existed locally but was not yet committed, so the remote checkout could not read it. It was committed, redeployed, and the run was relaunched as `v4-A1b`. No GPU-hours are charged for the aborted launch (no `train.json` was produced, hence nothing in the ledger).
2. The remote `res` checkout arrived dirty with one modified derived file (`v3/analysis/gpu_ledger_v3.json`). Its content differed from the committed local copy only in `created_utc` (the local commit is a regeneration of the same ledger, both containing the same two charged entries and the same 0.07622470870083715 h total). Local Git is the source of truth, so the remote copy was restored to the committed revision before deploying. Nothing that was not already in Git was discarded.

---

## 10. Artifacts and reproduction

```
tracks/structural-sufficiency/lcrc_lipo_local_chem_pilot_v4/
  REPORT.md, DECISION.md
  configs/lcrc_lipo_local_chem_pilot_v4.yaml
  code/          v4_common, v4_atomchem, v4_models, v4_audit, v4_descriptive,
                 v4_evaluate, v4_registry, run_v4
  tests/test_v4_pilot.py                       # 16 fast tests
  analysis/protocol_freeze_v4.json             # 1085d817…
  analysis/gpu_authorization_v4.json
  analysis/cpu_gates_v4.json                   # READY_FOR_GPU_AUTHORIZATION
  analysis/atomchem_audit_v4.json              # 0 mismatches, sums equal
  analysis/fold_alignment_v4.json              # 1568/784 mol, 42198/21033 atoms
  analysis/atom_norm_v4.json                   # train-only mean/scale
  analysis/forward_equivalence_v4.json         # +512 params, zero-extra ≡ original
  analysis/broadcast_control_v4.json           # P2 audited, not trained
  analysis/denorm_check_v4.json                # replay 1.68e-06
  analysis/descriptive_v4.json                 # §2 diagnostics
  analysis/temp_training_v4.json               # CPU smoke, temp dirs only
  analysis/frozen_extras_v4.json / .npz        # frozen 4-column arrays
  analysis/run_registry_v4.json                # exactly 2 jobs, 1 conditional
  analysis/budget_v4.json                      # 0.0871 h estimate ≤ 0.20 cap
  analysis/results_v4.json, paired_v4.json, decision_v4.json,
  analysis/subgroup_results_v4.json, table_v4.csv, summary_v4.json
  analysis/gpu_ledger_v4.json                  # 0.0442 h charged
  analysis/runs/F1_A_AtomChem4_R3_s0/{train.json,soup.pt,predictions.npz}
```

Reproduce:

```bash
cd tracks/structural-sufficiency/lcrc_lipo_local_chem_pilot_v4
uv run --group lipo pytest -q tests                     # 16 passed
uv run --group lipo python code/run_v4.py verify        # freeze + v1/v2/v3 immutability
uv run --group lipo python code/run_v4.py analyze       # regenerate table/paired/decision
```

v1, v2 and v3 files are untouched; `run_v4.py verify` re-checks the v3-pinned v2 manifest and the v1 freeze on every invocation, and the same guard ran inside the freeze build and inside the GPU job.

**This round is closed. No follow-up training is authorized by this report.**
