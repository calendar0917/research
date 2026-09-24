# E2E-DictEnv-v0 — analysis

Round `e2e_dictenv_v0` · study `zinc-context-gap` · protocol `e2e_dictenv_v0`
Preregistration: `notes/e2e_dictenv_v0_preregistration.md`
Amendment A1 (Stage-1 waiver): `notes/e2e_dictenv_v0_amendment_a1.md`
Implementation: `notes/e2e_dictenv_v0_implementation.md`
Prior-artifact audit: `notes/e2e_dictenv_v0_prior_artifact_audit.md`

```text
FROZEN VERDICT:  E2E_DICTENV_ABSOLUTE_WEAK
(frozen verdict table, case D: M_S > 0.145)
```

Formal run commit `eeeb6b34f641260a0373a3daeda1286a855717d7`, remote
A100-SXM4-40GB GPU1 (both arms sequential on GPU1; GPU0 was occupied by a
foreign 37 GB job), seed 0, official train 10 000 / official valid 1 000,
official test never loaded.

## 1. Headline

| quantity | value | gate | result |
|---|---:|---|---|
| `M_S` SparseDictEnv soup | **0.145508** | band ≤ 0.145 (viable) | **weak band, FAIL by +0.000508** |
| band | **weak** | — | — |
| `M_D` DenseTiedEnv soup | **0.313047** | — | — |
| `G_sparse = M_D − M_S` | **+0.167539** | ≥ 0.003 | **PASS** (≈ 56× the gate) |
| `M_zero` (α → 0) | 1.000347 | — | — |
| `G_dict-use = M_zero − M_S` | **+0.854839** | ≥ 0.010 | **PASS** |
| `M_shuffle` (5 perms) | 0.162797 | — | — |
| `G_assign = M_shuffle − M_S` | **+0.017289** | ≥ 0.010 | **PASS** |
| dictionary health | 27/32 active, eff 13.89 | ≥ 24/32 & ≥ 8 | **PASS** |
| FEC-S1 anchor (context) | 0.130422 | — | `M_S − anchor = +0.015086` |

**Every mechanism gate passed and the dictionary-specific contrast is enormous
(`G_sparse` = 0.168). The round nevertheless lands in the *weak* absolute band
because `M_S` = 0.145508 exceeds the frozen 0.145 boundary by 0.0005.**

## 2. What the causal contrast shows

The only difference between the arms is the tied coding operator applied to the
*same* 65×32 matrix: `α = IHT(D̄, φ)` exact top-8 (Sparse) versus `z = φ D̄`
dense (DenseTied). Everything else — parameters (66,158), initialisation, data
order, λ_rec, optimizer, reader — is bit-matched at step 0 (Gate G11).

* **The sparse dictionary coordinate is strongly load-bearing.** Neutralising
  the 32-d coordinate (`α → 0`, DC `1` kept, chemistry/bond/six-scalar/backend
  untouched) raises MAE from 0.1455 to 1.0003 (`G_dict-use` = 0.855; mean
  prediction shift 0.981, max 4.54). Almost all task signal flows through
  `IHT(D, φ)`.
* **The value is assignment-specific, not a marginal.** Breaking the
  `α_v ↔ q_v` correspondence within each (root, shell) — node set, shell, α
  multiset, q multiset, DC marginal, bond branch and global context all
  preserved — degrades MAE by `G_assign` = 0.0173 across all five permutations
  (0.15954–0.16458 vs clean 0.14551). The coordinate must be *correctly bound*
  to chemistry, not merely present.
* **Sparsity specifically matters, not just a 32-d tied coordinate.** The
  parameter-identical dense tied coordinate collapses to `M_D` = 0.313047 —
  more than twice as bad as Sparse. This is the sharpest result of the round:
  the same matrix, same width, same reconstruction target, but a dense linear
  code performs far worse than an exact-top-8 IHT code. (The dense code is a
  linear projection of `φ`; the sparse code is an adaptive overcomplete
  selection. The gap is the value of the sparse selection itself.)
* **Both arms reconstruct `φ` almost perfectly** (normalised reconstruction
  9.7e-5 sparse, 8e-6 dense) — so the difference is *not* reconstruction
  fidelity; it is how the coordinate organises the task.

## 3. Dictionary health (trained Sparse soup)

| diagnostic | train | valid |
|---|---:|---:|
| active atoms | 27/32 | 27/32 |
| effective atom count | 13.86 | 13.89 |
| support entropy | — | 2.6311 |
| top-1 / top-8 support mass | — | 0.1250 / 0.7733 |
| exact top-8 fraction | 1.000000 | 1.000000 |
| task gradient to `D` | 0.0874 | — |
| train–valid usage Spearman | 0.9982 | — |

The dictionary is heavily used and reusable: 10 atoms cover ≥ 98 % of molecules,
four atoms are used on effectively every molecule, and per-atom molecule
coverage spans 118–10 000. Five atoms (indices 2, 4, 11, 15, 29, 0-based) are
dead on the train distribution. `D` moves far from the K-SVD initialisation
(soup `‖D̄ − D̄_init‖_F` = 5.94, relative 1.05); effective rank 10.83;
coherence max 0.862, mean 0.175. Coefficients: mean |α| 0.849, median 0.444,
p90 1.930, max 2.34.

## 4. Frozen report questions Q1..Q8

| # | question | answer |
|---|---|---|
| Q1 | architecture purity (no MP / recurrence / writeback / FEC-S1 bypass)? | **PASS** (G1, G5, G6, G7, G8, G9; 15/15 CPU tests) |
| Q2 | dictionary centrality (`E`'s fine structure only from `IHT(D,φ)`)? | **PASS** — `G_dict-use` = 0.855; α → 0 destroys the model |
| Q3 | end-to-end coupling (task gradient to `D`; movement from K-SVD init)? | **PASS** — grad to `D` 0.0874 at the trained state; `D̄` moved 5.94 (rel. 1.05) |
| Q4 | structural anchoring (sparse / reusable / reconstructive / non-collapsed)? | **PASS** — exact top-8 (fraction 1.0), 27/32 active, eff 13.89, Spearman 0.998, rank 10.83, rec 9.7e-5 |
| Q5 | absolute band of `M_S`? | **weak** — `M_S` = 0.145508 > 0.145 |
| Q6 | `M_D − M_S ≥ 0.003`? | **yes** — `G_sparse` = +0.167539 |
| Q7 | zero-code Δ ≥ 0.010; shuffle Δ ≥ 0.010? | **yes** — +0.854839 and +0.017289 |
| Q8 | verdict from the six allowed verdicts? | **`E2E_DICTENV_ABSOLUTE_WEAK`** (case D) |

## 5. Caveats (must travel with the claim)

1. **Stage-1 waiver (Amendment A1).** The preregistered Stage-1 train-only
   smoke failed its `atoms_active ≥ 24/32` sub-gate (23/32) and was treated as
   passed by explicit user authorisation. The waiver is now *retrospectively
   justified*: the fully trained soup has **27/32** atoms active, so the smoke
   value was a 512-molecule/3-epoch transient, not a collapse. All other
   Stage-1 sub-gates passed then and the formal health gate passes now. The
   original `atoms_active: false` value is preserved in `smoke_gate.json`.
2. **Single seed.** The preregistration forbade seed 1. The whole result is
   one seed, chosen before the run.
3. **GPU protocol is seeded but not bit-deterministic.** The inherited protocol
   runs `index_add_` scatter with `use_deterministic_algorithms(False)`
   (`zinc_static_dictionary_pair._configure_determinism`). Re-running the same
   seed gives visibly different trajectories (e.g. sparse epoch-1 valid MAE
   0.912, 0.924, 1.008 across three attempts). **`M_S` = 0.145508 is only
   0.0005 above the 0.145 band boundary, so the weak-vs-viable band assignment
   is inside single-seed execution noise.** The mechanism contrasts (0.855 /
   0.0173 / 0.168) are far outside that noise; the band label is not.
4. **No official test** was loaded (raise-guarded, G12).
5. The historical FEC-S1 anchor (0.130422) is context only; the causal
   comparison is Sparse vs DenseTied and is matched by construction.

## 6. Reading

The E2E-DictEnv hypothesis is **substantially supported on mechanism and
specificity, and fails only on absolute strength**:

* a shared, task-coupled, sparse pure-topology dictionary can be the *only*
  fine-grained learned learned structural coordinate of a competitive
  no-message-passing environment (Q1/Q2/Q3);
* its value is *specific* and *assignment-mediated* — the matched dense tied
  coordinate is 2.15× worse and neutralising or shuffling the coordinate
  destroys or materially degrades the model (Q6/Q7);
* but the resulting architecture (`M_S` = 0.1455) does not reach the viable
  band and sits 0.0151 behind the FEC-S1 static-composition anchor.

So: the sparse dictionary core is the *right mechanism* but not yet the
*strongest architecture*.

## 7. Recommendation

Per the frozen table this round is **case D (absolute weak): STOP.** No seed 1,
no K/s/IHT/LISTA/dictionary-count/decoder/attention/LayerNorm/λ/epoch rescue is
authorized. Open questions worth a *new* preregistration:

1. **Resolution of the band boundary.** `M_S` = 0.1455 vs 0.145 with a
   non-deterministic single-seed protocol cannot distinguish "weak" from
   "viable but weaker". A future round should either stabilise the execution
   regime (deterministic scatter) or budget a small paired seed set for the
   band *label* (not for a rescue).
2. **Why dense-tied collapses.** `M_D` = 0.313 with perfect reconstruction
   suggests the value is in the *selection*, not the reconstruction — worth a
   targeted mechanism study (e.g. support statistics vs task error), under its
   own preregistration.
3. Whether the sparse dictionary core can be lifted into the viable band by a
   *different* composition backbone — a new architecture question, not a rescue.

## 8. Evidence index

* `results/e2e_dictenv_v0/artifact_identity.json`, `parameter_accounting.json`
* `results/e2e_dictenv_v0/correctness.json` (G0..G12 PASS)
* `results/e2e_dictenv_v0/lambda_calibration.json` (λ_rec = 135.834921)
* `results/e2e_dictenv_v0/smoke_gate.json` (Stage-1, with A1 override recorded)
* `results/e2e_dictenv_v0/sparse_seed0.json`, `sparse_curve.csv`
* `results/e2e_dictenv_v0/dense_tied_seed0.json`, `dense_tied_curve.csv`
* `results/e2e_dictenv_v0/dictionary_health.json`
* `results/e2e_dictenv_v0/mechanism_zero.json`, `mechanism_shuffle.json`
* `results/e2e_dictenv_v0/decision.json`, `REPORT.md`, `DECISION.md`
* `results/e2e_dictenv_v0/states/` (sparse/dense selection + soup checkpoints)
* `experiments/luyin16/e2e_dictenv_v0.py`,
  `experiments/luyin16/zinc_e2e_dictenv_v0.py`,
  `tests/test_e2e_dictenv_v0.py`
