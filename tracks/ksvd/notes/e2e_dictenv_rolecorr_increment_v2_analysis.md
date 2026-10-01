# E2E-DictEnv-RoleCorr-Increment-v2 — analysis (route 1, seed 0)

Round: `e2e_dictenv_rolecorr_increment_v2` · protocol `zinc-context-gap` ·
run `20261001-110113-4ed73ccc` · revision `8244602` · device **cpu**
(pre-registration amendment 1: user-authorised local CPU regime because the
remote A100 host could not initialise CUDA; see
`notes/e2e_dictenv_rolecorr_increment_v2_preregistration.md` §0b).

## Decision (first line)

**`INCREMENT_NO_MATERIAL_GAIN`** — on top of the *full* frozen K32/s8 structural
budget, appending the frozen sparse dictionary code of the role↔attribute
correspondence object (B) improves official-valid Top-5 soup MAE over the matched
appended structural control (A) by only **+0.000731**, far below the frozen
absolute gate of **0.003**. The correspondence block is causally **used** inside
B (zero +0.0173, within-molecule shuffle +0.0188, within-group object shuffle
+0.0211); the auxiliary comparison shows **no sparse advantage**
(B − C = +0.0024 in favour of the dense PCA16 control). Route 1 stops here.

## 1. Primary table (official valid 1000, Top-5 soup)

| arm | appended 16-wide block | best MAE | best ep | soup MAE |
|---|---|---:|---:|---:|
| A `EXTRA-STRUCT` | frozen `D_S` structural residual, K16/s4 | 0.13176929 | 288 | 0.12639307 |
| B `CORR-ADD` | frozen scaler + `D_C` correspondence, K16/s4 | 0.12874733 | 317 | 0.12566183 |
| C `CORR-PCA-ADD` | affine PCA16 of the same object (train-fitted) | 0.13017120 | 311 | 0.12329247 |

* `improvement_B_minus_A = M_A − M_B = +0.000731` vs gate `0.003` → **not fired**.
* `B_minus_C = +0.002369` → sparse advantage **not supported** (tolerance 0).
* Descriptive, **not a pre-registered primary comparison**: `M_A − M_C = +0.003101`
  (C better by 0.0031 in mean soup MAE; C better in 51.2% of molecules, median
  per-molecule delta −0.001380). Same architecture, same trainable budget, same
  shared init; C's block is dense/continuous (no atomic constraint) while A/B use
  exactly `l0 ≤ 4` frozen atoms, so the gap mixes block content with block family.

Soup members (epoch): A `[238, 288, 289, 300, 305]`, B `[303, 307, 308, 316, 317]`,
C `[280, 301, 310, 311, 316]`.

## 2. Mechanism probes (soup states, ΔMAE of the probe vs intact)

| probe | A | B | C |
|---|---:|---:|---:|
| zero the appended block | +0.055292 | **+0.017293** | +0.011696 |
| within-molecule row-shuffle (seeds 11/22/33/44/55) | +0.076695 | **+0.018822** | +0.024157 |
| within-group attribute-permuted object (seeds 101/202/303/404/505) | — | **+0.021111** | — |
| zero the base 32-coordinate block (diagnostic) | +0.024401 | +0.027531 | +0.033999 |

The B block is load-bearing: all three probes exceed both the directional (0.003)
and the clear (0.010) thresholds. It is *used* — it simply does not pay. Absolute
probe MAEs: B intact 0.125662, block zeroed 0.142955, base zeroed 0.153193
(A: 0.126393 / 0.181685 / 0.150794; C: 0.123292 / 0.134989 / 0.157292). Code
usage of B's block: 16/16 atoms active, effective atoms 7.89, exact `l0` mean 2.28,
top1 share 0.531 (base path: 31/32 atoms, effective 21.48, exact `l0` mean 8.0,
bit-identical across arms).

## 3. Paired per-molecule (official valid, n = 1000)

| comparison | mean Δ | median Δ | fraction second arm better |
|---|---:|---:|---:|
| A vs B | −0.000731 | +0.000621 | 0.493 |
| A vs C | −0.003101 | −0.001380 | 0.512 |
| C vs B | +0.002369 | +0.001416 | 0.485 |

The A-vs-B difference is a wash in both directions (mean says B slightly better,
median and fraction say A), i.e. there is no broad per-molecule shift to explain;
the correspondence code is not paying for the 16 appended readout columns.

## 4. Budget, cost and cleanliness

| item | value |
|---|---|
| trainable parameters (all arms, identical) | 99,469 |
| total parameters A / B / C | 102,589 / 110,125 / 101,549 |
| wall per arm A / B / C | 1832.5 s / 2165.1 s / 2142.7 s |
| seconds per epoch A / B / C | 5.73 / 6.77 / 6.70 |
| control-plane runtime | 6216.9 s (~103 min) |
| epoch budget | 320 (all arms completed; no truncation) |
| tracked code state | clean at revision `8244602` (`diff_hash` empty; `dirty` only because two pre-existing untracked result directories were present) |
| official test | never instantiated (`test_access: blocked`, `official_test_loaded: false`) |

Correctness gates all passed (`correctness.json: all_passed=true`: geometry,
base/appended bit-identities, binding-extension layout, shared init, purity,
freezing, wiring), smoke passed (`smoke.json: passed=true`, appended-binding
gradient present). Peak GPU memory: n/a (CPU regime).

## 5. Interpretation — what class of evidence is this?

* **Channel merely used**: yes, demonstrated (probes ≥ 0.017), but that is not the
  round's question.
* **Encoding-object increment (sparse correspondence on top of the full
  structural budget)**: **not supported** — +0.000731 ≪ 0.003.
* **Sparse advantage over a dense code of the same object**: **not supported** —
  directionally negative (B − C = +0.002369).
* **"Correspondence object is worthless"**: not claimed and not measured; the
  object's information may still be recoverable by a *different* encoding (see the
  descriptive C observation), and this round only tested the frozen sparse
  encoding at this placement.
* **Insufficient evidence**: only for the dense-control thread (single seed, not
  a pre-registered comparison).

Relation to RoleCorr-v1 (`e2e_dictenv_rolecorr_v1`): v1 replaced half of the
structural budget with the correspondence coordinate at width 33 and lost 2.6%;
this round removed that confound by keeping the full K32/s8 budget and merely
appending 16 columns. The added budget removes the *damage* (B is now slightly
better than A instead of 2.6% worse) but does **not** create a material gain, and
the dense affine code of the same object is the only arm that moved the mean by
the gate magnitude (descriptively). Taken together: the frozen sparse dictionary
coding of this object is not the mechanism that converts its information into
accuracy at either placement.

## 6. Scope discipline

Not claimed: seed 1 or any significance statement; any other width, K, sparsity,
scaler, dictionary, PCA rank, horizon or lr; any route-2 statement; any terminal
(official-test) statement; any general statement about correspondence objects or
about dense codes (C is a descriptive control). Not authorised without a new
pre-registration: any rerun/extension of this round, promoting C post hoc,
end-to-end dictionary fine-tuning, and route 2 (prepared objects only:
`joint_scaler`/`joint_cache`/`joint_objects` stages; its training stages are
deliberately not implemented).

## 7. Exact reproduction

```bash
# as executed (committed revision 8244602, CPU regime)
uv run research run zinc_e2e_dictenv_rolecorr_increment_v2 \
  --study zinc-context-gap --mode screen \
  --purpose "Increment-v2 route1 primary screen (user-authorized local CPU regime, prereg amendment 1)" \
  --set runtime.device=cpu
# evidence
uv run research show 20261001-110113-4ed73ccc
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_rolecorr_increment_v2.py
```

Artifacts: `results/e2e_dictenv_rolecorr_increment_v2/` (REPORT.md, DECISION.md,
summary.json, interventions.json, run_*.json, soup_*.json, curve_*.csv,
per_molecule_errors.npz, checkpoints/), promoted run record
`records/runs/20261001-110113-4ed73ccc.json`.
