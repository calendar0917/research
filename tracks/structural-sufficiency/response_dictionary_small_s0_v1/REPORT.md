# Response dictionary small screen, seed 0 (`response_dictionary_small_s0_v1`)

**Status:** completed, 4/4 formal units, development screen only.
**Protocol:** `PROTOCOL.md` (frozen before formal runs), `protocol_hash=861b778da2a7488db84579a29e9cc74d0703f2eb64b0583e6ed9d3850b9fcf17`.
**Test use:** `test_policy: always`; every Test number below is **`DEVELOPMENT_TEST`**. Test never selected a checkpoint, dictionary, or hyper-parameter.
**Machine-readable:** `summary.json`, `run_manifest.json`, `main_table.csv`, `gains.csv`, `bootstrap.csv`, `dictionary_table.csv`, `fusion_subgroups.csv`.

## 1. What was run

One shared fragment encoder (GINE, hidden 96, 3 layers, latent 16) and one seeded initial state
(`ccc8171d4cdc47da…`) for both arms. The only architectural difference is the fusion operator:

- `RESP`: `T = sum_j w_j q_j k_j / sqrt(16)`
- `ADD`:  `T = sum_j w_j (q_j + k_j) / sqrt(32)`

Both arms have 198,262 parameters. The `w` vector starts at exact zero, so the two arms produce
identical initial predictions; they diverge only after `w` receives gradient. For each molecule the
model averages up to 4 real single-non-ring-bridge cuts (smaller side 1–8 heavy atoms); a molecule
with no eligible cut goes through a separate whole-molecule fallback head.

| task | arm | best-Val epoch | Train RMSE | Val RMSE | Test RMSE | Val MAE | Test MAE | fallback Val |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Lipo | RESP | 63 | 0.3475 | **0.6287** | **0.6760** | 0.4821 | 0.5235 | 0.093 |
| Lipo | ADD | 91 | 0.3018 | 0.6762 | 0.7121 | 0.5010 | 0.5493 | 0.093 |
| ESOL | RESP | 154 | 0.3361 | **0.7278** | **0.8995** | 0.5621 | 0.6864 | 0.363 |
| ESOL | ADD | 105 | 0.4386 | 0.7703 | 0.9091 | 0.6103 | 0.7324 | 0.363 |

All units completed all 360 epochs with a full Train/Val/Test evaluation each epoch (no NaN, no
OOM fallback triggered, no early stopping). The selected state is the best-Val epoch after epoch
360; Test at that state is reported as development evidence. Final-epoch and last-10-epoch values
are descriptive only:

| task | arm | final Val | final Test | last-10 Val | last-10 Test |
|---|---|---:|---:|---:|---:|
| Lipo | RESP | 0.6518 | 0.6936 | 0.6518 | 0.6931 |
| Lipo | ADD | 0.7106 | 0.7298 | 0.7104 | 0.7296 |
| ESOL | RESP | 0.7497 | 0.9050 | 0.7503 | 0.9045 |
| ESOL | ADD | 0.8456 | 0.9616 | 0.8441 | 0.9593 |

The `HORIZON_LIMITED` rule is false for all four units: last-30 minus previous-30 Val RMSE is
`+0.00024` (Lipo RESP), `-0.00068` (Lipo ADD), `+0.00002` (ESOL RESP), `+0.00004` (ESOL ADD), all
far above the thresholds `-0.00606` (Lipo) and `-0.01033` (ESOL). The global best epochs are far
from the last 10 epochs, so no continuation is indicated.

## 2. RESP versus the matched additive control

The primary comparison is the best-Val state of the two arms on the same split and initialization.

| task | Val gain (ADD−RESP) | 95% molecule-bootstrap interval | Test gain (ADD−RESP) | after dropping top-1 Val molecule |
|---|---:|---:|---:|---:|
| Lipo | +0.0475 | [+0.0195, +0.0757] | +0.0361 | +0.0439 |
| ESOL | +0.0425 | [−0.0363, +0.1243] | +0.0096 | +0.0238 |

The interval is a molecule-row bootstrap (2000 draws, seed 1703) at the fixed selected state; it is
not a training-seed interval. On Lipo, 100% of bootstrap draws favor RESP and the top-1 Val molecule
accounts for only 8.2% of the total gain; removing it leaves +0.0439. On ESOL, 84.25% of draws favor
RESP, the interval crosses zero, and one Val molecule (official row 733) accounts for 45.3% of the
total gain; removing the top-1 still leaves +0.0238, but removing the top five turns the Val gain to
−0.0120.

The Lipo result is not a “better fit” artifact: RESP has **worse** Train RMSE than ADD
(0.3475 vs 0.3018) but better Val and Test. On ESOL, RESP is better on Train too (0.3361 vs 0.4386);
its Train advantage (+0.1025) is larger than its Val (+0.0425) and Test (+0.0096) advantages, so an
optimization/fit component cannot be excluded. This is why ESOL is labelled conditional-uncertain,
not a clean replication.

Two mechanistic diagnostics are consistent with a real interaction term but do not prove it:

- **View dispersion** (global RMS of per-view scores around the molecule mean, raw units):
  Lipo RESP `0.187` Val / `0.188` Test vs ADD `0.223` / `0.239`;
  ESOL RESP `0.198` / `0.218` vs ADD `0.338` / `0.353`.
  RESP's per-view scores are less dispersed, i.e. the multi-view aggregation is more stable.
- **Fixed-weight `w=0` probe** on the best RESP state (not a necessity proof): zeroing the learned
  fusion weights raises Lipo Val RMSE `0.6287→0.6979` and Test `0.6760→0.7249`; ESOL Val
  `0.7278→0.7828` and Test `0.8995→0.9176`. The trained weights are therefore used, but this probe
  cannot separate “useful response” from “any learned scalar gate”.

A post-hoc subgroup read (not a pre-registered endpoint) shows where the ESOL Test difference is
generated. Molecules that actually use the fusion head (72 Val / 80 Test) favor RESP on Val
(+0.0504) and Test (+0.0770). The no-cut fallback molecules (41 Val / 33 Test), which bypass the
fusion head entirely, favor ADD on Test by −0.1375 and only weakly favor RESP on Val (+0.0292).
The whole-split Test gain (+0.0096) is the formal endpoint; this decomposition explains its
attenuation and warns that the ESOL comparison is carried by a small, selected subgroup.

## 3. Historical position (not replayed)

Historical references are the lower prior Val RMSE among the two reports named in the brief:
Lipo C1 `0.7052`, ESOL C1 `0.7958`. “NEAR” means Val ≤ 1.05× that reference by the frozen rule;
the GINE gap is reported separately and is not replaced by the C1 pass.

| task | model | Val RMSE | Test RMSE | source |
|---|---|---:|---:|---|
| Lipo | C1/L | 0.7052 | 0.7139 | `sq-pilot-small-v1` |
| Lipo | C1/L | 0.7069 | 0.7293 | `gnn_route_clarification_8h_v1` |
| Lipo | GINE/G | 0.6091 | 0.6456 | `sq-pilot-small-v1` |
| Lipo | GINE/G0 | 0.6161 | 0.6800 | `gnn_route_clarification_8h_v1` |
| ESOL | C1/L | 0.7958 | 0.9534 | `gnn_route_clarification_8h_v1` |
| ESOL | C1/L | 0.8686 | 0.8570 | `sq-pilot-small-v1` |
| ESOL | GINE/G0 | 0.8587 | 0.8902 | `gnn_route_clarification_8h_v1` |
| ESOL | GINE/G | 0.9353 | 0.9354 | `sq-pilot-small-v1` |

- Lipo RESP is better than historical C1 (0.6287 vs 0.7052; 0.89×) and better on Test (0.6760 vs
  0.7139). It remains behind the stronger historical GINE on both splits (GINE Val 0.6091–0.6161,
  Test 0.6456–0.6800; Lipo RESP Test 0.6760 is between the two GINE Test values). Lipo ADD is also
  below C1 (0.6762 / 0.7121), so the C1 pass is not unique to RESP.
- ESOL RESP is better than historical C1 (0.7278 vs 0.7958; 0.91×) and better than the C1 Test
  (0.8995 vs 0.9534), but roughly level with / slightly behind the stronger historical GINE on Test
  (0.8902; it is better than the other GINE’s 0.9354). ESOL ADD is also below C1 (0.7703 / 0.9091).
- These are application references with different recipes/features; the comparison is positional,
  not a controlled baseline replay. No historical baseline was retrained.

## 4. Response dictionary stage

The dictionary stage is post-hoc and frozen: it uses each task's best-Val RESP state, Train-only
inventory/frequencies, and replaces `a(S), q(S)` of a fragment by a convex combination of at most
`K=64` prototypes. Val/Test fragments receive `alpha` from their structure through the frozen model
(no labels). Three dictionaries are compared:

- `D_RESPONSE`: prototypes selected in the response matrix `R(S)=s(C_j,S)-s(C_j,D0)` over a
  16-environment Train panel (Frank–Wolfe, 64 steps).
- `D_LATENT`: farthest-first prototypes in the latent `k(S)` space.
- `D_FREQ`: most frequent Train words per `tau` class.

`delta Val` is `RMSE_D(Val) - RMSE_RESP(Val)`; the pre-registered “worth further study” reference
is `delta Val <= 0.01 * Train_std` and supported views ≥ 80%. Fidelity is the RMS change of the
frozen model's own scalar output when `a,q` are replaced (a self-fidelity check, not task fidelity).

| task | dictionary | words/candidates | exact-storage classes | support | fallback Val | Train RMSE (fid) | Val RMSE (fid) | Test RMSE (fid) | delta Val | dev pass |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Lipo | D_RESPONSE | 280/702 | 13/16 | 1.000 | 0.093 | 0.3499 (0.0177) | 0.6244 (0.0291) | 0.6771 (0.0255) | −0.0043 | yes |
| Lipo | D_LATENT | 280/702 | 13/16 | 1.000 | 0.093 | 0.3589 (0.0774) | 0.6266 (0.0553) | 0.6774 (0.0449) | −0.0021 | yes |
| Lipo | D_FREQ | 280/702 | 13/16 | 1.000 | 0.093 | 0.3491 (0.0145) | 0.6241 (0.0255) | 0.6783 (0.0227) | −0.0047 | yes |
| ESOL | D_RESPONSE | 120/169 | 10/11 | 0.996 | 0.363 | 0.3374 (0.0141) | 0.7385 (0.1123) | 0.9009 (0.0160) | +0.0108 | yes |
| ESOL | D_LATENT | 120/169 | 10/11 | 0.996 | 0.363 | 0.3360 (0.0046) | 0.7382 (0.1117) | 0.8982 (0.0150) | +0.0104 | yes |
| ESOL | D_FREQ | 120/169 | 10/11 | 0.996 | 0.363 | 0.3369 (0.0098) | 0.7387 (0.1126) | 0.8968 (0.0211) | +0.0109 | yes |

All six cells pass the pre-registered development reference (Lipo threshold 0.0121, ESOL 0.0207;
support 1.000/0.996). The compact dictionary therefore *enters the frozen prediction path* with a
small Test loss: Lipo Test 0.6771 vs RESP 0.6760 (+0.0011), ESOL Test 0.9009 vs 0.8995 (+0.0014)
for `D_RESPONSE`. But most classes are stored exactly: 13/16 Lipo and 10/11 ESOL classes use all
candidates. Compression is concentrated in the three Lipo `tau` classes with 442→64, 81→64 and
91→64 words (the frequent carbon, aromatic-carbon and nitrogen port classes) and the single ESOL
carbon class with 113→64 words. The aggregate 280/702 and 120/169 ratios overstate how much
structural generalisation is being tested.

The ESOL Val “loss” is one molecule. The worst Val fidelity change for every ESOL dictionary is
the same molecule (official row 1098; |Δ prediction| 1.077, label −2.92). Excluding it, Val RMSE is
0.7261 for `D_RESPONSE` vs 0.7293 for the frozen RESP model, i.e. the dictionary is slightly better;
the reported +0.0108 is not a broad degradation. Lipo's worst case is much smaller (row 1647,
|Δ| 0.41; excluding it `D_RESPONSE` Val 0.6161 vs RESP 0.6167).

**Does response-based selection preserve more prediction than latent/frequency?** No consistent
answer in this screen: on Lipo, `D_FREQ` is marginally best on Val and fidelity, and `D_RESPONSE`
only beats `D_LATENT`; on ESOL, all three are within 0.0007 Val RMSE and `D_FREQ`/`D_LATENT` have
the lower Test RMSE. `D_RESPONSE` has no reproducible advantage over the cheaper baselines. The
dictionary result is therefore “compact storage is feasible”, not “the response dictionary is the
useful structure”. This is the main reason not to upgrade the dictionary claim.

## 5. Real paired-edit diagnostic

The only real edit set available is the existing Lipo amide-difference catalogue (74 rows, 73 in the
official Train split, 0 in Val/Test). Official split membership was recomputed; the catalogue's own
split column was ignored because it disagrees with the official splits. On the 73 Train pairs:

| arm | paired-difference RMSE | paired-difference MAE | mean signed error |
|---|---:|---:|---:|
| RESP | 0.4745 | 0.3501 | +0.0944 |
| ADD | 0.3909 | 0.3137 | +0.0659 |

ADD is better on this small Train-only set. This is counter-evidence for task relevance, not a
refutation of the Lipo Val/Test signal: the support is 73 pairs, restricted to one chemistry and one
split, and the catalogue was not designed for this screen. It is nevertheless enough that the round
cannot claim real-task edit relevance, hence **`TASK_RELEVANCE_NOT_ESTABLISHED`**.

## 6. Labels

| scope | label | basis |
|---|---|---|
| Lipo | `CONDITIONAL_SIGNAL_S0` | Val gain +0.0475 ≥ 0.02, Test gain +0.0361 > 0, top-1 removal +0.0439 > 0 |
| Lipo | `NEAR_HISTORICAL_C1` | Val 0.6287 ≤ 1.05 × 0.7052; still behind historical GINE |
| ESOL | `CONDITIONAL_SIGNAL_S0` | Val gain +0.0425 ≥ 0.03, Test gain +0.0096 > 0, top-1 removal +0.0238 > 0 |
| ESOL | `SAMPLE_UNCERTAIN` | bootstrap interval crosses zero; top-5 removal is negative; one molecule carries 45% of the gain |
| ESOL | `NEAR_HISTORICAL_C1` | Val 0.7278 ≤ 1.05 × 0.7958; Test roughly level with the stronger historical GINE |
| global | `TASK_RELEVANCE_NOT_ESTABLISHED` | only 73 Train edit pairs; ADD better on their paired differences |
| all units | `HORIZON_LIMITED = false` | last-30 vs previous-30 deltas are far above threshold |

`CONDITIONAL_SIGNAL_S0` is a screening label, not a claim of a chemical causal law, a new method, or
cross-seed stability. No `BUDGET_NOT_RUN` or `ENGINEERING_BLOCKED` label is needed: all four units
completed.

## 7. Budget and provenance

Formal unit times (each on one A100-SXM4-40GB, process backend on `res`, node `a100-2`,
driver 550.163.01, torch 2.5.1+cu124, TF32 off):

| unit | pool | run id | seconds | A100·h |
|---|---|---|---:|---:|
| Lipo RESP | `res-gpu1` | `rd-lipo-resp-s0-20261120-130756-69e349c3` | 601.6 | 0.167 |
| Lipo ADD | `res-gpu0` | `rd-lipo-add-s0-20261120-130756-bed58ba2` | 1106.8 | 0.307 |
| ESOL RESP | `res-gpu1` | `rd-esol-resp-s0-20261120-132028-5515874d` | 128.0 | 0.036 |
| ESOL ADD | `res-gpu0` | `rd-esol-add-s0-20261120-132804-db9e17bf` | 300.5 | 0.083 |
| **formal total** | | | **2136.8** | **0.594** |

Including the GPU smoke and the two aborted attempts (stopped after <5 minutes to apply the
TF32-off/precision and horizon-rule fixes; no numbers used), the round is still well under
1 A100·h, below the 4 A100·h soft budget and 6 A100·h hard budget. Wall clock including setup was
well under the 2 h target.

Provenance:

- Lipo units executed from commit `945d9881080b…`; ESOL units from `8f1badf28391…`. The only
  difference between those commits is `code/rd_analyze.py`; the executed training files have
  identical committed SHA-256 hashes (`rd_run.py c5ab23b8…`, `rd_train.py 7bf87269…`,
  `rd_model.py 62672faf…`, etc., pinned per commit in `run_manifest.json`). The current
  `rd_run.py` differs from the executed version only by the start-commit capture
  (`a1497cbb…` working tree vs `c5ab23b8…` executed); training behaviour is unchanged.
- All four units share init-state hash `ccc8171d4cdc47da…` and the same per-dataset train-order
  hash (`lipo 361af4df…`, `esol 36f0a997…`), so within a task the arms differ only in the fusion
  operator and the resulting optimization trajectory.
- Protocol hash `861b778d…`; config `configs/rd_small_s0.json` SHA-256 `8d0fd9ae…`; official cache
  and fragment-cache SHA-256 values are pinned in `run_manifest.json`; fragment caches are
  `molesol fac644d0…` and `mollipo ba6e9a41…`.
- The Lipo ADD process started before the later deploy; its `summary.json` records the checkout
  commit at finish (`8f1badf2`), not the executed commit. `status.json` and `run_manifest.json`
  record the executed commit (`945d9881`) and the difference is only the analysis file. This is
  why the report cites the execution commit, not the summary's end-of-run checkout.
- Checkpoints, raw predictions, logs and train orders are not committed (see the track
  `.gitignore`); compact summaries, curves, tables and manifests are committed.

## 8. What this round cannot answer

- Cross-seed stability, or whether the Lipo/ESOL ordering survives a second initialization.
- Generalization to unseen fragment–environment combinations; the screen only recombines a fixed
  Train vocabulary/panel and does not test a held-out fragment class.
- Chemical causality or a mechanistic explanation of the interaction. The `w=0` and dispersion
  probes are consistency checks, not identification.
- Whether a compact dictionary helps **end-to-end training**. The dictionary stage replaces
  `a,q` post hoc in a frozen model; it does not train a compact dictionary or fine-tune around it.
- Whether `RESP` is specifically better than other bilinear/low-rank fusions, or whether any
  learned scalar gate would produce the same Lipo gain.
- Real-task edit relevance beyond 73 Train amide pairs.
- Larger OGB datasets and the full method-validation matrix; those were out of scope.

## 9. Recommendation

Proceed only with a narrow follow-up, not a method expansion:

1. Replicate the **Lipo RESP vs ADD** pair on 2–3 more seeds with the identical frozen recipe. Lipo
   is the clean signal: worse Train fit, better Val/Test, tight bootstrap interval, top-1 removal
   survives, and view dispersion is lower.
2. Treat ESOL as **conditional-uncertain**: repeat it, but pre-register the fusion-subgroup read and
   the fallback fraction, because one third of ESOL Val molecules bypass the fusion head and the
   Test gain is carried by the view-covered subgroup.
3. Do not promote the response dictionary yet. First show that `D_RESPONSE` beats `D_FREQ` and
   `D_LATENT` on a task where compression actually bites (more than 3/16 or 1/11 classes), and add
   a Val/Test real-edit set. Until then the honest dictionary claim is “compact storage survives a
   frozen replacement”, not “response-selected words are better words”.
4. Keep `TASK_RELEVANCE_NOT_ESTABLISHED` until a larger, split-respecting edit set is available;
   the current 73 Train pairs point the other way.

No un-authorized rescue runs (new seeds, wider models, extra ranks, new cuts, extra datasets) were
started. This report closes the authorized S0 screen.
