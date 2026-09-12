# SBCI Fit-Matched Generalization Frontier

**Track:** `ksvd` · **Protocol:** `sbci_fit_matched_generalization_frontier_v1`
**Module:** `tracks/ksvd/experiments/luyin16/zinc_sbci_fit_matched_generalization_frontier.py`
**Results:** `tracks/ksvd/results/sbci_fit_matched_generalization_frontier/`
**Tests:** `tracks/ksvd/tests/test_sbci_fit_matched_generalization_frontier.py` (29 pass)
**Official valid:** never loaded. **Official test:** never loaded.
**Gradient updates:** zero. Every number below is a `model.eval()` forward pass
over already-saved N3600 / I0 / T0 checkpoints, plus the deterministic
D3600/800/2000 views of the frozen official-train split.

**Final verdict: `FIT_DEPENDENT_FRONTIER_CROSSING` — the SBCI diagnostic branch
is closed for architecture inference.** At matched D3600 empirical training fit,
SBCI's 800 selection-generalization difference changes sign across fit anchors:
three anchors show SBCI materially worse (`D_k >= +0.003`), three anchors show
SBCI materially better (`D_k <= -0.003`). There is no single ordering, so
neither "capacity failure" (DSP) nor "wrong inductive bias despite adequate fit"
(SSOD) is justified. The 2000 internal probe was correctly **not** opened.

---

## 1. Motivation

The compact-v4-SBCI run at N=3600 seed0 is a clean sample-efficiency NO-GO:
SOUP probe MAE **0.182924** vs the reused `compact-v4-smallhead` baseline
**0.176633** (`delta_SE = -0.006291` SOUP, `-0.006215` RAW, DDR `-0.128`), with
a SOUP paired bootstrap 95% CI `[-0.012760, -0.000176]`. The question is no
longer *whether* SBCI lost the endpoint, but whether its learned function
generalizes systematically worse **at the same empirical training fit**.

## 2. Why this is the final SBCI diagnostic

The previous zero-training fit triage
(`tracks/ksvd/notes/sbci_fit_generalization_triage.md`) compared the two runs
only at their *selected endpoints* and at a handful of aggregate fit statistics
(`dF_raw`, `dF_soup`, `dF_min`, `dF_late`). Those aggregates mixed an
endpoint-fit confound with the actual function question and were sign-
conflicting, so it returned `FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE`. This audit
removes the empirical-fit-level confound directly: it compares the two models at
*equal* D3600 training fit using individual frozen trajectory snapshots. By the
pre-registered stop rule, whatever this audit returns ends SBCI-derived
architecture inference.

## 3. Existing SBCI NO-GO

`compact_v4_sbci_sample_efficiency.md` and
`records/decisions/decision-compact-v4-sbci-nogo-20261007.yaml` established the
NO-GO. That decision is **not** re-judged here. SBCI remains a NO-GO for the
sample-efficiency claim; this audit only asks whether its failure carries a
usable architecture-family signal.

## 4. Why previous fit triage was mixed

The triage measured four D3600 fit deficits that pointed in opposite directions:
RAW fit slightly better for SBCI (`dF_raw = -0.001938`), SOUP fit worse
(`dF_soup = +0.004554`), best-achieved `dF_min = +0.003042` just over the
adequate-fit gate, and late-10 median `dF_late = -0.000952`. Because the
comparison lived at each run's own selected endpoint, a genuine matched-fit
ordering could be hidden by the fact that the two runs had reached different
empirical fit levels. The fit-matched frontier removes exactly that confound.

## 5. Scientific question

$$
\boxed{ MAE_{\rm generalization} \mid MAE_{\rm train} \approx c }
$$

Does there exist a **stable** ordering between baseline and SBCI at equal
empirical D3600 training fit? Only a stable ordering authorizes a next
architecture family.

## 6. RAW snapshot principle

This audit deliberately does **not** use SOUP states to construct the frontier.
A soup is an average of checkpoints at different fit levels, so it has no clean
"trajectory fit coordinate"; averaging also changes the object under study.
The frontier therefore uses **individual frozen RAW trajectory snapshots**. The
final RAW/SOUP selected endpoints appear only as contextual markers.

## 7. Phase F / G800 / G2000 firewall

* **Phase F (fit only).** Reads D3600, train targets, snapshot weights, and
  optimizer-step metadata. Forbidden: stored 800 MAE, 800 predictions, 2000
  targets/predictions, probe error. Produces the eligible snapshots, the common
  fit interval, K=7 anchors, the matched pairs, and the SHA-256 lock.
* **Phase G800.** Runs only after Phase F is frozen; loads the 800
  selection-generalization set and evaluates the locked pairs.
* **Phase G2000.** Runs only if Stage 1 at 800 yields a pre-registered clean
  candidate verdict; uses the exact same locked pairs, no rematching.

The firewall is enforced in code (`_select_graphs` / `_probe_graphs` raise
`PhaseFirewallError` before unlock) and audited in
`phaseF_fit_only_lock.json`. Phase F records
`selection_800_loaded = false`, `probe_2000_loaded = false`,
`matching_used_train_only = true`.

## 8. Snapshot inventory

Counts are read from the real frozen manifests and on-disk snapshot
directories — never hardcoded:

| model family | saved snapshots | optimizer range | init hash matches |
|---|---:|---|---|
| `compact-v4-smallhead` | **106** | 57 … 6042 | yes (`63f2cecb…`) |
| `compact-v4-SBCI` | **102** | 57 … 5814 | yes (`f3479439…`) |
| total | **208** | | |

Protocol compatibility is **16/16**: identical D3600 index hash
(`3be9f7e1…`), identical 800 and 2000 index hashes, same optimizer recipe
(Adam, lr `1e-3`, wd `1e-5`, batch 128, L1, clip 5.0), same step budget
(13,680), same 57-step evaluation cadence, same patience (40), same I0/T0 seed
family, and the 20 shared tensors bit-identical (`max_abs_diff = 0.0`). Any
mismatch would have been `INVALID FRONTIER COMPARISON`.

## 9. Early-transient exclusion

The audit studies learned-function generalization, not random-init dynamics.
With `S_common = min(6042, 5814) = 5814`, only snapshots with
`optimizer_step >= 0.25 * 5814 = 1453.5` are eligible. This leaves **81**
baseline and **77** SBCI snapshots. The 25 % fraction was fixed before any
matching and was not changed afterwards.

## 10. Common training-fit overlap

Model-wise eligible D3600 train MAE ranges:

* baseline `[0.055960, 0.267877]`
* SBCI `[0.059002, 0.168083]`

The common overlap is therefore `L, U = [0.059002, 0.168083]` (width 0.10908).
Trimming 10 % off each end gives the frozen fit interval

$$
[L', U'] = [0.0699098,\ 0.1571750], \qquad U' - L' = 0.0872652 > 0.010 .
$$

## 11. Seven locked fit anchors

With `K = 7` equally spaced anchors on `[L', U']` (constructed from D3600
training fit only):

| k | c_k | baseline fit | SBCI fit | gap |
|---:|---:|---:|---:|---:|
| 0 | 0.0699098 | 0.0702688 | 0.0702693 | 0.0000005 |
| 1 | 0.0844540 | 0.0841392 | 0.0847239 | 0.0005847 |
| 2 | 0.0989982 | 0.0987839 | 0.0997952 | 0.0010113 |
| 3 | 0.1135424 | 0.1135389 | 0.1144077 | 0.0008688 |
| 4 | 0.1280866 | 0.1266381 | 0.1264636 | 0.0001745 |
| 5 | 0.1426308 | 0.1417531 | 0.1437185 | 0.0019654 |
| 6 | 0.1571750 | 0.1649479 | 0.1569509 | 0.0079970 |

Anchor 6 is invalid because the baseline trajectory has no eligible snapshot
near 0.15718 (its closest is 0.16495; `|diff| = 0.00777 > 0.002`).

## 12. Deterministic unique matching

For each family, a deterministic minimum-cost bipartite assignment
(`scipy.optimize.linear_sum_assignment`) matches the 7 anchors to 7 **distinct**
snapshots. The cost is `|F(snapshot) - c_k|`; ties are broken toward the **later
optimizer step** with a tiny deterministic epsilon (`1e-12`). No held-out
quantity (800 MAE, probe MAE, checkpoint-selection rank) participates. The
matched snapshots are unique within each family, and re-running the assignment
from the parquet reproduces the lock exactly.

## 13. Match-quality gates

* anchor proximity `|F - c_k| <= 0.002` for both families;
* cross-model fit gap `|F_base_k - F_SBCI_k| <= 0.002`;
* `K_valid >= 5`;
* `median_k |F_base_k - F_SBCI_k| <= 0.001`.

Observed: `K_valid = 6`, median gap `0.0008688`, max valid gap `0.0019654`.
The lock is **LOCKED**. The `fit_anchor_lock.json` and
`matched_snapshot_pairs.csv` were SHA-256-locked before any 800 access
(`lock_sha256 = 85fbd6cc…`, pairs `44d6a246…`).

## 14. 800 frontier result

800 is a **selection-generalization diagnostic set**, not a pristine holdout —
it was used for checkpoint selection during the original runs. It was opened
only after the fit lock. Per-anchor differences (`D_k = MAE_800(SBCI) -
MAE_800(baseline)`), equal-weight across the 6 valid anchors:

| k | D_k |
|---:|---:|
| 0 | **+0.006618** |
| 1 | **-0.003695** |
| 2 | **+0.008842** |
| 3 | **+0.021548** |
| 4 | **-0.003175** |
| 5 | **-0.010575** |

Aggregate: `D_mean = +0.003261`, paired-molecule bootstrap (B=2000, seed
20260912) 95 % CI `[-0.004861, +0.012669]`, median `D_k = +0.001722`,
`f_+ = 0.50`, `f_{<=0.002} = 0.50`. The CI includes zero. Note the bootstrap
resamples 800 **molecules**; the trajectory snapshots are fixed locked model
states, not independent observations.

## 15. Frontier crossing check

Material frontier crossing requires at least 2 anchors with `D_k >= +0.003`
and at least 2 anchors with `D_k <= -0.003`. Observed: **3** anchors with
`D_k >= +0.003` (k=0,2,3) and **3** anchors with `D_k <= -0.003` (k=1,4,5).
The frontier **materially crosses**. This is a direct sign that SBCI's relative
generalization at matched empirical fit depends on the fit level, so no single
"capacity failure" or "wrong inductive bias" label is justified.

## 16. Conditional 2000 confirmation

Because the 800 frontier already produced a material crossing,
`stage1_800_decision.json` set `authorize_2000 = false`. The 2000 internal probe
was **not** opened, and no `stage2_2000_*` placeholder files exist. This is the
pre-registered stop rule: a crossing at 800 is `INCONCLUSIVE` and does not
authorize further confirmation.

## 17. Matched-fit interpretation

At comparable D3600 training fit, SBCI is neither systematically worse nor
systematically better on the 800 set. Some matched fit states favor SBCI, some
favor the baseline, and the magnitude of the differences is large relative to
the mean (`D_k` ranges from `-0.0106` to `+0.0215`). The earlier endpoint
ordering (`+0.0063` worse SOUP probe) is a single selected fit state, not a
stable property of the learned function at equal fit. The 800 per-snapshot MAE
is itself strongly oscillatory, which is consistent with the crossing being
dominated by trajectory/selection variability rather than a stable architecture
effect.

## 18. Why this does not reopen SBCI

Matched-fit non-inferiority or crossing does **not** rescue SBCI. The official
sample-efficiency NO-GO stands: at N=3600 seed0 the selected SBCI endpoint is
significantly worse on the 2000 probe. This audit neither re-judges that result
nor authorizes a rescue. It only shows that the endpoint failure does not
identify a clean architecture-family cause.

## 19. DSP implication if worse

Had the 800 frontier given a wrong-bias candidate **and** the locked 2000 probe
confirmed it, the conclusion would have been
`SBCI_MATCHED_FIT_GENERALIZATION_WORSE`, authorizing DSP design only (no
training), and closing SBCI / SSOD / latent-basis sharing as the leading
premise. That branch was **not** taken.

## 20. SSOD implication if noninferior

Had the 800 frontier given a non-inferior/favorable candidate **and** the
locked 2000 probe confirmed it, the conclusion would have been
`SBCI_MATCHED_FIT_GENERALIZATION_NONINFERIOR`, authorizing SSOD design only
(no training). That branch was **not** taken.

## 21. Stop rule if mixed

The observed crossing is exactly the pre-registered stop condition. No SSOD, no
DSP, and — per the budget discipline — no fourth SBCI diagnostic: no fit-bin
subdivision, no 5/9/11 anchors, no tolerance sweep, no subgroup analysis, no
seed1 SBCI, no K rescue. Future architecture ideas must come from independent
principles, not post-hoc interpretation of SBCI's failure.

## 22. Official-valid / test lock

Official valid was never loaded. Official test was never loaded. The reused
environment is the frozen official-train 7200/800/2000 split; the module's
firewall blocks the official valid/test extraction paths. No official test MAE,
prediction, subgroup or checkpoint exists.

## 23. Final verdict

**`FIT_DEPENDENT_FRONTIER_CROSSING`** — SBCI failure does not identify a next
architecture family. `authorized_for_design = false`, `family = null`,
`full_training_authorized = false`,
`sbci_branch_closed_for_architecture_inference = true`.

```
baseline N3600 seed0 SOUP probe   0.176633   (soup train 0.048170, soup 800 0.208414)
SBCI     N3600 seed0 SOUP probe   0.182924   (soup train 0.052724, soup 800 0.214427)
valid fit anchors / K             6 / 7
median cross-model fit gap        0.0008688
800 D_mean (SBCI - baseline)      +0.003261
800 paired bootstrap 95% CI       [-0.004861, +0.012669]
800 material frontier crossing    YES (3 anchors >= +0.003, 3 anchors <= -0.003)
2000 probe                        NOT opened (correctly)
```

## Provenance: convergence ambiguity (not re-judged)

The prior triage recorded a conflict between the literal pre-registered
convergence-slope rule (which would have fired `OPTIMIZATION_AMBIGUOUS`) and the
statistical interpretation (the negative slope is indistinguishable from
trajectory noise, and SBCI's best-so-far training fit had plateaued). This audit
does not modify any prior record. The fit-matched design deliberately reduces
dependence on endpoint convergence: it compares states the two trajectories
**actually visited** at the same empirical D3600 training fit, rather than
inferring a mechanism from where each run happened to stop.

## Q1–Q20

* **Q1** Saved snapshots: baseline **106**, SBCI **102** (208 total).
* **Q2** Common final-step threshold: `0.25 * 5814 = 1453.5`.
* **Q3** Eligible snapshots after burn-in: baseline **81**, SBCI **77**.
* **Q4** Common train-fit overlap: `[0.0590017, 0.1680832]`.
* **Q5** Trimmed interval: `[0.0699098, 0.1571750]` (width `0.0872652`).
* **Q6** Seven fit anchors: `0.0699098, 0.0844540, 0.0989982, 0.1135424,
  0.1280866, 0.1426308, 0.1571750`.
* **Q7** Per-anchor baseline / SBCI train fit: see §11 table.
* **Q8** Max cross-model fit gap: **0.0079970** (anchor 6, invalid).
* **Q9** Median cross-model fit gap: **0.0008688**.
* **Q10** Valid anchors: **6**.
* **Q11** 800 `D_k`: `[+0.006618, -0.003695, +0.008842, +0.021548, -0.003175,
  -0.010575]`.
* **Q12** 800 `D_mean`: **+0.003261**.
* **Q13** 800 paired-bootstrap CI: **[-0.004861, +0.012669]**.
* **Q14** 800 material frontier crossing: **YES**.
* **Q15** 800 verdict: **`MATERIAL_FRONTIER_CROSSING`** (not a wrong-bias or
  non-inferior candidate).
* **Q16** Authorize 2000: **No**.
* **Q17** 2000 `D_k` / mean / CI: **not opened** (`null`).
* **Q18** 800 vs 2000 ordering consistency: **n/a** (2000 not run).
* **Q19** Matched-fit result supports: **neither DSP nor SSOD**.
* **Q20** Final architecture authorization: `authorized_for_design = false`,
  `family = null`, `full_training_authorized = false`.

## Artifact map

* Locks: `audit_protocol_lock.json`, `protocol_compatibility.json`,
  `snapshot_inventory.json`, `phaseF_fit_only_lock.json`,
  `fit_anchor_lock.json`, `match_quality.json`.
* Fit: `train_fit_inventory.parquet`, `fit_inventory.summary.json`,
  `train_fit_overlap.json`, `matched_snapshot_pairs.csv`.
* Stage 1: `stage1_800_per_anchor.csv`, `stage1_800_per_molecule.npy`,
  `stage1_800_bootstrap.json`, `stage1_800_decision.json`.
* Decision: `endpoint_context.json`, `final_decision.json`,
  `next_architecture_family.json`, `answers_q1_q20.json`, `integrity_tests.json`.
* Figures: `figures/figure1_fit_matched_frontier.png`,
  `figures/figure2_anchor_differences.png`,
  `figures/figure3_aggregate_matched_fit.png`.
