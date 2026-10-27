# PROTOCOL — zinc-statistical-operator-closure-v1 (SOC-v1)

Status: **pre-registered before any GPU training in this round.**
Round directory: `tracks/structural-sufficiency/zinc_statistical_operator_closure_v1/`
Track: `structural-sufficiency`
Date: 2026-10-03 (UTC)

## 1. Question and target

On the official PyG ZINC subset (10,000 train labels, raw-`y` MAE), can a
**composable statistical operator** — an explicitly iterative, weight-tied
moment-closure computation over the bond/junction incidence graph — beat the
frozen DCC-L champion and move the official test MAE toward **< 0.060**?

The primary deliverable is a **single model**, trained on the official 10,000
labels, evaluated once on the official test split. Ensembling, test-time
adaptation, extra labels and label-derived inference features are out of scope.

## 2. Why this mechanism (evidence carried into the design)

The following facts are from the frozen prior rounds (see the reports cited in
`notes/evidence.md`):

* DCC-L (494,193 params) is the champion: official-valid 4-seed mean
  **0.075990**, official-test 4-seed mean **0.069400**. It is one-shot: no
  recursive local message passing.
* The residual error is dominated by **unseen radius-2 environments**
  (81.6% of total error), not by rings or seed noise; it is dispersion /
  heteroscedasticity, not a mean shift (`zinc_dccl_bottleneck_v1`).
* Adding a *global* attention / relation branch (`DBT`, `BAMF`, `COC`, `BOSI`,
  `native-relation-v1`) costs 0.004–0.006 MAE; the optimization cost of extra
  branches, not the absence of a mechanism, is the recurring failure mode.
* A frozen-reader audit (`FTA-v1`) shows 77% of the 4k→8k sample-size gain is
  re-extractable from the frozen 513-d DCC-L representation by a 33k-param
  reader. Operator changes are therefore measurable at the head level before
  full training.
* Two CPU diagnostics were run **before** this protocol (no GPU, train labels
  only):
  1. explicit radius-1/radius-2 composition counts added to frozen 513-d `z`
     improve a small head by only ~0.008–0.013 MAE, and the permutation control
     explains most of it; the real correspondence effect is ~0.002–0.003.
  2. naive alpha-blended iteration on **frozen** encoders degrades Monitor MAE
     monotonically (0.13384 → 0.13400 → 0.13512 → 0.16317 → 0.43316).
     Conclusion: iteration must be **trained end-to-end**, not bolted onto a
     frozen trunk.

SOC-v1 therefore adds the *smallest possible* end-to-end iterative operator on
top of the frozen champion, initialised to be numerically identical to DCC-L,
with zero new global mixing and no relation partition.

## 3. Architecture (frozen for this round)

Base: the exact frozen DCC-L `DualChannel` (P3 encoders → group composer → local
composer → 513-d reader). New module: `MomentClosure`, one shared operator
applied **K = 3** times.

For each object type (atom, bond, junction), with state width 96:

```
group1, group2 =
    atom      : (incident bond states, empty)
    bond      : (two endpoint atom states, incident junction states)
    junction  : (three patch atom states, two incident bond states)

m1 = mean(group1);  s1 = mean(group1^2);  c1 = count(group1)
m2 = mean(group2);  s2 = mean(group2^2);  c2 = count(group2)

x = [ h, m1, s1, m2, s2, log1p(c1), log1p(c2), role_embedding(8) ]
u = MLP( LayerNorm(x) -> 24 -> 48 -> 96 )
h' = h + sigmoid(gate_role) * u
```

All three object types are updated **simultaneously** (Jacobi step) from the
same pre-step states. The **same** projection and core MLP are used for all three
roles and all K iterations; the only role-specific parameters are an 8-d role
embedding and a scalar gate. There is no attention, no pairwise edge network,
no global token, and no per-iteration parameter set. This is a learned,
weight-tied fixed-point computation of explicit first/second moments; it is
honestly **iterative state propagation**, not "no message passing".

Initialisation: `gate_bias = -5` → `sigmoid = 0.006693`. The backbone tensors are
copied bit-for-bit from the frozen DCC-L initial state; the closure is the only
new parameter block. Forcing `gate_bias = -30` reproduces DCC-L predictions
exactly (measured max abs difference **0.0** on 16 real molecules).

Parameter counts (measured, CPU):

| arm | readout | parameters | delta vs DCC-L |
|---|---|---:|---:|
| DCC-L (frozen anchor) | — | 494,193 | — |
| **R** | final state `(a^K,b^K,w^K)` | **512,888** | +18,695 (+3.78%) |
| **T** | time mean over `t=0..K` | **512,888** | +18,695 (+3.78%) |
| **CTRL** | no closure, reader 256→274 | **513,021** | +18,828 (+3.81%) |

CTRL is the parameter-matched capacity control. Its extra reader units are dead
at initialisation (zero outgoing columns), so it also starts at DCC-L.

## 4. Data, recipe and selection

* Screen: BJOC `fit8k` (8,000 train rows), Monitor 1k for checkpoint selection,
  47,400 updates, official valid evaluated every 20 epochs for description only.
  Locked and official test are **not loaded**.
* Full: all 10,000 official-train rows, official-valid 1k for checkpoint
  selection, same 47,400 updates. Official test is **not loaded by training**.
* Normalisation: fit-row-only statistics (the frozen `fit_only_norm`). The full
  run uses all 10,000 train rows.
* Recipe: exactly the frozen champion: AdamW, lr 1e-3 → 1e-5 cosine, 500-step
  warmup, wd 1e-5, batch 128, clip 5, raw-`y` L1, 600 epochs = 47,400 updates.
* SSL init: the frozen masked-SSL P3 encoders `s0`/`s1`, identical to DCC-L.
* No dropout, no EMA, no SWA, no seed soup, no early-stop-on-test.

## 5. Arms and gates (this is the decision rule)

### 5.1 Screen (Fit8k, 2 seeds: 0, 1)

Frozen DCC-L anchors at the same rows/recipe:

| seed | Monitor MAE |
|---|---:|
| 0 | 0.12315087952744216 |
| 1 | 0.1277715681968257 |
| mean | 0.12546122386213394 |

Run **R** and **T** with seeds 0 and 1. **Primary gate** for an arm:

```
seed_mean(Monitor) <= 0.1215          (>= 0.004 better than the L mean)
AND Monitor(seed0) < 0.12315088
AND Monitor(seed1) < 0.12777157
```

If neither arm passes, **stop the round** and report the negative result. No
rescue by changing K, rank, gate bias, readout, dropout or recipe after seeing
Monitor. If exactly one arm passes, that arm advances; the other is closed.

If an arm passes, run **CTRL** with the same two seeds (same rows/recipe). CTRL
must not beat the passing arm by more than 0.001 on the seed mean; otherwise the
gain is attributable to capacity, and the round is reported as such (not a
mechanism win).

### 5.2 Full 10k confirmation

Take the passing arm (and, if needed, CTRL) to all 10,000 train rows, 2 seeds,
checkpoint selection on official valid. **Full gate**:

```
seed_mean(official_valid) <= 0.0720
AND official_valid(seed0) <= 0.0750
AND official_valid(seed1) <= 0.0750
```

The DCC-L 4-seed official-valid mean is 0.075990. A pass is a >= 0.004 seed-mean
improvement. If the full gate fails, **do not read official test**; report the
valid result as a negative / near-miss.

### 5.3 Terminal test read

Only if the full gate passes: freeze both seed states and the selection rule,
then read official test **once**. Report both seeds and the seed mean. The
success target is:

```
single-model official test MAE < 0.060
```

If the test result is >= 0.060, the round is a **negative/near-miss** for the
0.060 target even if it beats DCC-L; no post-hoc tuning on test is permitted.

## 6. Budget

| item | plan | cap |
|---|---:|---:|
| CPU diagnostics + gates | 0 A100·h | 0 |
| GPU smoke (1 card, 64 updates) | 0.02 | 0.1 |
| Screen R/T × 2 seeds | 2.4 | 3.0 |
| CTRL × 2 seeds (conditional) | 1.2 | 1.5 |
| Full 10k × 2 seeds (conditional) | 1.2 | 2.0 |
| Test scoring | 0.02 | 0.1 |
| **round total** | **4.8** | **16.0** |

Stop early if the screen fails. Every job (smoke, failed, retried, formal) is
recorded in `runs/` and `notes/ledger.md`.

## 7. What would count as evidence for / against

* **For**: the screen and full gates pass, CTRL does not explain the gain, and
  the terminal test is < 0.060.
* **Against**: the screen fails (the operator does not improve held-out
  accuracy at matched data/recipe), or the gain is reproduced by CTRL (capacity,
  not closure), or the full-10k valid gate fails (the screen gain does not
  survive the label-count change).
* **Not evidence**: train-loss improvements, Monitor-only gains that vanish on
  official valid, or a test read without the pre-registered full gate.

## 8. Honesty clause

The closure is a learned iterative state-propagation mechanism. It is not a
one-shot feature transform and it is not "message passing-free". Any report must
state: K=3 shared iterations, Jacobi updates, first/second moments only, no
attention, no global mixing, +18,695 parameters, and the exact parameter-matched
control. The round must not claim a "new class of model" from a single seed or a
single split.
