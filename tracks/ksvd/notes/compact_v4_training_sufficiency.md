# Compact-v4 Training Sufficiency & Protocol Calibration

> **Question.** Is the historical canonical 60-epoch compact-v4-hinge training
> protocol substantially under-training the existing architecture, or is the
> observed +0.00909 canonical late-readout *continuation* gain an
> optimisation-regime (fresh Adam / batch 512 / weight-decay 0) effect rather
> than a training-horizon effect?
>
> **Verdict (seeds 0/1/2/3, frozen test authorised).** **HISTORICAL UNDERTRAINING
> CONFIRMED.**
> Simply extending the *unchanged* canonical optimisation regime
> (Adam lr 1e-3, batch 128, wd 1e-5) from 60 to 240 epochs moves official-valid
> MAE from **0.170066** to **0.146420** on seed 0 (+0.02365) and from
> **0.163167** to **0.149332** on seed 1 (+0.01383); 4-seed paired valid gain
> **+0.02225**, 4/4 positive. Batch 512 and weight decay 0 do **not** provide
> an independent gain (both make things worse); the continuation signal was a
> horizon / early-stopping-rule effect, not an optimisation-regime switch. The
> frozen one-shot official test confirms it: optimized test mean
> **0.125179 ± 0.003374** vs historical **0.136885**, paired test gain
> **+0.011706** (4/4 positive). The new baseline is **longer single-stage
> compact-v4-hinge training** — no two-stage protocol and no scheduler.

Module: `experiments/luyin16/zinc_compact_v4_training_sufficiency.py`
Results: `results/compact_v4_training_sufficiency/`
Tests: `tests/test_compact_v4_training_sufficiency.py` (12 tests)

---

## 1. Motivation

Every architecture / representation / statistic route in the track terminated
NO-GO, but the strongest surviving local signal was a **late-readout
adaptation** effect: taking the canonical selected compact-v4-hinge checkpoint
and running a *full-model* continuation (fresh Adam, lr 1e-3, batch 512,
wd 0, 94 epochs) improved official valid from `B0 = 0.170066` to
`C = 0.160976` (`Delta_C = +0.00909`;
`results/canonical_late_readout_adaptation/`).

Before attributing that +0.00909 to any *representation/architecture* property,
we must rule out the mundane explanation:

> the historical 60-epoch protocol simply stopped training too early.

If so, every future architecture candidate would otherwise be compared against
an **under-trained** baseline.

## 2. Historical protocol (reconstructed, Stage 0)

Read from the committed canonical config
`configs/luyin16/zinc_compact_v4_topology_hinge.yaml` (sha256
`b9205b05…`) and the seed-0 run `20260909-194445-182c7021`:

| field | value |
|---|---|
| optimizer | Adam |
| learning rate | 1e-3 |
| weight decay | 1e-5 |
| batch size | 128 |
| max epochs | 60 |
| patience | 12 |
| scheduler | none |
| grad clip | 5.0 |
| loss | L1 / MAE |
| checkpoint selection | best official-valid MAE |
| seed | 0 |
| parameters (selection phase) | 99,613 |
| historical seed-0 valid | **0.170066** (best epoch 53) |
| historical seed-1 valid | **0.163167** (best epoch 48) |

`historical_protocol.json`, `historical_learning_curve.csv`.

## 3. Why architecture experiments were paused

The task is **optimization / training-protocol sufficiency**, not
representation information, architecture capacity, higher-order structure,
pooling, or head-family selection. Only training-protocol variables were
searched: horizon, patience, batch size, weight decay, learning rate (and one
scheduler family only if needed). Tokenizer, patch representation, pair
encoder, centre update, pooling, topology, head, loss and target are frozen.

## 4. Continuation signal (the anchor)

`0.160976` is carried as an **optimisation-headroom anchor**, not a benchmark
candidate: a new from-scratch protocol that reaches ≈0.160–0.163 would already
"absorb" the continuation signal. Note the anchor came from a *single
end-of-budget evaluation* with **no** validation selection.

## 5. Search design (small structured search)

- Stage 1A: geometric horizon ladder 60 / 120 / 240 at fixed
  Adam lr 1e-3, batch 128, wd 1e-5 (seed 0).
- Stage 1B (conditional): 2×2 batch {128, 512} × wd {1e-5, 0} at 240 epochs
  (seed 0).
- Stage 2 (conditional): lr {3e-4, 1e-3} on the winning regime.
- Stage 3: seed-1 replication, then optional seed 2/3.
- No Optuna / random / Bayesian search; only 5–10 interpretable protocols.
  official **test is never accessed** during any search stage.

`search_preregistration.json`.

## 6. Horizon audit (Stage 1A, seed 0)

| protocol | max ep | patience | best epoch | train@best | valid@best | epochs run | optimizer steps | wall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A0 historical | 60 | 12 | 53 | 0.150202 | **0.170066** | 60 | 4,740 | 291 s |
| A1 moderate | 120 | 24 | 74 | 0.132462 | **0.161252** | 98 | 7,742 | 470 s |
| A2 long | 240 | 40 | 169 | 0.084949 | **0.146420** | 209 | 16,511 | 1,012 s |

- A0 reproduced the historical result **bit-exactly** (`best_valid =
  0.17006561887910357`, best epoch 53, full valid curve max diff `0.0`).
- `A0 − A2 = +0.02365`, best epoch `169 > 60` ⇒ **H1 (clear undertraining)**.
- A1 already matches the continuation anchor (0.161252 vs 0.160976) and A2
  goes *far below* it (0.146420). Horizon **alone** explains and exceeds the
  continuation gain ⇒ **H2**.
- `H3 (longer still stalls)` false.

Stage gate: A2 explains the continuation gain, so a batch/wd **reset is not
required**; Stage 1B was nevertheless run (cheap, answers Q7/Q8) and confirms
the point (§7).

### 6.1 Selection-noise caveat (important)

The batch-128 valid curve is very noisy (late-phase per-epoch std ≈0.011), and
"best valid" over a longer horizon draws more samples. The honest picture:

| protocol | best valid | late-phase median valid | epochs < 0.15 |
|---|---:|---:|---:|
| A0 (60 ep) | 0.17007 | 0.19377 | 0 / 60 |
| A1 (98 ep) | 0.16125 | 0.17494 | 0 / 98 |
| A2 (209 ep) | 0.14642 | 0.16701 | 2 / 209 |
| B1 (b512, 240 ep) | 0.15392 | 0.16443 | 0 / 240 |
| B3 (b512/wd0, 240 ep) | 0.15115 | 0.16695 | 0 / 240 |

The *whole late-phase distribution* shifts down with horizon (0.194 → 0.175 →
0.167), so the horizon effect is real; but the single best epoch is partly a
lucky noise dip. The smoother batch-512 regimes have a floor ≈0.151–0.154.
The frozen one-shot official test (§14) is the ultimate adjudicator.

## 7. Batch / weight-decay decomposition (Stage 1B, seed 0)

240 epochs, patience 40, lr 1e-3:

| protocol | batch | wd | best valid | best epoch | train@best | epochs | steps |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | 128 | 1e-5 | **0.146420** | 169 | 0.084949 | 209 | 16,511 |
| B1 | 512 | 1e-5 | 0.153920 | 204 | 0.106572 | 240 | 4,800 |
| B2 | 128 | 0 | 0.159668 | 79 | 0.124131 | 119 | 9,401 |
| B3 | 512 | 0 | 0.151145 | 201 | 0.112573 | 240 | 4,800 |

`B0→B1 = -0.007500` (batch 512 worse), `B0→B2 = -0.013248` (wd 0 worse),
`B0→B3 = -0.004725` (both worse). **Q7/Q8: neither batch 512 nor wd 0 gives an
independent gain.** The canonical B0 regime remains the best. No step-matched
batch control was needed (batch 128 dominates on both epochs and updates).

## 8. Learning-rate calibration (Stage 2, seed 0)

Winning regime = B0 (batch 128, wd 1e-5, 240/40):

| protocol | lr | best valid | best epoch | epochs |
|---|---:|---:|---:|---:|
| S_lr1e-3 | 1e-3 | **0.146420** | 169 | 209 |
| S_lr3e-4 | 3e-4 | 0.146615 | 232 | 240 |

3e-4 is essentially tied (Δ = 0.0002) but its best epoch (232/240) sits at the
horizon boundary, i.e. it is still improving when the budget ends; 1e-3
converges cleanly at epoch 169. lr 1e-3 is retained and no scheduler is
introduced (the best epoch is not near the horizon, so a fixed LR is not
obviously limiting convergence).

## 9. Learning-curve interpretation

- Historical: validation minimum at epoch 53, then *rises* (epoch 60 valid
  =0.192). With `patience 12` and `max_epochs 60` the run simply stopped at the
  budget — it never observed the second, far-lower basin that appears around
  epoch 74 and later.
- A1 (patience 24) finds it at epoch 74 (0.16125); A2 (patience 40) finds the
  deeper floor (0.14642 at epoch 169). Early stopping itself fires only *after*
  the better basin (A2 stops at epoch 209 = best 169 + 40).
- Train L1 keeps falling monotonically the whole time (0.150 → 0.085 → 0.072);
  the valid minimum is genuinely reached later, not at the train-loss floor.

## 10. Training saturation

- A2 best epoch 169 / 240 = 70 %, i.e. **not** pinned to the horizon; fewer than
  the 10 % boundary distance (24 epochs) remains. No horizon extension needed.
- B1/B3 (batch 512) do hit 240 without early stop, but their late-phase valid
  is flat/slightly rising, so they are not obviously truncated.
- A2's late-phase valid shows a clear plateau/rise after epoch 169.

## 11. Seed-0 candidate

`candidate_protocol.json`:

- **P\*** = A2/B0: Adam, lr 1e-3, batch 128, wd 1e-5, max_epochs 240,
  patience 40, no scheduler, L1.
- seed-0 valid **0.146420**, best epoch 169, 16,511 optimizer updates,
  **Δ vs historical = +0.023645**.

## 12. Seed-1 replication

`seed1_replication.json` (candidate frozen before this run):

| seed | historical valid | optimized valid | gain |
|---:|---:|---:|---:|
| 0 | 0.170066 | 0.146420 | **+0.023645** |
| 1 | 0.163167 | 0.149332 | **+0.013834** |

Paired mean gain **+0.018740**. Gates: seed0 ≥ +0.003 ✓, seed1 > 0 ✓,
mean ≥ +0.003 ✓ ⇒ **TRAINING PROTOCOL GO**. No retuning on seed 1.

## 13. Additional seeds (2, 3)

Both seed gains are large (> +0.004) and same-direction, so seeds 2 and 3 were
run *for confirmation / variance estimation only* (candidate already frozen;
results cannot change it).

| seed | historical valid | optimized valid | gain | best epoch (opt) |
|---:|---:|---:|---:|---:|
| 0 | 0.170066 | 0.146420 | **+0.023645** | 169 |
| 1 | 0.163167 | 0.149332 | **+0.013834** | 104 |
| 2 | 0.174149 | 0.151455 | **+0.022694** | 127 |
| 3 | 0.170421 | 0.141595 | **+0.028826** | 161 |

**4-seed summary** (`valid_summary.csv`): historical valid mean **0.169451 ±
0.003965**; optimized valid mean **0.147201 ± 0.003697**; paired mean gain
**+0.022250**, 4/4 positive. All optimized best epochs lie beyond the
historical 48–60 epoch window.

## 14. Final protocol lock

`final_training_protocol_lock.json` freezes the exact config, seed policy,
tokenizer/topology fingerprints, parameter count and checkpoint-selection rule
before any test access. It is not modified after test results are seen.

## 15. Official test (frozen, one-shot)

Authorised only after `final_training_protocol_lock.json` existed and
`final_decision.json` read `HISTORICAL_UNDERTRAINING_CONFIRMED`. Each confirmed
seed's **valid-selected checkpoint** was evaluated on official test **once**;
no test-based selection, early stopping, tuning or train+valid refit.

| seed | optimized valid | **optimized test** | historical valid | historical test | test gain |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.146420 | **0.126512** | 0.170066 | 0.133901 | +0.007389 |
| 1 | 0.149332 | **0.127506** | 0.163167 | 0.132006 | +0.004500 |
| 2 | 0.151455 | **0.127327** | 0.174149 | 0.144614 | +0.017287 |
| 3 | 0.141595 | **0.119371** | 0.170421 | 0.137019 | +0.017648 |

- optimized valid **0.147201 ± 0.003697**; optimized test **0.125179 ± 0.003374**.
- historical test mean **0.136885**; **paired test gain +0.011706**, 4/4 positive.
- Honest reporting: the valid gain (+0.0223) is roughly 2× the test gain
  (+0.0117), i.e. **part of the valid-selected gain was validation-selection
  bias**, but a real, substantial official-test improvement remains — the
  hypothesis is not a validation-only artifact.

## 16. Compute cost

`compute_accounting.csv` / `compute_accounting.json`. Selection-phase
parameters 99,613 for every run (unchanged architecture). The optimized
protocol trains 144–209 epochs (11,376–16,511 updates) instead of 60 (4,740) —
a ≈2.4–3.5× optimisation-compute increase for the per-seed improvements.
Parameter efficiency is unchanged; **optimisation compute increases** and is
reported separately.

## 17. What this changes about previous experiments

The previous NO-GO diagnostics (pair endpoint association, centre-incidence
co-occurrence, triadic binding, function-basis accessibility, conditional
readout sufficiency, head-family) were all run on the **historical
compact-v4-hinge latent manifold / historical training regime**. They remain
valid *for that regime*. The optimized representation **drifts materially**, so
those audits describe a **not-yet-optimised** manifold and should be framed as
such — but they are not automatically invalidated.

### 17.1 Representation drift (descriptive)

`representation_drift.json` (seed 0, 1000 fixed valid probes):
`normalized_L2_drift = 0.314`, `mean_cosine_similarity = 0.950`. The optimized
training moves the graph representation `R` materially in magnitude while
keeping its direction largely aligned. A stronger baseline must therefore be
used for any future architecture claim.

## 18. What is and is not proven

**Proven (within seed 0/1/2/3):** the 60-epoch protocol materially
under-trained compact-v4-hinge; the gain comes from horizon/early-stopping, not
from a batch/wd/fresh-optimizer regime switch; the continuation anchor is
absorbed by simple longer training.

**Not proven:** that longer training is optimal in general; that the full
valid-selected gain (+0.022) transfers one-for-one to test (it does not — the
test gain is +0.0117, so ≈ half of the apparent valid gain was
validation-selection bias); that any representation/architecture route is now
open.

## 19. Final optimized baseline

From-scratch **single-stage longer canonical compact-v4-hinge**:
Adam lr 1e-3, batch 128, wd 1e-5, max_epochs 240, patience 40, no scheduler,
L1, valid-selected checkpoint. 4-seed optimized valid **0.147201 ± 0.003697**
and frozen test **0.125179 ± 0.003374** (historical: valid 0.169451 ± 0.003965,
test 0.136885). All future architecture claims must compare against **this**
stronger baseline rather than 0.170.

## 20. Next research direction

Re-run the *decisive* representation-diagnostic probes against the optimized
latent manifold before opening any new architecture family, because a stronger
baseline can change which discarded statistic becomes task-relevant. Two-stage
full-model optimisation is **not** retained (longer single-stage already
explains the gain).
