# REPORT — zinc-directed-bond-transformer-v1 (DBT)

Track `structural-sufficiency`, round `zinc-directed-bond-transformer-v1`.
Design and pre-registration: `DESIGN.md`; recipe: `configs/dbt_v1.json`;
checks: `analysis/cpu_checks.json`; curves: `analysis/curves.png`;
frozen test read: `analysis/test_freeze.json` + `analysis/test_report.json`.

**Status: complete.** One architecture, one seed-0 trajectory, one declared
horizon (1,500 epochs / 235,500 updates), best-valid selection on the full
1,000-molecule valid split, and one descriptive read of the official ZINC test
split with that frozen checkpoint.

---

## 0. One-paragraph verdict

The directed-bond Transformer is a **single model of 452,523 parameters** —
shared P3 atom/bond/junction encoders (train-only masked-SSL initialisation,
fine-tuned), a static directed-bond token that keeps source/target roles apart,
four layers of full-graph relation-conditioned attention whose relation is the
exact simple-path count and distance of the four endpoint blocks of two
directed bonds, and one amplitude-preserving reader. It reaches **valid
0.077768 (epoch 949)** and **test 0.069401**. That is a **wash against P3B on
test** (Δ = −0.00046, paired 95 % CI [−0.0053, +0.0044]) and **worse than L on
test** (+0.00314, CI [−0.0023, +0.0084]); on valid it is 0.0015 behind P3B and
0.0008 ahead of L, both inside noise. **The single-model test target
[0.050, 0.060) was not reached**, and nothing here shows that the
directed-bond-role hypothesis buys generalisation at this scale and recipe.
The one notable observation is diagnostic, not a result: the valid gap to P3B
is dominated by a single extreme-label molecule (row 172) on which DBT is much
worse, and on the other 999 valid molecules DBT is better than both references
while the test contrast is a tie.

---

## 1. What was actually built (and what is new relative to earlier trunks)

One model, one forward pass, no ensemble / router / concatenated twin / teacher
/ distillation. All weights are inference weights; nothing is frozen.

| stage | computation | parameters |
|---|---|---:|
| shared local encoders | P3 `atom_encoder`, `bond_encoder`, `junction_encoder`; seed-0 **train-only masked-SSL** init (`pretrain_encoder_s0.pt`, sha256 `006fabc1…`), fine-tuned | 202,048 |
| static directed-bond token | `LN(InitMLP([a_u, a_v, b_uv, J_src, J_dst, log1p n_src, log1p n_dst]))`, width 64; `J_src` = mean of the real junction states centred on the **source** that pair the target with each other neighbour; roles concatenated, never summed | 37,504 |
| relation pre-encoder | 65-dim raw relation → 32 → 24 (shared by all layers) | 3,098 |
| full-graph trunk | 4 × pre-LN blocks, width 64, 4 heads, FFN ×2; `α = softmax(q·k/√d + B(r))`, `value = V(h)·(1+tanh(G(r))) + U(r)`; all directed objects of a molecule, padding/cross-graph masked | 147,088 |
| static count injection | shared `4→64` (log source degree, log target degree, log atoms, log bonds) | 320 |
| reader | `[0.5Σ h_e, 0.5Σ h_e², Σ a_v, Σ a_v², global33, counts] → 357→128→128→1` | 62,465 |
| **total** | | **452,523** |

Differences that matter relative to this line's earlier work:

* the object is the **directed physical bond**, not an atom pair (A), not a
  junction-summed bond (P3/L/C), not a rooted induced path (J), not a whole-graph
  trace (B);
* the atom-pair path counts exist **only as a relation between two bond
  objects** — they are never pooled into a per-root or per-graph bag first;
* the relation enters the **read content** (`G` multiplicative gate and `U`
  additive term per channel), not just the logits (STA/ZSRT put structure in the
  bias);
* interaction is **four full-graph layers** over all directed bonds, with no
  endpoint-sharing mask and no hidden-state propagation along the local
  adjacency.

The path relation is a **bounded exact** count: all simple paths with 1–12 edges
are enumerated; nothing is sampled or truncated per graph. Max count over
train+valid+test is 12 (uint8 storage is safe), and the full 10k-train cache
costs 16 s of one CPU core, so no approximation or `L` change was needed.

## 2. Main table

Valid is the official ZINC (subset) valid split, raw-`y` MAE; best epoch by
strict `<` with ties to the earlier epoch. Historical rows are read unchanged
from `zinc_arch_exploration_v3/analysis/{main_table,test_reporting}.json` and
from the dual-channel composer's frozen test predictions — they were not
retrained and their test numbers were not recomputed.

| model | params | best valid | best ep | eval-train@best | test MAE | updates | status | GPU-h |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| P3B_s0 (reused) | 413,505 | 0.076223754 | 377 | 0.014680 | **0.069864841** | 47,400 | completed | 0.30 |
| L_s0 (reused) | 494,193 | 0.078537152 | 489 | 0.005510 | **0.066266555** | 47,400 | completed | 0.35 |
| A_s0 (reused) | 488,601 | 0.082189201 | 632 | 0.011543 | 0.080528948 | 188,400 | completed | 2.72 |
| A_SSL_s0 (reused) | 488,601 | 0.079876146 | 439 | 0.017018 | 0.084753448 | 141,300 | completed | 1.84 |
| A_attn_s0 (reused) | 488,889 | 0.082637432 | 519 | 0.012976 | 0.079930412 | 141,300 | completed | 2.02 |
| **DBT_s0 (this round)** | **452,523** | **0.077768402** | **949** | **0.015964** | **0.069401494** | **235,500** | completed | **3.69** |

Final (not selected) DBT state: valid 0.081313816, eval-train 0.007638685 at
epoch 1,500 — i.e. the last 551 epochs bought nothing and the fixed model
overfits further (train halves, valid rises).

Test metrics of the frozen checkpoint: RMSE 0.153396, median AE 0.036023,
p90 AE 0.142025, p95 AE 0.223497, max AE 2.210739, mean signed error −0.005412.
Valid metrics of the same state: MAE 0.077768, RMSE 0.299432, median AE
0.036680, p95 AE 0.202538, max AE 7.126240. Test − valid = −0.008367, the same
direction (test easier) as every reference in this repository.

### Curve facts (`analysis/curves.png`, `analysis/observations.json`)

| epoch | 100 | 300 | 600 | **949** | 1200 | 1500 |
|---|---:|---:|---:|---:|---:|---:|
| valid MAE | 0.129797 | 0.094008 | 0.081551 | **0.077768** | 0.082554 | 0.081314 |
| LR (per update) | 2.94e-4 | 2.76e-4 | 1.88e-4 | 9.21e-5 | 3.17e-5 | 3.00e-6 |

* Best valid at **epoch 949 = 63 % of the declared horizon**, LR 9.2e-5.
* Best valid inside the reference `lr ∈ (1e-4, 3e-4]` band: **0.079031 @614**.
* Mean epoch 8.85 s (157 updates), 3.69 GPU-h total, peak memory 2.05 GB.
* Online train MAE falls monotonically (0.0339 at the last epoch) while valid
  is flat after ~epoch 950: the capacity is not the binding constraint.

## 3. Selection, test read, and the valid/test discrepancy

Selection used only the full valid split; the freeze file was written
**before** the test object cache was built and before any test label was read
(`analysis/test_freeze.json`, `frozen_at 2026-10-22T02:28:12+0800`, checkpoint
sha256 `5810aaf7…`, epoch 949). Re-loading the checkpoint reproduces the valid
MAE to 7e-9, so the scored state is the frozen state.

Provenance note: the first terminal attempt (run `dbt-test`, 02:24) wrote the
freeze, built the **label-free** test relation cache (path counts / distances
only, sha256 `235de3e5…`), and then failed while loading the label-bearing
official test object cache, which did not exist on the compute host — so it
read **no** labels. The harness was changed to build a compact label-free
object cache (bit-identical to the earlier rounds' `ad_valid.pt` on all eight
fields it uses, verified inside the scoring run); the second attempt rewrote
the freeze with identical checkpoint/epoch/valid content and performed the
single test-label read at 02:28. No model, epoch, recipe or checkpoint changed
between the two attempts, and no test-derived statistic was used for any
choice.

### Paired descriptive contrasts (10,000 resamples, Δ = AE(DBT) − AE(ref))

Test — same 1,000 rows, frozen reference predictions:

| ref | MAE ref | MAE DBT | mean Δ | 95 % CI | DBT better on |
|---|---:|---:|---:|---|---:|
| P3B_s0 | 0.069864841 | 0.069401494 | −0.000463 | [−0.005308, +0.004362] | 50.3 % |
| L_s0 | 0.066266555 | 0.069401494 | +0.003135 | [−0.002272, +0.008404] | 47.5 % |

Valid — the same checkpoint, with the references' own valid-best predictions:

| ref | MAE ref | mean Δ | 95 % CI | DBT better on | ref AE at row 172 | mean Δ without row 172 |
|---|---:|---:|---|---:|---:|---:|
| P3B_s0 | 0.076223754 | +0.001545 | [−0.011511, +0.018792] | 52.5 % | 1.5630 | **−0.004023** |
| L_s0 | 0.078537152 | −0.000769 | [−0.011279, +0.012395] | 50.8 % | 6.0993 | **−0.001797** |

Row 172 is the extreme-label valid molecule (label −20.34) that earlier rounds
already flagged. DBT's AE there is 7.126 = **9.2 % of the whole valid MAE**
(the next worst is far smaller; 4 valid molecules have AE > 1). Without that
one molecule DBT's valid MAE is 0.070713 versus P3B 0.074735 and L 0.072510.
This is **not** used for selection, not a headline and not a substitute for the
overall number — it only explains why the valid point estimate and the test
contrast point in different directions.

## 4. Cost and provenance

| item | value |
|---|---|
| GPU | 1 × A100-SXM4-40GB on `res` (`res-gpu0`), driver 550.163.01, torch 2.5.1+cu124 |
| formal training | 13,290.5 s = **3.6918 GPU-h** (1,500 epochs, 235,500 updates) |
| timing run | 37 s (2 epochs, 9.41 s/epoch projected 3.92 h — matched the real run) |
| terminal test | 11 s |
| CPU prep | 16 s train + 2 s valid path caches; 1.1 s compact valid cache; 1.5 s test object cache |
| peak GPU memory | 2,052 MB |
| checkpoint | `runs/dbt_s0/best.pt`, sha256 `5810aaf7b7efa9855e52ccfe920338924d50ffce1d6b9dae8573a09d8933e000` |
| recovery state | `runs/dbt_s0/last.pt` + `final.pt` (state_dict, optimizer, RNG, epoch, global_step, best, curve); resume continues the same cosine, refuses a horizon change |
| data versions | train label sha256 `f0222d20…`, valid label sha256 `72a9cfc3…`, path cache `dbt-path-v1-L12` (train `bc76b430…`, valid `50f5d53a…`), test path cache sha256 `235de3e5…`, test label sha256 `66fd23ec…`, P3 objects `14326a7a…` — all recorded in `runs/dbt_s0/train.json` and `analysis/test_report.json` |
| git | implementation `c296a6f`, analysis/test harness `971f7f5` |

No budget cap was applied, no ledger balance was consulted, no run was
cancelled on time grounds. The declared horizon was completed.

## 5. Conclusion

**What was observed.**
One directed-bond Transformer with 452,523 parameters, trained once for the
declared 235,500 updates, reaches valid 0.077768 @949 and test 0.069401.
Relative to the two primary references it is a test-level tie with P3B
(Δ −0.00046, CI spans 0) and a point-estimate loss to L (+0.00314, CI spans 0);
on valid it is 0.0015 behind P3B and 0.0008 ahead of L. The valid gap to P3B is
almost entirely one extreme-label molecule; on the other 999 valid molecules it
is ahead of both, while test says tie/worse. Train MAE keeps falling to 0.0076
by epoch 1,500 while valid is flat after epoch ~950, and the best checkpoint
sits at 63 % of the horizon with LR 9.2e-5. The full 500k budget is not used
(47k headroom) and the model is not compute-bound (2 GB, 8.9 s/epoch).

**What it may mean.**
Within this implementation and recipe, keeping both bond-endpoint roles and
driving them with four layers of exact endpoint-bound path-count relations does
**not** move single-model generalisation beyond the existing static trunks. Two
readings are compatible with the data and neither is proven: (i) the extra
interaction capacity is spent on fitting the training set (train 0.016 → 0.008
while valid rises), i.e. the relation-conditioned attention is not the missing
sample-efficiency piece at this data size; or (ii) the valid/test disagreement
is dominated by one molecule and by split noise, so this single seed simply
cannot separate 0.069–0.070 from 0.066–0.070. What the round does show is that
the mechanism *works as declared* — the relation changes the read content with
or without identical token states, the model is exactly invariant to atom
relabelling / edge-direction input / batch composition, and the endpoint roles
and junction incidence are correct. It does not show that "endpoint identity is
useless", that "attention does not help", or that a no-message-passing ceiling
has been reached.

**Is it worth continuing.**
Not by scaling this recipe: a second seed, more layers or a wider trunk would
answer "is 0.0005 real", which is below the resolution of a 1,000-molecule valid
split and a spent test split. The only follow-up with a clear, separable
question would be whether the *training-set fit* is the problem rather than the
representation — e.g. one run with the same trunk and a materially stronger
regularisation/early-stopping regime, or a matched-capacity control that keeps
the directed objects but removes the path-count relation (which would test the
relation itself, not the object). Those are new purchases that need their own
pre-registration; none is started here, and no seed 1 / depth / width /
path-length grid was run.

## 6. Limitations

* Single seed, single recipe, single declared horizon. Every interval above is
  descriptive, not confirmatory.
* The official ZINC test split is already spent in this repository; the test
  numbers are one honest read of a frozen checkpoint, not a blind evaluation.
* `L = 12` is a bounded exact count, not all-length structure; counts are small
  on ZINC (max 12), so the relation is close to a distance/ring descriptor here.
* The valid comparison is dominated by one extreme-label molecule; the
  "without row 172" numbers are diagnostic only.
* Only the chemical-regression task on ZINC subset was tested; nothing here
  speaks to other datasets, to larger graphs, or to the literature models.
