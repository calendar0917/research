# BJOC-Null + SSL Recipe Tuning v1 — REPORT

Protocol `cscl-bjoc-recipe-tuning-v1` · commit `4f91d204d600` (frozen
experiment revision) · config
`configs/bjoc_recipe_tuning_v1.yaml` (`sha256 19280c71…`) · driver
`code/run_bjoc_recipe_tuning_v1.py` (`sha256 39bf5479…`) · note
`notes/bjoc_recipe_tuning_v1.md` (preregistration + pre-GPU amendments).

**Scope.** Supervision-phase recipe only. BJOC-Null architecture (413,505
params), full-10k masked-SSL encoder, composer/reader init, data order, raw-y
L1, clip 5, batch 128 and the last-5-epoch soup rule are all frozen. Official
test was never loaded, forwarded or read in any form. Official valid is an
exposed development split.

---

## A. One-line conclusion

**`RECIPE_GAIN_CANDIDATE` — the 360-epoch cosine schedule (R3) is the only
candidate that passes the pre-registered screening and confirmation gates
(seed-mean valid Δ = −0.003705, bootstrap CI95 [−0.006390, −0.001011],
ordinary molecules improve, gain not carried by `valid:0172`), but it is a
selection-conditioned candidate result on the exposed valid split, not a
blind confirmation, and it costs 1.5× supervision updates with a small
seed-unstable tail regression.**

---

## B. Core tables

### B1. Seed-0 screening grid (official valid, n = 1000, paired vs R0/P0)

| Recipe | Seed | Epochs | LR schedule | WD | Updates | Fit MAE | Valid MAE | Δ vs R0 | Bootstrap CI95 | GPU wall |
|---|---:|---:|---|---:|---:|---:|---:|---:|---|---:|
| **R0 (P0)** | 0 | 240 | constant 1e-3 | 1e-5 | 18,960 | 0.025333 | 0.084786 | — | — | replay-only (historical 620 s) |
| R1 | 0 | 240 | cosine 1e-3→1e-5 | 1e-5 | 18,960 | 0.003457 | 0.081737 | **−0.003049** | [−0.008213, +0.001395] | 452 s (438 s in-job) |
| R2 | 0 | 360 | constant 1e-3 | 1e-5 | 28,440 | 0.034737 | 0.091458 | +0.006673 | [+0.002078, +0.011354] | 658 s (644 s) |
| **R3** | 0 | 360 | cosine 1e-3→1e-5 | 1e-5 | 28,440 | 0.001782 | **0.079101** | **−0.005684** | [−0.009759, −0.001967] | 693 s (678 s) |
| R4 | 0 | 240 | constant 1e-3 | 1e-4 | 18,960 | 0.033612 | 0.089678 | +0.004892 | [−0.001436, +0.010832] | 465 s (451 s) |

Screening gate (pre-registered): Δ ≤ −0.003 **and** ordinary (y > −3) Δ ≤ 0
**and** Δ excluding `valid:0172` ≤ −0.002 **and** numerics/audits OK.

| Candidate | Δ MAE | Ordinary Δ (n=935) | Tail Δ (n=65) | Δ w/o `valid:0172` | Top-10-removed Δ | Gate |
|---|---:|---:|---:|---:|---:|---|
| R1 | −0.003049 | −0.000090 | −0.045600 | −0.001780 | −0.000196 | **fail** (extreme-row) |
| R2 | +0.006673 | +0.001641 | +0.079056 | +0.006073 | +0.006167 | fail |
| **R3** | **−0.005684** | **−0.003865** | −0.031846 | **−0.004653** | −0.003091 | **pass (unique)** |
| R4 | +0.004892 | +0.004391 | +0.012107 | +0.006187 | +0.005495 | fail |

Selection rule: lowest seed-0 valid MAE among passers → **R3** (no tie-break
needed). R1 failed only the extreme-row gate: its headline gain is carried by
`valid:0172` (row Δ = −1.270; excluding the row only −0.00178).

### B2. Confirmation (unique winner R3 vs historical P, both seeds)

| Model | Seed | Valid MAE | Δ vs P (same seed) |
|---|---:|---:|---:|
| R3 | 0 | 0.079101 | −0.005684 |
| P | 0 | 0.084786 | — |
| R3 | 1 | 0.080586 | −0.001727 |
| P | 1 | 0.082313 | — |
| **R3 seed-mean** | 0,1 | **0.079844** | **−0.003705** |
| P seed-mean | 0,1 | 0.083549 | — |

* Bootstrap (molecule-level, 5000 resamples, seed-mean error vector):
  mean −0.003705, **CI95 [−0.006390, −0.001011]** (entirely < 0);
  56.4 % of molecules better (ordinary: 56.7 %).
* Ordinary (n = 935): Δ = **−0.004133** (both seeds improve: −0.0039 / −0.0044).
* Tail (n = 65): Δ = **+0.002443** seed-mean but seed-unstable
  (seed0 −0.0318, seed1 +0.0367) → disclosure flag `TAIL_WORSE`.
* RMSE: 0.227499 vs 0.230007 (improves).
* Top-10-removed Δ = −0.003635; `valid:0172`-excluded Δ = −0.004209. The
  extreme row itself is worse under R3 in seed-mean (row Δ +0.499:
  seed0 −1.036, seed1 +2.034), so the gain is not carried by it.
* Per-seed fit MAE: R3 0.001782 (s0), 0.001702 (s1).
* All six pre-registered confirmation conditions pass; CI upper < 0 →
  `RECIPE_GAIN_CANDIDATE` (not `WEAK_OR_INCONCLUSIVE`). Seed 1 alone
  (−0.001727) would not clear the −0.003 magnitude; the pre-registered gate is
  on the seed mean.

### B3. Cost and absolute accuracy

* R2/R3 use **1.5× supervision updates** (28,440 vs 18,960). On the A100 the
  extra 120 epochs cost ≈ +4 min per run (R3 678 s vs R1 438 s in-job).
* Seed-mean valid: 0.083549 → 0.079844 is a **4.4 % relative MAE reduction**
  at 1.5× supervision compute. This is not a zero-cost gain.
* Total GPU time this round (job walls, GPU1): smoke 17 s + 452 + 658 + 693 +
  465 + 690 s = **2,975 s ≈ 0.83 GPU-h** (in-job training time 2,887 s ≈
  0.80 h), well below the 2.5 GPU-h budget. R0 used 0 GPU time.

---

## C. Mechanism judgment (Q1–Q8)

### C1. Direct answers

**Q1 — R1 vs R0 at equal 240 epochs: does cosine help? Suggestive, not
robust.** Mean Δ = −0.003049, but ordinary Δ ≈ 0 (−0.00009), the gain nearly
disappears when `valid:0172` is excluded (−0.00178), top-5-removal flips
positive (+0.00049), and the CI crosses zero. Observation: cosine at 240
mostly trims the single worst row; possible explanation: late low-LR steps
reduce the largest errors rather than shifting the bulk distribution.
Research decision: not selected, not a confirmed gain.

**Q2 — R2 vs R0: do ~50 % more supervision updates help? No, clearly
harmful.** Δ = +0.006673 (CI entirely positive), ordinary +0.001641, tail
+0.0791, and train fit also gets *worse* (0.0347 vs 0.0253). Observation:
under constant 1e-3 LR the extra 120 epochs do not just fail to generalize,
they also fit the training rows worse at the soup point. Possible explanation:
constant-LR late training wanders (noise-dominated updates), so the last-5
soup averages a worse region than at epoch 240. There is no raw
"undertraining" deficit to be fixed by more constant-LR updates.

**Q3 — R3 vs R2 at equal 360 epochs: does cosine add value? Yes, large.** R3 −
R2 = −0.012357 (CI [−0.018316, −0.007377], entirely negative; tail also
better by −0.111). Against its own 240-epoch cosine sibling, R3 − R1 =
−0.002636 (CI [−0.005818, +0.000585], crosses zero). Observation: the cosine
increment at 360 is ~4× the cosine increment at 240.

**Q4 — R4 vs R0: does 10× weight decay help at fixed length? No.** Δ =
+0.004892 (worse direction, CI includes zero), ordinary +0.004391, fit worse
(0.0336 vs 0.0253). Possible explanation: 1e-4 over-shrinks the small
composer/reader while the frozen SSL encoder dominates capacity.
Research decision: reject this WD change for BJOC-Null.

**Q5 — Where does R3's advantage come from? Primarily the epoch × schedule
interaction.** Seed-0 2×2 factorial (per-molecule paired Δ vs R0):

| Effect | Value |
|---|---:|
| epochs 240→360 at constant LR (R2−R0) | **+0.006673** |
| epochs 240→360 at cosine (R3−R1) | **−0.002636** |
| schedule const→cosine at 240 (R1−R0) | **−0.003049** |
| schedule const→cosine at 360 (R3−R2) | **−0.012357** |
| interaction | **−0.009308** |

More updates alone hurt; cosine alone (at 240) is small and fragile; the
combination is where the real gain appears. What screening *can* identify:
this interaction at seed 0 on the exposed valid set (fixed grid, paired
comparisons). What it *cannot*: (a) whether the interaction is seed-stable —
only R3 was purchased at seed 1, so R1/R2/R3-on-seed-1 exists only for R3;
(b) whether it is "longer training", "more annealed total decay" or
"late low-LR averaging" that matters; (c) whether R3 is the best point of the
(O(1) untested) schedule/length space, since no post-hoc search was allowed.

**Q6 — Is the gain only extreme molecules? No, ordinary molecules benefit
broadly.** Ordinary (n = 935, y > −3): seed-mean Δ = −0.004133, improved in
both seeds, 56.7 % of ordinary molecules better; RMSE −0.0025; top-10-removal
−0.0036; `valid:0172`-excluded −0.0042. The tail (n = 65) is the one
non-improving subgroup and its sign flips across seeds, so tail behavior is
not a stable property of R3. Structure strata (seed-mean Δ): junction-rich
−0.0043, atom-rich −0.0037, large-diameter −0.0039 — i.e. the ordinary gain
is not concentrated in one structural regime.

**Q7 — When fit error goes down, does valid go down too? Across these five
recipes, yes — but that is not evidence that better fitting causes the gain.**
Fit MAE: R3 0.0018 < R1 0.0035 < R0 0.0253 < R4 0.0336 < R2 0.0347; valid
MAE ranks identically (Spearman ρ = 1 over 5 configs). The classic
"fit improves, valid worsens" dissociation does **not** occur here. However,
the huge fit drop is almost entirely a final-LR artifact — the cosine runs end
at LR ≈ 1e-5, so their last-5 soup sits in a sharply fitted basin, whereas the
constant-LR runs are still moving at 1e-3. With n = 5 and schedule
confounded, this ranking cannot separate "genuinely better approximation" from
"lower terminal LR gives lower reported fit".

**Q8 — Do the results say the recipe is hard to improve with these cheap
changes? Partly.** Not all candidates failed: R3 cleared both gates. But of
the three *single-variable* cheap changes, two (R2 more updates, R4 stronger
WD) are clearly harmful and one (R1 cosine at 240) fails the robustness gate.
The defensible statement: the 240-constant recipe is **not optimal under the
specific 360-cosine change**, and this improvement requires changing both
length and schedule together; it is **not** evidence that "all hyperparameters
are optimal" — hidden sizes, dropout, batch size, mask ratio, η_min, warmup,
soup window and loss form were not tested, and this round has no license to
extend the grid.

### C2. The seven required mechanism bullets

* **Training-length deficit?** Not in the raw sense — +50 % constant-LR
  updates hurt (Q2). The exploitable deficit is optimization-shape (an anneal
  at a longer horizon), not raw step count.
* **LR-schedule gain?** Yes, conditional on length: large at 360
  (−0.0124 vs constant), small/fragile at 240.
* **Weight-decay gain?** No; 1e-4 moves MAE in the wrong direction and also
  raises fit error.
* **Fit or generalization?** Both move together, but the fit drop is
  schedule-confounded (Q7); the ordinary-molecule and RMSE improvements in
  both seeds indicate a genuine distribution-level shift, not pure fit
  memorization. The unstable tail shows it is not uniformly better.
* **Typical fit-improves/valid-worsens?** Not observed in this grid (neither
  among candidates nor for R3's soup, whose fit is lowest and valid best).
* **Is the pretraining advantage still the main mechanism?** Yes, by an order
  of magnitude: SSL P vs scratch B seed-mean valid is −0.0156
  (0.083549 vs 0.099202), ~4× the R3 recipe delta (−0.0037). The recipe
  gain is a secondary tuning effect on top of the SSL representation.
* **What is only a current-config/current-split phenomenon?** The exact R3
  magnitude (+ its seed-1 attenuation), the interaction estimate, the tail
  sign, and every R1/R2/R4 claim (seed 0 only). Valid is exposed, and the
  candidate was selected on it.

---

## D. Reproduction materials

### Commits

| Commit | Content |
|---|---|
| `2cde875` | preregistration (config, runner, tests, note) |
| `ed15fa03` | pre-GPU amendment #1 (portable R0 replay reference) |
| `2427560` | pre-GPU amendment #2 (tolerance + committed label-free reference arrays) |
| `4f91d20` | force-add the two reference `.npy` files (globally git-ignored) |
| experiment revision `4f91d204d600` | deployed to `res`, frozen in `protocol_freeze.json` |
| *(report commit)* | this REPORT + analysis artifacts (see `git log`) |

No historical runner, constant, report or frozen checkpoint was modified.

### Hashes

* Config `sha256 19280c71fc5d189706b5508e88d1df812c069958515252c2868441328bc72853`; driver `39bf547918bc…`; frozen candidate block `5d21ddbd…`.
* Inputs: `objects_train10k.pt 14326a7ae6f9bb67…`, `objects_valid.pt 61fca36a097f859b…`, `prep_train10k.json d0fb86a349ebdd37…`.
* Label read: `valid_y_read_once.npy`, y sha256 `8f6ee636fd23ee97…`, `y_raw_vs_encoded_max_abs = 0.0`, single read recorded in `valid_read.lock`.
* Encoder init files: `pretrain_encoder_s0.pt 006fabc11efede09…`, `s1 ff8d6474896c2744…`; used encoder state `7165eb90ceaaec84…` (s0) / `54a232edfc280537…` (s1).
* Init hashes — seed 0: full `d06d5e168138b76b…`, group `d9ffbd15356659ed…`, encoder `91af980a414ac945…`; seed 1: full `c79e6b02521c224e…`, group `98d11fe93b03368e…`, encoder `83d8b086bf1f9ee6…`. All match the recorded P init.
* Batch order (independent recomputation and P-prefix match, 240-epoch prefix): seed 0 `94a8ff7fde7b0033…`, seed 1 `41add6c8f91a127a…`. 360-epoch full hashes differ between R2 and R3 (`3c964a71…`) as required (R1/R4 full = prefix).
* Soup checkpoints: `soup_R1_seed0 83199dfc…`, `soup_R2_seed0 d60c8b3a…`, `soup_R3_seed0 a021d36e…` (= file sha), `soup_R4_seed0 5d22acf0…`, `soup_R3_seed1 b622824a…`. Soup members are each candidate's own final 5 epochs (236–240 / 356–360); the 240-window is never reused for 360-epoch runs.
* R0 replay: local cross-check bit-exact (`max|Δpred| = 0.0` on both seeds vs the frozen npz); on `res` numeric replay vs the committed label-free reference arrays `max|Δpred| = 1.9e-6 / 2.9e-6`, under the pre-declared tolerance `1e-4`. `r0_replay.json` / `input_audit.json` record both.

### Execution regime and jobs (all `res` pool `res-gpu1`, node `a100-2`, GPU1, driver 550.163.01, torch 2.5.1+cu124/cuda 12.4, `CUDA_VISIBLE_DEVICES=1`, commit `4f91d204d600`)

| Job | Run ID | Wall |
|---|---|---:|
| prep + freeze | `recipe-prep3-20261013-112851-757daf2a` | 12 s |
| GPU smoke R1 (2 ep) | `recipe-smoke-r1-20261013-113038-694b7d62` | 17 s |
| R1 seed 0 | `recipe-r1-s0-20261013-113452-ec446730` | 452 s |
| R2 seed 0 | `recipe-r2-s0-20261013-113524-14fbefec` | 658 s |
| R3 seed 0 | `recipe-r3-s0-20261013-114726-1379ff52` | 693 s |
| R4 seed 0 | `recipe-r4-s0-20261013-120213-e80d301e` | 465 s |
| R3 seed 1 | `recipe-r3-s1-20261013-123329-2f2e0fee` | 690 s |

Failed pre-GPU prep jobs (infrastructure only, no training): `recipe-prep-…-2e70a0fa`, `recipe-prep2-…-e6a89029`; inspect `recipe-inspect-…-d37f9cd2`.

### Reproduce

```bash
rr deploy res --pool res-gpu1
rr run res recipe-prep3 --pool res-gpu1 --result tracks/structure-contribution/results/bjoc_recipe_tuning_v1 \
  -- uv run python tracks/structure-contribution/code/run_bjoc_recipe_tuning_v1.py prep
# per candidate: train --recipe R{1,2,3,4} --seed 0 --device cuda
# winner seed 1: train --recipe R3 --seed 1 --device cuda
# local: predict --recipes R1 R2 R3 R4 --seeds 0 --device cpu
#        freeze-predictions --stage screening; score-screening; screen
#        predict --recipes R3 --seeds 1 --device cpu
#        freeze-predictions --stage confirmation; score-confirmation; decision; analysis
```

Params: 413,505 exact on every candidate (atom 26,336 + bond 62,816 +
junction 112,896 + composer 62,976 + reader 148,481). Checkpoints:
`results/bjoc_recipe_tuning_v1/soup_R*.pt`; evaluation outputs
`prediction_freeze.json`, `confirmation_prediction_freeze.json`,
`predictions/valid_R3{0,1}.npz`, `screening_metrics.json`,
`screening.json`, `confirmation_metrics.json`, `scientific_decision.json`,
`analysis.json`, `post_hoc_analysis.json`. Evaluation code = runner
subcommands; frozen references `configs/frozen/README.md`.

### Protocol deviations / caveats (all disclosed)

1. **R1 and R2 overlapped on GPU1** for ~7 min because the launcher loop's
   first iteration did not wait for the already-launched R1. Both jobs exited
   0 with exact init/order/schedule audits and no OOM; float32 training is
   deterministic given seed/order, so **no scientific number is affected**,
   but their reported wall times are contention-inflated and no decision uses
   them. Subsequent runs were strictly serialized.
2. **R0 remote replay tolerates 1e-4** instead of an exact npz-hash match,
   because the `res` CPU reduces float32 differently and the historical npz
   is git-ignored (amendments #1/#2, both made **before** any GPU run). The
   exact check remains in place locally (0.0).
3. **Seed 1 alone is only −0.001727**; the gate that passes is the
   pre-registered seed-mean (and both-seeds-negative) condition. The CI and
   seed-1 check are **selection-conditioned on the exposed valid split and
   are not an independent blind test** (as pre-stated in the protocol).

---

## E. Next-step suggestions (not started)

*Independent-confirmation branch (recommended first, because a candidate
exists).* Re-derive an internal train/valid/test split that excludes both
official valid and test; retrain the SSL encoder **only on that split's
permitted structures** (the exposure rule that motivated full-10k SSL is part
of the question); then run R0 vs R3 with ≥3 seeds, pre-registering the same
paired gates, on the new internal set. Only that can turn
`RECIPE_GAIN_CANDIDATE` into a confirmed recipe decision.

*Mechanism-addressing branch.* To decide *which* part of R3 matters, a
pre-registered follow-up could hold 360 epochs fixed and compare
cosine→1e-5 vs cosine→0 vs linear vs constant-with-EMA, and compare last-5
soup vs last-1 vs EMA in the low-LR region. Any such follow-up must be a new
pre-registered round; this round's budget is exhausted and no extension is
proposed as part of it.

*If the signal had been negative* (not the case here), the task's fallback
would apply: shift to the Scale-separated Readout algebraic/information audit
or the atom-only / bond-only pretraining ablations, since the SSL effect
(−0.0156) dominates any recipe effect.

*Research-decision note.* Keeping 360-cosine would raise future BJOC
supervision cost 1.5× for a seed-mean 4.4 % relative valid MAE reduction that
is not yet independently confirmed; a pragmatic default is to adopt it for
internal development runs while treating the magnitude as provisional and
re-validating on any future independent split. The wd 1e-4 change should be
dropped, and 360-constant is strictly worse than both 240-constant and
360-cosine at seed 0.
