# REPORT — SSRC Research Tree v1 (CMA / SPC)

**Round:** `zinc-compositional-research-tree-v1`
**Track:** `tracks/structural-sufficiency/zinc_compositional_research_tree_v1/`
**Protocol:** `PROTOCOL.md` (frozen before the first formal GPU run)
**Ledger:** `TREE_LEDGER.md` (implementation fixes, gates, budget, deviations)

---

## 0. 中文摘要（一句话版）

按任务冻结设计实现两棵子树——**CMA（有向 3-bond 路径耦合：两侧弦型联合编码）** 与
**SPC（割点—分支端口组合）**，每棵子树都与其**参数、形状、初始状态完全一致**的 NULL
臂配对训练。CPU 门禁 17/17 组通过（§4 的 12 条规则）后按 Stage 0→IV 执行：**SPC 在 Stage II 被否证**
（REAL−NULL = **+0.0013**，CI 跨 0；绝对误差 0.1451，比 DCC-L 差 0.0219）；**CMA 通过
Stage I/III 全部晋级条件**，两个 seed 方向一致（seed0 **−0.0109**，seed1 **−0.0045**，
full Monitor），在**从未碰过、也未参与选择的 Locked1k** 上复现（**−0.0070**，
CI [−0.0118, −0.0024]，ordinary −0.0058）。Stage IV（官方 train10k 训练、官方
valid1k 选择）两个 seed 再次复现同方向、且更大的匹配差（**−0.0167 / −0.0197**），
但该差在 valid 上主要由 65 个 tail 分子承载（seed0 90%，seed1 54%）。
CMA 的**绝对水平并未超过 DCC-L 参考**（Monitor 0.1249 vs 0.1232，+0.0018；
Locked 0.0905 vs 历史 0.0836，+0.0069；official valid 0.0843/0.0849 vs
0.0785/0.0738，+0.0057…+0.0110）。
结论：**「ZINC 的结构贡献主要是局部联合类型（composition）而非边际特征累加」这半边
假说得到匹配机制证据支持；「割点—分支端口」这一具体分解被否证；两者都还没有把绝对
误差推到 DCC-L 之下。** 未读官方 test。

---

## 1. What was asked, what was done

The task (`SSRC Research Tree v1`, §1–§7) fixes a shared skeleton, two
composition mechanisms, matched REAL/NULL arms, CPU gates, one recipe, one
evaluation slicing, four execution stages and a promotion/stop rule. This round
implements that design literally and executes it. It does **not** invent models,
losses, metrics, ensembles, distillation, lookup tables or extra seeds.

| stage | what the frozen rules allowed | what was run | outcome |
|---|---|---|---|
| 0 | CPU gates; stop if coverage < 60 % | `shared/xs_gates.py` (the 12 §4 rules, 17 checks) | **17/17 PASS**; SPC cut-vertex coverage **100.0 %** → proceed |
| I | CMA REAL/NULL, seed 0, Fit8k | 2 × 47,400 updates | REAL 0.124922 vs NULL 0.135866 → **promote** |
| II | SPC REAL/NULL, seed 0, Fit8k | 2 × 47,400 updates | REAL 0.145107 vs NULL 0.143796 → **reject SPC** |
| III | CMA seed 1 + frozen one-shot Locked read | 2 × 47,400 updates + 1 Locked read | seed1 REAL 0.132780 vs NULL 0.137275 (same sign) → **Stage IV authorised** |
| IV | full train10k, selection on official valid1k | 4 × 47,400 updates (REAL/NULL × seed 0/1) | see §6 |

Total formal GPU time: see §7 (`analysis/budget.json`). Official **test was
never loaded**; `xs_common.load_cache("test")` raises `PermissionError` and gate
g10 statically re-checks that no file outside the boundary module mentions the
test cache.

## 2. Fixed skeleton and mechanism resolution (§1–§3)

Shared skeleton exactly as frozen — `raw108 → 128 → 96` atom encoder,
`[a_u+a_v, |a_u−a_v|, a_u·a_v, bond7] (295) → 160 → 96` bond encoder,
`g_base = [Σa, Σa², Σb, Σb², global33, log1p n_atoms, log1p n_bonds]` (419),
`rho_base: 419→128→64→1` (SiLU, no dropout), and
`y = rho_base(g_base) + Σ_anchor rho_event(anchor, event, event², log1p count)`
with `rho_event: 225→128→1`, **zero-initialised output**, so training starts from
the pure global baseline. Both families reuse the same skeleton and the same
`rho_base`; the only difference between the two families is the event module.

**CMA (Coupled Motif Algebra).** Every directed 3-bond simple path `(i,j,k,l)` is
enumerated in both orientations; `z_abc = 0.5[F2(a,b,c) + F2(c,b,a)]` with
`F2: 484→96→48`; the chord types of the two side closures ∈ {no-edge,
bond1, bond2, bond3} are concatenated with `z` to `252` dims, `F3: 252→96→64`,
and the event of physical bond `e` is `m_e = Σ F3 / (2√(1+n_e))` over its paths.
The **NULL arm** keeps the identical tensors and parameter count but composes on
**fragment means**: the two directed records of a path are averaged *before* the
chord concatenation, so the joint correspondence of the left/right environments
is destroyed and `|z_L − z_R| ≡ 0` (gate 3 measures this directly).

**SPC (separator-port composition).** Tarjan articulation points; for each cut
vertex `v` the components of `G−v` are the branches; each branch is summarised
from its atom/bond features plus `Fport: 291→96→48`; branch pairs are composed by
`Fpair: 240→96→64` and `t_v = Σ pairs / √(1+n_pairs)`. Molecules without cut
vertices contribute exactly zero (measured: none in Fit8k). The **NULL arm**
composes on the **mean port** of the same branch set (same shapes, same
parameters, no pair coupling).

Parameter counts (frozen, verified by gate 6): **CMA 262,002**, **SPC 242,322**
(REAL = NULL exactly; shared skeleton bit-identical across families at the same
seed, hash `87d54a23…`), all ≤ 500,000.

## 3. CPU gates (§4) — all pass before any GPU run

`analysis/gates/cpu_gates.json` (17 checks covering the 12 §4 rules, `all_pass = true`). Highlights:

* **Invariance** (atom renumbering, bond flip, adjacency reorder, batch split):
  max |Δ| ≤ 3e-5, all four family × arm cells.
* **Enumeration vs brute force**: 26 graphs (6 hand-built + 20 real ZINC),
  1,786 directed path records, **0 mismatches**; every reverse record present with
  swapped side chords; `#directed = 2·Σn_e`; F3 reversal difference = 0.0.
* **REAL ≠ NULL**: CMA per-bond `m` differs (max 0.087 on a 32-molecule batch;
  0.198 on the fixed `n_e = 2` example), SPC `t` differs (0.068; 0.151 on the
  fixed 3-branch star); both are **exactly 0** where the anchor count is 0.
* **Parameter matching / identical init**: REAL = NULL counts, identical initial
  state hashes.
* **Gradients**: finite and non-zero for every module; loss decreases;
  checkpoint round-trip exact.
* **No label-derived input**: forward arguments ⊆ whitelist; mutating labels
  leaves predictions bit-identical.
* **Coverage stop (§3)**: CMA 44.0 undirected paths per molecule (median 43);
  SPC **100.0 %** of Fit8k molecules have a cut vertex → the pre-registered
  < 60 % stop did not trigger.

## 4. Stage I / II / III — Monitor1k results (Fit8k training)

Frozen recipe for every run: AdamW, batch 128, warmup 500, cosine 1e-3 → 1e-5 per
update, wd 1e-5, clip 5, raw-y L1, **47,400 updates**, full Monitor every epoch,
strict-min checkpoint. Single model per run. The full per-epoch curves (Monitor
MAE for all 753 epochs) are in `runs/*/train.json:curve`; the last-100, last-20
and best-20-rolling statistics in the table are computed from them.

| run | best Mon MAE | @ep | roll-20 median | last-100 median | last-20 median | ordinary (y>−3) | tail (y≤−3) | unseen-ordinary | seen-ordinary | −worst10 | GPU·h |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `stage1-cma-real_s0` | **0.12492** | 531 | 0.12532 | 0.12534 | 0.12532 | 0.07417 | 0.75995 | 0.12736 | 0.05217 | 0.07778 | 0.362 |
| `stage1-cma-mean_s0` | 0.13587 | 550 | 0.13636 | 0.13643 | 0.13642 | 0.08243 | 0.80456 | 0.13426 | 0.06098 | 0.08601 | 0.435 |
| `stage2-spc-real_s0` | 0.14511 | 433 | 0.14585 | 0.14588 | 0.14587 | 0.09208 | 0.80873 | 0.14496 | 0.07020 | 0.09498 | 0.291 |
| `stage2-spc-mean_s0` | 0.14380 | 607 | 0.14430 | 0.14436 | 0.14437 | 0.09145 | 0.79883 | 0.14860 | 0.06780 | 0.09386 | 0.330 |
| `stage3-cma-real_s1` | **0.13278** | 568 | 0.13320 | 0.13322 | 0.13322 | 0.08011 | 0.79186 | 0.13367 | 0.05795 | 0.08381 | 0.353 |
| `stage3-cma-mean_s1` | 0.13727 | 584 | 0.13779 | 0.13783 | 0.13783 | 0.08486 | 0.79322 | 0.13804 | 0.06285 | 0.08892 | 0.437 |

Matched REAL − NULL contrasts (paired bootstrap over molecules, 5,000 resamples,
negative = REAL better). Monitor slice sizes: n = 1000, ordinary 926, tail 74,
any-unseen 317, unseen-ordinary 271, seen-ordinary 655.

| family / seed | full | ordinary | unseen-ordinary | −worst10 | tail-only? |
|---|---|---|---|---|---|
| CMA s0 | **−0.01094** [−0.01623, −0.00594] | **−0.00825** [−0.01261, −0.00384] | −0.00691 [−0.01903, +0.00533] | −0.00823 | no |
| CMA s1 | **−0.00449** [−0.00931, +0.00036] | **−0.00475** [−0.00929, −0.00008] | −0.00437 [−0.01693, +0.00809] | −0.00511 | no |
| SPC s0 | +0.00131 [−0.00371, +0.00613] | +0.00063 [−0.00359, +0.00478] | −0.00364 [−0.01354, +0.00577] | +0.00112 | no |

Three facts that matter for the reading:

1. **CMA's gap is not a tail artefact.** It is present on ordinary molecules
   (−0.0083 / −0.0048), on molecules whose radius-2 environment never occurs in
   Fit8k (−0.0069 / −0.0044), survives deleting the fixed worst-10 Monitor rows
   (−0.0082 / −0.0051), and survives deleting the single dominant row 1656
   (−0.0110). It is also spread rather than carried: REAL is better on 557 of 1000
   Monitor molecules (worse on 443), the largest single improvement is 10.2 % of
   the net gap, the top-5 rows sum to 28.8 % and the fixed worst-10 rows to 25.6 %
   — a broad 56/44 per-molecule majority, not one row doing the work.
2. **It is not an epoch-selection fluke.** REAL is better than NULL in **90.8 %**
   of the 753 training epochs, and the gap is stable in every 100-epoch block
   (−0.023, then −0.010, −0.011, −0.012, −0.011, −0.011, −0.011, −0.011).
   Online train L1 at the end is identical (7.87e-4 vs 7.83e-4): the NULL arm fits
   the training set just as well, so the difference is generalisation structure,
   not optimisation.
3. **The two seed-0 arms are identical in every respect except the coupling.**
   Parameters 262,002 both, initial shared-skeleton state hash identical, same
   recipe, same data order, same seed.

## 5. Stage III — frozen Locked1k read (one shot, CMA s0)

Before the read, both checkpoints were written to `analysis/locked_freeze.json`
with SHA-256 (`28960b01…` REAL, `963d8c03…` NULL), git commit and Monitor values;
the reader verifies those hashes and refuses to run without the freeze file. The
Locked rows were never used for training or for any selection.

| arm | Locked1k MAE (n = 1000) |
|---|---|
| CMA REAL s0 | **0.090529** |
| CMA NULL s0 | 0.097547 |

Matched delta **−0.00702** [−0.01182, −0.00235], win-rate 55.1 %; ordinary subset
(n = 932) **−0.00579** [−0.01044, −0.00122]; tail (n = 68) −0.0238. The sign,
size and ordinary-locality of the Monitor gap reproduce on rows the model has
never seen and that played no part in checkpoint selection. This is the strongest
piece of evidence in the round: the effect is a property of the mechanism, not of
the Monitor slice.

## 6. Stage IV — full train10k, official valid1k selection

Authorised by §7 because the two CMA seeds agreed in sign on every slice and no
ordinary-molecule degradation appeared. Same recipe, same 47,400 updates, but
trained on the full official train subset (10k) and selected on the official
valid1k split (frozen slice definitions re-applied to valid rows; the two
Monitor-only slices — fixed worst-10 and fixed row172 — are *not defined* on
valid rows and are reported as N/A rather than silently transplanted).

| run | best valid1k MAE | @ep | roll-20 | final | ordinary | tail (65) | unseen-ord | GPU·h |
|---|---|---|---|---|---|---|---|---|
| `stage4-cma-real_s0` | **0.08428** | 368 | 0.08659 | 0.08656 | 0.07364 | 0.23734 | 0.13049 | 0.353 |
| `stage4-cma-mean_s0` | 0.10101 | 392 | 0.10186 | 0.10187 | 0.07541 | 0.46928 | 0.13119 | 0.426 |
| `stage4-cma-real_s1` | **0.08488** | 445 | 0.08650 | 0.08657 | 0.07425 | 0.23790 | 0.13055 | 0.361 |
| `stage4-cma-mean_s1` | 0.10459 | 408 | 0.10692 | 0.10713 | 0.08385 | 0.40284 | 0.14287 | 0.429 |

Matched REAL − NULL on valid1k (n = 1000, ordinary 935, tail 65):

| seed | full | ordinary | unseen-ordinary | tail |
|---|---|---|---|---|
| 0 | **−0.01673** [−0.04417, −0.00056] | −0.00177 [−0.00612, +0.00258] | −0.00070 [−0.01340, +0.01169] | −0.23193 |
| 1 | **−0.01970** [−0.03939, −0.00756] | **−0.00961** [−0.01432, −0.00502] | **−0.01232** [−0.02445, −0.00026] | −0.16494 |

**Trend check (the only thing Stage IV is allowed to decide).** The direction
reproduces on the official valid split at both seeds and with a *larger* matched
gap than on Monitor (−0.017 / −0.020 vs −0.011 / −0.0045), and REAL's valid error
is remarkably seed-stable (0.0843 / 0.0849) while NULL's is not (0.1010 /
0.1046). The round's frozen `decision` rule (sign agreement + no ordinary
degradation) passes.

**But the composition of the gap changes, and this must be said plainly.** On
valid1k the gap is carried mostly by the 65 tail rows: seed 0 gets 90 % of its net
−0.0167 from tail rows (−0.232 per tail row) and only −0.0018 from ordinary rows
(CI spans 0); seed 1 is broader (54 % tail, ordinary −0.0096 with CI excluding 0).
On Monitor1k the same contrast was broad (−0.0083 ordinary, 7 % of rows in tail).
So the honest statement is: *the joint-typing advantage is unambiguous and
reproducible for extreme negative-label molecules, and it is real but
seed-dependent (and on one of two valid seeds not individually significant) for
ordinary molecules on truly unseen data.*

**Absolute level.** With the same protocol (train10k, valid selection) the DCC-L
reference reads 0.07854 / 0.07384 by seed (historical recomputation in
`zinc_dccl_bottleneck_v1`, 4-seed mean 0.075990). CMA REAL is 0.08428 / 0.08488,
i.e. **+0.0057…+0.0110 worse than DCC-L**, and the NULL arms are far worse
(0.1010 / 0.1046). Stage IV therefore confirms the mechanism contrast and
confirms that this tree does not beat the established bottleneck baseline.

## 7. Budget, execution regime, boundary

| stage | runs | GPU·h | stage cap |
|---|---|---|---|
| timing smokes (3 × 126–158 updates) | `stage1/2/4-timing-*` | 0.0064 | — |
| I — CMA s0 REAL/NULL | 2 | **0.7965** | 4.0 |
| II — SPC s0 REAL/NULL | 2 | **0.6216** | 4.0 |
| III — CMA s1 REAL/NULL | 2 | **0.7902** | 3.0 |
| IV — train10k REAL/NULL × seed 0/1 | 4 | **1.5683** | 3.0 |
| **total** | 10 formal + 3 smoke | **3.7829** | soft 14.0 / hard 16.0 |

Wall clock: formal runs span 01:00→04:35 on `res` (3 h 35 min with at most 2
GPUs busy, no queue waits); the whole round — protocol freeze to final analysis —
is ≈ 5.5 h of the 8 h cap, and only 3.78 GPU·h of the 14 GPU·h soft cap were
spent. Unused budget (≈10.2 GPU·h of
the soft cap, ≈1.4 GPU·h of the Stage IV cap) was **not** reallocated: no extra
seeds, architectures or reruns were bought with it. Every run's `train.json`
carries `gpu_hours`, `git_commit`, checkpoint SHA-256 and the execution regime
(driver/torch/node), so the numbers above are auditable per run.

Execution regime for every run: host `res` (`a100-2`, 2 × A100-SXM4-40GB,
driver 550.163.01, torch 2.5.1+cu124), process backend, one run per GPU, git
commit recorded per run (`train.json.git_commit`, `rr meta.json`), `git_dirty =
false` for all formal runs. Structure caches (`cma_*.pt`, `spc_*.pt`) were built
once on the host and are byte-identical across runs (hashed in `train.json`).

Data boundary: only the official **train** subset (10k) and, in Stage IV only, the
official **valid** subset were opened. The internal split is the frozen
`bjoc_predictive_bottleneck_v1` split A (Fit8k 8000 / Monitor 1000 / Locked 1000),
row hashes re-verified on every load (`1cf6a2ec…`, `d493dc7f…`, `9cbefd5e…`).
Official test: never opened, never hashed by this round, `PermissionError` on any
attempt. SSL encoder initialisation reuses the existing train-only SSL
checkpoints (`pretrain_encoder_s0/s1.pt`); **disclosure**: those encoders were
pretrained on the complete official train10k structures, so Monitor and Locked
are not fully *structural*-blind (they are label-blind). This is the same regime
as the DCC-L reference numbers; no historical supervised checkpoint was used as
an initialisation.

## 8. Interpretation

**What the CMA contrast isolates.** REAL and NULL differ only in *when* the two
directed views of a path are merged: REAL concatenates the joint `(z_L, z_R)`
with the two side chord types and lets `F3` see their *cross-product*; NULL
averages the two views first, which sets `|z_L − z_R| ≡ 0` and removes the
left/right correspondence. The consistent −0.005…−0.011 MAE gap therefore says:
for ZINC, the predictive content of a local environment is not exhausted by its
two marginal half-environments — the *joint* type of the pair, and how that pair
is closed by chords, is worth **8.0 % of the NULL arm's own Monitor error at seed
0 and 3.3 % at seed 1** (16.6 % / 18.8 % of the NULL arm on the official valid
split), which the marginal summary cannot recover. That is the "composition, not
accumulation" half of the round's hypothesis, and it survives a null that is
perfectly matched in shape, parameter count, initialisation and optimiser.

**Why SPC fails.** Three observed facts, then the reading we are willing to
defend. Observed: (i) the REAL/NULL contrast is ≈ 0 (+0.0013 full, +0.0006
ordinary, CIs span 0) — the port *coupling* adds nothing measurable; (ii) the
mechanism fires on 100 % of molecules (9.2 cut vertices and 13.3 branch pairs per
molecule on average), so this is not an under-coverage artefact; (iii) SPC is the
weakest arm in absolute terms (0.1451 for REAL, 0.1438 for NULL) even though it
shares the identical skeleton with CMA, whose NULL arm reaches 0.1359. Reading
(interpretation, not measurement): `t_v` is a **sum of per-cut-vertex terms**, so
the event enters the prediction as an additive local correction over a coarse
summary (a 195-dim pooled description of an entire side branch), not as a global
composition; and "compose on the port mean" is a weak null here, because for the
common 2-branch separator the pair is nearly the multiset of ports, so REAL and
NULL see close to the same information. The defensible conclusion is that the
cut-vertex/separator algebra is the wrong compression of ZINC structure — not that
"composition" is the wrong idea (the CMA contrast says the opposite).

**What this does not show.** CMA does not beat DCC-L. On the same Monitor split
CMA REAL is 0.1249 vs DCC-L 0.1232 (+0.0018), on Locked 0.0905 vs the historical
DCC-L 0.0836 (+0.0069), and on the official valid split 0.0843/0.0849 vs
0.0785/0.0738 (+0.0057…+0.0110). The mechanism buys a real, reproducible matched
increment, but the round's absolute bar (≤ DCC-L + 0.010) is met only in the weak
sense of "not much worse". The frozen rules therefore do not license any claim of
a new state of the art, and none is made.

**Where the advantage actually lives.** The one place where the two arms separate
*mechanistically* rather than statistically is the extreme tail. On valid1k, REAL
spends 0.237 MAE on the 65 tail rows at *both* seeds, while NULL spends 0.469 and
0.403 — a 1.7–2.0× difference that no other slice shows, and the one slice where
REAL is seed-stable and NULL is not. The CMA event term gives each bond a
composition of its local path algebra; for a molecule whose label is far from the
bulk, that local algebra is apparently what stops the model from regressing to the
conditional mean. For ordinary molecules the same term acts like a small,
seed-dependent regulariser (−0.002…−0.010). This is the honest, non-inflated
shape of the result: **a tail-sharpening mechanism with a real but modest
ordinary-molecule component**, not a uniform improvement.

## 9. Deliverables and reproduction

| file | content |
|---|---|
| `PROTOCOL.md` | pre-registration (skeleton, mechanism resolutions, slices, stages, caps) |
| `TREE_LEDGER.md` | chronological log: fixes, gates, runs, budget, deviations |
| `DECISION.md` | the promotion/stop decisions and what they rest on |
| `REPORT.md` | this file |
| `analysis/main_table.csv` | one row per formal run (all slices, hashes, GPU·h) |
| `analysis/slices.csv` | long-form slice table for best and final checkpoints |
| `analysis/contrasts.json` | matched REAL − NULL deltas with bootstrap CIs |
| `analysis/budget.json` | GPU·h per stage vs caps, wall clock |
| `analysis/frozen_baseline.json` | DCC-L reference recomputation, fixed worst-10, fixed row172 |
| `analysis/monitor_slices.npz`, `valid_slices.npz`, `valid_freeze.json` | frozen slice masks (Monitor1k, valid1k) |
| `analysis/locked_freeze.json`, `locked_read.json`, `locked_contrast.json` | frozen checkpoints and the single Locked read |
| `analysis/stage1_detail.json`, `analysis/stage4_detail.json` | CMA curve/row decomposition and the valid-split tail decomposition |
| `analysis/gates/cpu_gates.json` | 17 checks (12 §4 rules), all pass |

Reproduction (from the repo root, host `res`):

```
rr deploy res
R=tracks/structural-sufficiency/zinc_compositional_research_tree_v1
python $R/shared/xs_gates.py                       # CPU gates, ~3 min
python $R/shared/xs_train.py --family cma --prepare-only --device cpu
rr run res ssrc-cma-real-s0 --pool res-gpu0 -- python $R/shared/xs_train.py \
    --family cma --arm REAL --seed 0 --device cuda:0 --tag stage1-cma-real_s0
# ... same for MEAN / SPC / seed 1 / --fit train10k --monitor valid ...
python $R/shared/xs_analyze.py freeze && python $R/shared/xs_analyze.py tables
python $R/shared/xs_analyze.py decision
python $R/shared/xs_analyze.py locked      # refuses without locked_freeze.json
```
