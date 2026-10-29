# TREE LEDGER — SSRC Research Tree v1

Chronological, append-only record of implementation, fixes, gates, runs and
budget. Science-level definitions live in `PROTOCOL.md` and are not edited here.

Times are local (`+08:00`), machine `res` = A100-40GB ×2, driver 550.163.01,
torch 2.5.1+cu124 (the same regime as the historical DCC-L / BOSI numbers).

## 2026-10-30 — setup and CPU phase

| time | event |
|---|---|
| ~23:20 | round directory created: `cma/`, `spc/`, `shared/`, `analysis/`, `runs/`; frozen split and recipe read from the BOSI/`zinc_arch_discovery_v1` caches |
| ~00:45 | first end-to-end forward passes for both families |
| ~01:05 | CPU gate suite `shared/xs_gates.py` written (12 task gates + SPC coverage stop) |
| 01:1x | gate run 1: 4 groups failed → see fixes below |
| 01:1x | gate run 2: all 17 checks pass (`analysis/gates/cpu_gates.json`, `all_pass = true`) |
| 01:18 | commit `7dd686b` — implementation + gates + frozen protocol + frozen baseline |
| 01:24 | `rr deploy res` (7dd686b); first prep job failed: `--arm` required for `--prepare-only` |
| 01:26 | commit `5a33e81`; redeploy; `ssrc-prep` exit 0 (18 s) — CMA/SPC structure caches for train10k + valid1k built on `res` |
| 01:28 | **Stage I timing** (`stage1-timing-cma-real_s0`, 126 updates = 2 epochs): 1.64 s/epoch, 0.0022 GPU·h |

### Engineering fixes (recorded before any formal run; no scientific change)

1. **`xs_data.BaseData.collate` bond offset bug** (found by gate 1). The batch
   `bond_index` used `atom_off[atom_owner[bond_owner]]` instead of
   `atom_off[bond_owner]`, so every molecule after the first in a batch read the
   first molecule's atoms. Fixed; gate 1 (permutation / flip / reorder /
   batch-split invariance) then passed for all four family×arm cells.
   *Affected runs:* none (no formal run had started). Smoke runs produced before
   the fix are discarded (`runs/` is git-ignored and the CPU smoke tag is not
   analysed).
2. **`xs_gates.mini_cache`** subtracted the molecule's atom offset from the
   cache's `bond_index`, which is already molecule-local (`ad-v1` convention).
   Gate-only helper; fixed.
3. **`spc_structure.build_split`** built `br_atom_ptr`/`br_bond_ptr` by
   appending arrays instead of extending offsets; fixed.
4. **`spc_structure.collate`** emitted one weight per branch instead of one per
   incidence; fixed.
5. **`cma_structure.build_split`** stored `path_z` with the split-global z
   offset while `collate` added it again; fixed to molecule-local rows.
6. Gate 8 initially reported `F2`/`F3` as missing gradients because the gate
   compared aggregated *module* keys (`F2.lins`) against bare names; the gate was
   fixed, the model was already correct (all modules receive finite, non-zero
   gradients by step 5).
7. Gate 12's secondary threshold (`loss < 0.5·std`) was replaced by the single
   pre-registered criterion “≥4× loss reduction on the synthetic task”; the
   target/optimiser/step count were not changed. (Declared explicitly because a
   gate criterion was edited.)
8. `xs_train.py --arm` made optional for `--prepare-only` (no scientific
   change).

### CPU gate summary (`analysis/gates/cpu_gates.json`, 17 checks, all_pass = true)

* g1 invariance: max |Δ| ≤ 3e-5 for atom renumbering, bond direction reversal,
  adjacency/bond-order permutation, and batch split/merge, both families, both
  arms.
* g2 CMA enumeration: 26 graphs (6 hand-built + 20 real ZINC), 1,786 directed
  records, 0 mismatches against an independent brute-force quadruple
  enumeration; every reverse record present with the two side chord types
  swapped and the same centre bond; `#directed = 2·Σn_e` everywhere; F3
  reversal difference = 0.0.
* g3 CMA REAL/MEAN: identical `n_e`/centre-bond tensors; per-bond `m` differs
  (max 0.087 on a 32-molecule batch; 0.198 on the fixed 5-node example with
  `n_e = 2`); `m ≡ 0` exactly when `n_e = 0`.
* g4 SPC: 44 graphs vs NetworkX — 0 cut-vertex, component, coverage, bond-set
  and cyclomatic-rank mismatches; permuting branch rows changes the prediction
  by ≤ 2.4e-7.
* g5 SPC REAL/MEAN: `t` differs (max 0.068; 0.151 on the fixed 3-branch star
  with `n_pairs = 3`).
* g6/g7: CMA 262,002 = 262,002 params, SPC 242,322 = 242,322, cap 500,000; REAL
  and NULL initial state hashes identical per family; shared skeleton identical
  across families at the same seed.
* g8: finite, non-zero gradient for `atom_encoder`, `bond_encoder`, `rho_base`,
  `rho_event` and the family modules; loss decreases; checkpoint round-trip
  exact.
* g9: forward keys ⊆ whitelist; mutating the label vector leaves predictions
  bit-identical.
* g10: split row hashes, full-label sha256 (= cache meta), SSL encoder sha256;
  official-test load refused; only `xs_common.py` (the boundary module) mentions
  the test cache.
* g11 coverage: CMA 44.0 undirected paths / molecule (median 43, max 110),
  1.77 paths / bond (83.3% of bonds carry ≥1 path); SPC **100.0%** of Fit8k
  molecules have a cut vertex, 9.2 cut vertices / molecule, 2.21 branches /
  cut vertex (max 4), 13.3 pairs / molecule.
* g12 synthetic learnability (synthetic graphs only, never in ZINC training):
  loss ratio after 200 steps 0.0045 / 0.0014 (CMA REAL/MEAN) and 0.011 / 0.018
  (SPC REAL/MEAN).

## 2026-10-30 — Stage I–IV formal runs

All runs: `res` (`a100-2`, 2 × A100-SXM4-40GB, driver 550.163.01, torch
2.5.1+cu124), process backend, `git_dirty = false`, 47,400 updates, single model.
Every number is reproduced from `runs/*/train.json` by
`shared/xs_analyze.py tables`.

| # | run dir | commit | best epoch | best split MAE | GPU·h | wall |
|---|---|---|---|---|---|---|
| 1 | `stage1-cma-real_s0` | 5a33e81 | 531 | Monitor 0.124922 | 0.362 | 01:31→01:57 |
| 2 | `stage1-cma-mean_s0` | 5a33e81 | 550 | Monitor 0.135866 | 0.435 | 01:31→01:58 |
| 3 | `stage2-spc-real_s0` | 5a33e81 | 433 | Monitor 0.145107 | 0.291 | 02:06→02:30 |
| 4 | `stage2-spc-mean_s0` | 5a33e81 | 607 | Monitor 0.143796 | 0.330 | 02:06→02:31 |
| 5 | `stage3-cma-real_s1` | f3b764f | 568 | Monitor 0.132780 | 0.353 | 02:30→02:54 |
| 6 | `stage3-cma-mean_s1` | f3b764f | 584 | Monitor 0.137275 | 0.437 | 02:30→02:56 |
| 7 | `stage4-cma-real_s0` | 43d2129 | 368 | valid 0.084278 | 0.353 | 03:45→04:09 |
| 8 | `stage4-cma-mean_s0` | 43d2129 | 392 | valid 0.101011 | 0.426 | 03:45→04:12 |
| 9 | `stage4-cma-real_s1` | c8dd9d4 | 445 | valid 0.084884 | 0.361 | 04:14→04:37 |
| 10 | `stage4-cma-mean_s1` | c8dd9d4 | 408 | valid 0.104589 | 0.429 | 04:17→04:42 |

Timing smokes (not ranked, cost counted): `stage1-timing-cma-real_s0` (126 upd,
1.64 s/epoch), `stage2-timing-spc-real_s0` (126 upd, 1.50 s/epoch),
`stage4-timing-cma-real_s0` (158 upd, 2.18 s/epoch) = 0.0064 GPU·h.

Budget: I 0.7965 / II 0.6216 / III 0.7902 / IV 1.5683 / timing 0.0064 =
**3.7829 GPU·h** (soft 14, hard 16), wall 3.78 h (cap 8 h). No reallocation of
unused stage budget.

Stage outcome (frozen rules, `analysis/decision_inputs.json`):
CMA seed-0 passes all five promotion conditions (`{"cma": true, "spc": false}`);
SPC fails all five and stops at Stage II; Locked read spent once (CMA only);
Stage IV authorised and run for both seeds.

### Deviations and fixes recorded after the first formal run

9. **`xs_analyze.py` run filter.** `tables` was picking up the un-tagged CPU smoke
   directory and emitting empty slice fields. Now only directories named `stage*`
   are ranked (smokes are still counted in the budget). Affects reporting only;
   no run, checkpoint or metric changed. The CSV/JSON artifacts were regenerated
   from the immutable `train.json` / `predictions_*.pt` files.
10. **`tail20` statistic added** to `xs_metrics.curve_stats` (task §6 asks for
    last-100 *and* last-20 curves). Recomputed for all runs; no selection rule
    uses it.
11. **valid1k slice pack** (`analysis/valid_slices.npz`, `valid_freeze.json`)
    added for Stage IV. Same frozen slice *definitions*, applied to valid rows
    with Fit8k as the support set; the Monitor-only slices (fixed worst-10, fixed
    row172) are reported as N/A on valid rather than transplanted.
12. **`decision` smoothing rule fixed.** The first implementation compared REAL's
    rolling-20 median against REAL's own best epoch, which is `>= 0` by
    construction and therefore always failed. The frozen rule ("the smoothed
    curve agrees in direction with the matched contrast") is implemented as
    REAL rolling-20 min median < NULL rolling-20 min median. This *changes no
    outcome*: SPC fails the other four conditions by a wide margin, and CMA
    already passed the manual comparison in `DECISION.md` D2 (0.12532 < 0.13636).
    Recorded because a decision rule was edited, not just plumbing.
13. **Run-tag typo.** `stage3-cma-mean-s1` was launched with `--tag
    stage3-cma-mean-s1` (dash) while `--result` declared `.../stage3-cma-mean_s1`,
    so `rr pull` reported the declared path as missing. The files were pulled with
    an explicit `--path` and renamed to `stage3-cma-mean_s1`; `train.json` inside
    is unmodified (tag string appears in `train.json`, dir name does not affect
    any computation).
14. `rr` refused two formal launches while the worktree was dirty
    (`local_commit != remote_commit`); the docs/analysis artifacts were committed
    and redeployed first, so every formal run still records `git_dirty = false`.

### Boundary attestation

* Official **test**: never opened. `xs_common.load_cache("test")` raises; gate g10
  re-checks statically that no module outside `xs_common.py` references the test
  cache; `analysis/*.json` all carry `official_test_loaded: false`.
* Official **valid**: opened only in Stage IV (`--monitor valid`), after the
  Stage III decisions were written to `DECISION.md` D1–D3.
* Locked1k: read exactly once, after the SHA-256 freeze of both checkpoints.

### Independent audit (post-hoc, after all runs)

Every formal run's headline number was re-derived from the saved prediction file
and the frozen label cache, independently of `train.json`:
`mean|pred − y|` over that run's monitor rows reproduces the reported
`best_monitor.monitor_mae` to **≤ 8.9e-9** for all 10 runs (the residual is
float32 storage of `pred`). The prediction row vectors are also checked against
the frozen Monitor/valid row lists inside `xs_analyze.tables`, which raises on any
mismatch. No discrepancy was found.
