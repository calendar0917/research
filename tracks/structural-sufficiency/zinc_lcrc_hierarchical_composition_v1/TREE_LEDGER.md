# TREE LEDGER — LCRC-HC3-v1

Append-only chronological record of implementation, fixes, gates, runs and
budget. Scientific definitions live in `PROTOCOL.md` and are not edited here.
Times are local (+08:00). Host `res` = A100-40GB ×2, driver 550.163.01,
torch 2.5.1+cu124 (the same regime as the historical DCC-L / LCRC-A numbers).

## 2026-11-07 — setup, erratum, CPU phase

| time | event |
|---|---|
| ~10:00 | round directory created (`code/`, `tests/`, `configs/`, `analysis/gates/`, `runs/`, `cache/`); frozen LCRC code imported read-only |
| ~10:20 | structure enumeration implemented; train10k cache 880,906 directed records / 346,358 z triples / 249,279 bonds; valid 87,634 / 34,520 / 24,846 |
| ~10:40 | H/F models implemented; base state per-tensor equal to frozen builder; step-0 H/F == B bitwise; param counts 323,391 / 323,384 |
| ~11:00 | **task 0 erratum** recomputed on frozen Stage IV predictions with the full train10k vocabulary; old files untouched (`ERRATUM_stage4_support.md`) |
| ~11:20 | CPU gate suite `code/hc_gates.py` (15 groups) written; all pass on the first complete run after the fixes below |
| ~11:40 | `PROTOCOL.md` and `configs/frozen_protocol.json` frozen; `configs/freeze_manifest.json` written |

### Task 0 erratum (no GPU, no retraining)

Stage IV was trained on full official train10k but grouped valid rows with a
Fit8k radius-2 vocabulary. With the correct full-train10k vocabulary:

* any-unseen 251 (old 297), unseen-ordinary 225 (old 267), seen-ordinary 710
  (old 668); 46 rows move from unseen to seen, none the other way.
* REAL−MEAN unseen-ordinary: seed 0 −0.002998 (old −0.000703, CI
  [−0.016662,+0.010513]); seed 1 −0.014535 (old −0.012320, CI
  [−0.028392,−0.000721]).
* Full/ordinary/tail are unchanged by the vocabulary. The historical decision
  is not rewritten; the erratum is additive.

### Engineering fixes (all before any formal GPU run)

1. `hc_structure._molecules` initially read a non-existent `n_bonds_graph`
   cache key; switched to `diff(bond_ptr)`. No run affected.
2. The structure cache did not carry `bond_ptr`, and the frozen `lc_data.collate`
   does not emit `bond_off`; both were derived locally for the event-count
   collate. Cache version bumped `lcrc-hc3-v1-a → b` before any run.
3. `_shuffled_struct` / gate helpers: dead reference code removed; the learned-
   vs-secondhop gate now builds its one-molecule cache from the frozen raw
   arrays instead of zero feature tensors. Gate-only.
4. `hc_train` early-return for the resume gate (`stop_after_steps`) and an
   indentation fix in the `initial_state.pt` save; the formal path is unchanged.
5. LR gate expectations corrected to the frozen 1-based warm-up (step 0 = 0,
   step 1 = 2e-6, step 500 = 1e-3). No model/recipe change.
6. Stage-C baseline replay tolerance set to 1e-5 (CPU float accumulation
   against the saved fp32 predictions; MAE difference ≤ 7e-9). No change to the
   historical checkpoint or predictions.

### CPU gate summary (`analysis/gates/cpu_gates.json`, all_pass = true)

* g1 parameter table: B 237,519; H 323,391 = base + F2 51,216 + F3 30,496 +
  W_comp 4,160; F 323,384 = base + Fflat 81,705 + W_comp 4,160; H−F = 7.
* g2 initial state: H/F trunk tensor-wise equal to the frozen builder for seeds
  0 and 1; B initial hashes equal the historical `11c698ae…` / `d9422859…`;
  H/F `W_comp` identical and exactly zero.
* g3 step 0: H−B and F−B maximum absolute difference **0.0** (bitwise) for both
  seeds.
* g4 enumeration: 19 graphs (5 hand-built + real ZINC) against an independent
  ordered-quadruple brute force — 0 path/chord/centre/n_e mismatches.
* g5 CMA semantics: 60 real molecules; z triples and path attributes equal to
  the frozen CMA enumeration, `bond_npaths` equal.
* g6 contrast with the second-hop pilot: the old block is a fixed
  non-trainable 96→32/96→32 projection on two-hop paths; the new path is a
  trainable 64-d three-bond encoder (81,712 / 81,705 parameters) with four
  distinct atoms and three bonds.
* g7 shared content: H and F see identical `a`/`b` tensors (and identical to
  B), all path atoms distinct, all chords in 0..3.
* g8 gradient path: `W_comp` gradient is non-zero at step 0; path-encoder
  gradient is **exactly zero at step 0** and non-zero/finite from step 1
  onward (i.e. step 2 of the update sequence).
* g9 invariance: atom renumbering ≤ 4.8e-7, bond reversal 0, bond-order
  permutation 0, path-record order 0, batch split/merge ≤ 1.2e-7; all three
  arms.
* g10 reversal: H forward/reverse `q_path` difference exactly 0; F 1.5e-8
  (float32 average); `m_e ≡ 0` on every `n_e = 0` bond.
* g11 data boundary: mutating labels leaves predictions bit-identical; no
  label key in a forward batch; test split refused by all three loaders.
* g12 recipe/checkpoint: LR warm-up/cosine endpoints and monotonicity; optimizer
  covers all 323,391 parameters; state round-trip exact; a 63-step run resumed
  to 126 steps has the **same final state hash** as an uninterrupted 126-step
  run.
* g13 data/norm: split row hashes and label hashes match the frozen sources;
  the historical full-train norm fingerprint is reproduced; Fit8k norm differs
  from a Monitor-only norm (no held-out rows used).
* g14 coverage: 249,279 physical bonds; 41,629 with `n_e = 0` (exact zero),
  207,650 with `n_e > 0`, max `n_e = 9`.
* g15 Stage-C baseline: replaying the historical `lcrcA_s0/s1` checkpoints on
  official valid reproduces their saved predictions (≤ 3.9e-6) and MAEs
  (≤ 7e-9), so the Stage-C baseline is protocol-matched.

## 2026-10-30 — formal Stage A execution and stop

| time | event |
|---|---|
| 08:08 | `hc3-prep` on `res` completes; train/valid structure caches match the freeze manifest (`c1fe6321…`, `de4f43b8…`) |
| 08:10–08:12 | 2-epoch Fit8k timing: H 3.04 s/epoch → 0.637 projected GPU·h; F 3.55 s/epoch → 0.743 projected GPU·h. B was not separately timed; see deviation below |
| 08:13:28–08:54:58 | **H seed 0 Fit8k** completes 47,400 updates; best Monitor 0.120233 @ epoch 626; 0.691 GPU·h |
| 08:13:41–08:57:33 | **F seed 0 Fit8k** completes 47,400 updates; best Monitor 0.119856 @ epoch 581; 0.730 GPU·h |
| 08:59:55–09:29:11 | **B seed 0 Fit8k** completes 47,400 updates; best Monitor 0.121258 @ epoch 486; 0.487 GPU·h |
| ~09:35 | results pulled; `code/hc_analysis.py` added (read-only analysis; no training/selection/protocol change) |
| ~09:45 | Stage A gate evaluation: **STOP**; Stage B and Stage C not purchased |

All three formal runs used commit `cccf380143a45af50ba8af54ad485db2b9a2d859`
with `git_dirty = false`, exactly 47,400 updates, Fit8k training and Monitor
selection. Checkpoint SHA-256 (best): B `5cfc7874…`, H `67edefe7…`,
F `757ffd43…`.

### Stage A result

| arm | best Monitor | W20 | T100 | final | GPU·h |
|---|---:|---:|---:|---:|---:|
| B | 0.121258 | 0.121718 | 0.121746 | 0.121743 | 0.487 |
| H | 0.120233 | 0.120483 | 0.120516 | 0.120517 | 0.691 |
| F | 0.119856 | 0.120167 | 0.120194 | 0.120211 | 0.730 |

Paired deltas (negative favors first arm), best checkpoint:

* full: H−B −0.001025 [−0.005601,+0.003401]; H−F +0.000377
  [−0.005164,+0.005500].
* ordinary: H−B −0.001612 [−0.005997,+0.002276]; H−F −0.000831
  [−0.006411,+0.003539].
* unseen-ordinary (Fit8k vocabulary): H−B −0.004731
  [−0.018378,+0.006854]; H−F −0.005104 [−0.022879,+0.008669].
* tail: H−B +0.006322; H−F +0.015487.
* Monitor \ fixed W10: H−B −0.003326; H−F −0.002285.

**Pre-registered gates failed:** Δ(H−B) full (needed ≤ −0.004, observed
−0.001025), Δ(H−F) full (needed ≤ −0.003, observed +0.000377), Δ(H−B)
ordinary (needed ≤ −0.002, observed −0.001612), W20 H−B and H−F, T100 H−F and
final H−F. The unseen-ordinary point gate passes, but removing molecule 374
(Monitor position 39) changes it to −0.000385. That one row contributes
−0.001178 to full H−B and −1.7841 to the row's H−F delta; excluding it, full
H−B is +0.000154 and full H−F is +0.002163. This is a **post-hoc descriptive
sensitivity, not a gate**; the pre-registered decision already fails without
it.

Fixed W10: B 5.120592, H 5.347371, F 5.083470. Net `AE(B)−AE(H)` = +1.0246;
W10 sum = −2.2678; share = −2.213. The share gate passes only because H is
worse on W10. `Δ(H−B)` on Monitor \ W10 = −0.003326 passes pointwise.

### Stage A decision

**STOP.** Per `PROTOCOL.md` §6, no seed 1, no Stage C, no width/LR/path/loss
changes. Seed 1 was not launched; official valid/test were not used for Stage
C. The negative result is reported in `REPORT.md` / `DECISION.md` as-is.

### Budget and deviations

* Stage A formal: 0.487 + 0.691 + 0.730 = 1.908 GPU·h; timing ~0.004; total
  **1.911 GPU·h**. Caps: soft 6, hard 8; max two GPUs respected.
* **Deviation:** B did not receive a separate 2-epoch timing run. The H/F
  timings and the historical LCRC-A timings bound B; actual B cost 0.487 h is
  below both H (0.691) and F (0.730). No purchase decision depended on the B
  projection because B is required in every case, and Stage A stayed far below
  the cap. Recorded here rather than silently omitted.
* No training or scientific-definition change was made after the freeze. The
  only post-run code addition is `code/hc_analysis.py`, which reads saved
  predictions and caches; it is not imported by `hc_train.py` and cannot affect
  the formal runs.

### Follow-up rule

If this question is revisited, it requires a **new pre-registered round** with a
new mechanism or a substantially different evaluation; it must not reuse this
round's Stage A to tune H toward F. This round is closed.

### Provenance note on dates and manifest dirtiness

The setup header in `PROTOCOL.md` / `configs/frozen_protocol.json` says
`2026-11-07`; the actual git commits (`1fb3e76`, `cccf380`), run logs and this
ledger are **2026-10-30**. The former is a setup-date typo. The frozen files
were left byte-identical to preserve their hashes. Separately, the freeze
manifest records `git_dirty=true` because gate outputs were still untracked when
it was generated; its code/cache SHA-256 entries are explicit, and all formal
run `train.json` files record `git_dirty=false` at `cccf380`.
