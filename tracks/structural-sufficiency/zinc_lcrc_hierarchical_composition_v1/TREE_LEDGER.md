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
