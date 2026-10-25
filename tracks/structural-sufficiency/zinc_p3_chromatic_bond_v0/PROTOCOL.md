# PROTOCOL — zinc-p3-chromatic-bond-v0

**Frozen before any GPU second of this round.** After this file is committed,
only implementation bug fixes are allowed. No arm, layer, width, head count,
relation feature, reader, dropout, normalisation, learning rate, epoch budget,
metric, subgroup or decision threshold may change. Deviations go in `REPORT.md`
and invalidate the affected claim.

- Track: `tracks/structural-sufficiency/zinc_p3_chromatic_bond_v0/`
- Start commit: `072eeaa` (working tree additionally contains an untracked
  historical audit directory; it is not part of this round)
- Dataset: official PyG ZINC `subset=True`, train 10,000 / valid 1,000
- Metric: raw-y MAE, lower is better
- Official test: **never constructed, loaded or scored**
- Valid: exposed development split; all statements are valid-only
- Seed: 0 only; one controlled C/S pair; no seed-1 purchase in this round
- Budget: soft 3.5 A100·h, hard 4.0 A100·h (sum of both cards, not wall clock)

## 0. Pre-check result (recorded before implementation)

The repository does **not** already contain a valid trained implementation of
per-output-feature-channel key softmax with the structural relation entering
that channel's logits. DBT and GOS use per-head softmax plus per-channel value
gating. GOS's `MixK` softmax branch was never run, uses an invalid 5-D mask on a
4-D tensor, and normalises on the query axis (`dim=1`) rather than the key axis.
The public CSA layer does contain the intended per-channel key softmax; this
round is a research composition on P3 bond objects, not a CSA reproduction.

## 1. Fixed architecture

- P3 encoders + DCC-L `group_composer` + real-ownership `local_composer`.
- Initialisation from the seed-0 train-only masked-SSL encoder `006fabc1…`;
  **not** from any supervised DCC-L checkpoint.
- Physical-bond token `h0 = LayerNorm(Linear(290,48)([b,c_glob,c_loc,
  log1p(line_degree),log1p(n_bonds)]))`.
- 5 layers, hidden 48, 4 heads, head_dim 12, FFN 48→96→48, pre-LN,
  BatchNorm 0, attention dropout 0, residual scale `1/sqrt(5)`.
- Full same-molecule attention over all physical bond pairs, including self;
  key padding is the only mask. No adjacency/line-graph mask.
- Relation `r_ef ∈ R^22`: 21 lifted atom-RW bond-pair values (`k=0..20`) plus
  one chordless-cycle (length 3..18) co-membership indicator.
- Per layer: `rel_head 22→4` (no bias), `rel_channel 22→48` (no bias, zero
  init), `rel_value 22→48` (no bias); `Q/K/V 48→48` (no bias); `W_o 96→48`;
  reader `323→128→64→1` with SiLU.
- Arm C: per-channel logits `QK + rel_head + rel_channel`.
- Arm S: per-channel logits `QK + rel_head + within-head mean(rel_channel)`;
  the mean is taken over the 12 channel outputs of each head, then broadcast.
- Both arms register identical parameter names, shapes and initial tensors.
- Expected deployment parameters: 477,017. Hard cap 500,000.

## 2. Data and cache

- Only `ad_train.pt` and `ad_valid.pt` are read. `ad_test.pt` is never opened.
- Relation cache is built from the raw bond list and atom pointer only. Labels
  are not read by the builder.
- The cache contains one float32 `[E,E,22]` block per molecule, in the same
  physical-bond order as the P3 cache. Both arms consume the same collated
  tensor.
- `train_mean_log1p_line_degree = 1.308120486872196` (train 10k only).
- `log1p(line_degree)` is the number of junction incidences on the bond (equal
  to the line-graph degree; verified).
- No target formula, no RDKit target descriptor, no external data.

## 3. CPU gates (must all pass before GPU)

The following are hard gate families (the implementation records 34 concrete assertions). Any failure ⇒ `STOP_BEFORE_GPU`, no GPU purchased.

1. Actual `named_parameters()` ledger ≤500,000; exact sum; optimizer covers all.
2. Encoder/composer/global-normalisation state exactly equals the frozen P3/DCC
   seed-0 initial state; SSL source hash `006fabc1…`; no supervised DCC-L
   checkpoint loaded as initialisation.
3. C and S state dictionaries are tensor-for-tensor identical; initial forward
   outputs agree within 1e-5.
4. A non-zero `rel_channel` perturbation changes C and, for a within-head
   zero-mean perturbation, leaves S unchanged; a non-zero-mean perturbation
   makes C and S differ numerically.
5. Gradients reach `rel_channel`, `rel_head`, `rel_value`, Q/K/V, fusion, both
   composers, encoders and reader; all are finite; expected tensors have
   non-zero gradients.
6. Switching the 22nd (ring) relation column to zero changes the real-model
   output; the ring column receives a non-zero gradient.
7. Cross-molecule batch isolation: single-molecule and batched predictions
   agree; permuting the batch order does not change per-molecule predictions.
8. Atom relabelling, bond permutation and endpoint flip leave the output
   invariant within 1e-4; relation equivariance is checked separately.
9. Empty-bond, single-bond, acyclic and multi-ring synthetic inputs produce
   finite outputs; no NaN/Inf.
10. `model.train()` and `model.eval()` give identical outputs (dropout=0, no BN).
11. The P3 local encoding, raw-y target and global33 standardisation are
    unchanged (state comparison plus a forward comparison against DCC-L D0;
    target hash equals the cache label hash).
12. Attention is finite, rows sum to 1 per channel, is not exactly uniform, not
    saturated (max weight < 0.999), and has non-zero weight on non-adjacent
    bond pairs. With a non-zero `rel_channel`, C has non-zero channel
    diversity and S has exactly zero channel diversity.
13. Relation cache: offsets, bond counts, bond pointers and atom pointers agree
    with the P3 cache; a sample is recomputed independently and agrees; ring
    indicator passes triangle/C4/C5/benzene/naphthalene/spiro/acyclic/K4
    examples; support statistics are recorded.
14. Source, checkpoint, cache and config SHA-256 values are recorded in
    `analysis/cpu_checks.json` and `analysis/parameter_ledger.json`.
15. No code path reads `ad_test.pt`; the audit records `official_test_loaded:
    false`.

No post-hoc R² or "feature utilisation" threshold is a stop gate.

## 4. Timing rule (before formal training)

Both arms run at least 3 complete official-train epochs on the full 10k split,
each ending with a full official-valid forward. Record per-epoch seconds.

Projection per arm:

```
projected_seconds = 1.20 * 600 * mean_full_epoch_seconds
                    + startup_seconds + final_eval_seconds + 60
```

If either arm's projection, or the sum of both, exceeds the hard budget of
4.0 A100·h (or the sum exceeds the 3.5 soft budget with no headroom), **STOP**.
Do not rescue by reducing layers, batch size, epochs or the LR horizon.

No `torch.compile`; if compilation is ever introduced, both arms must use the
identical scheme and a forward/gradient equivalence check must precede timing.

## 5. Formal training recipe

| item | value |
|---|---|
| train rows | all official train 10,000 |
| valid rows | all official valid 1,000 |
| seed | 0 |
| epochs | 600 |
| batch | 128 (79 updates/epoch) |
| optimizer steps | 47,400 |
| optimizer | AdamW |
| LR | 1e-3 → 1e-5, per-step cosine, 500-step warmup |
| weight decay | 1e-5 |
| gradient clip | 5.0 |
| loss | raw-y L1 |
| valid | full 1,000 every epoch |
| selection | strict best-valid; ties → earliest epoch |
| EMA/SWA/soup/auxiliary | none |
| compile | none |
| batch order | same generator seed for both arms |

C runs on `res-gpu0`; S runs on `res-gpu1`. Same committed revision, same data
cache, same relation cache, same initial state. No arm gets a different width,
depth, dropout, normalisation or relation feature.

## 6. Pre-registered records

For each arm keep: best-valid, best epoch, W20, tail-100 median valid MAE,
final valid, eval-mode train MAE at best and final, full valid curve, per-
molecule valid predictions, attention channel diversity, `rel_channel` effective
amplitude, and the fixed-weight prediction change when the ring column is
switched off. W20 is the minimum over contiguous 20-epoch windows of the window
median valid MAE. Tail-100 is the median of epochs 501..600.

## 7. Decision rule (frozen)

Let `C_best`, `S_best` be best-valid; `C_W20`, `S_W20` the W20 values.
`gain = S_best - C_best` (positive means C better).

**Stability screen first.** `UNSTABLE_OR_UNDERFIT` if any of:

- any non-finite loss/valid value, or a non-finite checkpoint;
- `max |valid_mae[t] - valid_mae[t-1]| > 0.05` for any `t > 10`;
- best-valid > 0.15 for either arm;
- eval-mode train MAE at best > 0.03 for either arm;
- final eval-mode train MAE > 0.03 for either arm;
- train-mode and eval-mode predictions differ by more than 1e-5.

If this fires, report it as "the performance question was not validly tested";
do not write "Color fails in principle", and do not tune/retry inside this
round.

Otherwise apply, in order:

1. **`COLOR_GAIN_AND_COMPETITIVE`** iff
   `gain >= 0.003`, `C_W20 <= S_W20`, `C_best <= 0.073`, `C_W20 <= 0.075`,
   and no catastrophic ordinary/tail degradation. Catastrophic is defined as
   `C_ordinary_MAE > S_ordinary_MAE + 0.002` or
   `C_tail_MAE > S_tail_MAE + 0.010` (ordinary `y > -3`, tail `y <= -3`).
2. **`COMPETITIVE_COLOR_UNPROVEN`** iff at least one arm is competitive but
   rule 1 did not fire. Competitive is defined as
   `best_valid <= 0.078537` (same-seed historical DCC-L) **and**
   `W20 <= 0.080000`.
3. **`COLOR_SIGNAL_ONLY`** iff `gain >= 0.003` and `C_W20 <= S_W20` but C is
   not competitive (`C_best > 0.078537`).
4. **`NO_GAIN`** otherwise.

The historical DCC-L anchors are: seed-0 valid 0.078537 and 4-seed mean valid
0.075990. The 4-seed mean is reported as a historical background line, not as a
paired random variable. No single-seed difference may be described as a
family-level causal advantage.

“Competitive” is deliberately a pilot line, not a statistical-significance
claim. Paired bootstrap intervals, ordinary/tail subgroups and the fixed
DCC-L-worst-10 removal are descriptive and are reported for every C/S contrast.

## 8. Forbidden in this round

- Reading official test in any form.
- Loading a supervised DCC-L checkpoint as initialisation.
- Changing the arm difference beyond C vs S.
- Adding an adjacency/line-graph attention mask.
- Adding recursive local message passing on the original or line graph.
- Adding target-formula features, external supervision, ensembles or soups.
- Adding seeds, retuning, width/depth/batch/epoch rescue, or a third arm after
  seeing results.
- Claiming that C and S have matched effective capacity.
- Calling this a CSA reproduction or claiming the CSA mechanism as new.

## 9. Outputs

`DESIGN.md`, this `PROTOCOL.md`, `code/`, `configs/chr_v0.json`,
`analysis/cpu_checks.json`, `analysis/parameter_ledger.json`,
`analysis/timing.json`, `analysis/main_table.json`, `analysis/decision.json`,
both valid curves, per-molecule valid predictions, `REPORT.md` (Chinese,
≤2 pages). Model weights and large caches stay out of Git.
