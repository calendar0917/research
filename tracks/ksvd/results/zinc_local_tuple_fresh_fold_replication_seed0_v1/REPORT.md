# REPORT — `zinc_local_tuple_fresh_fold_replication_seed0_v1`

**Question.** Does the historical B→M_J chemical-component (`g = y − c`) improvement survive on a
pre-fixed new 8000/2000 fold (new dev ∩ old dev = ∅, new dev ⊂ old fit, |new fit ∩ old fit| =
6000)? Exactly two seed-0 trajectories: `B` (original compressed M_g skeleton, 267,611 params) and
`M` (same skeleton + original M_J local tuple encoder and fusion first-layer injection, 297,499
params). Fit-dependent objects (label constants, body standardizers, tuple phi scaler, kappa
sample) are new-fit-only. Official valid/test never loaded.

## Opening table — the five closing questions

| # | question | answer |
|---|---|---|
| 1 | Does the old B→M_J `g`-MAE advantage hold on the new fold? Five gate items? | **Direction only, not at the frozen bar.** All four endpoints stay M-better in point estimate: G0 cal `+0.000549`, overall cal `+0.000888`, G0 raw `+0.002520`, overall raw `+0.003093`. Gate: `G0_cal≥0.003` **false**; `overall_cal≥0.003` **false**; `G0_cal CI lower >0` **false** (`[−0.003438,+0.004754]`); `G0_raw>0` **true**; `overall_raw>0` **true** → **2/5**, classification `DIRECTIONAL_NOT_CONFIRMED`. The old G0 cal gain was `+0.003659`; on the new fold it is `+0.000549` (≈6.7× smaller) and the overall cal gain falls from `+0.004336` to `+0.000888`. |
| 2 | Best `g`-MAE and the gap to 0.09? G0 contribution and the zero-other-groups floor? | Best = M dev overall cal `0.099304` (G0 cal `0.097383`; raw overall `0.100098`, G0 raw `0.098265`). Gap to 0.09: `+0.007383` on G0 cal, `+0.009304` on overall cal. G0 (k=0, n=1935) contributes `0.094218` to the overall MAE; even zeroing every non-G0 error leaves overall cal ≥ `0.094218`, still `+0.004218` above 0.09. This is an error-budget statement, not a theoretical lower bound. |
| 3 | Do raw and cal agree, is it bias/single-row driven, do fit and dev move the same way? | Same sign everywhere (M better) but different amplitude: raw gains are ~3–4× the cal gains, so the fit-median bias *reduces* rather than creates the advantage (B bias `−0.022919`, M bias `+0.013944`). Dropping the unique worst mean-error dev row (pos 1403, gid 6839, in G0) leaves G0 cal `+0.001026`, overall cal `+0.001349` — direction unchanged, no single-row dominance. Fit/dev are same-direction this round: M fit cal `0.030535` < B `0.031167`, M dev cal `0.099304` < B `0.100192` (on the old fold M was *worse* on fit). |
| 4 | Are targets/scalers/sample new-fit-only, and did the actual training graphs change? | **Yes.** New constants fit on 8000 new-fit rows only; permuting new-dev raw label fields leaves them bit-identical; every body standardizer equals a direct new-fit recomputation exactly; phi/kappa sample are new-fit-only; the payload's structure arrays are all-10000 raw indices (`node_sizes`/`root_atom` match the env cache). Actual training global graph-ID stream sha256 `69187f13fba5def8…` is identical across arms (paired) and differs from the old-fold position schedule hash `7b11a529…`; both arms ran the expected `7b11a529` position schedule, 15,120/15,120 steps. |
| 5 | Performance, partition stability or mechanism? Which luyin19 claims remain untested? | Confirmed: the added local-interface scheme still has a positive-direction `g` advantage on the new fold, but the frozen replication bar is not met — calibrated partition stability is **not** confirmed. Mechanism: `HEALTHY` (A drift `0.713`, `W_loc` norm `7.46`, 0/64 dead code dims, code RMS `0.507→0.339`, injection RMS `0→0.594`, dev zero-ablation mean `|Δpred| = 0.574`, cal MAE `0.0993→0.5841`). Untested: dictionary-vs-MLP win/equivalence, joint-vs-independent correspondence, incidence-permuted control, and luyin19's transferable structure–property claim. None of those were arms this round, so this must not be written as "the dictionary/correspondence works". |

## 1. Frozen setup and input witnesses

Protocol/contract/evidence-scope/errata and all fresh artifacts were committed **before** the two
runs (`PROTOCOL.md`, `METHOD_CONTRACT.md`, `EVIDENCE_SCOPE.md`, `ERRATA.md`, commit `2e4ee3f70b14`).

* Fold: `fresh_fold.npz` sha256 `941125505af5…`; new fit `2a21cb8771f6…`, new dev `270ab4126b0f…`
  (match the pre-registered hashes); sizes 8000/2000, disjoint, cover 10000,
  `new_dev ∩ old_dev = ∅`, `|new_fit ∩ old_fit| = 6000`. Canonical-group cross: 1 shared group,
  1 fit row and 1 dev row.
* Targets: `fresh_targets.npz` sha256 `2660ecbe16bd…`; new constants `sigma_logP=1.434428173759835`,
  `sigma_SA=0.8327498022638992`, `mu_SA=-3.1924626828735625`, `sigma_cycle=0.2885551506645267`,
  `mu_cycle=-1.3447189184664423e-05` with `mu_logP` fixed at `2.4570953396190123`. New k groups:
  fit `7693/269/33/5`, dev `1935/56/7/2`. Lineage anchor: the same raw fields refit on the old fit
  reproduce the historical saved constants (`sigma_logP` Δ `6.7e-16`, others 0.0) and old `k`/`c`
  exactly. Dev raw-field shuffle → constants unchanged (all diffs 0.0).
* Prep: `fresh_prep.npz` sha256 `5804dadb54e2…`; all eight fit mean/scale vectors equal the direct
  new-fit recomputation (max abs 0.0) and the cache-recovery constants are used only to invert.
* Payload: `fresh_tuple_payload.npz` sha256 `9acf100136ec…`; phi scaler on new-fit realized tuples,
  `kappa_D = 2.0631041526794434` on the new 8192-root sample (sha `6702326d5059…`, all in new fit);
  `kappa_M = 2.00007850651312` (one-shot label-free RMS match).
* Init/stream witnesses: 36 shared tensors byte-identical (`max Δ 0.0`), `A_raw = D_loc_init.T`,
  `W_loc ≡ 0`, post-construction RNG states equal (`a2e8a8ab…`); both arms' training RNG starts
  equal (`1ccf1725…`); the position schedule hash is the frozen `7b11a529…`; per-graph operator
  checks on 5 non-first dev molecules show model codes vs reference max abs `0.0`, batch-vs-single
  concat max abs `0.0`, and raw-graph atom order matches the env cache. Smoke (fit-only, 3 steps
  each, states discarded): both arms pass; M shows `W_loc` task grad nonzero at step 1, `A_raw`
  grad zero at step 1 then nonzero after the first `W_loc` step; save/reload forward exact.

## 2. Performance

Dev calibrated `g`-MAE (one fit-median bias per arm):

| arm | fit overall cal | dev overall raw | dev overall cal | dev G0 raw | dev G0 cal | bias |
|---|---:|---:|---:|---:|---:|---:|
| B | 0.031167 | 0.103191 | 0.100192 | 0.100786 | 0.097932 | −0.022919 |
| **M** | **0.030535** | **0.100098** | **0.099304** | **0.098265** | **0.097383** | +0.013944 |

Paired bootstrap (1000 draws, seed `20261005`, shared endpoint indices; positive = M better):

| gain | point | 95% CI |
|---|---:|---|
| G0 cal | +0.000549 | [−0.003438, +0.004754] |
| overall cal | +0.000888 | [−0.003031, +0.004820] |
| G0 raw | +0.002520 | [−0.001438, +0.006729] |
| overall raw | +0.003093 | [−0.000811, +0.006975] |

Bootstrap witnesses pass exactly (identical predictions → 0 with `[0,0]`; swapped mirror; constant
shift exact/bounded). Pre-fixed drop-worst-row sensitivity: G0 cal `+0.001026`, overall cal
`+0.001349`.

Per-k calibrated MAE with contribution `Σ|err|/N_dev` (N=2000):

| arm | k=0 (1935) | k=−1 (56) | k=−2 (7) | k≤−3 (2) |
|---|---:|---:|---:|---:|
| B | 0.097932 (.094749) | 0.171035 (.004789) | 0.114970 (.000402) | 0.251070 (.000251) |
| M | **0.097383 (.094218)** | **0.155907 (.004365)** | 0.135217 (.000473) | 0.247386 (.000247) |

The overall M advantage is G0 (+0.000399 contribution) plus k=−1 (+0.000424 contribution), partly
offset by k=−2 (−0.000071). Group-contribution and group-gain sum identities hold to ~1e-17.

Side-by-side with the old fold (read-only, different dev; not pooled):

| endpoint | old B | old M | old gain | new B | new M | new gain |
|---|---:|---:|---:|---:|---:|---:|
| G0 cal | 0.101875 | 0.098216 | +0.003659 | 0.097932 | 0.097383 | +0.000549 |
| overall cal | 0.103717 | 0.099381 | +0.004336 | 0.100192 | 0.099304 | +0.000888 |
| G0 raw | 0.107251 | 0.100434 | +0.006817 | 0.100786 | 0.098265 | +0.002520 |
| overall raw | 0.108956 | 0.101414 | +0.007542 | 0.103191 | 0.100098 | +0.003093 |

Both models score better on the new dev (G0 share 96.75% vs 95.75% old; k=−1: 56 vs 73 rows), and
the M advantage shrinks by ~4–7×. Old/new folds are never pooled into a "4000 independent rows"
confirmation.

## 3. Frozen classification and mechanism

Frozen order fired `DIRECTIONAL_NOT_CONFIRMED`: the performance gate is not fully passed, but G0
cal and overall cal gains are both > 0. `PERFORMANCE_REPLICATED` conditions were individually
recorded (2/5); `NOT_REPLICATED` not reached. This is **not** evidence of M_J–B equivalence: the
G0 cal CI `[−0.003438,+0.004754]` is wider than ±0.003, so neither an advantage of ≥0.003 nor
equivalence is established.

Mechanism `HEALTHY`: soup A drift `0.713`, soup `W_loc` norm `7.455`, 0 dead code dims, root-code
RMS `0.507→0.339` over epochs, injection RMS `0→0.594`, first-step probes at epochs 1/40/120/240
have `W_loc` task gradient `0.247/0.332/0.221/0.136` and `A_raw` gradient `0.0/0.028/0.037/0.032`.
Dev zero-ablation of the local path at the soup changes mean `|Δpred|` by `0.574` and cal MAE
`0.0993 → 0.5841` (dependence only, not an information share). Replay: both soups independently
reloaded; max abs vs stored `1.4e-6` (B) and `9.5e-7` (M) ≤ 1e-5.

## 4. Boundaries

One new fold, one seed, two fixed trajectories, one soup each, no search. The new dev was in the
old fit (old models trained on it) and the fits overlap; this is a partition-stability check of
the training-side decomposition, not an unseen test. The comparison isolates the added local
interface as a package (capacity + optimisation included), not correspondence, structure or a
dictionary. No 0.09 crossing; no official-valid/test read is bought. luyin19's transferable
structure–property relation, the dictionary necessity, and J-vs-I correspondence remain untested
by this round and must not be inferred from it.

## 5. Direct answers

1. **Old B→M_J body gain on the new fold:** sign preserved on all four endpoints, magnitude much
   smaller; gate 2/5 (`G0_raw>0`, `overall_raw>0` only) → `DIRECTIONAL_NOT_CONFIRMED`.
2. **Best score / 0.09 gap:** M overall cal `0.099304` (G0 `0.097383`); gap `+0.009304` overall
   (`+0.007383` G0). G0 contributes `0.094218`; zeroing all other groups still leaves `≥0.094218`.
3. **raw/cal and bias:** same direction; cal gains smaller than raw gains; not bias-created; no
   single-row dominance; fit and dev move together this fold.
4. **Fresh pipeline / stream:** targets, scalers, sample all new-fit-only and witness-checked; the
   actual training graph-ID stream changed and is arm-identical.
5. **What is confirmed:** a small positive-direction, CI-crossing `g` advantage of the added local
   interface on one new partition, with an active mechanism — not calibrated performance
   replication, not dictionary/correspondence evidence.

**Action:** freeze this local interface; stop further encoder/fold/seed/threshold investment. Keep
M_J documented as an implementation-verified working reference for the local channel and keep the
old dictionary/D_J references; shift the next architectural responsibility to the G0/global
residual rather than this interface.
