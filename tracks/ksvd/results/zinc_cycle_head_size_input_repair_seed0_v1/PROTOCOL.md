# PROTOCOL — zinc-cycle-head-size-input-repair-seed0-v1

The machine-readable frozen protocol is
`tracks/ksvd/protocols/zinc-cycle-head-size-input-repair-seed0-v1.yaml`; this
file is the self-contained summary committed with the results.

## Question and single change

Give the existing independent cycle head `q` the actual graph size
`N = num_nodes` and `E = number of unique undirected pairs` in addition to the
frozen `T25` input.  Measure (1) full-fit exact-input conflicts before/after,
(2) a controlled pair of heads `Q0` (T25 only, size gate 0) vs `QNE` (T25+N/E,
size gate 1) on the old 8000/2000 fold, seed 0, CPU.

## What is frozen

* Fold: fit `165e87ef…` / dev `fb8b7806…`; fit strata `7702/260/33/5`, dev
  `1926/65/7/2`.  Reused diagnostic dev; results are exploratory.
* Chemistry branch: frozen `O_seed0` raw soup `61d4aeb…`; `O` predictions
  `5cee5782…`; `T25` array `dc2e1516…`; prep blob `968e82dd…`; train-only
  decomposition `target_decomposition.npz` for `c/g/k`.
* Train inputs only; official-valid/test never loaded (including old caches).
* No new Full training, no dictionary/ISTA change, no re-weighting, no
  input/width/epoch/optimizer search, no extra seed, no ensemble.

## Inputs

* `X25 = T25` (25 dims, unchanged).
* `N`, `E` from the actual graph; `X27 = [T25, (N−mean)/std, (E−mean)/std]`
  float32, scaler = unweighted graph-level fit-8000 mean/std (float64, floor
  1e-6); `N: 23.18275 ± 4.488552`, `E: 24.9435 ± 5.288602`.
* N/E contain no labels/ids/SMILES/precomputed cycle targets; they are
  recomputed from graph edges at deploy time.

## Structure gate (fit only, before any head)

1. identity/no-label/count invariants (NetworkX reference, renumbering,
   endpoint/self-loop/duplicate audits) all pass;
2. `global_min_l1(X25) − global_min_l1(X27) ≥ 0.001`;
3. one original conflict `k<=-3` row has X27 class target span `≤ 1e-10`.

Failure → `INPUT_REPAIR_NOT_SUFFICIENT`, no training.

## Heads (only if the structure gate passes)

```
z = Linear_T25(25,64) + gate * Linear_size(2,64,bias=False)
q = Linear(32,1)(SiLU(Linear(64,32)(SiLU(z))))
```

`Q0` gate 0, `QNE` gate 1; 3905 params each; shared fresh seed-0 init copied
from the original head (last layer `W=0`, `bias=median_fit(c)`), `W_S=0`;
Adam lr 1e-3 coupled wd 1e-5, clip 5, batch 128, unweighted mean `L1(q,c)`,
300 epochs = 18900 steps, soup 296–300, schedule generator seed 20261003 shared
by both arms.  `P_raw = h_raw + q`, one fit-only `b_P` per arm,
`P_cal = P_raw + b_P`; `h` frozen.

## Performance gate (dev)

* overall cal gain ≥ 0.003 and paired-bootstrap CI lower > 0;
* overall raw gain > 0;
* G0 cal worsening ≤ 0.001;
* identity / no-label / single-calibration / replay contracts pass.

Bootstrap 1000 draws seed 20261004; main CI uniform over all dev rows,
same indices both arms; G0 within G0; severity-stratified CI secondary only.
Fit-improvement marker (descriptive): 5 fit `k<=-3` rows q-MAE drop ≥ 25 %.

## Decision table

`INPUT_REPAIR_NOT_SUFFICIENT` / `INPUT_REPAIRED_NOT_FIT` /
`FIT_GAIN_NO_TRANSFER` / `TRANSFER_SIGNAL` / `INCONCLUSIVE` / `INVALID` as in
the round instructions.

## Stop rule

After Phase A and, only on gate pass, the two Phase-B heads, all compute
stops.  No third arm, no search, no Full run, no official split, no remote job,
no push/merge.
