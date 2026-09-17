# FSAR-R2-AR0-EDGE pre-registration — edge structural-role ↔ bond-type assignment residual

Branch `exp/fsar-r2-ar0-edge-binding-zinc`, created from the durable AR0 verdict
tip `69b8fe9`.  Pre-registered before any formal ZINC run.  **Official ZINC test
is never loaded.**

This round continues the same fixed radius-2 explicit-coordinate framework.  It
keeps the verified node assignment residual `B_V` and adds exactly one new
first-order statistic: the assignment between **edge structural role** and
**bond type**.  It does not touch the model definition, protocol, gate or
verdict of the node round.

## 0. Question

> With the already-supported node-level assignment residual held fixed, does
> the real "edge structural-role ↔ bond-type assignment" provide a stable
> predictive increment over the node baseline **and** over a matched,
> assignment-independent edge marginal control, under the same fixed radius-2
> pure-topology coordinates and identical training protocol?

## 1. Precondition (phase-1 mechanism round, revision `2d66ca7`)

No correctness failure was found: `C`/scaler correct, permutation zero-mean
holds, cache/model mapping correct, branch outputs map to the correct
checkpoints.  The AR0 caveat "`C` is close to rank-1" was an **audit slicing
bug** and is retracted; the full flattened `C` has effective rank `~57`
(`C~` `~116`).  The node assignment signal is a stable, high-dimensional
functional: branch outputs agree across seeds at Pearson `≥0.988`, ~67–71 % of
the `MB` gain is recovered by a frozen `M0` + linear `C`, and `PC1` explains
`<2 %` of the trained `B`.  Full analysis:
`notes/fsar_r2_ar0_node_binding_mechanism.md`.  Therefore the edge round is
authorised.

## 2. Fixed definitions (no sweep, no new primitive)

Radius stays 2.  The `S` encoder, `A` marginal, the node coordinate `phi_v`
(65-D), the node statistic `C_V` and the node branch `B_V` are **unchanged**.

For every original **undirected** edge `e = (u, v)` (PyG directed entries are
canonicalized to one entry per bond):

```
psi_e = [ phi_u + phi_v ,  |phi_u - phi_v| ]   in R^130
```

`psi_e` is strictly endpoint-swap invariant and chemistry-free.  Forbidden in
`psi_e`: bond type, atom type, typed path, chemistry-aware edge descriptor,
learned message passing, any new motif.  The 4-padded bond-type space is
verified from the raw dataset schema (`data/ZINC/raw/bond_dict.pickle`:
`NONE=0, SINGLE=1, DOUBLE=2, TRIPLE=3`) and matches `A`'s existing bond
multiset.

## 3. Edge assignment residual

With `r_e = onehot(bond_type_e) in R^4`:

```
J_E = sum_e psi_e r_e^T
P_E = (1/m) (sum_e psi_e) (sum_e r_e)^T
C_E = J_E - P_E = sum_e (psi_e - psibar)(r_e - rbar)^T     # sum-centred
```

For fixed topology and fixed bond-type multiset, `E_pi[C_E] = 0` exactly under
a uniform random permutation of bond types across the undirected edges.
Train-only per-coordinate RMS scaling, no dataset-mean subtraction:

```
D_E = sqrt(E_train[C_E^2] + eps);  C~_E = C_E / D_E
D_PE = sqrt(E_train[P_E^2] + eps); P~_E = P_E / D_PE
```

zero-RMS coordinates are masked to zero and counted.

## 4. Three models (only these)

```
BV    yhat = F0([z_S, a]) + <W_B, C~_V>
BVE   yhat = F0([z_S, a]) + <W_B, C~_V> + <W_E,  C~_E>
BVEM  yhat = F0([z_S, a]) + <W_B, C~_V> + <W_ME, P~_E>
```

`W_B in R^{65x28}`, `W_E, W_ME in R^{130x4}`: bias-free plain parameters,
zero-init, no MLP / activation / hidden bypass after either statistic.  At the
same seed the base `F0` and `W_B` must be bit-identical across all three
variants; `BVE`/`BVEM` differ only by the single edge parameter; all three
predict exactly the base + node term at step 0.  No other model, width, radius,
primitive, gate or optimizer variant is created.

## 5. Pre-registered correctness tests (`tests/test_fsar_r2_ar0_edge.py`)

1. edge endpoint-swap invariance `psi_uv = psi_vu`
2. `psi_e` chemistry purity
3. node relabel invariance of `BV`/`BVE`/`BVEM`
4. bond-assignment permutation leaves `S`, `A`, `C_V`, `P_E` invariant
5. bond-assignment permutation changes `C_E` on an asymmetric toy
6. exact small-graph enumeration `mean_pi C_E = 0`
7. scalar `mean_pi <W_E, C_E> = 0`
8. undirected bond canonicalization (directed entries collapse to one bond)
9. train-only RMS scaler (not a centred std)
10. batched edge statistics equal the numpy reference
11. `W_E` / `W_ME` nonzero task gradient
12. no edge bypass (zeroing the statistic reproduces base + node term)
13. `BV`/`BVE`/`BVEM` shared initialization identity
14. zero-init prediction identity
15. parameter counts
16. assignment-only positive control (+ marginal-only negative control)

## 6. Synthetic controls (must pass before any formal ZINC run)

**Positive (assignment-only).**  Fixed 5-node tree with non-equivalent edge
roles (`0-1, 1-2, 2-3, 2-4`), all-`C` atoms (so `C_V = 0`), fixed bond multiset
(one `DOUBLE`, three `SINGLE`); only *which* edge carries the `DOUBLE` changes,
and the label is `1` iff that edge is the branch edge `(1,2)`.  `A`, `S` and
`P_E` are identical across examples; only `C_E` differs.
Requirement: `BVE` learns (valid MAE `<< BV`), `BV` cannot distinguish, `BVEM`
does not gain the real assignment, `||W_E|| > 0`.

**Negative (marginal-only).**  Label depends only on the topology marginal and
the atom/bond counts; then the atom assignment and the bond assignment are
randomised within every molecule.  Requirement: `BVE` shows no stable,
repeatable advantage over both `BV` and `BVEM`.

If the positive control fails, **no formal ZINC run is launched**.

## 7. Short runs

`BV`/`BVE`/`BVEM` seed 0 for at most 60 epochs (AR0 smoke protocol).  Used only
for finiteness, gradient, branch liveness, curve sanity and memory.  Short-run
MAE does not select a model.

## 8. Formal protocol (inherited verbatim from AR0)

Adam, `lr = 1e-3`, `weight_decay = 1e-5`, batch 128, L1 loss, grad clip 5.0,
no scheduler, `max_epochs = 240`, `patience = 40`, best official-valid
checkpoint, fixed equal-weight Top-5 soup, `torch.use_deterministic_algorithms
(True)`.  No optimizer / width / radius / primitive change.  Official test
locked.

**Budget.**  First round: `3 models x 2 paired seeds = 6 runs` at seeds `0, 1`.

**GPU policy.**  GPU 0 is occupied by an unknown third-party process and is
never touched or co-tenanted.  Only GPU 1 is used, and (per the current
instruction) all runs are **sequential** on GPU 1.  No DDP.  Every launched job
is waited on inside the agent's shell call.

## 9. Formal gate (pre-registered, automatic)

```
Delta_edge         = MAE(BV)   - MAE(BVE)
Delta_edge_control = MAE(BVEM) - MAE(BVE)
```

Seed 2 is authorised only if **both** seeds 0 and 1 satisfy
`Delta_edge > 0` **and** `Delta_edge_control > 0`.  A small but positive delta
still buys seed 2 by the sign gate, but the result is then labelled a weak
effect.  If the direction flips on the two seeds: no extra seed, no tuning, no
edge-role change, no radius change — go straight to mechanism diagnostics and a
verdict.

Verdicts allowed (exactly one): `EDGE_ASSIGNMENT_SUPPORTED`,
`EDGE_ASSIGNMENT_NOT_SUPPORTED_OR_UNSTABLE`, `EDGE_EXPERIMENT_INVALID`.

## 10. Evaluation-only bond-type permutation diagnosis

On the trained `BVE` soup: keep binary topology, atom attributes, atom
assignment and the bond-type multiset fixed, and randomly permute bond types
among the **undirected** edges.  Verify that `S`, `A`, the node statistic `C_V`
and `P_E` are exactly invariant while `C_E` and the prediction change.  Report
`actual MAE`, `MAE(E[yhat_perm])` and `E[MAE(yhat_perm)]` separately (never
confused).  This is a mechanism diagnostic, not a substitute for the
independently trained controls.

## 11. Branch liveness (each formal soup)

Node branch: `||W_B||`, output std.  Edge branch: `||W_E||` (or `||W_ME||`),
output std, exact zero count, effective non-masked coordinates, task gradient
norm.  If the exact-zero fraction equals the zero-RMS mask fraction it is **not**
called learned sparsity.

## 12. Forbidden this round

No official ZINC test; no pair-level node binding; no second-order
attribute-pair interaction; no radius > 2; no C4/C5; no RRWP; no homomorphism;
no attention / Transformer; no message passing inside the explicit primitive
definition; no width/depth sweep; no optimizer/wd sweep; no feature search; no
`B-full` teacher / distillation; no mixed descriptor; no seed hunting; no
changing `A_exact`; no validation-time change of the edge-role definition; no
automatic "fix the model" if the edge round is negative.  Pair-level `B^(2)` is
**not implemented** this round; it may only be proposed in the final discussion
if a first-order collision witness is found.

## 13. Files

* model / features: `experiments/luyin16/fsar_r2_ar0_edge.py`
* runner: `experiments/luyin16/zinc_fsar_r2_ar0_edge.py`
* tests: `tests/test_fsar_r2_ar0_edge.py`
* results: `results/fsar_r2_ar0_edge/` (gitignored)
* durable note: `notes/fsar_r2_ar0_edge_binding.md` (written after the pulled
  formal results)

## 14. Local pre-flight (recorded before the remote deploy)

* `tests/test_fsar_r2_ar0_edge.py`: **17/17 pass**.
* `sanity`: **16/16 checks pass**, `all_pass = true`.
* `parameter_accounting.json`: `BV 24,797`; `BVE 25,317`; `BVEM 25,317`
  (node binding `1,820`; edge binding `520`; shared-init identity `true`).
* `feature_audit.json` (500 molecules): `psi` 130-D, effective rank `20.1`,
  22 zero-std coordinates, no NaN/Inf; `C_E` effective coordinates `324/520`;
  `P_E` `330/520`; `CE_rms_max 0.824`, `PE_rms_max 75.72`; train-only,
  no dataset mean.
* `preprocess`: 10,000 train / 1,000 valid, 36.6 s, official test not loaded.
* `synthetic_controls.json`:
  * positive **pass** — `BV 0.2531`, `BVE 0.0173`, `BVEM 0.3228`, `||W_E||≈0.05`;
  * negative **pass** — `BV 0.0206`, `BVE 0.0666`, `BVEM 0.0420` (no `BVE`
    advantage).

## 15. Durable outputs

After the formal results are pulled back and analysed locally: write the
complete mechanism/verdict note, update claims/decisions, update `STATE.yaml`
if appropriate, and commit + push.  Every scientific conclusion is based only
on pulled formal results.
