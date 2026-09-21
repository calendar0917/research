# Pre-registration — TCCD-v2: Task-Learned Local Prototype Vocabulary (ZINC)

Round name: **TCCD-v2** (*Task-Learned Local Prototype Vocabulary*).

Lineage: TCCD-v0 fixed-coordinate patch audit → TCCD-v1 task-coupled dictionary.
This is a new round. It does not amend or reopen either earlier round.

Protocol version: `tccd_v2`. Study: `zinc-context-gap`.

## 1. Why reconstruction dictionaries are closed

TCCD-v0 established that canonical raw-patch Euclidean geometry lacks the
required structural continuity. TCCD-v1 established assignment-sensitive
composition and task reorganization, but the tied exact-sparse dictionary was
far below the matched dense local latent (`TASK-D 0.938099` vs `DENSE 0.403933`
on the internal CPU comparison). Therefore this round stops classical K-SVD,
OMP, IHT reconstruction, reconstruction loss, K-sweeps, radius-sweeps and
reader/encoder rescue.

The new hypothesis is that local environments should be learned as shared,
task-relevant prototypes after a learned local latent map, rather than as a
raw-patch reconstruction basis.

    x_v -> z_v = phi_theta(x_v) -> c_v over shared prototypes
    C_G = [c_1^T, ..., c_n^T]^T
    M_G^(r) = C_G^T R_G^(r) C_G
    h_G = [sum_v c_v, vec_sym(M_G^(r))_r]

The local prototype vocabulary is a task bottleneck, not a reconstruction
basis.

## 2. Frozen execution discipline

* Local repository is source of truth.
* Formal remote experiments use **GPU1 only** via `bash scripts/run_remote.sh 1 ...`
  or the durable detached equivalent.
* GPU0 is forbidden for formal experiments.
* Formal experiments require a clean committed revision and deployment.
* Remote checkout is compute-only; unknown remote dirty changes are never
  overwritten.
* Official ZINC test is never loaded, referenced, or evaluated.
* Official full-data training is blocked unless the preregistered internal
  gates authorize it.
* No K-SVD refit, OMP/IHT, raw reconstruction loss, K/radius/prototype-count
  sweep, temperature sweep, reader sweep, relation-set sweep, message-passing
  depth sweep, or official test.

## 3. Reused data, split and relations

Reuse the TCCD-v1 cached radius-2 canonical/raw patch records and relation
operators. No patch extraction or relation construction is repeated.

* canonical official-train records: `tccd_v0_train_r2_M14_A21_B3_n10000.pkl`
* internal split: seed `20260922`, 8,000 train / 2,000 dev
* patch input: `x_v in R^714`
* frozen relation set: overlap relation, three native bond relations, and
  global relative-position relation, exactly as TCCD-v1
* composition: BAG plus `C^T R C` symmetric upper-triangle features
* reader: the TCCD-v1 lightweight linear head and identical optimizer protocol
  (Adam, batch 32, lr 1e-3, weight decay 1e-5, gradient clip 5, at most 240
  epochs, patience 40, Top-5 soup reporting)

The historical TCCD-v1 DENSE-REL result is reusable only because it uses the
same split, input, 714-to-64 local encoder, relation set, reader, optimizer,
loss and stopping protocol. Reused value: best internal-dev MAE `0.403933`;
Top-5 soup MAE `0.387409`. If any protocol mismatch is found, exactly one
GPU1 Dense-REL rerun is allowed before Gate A.

## 4. Matched local encoder

Dense and Prototype arms share exactly the TCCD-v1 DENSE local encoder:

    z_v = phi_theta(x_v) = x_v W_theta,  W_theta in R^(714 x 64)

No GNN, Transformer, message passing, raw-graph bypass, or extra local input.
The encoder uses the same D0-shaped initialization convention as TCCD-v1
DENSE (`W_theta` initialized from the reused TCCD-v0 `(714,64)` artifact),
with the same normalization/optimizer convention. Prototype arms may only
apply the registered L2 normalization needed for cosine assignment; they may
not pass continuous `z_v` directly to the graph reader.

Frozen dimensions:

* local latent dimension `d = 64`
* prototype vocabulary size `K = 64`

## 5. Prototype assignment

Learn a prototype matrix `D_proto in R^(64 x 64)`. For every patch:

    zbar_v = z_v / ||z_v||_2
    dbar_k = d_k / ||d_k||_2
    s_vk = zbar_v^T dbar_k
    c_vk = softmax_k(s_vk / tau)

Temperature is one global trainable scalar:

    tau = 0.05 + 0.95 * sigmoid(a)

It is initialized to `tau = 0.2` and is never per-prototype or per-graph.
Prototype rows are initialized from a fixed-seed normalized Gaussian draw,
independent of labels and graph relations; the exact seed and fingerprint are
recorded in the run JSON. No reconstruction target is used.

The Prototype reader receives only `C_G` and fixed relation contractions. It
never receives raw `z_v`.

## 6. Prototype regularization

For each training batch, with `N_patch` real patches:

    H(c_v) = -sum_k c_vk log(c_vk + eps)
    L_local = mean_v H(c_v)
    cbar = mean_v c_v
    L_balance = KL(cbar || Uniform(K))
    L = L_task + lambda_local L_local + lambda_balance L_balance

Weights are calibrated once on the first fixed calibration batch, using
**detached initial magnitudes**, then frozen forever:

    lambda_local   = 0.05 * L_task / max(L_local, 1e-6)
    lambda_balance = 0.05 * L_task / max(L_balance, 1/K)

The `1/K` floor is the preregistered stable normalization for the balance term,
which can be numerically close to zero when initialization is nearly uniform.
It prevents an unstable infinite/very-large weight; the realized initial
contributions and raw magnitudes are always logged. No dev MAE is used to tune
these weights.

## 7. Gate 0 — implementation sanity

Before formal Gate A, targeted CPU tests and one GPU1 smoke must verify:

1. permutation invariance of patch inputs, assignment multiset, `C^T R C`, and
   prediction;
2. nonnegative assignments summing to one;
3. task gradients reach the local encoder, prototypes, and temperature;
4. no-bypass assertion: graph reader inputs are derived only from `C_G` and
   fixed relations, never continuous `z_v`;
5. fixed code multiset + shuffled graph assignment changes representation or
   prediction materially;
6. no NaN, no prototype norm collapse, and valid bounded temperature.

No Gate 0 check uses official test data.

## 8. Gate A — matched internal train/dev comparison

Primary internal-dev metrics use the same fixed split and report best-checkpoint
MAE as the decision metric; Top-5 soup MAE is reported diagnostically.

Arms:

* **Dense-REL:** historical matched TCCD-v1 DENSE result, or one GPU1 rerun if
  the exact protocol is not reusable.
* **Prototype-BAG:** task-learned prototype model using only `sum_v c_v`.
* **Prototype-REL:** task-learned prototype model using BAG plus all frozen
  `C^T R C` relation features. This is the primary model.
* **Prototype-REL-SHUFFLE:** evaluate the trained Prototype-REL checkpoint with
  the same per-graph assignment multiset randomly permuted across the fixed
  relation rows; do not permute relations. It is not retrained on shuffled
  assignments.

The Prototype-BAG model is trained with the same local encoder, prototype
assignment, task loss, regularizers, optimizer and stopping protocol, but its
reader input is BAG only. Prototype-REL and its shuffle control share one
trained checkpoint.

Composition gain:

    Delta_comp = MAE_ProtoShuffle - MAE_ProtoREL

* PASS if `Delta_comp >= 0.010`.
* FAIL / STOP if `Delta_comp < 0.010`.

Prototype bottleneck gap:

    Delta_proto = MAE_ProtoREL - MAE_DenseREL

* PASS if `Delta_proto <= 0.010`.
* STRONG PASS if `Delta_proto <= 0`.
* FAIL / STOP if `Delta_proto > 0.030`.
* AMBIGUOUS only for `0.010 < Delta_proto <= 0.030`; exactly one paired seed
  (`1`) is allowed, and the two-seed mean must be `<= 0.015` to pass.

No temperature, K, reader, encoder, regularizer, or relation adjustment is
allowed after an ambiguous or failed outcome.

## 9. Gate B — vocabulary quality (only after Gate A PASS)

Report, on the trained Prototype-REL model and internal-dev patches:

* active/dead prototypes;
* effective prototype count;
* per-prototype molecule coverage;
* top-1 and top-8 assignment mass;
* mean local assignment entropy;
* global assignment entropy;
* final learned temperature;
* top activating real patches and existing structural-coherence diagnostics.

A prototype is dead if dataset-level mean assignment is below `1e-6` (fixed
before observing results). At least `48/64` prototypes must be active. No small
minority may hold the overwhelming assignment mass; top-8 mass is reported and
interpreted against the preregistered collapse criterion `top-8 > 0.80`.

If prediction passes but vocabulary collapses, the result may only be called a
low-rank task bottleneck; the reusable chemical vocabulary claim fails and the
round stops.

Structural continuity remains diagnostic only; no AUC threshold is reused.

## 10. Gate C — absolute representation gap

Only if Gate A and vocabulary quality pass, inspect internal Prototype-REL MAE.
If `MAE_ProtoREL > 0.30`, stop before any full-data run and conclude that the
simple 714-to-64 local encoder plus second-order composition has an absolute
capacity gap.

Only if internal `MAE_ProtoREL <= 0.30` and all prior gates pass may one full
official-train to official-valid GPU1 run be authorized. Official test remains
frozen.

For the authorized full-data run only:

    Delta_abs = MAE_ProtoREL_full - 0.119818

Bands:

* competitive: `Delta_abs <= 0.015`
* promising but insufficient: `0.015 < Delta_abs <= 0.05`
* not viable: `Delta_abs > 0.05`

## 11. Seeds and stopping

Primary seed is `0`. Seed `1` is permitted only for the preregistered
Prototype-vs-Dense ambiguity. Formal experiments are sequential on GPU1. Any
composition FAIL, bottleneck FAIL, vocabulary collapse, or internal absolute
MAE above `0.30` stops the round and forbids later stages.

## 12. Durable record

Every formal result records local and remote commits, GPU1, seed, split, wall
clock, architecture, exact losses, initial calibration magnitudes and weights,
final temperature, MAE, usage diagnostics, PASS/FAIL and stopping reason.
Update `tracks/ksvd/STATE.yaml` and add Gate A/vocabulary/absolute notes only
after results are pulled and analyzed locally.

**Frozen before formal experiments.**
