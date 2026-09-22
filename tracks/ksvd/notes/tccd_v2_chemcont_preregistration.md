# Pre-registration — TCCD-v2 CHEM-CONT: label-free chemical continuity regularizer

Round name: **TCCD-v2 CHEM-CONT** (a single intervention on top of frozen TCCD-v2
Prototype-REL). Study: `zinc-context-gap`. Protocol version: `tccd_v2_chemcont`.

This is a **diagnostic single-arm intervention**. It does not amend TCCD-v2, does
not reopen the TCCD-v0/v1/v2/v3/v4/v5/v6/v7 results, and does not retrain BASE.

## 1. Scientific question

The frozen TCCD-v2 continuity audit
(`tracks/ksvd/audit/tccd_v2_continuity/`) shows that the learned local latent
`z` is an exact-anchored discrete motif vocabulary: the first non-identical step
consumes 61% (Z) / 75% (C) of the representation range, VeryNear Z cosine 0.428,
C cosine 0.301, same top-1 prototype 0.090, and excluding-Exact graded Spearman
0.41 (Z) / 0.37 (C).

**The one question this round answers:**

> Is the local discontinuity caused by the task objective never *asking* for
> chemical continuity, or is the canonical 714-D patch plus linear 714→64
> encoder itself unable to form a continuous chemical geometry?

Composition is explicitly out of scope this round.

## 2. Frozen BASE — BASE NOT RERUN

Everything below is read from existing artifacts; no BASE training, no BASE
continuity re-audit, no baseline rebuild.

* checkpoint: `tracks/ksvd/results/tccd_v2/prototype_rel_seed0_best.pt`
  (sha256 `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`).
* formal Gate A: `tracks/ksvd/results/tccd_v2/gateA_seed0.json`
  (commit `69a985a`): internal-dev best MAE `0.2862437069416046`, Top-5 soup
  `0.26635152101516724`, delta_comp `0.552572`, delta_proto `-0.124546`.
* vocabulary: `tracks/ksvd/results/tccd_v2/vocabulary_seed0.json`
  (64/64 active, effective count 62.5755, tau 0.094572).
* continuity audit:
  `tracks/ksvd/audit/tccd_v2_continuity/tccd_v2_continuity_audit.json` and
  `REPORT.md` (commit `27937d9`).
* claims/decisions:
  `records/claims/claim-tccd-v2-task-learned-prototype-vocabulary-20260922.yaml`,
  `records/decisions/decision-tccd-v2-stop-absolute-gap-20260922.yaml`.

## 3. Only new arm: CHEM-CONT (seed 0)

Architecture and protocol are **byte-for-byte the TCCD-v2 Prototype-REL path**:

* same 714-D canonical radius-2 input records (`tccd_v0_train_r2_...pkl`);
* same `714 → 64` linear encoder `W`, initialized from the frozen TCCD-v1 D0
  artifact;
* same `K = 64` cosine prototypes, same trainable temperature parameterization
  `tau = 0.05 + 0.95·sigmoid(a)`, init 0.2;
* same `BAG + CᵀRC` reader, linear head;
* same Adam, batch 32, lr 1e-3, wd 1e-5, clip 5, ≤240 epochs, patience 40,
  Top-5 soup, seed 0;
* same internal split seed `20260922` (8000 train / 2000 dev);
* same local-entropy and balance regularizers with the same one-shot
  calibration;
* same stopping protocol.

**The only change** is one added label-free latent term `λ_chem · L_chem`.
Prototypes `P`, temperature, reader, composition, relations and optimizer are
untouched by `L_chem` (it acts on `z` only).

## 4. Positive / negative definition (training split only)

Patch chemistry is decoded deterministically from the frozen 714-D coordinate
exactly as in the continuity audit (root, shell 1/2, atom/bond types, brute-force
canonical radius-1 key `L1`, canonical radius-2 key). No target value `y` is read.

For an anchor patch `i` (only training-split patches are ever anchors):

* **Positive `j` — VeryNear, non-identical:** `L1(i) = L1(j)`,
  `key(i) ≠ key(j)`, `graph(i) ≠ graph(j)`. This is the exact
  `Exact → VeryNear` cliff target.
* **Negative `k` — matched HardNegative:** `root(i) = root(k)`,
  `degree(i) = degree(k)`, `|n_atoms(i) − n_atoms(k)| ≤ 1`,
  `c1(i,k) ≥ 2` radius-1 edits, `graph(i) ≠ graph(k)`, with
  `c1 = ½·‖s1atom_i − s1atom_k‖₁ + ½·‖rbond_i − rbond_k‖₁`. This is the
  audit's deterministic HardNegative definition.
* Exact pairs are excluded. Random negatives are not used as the primary
  negative.

Pair cache is built **once**, deterministically, only from the internal training
split, with fixed seed `PAIR_SEED = 20260922`, cross-molecule, at most
`MAX_PAIRS_PER_ANCHOR = 4` positives and 4 negatives per anchor (rejection
sampling, ≤128 attempts). Anchors missing a valid positive or a valid negative
are dropped. Reported: anchors used, positive-pair count, negative-pair count,
unmatched-anchor fraction, tier sanity re-check counts.

## 5. Loss and λ calibration (no sweep)

`z̄ = z / ‖z‖`, `s_pos = cos(z_i, z_j)`, `s_neg = cos(z_i, z_k)`:

    L_chem = mean softplus(s_neg - s_pos)

No projector, no extra network, no prototype/argmax constraint, no triplet
margin sweep.

`λ_chem` is calibrated once on the first fixed training batch (first 32
training graphs), detached:

    λ_chem = 0.05 · L_task_initial / max(L_chem_initial, 1e-8)

and then frozen for the whole run. Never tuned against dev MAE or continuity.
Recorded: `initial_task`, `initial_chem`, `lambda_chem`, and the realized initial
weighted contribution.

    L_total = L_task + λ_local·L_local + λ_balance·L_balance + λ_chem·L_chem

## 6. Sanity checks before the formal run

1. `L_chem` has non-zero gradient to the encoder `W`.
2. positive/negative definitions use no label `y` (checked by scrambling `y`).
3. no dev pair is used for training (anchors/partners restricted to train).
4. permutation invariance / relation semantics unchanged (architecture untouched).
5. with `λ_chem = 0` the forward pass and task/local/balance losses match
   `tccd_v2.train_model` exactly on the same seed.
6. official test is never loaded (asserted in the runner).

No large BASE smoke training is run.

## 7. Readouts

Primary (representation, not score), on the **internal-dev** patches with the
existing audit code:

| metric | frozen BASE | CHEM-CONT |
|---|---|---|
| VeryNear Z cosine | 0.428 | ? |
| Moderate Z cosine | 0.332 | ? |
| HardNegative Z cosine | 0.206 | ? |
| Z graded Spearman excl. Exact | 0.41 | ? |
| VeryNear C cosine | 0.301 | ? |
| C graded Spearman excl. Exact | 0.37 | ? |
| VeryNear same top-1 prototype | 0.090 | ? |
| Exact→VeryNear Z cliff `1−cos` | 0.572 | ? |

Guardrail (not an optimization target): internal-dev best MAE and Top-5 soup
versus frozen BASE `0.286244` / `0.266352`.

## 8. Pre-registered interpretation (no post-hoc story)

* **A** — Z continuity clearly improves and MAE is roughly preserved:
  discontinuity is objective-induced.
* **B** — Z continuity improves but C-space barely improves:
  prototype coding/quantization is the next bottleneck.
* **C** — Z continuity barely improves: the 714-D canonical + linear 714→64
  representation is the binding limit; change the local input, not the loss.
* **D** — Z improves but MAE degrades materially: the chemical similarity notion
  conflicts with task distinctions / is too coarse at radius 1.
* otherwise `INCONCLUSIVE`.

"Clearly improves" requires VeryNear Z cosine to rise, graded Z Spearman to rise
in the same direction, and the Exact→VeryNear cliff to shrink; a single scalar
moving with all other continuity metrics flat is not enough. Bootstrap CIs are
reported. Small effects with overlapping CIs are reported as no clear change.

## 9. Stop rule

One arm, seed 0, one λ_chem. No seed-1 rescue. No λ/K/temperature/reader/relation/
radius sweep. No official test. If the run fails, the round reports the failure
and stops. Even a favourable MAE does not authorize any further run (TCCD-v7
closed the standalone performance route).

**Frozen before the formal CHEM-CONT run.**
