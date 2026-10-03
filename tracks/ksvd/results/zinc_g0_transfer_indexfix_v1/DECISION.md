# DECISION — zinc-g0-transfer-indexfix-v1

Start = first tool call; stop-new-compute after Phase 2; this note = finalisation.
All compute stopped.  No training, no GPU/remote job, no optimizer/backward.

## Outcome

The located bug (missing per-graph endpoint offset in the edge keys) is
**confirmed and fixed**, proved by four invariants plus node-key invariance.  The
training path is **not** affected (`env_collate` offsets via `Batch.ptr`).

Part C was recomputed under the **unchanged frozen protocol**:

* edge keys are ~6–7× more numerous (unique support keys 169 → 1037/1095;
  descriptor 207 → 2123; joint 317 → 1257/1317);
* `structure_rare` is ~13–28× larger than the buggy value (still ~1–2 % of
  molecule-edges);
* the main exposure `joint_rare_marginals_seen` is still ~0.2 % (residual) /
  ~0.35 % (descriptor) per molecule;
* the `≥0.25` high group is still **empty** in every fold × view × key × split;
  matching still yields **0 pairs**.

## Decision (case 3 of the task)

**This coverage proxy, under the fixed definition, still cannot locate the
bottleneck.**  The conclusion is *not* that the rare-combination hypothesis is
falsified, and *not* that structure-semantics fusion is useless.  The pre-fix
statement "high group empty / matching not estimable" survives, but it now rests
on correct edge keys; the old "coverage saturated / structure_rare unpopulated"
wording is **withdrawn**.

* **Stop this coverage branch** (the combination-rarity proxy).  Do not add
  support-set interventions, do not change keys/thresholds to chase a signal,
  and do not auto-purchase any training from this result.
* Part A and Part B are unchanged and were not refit; Part B's exact
  class-median shrinkage rule remains **closed** (its earlier gate FAIL stands).

## Evidence still missing before any new experiment

1. A representation-level diagnostic that can separate "coarse but load-bearing
   structural key" from "structure channel not trained", on the **alive** F_A
   edge channel, without relying on the saturated coverage counts.
2. A channel-health measurement over the full held-out G0 (not a small batch)
   for both folds, quoted with its scope, before any cross-fold channel claim.
3. A decision about whether the fix changes the planned diagnostic; it does not
   change the data, only the association estimate — which is still uninformative.

## Single next action

Run the already-planned **forward-only structural-coordinate intervention** on
the frozen F_A soup: perturb only `coord` (residual support/coefficients) on the
corrected G0 rows and measure the induced prediction change against the base
residual.  Its purpose is to test whether the (now correctly indexed) structural
coordinate carries usable signal on a load-bearing channel — the one piece of
evidence this coverage proxy cannot supply.  If flat, the next purchase is a
representation/regularisation question, not a fusion change; no parameter-search
menu is proposed.

## Guardrails honoured

* Train-only loading; `load_split("train")` avoided (it deserialises
  `encoded_valid.pt`).
* Official test never instantiated/loaded/evaluated; official-valid not
  loaded/re-read/re-evaluated.
* Old results, old caches and old reports byte-for-byte preserved.
* Only this task's code and new results committed; isolated branch
  `zinc-g0-indexfix-v1`; **no push, no merge**.