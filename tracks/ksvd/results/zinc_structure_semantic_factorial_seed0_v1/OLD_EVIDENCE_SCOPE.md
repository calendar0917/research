# Old-evidence applicability (one page, read-only)

This round reuses three pieces of prior evidence **as background only**. None
of them was re-audited and none of their numbers was recomputed here.

## 1. `docs/luyin/luyin19.txt` — the design claim

The transcript's claim is that a *shared* structure dictionary can learn a
structural–attribute relation that transfers to unseen graphs, and that the
combination of structure and attribute should be done by an explicit
structure–attribute cross/combination rather than by enumerating joint
categories (which is said not to converge). It also says local-environment
composition ("environment to environment") is unresolved and that the
structure/attribute dictionary should first be kept separate from fusion. The
concrete statement is a *transfer* claim about a shared representation, not a
claim about seed-level MAE.

Where the current canonical `Full` agrees with the transcript: a shared
trainable structural matrix `D` is used for every molecule, an explicit
structure×attribute binding exists at the node/edge slot level, and a separate
task dictionary handles the environment width.

Where it is **not yet honoured**: the transcript's stronger claim ("learned
structural–attribute relation reusable on unseen graphs") is exactly the
generic-transfer claim this round is designed to *test* on the current
architecture, not something already demonstrated. The earlier independence
experiments were on a smaller, earlier model and did not include CSSD-q1,
Sem108, the task dictionary or the widened task path.

## 2. `e2e_dictenv_clean_mechanism_v1` — early mechanism evidence

Applicable as prior directional evidence only:

* From-scratch C6+indep edge controls were **worse** than the paired edge
  binding by `+0.00755` mean (`+0.00900 / +0.00471 / +0.00895`, seeds 0/1/2).
  Node independence was not adopted (`+0.00434` mean, max `+0.00925`).
* Matched DenseTied was marginally **better** than the sparse tied-IHT arm
  (`G_dict = −0.002936`; gate needed ≥ `+0.003`), so sparse specificity was
  "not established" — not "disproved".
* The binding operator is *load-bearing as information* (zeroing/shuffling it
  costs 0.15–1.45 MAE), but per-occurrence correspondence itself was
  inconclusive-to-rejected for simplification at that time.

## 3. Why those numbers do **not** transfer to the current `Full`

Since that round the canonical `Full` changed materially:

* `CSSD-q1` replaced the raw code: a fixed common subspace `U` and a
  column-normalised residual dictionary `Dbar = colnorm((I−UUᵀ)D_raw)` are
  projected every forward, and the code is `[d/common_rms; IHT10(Dbar, r)]`.
* `Sem108` replaced the anchor62→32 compression with the shell-resolved
  `[Sem108(108); size2(2)]` interface; the fusion is now `446→342→144`.
* A shared task dictionary `D_L [144,288] / V_L [288,144]` was inserted between
  the fusion and the downstream composer, and the whole task path was widened
  to 408,651 parameters.
* The earlier node channel died and the edge channel was unstable across
  models; a *new* round is required to know whether the current Full still
  generalises through the same mechanism.

## 4. What this round is allowed to conclude from that background

* The question "does the per-occurrence binding beat the independent-pairing
  expectation, and does sparse coding interact with it, in the **current**
  Full?" was **not** settled by the old round. It is the new question.
* The old `+0.00755` edge result cannot be quoted as the current effect size.
* The old `−0.002936` dense advantage cannot be quoted as evidence that the
  current dense control will win; the current dense control is amplitude-
  matched at initialisation and runs through CSSD-q1.
* No other historical audit, ablation or valid/test split is used to select,
  calibrate or interpret this round.
