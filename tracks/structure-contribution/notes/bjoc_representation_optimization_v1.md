# bjoc-representation-optimization-v1 — design and pre-registration record

Round `cscl-bjoc-representation-optimization-v1` (track `structure-contribution`).
Pre-registration: `configs/bjoc_representation_optimization_v1.yaml`, frozen in
`results/bjoc_representation_optimization_v1/protocol_freeze_manifest.json` before the first
formal trajectory. Driver: `code/run_bjoc_representation_optimization_v1.py`.

## Question

The terminal BJOC evaluation left the predictor ~0.003 better than historical Full-Y on official
test with a confidence interval crossing zero, and showed the real bottleneck is the *gap* between
fit (~0.04-0.05 MAE) and unseen molecules (~0.12-0.14 MAE). This round asks which of three
candidate improvable points is real:

1. **optimization stability** — the last 40 epochs of the frozen schedule still buy something and a
   decayed schedule would be worth a trajectory (arm `S`);
2. **learned junction content** — the per-centre junction reader contributes predictive content
   (measured by removing it: arm `J`);
3. **missing typed structure** — the original objects cannot express which bond types short paths
   carry or how rings are fused, and adding that helps (arm `T`).

## Arms and capacity

| arm | model | new channels | parameters |
| --- | --- | --- | --- |
| B | `ExtendedBJOC` (original widths) | present but exactly zero | 432,705 |
| T | `ExtendedBJOC` (original widths) | real, standardised | 432,705 |
| J | `NoJunctionBJOC` (original input dims) | none | 432,705 |
| S | `ExtendedBJOC` + staged LR | zero | 432,705 |

* Published BJOC backbone: **413,505** parameters → all arms are **+4.64%** because the new input
  rows cost parameters. B absorbs that cost with structurally zero inputs, so the T−B contrast is
  capacity-free.
* J is matched to **B**, not to the published backbone: J−B is the causal contrast, and an
  undersized J would bias the junction verdict in the direction that flatters our hypothesis.
* B and T are the *same class with the same widths and the same seed*, so they share an identical
  initial state (verified: identical `state_dict` hash) and an identical data order.

## New representation (135 atom + 12 bond channels)

* typed length-2 and length-3 **bond-type sequences** rooted at each atom (16 + 64 log1p counts);
* how many of those path bonds are ring bonds (3 + 4);
* the atom category at the far end of each length-3 path (28);
* path totals (2);
* Tarjan **biconnected ring-block** descriptors per atom (18) and per bond (12): block sizes,
  local cycle rank, attachments, cut-vertex share, spiro/monocyclic/polycyclic flags, block
  bond-type fraction and a capped simple-path fused-edge test.

Feature-information audit (`feature_information_audit.json`): **53.4%** of the original-feature
groups (5,715 / 10,699 atom rows over the 10k set) are split by the new channels, so the new blocks
are not a relabelling of the old ones. Sparsity is disclosed: only 6 of the 9 and 12 of the 27
bond-type sequences that can occur at all appear in the training set; `ring_is_spiro_center` fires
on 199 atom rows, so no spiro-specific claim is possible.

## Recipe (identical for every arm)

Adam lr 1e-3, weight decay 1e-5, batch 128, grad clip 5, 240 epochs, FP32, no AMP, no scheduler,
raw-y L1, last-5 parameter soup (epochs 236-240). Global features are standardised with a
standardiser fitted on that split's **fit rows only** (the same rule the upstream prep stage used);
the new channels are standardised the same way, and the zeroed arms use the fit mean (0 in
standardised space).

The first stage-1 launch ran with unstandardised global features (mean 0 / scale 1). That is a
deviation from the upstream recipe, it pushed the monitor MAE to ~0.175, and it was cancelled
before any trajectory completed; nothing from it enters the results. The fix applies to all arms
identically.

## Splits

Canonical-SMILES groups, ordered by `sha256(salt | smiles)`, greedy fill to 8,000 fit / 1,000
monitor / 1,000 locked. Split A uses `cscl-bjoc-repr-opt-v1-discovery`, split B
`cscl-bjoc-repr-opt-v1-replication`. Row-set hashes are frozen in `split_a.json` / `split_b.json`.
The split building and the objects cache reproduce **bit-identically** on the remote host.

Disclosure: split B's locked molecules were mostly inside split A's fit set (805/1000), so B
replicates the *fit-set* variation, not the locked rows.

## Read protocol

Monitor labels are readable (they are part of the pre-registered gate). Locked-dev labels are read
**exactly once**, inside `score`, after every prediction file has been hashed; `predict`,
`score --preview`, `decide --preview` and `mechanism` were rehearsed on monitor rows first.
Official valid and official test are never loaded in any form.

## Pre-registered decisions

* `T_CONFIRMATION_PURCHASE` iff monitor Δ(T−B) ≤ −0.006 **and** locked Δ < 0 **and** ordinary
  molecules (y > −3) not worse **and** not single-row-driven (top-10-removed Δ < 0, worst-row share
  < 0.05) **and** correctness/capacity gates pass.
* Junction reading: supported if locked J−B > +0.005 with CI lower > 0; simplification signal if
  < −0.005 with CI upper < 0; unresolved if |Δ| ≤ 0.005; otherwise directional only.
* Stability: buy only if the curve audit shows a late plateau plus a soup gain, and the seed gap is
  an optimization artefact.

## Budget

10 formal trajectories maximum, 1 GPU, ≤ 2.5 GPU-hours, one trajectory at a time, total wall clock
≤ 7h45m with ≥ 35 min reserved for pull / verify / report / git.
