# FSDP v0 — Fixed-Source Structural Distribution (MEANFIELD) pilot

- Round: `tracks/structural-sufficiency/fixed_source_structural_distribution_pilot_v0/`
- Protocol id: `fsdp-v0`
- Round id: `fixed-source-structural-distribution-pilot-v0`
- Data: `ogbg-molpcba` official scaffold split, **complete** Train (350,343) / Valid (43,793);
  **no Test access at all** in this round (neither scoring nor labels).
- Seed: 0. One new model: `MEANFIELD`. No architecture search, no multi-seed,
  no distillation, no ensemble, no auxiliary loss, no hyper-parameter search.
- Budget: wall clock target 45–60 min, hard cap 75 min; 0.8 GPU-hour soft / 1.0 hard;
  one GPU, one training process.
- Frozen before the first training step: everything below.

## 1. Question

Does FIXED1's performance depend on the correspondence between *distinct local
structural content* and the *actual centre* it is aggregated into? Or is it
enough to keep the molecule's distribution of structural content plus the real
per-centre relation-type counts?

## 2. Reused base (read-only)

`tracks/structural-sufficiency/direct_structural_computation_recurrence_v0/`
(frozen DSCR v0). Kept bit-identical:

- the W2 trunk (`bs_model.BSModel`) and the S1/S2 local content;
- both FIXED1 blocks (`sg_model.MessageFFN`), LayerNorm, `bond_proj`, residuals, FFN;
- the existing `f`, `rho` and attention readout;
- data identity, initialisation rule and the training recipe.

The RSSC low-rank branch is **not** used. Parameter count must equal the original
FIXED1: **1,128,468** (audited tensor-by-tensor, see check P1).

## 3. MEANFIELD definition

Per block, per real directed edge `j -> i` (both directions of every physical bond):

```
p            = SiLU(z_source[j] + edge_msg)                 # FIXED1 message, unchanged
group key    = (graph_id, raw_bond_feature_signature)       # raw 13-d discrete tuple
group_mean   = segment_sum(p, group_id) / group_count
p_meanfield  = group_mean[group_id]
m[i]         = sum_{j -> i} p_meanfield                     # vectorised segment reduce
h_next       = h_current + FFN(z_target, m)
```

Blocks:

```
block 1:  z_source = LN1(h0),  z_target = LN1(h0)
block 2:  z_source = LN2(h0),  z_target = LN2(h1)
```

`h1[src]` / `h2[src]` are never read. No parameter is added, no local encoding is
recomputed, no random permutation is used, and no per-molecule Python forward loop
is allowed (vectorised `index_add_` segment reduce only).

The bond feature has 13 binary (one-hot) dimensions — verified on the frozen cache
before the round (`min=0, max=1`, integer valued). Grouping therefore uses the full
raw tuple exactly (`torch.unique(dim=0)`), never a learned embedding and never a
manual bucket. If the raw features had not been discrete the round would have
stopped.

**Explicit information-access statement.** MEANFIELD still uses every real directed
edge and preserves each centre's real in-edge count and in-edge bond-type counts,
but the distinct source content attached to each edge instance is replaced by the
mean content of its group. Inside one (molecule, bond-type) group the source→centre
binding is therefore shared molecule-wide. MEANFIELD is **not** an
information-access-equivalent variant of FIXED1.

Structural consequence (checked, not assumed): both directions of one physical bond
carry the same raw bond feature, so they always land in the same group. Directed-edge
group sizes are therefore even (never 1), and MEANFIELD is invariant under a global
direction flip while FIXED1 is not.

## 4. Correctness checks (all must pass before training)

1. `P1` parameter count **and every initialised tensor** identical to the original FIXED1 (same seed).
2. `P2` no mixing across molecules or across bond-signature groups.
3. `P3` per-(molecule, signature) message-sum conservation.
4. `P4` per-centre in-edge count and per-centre bond-type counts unchanged.
5. `P5` witness: same-group distinct messages really change the source→centre binding.
6. `P6` atom-renumbering (within molecule), edge-order and direction-reversal invariance.
7. `P7` empty bond graph handled, gradients still reach both blocks.
8. `P8` Test isolation (static scan + all Test labels non-finite).

Real-Valid contrast (fixed sample: the first 1,024 rows of the frozen Valid split,
no shuffling, model at its official initialisation) reports: group-size
distribution, singleton-group ratio, changed-message fraction, actual aggregated
vector change RMS (block 1 and block 2), and the group-sum conservation error.

**Contrast gate (frozen).** If `block1_relative_rms_change < 1e-2` **or**
`changed_message_fraction < 0.5`, the round is labelled `INSUFFICIENT_CONTRAST` and
**no formal training** is started.

## 5. Training

All parameters trained from the original FIXED1 random initialisation.

- complete PCBA Train, batch 128, the frozen seed-0 40-epoch order plan;
- the frozen 40-epoch per-step cosine **prefix** (1-epoch linear warmup
  3e-5 → 3e-4, then per-step cosine to 1e-5); **the cosine is never recompressed**;
- AdamW (β = 0.9/0.999, eps 1e-8, wd 1e-3), dropout 0.10, clip 5, masked BCE, fp32;
- formal horizon **12 epochs**; if 12 epochs cannot be completed the run stops and
  is reported as `INCOMPLETE` instead of appending an incomparable partial result;
- evaluation: full Valid AP/BCE every epoch; full Train AP/BCE at epochs 4, 8, 12
  plus a re-evaluation of the best checkpoint; **no Test evaluation**;
- checkpoints: best (highest full official Valid macro AP, ties → earlier epoch) and
  last, with a SHA-256 manifest.

## 6. References and analysis

Historical FIXED1 12-epoch prefix (re-verified from
`direct_structural_computation_recurrence_v0/results/runs/FIXED1_seed0/train_curve.csv`):

| quantity | FIXED1 (12-epoch prefix) |
|---|---:|
| best Valid AP | 0.26999159 (epoch 12) |
| endpoint Valid AP | 0.26999159 |
| last-3 mean Valid AP | 0.263256 |
| Train AP at epoch 12 | 0.447919 |

Same-horizon background references, read (never retrained): `SELF` seed 0 and `W2` seed 0.

Reported for every arm: official all-task macro AP, endpoint and last-3 mean, Train
AP/BCE, Valid BCE, high-support-task AP (Valid positives ≥ 20, `SUPPORT_MIN = 20`),
per-task improve/degrade counts, paired task bootstrap (2,000 resamples, seed 0),
and the sensitivity after removing the three tasks with the largest absolute
difference. The official all-task macro AP stays the primary metric; the high-support
subset never replaces it.

Optional, only if budget remains: one full Valid forward on the historical FIXED1
best checkpoint with the MEANFIELD replacement, no weight change. Frozen-weight
intervention and independent training results are reported separately.

## 7. Decision rule (frozen, single seed)

Primary comparison: MEANFIELD vs FIXED1 at the same 12-epoch horizon.

- AP gap < −0.008 and high-support tasks in the same direction →
  `LOCAL_ALLOCATION_ADVANTAGE_SIGNAL`
- |AP gap| ≤ 0.003 with no conflict in the late curve or the high-support subset →
  `DISTRIBUTIONAL_COMPUTATION_PROMISING`
- AP gain ≥ +0.005 with broad task support → `DISTRIBUTIONAL_PERFORMANCE_ADVANCE`
- otherwise → `INCONCLUSIVE`

These scales are a practical reference for a single-seed short run; they are not a
statistical-equivalence or confirmatory claim.

Interpretation limits, fixed in advance: if MEANFIELD is worse, we do **not** claim
that real relation binding has an independent causal benefit, because the
computation also changed both local distinctness and global access. If MEANFIELD is
close to FIXED1, we do **not** claim that real topology is useless, because `h0`, the
relation counts and the original global descriptors still carry structural
information.

## 8. Stop rule

Deliver only: implementation code, the necessary tests, this protocol, `main_table.csv`,
`epoch_curves.csv`, `paired_comparisons.json`, `allocation_checks.json`, the checkpoint
manifest and `REPORT.md`. No additional model, seed, grouping definition or training
length is added after seeing results.
