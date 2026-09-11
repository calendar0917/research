# Explicit triadic relation binding witness audit — output manifest

**Frozen-feature witness diagnostic** on the frozen compact-v4-hinge OOF states
(official TRAIN molecules only; official valid/test never loaded). It asks whether
explicitly binding the three pair relations of a patch triple — `(q_ij, q_ik, q_jk)` —
into a permutation-invariant higher-order object carries task-relevant information that
the current individual/marginal pair representation discards.

No backbone is trained; compact-v4 is unchanged; no pair state is recomputed; no patch
state is updated; there is no second round of state propagation, no attention and no
message passing.

Full scientific note: `tracks/ksvd/notes/triadic_relation_binding_witness_audit.md`.
Module: `tracks/ksvd/experiments/luyin16/zinc_triadic_relation_binding_witness.py`
(stages `inventory → spec → profile → gate0 → unbound → splits → stage1 → stage1b →
stage2 → figures → decision → all`).
Tests: `tracks/ksvd/tests/test_triadic_relation_binding_witness.py` (13 tests, all pass).

CSV/JSON/PNG artifacts below are generated outputs (git-ignored by policy); this README,
the note, and the claim/decision YAML records are the tracked scientific record.

## Result in one line

**NO-GO (Case C):** true triad witness MAE `0.192152` vs unbound matched control
`0.190303` (`mean ΔU = −0.001849`, 95% CI `[−0.00294, −0.00079]`, `P(>0)=0.0003`, 0/5
folds) and vs R-only `0.190499` (`mean ΔR = −0.001653`, 1/5 folds). Second backbone and
Stage-1b not spent.

## Frozen triad construction

Inputs come from the already-verified `frozen_state_export_v4_centre_incidence` (same
frozen OOF checkpoints; no new inference). For every unordered triple `i < j < k`:

```
a = q_ij, b = q_ik, c = q_jk
s1    = a + b + c
s_abs = |a−μ| + |b−μ| + |c−μ|,  μ=(a+b+c)/3
s2    = a⊙b + a⊙c + b⊙c
triad_raw = [s1 ; s_abs ; s2]  ∈ R^48
z = standardize(triad_raw) @ W_T,  W_T fixed orthonormal 48×16 (seed 20260914)
T_graph = [mean(z) ; std(z)] ∈ R^32
```

Value-lexicographic canonical ordering makes the representation exactly invariant to
pair-slot permutation. Full `O(n^3)` enumeration is used (median `n=23`, max `n=36`,
~8.3 s per 2000-molecule fold; no sampling).

## Unbound matched control

Keep `q_ij, q_ik`; deterministically permute the outer `q_jk` across the molecule's
triples (molecule-ID-seeded single cycles within same-outer-bucket groups). Pair-state
multiset and outer-bucket distribution are exactly preserved; 97.4% of triples are
genuinely rebound; bucket mismatch 0.0; fixed-point rate `≤ 4.95e-7`.

## Stage 1 (frozen OOF backbone seed 0 × 5 folds × 1 init)

| reader | architecture | params | mean eval MAE |
|---|---|---:|---:|
| B0 frozen baseline | — | — | 0.193160 |
| B1 R-only | `302 → 13 → 13 → 1` | 4135 | 0.190499 |
| B2 unbound control | `(302+32) → 12 → 12 → 1` | 4189 | 0.190303 |
| E true triad | `(302+32) → 12 → 12 → 1` | 4189 | 0.192152 |

`mean ΔR = B1 − E = −0.001653` (1/5 folds); `mean ΔU = B2 − E = −0.001849` (0/5 folds).
