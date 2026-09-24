# E2E-DictEnv-M1 — pre-registration (frozen before any implementation run)

Round **E2E-DictEnv-M1** · protocol `e2e_dictenv_m1` · study `zinc-context-gap`
(Workstream M of the 2026-09-24 two-workstream round).

This is the frozen specification.  It is written **before** any M1
implementation, dictionary fit, cache build, training or GPU use.  Any
deviation requires a written amendment recorded here before the affected run.

The paired, independent Workstream Z (ZINC absolute-performance search) is
pre-registered separately in `e2e_dictenv_p2_abs_preregistration.md`.  The two
workstreams share only the operational concurrency gate; their results
directories, states, logs, checkpoints and freeze files are fully disjoint.

---

## 1. The single scientific question

> Can the P1 clean primitive-only dictionary-core / no-message-passing
> architecture transfer as an **architecture** to OGBG-MolHIV, i.e. reach a
> reasonable official-valid and official-test ROC-AUC **without any
> MolHIV-specific architecture search**?

This is **architecture transfer**, not vocabulary transfer.  The dictionary
initialization is therefore re-fit label-free on the MolHIV official-train
topology; the ZINC-trained `D` is *not* used as the final M1 initialization.

---

## 2. Dataset

```text
dataset   ogbg-molhiv
split     official OGB scaffold split
```

Actual train / valid / test sizes and positive counts are read at runtime from
the OGB loader and recorded; the pre-registration does not hard-code them as the
source of truth.

Every data-fitted transform (dictionary fit, anchor scaler, topology
standardizer, chemistry schema inspection, `lambda` calibration) uses **official
train only**.  Valid is used only for checkpoint selection and the Top-5 soup.
Official test stays blocked until the freeze stage.

---

## 3. Pure-topology coordinate (exact P1 transfer)

```
phi_v in R^65   = exact FSAR-R2-AR0 `fsar_r2_ar0.build_phi` coordinate
```

Chemistry-free, label-free, node-centric, identical mathematical definition to
P1.  Only the *data interface* is ported (OGB graph → the same `build_phi`).
A synthetic equivalence gate confirms that identical topology produces identical
`phi` (G0).

---

## 4. MolHIV dictionary initialization

Re-fit on MolHIV official-train `phi_v^65` only, no valid/test:

```text
K          = 32
s          = 8
epochs     = 10      (SDB-v0 K-SVD family, same convention)
seed       = 20260924 (SDB DICT_SEED)
solver     = sdb_v0.fit_ksvd -> tccd_v0.ksvd_fit
normalization = column-normalized Dbar_k = D_k / max(||D_k||_2, eps)
```

If official-train node count `<= 500_000`, use all nodes; otherwise a
deterministic `500_000`-node sample with a fixed seed and no valid/test
contamination.  Sample provenance is recorded.

```
D_0^M in R^{65 x 32},   D.requires_grad = True
alpha_v = IHT_{s=8}^{T=10}(Dbar, phi_v)     (exact top-8)
```

---

## 5. OGB chemistry representation

Categorical schema is read from the OGB runtime:

```python
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
```

No hand-written category counts.

```text
atom chemistry : one Embedding(field_cardinality, 24) per OGB atom field,
                 summed                                -> q_v^chem in R^24
bond chemistry : one Embedding(field_cardinality, 12) per OGB bond field,
                 summed                                -> b_e^chem in R^12
```

These embeddings may read **only** the chemistry fields.  They may not read
degree, shell, neighbour identity, dictionary code or any graph state.

---

## 6. MolHIV primitive anchor `p_i in R^62`

Semantics are exactly P1: the anchor may only answer "what is the root / what is
in the patch / how big is the patch".

```
p_i = [ q_i^chem (24) ; sum_{v in P_i} q_v^chem (24) ;
        sum_{e in E_i} b_e^chem (12) ;
        log(1+|V_i|), log(1+|E_i|) (2) ]          -> 62
```

Train-only per-coordinate standardizer (fit at initialization, frozen).  No
shell conditioning anywhere in the anchor.

---

## 7. Node dictionary × chemistry binding

```
d_A = 96
W_A^S in R^{32x96}, W_A^C in R^{24x96}      (bias-free)
u_v^A = ((alpha_v W_A^S) o (q_v^chem W_A^C)) / sqrt(96)
A_{i,s} = sum_{v : dist(i,v)=s} u_v^A in R^96      (shell = routing index only)
A_i = [A_{i,0}; A_{i,1}; A_{i,2}] in R^288
```

---

## 8. Edge dictionary-role × bond-chemistry binding

No edge dictionary is created.

```
g_uv = [alpha_u + alpha_v ; |alpha_u - alpha_v| ; alpha_u o alpha_v] in R^96
d_E = 48
W_E^S in R^{96x48}, W_E^C in R^{12x48}      (bias-free)
u_e^E = ((g_uv W_E^S) o (b_e^chem W_E^C)) / sqrt(48)
E_{i,p} = sum_{e : shellpair(e)=p} u_e^E in R^48     (6 historical shellpair slots)
E_i^edge = [E_{i,0}; ... ; E_{i,5}] in R^288
```

---

## 9. Local decoder

```
z_i = [ p_i^62 ; A_i^288 ; E_i^{edge,288} ] in R^638
E_i = W_2 SiLU(W_1 z_i)                        638 -> 102 -> 48
```

No `patch_cont`, no typed/parent token, no message passing, no recurrence, no
attention, no LayerNorm, no pair→centre.

---

## 10. Pair composition

The P1 static read-only composer is kept.  Only the **pure-topology** pair
relation is used; MolHIV bond chemistry enters the environment *only* through the
`dictionary endpoint role × bond chemistry` edge binding.

MolHIV has a 6th relation class (disconnected salt pairs) beyond the ZINC
5-bucket range.  Per the round rule, P1's distance bucketing/clipping semantics
are reused **without adding new bins**: `bucket = clip(distance, 1, 5)` with the
disconnected class clipped into the top (`5+`) bucket; the relation vector is
`[one_hot(bucket,5) ; log1p(distance) ; overlap(5) ; boundary(3) ; log1p(path_count)]`
(15-D, exactly P1's pure-topology slice family).  The clipping frequency
(disconnected pairs and pairs with distance > 5) is recorded.  No new distance
bins are added in this round.

```
u_i = W_P E_i in R^16                     (bias-free)
inputs = [u_i+u_j ; |u_i-u_j| ; u_i o u_j o gate(d_ij) ; relation(15)]
pair encoder = _MLPBlock(64, 64, 16)
distance gate = Embedding(5, 16)
```

---

## 11. Global context and readout

```
global      = [ pure topology^30 ; atom chemistry aggregate^24 ;
                bond chemistry aggregate^12 ]                 = 66 -> 32 -> 32
topology    = 25-D generic pure-topology (hinge) features     25 -> 16 -> 8
graph repr  = [unary^97 ; pair moments^165 ; global^32 ; topology^8] = 302
reader      = GenericReader(302, (13, 13)) -> 1 logit
```

The chemistry aggregate is a chemistry-only whole-molecule mean
(`mean_v q_v^chem`, `mean_e b_e^chem`); no local shell-conditioned statistics.
The topology branch uses the repository's graph-generic 25-D `zinc_topology_features`
hinge vector (cycle spectrum / MCB / hinge ladder), standardized on official
train only.

---

## 12. Parameter audit

No width is forced to match P1's 97,865.  The exact count is computed from the
runtime OGB field dimensions and is the source of truth.  Chemistry embedding
tables, dictionary, node binding, edge binding, decoder, backend and whole model
are recorded.  Hard budget:

```text
80_000 <= total trainable params <= 130_000
```

Out of budget: STOP and inspect the implementation, do not widen to match ZINC.

---

## 13. Reconstruction coefficient

BCE has a different scale from the ZINC L1 target, so `33.9587` is not copied.
Train-only calibration on the **first 512 official-train graphs** with the
initial model and a fixed seed:

```
lambda_base = L_task^init / (L_rec^init + eps)
lambda_M    = 0.25 * lambda_base            (frozen, no sweep)
```

---

## 14. Training protocol

The existing repository MolHIV faithful-transfer training family is used with no
new degrees of freedom:

```text
seed = 0, 1
Adam, lr 1e-3, weight_decay 1e-5, batch 128, grad clip 5
BCEWithLogitsLoss (UNWEIGHTED) + lambda_M * reconstruction
max epochs 240, patience 40, valid ROC-AUC every epoch
```

If batch 128 OOMs, only a runtime batch reduction `128 -> 64` is permitted and
is recorded.  No architecture change.

---

## 15. Selection

Per seed, reported independently:

```
raw  : checkpoint with the highest official-valid ROC-AUC (ties -> earliest)
soup : equal-weight parameter average of the 5 highest-valid-ROC-AUC checkpoints
```

Both raw and soup are reported in full; there is no post-hoc selection.

---

## 16. Correctness gates (all PASS before the formal run)

```
G0  phi65 identity vs a fresh build_phi (bit-identical)
G1  dictionary identity: shape / sha256 / column norms
G2  exact top-s: max l0 <= 8, exact-8 fraction >= 0.99
G3  real BCE backward gives ||grad_D|| > 0
G4  chemistry purity: chemistry relabelling leaves phi/alpha/topology-incidence
    unchanged (phi is topology-only)
G5  forbidden local descriptors absent (AST + NaN-poisoned patch_cont)
G6  primitive anchor recomputed from raw primitives; no shell-conditioned chem
G7  edge role symmetry g_uv == g_vu
G8  pair-relation chemistry blocker: pair relation is topology-only; no
    path_bond_mean / adjacent_bond_type in the model
G9  environment frozen under pair mutation
G10 no pair->centre: monkeypatched raise, forward still succeeds
G11 once-only composition: pair/relation/env encoders called exactly once
G12 relabel invariance
G13 parameter budget 80_000 <= actual == accounted <= 130_000
G14 official test blocker
```

A GPU smoke (512 graphs, 3 epochs) must also pass: finite loss, non-zero
`||dL/dD||`, exact top-8, active-atom retention, environment rank, node/edge
binding gradients non-zero.

---

## 17. Freeze and test

After seed0 and seed1 train→valid complete, write
`architecture_freeze.json` with commit, split hashes, chemistry schema,
dictionary fit provenance, `D_0` hash, `lambda_M`, seeds, selected raw
checkpoint hashes, Top-5 soup member hashes, valid metrics and
`official_test_loaded_at_freeze_time = false`; commit the freeze.  Then write
`official_test_unlock.json` asserting:

```
the user explicitly authorised the test
the test is terminal reporting only
MolHIV test is NOT project-wide pristine
no post-test model change
```

The test is loaded once, in one process, and evaluated on seed0 raw/soup,
seed1 raw/soup, plus a 2-seed raw prediction ensemble and a 2-seed soup
prediction ensemble.  Only ROC-AUC is reported.  No train+valid refit and no
retraining after the test read.

---

## 18. Context anchors (not tuning targets)

Reported as context only, with explicit protocol differences: the prior
no-MP patch-path pooling port, the prior recurrent pair--centre port, and the
GIN / GPS / CIN references already recorded in the repository.

---

## 19. Final verdicts

| id | condition |
|---|---|
| `M1_TRANSFER_STRONG` | soup valid AUC >= 0.80 and test soup AUC >= 0.78 (2-seed mean) |
| `M1_TRANSFER_COMPETITIVE` | soup valid AUC >= 0.78 and test soup AUC >= 0.75 |
| `M1_TRANSFER_PLAUSIBLE` | soup valid AUC >= 0.72 (above a trivial/marginal baseline) |
| `M1_TRANSFER_WEAK` | soup valid AUC < 0.72 |

The verdict answers only "does the clean dictionary-core architecture transfer
at all"; no dictionary-specificity claim may be made from M1.

---

## 20. Forbidden (regardless of how close)

```text
NO MolHIV architecture tuning / width tuning / reader tuning
NO patch_cont ; NO atom_shell ; NO bond_shell ; NO local shell x chemistry
histogram ; NO pair bond-chemistry relation ; NO message passing ; NO recurrence
NO attention ; NO K/s/IHT change ; NO lambda sweep ; NO class weighting ;
NO test access before the freeze
```

---

## 21. Durable artifacts

```text
notes/e2e_dictenv_m1_preregistration.md
notes/e2e_dictenv_m1_implementation.md
notes/e2e_dictenv_m1_analysis.md
notes/e2e_dictenv_m1_test_read.md

results/e2e_dictenv_m1/
    schema.json  dictionary_fit.json  parameter_accounting.json
    correctness.json  smoke.json
    run_seed0.json  run_seed1.json  curve_seed0.csv  curve_seed1.csv
    states/
    architecture_freeze.json
    official_test_unlock.json  official_test_results.json
    REPORT.md  DECISION.md
```
