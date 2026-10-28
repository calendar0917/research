# PROTOCOL — zinc_path_binding_sufficiency_audit_v1

**Round id:** `zinc-path-binding-sufficiency-audit-v1`
**Track:** `tracks/structural-sufficiency/`
**Created / frozen:** 2026-10-28 (written and committed **before** any audit number was computed)
**Status:** frozen-at-commit; see `PROVENANCE.json` for the freeze commit.
**Budget:** **0 GPU·h**. CPU only, 16 threads, no CUDA device is touched.
**Upstream round (read-only):** `zinc_local_relation_composition_v1` (LCRC-v1), commit `d557e6e`.

---

## 1. Question

LCRC-A's local input `X_v` contains a fixed, low-rank summary of the rooted typed
two-hop paths (`S1` joint neighbour code, `S2` path-type / end-type / rank-32
product / closure / gated-endpoint-content). The proposed DFPB candidate instead
binds the *whole* ordered two-hop path:

```
T_real(v) = n_v^{-1/2} * sum_{(v,w,u)} Q(a_v,b_vw) ⊙ K(a_w,b_wu) ⊙ V(a_u)
```

**Does LCRC-A's existing S1/S2 statistics already retain everything the dynamic
two-hop path binding can provide, and is there a reason to pay for a DFPB GPU
experiment?**

Sub-questions (must be answered separately):

1. Is the difference *mathematical expressiveness*, *finite-dimensional
   compression*, or *only parameterisation*?
2. Which part of `T_real` is recoverable from the existing statistics on real
   molecules, and which part is not?
3. Does the "new" information come from real path-content correspondence, or from
   simply feeding learned atom/bond context through a new product?
4. Is a GPU round justified? If not, stop explicitly.

## 2. Non-goals (hard)

- No property labels are read at any point (`y` is poisoned at load time and any
  access raises). No Fit/Monitor/Locked split is created. No official test cache
  is opened (`lc_test.pt` is never loaded; `prep` refuses `test`).
- No training, fine-tuning, or re-initialisation of LCRC-A. The two frozen A
  checkpoints are used exactly as committed.
- No new ZINC property-prediction model. No architecture search.
- No modification of LCRC-v1 code, weights, caches, or reports; the round is
  imported read-only through `sys.path`.
- The audit cannot and does not claim that any measured difference improves MAE.

## 3. Data and frozen artefacts

| item | value |
|---|---|
| splits | official PyG ZINC `subset=True`, train 10,000 / valid 1,000 |
| train cache | `zinc_local_relation_composition_v1/cache/lc_train.pt`, sha256 `8777494f…` |
| valid cache | `…/cache/lc_valid.pt`, sha256 `aae2caa5…` |
| atom scalar normaliser | `…/analysis/norm.json`, fingerprint `8c3e8813…` |
| A_s0 | `…/runs/lcrcA_s0/best.pt`, sha256 `06340bcd9e45af03…` |
| A_s1 | `…/runs/lcrcA_s1/best.pt`, sha256 `7225fe11fce0a24d…` |
| feature version | `lcrc-v1-atom108-s1joint-s2typedpath` |

Atom/bond encoders inside those checkpoints are the LCRC-v1 fine-tuned encoders
(SSL initialised, then fine-tuned by LCRC-v1 itself). They are frozen here.

## 4. Frozen definitions

Per atom `v`: `a_v ∈ R^96`, per bond `e`: `b_e ∈ R^96`
(`b_e = BondMLP([a_u+a_v, |a_u-a_v|, a_u⊙a_v, bond_feat])`).

Path set `P(v)` = all ordered non-backtracking two-hop paths `v→w→u`, `u≠v`
(closure `u~v` included), with multiplicity, exactly as LCRC-v1 enumerates them
and **additionally carrying the two bond ids `e1=(v,w)`, `e2=(w,u)`**.

Fixed projections (label-free, never searched, never selected on any outcome):

```
W_Q, W_K ∈ R^{32×192}   W_V ∈ R^{32×96}     W ~ Normal(0, 1/d_in)
projection seeds: 20261030, 20261031, 20261032   (all three always reported)
Q_p = W_Q [a_{v} ; b_{e1}]     K_p = W_K [a_{w} ; b_{e2}]     Z_p = W_V a_{u}
```

```
T_real(v) = n_v^{-1/2} Σ_{p∈P(v)} Q_p ⊙ K_p ⊙ Z_p
T_null(v) = n_v^{+1/2} (q̄_v ⊙ k̄_v ⊙ z̄_v),   q̄_v = n_v^{-1} Σ_p Q_p,  (same for k,z)
D(v)      = T_real(v) − T_null(v)
```

`T_null` uses the **same** per-path factors, the **same** `n_v`, and means taken
**inside one root only** (never across molecules or batches). Under independent
`q,k,z` along paths `E[T_real] = E[T_null]`, so `D` is a matched-scale
"joint binding" residual, not a norm artifact (unit-tested by Monte-Carlo).

Secondary binding control (reported, not decisive): `T_perm(v)` = same factors
with `Q,K,Z` re-paired by a fixed within-root random permutation, and
`D_perm = T_real − T_perm` (keeps the marginals exactly, destroys only the
alignment).

Type-only content control: `ā[t]` = **train-only** mean of `a` over atoms of
type `t`; `b̄` = **train-only** mean of `b` over bonds with the same unordered
`(type(u), type(v), bond_type)` triple. Same projections, same graphs, same
normalisation → `T_type`, `T_null_type`, `D_type`. `T_type` is a statistic of the
typed path multiset only; `T_real − T_type` is exactly the within-type learned
context, so this control separates "real path-content correspondence" from
"more learned context".

Input sets for reconstruction (all label-free, all per atom):

| name | content | dim |
|---|---|---|
| `S` | `b_sum, b_mean, S1joint(32), path_agg(104), content_agg(96), scalars(9)` — existing S1/S2 statistics + local scalars, **no `a_v`** | 433 |
| `X` | full input to `LocalContent.mlp` (`S` + `a_v`) | 529 |
| `h` | trained LCRC-A local output | 64 |

Targets: `T_real, T_null, D, T_type, D_type` (32 ch each) and, as a probe
capacity reference, `h` (64 ch) reconstructed from `X`/`S`.

Relative binding: `ρ_v = ‖D_v‖₂ / RMS_T`, `RMS_T = sqrt(mean_train ‖T_real‖₂²)`
(train-defined scale; no valid/target information).

## 5. Reconstruction protocol

- Fit on official-train atoms only, evaluate on official-valid atoms only.
  Standardisation (inputs and targets) uses train means/stds only.
- Selection of `λ` / epoch count uses a train-internal monitor only:
  `RandomState(20261029).permutation(10000)[:2000]` molecules are held out of
  fitting; valid is never used for any choice.
- Ridge: closed form on standardised features, `λ ∈ {1e-3,1e-2,1e-1,1,10,100}`
  selected by monitor R², refit on all 10,000 train molecules.
- Nonlinear: MLP `d → 256 → 128 → n_out`, SiLU, AdamW(1e-3, wd 1e-5),
  batch 2048, ≤60 epochs, early stop (patience 8) on monitor R², seed 20261030.
- Permutation control: identical pipeline with train target rows permuted
  (`RandomState(20261101)`), evaluated against the true valid targets.
- Metrics: per-channel R², block R² (variance-weighted over **non-degenerate**
  channels), standardised RMSE, effective residual rank (entropy and
  participation ratio), residual-variance fraction.
- Degenerate channel: train std `< 1e-6` → excluded and flagged. Near-constant
  channel: train std `< 1e-2 ×` median channel std of that target block →
  reported separately, never used to inflate a headline R².

## 6. Graph-level collapse audit

Per molecule aggregates of `T_real`, `D`, `D_type`: `sum`, `mean`, `mean of
squares` (3 × 32) + `log1p(n_atoms)`. Existing graph-level input
`E = [global33, mean_v h, mean_v h², log(1+|V|)] ∈ R^162` (exactly LCRC-A's `ρ`
input). Ridge and the same MLP fit on train graphs, evaluated on valid graphs,
in both directions (`E→G`, `G→E`).

## 7. Support coverage audit

Radius-2 structural keys exactly as the existing definition
(`zinc_composition_transfer_v1/code/zct_keys.py`, recursion depth 2 over
`(atom type, sorted (bond type, neighbour key))`), rebuilt from the LCRC caches
and cross-checked against the committed `keys.npz` vocabulary. Support of a key =
number of **train molecules** containing it. Valid atoms: `unseen` (support 0),
`low` (1–3), `seen` (≥4). Molecule: `has_unseen` / `no_unseen`. Only structure,
feature variance and reconstruction residuals are compared inside these groups.

## 8. Decision rule (frozen before running)

Three separate levels are reported and never mixed:

- **L1 mathematical**: equivalence / non-equivalence proofs and counterexamples.
- **L2 empirical redundancy**: the R² and ρ measurements below.
- **L3 predictive value**: *not measured* (no labels). No GO may be justified by
  an imagined MAE gain.

`D 广泛非零` (BROAD) holds iff, for a given checkpoint / projection seed:
median `ρ_median_atom ≥ 0.15`, p10 of `ρ ≥ 0.05`, fraction of atoms with
`ρ ≥ 0.05` at least `0.90`, median per-molecule `ρ` at least `0.10`, and the
top 1% of atoms carry at most `0.25` of the total `Σ‖D‖²`.

`D 难恢复` (HARD) holds iff for `X`: `min(ridge, MLP) R²(D) < 0.80` **and** for
`h`: `min(ridge, MLP) R²(D) < 0.70`.

`CHECKPOINT-CONSISTENT` iff for every (input, reconstructor) pair
`|R²(A_s0) − R²(A_s1)| ≤ 0.15` and both checkpoints are on the same side of the
HARD thresholds.

`NO-GRAPH-COLLAPSE` iff `max(ridge, MLP) R²(G_D | E) < 0.80`.

`TYPE-BINDING-PRESENT` iff `D_type` is BROAD (same rule) **and**
`min(ridge, MLP) R²(D_type | X) < 0.80`.

### Verdict

**REPRESENTATION GAP / PILOT-ELIGIBLE** requires **all** of: BROAD (≥2 of 3
projection seeds, both checkpoints), HARD, CHECKPOINT-CONSISTENT,
NO-GRAPH-COLLAPSE, TYPE-BINDING-PRESENT.

**REDUNDANT / NO-GO** if any of:
- not BROAD for ≥2 of 3 seeds or for either checkpoint;
- `R²(D|X) ≥ 0.90` for both reconstructors and both checkpoints;
- `R²(G_D|E) ≥ 0.90` for both reconstructors and both checkpoints;
- `TYPE-BINDING-PRESENT` fails (then the "new" statistic is a re-parameterisation
  of learned content already present in `X`).

**INCONCLUSIVE** in every other case — including any criterion inside ±20% of its
threshold, any result that flips sign across the 3 projection seeds, and any
result that is carried by ≤1% of atoms or molecules.

No verdict may be upgraded by "the new feature exists"; only the rule above is
allowed to produce PILOT-ELIGIBLE, and PILOT-ELIGIBLE still does **not** assert a
MAE improvement.

## 9. Known historical defect to document (not to repair here)

LCRC-v1 `RelationHead.aggregate(mode="N")` uses the **whole-batch** `z` mean
(`z.sum(0) / z.shape[0]`), not a per-molecule mean, so N's output for one
molecule depends on the other molecules in the batch. The historical G2
batch-invariance gate could not see this because the relation head is
zero-initialised. This round (a) adds a minimal, label-free batch-invariance
regression test with a **non-zero** relation head, (b) provides a corrected
per-molecule reference aggregator, and (c) records the interpretation limit of
the original R/N contrast. It does not change any historical number and is not a
reason to retrain LCRC-v1.

## 10. Deliverables

`PROTOCOL.md`, `REPORT.md`, `DECISION.md`, `PROVENANCE.json`,
`code/pb_*.py`, `tests/test_pb_*.py`, `analysis/*.json`.

Final sentence required in the report: which of **statistical content**,
**statistics composition function**, or **`X_v → h_v` compression** is the most
worthwhile thing to improve in current LCRC-A.
