# Pre-registration — PEC-I1: Static Composition Interface Audit (ZINC)

Round **PEC-I1** · study `zinc-context-gap` · protocol `pec_i1`.
Prior-artifact audit: [`pec_i1_prior_artifact_audit.md`](pec_i1_prior_artifact_audit.md).
Written **before** any PEC-I1 training or full-data run.

This round is an **interface audit**, not an architecture search.  It changes
**exactly one** scientific object relative to PEC-C1: the graph-level
composition / readout statistics interface.  Everything else is frozen and
imported from `pec_v0.py` unchanged.

Frozen parent verdicts stay unchanged:

```
PEC-C1:  PURE_ENV_COMPOSITION_ABSOLUTE_WEAK   seed1_authorized = false
PEC-v0 Gate 1: historical frozen FAIL (not reopened, not recalibrated)
```

`official ZINC test` is **never** loaded.  Only `train` and `val` splits are
read, through a guarded loader that raises for `"test"`.

---

## 0. Scientific hypothesis

> PEC-C1's `0.151767` deficit against the strict-static S0 band is caused mainly
> by compressing a real environment-composition mechanism into a graph-level
> `mean`/`max` interface that is too poor — not by missing local chemistry in
> the environment formation.

## 1. Purity contract (hard, unchanged)

The model may **not** read: raw `patch_cont`; typed token; parent token; B-Full;
B-Bag; B-Null recurrent state; global typed atom/bond histogram;
`path_bond_mean`; adjacent bond one-hot; any direct raw atom/bond graph readout;
the strict-static mixed pair relation; any MP / recurrence / attention output.

Additional PEC-I1 invariants:

* no `c_ij -> E_i` and no `c_ij -> E_j` (no message passing);
* `E_i` is computed before any pair information exists and is frozen;
* the reader reads chemistry only through the pooled `E_i` / `c_ij`;
* no typed global bypass (`global_topo` stays the 8-D PEC topology vector).

## 2. Frozen objects (identical to PEC-C1 unless noted)

Roles (Amendment A1 reading): node occurrence role `b^V ∈ R^11`, edge occurrence
role `b^E ∈ R^15` from `fsar_v2._explicit_basis_for_patch`; atom primitive 28
one-hot; bond primitive 4 one-hot.

Dense role coordinate: `alpha^V = M_V b^V`, `alpha^E = M_E b^E`,
`Linear(11→16)` / `Linear(15→16)`, initialized from `Dᵀ` of the matching
(full-train or Gate-2-train) K-SVD dictionary.  **No dictionary arm is run this
round.**

Environment (formed once per root, then frozen):

```
r^V = [ one_hot(shell,3) ; alpha^V ]          # 19
r^E = [ one_hot(shellpair,5) ; alpha^E ]      # 21
B^V = sum r^V ⊗ a(q_v)                         # 19 x 28
B^E = sum r^E ⊗ c(b_e)                         # 21 x  4
E_i = H([ a(q_i) ; vec(B^V_i) ; vec(B^E_i) ; S_i ]) -> R^48
H: 650 -> 96 -> 48, SiLU
```

Static composition (one read-only pass):

```
rho_ij ∈ R^18   # S0 relation minus path_bond_mean(4) and adjacent-bond(4)
c_ij = F([ E_i+E_j ; |E_i-E_j| ; E_i⊙E_j ; rho_ij ]) -> R^48
F: 162 -> 64 -> 48, SiLU, once per unordered pair
```

No `F` architecture change, no chemistry in `rho_ij`, no second pair pass.

## 3. The single change — S0-style statistical pooling

PEC-C1:

```
R_unary = [mean(E), max(E)]      # 96
R_pair  = [mean(c), max(c)]      # 96
reader  = 200 -> 64 -> 1
```

PEC-I1:

```
R_unary = [ mean_i E_i ; std_i E_i ; log1p(n) ]                    # 2*48+1 = 97
R_pair  = concat_{d=0..4} [ mean c_ij ; std c_ij ; log1p(n_d) ]    # 5*97  = 485
y_hat   = G([ R_unary ; R_pair ; global_topo ])                    # 590 -> reader_hidden -> 1
```

`max` is removed.  The pair pooling uses the **audited S0 shortest-path distance
bucket semantics**, reused verbatim from `zinc_patch_path_pooling._pair_relation`
and `_pool_pairs`:

* **bucket count = 5** (`DISTANCE_BUCKETS = 5`, the S0 audited pooling width,
  readout `1,2,3,4,5+`);
* **distance clipping**: `bucket = min(max(distance,1),5) - 1`;
* **zero-pair behaviour**: an empty bucket contributes `mean = 0`,
  `std = sqrt(0 + 1e-8) = 1e-4`, `log1p(0) = 0` (S0 `mean_std` convention);
* **std convention**: population std = `sqrt(clamp(E[x²]−E[x]², min=0) + 1e-8)`;
* **count transform**: `log1p(count)`.

Rationale for 5 and not PEC's 8-D relation one-hot: the round brief explicitly
requires reusing S0's already-audited bucket semantics and forbids designing a
new bucket.  The exact bucket index is derived deterministically from the
already-frozen `pair_rho` distance one-hot as
`min(argmax(pair_rho[:, :8]), 4)` — no new feature is defined.

`global_topo` remains PEC's 8-D vector: no typed marginals are added.

## 4. Reader parameter matching

Pooling width changes, so the reader input width changes.  The reader hidden
width is chosen deterministically to keep the **total parameter count** within
`1 %` of PEC-CD, without touching `env_hidden`, `env_dim`, `pair_hidden` or
`pair_dim`.

Pre-committed rule: with `W = 590` the reader costs `(W+2)·h + 1` parameters.
Pick the integer `h ≥ 1` minimising `|params(CD-I1) − params(PEC-CD)|`; ties
resolve to the smaller `h`.  With PEC-CD `= 94 049`:

```
non-reader params = 81 120
h = 22  ->  total 94 145   (Δ = +96,   +0.102 %)
h = 21  ->  total 93 553   (Δ = −496,  −0.528 %)
=> frozen choice h = 22
```

The exact counts are re-measured and recorded; if the realised relative
mismatch exceeded `1 %` the round would STOP and report the reason (it does
not).

## 5. Correctness gates (all must pass before any training)

* **G0 environment equivalence** — CD-I1 vs PEC-CD with the same seed: `environment`
  and `pair` state-dict entries bit-identical, and `max|E_old − E_new| = 0` on a
  real batch.
* **G1 pair equivalence** — `max|c_old − c_new| = 0` before pooling: the only
  change happens in the aggregation.
* **G2 no MP** — patching `PECI1Model.compose_pairs` to also feed `c_ij` back
  into any environment/centre route raises; a plain forward with pair→centre
  monkeypatched to `raise` succeeds.
* **G3 environment freeze** — changing pair relations leaves `E_i` bit-identical.
* **G4 chemistry purity** — changing only atom/bond categories leaves `rho_ij`
  and every `E_i`-independent topology quantity unchanged; the pair relation is
  invariant to chemistry.
* **G5 relabel invariance** — node relabel / edge reorder changes prediction by
  `< 1e-5`.
* **G6 bucket correctness** — on hand-built toy graphs, every pair lands in the
  correct S0 bucket (incl. empty and `5+`), counts are right, and mean/std match
  a numpy reference (including the empty-bucket `std = 1e-4` convention).
* **G7 gradients** — a real batch backward gives finite, non-zero gradients for
  the dense role maps, `H`, `F` and the reader.
* **G8 parameter matching** — relative mismatch `≤ 1 %`.

Any FAIL → **STOP**.

## 6. Stage A — zero-training Patch-B recoverability audit

Label-free / no-`y`.  Compare PEC environment **input primitives** (not `E_i`)
against S0 `patch_cont` blocks for the same radius-2 rooted patch:

```
atom_shell    3 x 28 = 84
bond_shell    6 x  4 = 24
root_atom     28
incident      4
scalars       6
```

Recovery is analytical/known-denominator wherever possible (train-fit linear
probes only if no analytic map exists).  Reported per block: `max abs error`,
`mean abs error`, `exact-match fraction`, `unexplained coordinates`, and a
rank/collision summary.  The shell-pair taxonomy audit of §3 of the prior
artifact audit is included (full train/valid occurrence frequencies).

**Pre-declared handling of definitional (scope) differences.**  A coordinate
that is the *same topology quantity at a different scope / normalization* as a
PEC primitive is reported separately and does **not** by itself constitute an
information gap, provided every chemistry block (atom shell, bond shell, root
atom, incident) is exact.  Specifically, S0's structural-scalar coordinate 5 is
the *patch-scoped mean molecule degree* (`degrees.mean()/4`, patch node set),
whereas PEC's `root_scalars[3]` is the *molecule-scoped* mean degree and PEC's
`b^V` carries only the *induced* patch degree.  It has no analytic map; it is
reported with its molecule-scoped counterpart and a train-fit ridge probe
(features: PEC `root_scalars` (6), `global_topo` (8), patch induced-degree
moments) fitted on train roots and evaluated on valid roots, with `R²` and MAE
recorded.

**A-PASS** (⇒ `LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT`): all chemistry blocks
recovered exactly or via explicit known-denominator/normalization transforms,
with the only unexplained coordinates being (i) the structurally empty
shell-pair classes and (ii) the documented scope difference above.

**A-FAIL** (⇒ `LOCAL_INFORMATION_GAP_FOUND`): a task-relevant, currently
inexpressible **chemistry** block is found (missing *non-empty* shell-pair
class, lost incident-bond correspondence, unrecoverable chemistry
normalization, or a missing root-conditioned chemistry block).  Then **STOP
PEC-I1 training**; the gap is written up and deferred to a separately
pre-registered round.  No feature is added ad hoc.

## 7. Stage B — cheap internal task screen

Reuse PEC-v0 Gate-2 split exactly:

```
train = official-train first 2 000
dev   = official-train 8 000..8 499
seed  = 0
protocol: Adam lr 1e-3, wd 1e-5, batch 64, L1, grad clip 5, no scheduler,
          up to 60 epochs, best-dev checkpoint, fixed equal-weight Top-5 soup
```

No official valid, no official test.

**Arms (both in the same session/device, pre-declared matched control):**

* `CD_matched` — PEC-v0 DenseRole reference reader (`200→64→1`), the device-matched
  control (audit §2.2; the historical `0.467108` is a CPU/host-`calendar`
  artifact, so one matched control is purchased in the same GPU session);
* `CD-I1` — the candidate (`590→22→1`).

Dictionaries are the deterministic `sdb_v0.fit_ksvd` fit on the 2 000 Gate-2
train molecules (default `DICT_SEED`), used only to `Dᵀ`-initialize the dense
role map, exactly as PEC-v0 Gate 2 did.

**Primary gate.**

```
Δ_interface_frozen  = 0.467108        − M_CD-I1     (Top-5 soup)
Δ_interface_matched = M_CD_matched    − M_CD-I1     (Top-5 soup)
Δ_interface         = min(both)
```

* `Δ_interface >= 0.010` → `INTERFACE_SIGNAL_STRONG` → authorize full-data seed 0.
* `0.005 <= Δ_interface < 0.010` → `INTERFACE_SIGNAL_WEAK` → STOP.
* `Δ_interface < 0.005` → `INTERFACE_NOT_PRIMARY_GAP` → STOP (a future
  `PEC-I2` direct-bond-chemistry proposal may be written; not implemented here).

If the two deltas straddle the `0.010` boundary, the smaller (conservative) one
decides.  No second pooling variant is run.

## 8. Mechanism controls (eval-only, on the trained CD-I1 soup)

Not selection gates; they confirm the new pooling did not turn the pair branch
into a decorative unary bag:

* **relation correspondence shuffle** — keep the environment multiset and the
  relation multiset, permute which `rho_ij` goes with which pair; prediction /
  MAE must change materially;
* **BAG** — zero the pooled pair representation; `CD-I1 TRUE` must still beat
  `CD-I1 BAG`.

No independent BAG model is retrained; both are interventions on the trained
checkpoint.

## 9. Full-data formal run — authorization and frozen decisions

Authorized **only if**: Stage A = `LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT`,
all G0–G8 pass, `Δ_interface >= 0.010`, and the mechanism is alive.

One arm only: `CD-I1` seed 0, full official-train 10 000 → official-valid 1 000,
240 epochs, no early stop, best official-valid checkpoint, Top-5 soup.
Historical comparator: PEC-C1 `CD` soup `0.151767` (matched, audit §2.1).
PEC-C1 `CD` is **not** re-run.

Frozen full-data decision on `M_I1`:

* **Case A** `M_I1 <= 0.145` → `PURE_STATIC_INTERFACE_VIABLE`.
  * if also `M_I1 <= 0.1408` → additionally record
    `MATCHES_OR_BEATS_HISTORICAL_STRICT_STATIC_S0` (orientation only).
* **Case B** `0.145 < M_I1 <= 0.148` **and** `0.151767 − M_I1 >= 0.005` →
  `INTERFACE_MATTERS_BUT_NOT_SUFFICIENT` → STOP.
* **Case C** `0.151767 − M_I1 < 0.005` → `STATIC_POOLING_NOT_PRIMARY_GAP` →
  STOP, move to a `PEC-I2` proposal.

If `0.148 < M_I1 <= 0.151767` and the improvement is `>= 0.005`, the round
records `INTERFACE_MATTERS_BUT_NOT_SUFFICIENT` (same as Case B: a material but
insufficient gain).  If `M_I1` is outside both Case A and the `>= 0.005`
improvement, Case C fires.

Seed 1 is not authorized by this pre-registration.  No `K`/`s` sweep, no
recurrence, no attention, no extra chemistry feature, no second pooling.

## 10. PEC-I2 (recorded, not implemented this round)

Only if PEC-I1 fails: does the composition operator need an explicit **direct
chemical bond relation primitive**

```
beta_ij = onehot direct bond type between roots i, j  (including NONE)
c_ij    = F(E_i, E_j, rho_ij_topo, beta_ij)
```

No direct bond type, path bond mean, or any chemistry relation may enter this
round.

## 11. Durable artifacts

```
notes/pec_i1_prior_artifact_audit.md
notes/pec_i1_preregistration.md           (this file)
notes/pec_i1_information_recoverability.md
notes/pec_i1_analysis.md
results/pec_i1/information_recoverability.json
results/pec_i1/correctness.json
results/pec_i1/parameter_accounting.json
results/pec_i1/internal_screen.json
results/pec_i1/internal_curve.csv
results/pec_i1/mechanism.json
results/pec_i1/REPORT.md
results/pec_i1/DECISION.md
results/pec_i1/decision.json
```

If the full run is authorized, additionally: `full_seed0.json`,
`full_seed0_curve.csv`, `states/`, soup artifact, remote provenance, claim YAML,
decision YAML, `STATE.yaml` update.

Every artifact records local/remote commit, dirty flag, GPU, seed, split
fingerprint, reused baselines, baselines explicitly not re-run, wall time, peak
memory, and `official_test_loaded = false`.

## 12. Final verdict vocabulary (only these)

`LOCAL_INFORMATION_GAP_FOUND` ·
`LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT` ·
`INTERFACE_NOT_PRIMARY_GAP` ·
`INTERFACE_SIGNAL_WEAK` ·
`INTERFACE_SIGNAL_STRONG_INTERNAL` ·
`PURE_STATIC_INTERFACE_VIABLE` ·
`INTERFACE_MATTERS_BUT_NOT_SUFFICIENT` ·
`STATIC_POOLING_NOT_PRIMARY_GAP`

No post-hoc fuzzy conclusion is permitted.  One test, one conclusion.
