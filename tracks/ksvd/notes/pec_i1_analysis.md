# PEC-I1 — analysis (Static Composition Interface Audit)

Round **PEC-I1** · study `zinc-context-gap` · protocol `pec_i1`.
Pre-registration: [`pec_i1_preregistration.md`](pec_i1_preregistration.md)
(frozen before any run). Prior-artifact audit:
[`pec_i1_prior_artifact_audit.md`](pec_i1_prior_artifact_audit.md).
Stage A note: [`pec_i1_information_recoverability.md`](pec_i1_information_recoverability.md).

> **PEC-C1's frozen verdict is untouched.**
> `PEC-C1 = PURE_ENV_COMPOSITION_ABSOLUTE_WEAK`, `seed1_authorized = false`.
> **PEC-v0 Gate 1 remains a historical frozen FAIL** and is not reopened.

Remote compute: `hxy@a100-2`, repo `/home/hxy/cy/research`, commit
`a84a6c86477d4a8ebab08f22370d6cf3c8d7e7dd`, 2 × NVIDIA A100-SXM4-40GB, CUDA
12.4, torch 2.5.1+cu124, Python 3.12.14.  `official_test_loaded = false` in
every artifact; only splits `["train", "val"]` were ever read.
**Provenance note:** the only `git status --porcelain` entry on the remote is
`?? tracks/ksvd/results/pec_i1/` (this round's generated results directory), so
every artifact records `git_dirty = true` while the tracked source is exactly
`a84a6c8`.  No tracked source file was modified during any run.

---

## 1. Frozen verdict

```
STATIC_POOLING_NOT_PRIMARY_GAP
```

The internal screen showed a strong interface signal, the full-data seed-0 run
was therefore authorized and executed, and the gain **did not transfer**:

| quantity | value |
|---|---:|
| `M_CD-I1` Top-5 soup (full official-valid) | **0.151657** |
| `M_CD-I1` best checkpoint | 0.159730 @ epoch 173 |
| PEC-C1 `CD` Top-5 soup (matched comparator) | 0.151767 |
| PEC-C1 `CD` best | 0.159782 @ epoch 235 |
| `0.151767 − M_CD-I1` | **+0.000110** |
| required material improvement | **≥ 0.005** |
| viability band | ≤ 0.145 |

`0.000110 < 0.005` → **Case C** fires:
`STATIC_POOLING_NOT_PRIMARY_GAP`.  STOP; the pooling hypothesis is closed.

## 2. Frozen provenance

| field | value |
|---|---|
| local / remote commit | `a84a6c86477d4a8ebab08f22370d6cf3c8d7e7dd` |
| remote dirty entries | `?? tracks/ksvd/results/pec_i1/` only |
| GPU | NVIDIA A100-SXM4-40GB (39.4 GiB) |
| seed / epochs | 0 / 240 fixed, no early stopping |
| train / valid | 10 000 official-train / 1 000 official-valid |
| role init | PEC-C1 full-train K-SVD `Dᵀ` |
| params | 94 145 (Δ +96 vs PEC-CD, +0.102 %) |
| wall / peak GPU | 1 194.7 s / 88.3 MB |
| official test | never loaded |

## 3. Stage A — local information is substantially present

`LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT` (2 000 train + 1 000 valid molecules,
all 69 544 roots, label-free):

| block | dim | max abs err | exact fraction | unexplained |
|---|---:|---:|---:|---:|
| atom_shell | 84 | 2.65e-08 | 1.000000 | 0 |
| bond_shell | 24 | 2.75e-08 | 1.000000 | 0 |
| root_atom | 28 | 0.00e+00 | 1.000000 | 0 |
| incident | 4 | 1.99e-08 | 1.000000 | 0 |
| scalars | 6 | 2.40e-01 | 0.837628 | 67 752 |

The only S0 shell-pair class PEC cannot express is `(0,0)`, whose train and valid
occurrence counts are **both exactly 0** (structurally impossible: shell 0 is the
singleton root).  `(0,2)` is expressible and also empty.  The only
non-recoverable coordinate is S0 scalar 5, the *patch-scoped* mean molecule
degree, a documented scope difference from PEC's *molecule-scoped*
`root_scalars[3]` (ridge probe valid `R² = 0.316`, MAE `0.0329`); topology, not
chemistry.

**Conclusion:** the PEC-C1 deficit is not plausibly caused by missing coarse
local chemistry in the environment formation.

## 4. Stage B — the internal screen gave a strong (non-transferring) signal

PEC-v0 Gate-2 exact split and protocol (2 000 train, dev = official-train
8 000..8 499, seed 0, Adam lr `1e-3` / wd `1e-5` / batch 64 / L1 / clip 5,
60 epochs, best-dev, fixed Top-5 soup).  Two arms in the same session/device
(the pre-declared device-matched control, audit §2.2):

| arm | params | best dev MAE | Top-5 soup |
|---|---:|---:|---:|
| `CD_matched` (PEC-CD reader `200→64→1`) | 94 049 | 0.474939 | 0.466020 |
| `CD-I1` (S0 pooling `590→22→1`) | 94 145 | 0.427904 | **0.424733** |

```
Δ_frozen  = 0.467108 − 0.424733 = +0.042375     (historical CPU comparator)
Δ_matched = 0.466020 − 0.424733 = +0.041287     (same-session A100 control)
Δ_gate    = min(...)           = +0.041287  >> 0.010
```

Control reproduction drift `M_CD_matched − 0.467108 = −0.001088` (device noise),
so the comparator is matched.  `INTERFACE_SIGNAL_STRONG`; the mechanism was
alive (relation-shuffle `+0.103946`, BAG `+0.463472`), so the full-data seed-0
run was authorized exactly as pre-registered.

## 5. Full data — the signal does not transfer

| model | best valid MAE | best epoch | Top-5 soup |
|---|---:|---:|---:|
| PEC-C1 `CD` (historical, matched) | 0.159782 | 235 | 0.151767 |
| `CD-I1` full seed 0 | 0.159730 | 173 | **0.151657** |

`Δ = +0.000110`, i.e. the two architectures are **statistically
indistinguishable** on this regime (the fold-over noise between the two
historical PEC-C1 arms and seeds is of the order 0.004–0.009).  The 240-epoch
fixed horizon is not the explanation: I1's 48-epoch block minima are
`[0.20047, 0.17822, 0.16472, 0.15973, 0.16155]` with the minimum at epoch 173
and a flat-to-noisy tail; CD's are `[0.21560, 0.17001, 0.16649, 0.16023,
0.15978]` with the minimum at epoch 235.  Both soups gain ≈ 0.008 over their
best checkpoint.

Mechanism at full scale (eval-only on the `CD-I1` soup):

| intervention | valid MAE | degradation | prediction shift |
|---|---:|---:|---:|
| composition relation shuffle | 0.974194 | **+0.822537** | 0.921245 |
| BAG (pooled pair representation zeroed) | 0.967820 | **+0.816163** | 0.904276 |

So the static pair composition branch is alive and load-bearing: the new
pooling did **not** turn composition into a decorative unary bag.  It simply did
not add task information at full data.

## 6. Answers to the round's questions

**Q1 — Local information.** *Substantially present.*  All four coarse local
chemistry blocks (atom shell, bond shell, root atom, incident bond composition)
are recoverable from PEC environment raw primitives with `exact fraction = 1.0`
and float32-round-off error (`≤ 2.7e-8`).  The only S0 shell-pair class absent
from PEC, `(0,0)`, is structurally empty on train and valid.  The single
non-recoverable coordinate is one topology scalar (patch-scoped mean molecule
degree), a documented scope difference, not chemistry.

**Q2 — Interface.** S0-style bucketed `mean/std/log-count` pooling significantly
improves pure PEC DenseRole **only in the cheap underfitting screen**
(`+0.041287`, dev MAE 0.4247 vs 0.4660) and gives **no full-data improvement**
(`+0.000110`, 0.151657 vs 0.151767).  The pre-registered gate was `≥ 0.010` at
full-data materiality `≥ 0.005`; neither transfers.  Exact deltas:
`Δ_screen = +0.041287` (matched) / `+0.042375` (frozen historical);
`Δ_full = +0.000110`.

**Q3 — Composition.** Yes.  With the improved pooling the model still depends on
the static pair composition: relation-correspondence shuffle costs `+0.822537`
and zeroing the pooled pair representation costs `+0.816163` valid MAE.  The
result is not a unary bag.

**Q4 — Purity.** Strictly preserved.  `no MP`, `no recurrence`, `no
pair→centre`, `environment frozen` (pair mutation leaves `E_i` bit-identical),
`no mixed chemistry pair relation` (the 18-D `rho_ij` is topology-only).  G0–G8
all `PASS`, including `max|E_old − E_new| = 0` and `max|c_old − c_new| = 0`
against PEC-CD under identical weights.

**Q5 — Next.** The only allowed verdict is
`STATIC_POOLING_NOT_PRIMARY_GAP`.  The pooling hypothesis is closed.  The next
scientific question (`PEC-I2`, proposal only) is whether the composition
operator needs an explicit **direct chemical bond relation primitive**
`beta_ij` (including `NONE`); the direct bond type, `path_bond_mean` or any
chemistry relation is **not** added in this round.

## 7. What this implies (and what it does not)

* The PEC-C1 `0.151767` deficit is **not** primarily a graph-level
  `mean`/`max` interface artifact.  Giving the pure model S0's full
  moment/count geometry leaves the full-data result unchanged.
* The strong screen result is real *in the screen regime* but is
  **not predictive** of the full-data interface effect: with 2 000 training
  molecules both models are far from converged (dev MAE ≈ 0.42–0.47) and the
  wider moment interface helps fit; at 10 000 molecules the two models converge
  to the same solution quality (dev MAE ≈ 0.152).  This is a methodological
  finding for this track: a cheap small-data screen can over-authorize an
  interface full run.
* It does **not** show the environment is bad, nor that composition is
  unnecessary: the mechanism controls are large and directionally correct.
* It does **not** reopen the dictionary route, and it does not authorize
  `PEC-I2` implementation.

## 8. Purity / discipline audit

* Only `pec_i1.py` and `zinc_pec_i1.py` were added; `pec_v0.py`,
  `pec_v0_gate0.py`, `zinc_pec_v0.py`, `zinc_pec_c1.py` are byte-identical to
  PEC-C1 (imported unchanged).
* The single architectural change is the pooling interface + the
  parameter-matched reader width; `environment` and `pair` parameters are
  **bit-identical** to PEC-CD under the same seed (G0).
* The reader width is the deterministic nearest-integer solution
  (`590→22→1`), chosen without touching `env_hidden` / `env_dim` /
  `pair_hidden` / `pair_dim`.
* Official valid was used only for checkpoint/soup selection; official test was
  never loaded; the loader raises for `"test"`.
* 27 targeted tests pass (`test_pec_i1.py`), 27 with `test_pec_v0.py`
  together (41 in the PEC-v0/C1 suite are unaffected).
* PEC-C1 `CD`/`CK`, PEC-v0 gates, strict-static S0, B-Null, B-Full and all other
  historical controls were **not** re-run.  Exactly one matched `CD` control was
  purchased in the same session, as pre-declared.

## 9. Limitations

* One seed (0), as required by the round.
* The verdict rests on the full-data comparison; the `0.000110` gap is far
  inside the known single-seed spread, so "no gain" is the correct reading, not
  "a tiny gain".
* The 5-bucket pooling width follows S0's audited geometry rather than PEC's
  8-D relation one-hot; the round explicitly forbids trying a second bucket
  scheme.
* `git_dirty = true` in the artifacts is caused solely by this round's untracked
  `results/pec_i1/` output directory; the tracked revision is `a84a6c8`.
* Stage A's scalar-5 probe `R² = 0.316` is a diagnostic, not a gate.
