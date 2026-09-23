# PEC-I1 — prior-artifact audit (Static Composition Interface Audit)

Round **PEC-I1** · study `zinc-context-gap` · protocol `pec_i1`.
Written **before** any PEC-I1 code/experiment.
Parent rounds: [`pec_v0_preregistration.md`](pec_v0_preregistration.md),
[`pec_c1_analysis.md`](pec_c1_analysis.md).
Recorded lineage HEAD at audit time: `cff7356` (clean worktree).

This note modifies no historical record. PEC-C1's frozen verdict stays:

```
PURE_ENV_COMPOSITION_ABSOLUTE_WEAK
seed1_authorized = false
```

---

## 0. Why this audit exists

PEC-C1 (`0.151767`) sits `+0.010973` above the strict-static S0 seed-0 Top-5
soup (`0.140794`) in the same 10 000 / 1 000 official-train → official-valid
protocol family, while its mechanisms are decisively load-bearing
(`+1.576104` chemistry-placement shuffle, `+1.967415` neutral dictionary,
`+1.392907` relation shuffle).  The open question is *where* the remaining gap
lives.  PEC-I1 tests exactly one candidate: the graph-level **composition /
readout interface** (`mean`/`max` pooling) may be too poor to expose the
environment-composition mechanism that S0 already knew how to read.

## Q1 — What is the single change between PEC-C1 and PEC-I1?

**Only the composition / readout statistics interface.**

| frozen object | PEC-C1 | PEC-I1 (planned) |
|---|---|---|
| radius / node set / patch | radius 2, `fsar_v2._explicit_basis_for_patch` | **identical** |
| node basis `b^V ∈ R^11` | audited FSAR explicit rooted node basis | **identical** |
| edge basis `b^E ∈ R^15` | audited FSAR explicit rooted edge basis | **identical** |
| atom primitive `a(q_v)` | one-hot 28 ZINC categories | **identical** |
| bond primitive `c(b_e)` | one-hot 4 ZINC categories | **identical** |
| chemistry primitive | unchanged | **identical** |
| dictionary / dense role definition | `Linear(11→16)` / `Linear(15→16)`, `Dᵀ`-init | **identical (DenseRole only; dictionary removed from the round, §19)** |
| environment MLP `H` | `650→96→48`, SiLU | **identical** (same shapes, same seed, same RNG order) |
| pair relation `rho_ij ∈ R^18` | topology-only | **identical, topology-only** |
| pair MLP `F` | `162→64→48`, computed once | **identical** |
| training objective | L1 / Adam / clip 5 | **identical** |
| **unary pooling** | `[mean(E), max(E)]` (96) | **`[mean(E), std(E), log1p(n)]` (97)** |
| **pair pooling** | `[mean(c), max(c)]` (96) | **per S0 distance bucket `d ∈ {1,2,3,4,5+}`: `[mean(c), std(c), log1p(n_d)]`, 5 × 97 = 485** |
| `global_topo` | PEC-v0 8-D vector | **identical (no typed marginals added)** |
| reader | `200→64→1` | **`590→22→1` (parameter-matched, §10)** |

No environment basis, chemistry primitive, dictionary/dense-role definition,
environment MLP, pair-MLP functional input, pair-relation chemistry, or training
objective is modified by PEC-I1.

## Q2 — Why is this not a redo of S0?

PEC-I1 keeps the PEC environment factorization **pure**.  It does **not**
restore any S0 mixed bypass:

* no raw `patch_cont` mixed input (PEC forms `E_i` only from rooted roles bound
  to chemistry; S0's local token concatenates raw chemistry descriptors);
* no typed token / parent token / B-Full / B-Bag / B-Null path;
* no typed global marginals (`global_topo` stays the 8-D PEC topology vector);
* no mixed pair chemistry (`rho_ij` stays the 18-D topology-only subset, i.e.
  S0's relation minus `path_bond_mean` and the adjacent-bond one-hot);
* no pair→centre / recurrence / centre update.

Only S0's **already-audited pooling geometry** is borrowed: first/second
moments plus log-count, per shortest-path distance bucket, exactly the
`zinc_patch_path_pooling.py` `mean_std` semantics (`compact_v4` readout
`unary 97 + pair 5×33`).

## Q3 — Why not a task-coupled dictionary?

* PEC-C1's absolute pure class is already weak; DenseRole is the stronger role
  coordinate inside it (`CK − CD = +0.009204`).
* The current bottleneck is the absolute capacity of the pure no-MP function
  class, not the sparse coordinate.
* Dictionary-specific questions must wait until a pure **Dense** architecture
  re-enters the acceptable absolute band.  PEC-I1 therefore touches **DenseRole
  only**; `CK`, K-SVD refit, `K`/`s` sweeps, neutral-dictionary and sparse-role
  diagnostics are all out of scope (§19).

## Q4 — Why not add bond chemistry directly?

Because that is a *different scientific variable*:
`composition relation primitive chemistry`.  PEC-I1 must first test whether the
**aggregation / interface alone** is a material gap under the existing pure
topology relation.  If it is not, the next question (`PEC-I2`, proposal only —
not implemented this round) becomes whether the composition operator needs an
explicit direct chemical bond relation primitive `beta_ij`
(including `NONE`).  Adding chemistry now would confound the interface test.

## Q5 — Is there any prior artifact that already answers this?

No.  Dedupe check against the nearest prior objects:

| prior | object | decisive difference from PEC-I1 |
|---|---|---|
| PEC-v0 / PEC-C1 | pure env composition, `mean`/`max` readout | this is the object PEC-I1 changes |
| strict-static S0 (`zinc_patch_path_pooling`) | mixed `patch_cont` backbone, `mean_std` bucket moments | mixed typed bypass + different environment formation; PEC-I1 borrows only the pooling |
| SDB-v0 Stage 4 | global `C~_D` added to S0 as linear residual | additive on a strong mixed backbone; not per-root composition |
| SDPK-v0 | dictionary coordinates inside the S0 static pair kernel | keeps the mixed local token; soft `tau`-gate kernel |
| compact-v4 readouts (`compact_v4_global_topology_channel.md`, `compact_v4_learned_centre_composer.md`) | moment pooling studied on the mixed backbone | never applied to a pure `E_i` environment-composition class |

**Dedupe verdict: no equivalent PEC-I1 implementation exists.**  No prior run
combines (a) PEC's pure frozen environment, (b) the pure-topology 18-D pair
relation, (c) a single read-only static pair composition, and (d) S0-style
bucketed mean/std/log-count pooling with a parameter-matched reader.

## 1. Frozen historical references (never re-run)

`official_test_loaded = false` everywhere below; official test is never loaded.

PEC-v0 internal 2 000 train / 500 dev (official-train molecules 8 000..8 499),
seed 0, Adam lr `1e-3` / wd `1e-5` / batch 64 / L1 / clip 5 / 60 epochs /
best-dev / fixed Top-5 soup:

```
C0_coarse soup = 0.466904   best = 0.470864
CD_dense  soup = 0.467108   best = 0.476846   <-- matched PEC-I1 comparator
CK_sparse soup = 0.461811   best = 0.465776
CK_BAG    soup = 0.496982
CK_SHUFFLE soup = 0.473408
```

PEC-C1 full official-train 10 000 → official-valid 1 000, seed 0, 240 epochs,
no early stop, Top-5 soup:

```
CD_dense  soup = 0.151767   best = 0.159782 @235
CK_sparse soup = 0.160971   best = 0.169599 @210
```

Strict-static / null orientation only (different computation class, never a
gate):

```
strict-static S0 seed0 soup = 0.140794   best = 0.145674 @163
strict-static S0 seed1 soup = 0.136423
B-Null ≈ 0.123
B-Full ≈ 0.1198
```

## 2. Matched-comparator audit (split / seed / optimizer / protocol)

### 2.1 Full-data comparator (PEC-C1 CD `0.151767`)

| quantity | PEC-C1 CD | PEC-I1 full (planned) | match |
|---|---|---|---|
| train | official-train 0..9 999 | same | yes |
| valid | official-valid 0..999 | same | yes |
| role | dense `Linear(11→16)`/`Linear(15→16)`, `Dᵀ`-init from full-train K-SVD | same | yes |
| seed | 0 | 0 | yes |
| optimizer | Adam lr `1e-3`, wd `1e-5`, batch 64, L1, clip 5, no scheduler | same | yes |
| epochs / selection | 240 fixed, no early stop, best official-valid ckpt, Top-5 soup | same | yes |
| device regime | A100-SXM4-40GB, CUDA 12.4, torch 2.5.1+cu124 | same remote, same stack | yes |
| code | `pec_v0.py` unchanged | imported unchanged | yes |

The full-data comparator is **matched**; no PEC-C1 CD re-run is purchased.
The only intentional difference is the pooling interface and the
parameter-matched reader width.

### 2.2 Internal-screen comparator (PEC-v0 CD `0.467108`)

| quantity | PEC-v0 Gate-2 CD | PEC-I1 screen (planned) | match |
|---|---|---|---|
| train | first 2 000 official-train | same | yes |
| dev | official-train 8 000..8 499 | same | yes |
| dictionaries | K-SVD fit on the 2 000 Gate-2 train molecules only (`sdb_v0.fit_ksvd`, default `DICT_SEED`) | same deterministic fit | yes |
| dense init | `m_node.weight = d_nodeᵀ` | same | yes |
| seed | 0 | 0 | yes |
| optimizer / loss | Adam lr `1e-3`, wd `1e-5`, batch 64, L1, clip 5, no scheduler | same | yes |
| epochs / selection | 60, best-dev checkpoint, fixed Top-5 soup | same | yes |
| **device** | **CPU, host `calendar`** (`git_commit 883e529`) | remote A100 (GPU) | **MISMATCH** |

Because the historical internal-screen artifact is a **CPU / host-`calendar`**
run while the round's screen executes on the A100, the device-regime mismatch
means exact bit-comparability **cannot be claimed**.  Per the round brief §13 and
the runner's GPU-baseline rule, PEC-I1 therefore **pre-declares exactly one
matched CD baseline control**, trained in the *same* session, device and code as
`CD-I1`:

* `CD_matched` — DenseRole, PEC-v0 `mean`/`max` reader (`200→64→1`), same split,
  same seed, same protocol, same device;
* `CD-I1` — the candidate.

Both `Δ_interface_frozen = 0.467108 − M_CD-I1` (the brief's primary) and
`Δ_interface_matched = M_CD_matched − M_CD-I1` (the device-matched audit) are
reported, and PASS requires **both** to clear `0.010`.  The reproduction drift
`M_CD_matched − 0.467108` is reported as a device-noise estimate.

### 2.3 What is *not* re-run

PEC-v0 (any gate), PEC-C1 (`CK`/`CD`), strict-static S0, B-Null, B-Full,
B-Bag, SDB/SDPK/SRDA, TCCD, FSAR, and any other historical control whose
protocol/identity is fixed by a durable artifact.  No `K`/`s` refit.

## 3. Shell-pair taxonomy audit (pre-committed to Stage A)

This is the one place where the S0 and PEC schemas provably differ and it must
not be silently ignored.

```
S0 (zinc_patch_path_pooling.SHELL_PAIRS, fsar_v2.SHELL_PAIRS) = 6 classes:
    (0,0) (0,1) (0,2) (1,1) (1,2) (2,2)

PEC (pec_v0.SHELLPAIRS, used in r^E one-hot and eocc_shellpair) = 5 classes:
    (0,1) (0,2) (1,1) (1,2) (2,2)
```

`(0,0)` is the class present in S0 and absent from PEC's role one-hot.  Stage A
must report its occurrence frequency on train / valid and decide whether it is a
task-relevant information gap.  The a-priori structural argument: shell 0 is the
singleton `{root}`, so a `(0,0)` induced bond would be a root self-loop, which
cannot exist in a simple ZINC graph.  `(0,2)` is present in PEC (index 1) but is
likewise structurally impossible (a distance-2 node adjacent to the root would
be distance 1).  Stage A confirms both frequencies empirically before any
training.

## 4. Frozen verdict vocabulary (only these)

`LOCAL_INFORMATION_GAP_FOUND` ·
`LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT` (Stage A) ·
`INTERFACE_NOT_PRIMARY_GAP` ·
`INTERFACE_SIGNAL_WEAK` ·
`INTERFACE_SIGNAL_STRONG_INTERNAL` (Stage B) ·
`PURE_STATIC_INTERFACE_VIABLE` ·
`MATCHES_OR_BEATS_HISTORICAL_STRICT_STATIC_S0` ·
`INTERFACE_MATTERS_BUT_NOT_SUFFICIENT` ·
`STATIC_POOLING_NOT_PRIMARY_GAP` (full).

No fuzzy "close but keep tuning" verdict is permitted.  One test, one
conclusion.
