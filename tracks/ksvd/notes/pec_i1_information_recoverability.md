# PEC-I1 — Stage A: zero-training Patch-B recoverability audit

Round **PEC-I1** · study `zinc-context-gap` · protocol `pec_i1`.
Pre-registration: [`pec_i1_preregistration.md`](pec_i1_preregistration.md) §6.
Prior-artifact audit: [`pec_i1_prior_artifact_audit.md`](pec_i1_prior_artifact_audit.md).

Remote: `hxy@a100-2`, repo `/home/hxy/cy/research`, commit
`a84a6c86477d4a8ebab08f22370d6cf3c8d7e7dd`, official test **never loaded**.
Label-free: the task target `y` is not read anywhere in this audit.

## Method

For the same rooted radius-2 patches, the PEC **environment raw primitives**
(`root_chem`, the coarse `shell × atom` binding block, the coarse
`shellpair × bond` binding block, the 6-D `root_scalars`, and the audited
topology bases `b^V` / `b^E` that feed the dense role map) are compared against
S0 `patch_cont` blocks (`zinc_patch_path_pooling._shell_descriptor`):

```
atom_shell   3 × 28 = 84
bond_shell   6 ×  4 = 24
root_atom          28
incident            4
scalars             6
```

Recovery is analytical with explicit known denominators wherever possible.  A
coordinate counts as exact when `|recovered − S0| <= 1e-5` (float32 round-off
headroom).  Audited molecules: 2 000 official-train + 1 000 official-valid,
**all** roots (69 544 rooted patches).

## Result

```
LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT
```

| block | dim | max abs err | mean abs err | exact fraction | unexplained |
|---|---:|---:|---:|---:|---:|
| atom_shell | 84 | 2.65e-08 | 2.24e-10 | **1.000000** | 0 |
| bond_shell | 24 | 2.75e-08 | 7.33e-10 | **1.000000** | 0 |
| root_atom | 28 | 0.00e+00 | 0.00e+00 | **1.000000** | 0 |
| incident | 4 | 1.99e-08 | 1.57e-09 | **1.000000** | 0 |
| scalars | 6 | 2.40e-01 | 7.72e-03 | 0.837628 | 67 752 |

Recovery maps (all exact):

* `atom_shell = ` coarse `shell × atom` binding `/ expm1(root_scalars[0])`;
* `bond_shell = ` coarse `shellpair × bond` binding `/ expm1(root_scalars[1])`;
* `root_atom = ` root chemistry one-hot;
* `incident = ` coarse `(0,1)` binding `/ root degree` (`root_scalars[2]`);
* `scalars[0..4] = ` exact known-denominator transforms of `root_scalars`
  (patch `n`/`m`, boundary fraction, patch cycle rank `= (m−n+1)/n`, root
  degree `/4`).

Rank/collision: recovered and target block ranks agree exactly
(atom_shell 36/36, bond_shell 9/9, root_atom 12/12, incident 3/3, scalars 6/6 in
the local smoke sample); no rank loss or collision was introduced by the
recovery map.

## Shell-pair taxonomy (the pre-declared focus)

S0 declares **6** shell-pair classes; PEC's role one-hot covers **5** and cannot
express `(0,0)`.  Occurrence frequencies:

| shell pair | train occurrences | valid occurrences | PEC expressible |
|---|---:|---:|---|
| `(0,0)` | **0** | **0** | no |
| `(0,1)` | 498 558 | 49 692 | yes |
| `(0,2)` | **0** | **0** | yes |
| `(1,1)` | 1 887 | 246 | yes |
| `(1,2)` | 688 942 | 68 548 | yes |
| `(2,2)` | 43 457 | 4 448 | yes |

`(0,0)` is **structurally impossible**, not merely absent: shell 0 is the
singleton `{root}`, so a `(0,0)` induced bond would be a root self-loop, which
cannot exist in a simple ZINC graph.  `(0,2)` is expressible in PEC (index 1)
and is also structurally impossible (a distance-2 node adjacent to the root
would be distance 1).  The only S0 shell-pair class PEC cannot express is
therefore *empty* on both splits: **no information is lost**.

## The one non-recoverable coordinate: S0 scalar 5

S0's structural-scalar coordinate 5 is the **patch-scoped mean molecule degree**
(`degrees.mean()/4`, averaged over the patch node set).  PEC's scalar block
stores the **molecule-scoped** mean degree (`root_scalars[3]`) and the audited
node basis `b^V` carries only the **induced** patch degree.  These are the same
topology quantity at a different scope, so no exact analytic map exists.

Quantified (train roots 46 461 → valid roots 23 083):

* molecule-scope analogue `root_scalars[3]/4` MAE = **0.046242**;
* train-fit ridge probe (features: `root_scalars` (6), `global_topo` (8),
  patch induced-degree mean/std/max) valid `R² = 0.316`, MAE = 0.032891,
  max |err| = 0.177174.

So scalar 5 is only partially explained, but it is (i) **topology, not
chemistry**, (ii) one of six coordinates, and (iii) a scope difference of a
quantity PEC already represents at molecule scope plus through the induced
degrees in `b^V`.  Under the pre-registered rule it is recorded as a
**documented definitional (scope) difference**, not an information gap.

## Decision

All four **coarse local chemistry** blocks (atom shell, bond shell, root atom,
incident bond composition) are recovered **exactly** (`exact fraction = 1.0`,
max error ~2.7e-8 = float32 round-off) from PEC environment raw primitives,
after explicit known-denominator transforms.  The only S0 shell-pair class PEC
cannot express is structurally empty on both splits.  The only non-recoverable
coordinate is one topology scalar with a documented scope difference.

```
Stage A verdict: LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT
=> PEC-I1 training is authorized (Stage B).
```

This does not claim the S0 `patch_cont` block is redundant or that PEC's
environment is equivalent to S0's token; PEC-I1 still removes the mixed
`patch_cont` bypass, typed tokens, typed marginals and mixed pair chemistry.
It claims only what the round asks: the PEC-C1 deficit is unlikely to be caused
by missing coarse local chemistry in the environment formation.
