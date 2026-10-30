# Structural Binding Pretraining — CPU Feasibility Gate

**Verdict: `NO_NONTRIVIAL_BINDING_SIGNAL`.**

The real mid–end pairing is distinguishable from a marginal-preserving
permutation, but that signal is the deterministic **local type-compatibility
rule**.  Once compatibility is preserved by the negative (hard control), the
remaining signal is small, is not reliably above simple atom/bond-type
baselines, is actually below a no-training context-conditioned frequency table
using degree/ring/type/closure, and transfers worse than a simple linear head
to unseen local combinations.  The binding-vs-marginal comparison is positive,
but the proposition's nontriviality clause is not met.  Do not start a GPU
pretraining experiment on this basis.

All numbers below are from the final 5-fold, 2-seed, 10-epoch CPU run
(`analysis/results.json`, `analysis/report.json`); the post-hoc simple-statistic
table is in `analysis/context_freq.json`.  No property label was read, official
valid/test were not loaded, and CUDA was not used.

## 1. Effective coverage

| Control | Valid roots | / 231,664 | Valid molecules | / 10,000 |
|---|---:|---:|---:|---:|
| Standard marginal-preserving permutation | 113,159 | **48.85%** | 9,986 | **99.86%** |
| Hard compatibility-preserving 2-swap | 370 | **0.16%** | 341 | **3.41%** |

The hard set is **0.33% of the standard-valid roots**.  It is not concentrated
in a few molecules: 314/341 molecules have exactly one hard root, the maximum
is 3, and the top 1% of hard-bearing molecules contain 2.7% of hard roots
(top 20%: 26.5%).  It is, however, structurally narrow:

- 354/370 (95.7%) hard roots are ring atoms;
- every hard root has 4–7 paths (4: 167, 5: 84, 6: 113, 7: 6);
- only four root atom-type indices occur: 0 (281), 4 (57), 2 (28), 1 (4);
- the standard set has 2–9 paths, 74.1% ring atoms, and 18 root type indices.

The standard set is almost always locally incompatible: only 243/113,159
(0.21%) standard roots are compatible on both sides.  A simple deterministic
compatibility rule therefore solves the standard control almost perfectly.

## 2. Standard control: the signal is the compatibility rule

Pair accuracy, 95% bootstrap CI over molecules (ties count 0.5):

| Head | Params | Pair accuracy | 95% CI |
|---|---:|---:|---|
| binding (seed 0) | 19,195 | 0.8677 | [0.8654, 0.8701] |
| binding (seed 1) | 19,195 | 0.8678 | [0.8654, 0.8703] |
| marginal-only (exact 0.5 control) | 19,195 | **0.5000** | [0.5000, 0.5000] |
| linear + compat | 3,975 | 0.9138 | [0.9118, 0.9157] |
| linear pair-only | 3,970 | 0.7895 | [0.7868, 0.7923] |
| pair-frequency table (no training) | 0 | 0.7747 | [0.7717, 0.7777] |
| context-frequency table (no training; degree/ring/type) | 0 | 0.8699 | [0.8676, 0.8722] |
| binding + compat bit | 19,259 | **0.99885** | [0.99866, 0.99904] |
| oracle compatibility rule | 0 | **0.99893** | [0.99878, 0.99906] |

The binding head beats the marginal head by **+36.77 pp** (CI
[+36.52, +37.01]), so the real pairing is not recoverable from the end-type
marginal alone.  But adding one simple compatibility bit takes the same
binding head from 0.8677 to 0.99885, matching the oracle rule.  The linear head
with the same bit reaches 0.9138; the binding head without the bit is worse
than that linear baseline by −4.61 pp (CI [−4.84, −4.39]).  The standard task
therefore measures whether a model learns the deterministic neighbor-type
constraint, not a nontrivial binding rule.

## 3. Hard control: the remaining signal is not beyond simple statistics

The hard negative preserves per-path local compatibility and the end-type
marginal.  Consequently the oracle compatibility rule and the marginal-only
head are exactly 0.5 on it.  Any remaining signal is a pairing-dependent
statistic.

| Head | Pair accuracy | 95% CI | Fold min–max |
|---|---:|---|---|
| binding (seed 0) | 0.5919 | [0.5415, 0.6423] | 0.5072–0.6265 |
| binding (seed 1) | 0.5838 | [0.5305, 0.6346] | 0.5072–0.6184 |
| marginal-only | **0.5000** | [0.5000, 0.5000] | 0.5000 |
| linear + compat (seed 0) | 0.5541 | [0.5052, 0.6060] | 0.5072–0.6301 |
| linear + compat (seed 1) | 0.5568 | [0.5053, 0.6085] | 0.5072–0.6438 |
| linear pair-only | 0.5595 | [0.5080, 0.6087] | 0.5072–0.6438 |
| binding + compat bit | 0.6162 | [0.5647, 0.6667] | **0.4348**–0.7349 |
| pair-frequency table (no training) | 0.5811 | [0.5308, 0.6317] | 0.5060–0.6849 |
| context-frequency table (no training; degree/ring/type) | **0.6514** | [0.6011, 0.7000] | 0.5542–0.7671 |
| oracle compatibility rule | **0.5000** | [0.5000, 0.5000] | 0.5000 |

The binding head is above chance, but so are the simple baselines.  Paired
molecule-bootstrap differences on the same 370 roots:

| Comparison | Difference | 95% CI |
|---|---:|---|
| binding − marginal | **+9.19 pp** | [+4.10, +14.16] |
| binding − pair-frequency table | +1.08 pp | **[−5.11, +7.18]** |
| binding − linear pair-only | +3.24 pp | **[−0.54, +7.32]** |
| binding − linear + compat | +3.78 pp | [+0.00, +7.84] |
| frequency table − marginal | **+8.11 pp** | [+3.01, +13.00] |
| linear pair-only − marginal | **+5.95 pp** | [+0.81, +11.01] |

The hard signal is not reliably larger than the simple pair-frequency table or
the pair-only linear head.  The 95% CI for binding − frequency is centred near
zero; the CI for binding − linear pair-only just includes zero.  The binding
head is not a unique detector of the real correspondence.

The strongest simple baseline is a **no-training context-conditioned
pair-frequency table**: it counts `(mid, end)` pairs in the training molecules
conditional on `(root_deg, root_ring, root_mring, mid_deg, mid_ring, closure)`,
with Dirichlet backoff to the global pair frequency (`alpha = 20`).  This uses
exactly the simple atom type, bond type, degree, ring-flag, min-ring, and
closure features named in the task.  It reaches **0.6514** on the hard control
[0.6011, 0.7000] and **0.8699** on the standard control [0.8676, 0.8722].
Its hard point estimate is **+5.95 pp** above the binding head (CI
[0.00, +11.75]), and it matches/beats the binding head on standard by +0.21 pp
(CI [+0.01, +0.42]).  The binding head therefore adds nothing beyond a simple
context-conditioned frequency statistic; the simple table is at least as good,
and numerically better on the hard control.

The frequency margin (sum of training log pair frequencies) and the binding
margin correlate at **r = 0.727**.  Binding accuracy is 0.688 on roots where the
frequency table prefers the real pairing and 0.458 where it prefers the fake.
The quartile profile is not monotonic, so the binding head is not literally
the frequency table; but the frequency table alone is already above chance,
and the binding head does not reliably exceed it.  A weak simple statistic is
therefore sufficient to explain the hard control.

The hard signal is also unstable at the fold/head level: the binding seed 0
fold minimum is 0.5072 (barely above chance), the binding seed 1 fold minimum
is 0.5072, and the binding+compat head has a fold at 0.4348 (below chance).
The hard set is small and structurally narrow, so the 95% CI spans roughly
±5 pp; this is not a basis for a general scientific claim.

## 4. Transfer to structurally novel local combinations

Natural novelty is sparse: 673/113,159 test roots (0.595%) have at least one
root key unseen in the training molecules; 43/113,159 (0.038%) have an unseen
`(mid, end)` pair type.  Pair accuracy:

| Group | n roots | binding | linear + compat | binding − linear |
|---|---:|---:|---:|---:|
| Seen keys | 112,486 | 0.8687 [0.8663, 0.8711] | 0.9139 [0.9120, 0.9158] | **−4.52 pp** [−4.74, −4.30] |
| Unseen key | 673 | 0.7088 [0.6712, 0.7453] | 0.9004 [0.8755, 0.9243] | **−19.17 pp** [−23.01, −15.36] |
| Unseen pair | 43 | 0.7442 [0.6206, 0.8571] | 0.9070 [0.7742, 1.0000] | −16.28 pp [−32.35, 0.00] |

The binding head's seen→unseen gap is **15.99 pp**; the linear head's gap is
**1.34 pp**.  The binding head does not transfer better to novel local
combinations; it transfers substantially worse than the simple linear model.
This argues against a compositional binding advantage.

## 5. Leakage and invariance checks

| Check | Result |
|---|---|
| Independent LCRC array rebuild vs frozen cache | all arrays bit-identical |
| Random atom renumbering (500 molecules, 11,575 roots) | 0 signature mismatches |
| Random path storage permutation (512 rows) | max output diff 0.0 |
| Marginal head real vs fake | max score diff 0.0; tie rate 1.0 on every fold/seed |
| Molecule-isolated folds | 0 duplicate fingerprints crossing folds |
| Random-label training, proper permuted test | 0.5005 / 0.5008 (seeds 0/1) |
| Untrained head, permuted test | 0.5028 |
| Labels accessed / official test loaded / CUDA used | false / false / false |

The random-label **clean-test** score is 0.541 / 0.524, slightly above 0.5.
This is an input/initialization asymmetry, not label leakage: the untrained
clean-test score is 0.509, while the proper permuted-label test is at chance
(0.5005 / 0.5008).  The permuted-label test is the relevant null.

## 6. Gate decision

Operationalization of the requested strict criterion:

| Criterion | Result |
|---|---|
| Hard binding above chance | pass: 0.5919, CI low 0.5415 |
| Hard binding beats frequency table | **fail:** +1.08 pp, CI [−5.11, +7.18] |
| Hard binding beats linear pair-only | **fail:** +3.24 pp, CI [−0.54, +7.32] |
| Hard binding beats context-frequency table | **fail:** −5.95 pp point estimate (CI [0.00, +11.75]); simple table at least as good |
| Both seeds above chance | pass: 0.5919 / 0.5838 |
| Seed gap < 5 pp | pass overall: 0.81 pp (fold 2 gap 10.1 pp) |
| Every fold above 0.5 | pass barely: min 0.5072 |
| At least 300 hard roots | pass: 370 |

**The decisive simple-statistic checks fail.**  The hard binding head is not
reliably beyond the pair-frequency table or the pair-only linear head, and it is
actually below the no-training context-frequency table; the standard signal is
the compatibility rule.  The gate returns `NO_NONTRIVIAL_BINDING_SIGNAL`.

## 7. Limitations

- The hard control covers only 370 roots in 341 molecules, all with 4–7 paths,
  95.7% ring atoms, and only four root atom-type indices.  It is too narrow
  for a general claim and too small to rule out a very small effect (<3 pp).
- The hard negative preserves per-path compatibility but not the joint
  pair-type marginal; a global pair-frequency prior remains, and the simple
  frequency table detects it.
- The hard negative is a uniform compatibility-preserving 2-swap, not a full
  conditional null over all compatible pairings.
- The standard control is trivially solvable by the compatibility rule, so it
  is not a binding test.
- The random-label clean-test score is slightly above chance due to input and
  initialization asymmetry; the permuted-label test is the proper control.

## 8. Reproducibility

- Protocol: `PROTOCOL.md`
- Code: `code/sbf_common.py`, `code/sbf_data.py`, `code/sbf_models.py`,
  `code/sbf_run.py`, `code/sbf_analysis.py`, `code/sbf_context_freq.py`
- Durable results: `analysis/results.json`, `analysis/report.json`,
  `analysis/coverage.json`, `analysis/context_freq.json`
- Local, gitignored per-root arrays (regenerable with the two run scripts):
  `analysis/raw_arrays.npz`, `analysis/context_freq_raw.npz`
- File hashes: `MANIFEST.json`
- Frozen binding-cache content SHA-256:
  `ffb25c800e102a6fb0796cb3bc84056e167309793160c2ef646ffe9e17a3d7bf`
- Frozen binding-cache file SHA-256:
  `0b1449d6ea5576e196058d3fc19628ccf85c3cc1d6dcf70272c9e0400030db3f`
- Run environment: Python 3.12.15, torch 2.5.1+cu124 (CPU only),
  numpy 2.1.3; run time 525 s.
- Repo commit at run time: `db13e5b805fab2de3c5a3a54833be3917b973804`.
  The track directory was an untracked working-tree addition at that time.
