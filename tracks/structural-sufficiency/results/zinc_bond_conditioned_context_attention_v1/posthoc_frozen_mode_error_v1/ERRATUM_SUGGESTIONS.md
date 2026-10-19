# ERRATUM suggestions — documentation only

**Status:** suggestions, not applied. The frozen `PROTOCOL.md`, `REPORT.md`,
config and result records are untouched and must remain so. These are wording /
code-description corrections, **not** evidence that the original training
execution was wrong, and they are not a reason to rerun the experiment.

References below are to the code that actually ran (`commit a6b51bb` /
`4e1ae84`) and to the frozen protocol/report.

---

## 1. C attention covers all junctions of the molecule, not only the bond's incident junctions

**Where the frozen text is misleading**

* `PROTOCOL.md` §4.1: “strictly within the bond's own molecule” then “(by
  construction) only to junctions incident to itself”, and “a dense
  `[n_bonds, n_junctions]` block-diagonal mask built from `junc_bond`”.
* `REPORT.md` §1 (arms): “per-bond conditional attention over the bond's own
  junctions”; `PROTOCOL.md` §4.2/§4.5 and the §441 restatement repeat the same
  wording.

**What the code does**

`bcca_models.pair_index` (lines 114–131) pairs each bond with every junction of
its molecule: `per_bond = jcount[bb]`, `joff[bb]`. It never consults
`junc_bond`. `junc_bond` is used only by `bond_junction_counts` (line 105) to
count `n_e`. The block-diagonal structure is **molecule-level**, not
bond-incidence-level.

**Suggested wording**

> Each bond attends over **all junctions of its own molecule**; the attention
> distribution is per-bond in C and shared across the molecule in G. The model
> does not restrict attention to the junctions incident to the bond.

**Consequence for interpretation**

“Per-bond conditional” means the mixture weights differ per bond; it does not
mean local bond-neighbourhood attention. C vs G still isolates per-bond versus
per-molecule marginal allocation, but it does not isolate locality.

---

## 2. `n_e` is the number of junctions incident to the bond, not twice that number

**Where the frozen text is wrong**

* `PROTOCOL.md` §3: `n_e = deg(u) + deg(v) - 2 = 2 |J_e|`; “each junction is
  counted once from each endpoint, so `n_e` is the number of endpoint
  incidences”; “the implementation checks `n_e == 2 * |J_e|`”.
* The same paragraph says the Null moments are “invariant to how many junctions
  a bond has; `n_e` only rescales.”

**What the code does**

`bond_junction_counts` (line 105) does one `index_add_` over
`junc_bond.reshape(-1)`. A bond `e=(u,v)` is incident to exactly
`deg(u)+deg(v)-2` junction entries, so

```
n_e = deg(u) + deg(v) - 2 = |J_e|.
```

The global identity `Σ_e n_e = 2 |J|` holds because each junction has two
incident bonds; it does **not** imply `n_e = 2 |J_e|` for each bond. The CPU
gate `sum_n_e_equals_2_times_junctions` (`bcca_runner.py` lines 639–662) is a
global identity, not a per-bond check.

**Related correction**

With `n_e = |J_e|` and `W_e = n_e * (1/|J_e|) Σ_j w_j`:

```
W_e = Σ_{j∈J_e} w_j,      V_e = Σ_{j∈J_e} w_j².
```

As written, “both are invariant to how many junctions a bond has” is inaccurate:
the per-junction mean `W_e / n_e` is invariant, but `W_e` itself scales with
`|J_e|`; `n_e` restores the sum. The sentence should be removed or rewritten.

**Consequence**

Documentation/description only. The code, frozen arrays and original scores are
unchanged, but the normaliser is a count-like sum, not a constant-times-mean.

---

## 3. Composer input is 296; reader input is 321

**Where the frozen text is wrong**

`PROTOCOL.md` §3 describes
`b_e = [h_u, h_v, e_uv, W_e, V_e]` as “(dim 96 each after the frozen
projections, 321 total before the 32-d reader)”. `321` is the reader width, not
the composer width, and the composer input has two more slots.

**What the code does**

For `BondContextBJOC` (extends `structural_relation_composer_v1.BJOC`):

* composer input (`BJOCConfig.compose_in`, `structural_relation_composer_v1.py`
  line 113):
  `b (96) + W (96) + V (96) + log1p(n_b) (1) + bond_feat (7) = 296`.
* reader input (`BJOCConfig.reader_in`, line 117):
  `z (96) + a_sum (96) + a_sq (96) + global (33) = 321`.

**Suggested wording**

> The composer receives 296 features per bond:
> `[b, W, V, log1p(n_b), bond_feat]` with widths `96+96+96+1+7`. The reader
> receives 321 features per molecule: `[z, a_sum, a_sq, g]` with widths
> `96+96+96+33`.

Do not use 417 for this round: that is the widened
`BJOCSecondMoment`/C1 reader in `bjoc_perfmech_long_v1.py`, which is a different
control and is not the model trained here.

---

## 4. “Frozen encoder” means frozen source/init/soup, not frozen weights during training

**Where the frozen text is misleading**

* `REPORT.md` §1: “same frozen P3 encoder”.
* `PROTOCOL.md` §4.2: “everything else in the encoder and composer is frozen
  P3”; §4.4: “`W_e, V_e` remain the frozen P3 Null moments”.

**What the code does**

`bcca_runner._train_loop` builds the optimizer with
`Adam(model.parameters())` (`bcca_runner.py` line 1069). In every formal C/G/F
run, the atom encoder, bond encoder, junction encoder, composer and reader are
all trainable and are optimized. “Frozen” refers to the historical P3
checkpoint/recipe/initialisation and to the frozen artefacts, not to
`requires_grad=False`.

**Suggested wording**

> Same P3 initialisation and same frozen R3 recipe; all shared weights remain
> trainable during the formal run.

This matters for interpretation: the C/G networks can co-adapt their encoders
and reader to the new context mapping. The current post-hoc result (uniform
restoration is much worse) is consistent with exactly that co-adaptation.

---

## 5. F is a parameter-count control, not an exhaustive capacity/optimization control

**Where the frozen text is too strong**

* `REPORT.md` header table (“capacity axis”) and §6 question 2: “the loss is
  not a capacity effect”.
* `PROTOCOL.md` §4.5: “C vs F differ only in where the new capacity sits”.
* The empirical C−F result itself is not in question; only the causal label is.

**What is actually controlled**

F has the same number of new parameters (6,144), but a different function class
and different initialisation: F uses `T` random / `U` zero and `b + U SiLU(T
LN(b))`, while C/G use `Q` zero / `K` random and a softmax over junctions. The
gradient paths, nonlinearity and effective capacity therefore differ.

**Suggested wording**

> F matches the raw added-parameter count. C−F therefore argues against a
> pure parameter-count explanation; it does not exclude all differences in
> effective capacity or optimization dynamics.

Keep the empirical C−F result, but do not label it a definitive capacity
decomposition.

---

## 6. Narrow the causal wording around “the loss comes from reweighting itself”

**Where the frozen text is too strong**

* `REPORT.md` §0 TL;DR item 4: “The loss comes from replacing the Null moments
  with a learned reweighting at all”.
* `REPORT.md` §9: “the freedom is spent in a direction that generalises worse”.
* The same causal shorthand appears in the header table (“the loss is not a
  capacity effect”) and §6 question 2.

**Why it should be narrowed**

The original comparison is between two **trained solutions** (P3 vs C/G), not a
randomised intervention on the context mapping with everything else held fixed.
The new post-hoc analysis shows that forcing the uniform Null mapping under the
C/G weights is much worse, not better. That is evidence of co-adaptation
between the learned context and the rest of the network; it does not isolate a
causal effect of “reweighting” alone. Same-weight mode switching is also an
off-training-distribution intervention.

**Suggested wording**

> The C/G-trained solutions generalise worse than P3. Under the same trained
> weights, restoring the uniform Null mapping is substantially worse, which is
> consistent with co-adaptation between the attention context and the rest of
> the network. The evidence does not isolate a causal effect of the
> reweighting operation alone.

Also prefer “associated with worse valid error in these trained solutions” over
“the effect on valid error is negative” when a causal reading is not intended.

---

## What not to do

* Do not edit the frozen `PROTOCOL.md`, `REPORT.md`, config, metrics or
  `valid_read.lock` to apply these suggestions.
* Do not describe these as training-execution bugs, and do not rerun the
  original six runs because of them.
* If the documentation is ever revised, make the revision an explicit erratum
  with a pointer to this file and to the code lines above.
