# ERRATA — ZINC G0 transfer audit v1 (Part A statistics + Part C scope)

This errata corrects the **paired bootstrap** of the previous round
(`zinc-topology-crossfit-diagnostic-v1`, result commit `8134630`, science commit
`240bd2b`) and records the scope corrections for the G0 coverage audit.  It does
**not** overwrite the historical metrics, scripts or `DECISION.json`; the old
directory is left byte-for-byte unchanged.  A machine-readable copy is
`part_a_paired_bootstrap.json`.

## E1 — `analyze.py::paired_boot` destroyed the row pairing

The frozen function drew **two independent** bootstrap index vectors:

```python
ea[rng.integers(0, n, n)].mean() - eb[rng.integers(0, n, n)].mean()
```

so arm `a` and arm `b` were scored on different resampled rows.  The comparison
is paired by construction (`p_a` and `p_b` are the same 2000 molecules), so this
only inflates the interval; it does not change the point estimate.
`phase0_corrected_metrics.py` already used the correct one-index form, but the
main `analyze.py` and its reported interval did not.

**Fixed procedure (this round, `part_a_errata.py::paired_boot_fixed`):** per
iteration draw one `idx`, resample the per-row difference
`d_i = |p_a,i - y_i| - |p_b,i - y_i|`; 1000 draws, seed `20261003`; A/B
predictions averaged **before** MAE (protocol preserved).

| positive = improvement | point | corrected 95 % CI | original (broken) 95 % CI |
|---|---:|---|---|
| `TP` over `base` (gain_base) | **−0.02635856** | **[−0.03347119, −0.01906922]** | [−0.06137628, +0.01017032] |
| `TP` over `P` (gain_topo) | **+0.01974322** | **[+0.00737954, +0.03168344]** | [−0.01763686, +0.06045402] |
| `const` over `base` (gain_const) | **+0.00009919** | **[−0.00000737, +0.00020156]** | [−0.03684821, +0.03966323] |

**What changed.** The CIs are now ~4–20× tighter.  Two qualitative changes:

* `gain_base` and `gain_topo` intervals **no longer cover zero**; the sign of
  the failure is now statistically clear (TP is significantly worse than base;
  the topology increment over the scalar prediction is now a clear, if small,
  positive).
* The `const` control is now correctly bounded: a pure global offset of
  `c = 0.00234013` cannot move MAE by more than `|c|`.

**What did NOT change.** The pre-registered gate outcome is unchanged: the
`gain_base ≥ 0.003`, `G0_worsening ≤ 0.001` and per-fold clauses all still
FAIL, and they fail *harder* (the corrected interval lies entirely on the
negative side).  The previous `FAIL` stands; it was simply under-quantified.

### E1 verification (all three pass)

| check | result |
|---|---|
| identical predictions `p_a = p_b` | gain 0, 95 % CI exactly [0, 0] |
| swap `p_a`/`p_b` | point flips sign, CI mirrors (`lo' = −hi`, `hi' = −lo`) |
| constant shift `c = 0.0023401305` | per-row MAE diffs in `[−c, +c]`; paired bootstrap `[−7.37e−6, +2.01e−4]` ⊂ `[−c, c]` |

The old `const` interval of ≈ ±0.04 violated the mathematical bound; the
corrected one does not.

### E1 damage statistics (P/TP vs base, 2000 outer-dev rows)

| arm | MAE gain | improved | worsened | top-10 share of damage | top-50 share of damage | abs-correction p50 / mean / max |
|---|---:|---:|---:|---:|---:|---|
| `P` | −0.046102 | 753 | 1247 | 0.175 | 0.308 | 0.061 / 0.102 / 6.06 |
| `TP` | −0.026359 | 785 | 1215 | 0.095 | 0.233 | 0.062 / 0.094 / 3.66 |

Damage is diffuse, not one outlier: `P` loses on 1247/2000 rows and the worst
50 rows carry only ~31 % of the positive damage.

### E1 typical row — dev 1792

| field | value |
|---|---|
| outer-dev row / train index / penalty | 1792 / 8895 / 0 |
| target | 1.9762 |
| base (A/B avg) | 2.0529; base err **0.0767** |
| A base prediction | 2.1254 |
| A `P` correction / A `P` prediction | **−12.165** / −10.040 |
| A/B-averaged `P` err | **5.987** |
| nearest meta-B neighbour under A base pred | base **2.1249**, target **−15.5405**, residual **−17.6654** |

The scalar nearest meta neighbour has a base prediction within `5e-4` of the
dev row but a residual of `−17.67`; the `min_samples_leaf=1` ExtraTrees `P`
readout copies that residual onto the dev row.  This is the readout-variance
explanation, demonstrated on one row; we did **not** train trees to attribute it
between the `squared_error` criterion, `leaf=1`, and tree averaging — those
remain competing explanations.  `TP > P` is the paired advantage of the two
fixed readouts, **not** proof of a usable topology gain and **not** evidence
that the original Full already uses topology.

## E2 — Part C scope corrections

* The Full node channel is **structurally dead in both folds**:
  `W_A_S ≈ 0` (‖·‖ ≈ 1e-18), `W_A_C = 0`, `node_encoder.0.weight = 0`; the node
  encoder output is constant across nodes (per-dim std 0).  Atom information
  still reaches the model through **Sem108** (`patch_cont[:, :108]`) and the
  anchor size2 block (`semantic_interface`), not through the node binding.
* The Full edge channel is **alive in F_A** (`‖W_E_S‖≈3.15`, `‖W_E_C‖≈1.36`,
  edge-encoder output varies) but **collapsed in F_B** (`‖W_E_S‖≈1.4e-3`,
  `‖W_E_C‖≈1.9e-6`, edge-encoder out ~1e-11).  Coverage/rarity statements about
  F_B's structure channel therefore describe a channel that is not load-bearing
  in that fold.
* The residual-support structure key is a **coarsened** key, not a graph
  isomorphism or full structural identity (as pre-registered).  In practice it
  is extremely coarse: `tied IHT s=8` gives every node exactly 8 nonzeros and
  only **202 (F_A) / 212 (F_B)** distinct support masks over 231,664 nodes.
* The pre-registered full `dict_phi[65]` descriptor key is **also
  low-cardinality**: only **1182 distinct 65-d rows** over 231,664 nodes
  (99.49 % duplicates), **169** distinct edge descriptor pairs, **317** distinct
  edge joint keys.  This is a property of the frozen structural input, not of
  our support coarsening.