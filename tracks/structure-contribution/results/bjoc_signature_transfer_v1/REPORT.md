# bjoc-signature-transfer-v1 — cross-molecule transfer of Junction signature importance

**Verdict (frozen rule, evaluated on merged molecule-unit k=4 contrasts): `CONTEXT_DEPENDENT_OR_PARTIAL_TRANSFER`**

Gates: `M0 ✓ G1 ✓ G2 ✓ G3 ✗ G4 ✓ G5 ✓ G6 ✓ G7 ✗ G8 ✓`.
Discovery: 256 frozen Phase-0 train molecules × 2 R3 seeds → molecule-capped table of
58 candidate signatures (≥20 molecules, 87.1 % of discovery junctions covered).
Validation: 1 000 official-valid structures, labels never read, natural-replacement
`|Δ|`, molecule unit, two seeds. Phase-0 replay is bit-exact (IG, top-4 sets,
reference rows, responses 0.0 diff over 256×2 molecules), so the reused attribution
is exactly the historical one.

## 1. The structure-level signal does transfer

| contrast (k=4) | mean | 95 % CI | sign p |
| --- | --- | --- | --- |
| SIG − RANDOM | +0.00946 | [+0.00772, +0.01120] | 2.3e-22 |
| SIG − RING | +0.01531 | [+0.01304, +0.01760] | 3.6e-45 |
| SIG − RING_DEGREE | +0.01382 | [+0.01155, +0.01606] | 6.8e-39 |
| SIG − DEGREE | +0.01164 | [+0.00940, +0.01381] | 3.4e-34 |
| SIG − SHUFFLE (16 perms) | +0.00960 | [+0.00801, +0.01116] | 8.4e-31 |
| SIG − RARE | −0.00058 | [−0.00376, +0.00249] | — |
| SIG − DEV | −0.03657 | [−0.04462, −0.03090] | — |

Absolute SIG mean 0.05430 vs RANDOM 0.04483. Both seeds separately positive
(+0.00944, +0.00949; CI lower > 0.0075). Cross-scaffold: the 961 valid molecules
whose Murcko scaffold is absent from discovery give +0.00920 [+0.00741, +0.01092]
(the 39 scaffold-sharing molecules +0.01605 [+0.00198, +0.03172]). Internal
leave-molecules-out control is positive (+0.00711 [+0.00511, +0.00924]),
so the pipeline recovers in-distribution transfer as well.

The class ordering itself transfers almost perfectly: Pearson 0.927,
Spearman **0.914 [0.885, 0.933]** over the 58 candidate classes (per seed 0.82 / 0.86),
i.e. the model up-weights the same signature classes on molecules it never saw in
weight training. 98.5 % of SIG top-4 junctions carry a candidate class;
85.3 % of all valid junctions do.

## 2. Why only partial

* A simple per-junction baseline beats the table at selection: encoder deviation
  DEV = 0.09087 vs SIG 0.05430, so G3 fails; within-molecule Spearman with model
  `|IG|` is 0.637 (DEV) vs 0.244 (SIG). Rarity is statistically indistinguishable
  from SIG (CI crosses 0).
* The top-4 response gain is only **17.8 %** of the oracle−random gap (G7 threshold
  0.25; ORACLE 0.09807, SIG 0.05430, RANDOM 0.04483). The class-mean table is too
  coarse for fine-grained top-k selection even though it orders classes well.
* Context modulates the magnitude: across classes with ≥2 strata the within-class
  mean-percentile range is 0.180 (median; p90 0.283) for center degree,
  0.102 (p90 0.264) for endpoint ring count, and 0.054–0.060 for molecule size and
  ring-count terciles. Example: class 56 (`C` center, two single C–O bonds, in ring,
  1 ring bond, T=0.332) sits at mean percentile 0.353 at degree 3 but 0.618 at
  degree 4+; class 4 (acyclic `C` with C=O/C–O, T=0.696) 0.711 (endpoint not in
  ring) vs 0.649 (endpoint in ring). Same signature, different context, different
  encoded importance.
* Signed `Δ` is not stable across seeds (mean −0.0023 / +0.0027), so no signed
  direction claim is made; all primary claims use `|Δ|`.

## 3. Chemical-semantics check (pre-selected classes)

Selected by the frozen rule: top-3 T (support ≥50) classes 4, 5, 7; bottom-3
classes 56, 54, 48; most frequent class 43. RDKit verification: ring-membership
11 745/11 745 identical to the signature fields; non-aromatic bond orders
6 191/6 191 exact. The 17 299 RDKit-aromatic bonds all carry the dataset's
kekulized SINGLE/DOUBLE words (raw 26 % "match" is an aromaticity-perception
convention, not a chemistry error; `semantics_rdkit_detail.json`). Instances show
the same signature varying from percentile 0.0 (counterexamples) to >0.88 across
different molecules. No chemical-group names are assigned: these are two-bond
signatures, and context effects (b) are demonstrated while higher-order connected
structure discovery (c) is out of scope.

## 4. Scope and limitations

Everything here is **predictor-internal encoded-source attribution of the frozen
R3 model** (no label was read, no property explanation is claimed). Official valid
is a historically exposed development set, not a blind test; the cache-level
canonical SMILES cannot express uncharged N/O/S H counts; only 39/1 000 valid
molecules share a discovery scaffold, so cross-scaffold evidence is dominated by
truly new scaffolds. No new model, training, GPU or search was used.

## 5. Judgment

Worth continuing as a **cross-molecule structure-level importance vocabulary** —
class ordering transfers strongly and beats random, ring/degree rules, shuffle
controls, cross-scaffold and both seeds. It is **not** yet a competitive
per-junction selector: encoder deviation is stronger at top-k, and the table
captures only ~18 % of the oracle response gap. The defensible claim is
context-dependent shared structure sensitivity: signature identity sets the
typical encoded importance, and center degree / substitution environment shift
the magnitude within a class. Next work should operate at the class×context
level and use per-molecule (not junction-only) units, rather than treating the
signature as a precise local-importance predictor.
