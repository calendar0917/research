# Knowledge cards — ISRD-v0

Rules are **whole-molecule threshold predicates** on frozen scaled structure counts
`x_s = count_s / sqrt(max(n_atoms,1))`. They are model-computational / predictive
associations only: no chemical interaction, local connection, or causal claim.
All conditions and supports come from Train; Valid/Test are reported as
`DEVELOPMENT_TEST`. Raw-unit coefficients are from `analysis/raw_coefficients.json`.
"Switch off" is a frozen-model intervention and measures dependence of the fitted
logit on the term, **not** a counterfactual molecular edit.

---

## Card 1 — high-support aromatic C–C–O triangle (COARSE 157)

**Structure object.** `COARSE` column 157; canonical key
`N[5,5,1;5,5,1;7,5,1]E[0-1:3;1-2:3]`; decoded
`a:Carx2 b:Oarx1`, bonds `a-a:AROMATICx1 a-b:AROMATICx1`, family size-3: two
aromatic carbons connected to an aromatic oxygen (a furan-like three-node motif),
with no fine atom attributes.

**Rule.** `S471`: `x_C[157] >= 0.223606795`; selected in the first 16 rules and in
R64/R128. 0.2236 = 1/sqrt(20): because of the size scaling, this threshold is
**not** a raw count of at least two occurrences; on Train the predicate is
activated by molecules with raw counts 1–2 (95.5 % of activations have count 1).
Raw-unit coefficient in R128: **+2.0553** (standardised +0.4733).

**Data support (molecules / scaffolds / positives / positive rate).**
Train 1849 / 704 / 98 / 0.0530 (inactive 0.0365); Valid 266 / 266 / 8 / 0.0301
(inactive 0.0190); Test 225 / 225 / 7 / 0.0311 (inactive 0.0316).
The active-group rate is higher than the inactive rate on Train and Valid, but
**equal on Test** — a clear limitation.

**Prediction evidence.** R128 uses the rule with a positive coefficient; switching
it off lowers the logit by exactly its contribution for every activated molecule
(verified to 1e−15). This card's logit effect is therefore faithful to the model,
but the Test support does not reproduce the Train association.

**Real cases.** Supporting (Train label 1): graph 434, 52 atoms, SMILES
`CC1OC(OCC2OC(Oc3c(-c4ccc(O)cc4)oc4cc(OC5OC(C)C(O)C(O)C5O)cc(O)c4c3=O)C(O)C(O)C2O)C(O)C(O)C1O`,
`x_C[157]=0.416` (about 9 instances). Counterexample (Train label 0): graph 69,
16 atoms, `Cc1cccc2c(=O)c3ccccc3oc12`, `x_C[157]=1.0` (16 instances).

**Competitions.** The same motif is a proxy for aromatic heterocycles and sugar-like
substituents; larger molecules can reach the threshold with fewer instances than
small ones. The R128 positive coefficient may partly represent the molecule's other
activated rules (95 COARSE rules in R128).

**Status.** Model-computational rule with a reproducible Train/Valid association;
the Test association is not reproduced. Not a chemical mechanism.

---

## Card 2 — readable presence threshold (FULL 1089)

**Structure object.** `FULL` column 1089; canonical key
`N[5,0,3,5,0,0,1,0,0;5,0,4,5,2,0,2,0,0;6,0,2,5,0,0,1,0,0]E[0-1:0,0,0;0-2:1,0,1]`;
decoded `a:C.sp2.deg3`, `b:C.sp3.H2.deg4`, `c:N.sp2.deg2`, bonds `a-b:SINGLE`,
`a-c:DOUBLE`; an sp2-carbon–sp3-CH2–sp2-N three-node chain (an imine/amide-like
local topology).

**Rule.** `S125`: `x_F[1089] >= 0.142857149`. This is one of the 53/512 rules for
which the scaled threshold is logically equivalent to presence: `1/sqrt(7)` with
raw count 1; on its `< t` side all molecules have raw count 0, and on its `>= t`
side all have raw count ≥1. Raw-unit coefficient in R128: **−4.7894**
(standardised −0.4693). Not in R64/R16, so it was added in a later selection batch.

**Data support.** Train 319 / 83 scaffolds / **0 positives** / rate 0.0000
(inactive 0.0378); Valid 61 / 61 / 1 / 0.0164; Test only 5 / 5 / 0 / 0.0000.
The Train "all-negative" association is not supported on Valid and has almost no
Test coverage.

**Prediction evidence.** Negative coefficient in R128; the logit change on switch-off
equals the coefficient. The tiny Test support prevents treating this as a
validated predictive rule.

**Real cases.** Supporting (label 0, Train): graph 746, 20 atoms,
`CCOC(=O)CC(C)=NNS(=O)(=O)c1ccc(N)cc1`, count 1. Counterexample (label 1):
graph 14423, 40 atoms, `Cc1ccc(C)c(NC(=O)CCC(=NNC(=O)c2c(O)ccc3ccccc23)C(C)C(=O)c2ccc(O)cc2)c1`,
`x=0.158` (count 1) — a genuine label-1 molecule satisfying the Train "all-negative"
condition, located in Valid.

**Competitions.** The Train-zero-positive pattern may reflect dataset composition:
this topology appears mostly in non-active hydrazone/sulfonamide-like molecules.
Molecule size matters: count 1 at 40 atoms produces the same predicate as count 1
at 7 atoms.

**Status.** A clearly expressible rule with a *weakened* association (Valid
counterexample) and very low Test support; a hypothesis, not a validated rule.

---

## Card 3 — aromatic carbon–sulfur family (COARSE 419) and its FULL children

**Structure object.** `COARSE` 419: key `N[5,5,1;5,5,1;5,5,1;15,5,0]E[0-2:3;1-2:3;2-3:0]`;
`a:Carx3 b:Sx1`, bonds `a-a:AROMATICx2 a-b:SINGLEx1`: an aromatic carbon bonded
to sulfur in a three-aromatic-carbon environment. Its four priority FULL children
are all genuine sub-instances (`overlap=1.0` with COARSE 419):

| FULL column | decoded sulfur environment | Train +/n | Valid +/n | Test +/n | B2 coefficient |
|---|---|---:|---:|---:|---:|
| 2078 | S sp3 deg4 H0, in a c–c–S–c topology | 197/529 (37.2 %) | 10/32 (31.3 %) | 28/52 (53.8 %) | +0.159 |
| 2525 | S sp3 deg2 H0 (thioether-like) | 38/240 (15.8 %) | 1/14 (7.1 %) | 1/20 (5.0 %) | +0.087 |
| 1788 | S sp3 deg4 H0, different aromatic substitution | 117/1519 (7.7 %) | 9/199 (4.5 %) | 36/204 (17.6 %) | −0.084 |
| 2275 | S sp3 deg4 H0, **in-ring** | 11/356 (3.1 %) | 0/54 (0 %) | 5/58 (8.6 %) | −0.009 |
| COARSE 419 | sulfur attributes collapsed | 332/3213 (10.3 %) | 15/392 (3.8 %) | 50/415 (12.0 %) | +0.245 |

**Rule.** `S271`: `x_C[419] >= 0.234082296` (Train 1584 active / 618 scaffolds).
R128 raw coefficient −0.1824 (standardised −0.039), near zero: the model uses the coarse presence
but assigns a small negative logit contribution at this threshold. **None of the four FULL
children entered the 512-rule candidate pool.** So the fine differences were not
used by the rule models, even though B2's frozen coefficients distinguish them.

**Local environment (real instances).** FULL 2078 (graph 3): aromatic C/S nodes with
six external bonds, including an S–C single bond and two S=O double bonds
(sulfonyl/aryl-sulfone context). FULL 2525 (graph 8): four external bonds, including
an S–S single bond. FULL 1788 (graph 67): five external bonds including S–N and two
S=O bonds. FULL 2275 (graph 2): sulfur in a ring with S–N and aromatic bonds
(phenothiazine-like). These are real attribute/topology differences, not merely
different labels for one identical motif.

**Data support and competing explanations.** All four children are subsets of
COARSE 419 but overlap each other little (pairwise Jaccard ≤0.069). Molecule-size
stratification on Train shows the positive association is partly size-driven:
for 2078, rate 0.318 at ≤20 atoms vs 0.726 at >50 atoms (whole-dataset base 0.024
and 0.172 respectively). The same molecule/scaffold can carry several instances
(e.g. graph 883 carries multiple COARSE-419 instances); no duplication artefact
was found at the molecule level. Valid/Test rates move substantially, e.g. 1788
4.5 % → 17.6 % while 2525 7.1 % → 5.0 %.

**Prediction evidence.** B2's frozen coefficients give opposite signs to 2078/2525
and 1788; the R128 coarse rule contributes only about −0.18 logit units when active. This weakens
the claim that the current rule model captured the fine sulfur distinction.

**Real cases.** Supporting for 2078: graph with `S(=O)(=O)O` aryl sulfone; label 1
counterexample from C–S support: graph 11, `O=C(O)Cc1ccc(SSc2ccc(CC(=O)O)cc2)cc1`
(label 1, COARSE 419 active). Counterexample: graph 2,
`CC(=O)N1c2ccccc2Sc2c1ccc1ccccc21` (label 0, FULL 2275 in-ring sulfur, COARSE 419
active). A label-0/1 pair with the same coarse predicate exists, so coarse presence
alone is not sufficient.

**Status.** A real, readable structural family with reproducible Train association
for COARSE 419; the child-specific label differences are a **data association /
hypothesis** with unstable Valid/Test magnitudes and no support in the fitted rule
model. No sulfur-causality claim.

---

## Card 4 — weak rule with an explicit counterexample (COARSE 410)

**Structure object.** `COARSE` 410: key
`N[5,5,0;5,5,1;5,5,1;6,5,1]E[0-1:0;1-2:3;1-3:3]`; decoded
`a:Cx1 b:Carx2 c:Narx1`, bonds `a-b:SINGLEx1 b-b:AROMATICx1 b-c:AROMATICx1`;
an aromatic N-heterocycle with an aliphatic substituent (size-4).

**Rule.** `S267`: `x_C[410] >= 0.2` (1/sqrt(25)); raw-unit coefficient in R128
**−0.0550** (standardised −0.0146). Not in R64/R16: a weak late addition.

**Data support.** Train 2513 / 1154 scaffolds / 86 positives / 0.0342
(inactive 0.0377 — slightly *negative* association); Valid 370 / 370 / 9 / 0.0243
(inactive 0.0192 — direction reversed); Test 306 / 306 / 4 / 0.0131 (inactive 0.0331).

**Prediction evidence.** Small negative R128 coefficient; switch-off changes the logit
by exactly the term. This is an explicit case where a Train-selected rule has no
consistent Valid/Test association: it should not be interpreted as a discovered
"regularity".

**Real cases.** Supporting (label 0, Train): graph 153, 17 atoms,
`CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1`, count 5. Counterexample (label 1, Train):
graph 818, 46 atoms, an iron-porphyrin-like structure
`CC(=O)C1=C(C)C2=Cc3c(C(C)=O)c(C)c4n3[Fe-3]35(Cl)n6c(c(C)c(CCC(=O)O)c6=CC6=[N+]3C(=C4)C(C)=C6CCC(=O)O)=CC1=[N+]25`.

**Competitions.** The rule is probably a proxy for heteroaromatic scaffolds and
molecule size; its activator set contains both inactive drug-like pyrimidines and
active metallo-complexes.

**Status.** A model-computational term with an observable Train/Valid/Test reversal;
kept as a negative/weak example, not promoted to an accumulated regularity.

---

## Joint-rule note (descriptive, not a fifth knowledge card)

The 16 selected real co-path pairs in `B2-U+P` are whole-molecule co-occurrences.
For example `P121 = (x_C[1] >= 1.1339) AND (x_C[495] >= 0.3780)` has Train cells
(11/10/01/00) = 71/9899/740/22191 molecules, with positive rates
0.042/0.037/0.085/0.036; Valid/Test `11` cells have 0/13 and 0/16 positives.
A stronger Train interaction (`P025`: cell `01` rate 0.245 vs 0.034 baseline)
does not carry over (Valid 8/80, Test 19/94). No instance-position check was used to
call any of these an interaction; the arm's Test performance is below the U64
baseline. See `analysis/phaseD_cases.json` for the full four-cell tables of the top
three pairs.
