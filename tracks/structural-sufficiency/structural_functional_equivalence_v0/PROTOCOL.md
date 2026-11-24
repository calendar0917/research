# SFEC v0 — Structural Functional Equivalence and Composition-Rule Discovery

Round: `structural_functional_equivalence_v0`
Track: `tracks/structural-sufficiency/`
Status of this file: **frozen before any label analysis**. It records the graph
decomposition, exact identity, context split, support gates, estimators,
statistics, and judgement rules. It is not changed after seeing label results.

No neural network, teacher, distillation, pretrained model, or model prediction
is used. Only raw graph structure and the dataset's raw task labels are used.
ZINC is a fast structural screen; PCQM4Mv2 is the larger-scale check. Lipo
results are cited from the historical report and are not re-run.

## 0. Fixed inputs

| item | choice |
|---|---|
| ZINC | official PyG `ZINC(root=data/ZINC, subset=True)`; splits train 10000 / valid 1000 / test 1000; raw node type `data.x[:,0]`, raw bond type `data.edge_attr`; raw `y` |
| PCQM4Mv2 | official `data/ogb/pcqm4m-v2`, `split_dict.pt`, `raw/data.csv.gz`; official Train and Valid only; raw HOMO-LUMO gap in eV; Test never read for labels |
| PCQM row plan | historical seed-42 permutation `pcqm_train1m/gids.npy`; first 100,000 rows first, then first 1,000,000 if structural support is insufficient; full Train only if support is still insufficient **and** measured throughput projects completion within budget |
| PCQM cache reuse | `lcrc_pcqm4mv2_transfer_pilot_v1/cache/pcqm_train1m` (SMILES, gids) and `cache/pcqm_valid`; raw labels re-read from `raw/data.csv.gz` |
| identity reuse | bridge-finding and exact coloured-graph certificate implementations are re-used from `response_dictionary_small_s0_v1/code/rd_fragment.py` for ZINC; PCQM uses RDKit canonical isomeric fragment SMILES with an explicit attachment dummy |

`labels` are attached to the structural tables only after the structural universe,
the 64-type vocabulary, the context split, and all support statistics have been
written to disk. No label is used to choose a cut, a fragment, a vocabulary, a
context split, a support threshold, or the scale extension.

## 1. Graph decomposition

For every original **connected** graph `G`, enumerate every bridge bond of the
simple undirected atom graph. For each bridge `e=(u,v)`:

* cut `e`; obtain the two connected components `S` and `C` (the full complement,
  including any other original components; multi-component molecules are
  excluded and counted).
* choose the smaller side as `S`; if the two sides have equal atom count, order
  the two oriented rooted fragments by their canonical key and take the
  lexicographically smaller key as `S`. This tie rule is deterministic and
  label-free.
* require `1 <= |S| <= 8`, `|C| >= 6`, and `|C| >= 2*|S|`.
* keep the cut bond type and both attachment roots.

ZINC: only bond code `1` (single) is cut; the raw bond-code convention is
recorded in `data_identity.json`. If the single-bond convention cannot be
verified, all bridges are cut and stratified by raw bond code.

PCQM4Mv2: only acyclic single bonds of the original RDKit molecule are cut.
All bridges are acyclic by definition; only `Chem.BondType.SINGLE` is retained.
No artificial molecule is generated and no absent combination receives a label.

## 2. Exact fragment identity

ZINC: exact coloured-incidence-graph certificate (pynauty) over raw atom type,
raw bond type, and a binary attachment-root colour. Two keys are equal iff the
rooted coloured graphs are isomorphic. No WL hash is used as identity.

PCQM4Mv2: the fragment is produced by keeping one side plus the cut neighbour,
turning the cut neighbour into a dummy atom in its original bond slot, and
serializing with `Chem.MolToSmiles(isomericSmiles=True, canonical=True)`. The
key is the exact canonical string. It retains formal charge, aromaticity,
isotopes, tetrahedral parity, and double-bond stereo encoded on the attachment
dummy bond. Keys are never replaced by a WL hash. `Molecule_key` is the
canonical isomeric SMILES of the whole original molecule.

Reconstruction gate: for every cut, the atom/bond counts and the cut bond type
must satisfy `|S|+|C|=n_atoms`, `|E_S|+|E_C|+1=n_bonds`, and both fragments must
be connected. A deterministic sample (all ZINC cuts; for PCQM all cuts in a
fixed 1% row sample plus every cut of the first 2,000 rows) is fully reassembled
and checked. The check is graph-exact; stereo at the cut root is additionally
checked by fragment-level stereo-aware substructure matching. Any failure aborts
the build.

Observed record per cut: `dataset, split, row_id, molecule_key, cut_id,
bond_type, C_key, S_key`. `molecule_id` is the exact whole-molecule identity;
duplicate rows with the same `molecule_key` are one molecule for independence
and are never counted as independent samples. Duplicate `(C,S,b)` observations
are de-duplicated to one cell; the cell keeps the number of distinct molecules
and the original row IDs, and reports the label spread inside the cell.

## 3. Support matrix and vocabulary

`Omega_b = {(C,S): observed in dataset for bond type b}`. A sparse bipartite
graph and a reverse index are built; no dense matrix is formed.

For each bond type, `S` types are ordered by Train **molecule document
frequency** (number of distinct Train molecules), then by canonical key. The
first 64 enter the primary crossed analysis. Only structural frequency is used.
Coverage before and after the cut is reported.

Reported structural quantities (full universe and primary 64-type universe):
cuts, usable molecules, distinct `S`/`C` types, cuts per molecule, S and C
degree distributions, fraction of contexts with exactly one combination,
connected components, largest component, 2-core size, and cycle rank
`m - n + k` (m unique cells, n bipartite vertices, k components).

For every primary pair `(A,B)` within a bond type, `k_AB` is the number of
distinct contexts containing both. The number of complete rectangles is
`sum_{A<B} C(k_AB,2) = sum_C C(k_C,2)`, computed without enumerating rectangles.
Counts of pairs with `k_AB >= 2,5,10,20` are reported. Shared-molecule rectangles
are not treated as independent samples.

## 4. Context split and transfer sample

Official Train/Valid/Test labels keep their official meanings. Inside official
Train only:

* `h(C) = SHA256("0|" + C_key)`, first 16 hex digits as an integer, `% 100`;
* `h(C) < 80` -> discovery context; `h(C) >= 80` -> evaluation context;
* salt is fixed at 0.

Every molecule identity that appears in **any** evaluation-context Train cut is
removed from discovery entirely (all its cuts). The full evaluation set is kept
for description. The primary evaluation set is selected greedily in ascending
`h(C)` order, accepting a context only if its full molecule-identity set is
disjoint from all already accepted contexts. Official Valid/Test contexts are
reported separately and never enter discovery. A Valid/Test cut whose complete
original graph duplicates a Train graph is excluded from the main transfer
statistics and reported.

Support gates for P1/P3 (practical screen, not a theorem):
at least 10 structure pairs with >=10 discovery contexts and >=5 primary
evaluation contexts, >=50 mutually molecule-disjoint primary evaluation
contexts, and >=100 primary evaluation pair observations. Otherwise the stage
is labelled `INSUFFICIENT_CROSSED_SUPPORT` and only structural/descriptive
results are reported.

P2 has its own gate in Section 8. Support statistics, the vocabulary, the
context split, and the configuration are written before label analysis.

## 5. P1 estimator

For canonical-ordered pair `(A,B)` and context `C`:

`delta_AB(C) = y(C,B) - y(C,A)`, where `y(C,S)` is the mean raw label over the
distinct molecules in cell `(C,S,b)`.

Discovery: `mu_AB = median_C delta_AB(C)` over discovery contexts containing
both. `D_AB(C1,C2)=delta_AB(C2)-delta_AB(C1)`.

Reported: delta median, MAD, quantiles, direction consistency, `|D|`
distribution, discovery-vs-evaluation response distribution. For pairs with
very many context pairs, at most 256 unordered context pairs are drawn with
seed 0; the full rectangle count is still reported and sampled pairs are not
counted as independent observations.

Evaluation on the primary evaluation set: ZERO (`0`) vs PAIR (`mu_AB`).
Report delta MAE, RMSE, sign accuracy, and skill score `1 - MAE/MAE_ZERO`.
Aggregate with context-equal weights and pair-equal weights separately.

Functional-equivalence candidates, frozen on discovery before evaluation:
`P80_C(|delta_AB(C)-mu_AB|) <= tau` and at least 10 discovery contexts, with
`tau=0.10` raw-y unit for both datasets (independent units, no cross-task
averaging); sensitivities `tau=0.05, 0.20`. Evaluation checks the fraction
inside `tau`, discovery/evaluation residual distributions, and `|mu_AB|`.
Near-zero fixed effects and clearly non-zero fixed effects are reported
separately. Approximate transitivity is never assumed: if `A~B` and `B~D`,
`A~D` is only reported if separately verified.

P1 judgement: `REUSABLE_RESPONSE_SIGNAL` if support is sufficient, PAIR beats
ZERO on evaluation, and discovery candidates retain small evaluation residuals;
`PARTIAL_RESPONSE_REUSE` if direction transfers but stable candidate coverage is
limited; otherwise `NO_SUPPORTED_RESPONSE_REUSE`.

## 6. SHARED composition model (P2)

For discovery contexts with at least two primary `S` types, fit

`F(C,S) = a_C + b_S`

by context-centred least squares, each context weighted equally:

`y~(C,S) = y(C,S) - mean_{S' in C} y(C,S')`
`b~(C,S) = b_S - mean_{S' in C} b_S'`
`min_b sum_C mean_{S in C} (y~(C,S) - b~(C,S))^2`

FP64 normal equations are assembled as sparse/dense Laplacians over the
<=64 primary S types per bond type. Each connected component receives the gauge
`sum_S b_S = 0`. This gauge is not interpreted as an absolute structural
contribution. Only differences inside one supported connected component are
used. Prediction: `delta_hat_AB = b_B - b_A`.

Natural unobserved replacements: pairs `(A,B)` that co-occur in at least one
evaluation context but never in a discovery context, both appear in discovery,
are connected through other observed structure relations, and lie in the same
supported component. No direct replacement label is used for prediction.
Report count, contexts, molecules, path length, error, and response magnitude.

Controlled masking (if natural pairs are scarce, and as an extra test): up to 20
pairs selected only by discovery support (descending), then evaluation support
(descending), then canonical keys (ascending), among pairs that have direct
discovery co-occurrence and remain indirectly connected after deleting the
direct co-occurrence. For each pair, delete the `B` cell in every discovery
context containing both `A` and `B`; keep `B` elsewhere and `A` elsewhere.
Verify no direct `A,B` discovery context remains. If `A,B` are then disconnected,
record the pair as inapplicable. Refit SHARED per masked pair. Evaluate on
primary evaluation contexts that were never used in discovery.

P2 gate: at least 10 masked pairs passing the structural support check, each with
at least 5 primary evaluation contexts. Comparisons: ZERO, SHARED, and
PAIR-FULL (unmasked direct discovery response; a reference with direct
supervision, not a fair unseen-replacement baseline). Report MAE improvement of
SHARED over ZERO and the remaining gap to PAIR-FULL. A triangle difference
inside one context is not a discovery.

P2 judgement: `COMPOSITION_TRANSFER_SIGNAL` if support is sufficient, SHARED
improves evaluation MAE over ZERO by at least 15%, and the paired interval
supports a positive gain; `COMPOSITION_SCREEN_ONLY` if the point estimate is
positive but uncertain; otherwise `NO_SUPPORTED_COMPOSITION_TRANSFER`. The 15%
is a practical screen, not a theoretical threshold.

## 7. LOCAL context rule (P3)

Context descriptor `q(C)`, fixed and readable:

* PCQM: attachment-root atomic number, formal charge, aromatic flag, and the
  sorted multiset of one-hop neighbour `(atomic number, formal charge, aromatic
  flag, bond type)` inside `C`; the cut edge is excluded.
* ZINC: root raw node type and sorted multiset of `(raw node type, raw bond
  code)` inside `C`; the cut edge is excluded.

No second hop, global fingerprint, embedding, or other descriptor is added.
`q` is one value per context (the context key already fixes the root).

On top of SHARED, fit for each supported `q`:

`delta_hat_AB(C) = b_B - b_A + d_(B,q(C)) - d_(A,q(C))`

with context-centred least squares, each context equal weight, and fixed ridge
`lambda=5 * sum_S d_(S,q)^2`. A `q` group needs at least 10 discovery contexts.
The correction is enabled for `(A,B)` only when both are in the same supported
connected component of that `q`'s observation bipartite graph; otherwise fall
back to SHARED. Model name: LOCAL.

Matched NULL: within each `(bond type, context row degree k_C)` stratum, randomly
permute the `q` assignments with seed 0, preserving the number of each `q` per
stratum. Fit LOCAL_NULL with the same estimator. Evaluation uses the original
true `q`. If too few contexts are permutable, mark NULL support insufficient.

Compare SHARED, LOCAL, LOCAL_NULL on the same evaluation observations. Report
overall delta MAE, delta MAE on high-non-additivity pairs, correction coverage,
correction magnitude, unseen-`q` fallback rate, and per-`q` support. The P2
masked pairs are also scored with condition corrections fitted after masking
(never restoring masked labels).

P3 judgement: `TRANSFERABLE_CONTEXT_RULE` only if LOCAL beats both SHARED and
LOCAL_NULL, evaluation delta MAE improves by at least 10%, true-condition
coverage is sufficient (>=50% of primary evaluation pair observations have both
corrections enabled and >=10 supported `q` groups), and the paired interval
supports a positive gain. Otherwise `CONTEXT_SUPPORT_INSUFFICIENT`,
`CONTEXT_RULE_SCREEN_ONLY`, or `NO_SUPPORTED_CONTEXT_RULE`, with the reason
recorded.

## 8. Statistics

All delta metrics use raw label units. Primary metric: delta MAE; RMSE and sign
accuracy are secondary. 95% intervals are 2000-replicate context-level paired
bootstrap percentiles (seed 0), resampling whole evaluation contexts. For PCQM,
the number of context scaffolds and a scaffold-cluster bootstrap sensitivity are
also reported. Intervals describe sample uncertainty under the fixed data and
analysis, not a population theorem.

## 9. Engineering positive controls

Small FP64 synthetic designs, error target <=1e-8:

A. additive `y(C,S)=a_C+b_S`: SHARED recovers all identifiable `b` differences;
   every complete-rectangle double difference is zero.
B. crossed `y(C,S)=a_C+b_S+d_(S,q(C))`: LOCAL improves SHARED on a supported
   noiseless design.
C. acyclic observation bipartite graph: free row/column offsets fit any observed
   label perfectly, so perfect training fit does not prove additivity.
D. unobserved replacement: after deleting direct `A,B` co-occurrence but keeping
   an indirect path, SHARED recovers the known additive difference.
E. unidentifiable control: `A` and `B` in different components yield no
   meaningful shared `b` difference.

Synthetic data are used only for engineering and mathematical controls.

## 10. Budget and stop rules

CPU only. Target 30-60 min wall time, hard ceiling 2 h. At most 24 CPU workers
on `res`; local runs use at most 8 workers and 1 BLAS thread per worker. Total
process RSS ceiling 64 GB (local machine is much smaller; the run is chunked and
never loads all cells as Python objects).

Stage order: structural build -> support/vocabulary/split -> checks -> P1 ->
P2 -> P3 -> report. A failed gate is recorded and later stages may still run
descriptively. If the budget is exhausted, completed stages and artifacts are
saved and the limitation is reported. No neural training, no GPU experiment, no
redesign of the functional vocabulary after seeing results.

## 11. Required outputs

`PROTOCOL.md`, fixed config and code, `data_identity.json`,
`structural_support.json`, `observed_cells.parquet` (or sparse equivalent),
`context_split.json`, `checks.json`, `rectangle_statistics.csv`,
`response_transfer.csv`, `functional_equivalence_candidates.csv`,
`unseen_composition_results.csv`, `context_rule_results.csv`, `summary.json`,
`REPORT.md` (Chinese, first page answers the eight required questions).