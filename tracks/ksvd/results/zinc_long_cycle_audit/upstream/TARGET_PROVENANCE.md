# ZINC target provenance (verified during Long-Cycle Audit)

## VERIFIED chain (code + data re-checked)

| step | VERIFIED fact | evidence |
|---|---|---|
| 1. source molecules | PyG `ZINC(subset=True)` official 10k/1k/1k subset of the **ZINC 250k** molecule set (249,456 drug-like molecules from the ZINC database): raw pools `220,011 + 24,445 + 5,000 = 249,456` | `data/ZINC/raw/*.pickle` sizes == `EXPECTED_POOL_SIZES`; matches the GVAE ``250k_rndm_zinc_drugs_clean.smi`` line count (249,456) |
| 2. dataset loader | PyG `ZINC.process()` reads `mols[idx]['logP_SA_cycle_normalized']` verbatim as `y`; trained on graph (atom_type, bond_type) only | `torch_geometric/datasets/zinc.py` sha256 recorded; string `logP_SA_cycle_normalized` present in source |
| 3. upstream pickle origin | `molecules.zip` (Dropbox `feo9qle74kg48gy`) created for graphdeeplearning/benchmarking-gnns; their `prepare_molecules.ipynb` only re-packages it (no property computation in that repo) | `prepare_molecules.ipynb` (fetched); repo tree search: no computation script, only `.index` split files |
| 4. property formula | **Kusner et al. 2017 GVAE** ``generate_latent_features_and_targets.py``: `logP = Descriptors.MolLogP(mol)`, `SA = -sascorer.calculateScore(mol)`, `cycle = -max(0, max(len(c) for c in nx.cycle_basis(nx.Graph(GetAdjacencyMatrix(mol)))) - 6)` (0 if no cycle / <= 6) | verbatim excerpt saved at `upstream/generate_latent_features_and_targets.py` (sha256 recorded) |
| 5. normalization | `y = (logP-mean)/std + (SA-mean)/std + (cycle-mean)/std` with **dataset** means/stds computed over all 249,456 (``np.mean``/``np.std``, ddof=0) | verbatim excerpt; stage `verify` re-derives and compares |
| 6. our loader | `zinc_long_range_proxy._load_zinc` -> PyG `ZINC` class; `zinc-context-gap` protocol uses official split indexes | `tracks/ksvd/experiments/luyin16/zinc_long_range_proxy.py:119` |

## ASSUMED / COMMUNITY cross-reference (NOT yet verified by the data itself)

* The constants `logP_mean=2.4570953396190123, logP_std=1.4498737504491803`,
  `SA_mean=-3.0536628609296634, SA_std=0.8315096218718826`,
  `cycle_mean=-0.048569687328769765, cycle_std=0.2859726330705808` are the
  JTNN-era community snippet constants.  Stage `verify` recomputes dataset
  stats over the full 249,456 and compares.
* The GVAE `cycle` statistic uses `nx.cycle_basis` (a *cycle basis*, not the
  longest simple cycle) on the graph from `GetAdjacencyMatrix(mol)` in
  **RDKit canonical SMILES atom order**.  networkx's `cycle_basis` output
  depends on node iteration order (Paton 1969 spanning-tree algorithm), so
  the label's cycle term is node-order dependent (see stage `perm`).

## Open item (tested in stage verify)

Whether the graph pickles in `molecules.zip` are exactly the GVAE SMILES
molecules (identity correspondence via canonical graph isomorphism), and
whether the stored `logP_SA_cycle_normalized` matches the GVAE formula to
float precision.
