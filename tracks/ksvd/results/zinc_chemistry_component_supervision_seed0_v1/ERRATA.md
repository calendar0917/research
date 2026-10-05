# ERRORTA — `zinc_chemistry_component_supervision_seed0_v1`

## Resolved (pre-formal-run)

* CUDA device-placement: the two component target tensors were initially created on CPU and
  compared to the GPU forward inside the loss; fixed by placing `target_ell`/`target_s` on `device`
  and indexing them with a device-local index tensor. Smoke + full runs completed after the fix.
  The pre-fix attempts (Slurm 56007/56008) failed before producing any dev score.

## Evidence scope

* Sources are read-only: the frozen fresh-fold artifacts under
  `tracks/ksvd/results/zinc_local_tuple_fresh_fold_replication_seed0_v1/` and the train-only raw
  label reconstruction of the fresh-fold runner. No `formula_verification_per_molecule.csv`
  (mixed 12k) was opened.
* The original g-only model is used only as an init-identity reference; its trained state is not
  loaded. `zinc_direct_bond_relation_seed0_v1` is referenced only to close that line.
