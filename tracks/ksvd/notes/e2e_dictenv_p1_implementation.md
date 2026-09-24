# E2E-DictEnv-P1 — implementation

Round **E2E-DictEnv-P1** · protocol `e2e_dictenv_p1` · study `zinc-context-gap`.
Pre-registration: [`e2e_dictenv_p1_preregistration.md`](e2e_dictenv_p1_preregistration.md).
Prior-artifact audit: [`e2e_dictenv_p1_prior_artifact_audit.md`](e2e_dictenv_p1_prior_artifact_audit.md).

## 1. Code layout

| file | role |
|---|---|
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_p1.py` | frozen P1 model + parameter accounting + primitive anchor + shuffle helpers + collate |
| `tracks/ksvd/experiments/luyin16/zinc_e2e_dictenv_p1.py` | runner: env cache, correctness G0–G14, smoke, training, health, mechanism, gates, freeze, analyze |
| `tracks/ksvd/tests/test_e2e_dictenv_p1.py` | focused CPU tests (no ZINC, no GPU) |

The additive `bond_u` / `bond_v` outputs were added to
`e2e_dictenv_v0.env_incidence` (P1-only consumers; v0/T1 ignore them and their
tests still pass).

## 2. Model wiring

```text
phi_v (65) --IHT(D, s=8, 10 steps)--> alpha_v (32)          [D trainable, K-SVD init]
node branch: (alpha_v W_A^S) o (q_v W_A^C) / sqrt(96)  -> per-shell SUM -> A_i   (288)
edge branch: g_uv = [a_u+a_v; |a_u-a_v|; a_u o a_v] (96)
             (g_uv W_E^S) o (b_uv W_E^C) / sqrt(48)     -> per-shellpair SUM -> E_i (288)
anchor:      [root one-hot(28); patch atom mass(28); patch bond mass(4); size(2)] -> p_i (62)
z_i = [p_i; A_i; E_i]  (638) --SiLU MLP 638->102->48--> E_i (48)
static read-only composer: u=W_P E, 15-D pure-topology relation, 64->64->16 pair encoder,
global 62->32->32, topology 25->16->8, GenericReader(302,(13,13))
```

Arms differ only in the coding operator (`Sparse`: tied IHT; `DenseTied`:
`phi @ Dbar`); initialization is bit-identical.

## 3. Primitive anchor / cache

`build_env_cache` rebuilds, from raw ZINC primitives, per-split:

```text
phi (65), atom category, occ_node/occ_root/occ_shell,
bond_root/bond_shellpair/bond_type, bond_u/bond_v,
anchor (standardized 62-D)
```

The 62-D raw anchor is fit with an official-train-only per-coordinate
`(mean, scale)` (`anchor_stats.json`), then standardized for both splits.  No
`patch_cont`, no `scalars` and no `atom_shell`/`bond_shell` are stored or read.

## 4. Local results (CPU)

* `pytest tracks/ksvd/tests/test_e2e_dictenv_p1.py`: **11 passed**
  (parameter accounting == 97,865; init matching; sparsity; coord-zero purity;
  edge symmetry; shuffle multiset preservation; relation slice; environment
  freeze; dictionary gradient; anchor semantics; AST source cleanliness).
* combined with the v0/T1 suites: **41 passed**.
* Local `correct` stage: **all 15 gates G0–G14 PASS** on the full local ZINC
  cache (valid `G0` 3/3 bit-identical; `G1` file sha `925d573a...` and column
  norms 1.0; `G12` relabel max pred diff `2.4e-7`; `G13` actual == accounted ==
  97865, in budget).
* Local env cache: train 10,000 mols / 231,664 nodes / 1,418,500 occurrences /
  1,232,844 bond occurrences; valid 1,000 / 23,083 / 141,287 / 122,934
  (identical counts to the v0 cache, as expected).

## 5. Remote execution regime

Formal runs use the repository's canonical A100 protocol (Adam lr 1e-3, wd
1e-5, batch 128, clip 5, 240 epochs, no scheduler, fixed Top-5 soup), λ fixed at
`33.95873017865987` for both arms.  Official test is never loaded before the
terminal unlock stage.  Formal run commit and per-run wall clock / peak memory
are recorded in `results/e2e_dictenv_p1/`.

## 6. Reproduce

```bash
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_p1.py
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_p1 identity
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_p1 env
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_p1 correct
# formal (GPU), resumable:
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_p1 all --device cuda
```

Stages are resumable: `env` reuses existing caches, `train` skips an existing
run JSON, and `run_all` stops at the first failed gate.
