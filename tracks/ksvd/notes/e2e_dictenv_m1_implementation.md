# E2E-DictEnv-M1 — implementation

Round **E2E-DictEnv-M1** · protocol `e2e_dictenv_m1` · study `zinc-context-gap`
(Workstream M).
Pre-registration: [`e2e_dictenv_m1_preregistration.md`](e2e_dictenv_m1_preregistration.md).

## 1. Code layout

| file | role |
|---|---|
| `tracks/ksvd/experiments/luyin16/e2e_dictenv_m1.py` | M1 model: OGB chemistry embeddings, 62-D primitive anchor, node/edge dictionary binding, 638->102->48 decoder, pure-topology 15-D pair relation, self-contained batch collate |
| `tracks/ksvd/experiments/luyin16/molhiv_e2e_dictenv_m1.py` | runner: dictionary fit, env cache, anchor stats, lambda calibration, correctness G0–G14, smoke, train, freeze, analyze, terminal test |
| `tracks/ksvd/tests/test_e2e_dictenv_m1.py` | focused CPU tests |

## 2. Chemistry schema

The categorical schema is read at runtime from
`ogb.utils.features.get_atom_feature_dims()` / `get_bond_feature_dims()` and
recorded in `results/e2e_dictenv_m1/schema.json`.  No category counts are
hand-written.  Each atom field has its own `Embedding(cardinality, 24)`; the
field embeddings are summed into `q_v^chem in R^24`.  Each bond field has its own
`Embedding(cardinality, 12)`, summed into `b_e^chem in R^12`.

## 3. Model wiring

```text
phi_v^65 --IHT_{s=8}(Dbar), 10 steps--> alpha_v^32      [D trainable, MolHIV K-SVD init]
node:  (alpha_v W_A^S) o (q_v^chem W_A^C)/sqrt(96)  -> per-shell SUM -> A_i (288)
edge:  g_uv=[a_u+a_v; |a_u-a_v|; a_u o a_v] (96)
       (g_uv W_E^S) o (b_e^chem W_E^C)/sqrt(48)     -> per-shellpair SUM -> E_i (288)
anchor p_i = [q_i^chem(24); sum_P q_v^chem(24); sum_E b_e^chem(12); log sizes(2)] (62)
z_i = [p_i; A_i; E_i] (638) --SiLU MLP 638->102->48--> E_i (48)
pair:  u=W_P E, 15-D topology-only relation, 64->64->16 pair encoder, Embedding(5,16) gate
global = [molhiv short(15)+long(15) topology; mean atom chem(24); mean bond chem(12)] = 66 -> 32 -> 32
topology branch = 25-D zinc_topology_features hinge -> 16 -> 8
reader = GenericReader(302, (13,13)) -> 1 HIV logit
```

Parameter audit: **102,325 total** (local 87,094 = chemistry 4,332 + dictionary
2,080 + node binding 5,376 + edge binding 5,184 + decoder 70,122; backend
15,231), inside the pre-registered 80,000–130,000 budget.

## 4. Pure-topology pair relation

The 15-D relation is `[one_hot(bucket,5); log1p(distance); overlap(5);
boundary(3); log1p(path_count)]`.  P1's 5 distance buckets are reused;
distance > 5 and disconnected salt pairs are clipped into the top (`5+`)
bucket.  The clipping frequency is recorded per split in `env_cache.json`.
Bond chemistry never enters the relation.

## 5. Dictionary / anchor / lambda

* Dictionary: `sdb_v0.fit_ksvd(X, atoms=32, s=8, epochs=10, seed=20260924)` on
  MolHIV official-train `phi65` only.  If train nodes > 500,000 a deterministic
  500,000-node sample (seed 20260924) is used.  Provenance and `D` sha256 in
  `dictionary_fit.json`.
* Anchor scaler: train-only per-coordinate `(mean, scale)` fitted at
  initialization (`anchor_stats.json`).
* Reconstruction weight: calibrated on the first 512 official-train graphs with
  the initial model; `lambda_M = 0.25 * L_task^init / L_rec^init`
  (`lambda_calibration.json`).

## 6. Training protocol

seed 0/1; Adam lr 1e-3, wd 1e-5, batch 128, grad clip 5; unweighted
`BCEWithLogitsLoss` + `lambda_M * reconstruction`; max epochs 240, patience 40;
valid ROC-AUC every epoch; raw = best valid AUC checkpoint (ties -> earliest);
Top-5 soup = equal-weight average of the 5 highest-valid-AUC checkpoints.

## 7. Correctness gates (G0–G14)

Enforced before any formal run and recorded in `correctness.json`; the GPU smoke
(512 graphs, 3 epochs) must also pass.  Official MolHIV test is blocked until
`unlock`, which requires the frozen `architecture_freeze.json` and a recorded
`decision.json`.

## 8. Reproduce

```bash
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_m1.py
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 dict
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 env
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 stats
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 calibrate
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 correct --device cpu
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 smoke --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 train --seed 0 --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 train --seed 1 --device cuda
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 freeze
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 analyze
uv run python -m tracks.ksvd.experiments.luyin16.molhiv_e2e_dictenv_m1 unlock --device cuda
```

Stages are resumable; `unlock` refuses a second test read in the same round.
