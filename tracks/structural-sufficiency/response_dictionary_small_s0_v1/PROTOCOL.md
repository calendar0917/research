# PROTOCOL — response dictionaries on the small official splits (seed 0)

Status: frozen before the formal runs. Code commit is recorded in every run's
`status.json` / `summary.json`; the data caches are content-addressed.

## Question

Under one fixed configuration, is a real conditional response between a real
fragment and a real environment more useful than a new matched additive
control; and can a response dictionary enter the real prediction path with an
acceptable loss? This is a fast developmental screen on one seed and the
official splits. It cannot establish cross-seed stability or unseen
fragment–environment generalisation.

## Data and identity

* OGB `ogbg-mollipo` (3360/420/420) and `ogbg-molesol` (902/113/113), official
  `get_idx_split()` rows only. The frozen raw-graph caches of
  `tracks/structural-query-learning/data/ogbg_{mollipo,molesol}_sq-pilot-cache-v1.npz`
  are re-verified by SHA-256 at every load (`sq_common.EXPECTED_CACHE_SHA256`).
* Targets are standardised with Train mean and population std (`ddof=0`) only;
  all reported metrics are back in raw units.
* `test_policy: always`; Test is read every epoch and reported as
  `DEVELOPMENT_TEST`. Test never selects a checkpoint, a dictionary or a
  hyper-parameter.

## Frozen structural representation

* Cuts: non-ring, single, bridge bonds of the physical bond graph. The smaller
  side is `S` (1–8 heavy atoms); equal sides keep both orientations; `C` is the
  complement in the whole molecule (all other components retained). The cut
  bond and both ports are stored; exact reassembly is checked at preparation.
* Identity: exact canonical serialization of the coloured incidence graph
  (atoms carry all 9 raw columns + port flag, bonds become auxiliary vertices
  carrying all 3 raw columns) via `pynauty.canon_label`. The stored key is the
  SHA-256 of that serialization; every hash match is confirmed by full-string
  equality, and a true collision aborts preparation.
* Fixed views: sort by `(|S|, key(S), key(C), bond_attributes)` and take
  `q = min(4, m)` with the frozen index rule. The same fixed view set is used
  for training and for all Train/Val/Test evaluation.
* `F(G)` is the mean of the at most four view scores. Molecules without a legal
  cut use one port-free whole-graph encoding and the shared fallback scalar
  head; they are counted separately and never count as dictionary coverage.

## Frozen model (both arms)

Shared encoder: width 96; 9 atom-category embeddings summed + binary port
embedding + internal-degree embedding (0–15, `>=16` one bucket); 3 bond
embeddings summed and used in both message directions; 3 `GINEConv` layers with
`96→192→96` MLP, SiLU, `train_eps=True`, residual, LayerNorm, SiLU,
dropout 0.10 (no BatchNorm). Descriptor
`concat(port_state96, sum96, mean96, log1p(n_atoms))` → `289→128`, SiLU,
LayerNorm → `z`. `a(S)`, `b(C)` are `128→64→1` SiLU MLPs. `U,V` are bias-free
`128→16`; `w ∈ R^16` starts at exact zero. Fallback head `128→64→1` only for
no-cut molecules.

```
RESP: T = sum_j w_j q(S)_j k(C)_j / sqrt(16)
ADD : T = sum_j w_j (q(S)_j + k(C)_j) / sqrt(32)
s(C,S) = b(C) + a(S) + T;  F(G) = mean_view s(C,S)
```

Category counts come from the official OGB feature definition, never from
Val/Test statistics. Both arms load one canonical `manual_seed(0)` state dict
(identical hash recorded) and reuse one frozen training order.

## Frozen training and evaluation

* seed 0, 360 epochs, batch 128 original molecules (tail kept), AdamW
  `lr=1e-3`, `wd=1e-5`, `betas=(0.9,0.999)`, `eps=1e-8`; 5-epoch linear warmup
  from `1e-4`, then per-optimizer-step cosine to `1e-5`; grad clip 5; FP32,
  TF32 off.
* Loss: per molecule, mean over its fixed views of `(s - y_std)^2`; then mean
  over original molecules. Fallback molecules contribute one squared error.
* Every epoch: `eval()` / `no_grad()` full Train/Val/Test with raw RMSE and MAE;
  evaluation batch 256 molecules, reduced to 128/64 only on OOM (recorded).
* Best-Val checkpoint (strict `<`, earlier epoch on ties), final checkpoint,
  all per-epoch curves and best/final raw predictions per row id.
* `HORIZON_LIMITED` if mean Val RMSE of epochs 331–360 is at least
  `0.005 * Train_std` below epochs 301–330 **and** the best of 331–360 lies in
  351–360. No continuation.

## Frozen dictionary stage (no new training)

Only the Train fixed views define words, frequency, `D0`, the environment
panel, prototypes and the response matrix. Connection type
`tau = (port atomic number, port formal charge, port aromatic flag, all three
cut-bond attributes)`. Per class: `D0` = highest Train molecule frequency
(tie: smallest canonical key); environment panel ≤16 by Train molecule
frequency (tie: canonical key); `K = min(64, train candidates)`; `D0` is always
in the dictionary. Response `R(S)[j] = s(C_j,S) - s(C_j,D0)` on the class
panel, computed from cached frozen `a,q,b,k,w` (the GNN is not re-run per pair).

Three selections share tau, K, panel, reference, alpha solver and fallback
rule: `D_RESPONSE` (farthest-first in R), `D_LATENT` (farthest-first on
L2-normalised `[a(S),q(S)]` with `1-cos`), `D_FREQ` (Train frequency). Alpha is
the same response-space simplex projection: fixed 64-step Frank–Wolfe starting
at the nearest prototype, exact 1-D line search clipped to `[0,1]`, zero
direction keeps the point; no extra L1; each prototype is forced one-hot.

Prediction path: `a_D = Σ α a(D_k)`, `q_D = Σ α q(D_k)`,
`s_D = b(C) + a_D + Σ_j w_j q_D[j] k(C)[j]/sqrt(16)`, `F_D` = mean over the
same fixed views. Original `a(S),q(S)` are not added back for dictionary
words; unsupported views keep the continuous interface and are counted.
Classes whose `K` equals all candidates are exact storage, not compression
evidence.

Compatibility note: the frozen model consumes two branch descriptors and never
re-splices a molecule, so every `(C_j,S)` pair inside one tau class is a legal
model input; `DICTIONARY_UNSUPPORTED` therefore counts classes with no Train
word or no Train environment panel, not a failed chemical re-join.

## Diagnostics and labels

Gain `= RMSE_ADD - RMSE_RESP` at Val-best, final and last-10 mean; Test follows
the same selected states. Paired molecule bootstrap 2000 draws, seed 1703, at
the two best-Val states (fixed training state, not a training-seed interval).
Top-1/top-5 removal is an appendix. `w=0` on the RESP best-Val state is a
fixed-weight dependence probe. View dispersion and fallback shares are
reported per split. `HORIZON_LIMITED`, `BUDGET_NOT_RUN` and `ENGINEERING_BLOCKED`
are never mixed into a model-negative conclusion.

Development references (not re-trained, not strictly paired): lower Val C1 RMSE
of two pre-declared historical sources, Lipo 0.7052 and ESOL 0.7958; a Val RMSE
within 1.05× of that value is `NEAR_HISTORICAL_C1`, otherwise
`PERFORMANCE_BEHIND_C1`. The dictionary is "worth further study" only when
`RMSE(D_RESPONSE) - RMSE(RESP) <= 0.01 * Train_std` on Val with supported view
fraction ≥ 0.8, and the comparison to D_LATENT/D_FREQ is reported alongside.

## Budget and stop rules

Four formal units at most (Lipo RESP/ADD, ESOL RESP/ADD), seed 0, 360 epochs,
no second recipe. Soft 2 h wall / 4 A100·h, hard 4 h wall / 6 A100·h including
smoke and failures. If the matrix cannot finish, Lipo both arms complete first
and ESOL is marked `BUDGET_NOT_RUN`. NaN/identity/resource failure stops the
affected unit and is recorded; no result-driven early stopping.
