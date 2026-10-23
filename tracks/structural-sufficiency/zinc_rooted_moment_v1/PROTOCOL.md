# PROTOCOL — zinc_rooted_moment_v1 (RMC)

Protocol id: `ssrc-zinc-rooted-moment-v1`
Track: `structural-sufficiency`
Baseline / reference: `tracks/structural-sufficiency/zinc_native_relation_v1`
(commit `74dea2d`, `P3NullModel`, 413,505 parameters).
Status: **FROZEN before the first formal GPU training run.** Config
(`configs/rmc_v1.json`), code, CPU checks (`analysis/cpu_checks*`) and this file
are committed together. Any change to the mathematics, cache contract, model,
recipe, decision rule or budget invalidates this protocol and needs a new id.

The official ZINC **test** split is never built, loaded or scored during
development. It is read at most once, and only if the terminal rule in §9 fires.

## 0. Question and hypotheses

Final target of the track is unchanged: official ZINC subset raw-`y` test MAE
`< 0.060`. This round does not promise to reach it. It asks one falsifiable
question:

> If every local neighbourhood statistic that the trunk can compute from the
> P3 objects is replaced by an **exact, label-free summary of the
> neighbourhood's empirical endpoint-feature distribution as seen from the
> root** — queried by the root itself — does that beat (a) the same trunk
> without it and (b) a parameter- and input-identical Gaussian closure that
> throws away the distribution's skew?

* **H1 (M vs P).** Candidate M improves overall valid MAE and/or the frozen
  225-molecule unseen+ordinary group relative to the same-round P3 reference P,
  repeatably across supervised seeds 0/1.
* **H2 (M vs G).** With identical parameters, initialisation, optimiser, data
  order, cache and readout, using the true third central moment C3 beats setting
  C3 = 0 (Gaussian closure). `M − G` is the only contrast that isolates *skew*;
  `M − P` confounds extra capacity and the extra readout with the mechanism.
* **H3 (test).** The frozen single-model candidate reaches official test
  `< 0.060`, evaluated only under §9.

`M − P` cannot by itself establish H2, and `M − G` does not show that moments
are a *necessary* mechanism (only that this particular closure is worse). No arm
here can claim that moment conditioning is new as a mathematical object (§10).

## 1. Fixed inputs

* `PyG ZINC(subset=True)`, official train 10,000 / valid 1,000; raw `y`, L1.
* Frozen P3 object cache (`zinc_arch_discovery_v1/cache/ad_{train,valid}.pt`):
  atom 108-d, bond 7-d, junction 32-d, `atom_ptr/bond_ptr/junc_ptr`,
  molecule-local `bond_index`/`junc_index`, global-33.
* Frozen P3 encoders (`P3Encoders`) produce `(a, b, w)`; the atom output `a` is
  the online 96-d atom state used by every arm.
* **Endpoint features `u_j` (fixed, never trained):**
  `u_raw = [atom_108 ‖ memory_atom_96]` (204-d), where `memory_atom_96` comes
  from the frozen SSL M1 atom encoder
  (`tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1/pretrain_encoder_s0.pt`,
  SHA-256 `006fabc11efede091420fc2126ef2ef1c46cb3948ab91648a96843c0352bfa97`),
  loaded once, `requires_grad=False`, **not** in the optimiser.
* Two fixed views: `R_v ∈ R^{204×12}`, `v ∈ {0,1}`, seeded by 7301 / 7302,
  `u_j^{(v)} = tanh(u_raw R_v)/sqrt(12)`. Projections are built from the
  train-only atom matrix, frozen into `cache/rmc_norm.pt`.
* Global-33 standardiser: the original P3 train-only one, reused unchanged.

## 2. Mathematics

Undirected simple adjacency `A`, `B = I + A`, scales
`K = {0,1,2,3,4,6,8,12}`, `W_k = B^k` (integer, computed by incremental
matrix powers in float64, verified equal to brute-force walk enumeration).

For root `i` and scale `k`:

```
m_ik      = sum_j W_k[i,j]                       (integer local mass)
p_ik(j)   = W_k[i,j] / m_ik                      (empirical measure on endpoints)
mu_ik     = sum_j p_ik(j) u_j
C2_ik     = sum_j p_ik(j) (u_j - mu_ik)(u_j - mu_ik)^T
C3_ik     = sum_j p_ik(j) (u_j - mu_ik) ⊗ (u_j - mu_ik) ⊗ (u_j - mu_ik)
```

Positive quadratic kernel `kappa(q,u) = (1 + q^T u)^2 + eps`, `eps = 0.05`,
`a = 1 + q^T mu`, `Z = a^2 + q^T C2 q + eps`, and the identity used verbatim in
the model and in the cache validator:

```
F_ik(q) = mu_ik + [ 2 a C2_ik q + C3_ik[q,q] ] / Z_ik
        = sum_j p_ik(j) kappa(q,u_j) u_j / sum_j p_ik(j) kappa(q,u_j)   (exact)
```

`C3_ik[q,q]` denotes `sum_{a,b} C3[a,b,:] q_a q_b`. `Z >= eps` always, because
`a^2 >= 0` and `C2 ⪰ 0`; no clamping is used anywhere. `C3` here is **not** a
three-node motif or a triangle/ring counter: it is the third central moment of
the same endpoint-feature distribution, i.e. its skew. Nothing in this round may
be described as a shortest path, a resonance or a substructure count.

Cache (**per root atom**, per scale, per view): `mu` (12), `C2` (78 packed
upper-triangle including diagonal), `C3` (364 packed: 78 diagonal `a<=b<=c`
plus 286 mixed `a<b<c` permutations, with the exact multiplicity folded in),
`log m` (1) → 455 float32 per (atom, scale, view), 7,280 float32 (29.1 kB) per
atom. Measured on the full split (14 worker processes, CPU): train 231,664 atoms
= 6.74 GB, valid 23,083 atoms = 0.67 GB, build 14.1 s / 1.9 s, wall 37.4 s,
peak RSS 9.2 GB, stored as **float32** after **float64** construction and
validation (float64 sums are kept only for the checks).

One-dimensional witness used in the checks (fixed prior `(0.5, 0.5)`, `q = 1`):
A: `0.3846153846`, B (with C3): `0.4526064213`, Gaussian closure of B:
`0.3846153846`, third central moment of B: `0.08838834765`. The direct sum
matches the moment form to `~1e-14`; the CPU identity gate requires `< 1e-10`.

## 3. Model (single predictor)

```
(a, b, w) = P3Encoders(batch)                              # frozen architecture, online weights
e_i  = MLP(290 -> 128 -> 96)([a_i, mean_j b_ij, mean_j w_ij, log1p deg_i, log1p njunc_i])
q    = 4 * tanh(MLP(96 -> 48)(e_i)) / sqrt(12)             # 4 heads x 12
F, logZ = rooted readout of (mu, C2, C3, log m) at q
          head 0,1 -> view 0 ; head 2,3 -> view 1 ; each head reads all 8 scales
readout = [ e_i (96) ‖ F (4*8*12 = 384) ‖ logZ (4*8 = 32) ‖ logm_std (8)
            ‖ global33 (33) ‖ deg_std (1) ‖ log1p n_atoms ‖ log1p n_bonds ]   # 556
h_i  = MLP(556 -> 224 -> 96)(readout)
t_i  = MLP(96 -> 128 -> 1)(h_i)
y_hat = graph_bias + sum_i t_i / train_mean_n_atoms
```

`F` is standardised by train-only `(F_center, F_scale)`; `log m`, `deg`,
`log1p n_atoms`, `log1p n_bonds` likewise. No LayerNorm, BatchNorm, dropout,
residual stream, attention, or old P3 prediction bypass. Molecule-level readout
is a sum of per-atom scalars, exactly as in P3.

Parameter audit (checked at build time, CPU, and remotely):

| block | parameters |
|---|---|
| P3 encoders (online) | 202,048 |
| e_mlp 290→128→96 | 49,632 |
| q_proj 96→48 | 4,656 |
| h_mlp 556→224→96 | 146,368 |
| t_mlp 96→128→1 | 12,545 |
| graph_bias | 1 |
| **M / G / U total (optimiser)** | **415,250** |
| frozen memory atom encoder | 26,336 (not in optimiser) |
| **deployment total** | **441,586** (< 500,000) |
| P reference | 413,505 |

## 4. Arms

* **P** — `P3NullModel` used verbatim from `zinc_recipe_trunk_v2/ad2_models.py`
  (same builder, same init RNG fork, same inputs, 413,505 params).
* **M** — full model above, with `C3`.
* **G** — identical to M with `use_c3 = False`; the readout then returns the
  Gaussian closure *with the same formula* (C3 contribution set to zero), so
  `M(forced C3=0) == G` bit-for-bit is a gate. Same parameter count, same
  initial state (cloned from a single canonical state per seed), same data
  order, same readout input dimensions.
* **U** (optional, only under §8) — molecule-internal uniform control: the same
  readout, but `mu/C2/C3` are replaced by the uniform measure over the
  molecule's own atoms (`p(j) = 1/n`), i.e. the distribution is destroyed while
  the marginal endpoint features and all shapes are kept.

Pairing: for each supervised seed `s ∈ {0,1}` the canonical initial state is
generated once with `torch.manual_seed(s)` inside the same `fork_rng` block and
cloned into M and G; P keeps the original `P3NullModel` construction path. The
paired-init gate requires `sha(encoders(P)) == sha(encoders(M)) ==
sha(encoders(G))` and `sha(state(M)) == sha(state(G))`.

## 5. Pre-registered CPU checks (all must pass before any GPU run)

Synthetic: witness values; `F`/`Z` moment form vs direct double-precision sum
(`< 1e-10`); `q = 0 ⇒ F = mu, Z = 1 + eps`; point mass `⇒ F = u`; single atom
collapses at every scale; packed vs full `C2/C3`; `C2 ⪰ 0`; `k = 0 ⇒ C2 = C3 =
0, mu = u`; `W_k` vs brute-force walk enumeration; `Z >= eps` at large `q`
(real cache, no clamp); `dF/dq` vs autograd of the direct sum.

Data: endpoint cache reproduces the frozen memory encoder (`< 2e-6`);
`q = 0` on the real cache returns `mu`; real-cache `C2` PSD (float64) and
`Z >= eps` under the model's own queries; batch vs single-molecule prediction
invariance and reversed-batch invariance; atom renumbering and bond-direction
invariance; empty membership (`deg = 0`, `njunc = 0`) gives exactly zero
`mean_b/mean_w` (no NaN, no leakage across molecules); all model groups receive
finite non-zero gradients and the optimiser covers exactly 415,250 parameters;
parameter counts and the frozen-norm hashes match the config.

## 6. Recipe (verbatim `recipe_variant="B"`, unchanged trainer)

`ad_train.train(..., recipe_variant="B")`: AdamW, lr 1e-3 → 1e-5 cosine, wd
1e-5, warmup 500 updates, grad-clip 5, batch 128 molecules (79 updates/epoch),
600 epochs = 47,400 updates, full valid every epoch, best-valid with strict `<`,
save `best.pt` + `final.pt`, no weight soup, no averaging, no early stop, no
HPO. Six formal runs: `{P,M,G} × {seed 0, 1}`.

## 7. Endpoints and decision rule (frozen)

`delta = MAE(candidate) − MAE(reference)` (negative is better).
Primary endpoint **T100** = median valid MAE over epochs 501–600. Reported per
seed, per arm, plus the two-seed mean; also best-valid value/epoch, W20 (epochs
1–20 mean), final epoch, eval-mode train MAE, and cost.

*Ordinary* = the frozen 1,000-molecule valid split restricted to the
train-ordinary stratum (as in `zinc_native_relation_v1`); *unseen* = the frozen
225-molecule group (single model, one number). Row 172 is reported separately.

Inference: per-molecule paired bootstrap for `M − P`, `G − P`, `M − G`
(same-molecule seed-mean first, 10,000 resamples, 95% percentile interval); the
2,000 per-seed predictions are never treated as independent. Sensitivity:
ordinary-only, and ordinary with the 10 largest molecules removed.

**Decision rule (frozen).** `d_T100(X) = MAE_T100(X) − MAE_T100(P)`.
1. **H1 supported** iff both seeds have `MAE_T100(M) < MAE_T100(P)` and the
   two-seed mean gain `−mean(d_T100(M)) >= 0.003`.
2. **H2 supported** iff both seeds have `MAE_T100(M) < MAE_T100(G)` and the
   two-seed mean gain `−mean(d_T100(M) − d_T100(G)) >= 0`.

Stability: both seeds agree in sign for `M − P` and for `M − G`. The headline is
H1; H2 is the mechanism contrast. Subgroup reading (`ordinary` best+final both
better, `unseen+ordinary` final both better) is reported next to the verdict but
does not change the pre-registered H1/H2 rules. If H1 fails, the report says the
body did not reproduce; if H1 holds but H2 does not, the report says the skew
term was inert at this budget (the Gaussian closure is as good).

## 8. Optional U arm trigger

U is run **only if all** of the following hold for the *prospective method*,
define as the arm with the better two-seed mean T100 among `{M, G}`: (a) both
seeds improve T100 versus P; (b) two-seed mean gain `-mean(d_T100) >= 0.003`;
(c) ordinary best and final are better on the two-seed mean; (d) `unseen+ordinary`
final is not worse on either seed (`gain >= 0`) and its two-seed mean gain is
`>= 0.005`; and (e) the budget still allows two complete 600-epoch runs (§11).
If the trigger fails, no U run happens and this round makes no claim about
uniform/degree-based controls beyond what M/G show.

## 9. Official test read policy

The test split is touched at most once, after: CPU checks pass, the six formal
runs finish, `analyze` has written the frozen per-molecule predictions, the
selection (best-valid `best.pt`, fixed inside `train.json`) is written, and the
model/config/cache hashes are committed. Then one CPU scoring pass over the
terminal checkpoints (`best.pt`) of M seed 0/1 (+ P, G for the record). If the
§8 trigger does not fire, this round reports **no** test number.

## 10. Novelty boundary (what is and is not new here)

| prior work in this repo | relation to this round |
|---|---|
| A / B arms (`zinc_recipe_trunk_v2`): normalisation placement, own-feature gating | no moments, no root conditioning |
| BCCA (`zinc_bcca_factorial_v1`): cross-attention between atom and bond/junction tokens | learned attention over tokens, not an exact moment summary |
| SF (`zinc_structure_features_v1`): scalar structural features injected | counts only, no feature-distribution moments, no query |
| NR (`zinc_native_relation_v1`): relation-type ↔ token-pair message passing | pair messages, no moments, no root query |
| R0/R1 (`notes/prior_work_boundary.md`) | rooted *count* features only; no C2/C3, no kernel readout |

Literatures this round leans on, none of which is claimed as new:
Linear Attention / kernelised attention (Katharopoulos et al., 2020;
Choromanski et al., 2021) for the feature-map view `kappa(q,u) = phi(q)^T phi(u)`;
TaylorShift (Nguyen et al., 2025) and polynomial/kernel GNNs (e.g. graph
polynomial filters) for the `(1 + q^T u)^2` polynomial kernel; SIGN (Frasca et
al., 2020) and MM-GNN (Zhu et al., 2021) for precomputed multi-scale
neighbourhood summaries; GRIT (Ma et al., 2023) for rooted, query-conditioned
readout. The contribution claimed here is narrow: an **exact**, label-free,
root-conditioned second/third-moment summary of the frozen endpoint-feature
distribution, with a same-parameter Gaussian-closure control.

## 11. Budget and provenance

Total 5.0 GPU·h measured as the sum of both A100s' wall time; pre-formal
(checks/timing) + formal ≤ 4.7 h, leaving 1.15 h slack. CPU prep + validation
target ~90 min wall. Every run records driver version, torch version, node,
GPU model, seed, wall time, GPU·h and the SHA-256 of the config, code, cache,
norm file, label arrays and data-order stream (`run_extra.json`). `rr` is used
for all remote execution; the two A100s are used in parallel. Nothing is
overwritten: this directory is new, and the old `zinc_native_relation_v1`
report/config stay untouched.

## 12. Non-claims

This round does not claim: a new kernel, a new message-passing scheme, a
state-of-the-art ZINC result, that moments are necessary for molecular
regression, that C3 encodes rings/substructures, or that a positive `M − G`
result transfers to larger scales. It also does not revisit the old HR /
normalisation question, and it makes no claim about the official test split
unless §9 fires.
