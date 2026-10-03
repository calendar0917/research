# REPORT — zinc-structure-semantic-factorial-seed0-v1

Canonical `Full` (`e2e_dictenv_scale_v1`, 408,651 trainable parameters),
source commit `8126300`, trained commit `118e2481362f`. **One seed (0), four
fresh 240-epoch trajectories**: `S_J` / `S_M` / `D_J` / `D_M`. Train-only 8000
fit / 2000 train-inner dev. Official-valid/test were never loaded or
instantiated (`official_test_loaded = False` in every artifact).

Method facts and file paths: `METHOD_CONTRACT.md`. Old-evidence boundary:
`OLD_EVIDENCE_SCOPE.md`. Numbers: `summary.json`, `main_table.csv`,
`group_table.csv`, `group_contribution_gains.csv`, `predictions.csv`,
`binding_collapse_diagnostics.json`, `operator_checks.json`.

## A. The four questions

### A1. How does the current Full predict, and where does luyin19 stand?

`Full` predicts from the structural code (`z = [d/common_rms; code(Dbar, r)]`,
`phi65 → D_raw[65,32]`, fixed `U[65,1]`), through two fused paths:
the **Sem108 interface** `[Sem108(108); size2(2)]` (shell-conditioned atom /
bond statistics) and the **node/edge binding slots** `(root, shell)` and
`(root, shellpair)` built from the structural code times the atom/bond
one-hots, then fusion `446→342→144`, the shared task dictionary
`D_L/V_L`, unary/pair moments, pair relations, global context, topology25→8 and
the reader. No message passing, no Transformer.

Consistent with `luyin19`: a single shared trainable structural matrix `D` is
used for every molecule and the architecture contains an explicit
structure×attribute cross. **Not yet delivered by this architecture**: the
transcript's transfer claim ("the learned structural–attribute relation
generalises to unseen graphs") is exactly what this round tests, and it is not
supported as a *training-time* inductive bias (see A2). The current node
channel is dead and three of four arms also killed the edge channel, so the
cross is largely not the mechanism that carries the current prediction.

### A2. Does the per-occurrence correspondence exceed the marginals, and does
it depend on the sparse code?

Two distinct measurements, and they point in different directions.

**(i) Across fresh arms (the pre-registered 2×2, dev, y units).**

| arm | G0 (k=0) cal | overall cal | overall raw | fit cal | fit/dev gap |
|---|---:|---:|---:|---:|---:|
| S_J | 0.102329 | 0.122601 | 0.123153 | 0.035538 | 0.087063 |
| S_M | **0.097347** | 0.120130 | 0.120122 | 0.037964 | 0.082166 |
| D_J | 0.100115 | 0.119293 | 0.122439 | 0.044809 | 0.074484 |
| D_M | 0.097941 | **0.117615** | 0.121349 | 0.038826 | 0.078789 |

Contrasts (positive = the first-named mechanism helps): `C_S = L(S_M) − L(S_J)`
(correspondence gain under sparse), `C_D` (under dense), `G_J = L(D_J) − L(S_J)`
(sparse gain under J), `G_M` (under M), `I = C_S − C_D`.

| endpoint | point | 95 % paired CI | classification (δ=0.003) |
|---|---:|---|---|
| **C_S, G0 cal** | **−0.004981** | **[−0.009020, −0.000740]** | **negative purchase signal** |
| C_D, G0 cal | −0.002174 | [−0.007165, +0.003112] | inconclusive |
| G_J, G0 cal | −0.002214 | [−0.006712, +0.002363] | inconclusive |
| G_M, G0 cal | +0.000594 | [−0.004359, +0.005915] | inconclusive |
| I, G0 cal | −0.002807 | [−0.009391, +0.003783] | inconclusive |
| C_S, overall cal | −0.002471 | [−0.009370, +0.006392] | inconclusive |
| C_D, overall cal | −0.001678 | [−0.008043, +0.004820] | inconclusive |
| G_J, overall cal | −0.003308 | [−0.008729, +0.002050] | inconclusive |
| G_M, overall cal | −0.002515 | [−0.008427, +0.003894] | inconclusive |
| I, overall cal | −0.000793 | [−0.009233, +0.007699] | inconclusive |

Raw-direction counterparts (full dev): G0 raw `C_S −0.005539, C_D −0.001676,
G_J +0.000556, G_M +0.004420, I −0.003863`; overall raw `C_S −0.003031,
C_D −0.001090, G_J −0.000714, G_M +0.001226, I −0.001941`. `G_J` changes sign
between raw and cal on G0 (+0.0006 vs −0.0022), so no raw/cal-consistent sparse
advantage exists.

**(ii) Within the trained S_J weights (frozen operator swap).** Only `S_J`
retains a live binding interface. Replacing its paired edge operator with the
independence expectation at evaluation costs **+0.010596 G0** and **+0.010446
overall** (raw). For `S_M`, `D_J`, `D_M` the swap changes predictions by
**exactly 0.0**.

Interpretation: conditional on a trained, *alive* edge interface, the paired
correspondence is strongly better than the marginals (reproducing the direction
of the old clean-mechanism `+0.00755`). But fresh training with the marginal
operator (`S_M`) drives the interface to zero and then **beats** the
correspondence-trained arm on G0 (`C_S` negative, CI excluding 0). The edge
interface is therefore *causally used when it survives*, but as a training-time
bias it does not pay for itself relative to switching the interface off.

Sparse dependence: no positive signal. `G_J` (sparse benefit) is negative/
inconclusive; `G_M` is near zero; `I` is inconclusive. The one observable
coding×binding structure is *survival*, not MAE: sparse+paired is the only
configuration whose edge binding weights stay non-zero (see A4).

Node/edge split: the **node** factor is unidentifiable (dead in all four arms);
no node evidence is claimed. All statements above are edge-channel statements.

### A3. Is there a G0 performance candidate worth a follow-up confirmation?

Not in the intended sense. The numerically best G0 arm is `S_M` (0.097347) and
the best overall arm is `D_M` (0.117615 cal). `S_M` does pass the stated numeric
candidate bar relative to `S_J` (G0 gain `+0.004981`, CI lower bound
`+0.000740 > 0`, overall cal change `−0.002471`, i.e. no worsening, raw same
direction) — **but it does so because its binding interface collapsed to
exactly zero**, not because its marginal operator is a better form of the same
interface. So it is a "trained binding-off" arm, not a clean simplification
candidate; it must not be promoted as such. No arm shows a raw/cal-consistent,
wide-margin G0 advantage. No seed-1 or full confirmation is bought here.

### A4. Mechanism post-mortem — the decisive observation

Trained raw-soup states (`binding_collapse_diagnostics.json`):

| arm | ‖W_A_S‖ | ‖W_A_C‖ | ‖W_E_S‖ | ‖W_E_C‖ | eval swap ΔG0 (indep−paired) |
|---|---:|---:|---:|---:|---:|
| S_J | 6.3e−40 | 6.3e−40 | **4.263** | **1.644** | **+0.010596** |
| S_M | 6.3e−40 | 6.3e−40 | 6.3e−40 | 6.2e−40 | 0.0 |
| D_J | 6.3e−40 | 6.3e−40 | 6.3e−40 | 6.2e−40 | 0.0 |
| D_M | 6.3e−40 | 6.3e−40 | 6.3e−40 | 6.2e−40 | 0.0 |

* The **node binding collapsed to denormal zero in all four arms** (slot
  variance 0 at epoch 240). The node factor is functionally unidentifiable.
* The **edge binding stayed alive only in sparse+paired (`S_J`)**; `S_M`, `D_J`
  and `D_M` zeroed it. Health curves show `S_J` edge-slot variance growing to
  9.3e−6 while the others reach 0.
* Therefore the 2×2 is **not a clean operator contrast at evaluation**: the M
  arms realize "no binding at all", not "same marginals". The `C_S` signal is a
  training-dynamics/live-vs-dead effect, not "marginals beat correspondence on
  an otherwise equal interface".

This is exactly the pre-registered warning case ("the J/M difference is
absorbed by collapse in the real model; the interface claims are not
sufficiently identified; evidence stops at interface deactivation") — with the
extra, load-bearing fact that conditionally the live interface is functional
and paired-beats-indep by 0.0106.

### A5. Group contributions (cal, contribution = Σ|error|/2000; sums equal MAE)

| group | n | S_J | S_M | D_J | D_M |
|---|---:|---:|---:|---:|---:|
| k=0 | 1926 | 0.098543 | 0.093746 | 0.096411 | 0.094317 |
| k=−1 | 65 | 0.005528 | 0.005587 | 0.005499 | 0.004241 |
| k≤−2 | 9 | 0.018531 | 0.020798 | 0.017383 | 0.019057 |
| overall | 2000 | 0.122601 | 0.120130 | 0.119293 | 0.117615 |

The G0 `C_S` advantage is broad-based over the 1926 rows
(contribution gain −0.004797). The rare tail behaves oppositely: M arms are
worse on `k≤−2` (`C_S` contribution gain +0.002267), and `D_M` buys a large
`k=−1` gain (−0.001259). Contribution identity `Σ_groups = overall` holds to
0.0 for all four arms.

## B. Operator / invariant checks (real code, real forward)

`operator_checks.json`, `mechanism_ok = True` (CPU; GPU smoke `mechanism_ok =
True`):

* `S_J` forward equals canonical `Full` on the same state: max abs diff
  `5.96e−7` (GPU, tolerance 2e−6; 0.0 on CPU).
* four arms: 408,651 trainable parameters each, trainable-parameter hash
  identical at init (`init_identity.json` `all_identical = True`).
* coding: both widths 33; sparse exactly 8 nnz/row, dense 100 % nonzero; dense
  code equals `kappa·(r@Dbar)` exactly; `kappa = 0.2976927507` from 185,462 fit
  nodes, `rms(kappa·dense) = rms(alpha_S)` to 2.4e−16 relative.
* micro-bucket enumeration: mean of J over all semantic permutations equals M
  (`n=2` 9.3e−10, `n=3` 3.0e−8), `n=1` J==M, `n=0` M==0. The tested functions
  are the production functions (`indep_bucket` is used by
  `_indep_environment_from_parts`).
* bucket locality: per-molecule slots equal the 2-molecule batch blocks
  (<1e−9 GPU), graph order changes predicted values by <3e−7.
* dispatch: M forward increments `indep_calls`; gradients reach `D`, `W_A_S`,
  `W_A_C`, `W_E_S`, `W_E_C`; `kappa` is a buffer, not a parameter; no
  message-passing/Transformer module.
* real inputs are non-degenerate for the operator: J−M non-zero bucket fraction
  node 0.579 / edge 0.292; n≥2 fraction node 0.585 / edge 0.293 — so the
  collapse is a *learned* phenomenon, not an input degeneracy.

## C. Learning diagnostics

All four runs: 240 epochs, 15,120 optimizer steps, soup 236–240, FP32,
Adam 1e−3/1e−5, batch 128, clip 5, `λ = 33.95873017865987`, no scheduler/AMP/
DDP. Train MAE (last epoch): S_J 0.0646, S_M 0.0685, D_J 0.0753, D_M 0.0772.
Residual-code rank: sparse ≈ 17.0–17.6, dense ≈ 9.2. Reconstruction
`train_rec` at 240: sparse 4.4e−5 / 3.9e−5, dense 8.4e−7 / 8.7e−7. The dense
arms fit the fit set worse (fit cal 0.0448 / 0.0388 vs sparse 0.0355 / 0.0380)
and have smaller fit/dev gaps (0.0745/0.0788 vs 0.0871/0.0822): the sparse
functional form over-fits more, consistent with the old concentration finding.

## D. Boundaries and what is not claimed

* One seed, one reused 2000-row train-inner dev; this is a mechanism screen /
  exploratory purchase signal, not a confirmation set. The CIs describe this
  seed and this dev only.
* No "all dictionaries vs no-dictionary MLP": the dense arms keep the trainable
  shared `D` and the task dictionary `D_L/V_L`.
* No claim that all structure–semantics interaction is useless: Sem108, size2,
  marginals, shell/shellpair positions, relations and topology stay in all
  arms. What is challenged is the incremental value of the explicit slot
  binding over those retained bypasses.
* `kappa` matches initial residual-code RMS only — not direction, rank or the
  full gradient geometry of the dense code.
* The old `+0.00755`/`−0.002936` numbers are not reusable as current effect
  sizes (`OLD_EVIDENCE_SCOPE.md`).
