# E2E-DictEnv-RoleCorr-v1 — pre-registration (frozen)

Round: **E2E-DictEnv-RoleCorr-v1** (`e2e_dictenv_rolecorr_v1`).
Protocol: `zinc-context-gap` (study `zinc-context-gap`), track `tracks/ksvd`.
Write-then-follow: nothing below may change after the first formal run.

任务来源（中文摘要）：在现有 ZINC + 跨分子共享字典路线上，保持
Sem108 直接语义接口与静态组合读出不变，只改变**字典学习对象**。新假设是
“化学属性落在哪种细粒度结构角色上”具有预测价值，并可通过共享字典跨分子复用。
对每个 radius-2 rooted patch，在每个节点 shell / 边 shellpair 内计算
``C_g = Σ_i (b_i − mean_g(b))(q_i − mean_g(q))^T``（角色 × 属性 的组内中心化
交叉统计），去掉组内恒定列后得到 ``C ∈ R^536``，在其上学习共享字典
``D_C ∈ R^{536×16}``；首轮冻结所有字典，只训练原有读出，检验该表示的任务价值。
本轮不做端到端字典微调、不做多 seed、不做超参数搜索。

---

## 1. Single question

Does replacing the pure-topology dictionary coordinate of the frozen
``CSSD-Sem108 + C6`` pipeline with a **role↔attribute correspondence**
coordinate — a shared sparse dictionary learned directly on the within-group
centred role×attribute statistic ``C ∈ R^536`` — give the otherwise unchanged
no-message-passing static composition a material task gain over the existing
pure-topology K32 dictionary coordinate, when **both** dictionaries are frozen
and only the readout is trained?

Subsidiary questions (only answered if the primary screening gate fires):

1. Is the gain specific to the **correspondence** between role and attribute?
   (shuffled-correspondence control ``C``; inference zeroing / shuffling of
   ``alpha_C`` on the trained candidate).
2. Does learning a **sparse dictionary** on ``C`` add value over a dense
   train-fitted **PCA16** code of the same object?

This is a representation screen, not a new end-to-end architecture search.

---

## 2. Frozen objects (read-only)

| object | definition | source |
|---|---|---|
| rooted patch | radius-2 induced ego frame, node order = ``graph.nodes``, patch order = root order | `fsar_v2._explicit_basis_for_patch`, `a1.patch_blocks` |
| node role basis ``b^V`` | 11-D: root(1) ‖ shell one-hot(3) ‖ log1p degree(1) ‖ log1p neighbour-by-shell(3) ‖ log1p walks A,A²,A³(3) | `fsar_r2_ar0.NODE_BASIS_DIM=11` |
| edge role basis ``b^E`` | 15-D: shellpair one-hot(6) ‖ log1p deg-sum, log1p |deg diff|, log1p common(3) ‖ log1p neighbour-sum(3) ‖ log1p |neighbour-diff|(3) | `fsar_r2_ar0.EDGE_BASIS_DIM=15` |
| atom attribute ``q`` | 28-category one-hot | `fsar_r2_ar0.ATOM_CATEGORIES=28` |
| bond attribute ``r`` | 4-category one-hot | `fsar_r2_ar0.BOND_CATEGORIES=4` |
| common subspace | frozen CSSD-q1 ``U ∈ R^{65×1}``, ``U = mu/‖mu‖`` (uncentred train mean direction), ``common_rms = 5.082852828320509`` | `results/e2e_dictenv_common_subspace_dictionary_v1/common_subspace.json` |
| baseline dictionary | frozen ``D_SDB ∈ R^{65×32}`` (SDB-v0 K-SVD, ``s=8``) | `results/sdb_v0/dictionary.pt` |
| base architecture | frozen ``SEM108Model`` (`e2e_dictenv_sem108_v1`): direct ``[Sem108 ; size2]`` (110-D) fusion, node 3×48 / edge 6×32 bindings, 48-D static environment, frozen relation/distance/global/topology backend, reader | `e2e_dictenv_sem108_v1.py` |
| mask | frozen C6 | `cm.C6_MASK` |

Nothing is refit except the two new dictionaries (and the readout, which is
the only trained part in this round). The common subspace, the SDB baseline
dictionary, ``Sem108``, the backend and the reader definition are frozen.

---

## 3. The new object ``C`` (frozen construction)

For one rooted patch let ``P`` be its node set with rooted distances
``s_v ∈ {0,1,2}`` (shells), and let ``E`` be its sorted undirected induced
edge list with shellpairs ``(s_u, s_w)``.

**Node groups** ``g ∈ {(shell 1), (shell 2)}``:

```
C^V_g = sum_{v in g} (b^V_v[:, 4:11] - mean_g(b^V[:, 4:11]))
                     (q_v          - mean_g(q))^T               in R^{7 x 28}
```

**Edge groups** ``g ∈ {(0,1), (1,1), (1,2), (2,2)}``:

```
C^E_g = sum_{e in g} (b^E_e[:, 6:15] - mean_g(b^E[:, 6:15]))
                     (r_e          - mean_g(r))^T               in R^{9 x 4}
```

Rules (frozen):

* **Sum, not mean**: no division by the group size.
* ``C_g = 0`` when ``|g| < 2`` (empty and singleton groups), including the
  always-singleton shell-0 node group and any empty shellpair group.
* Undirected edges are counted **once** (``fsar_v2`` sorted induced edge list).
* Role columns that are constant inside a group (root/shell indicators for
  node groups, shellpair indicators for edge groups) are removed exactly;
  keeping them would only add identically-zero coordinates.
* Node groups store shells 1 and 2 only; the root shell is a singleton.

**Fixed concatenation order** (all float64 during construction):

```
C = [ C^V_shell1 (196) ; C^V_shell2 (196) ;
      C^E_(0,1) (36) ; C^E_(1,1) (36) ; C^E_(1,2) (36) ; C^E_(2,2) (36) ]
  in R^536
```

``C`` is invariant to node renumbering and to edge endpoint exchange, and it
never reads the target ``y``.

### 3.1 Relation to prior objects (not a re-run)

* **A1 ``chi_REAL``** was the *whole-patch* joint ``sum_v b_v q_v^T`` (plus
  ``phi``), i.e. no within-shell centring; A1 stopped at Gate 0.
* **FEC-D1** bound an *already learned* 32-D topology code ``alpha_v`` to
  chemistry inside FEC-S1 (``C = Σ (alpha_v − ā)(q_v − q̄)^T``); it never
  learned a dictionary on the correspondence object itself.
* **This round** learns a shared dictionary **directly on the raw role×
  attribute correspondence** ``C`` and feeds its sparse code through the
  existing Sem108 static-composition readout.

---

## 4. Scaling (train only, frozen)

Reuse ``a1.fit_block_scaler`` verbatim on the two blocks separately:

* layer 1: per-coordinate train RMS ``sqrt(E[z^2] + 1e-12)`` with a zero-RMS
  mask at ``1e-9``;
* layer 2: block energy equalisation ``w_b = 1/sqrt(E_train[‖block_b‖²]+1e-12)``
  so the node (392) and edge (144) blocks carry equal mean squared energy;
* **no** extra global mean centring (within-group centring already defines the
  zero point of every group).

``D_C`` and the PCA16 control are fit on the **scaled** train object only.
Official valid is never used to fit any scaler, dictionary or PCA.

---

## 5. Frozen coordinate and model

```
coordinate  z = [ c~ (1) ; alpha_S (16) ; alpha_C (16) ]   in R^33
c~          = (phi · U) / common_rms                        (frozen CSSD-q1)
alpha_S     = IHT10( colnorm( (I-UU^T) D_S ), (I-UU^T) phi )  exact top-4
alpha_C     = IHT10( colnorm( D_C ), C_scaled )               exact top-4
```

* ``D_S ∈ R^{65×16}``: unlabeled K-SVD on the train residual ``r = (I-UU^T) phi``.
* ``D_C ∈ R^{536×16}``: unlabeled K-SVD on ``C_scaled`` (official train only).
* K-SVD pipeline: ``sdb.fit_ksvd`` (existing, reused verbatim):
  ``atoms=16``, ``s=4``, ``epochs=10``, ``seed=20260924``, full train rows,
  ``chunk=16384``. No K / sparsity / epoch search.
* IHT: ``v0.tied_iht_codes``, 10 steps, exact top-4 each step.
* ``D_S`` is projected onto the common-1 orthogonal complement and column
  normalised at use (identical hard constraint to CSSD); ``D_C`` is column
  normalised at use.

The model is the frozen ``SEM108Model`` with coordinate width 33 and the
**same** binding widths as the Sem108 baseline (``W_A_S: 33×48``,
``W_E_S: 99×48``); only ``code()`` changes. No new module, no width change, no
nonlinearity, no message passing, no recurrence.

### 5.1 Baseline A (``TOPO``) — frozen pure-topology coordinate

Exactly the frozen Sem108 dictionary route, with the historical dictionary
**frozen**:

```
z_A = [ c~ (1) ; IHT10( colnorm((I-UU^T) D_SDB), (I-UU^T) phi ) (32) ]  in R^33
```

``D_SDB`` is the frozen SDB-v0 K-SVD dictionary (``K=32``, ``s=8``). It is not
refit in this round, so A is *not* a historical end-to-end number: it is the
same frozen-dictionary protocol as B.

### 5.2 Candidate B (``CORR``)

Section 5 with ``D_S`` / ``D_C`` frozen. Readout parameter shapes are identical
to A; the two arms are initialised from the **same readout state** (the A
initialisation is copied into B and the bit-identity is verified and recorded),
so the only difference at step 0 is the coordinate content.

### 5.3 Forbidden

No refinement of ``U``, no dictionary fine-tuning / task coupling, no
``lambda_rec`` reconstruction term (frozen dictionaries make it inert; it is
set to 0 and the inertness is checked), no Sem108 / backend / reader change, no
new descriptor, no message passing, no ``alpha_C``-only bypass into the reader,
no extra width or heads.

---

## 6. Training protocol (frozen, inherited from Sem108)

* data: official ZINC train 10000 / official valid 1000; **official test never
  loaded** (non-terminal control-plane mode enforces this).
* seed 0; CPU; 8 torch threads.
* Adam ``lr=1e-3``, ``weight_decay=1e-5``, batch 128, grad clip 5.0.
* loss: task L1 (+ ``0.0 ×`` frozen reconstruction diagnostic).
* 320 epochs, no early stop.
* Top-5 soup by valid MAE over the 320 epochs; best checkpoint as well.
* data order: the frozen Sem108 loader seeds (``seed + TRAIN_SHUFFLE_OFFSET`` /
  ``seed + EVAL_SHUFFLE_OFFSET``).
* arms A and B run in one runner invocation; identical loaders, optimizer
  schedule and soup rule. No tuning of anything.

---

## 7. Correctness checks (all before training; any FAIL ⇒ STOP)

* **G0 construction identity** — ``C_g`` matches an independent float64 numpy
  hand computation for sampled patches (node shell 1/2, edge shellpair).
* **G1 invariance** — within a molecule, a random node relabelling and edge
  endpoint exchange leave ``C`` unchanged (float64 tolerance ``1e-9``).
* **G2 non-triviality** — a within-group attribute permutation keeps the coarse
  chemistry marginals ``Σ_g b`` / ``Σ_g q`` (to ``1e-12``) and keeps the role
  and attribute multisets, but changes ``C`` (nonzero delta on at least one
  sampled patch).
* **G3 zeros** — empty / singleton groups and constant-role columns produce
  exactly zero blocks; shell-0 node groups are always zero.
* **G4 label-free** — the cache builder and the object builder are AST-checked
  to reference no ``y`` / target; the C cache is built from topology and
  attributes only.
* **G5 scaler scope** — the scaler is fit on official train only, valid is not
  in the fit; RMS/mask statistics and effective-coordinate counts recorded.
* **G6 dictionary shapes / energy** — ``D_S: 65×16``, ``D_C: 536×16``, columns
  unit norm, finite; K-SVD final MSE recorded; IHT codes have exact
  ``l0 <= 4``.
* **G7 coordinate layout / freezing** — coordinate width 33 with the frozen
  slices; ``D_S``/``D_C`` ``requires_grad=False``; no trainable parameter of
  the coordinate path exists.
* **G8 shared readout initialisation** — every readout parameter of A and B is
  bit-identical at step 0 and the trainable parameter count is identical.
* **G9 zero-``alpha_C`` purity** — zeroing only ``alpha_C`` leaves the other 17
  coordinate columns bit-identical and does not touch Sem108 or any other
  input.
* **G10 official-test blocker** — any ``official_test_loaded`` payload with
  anything but ``false`` raises; the test split is never instantiated.

Plus a smoke run (8 epochs × 1024 train / 512 valid = 64 steps) per arm:
finite loss, non-zero gradients in the coordinate path and the readout,
C perturbation changes the prediction.

---

## 8. Frozen inference probes (soup state only, no retraining)

| probe | content |
|---|---|
| ``M_A``, ``M_B`` | soup valid MAE of A and B |
| ``M_C0`` | B soup, ``alpha_C → 0`` (common + ``alpha_S`` untouched) |
| ``M_S0`` | B soup, ``alpha_S → 0`` (common + ``alpha_C`` untouched), diagnostic |
| ``M_A0`` | A soup, ``alpha32 → 0``, diagnostic |
| ``M_Cshuf(s)`` | B soup, valid ``C`` replaced by the within-group attribute-permuted object (5 seeds 101/202/303/404/505), same scaler and same ``D_C`` |
| per-molecule errors | paired A/B per-molecule absolute errors on official valid |

``G_C0 = M_C0 − M_B``, ``G_Cshuf = mean_s M_Cshuf(s) − M_B``.

---

## 9. Frozen screening gate and decision

Primary gate (2 %, a screening rule only — **not** a significance claim):

```
relative = (M_A − M_B) / M_A
PROCEED  iff relative >= 0.02     (equivalently M_B <= 0.98 * M_A)
```

* If **not** met: record A/B curves, best and soup MAE, per-molecule
  difference, ``M_C0``, ``M_Cshuf``; verdict
  ``ROLE_CORR_NO_MATERIAL_GAIN``; STOP. No width / horizon / seed /
  regularization rescue; do not run the controls.
* If met: run the two frozen controls with the identical protocol (seed 0):
  * **C — shuffled correspondence** (``CORR-SHUF``): train and evaluate with
    within-group attribute-permuted ``C`` (same scaler, same patch order),
    refit ``D_C`` on the shuffled train object only; everything else identical.
  * **D — PCA16** (``CORR-PCA``): replace ``alpha_C`` by the affine PCA16 code
    of the same real scaled ``C`` (mean + 16 components fit on official train),
    keep ``common`` + ``alpha_S`` and everything else identical.
* Answers (frozen wording):
  1. ``B < C`` and ``G_C0 >= 0.003`` and ``G_Cshuf >= 0.003`` ⇒ the
     correspondence information is worth keeping (report whether it is
     directional or clear; 0.003 is a direction check, 0.010 is clear).
  2. ``M_B <= M_D`` (or ``M_B < M_D`` beyond the run-to-run noise scale) ⇒
     sparse-dictionary advantage; otherwise explicitly report
     *object valuable, sparse-dictionary advantage not supported*.
* Verdicts: ``ROLE_CORR_SCREEN_PROCEED`` (A/B stage),
  ``ROLE_CORR_SUPPORTED_SPARSE_DICT`` / ``ROLE_CORR_SUPPORTED_NOT_SPARSE_SPECIFIC``
  (controls). No multi-seed in this round; no end-to-end dictionary tuning.

---

## 10. Stop discipline and reporting

One runner invocation for A + B; a second invocation only for C + D if the
gate fires. No seed 1, no horizon/width/lr/K/s change, no official test, no
reuse of historical numbers as if they were matched. Every output records
``protocol_version``, ``git_commit``, ``device``, ``seed``,
``official_test_loaded=false`` and the data/split fingerprints via the control
plane. Deliverables: this pre-registration, the core module, the stage runner,
the control-plane runner + config, focused CPU tests, the result directory
(curves, checkpoints, probes, per-molecule errors), ``REPORT.md`` and
``DECISION.md``, plus a STATE/claim/decision update.
