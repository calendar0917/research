# E2E-DictEnv-A2 — compute-budget amendment (user-requested contraction)

Status: **amendment to a frozen round**.  The parent preregistration
(`tracks/ksvd/notes/e2e_dictenv_a2_preregistration.md`, commit `1813f53`) is
**not edited** by this document and stays in force for everything it froze.

---

## 1. Why this amendment exists

* **Reason**: the user requested a *compute-budget contraction* of the
  running `E2E-DictEnv-A2` round (explicit instruction, 2026-09-25): "对当前
  正在执行的 E2E-DictEnv-A2 做一次计算预算收缩 … 请优先保证科学记录完整，同时
  停止不必要的后续计算".
* **Not motivated by**: an observed validation result, an architecture
  performance reading, or a desired outcome.  The amendment was requested
  without reference to any stage verdict, and it changes **no** scientific
  gate, threshold, endpoint or decision rule of the parent round.  Its only
  scientific object is a *smaller question* asked in a *separately labelled*
  protocol.
* **Effect**: truncate the original A2 execution plan at the next safe
  experimental boundary and continue only the minimum paired screen required to
  answer the primary question — *does the real structure↔attribute pairing
  (`REAL`) have task value over the assignment-independent control (`INDEP`)
  under the frozen exact-OMP code?*  Everything else is
  `DEFERRED_PENDING_USER_AUTHORIZATION`, not cancelled.

Device discipline is unchanged and remains hard: **physical GPU1 only**
(`CUDA_VISIBLE_DEVICES=1`), no GPU0, no DDP, no multi-GPU, no parallel CUDA
processes, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`.

## 2. Metrics already observed before this amendment

The lite continuation is **not blind**.  At the moment of writing, the
following numbers had been produced by the running A2 round (commit
`24d5286`) or were already in the repository.  They are listed so that the
post-amendment path is never described as a fully blind preregistration.

Already observed (A2, remote GPU1, complete artifacts):

| what | observed | artifact |
|---|---|---|
| A2 artifact identity | `all_passed = true`, 26/26 entries, 130 s; 3 dictionary sha pins, 10 cache sha pins, 2 scalers, 6 exact-OMP recomputations, matched-init no-op | `results/e2e_dictenv_a2/artifact_identity.json` |
| Gate-0 correctness | 19/19 | `correctness.json` |
| Gate-0 assignment | REAL joint max abs change `3.6243410110473633`; INDEP `0.0`; marginals preserved | `assignment_semantics.json` |
| Gate-0 health (holdout normalised reconstruction) | TOPO `1.410e-05`, INDEP `2.109e-02`, REAL `1.971e-01`; used atoms 32/32/31; effective 4.03/18.59/19.99; usage Spearman 0.9989/0.9989/0.9993 | `dictionary_health*.json` |
| Gate-0 accounting + official-test blocker | pass; official test never loaded | `parameter_accounting.json` |
| continuity-v2 (diagnostic, pooled) | REAL x `0.2971` / code `0.2464`; INDEP x `0.2750` / code `0.4014`; TOPO x `0.1350` / code `0.1329`; strata small/medium/large as in the implementation note; A1 Path-D reconciliation bit-exact (REAL x 0.28699 / code 0.26007; INDEP 0.28085/0.35470; TOPO 0.12823/0.13306) | `continuity_v2.json` |
| A2 plumbing smoke (GPU1) | all curves finite, OMP dictionaries frozen, IHT dictionary gradient `6.399` | `smoke/smoke.json` |
| frozen Stage-1 TOPO arm (`T0`) | in flight during the amendment; its per-epoch log lines were visible. TOPO is excluded from the lite screen, so it cannot inform `REAL` vs `INDEP` | `results/e2e_dictenv_a2/` (boundary record, §5) |

Already in the repository before the amendment (parent / earlier rounds):

* A1's Gate-0 continuity verdict: REAL code-space AUC `0.534375 < 0.70`
  (INDEP code AUC `0.705625`) — the frozen reason for A1's
  `REPRESENTATION_NOT_QUALIFIED`.
* A1's `decision.json` records `stage1_omp = stage2_coder = stage3_e2e = null`:
  **no REAL-vs-INDEP task comparison existed anywhere before this amendment**.
* A1's attributed-WL continuity numbers (Path R / Path D), reconciled bit-exactly
  by A2's continuity-v2 diagnostic.

Not observed (and not inferable) at amendment time: any `REAL`/`INDEP`
prediction value, the Stage-1 `G_pair_OMP`, any coder/E2E/mechanism number.

Note on the direction of prior evidence: the only pre-existing task-side reads
(A1 Gate-0, continuity-v2) are *code-similarity* diagnostics, and in those the
INDEP code space looked **more** continuity-preserving than REAL.  The lite
screen's decision table is symmetric (`REAL` better ⇒ more compute justified;
INDEP better ⇒ stop), its threshold (`0.006`) and the dense fallback bar
(`0.003`) are the user's instruction, and no gate was chosen after seeing a
lite-screen value.

## 3. What is truncated, and what stays valid

* The frozen round is recorded as
  `E2E-DictEnv-A2: COMPUTE-BUDGET-TRUNCATED`.
* Every artifact the frozen round already completed stays **valid and
  unchanged** (identity, qualification, continuity-v2, and the completed
  frozen Stage-1 TOPO arm).  Nothing is deleted, re-run or edited; the local
  copy is wiped before pulling so a local pass can never mask a remote failure.
* Stages the frozen order had not reached are recorded as
  `NOT RUN — deferred by user compute-budget amendment`, never as a result.
* Reuse is by *reference*: the lite round re-reads the completed A2 identity
  record and re-checks the cheap irreversible digests (dictionary array shas,
  OMP cache bytes, sha256 of the A2 producer modules and of the frozen parent
  preregistration).  It **never** recomputes the A1 433-D caches, the scaler,
  the K32/s8 dictionaries or the exact-OMP codes.

## 4. Lite continuation protocol (frozen *before* the screen ran)

Identity separation is mandatory: the lite continuation is a **different
protocol** and may never be reported as the frozen A2 Stage 1.

| field | value |
|---|---|
| round | `E2E-DictEnv-A2-Lite` |
| protocol id | `e2e_dictenv_a2_lite` |
| parent round | `E2E-DictEnv-A2` (protocol `e2e_dictenv_a2`), parent commit `24d528635fab49c082115d90592cb7d5938eeb37`, parent prereg commit `1813f53` |
| amendment / prereg commit | the commit that introduced this note and the lite modules (chain recorded in `lite_verify.json`) |
| result directory | `tracks/ksvd/results/e2e_dictenv_a2_lite/` (own decision + report artifacts) |
| decision record | `lite_screen_decision.json`, `lite_pca_decision.json`, `lite_report.json`, `DECISION.md`, `REPORT.md` |
| arms | `REAL`-OMP and `INDEP`-OMP only.  **No TOPO arm.** |
| horizon | **160 epochs** (screen), same H1 trainer: same seed 0, same init, same batch order (`SEED + 91011` / `+ 91012`), same optimizer/LR/weight-decay/clip, same λ, same Top-5 soup rule, same frozen dictionary/codes, same physical GPU1, one process at a time |
| official test | never loaded |
| smokes/sweeps | none; no K/s/radius/feature/optimizer sweep; no seed hunting |

### 4.1 Frozen screen rule

```
G_pair_screen = MAE(INDEP-OMP) - MAE(REAL-OMP)          (Top-5 soup, valid split)

late window    = epochs 121..160 (paired)
direction      = delta(e) = MAE_INDEP(e) - MAE_REAL(e)
direction_stable = (fraction of delta(e) > 0) >= 0.75
                   and mean(delta(e)) > 0
                   and (min over window MAE_INDEP) - (min over window MAE_REAL) > 0

strong_positive = curves_finite and G_pair_screen >= 0.006 and direction_stable
```

* `strong_positive` ⇒ verdict `PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION`:
  **stop**; the 320-epoch confirmation, the IHT coder, the task-coupled E2E
  dictionary, the mechanism and the DenseTied specificity control are *not*
  started; the user decides whether to buy more compute.
* `G_pair_screen < 0.006` or an unstable direction ⇒ only the cheap dense
  diagnostic below is authorised.
* unusable (missing/non-finite) curves ⇒ `LITE_SCREEN_UNRESOLVED_PENDING_REPAIR`:
  stop and report; the dense route is not authorised on broken evidence.

### 4.2 Frozen cheap diagnostic (only when the screen is not strong)

```
ATTR-INDEP-PCA32 vs ATTR-REAL-PCA32
train-only fit (official train only), rank 32, same normalised 433-D object,
same H1 trainer, same seed, same horizon 160, same GPU1,
PCA delegated to sdb_v0.fit_pca_rank through e2e_dictenv_a2.fit_dense_rank
G_pair_PCA = MAE(INDEP-PCA32) - MAE(REAL-PCA32)
```

Frozen labels:

| condition | label |
|---|---|
| `G_pair_screen >= 0.006` and stable | `PAIRING_SIGNAL_WORTH_FULL_CONFIRMATION` |
| screen weak/unstable and `G_pair_PCA >= 0.003` | `ATTRIBUTED_PAIRING_SUPPORTED_COMPRESSION_BOTTLENECK` |
| screen weak/unstable and `G_pair_PCA < 0.003` | `NO_ATTRIBUTED_CODE_FORMATION_SIGNAL_AT_32D` |
| curves unusable at either step | `LITE_SCREEN_UNRESOLVED_PENDING_REPAIR` |

The lite screen is a *continue/stop* screen: a 160-epoch value is never
presented as the frozen A2 320-epoch Stage-1 claim, and the parent round's
`MATERIAL = 0.003` gate is untouched.

### 4.3 Deferred (never auto-started)

TOPO-OMP predictor baseline; IHT-10/30/100/200 coder qualification;
task-coupled E2E sparse dictionary (E0/E1/E2); REAL→INDEP code-pairing-removal
mechanism; node-only / edge-only interventions; DenseTied specificity control;
seed 1; official test; K64; s12; any architecture or hyper-parameter sweep.
All are `DEFERRED_PENDING_USER_AUTHORIZATION`.

### 4.4 Stop rule

The lite round stops after `DECISION.md`/`REPORT.md` exist and the results are
pulled, analysed and recorded locally.  It does not enter any further stage on
its own, whatever the label.

## 5. A2 truncation boundary (appended after the boundary was reached)

See the `A2 truncation boundary` section of
`tracks/ksvd/results/e2e_dictenv_a2/README.md` for the byte-level boundary
record (completed artifacts, partial checkpoints — there are none, because the
frozen `train_arm` writes state/curve/JSON only after the final epoch — and the
exact stop point).  This section is appended in a later commit than §1–§4 and
changes no rule: the protocol that the lite run executes was frozen in the
commit that first introduced this document.
