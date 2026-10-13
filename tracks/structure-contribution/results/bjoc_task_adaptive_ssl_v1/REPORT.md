# BJOC task-adaptive SSL v1 — final report

* Protocol: `configs/bjoc_task_adaptive_ssl_v1.yaml` (`cscl-bjoc-task-adaptive-ssl-v1`), frozen 2026-10-13 (commit `388c9a7`, remote prep on `res`).
* Track: `tracks/structure-contribution`.
* Status: **complete**. Frozen decision: **`TASSL_NO_GAIN`**. The round stops here: no compute control, no Locked read, no second λ/frequency, no third seed, no full-10k refit.
* GPU spend: **2.537 of 3.0 GPU-hours** (`gpu_budget_ledger.json`).
* The historical best model (BJOC-Null + masked SSL + R3) was the background reference only; no official-valid or official-test number was produced by this round.

## 1. Question

Under the same pretrained initialisation and the same supervised training steps, does keeping a low-frequency masked-atom self-supervised auxiliary task alive during supervised training (**TA-SSL**) improve held-out generalisation over the plain P3 training programme? The three separations the round was designed to keep apart: (1) SSL-as-initialisation is already known to help; (2) does a continuous structural-recovery task constrain supervised training; (3) is any gain only extra computation.

## 2. Data access and provenance

* **Official test was never loaded** in any form (no code path; `official_test_loaded: false` in every artifact). **Official valid was not loaded** in this round either.
* Split D is an internal group-aware re-partition of the official ZINC train 10 000, pre-registered (as a fingerprint payload) but never purchased in the architecture round. This round verified the committed architecture-round config fingerprints (`matches_pre_registered_arch_config: true`), zero duplicate groups crossing sides, and scanned all D-tagged artifacts: **no prior D objects, SSL, training, Monitor or Locked read existed** (`prior_d_training_detected: false`).
* Split D **overlaps the previously used A/B splits** (vs A: 6 389 Fit / 806 Monitor rows already in the previous Fit side, 1 611 Fit / 194 Monitor / 195 Locked rows in previously read zones; vs B: 6 388 / 805 / 807 and 1 612 / 195 / 193). This is disclosed, not hidden: D is **not an independent external test split**, absolute MAE levels are not comparable to official-split numbers, and only within-split matched contrasts are interpreted.
* `ab.LabelGuard` made Monitor/Locked `.y` unreachable during preprocessing, SSL, training and prediction. Object-build alignment check is Fit-only (max |Δy| = 0.0); `forbidden_label_access_attempts = 0` in `objects_manifest_D.json`, both SSL runs and all four arms. Fit-only standardisation; Fit raw-y supervision only (`labels_used: "train Fit rows raw y only"`).
* Label reads: exactly **one declared Monitor read** (`monitor_read_D.lock`, 2026-10-13T18:24Z, y sha `bbcf6faf…`, 1 000 rows), after all 8 prediction files were frozen and hash-manifested (`prediction_manifest_D.json`; the lock records its sha). **Locked was not read**: the frozen purchase rule requires a Monitor gate pass and the gate failed. `decision.json` records `locked_read_authorized: false`; no `locked_read_D.lock` exists.

## 3. Frozen designs and controls

**P3 (reference)** — BJOC-Null (413 505 params), split-D Fit-only masked atom+bond SSL (7 560 updates, batch 128, Adam 1e-3/wd 1e-5/clip 5), then R3: 22 680 supervised updates, raw-y L1, per-step cosine 1e-3→1e-5, last-5 epoch-state soup.

**TA-SSL (candidate)** — same per-seed SSL encoder and same atom head captured from that same SSL run; during the identical R3 trajectory, every 4th minibatch (`t % 4 == 0`, 5 670 auxiliary steps) adds an atom-only masked reconstruction loss λ = 0.1:

    L_t = L_property + 0.1 · CE(h_atom(E_θ(masked view)), hidden atom type)

The masked view is built from the **same Fit molecules of the current batch** with the frozen `mp.build_masked_sample` path (15% atoms / 15% bonds, full rebuild); the atom context is the same static-incidence formula as the frozen `PretextModel`; one joint optimizer step per minibatch; joint clip 5; the R3 schedule, batch order and last-5 soup rule are P3's; the auxiliary head (66 108 params) is removed at inference, which is exactly BJOC-Null 413 505 params. No search after the freeze.

## 4. CPU Stage A gates (before any GPU trajectory)

All passed on `res` (`stage_a.json`, commit `388c9a7`):

* **A1 frozen-loop replay.** Same Fit-D rows, seed 0, initialisation, batch order and schedule: batch-order hash **identical** (`full_matches: true`), init full hash identical, LR values at every epoch exactly equal, train-MAE curve within the machine noise floor, soup within the noise floor (new-vs-bp max abs diff `1.27e-7`; bp-repeat max abs diff `1.27e-7`; tolerance `1e-6`; `equivalent_within_noise: true`). On the local workstation all four soups (bp ×2, adaptive-loop ×2) were **bitwise identical (max abs diff 0.0)**.
* **A1 amendment (disclosed).** The original frozen wording required bitwise soup identity. The remote CPU is run-to-run non-deterministic at ~1e-7 in float32 parameters **for the same frozen loop** (two `bp.train_supervised` runs differ; the M4 local workstation is bitwise). The gate instrument was therefore changed to the noise-floor comparison above; the change is recorded in `post_freeze_amendments.json`, was made before any GPU trajectory and changed no model, recipe, auxiliary schedule, λ, seed, threshold, label rule or budget.
* **A2 SSL wrapper and masked-view audits.** The head-capturing wrapper round the frozen `bp.pretrain_masked` is **encoder-bit-identical** to it and reproduces its loss steps; hidden-truth mask-swap invariance 100/100 atom + 100/100 bond cases; masked-rebuild leak audit 50/50 cases (`MASKED_REBUILD_LEAK_FREE`).
* **A3 gradient diagnostics** (recorded, never used to tune λ). At the SSL init on real Fit minibatches: g_atom/g_property norm ratio 0.369 (seed 0) / 0.403 (seed 1), cosine −0.025 / +0.012, head active, all finite. After two short epochs: ratio 6.71, cosine +0.15 — the atom task is a small, mildly conflicting gradient at the start, not a dead or dominating channel.
* **A4 invariance and inference identity.** Relabel / bond-endpoint swap / batch membership / save-load replay max diff 8.2e-8; inference path parameter audit exact 413 505; no auxiliary-head attribute on the exported model.

## 5. Stage B training (split D Fit, matched per seed)

| arm | seed | SSL encoder (sha) | head (init sha) | updates | supervised wall | soup Fit MAE |
|---|---|---|---|---|---|---|
| P3 | 0 | `2c2bbd3f…` | — | 22 680 | 538.7 s | 0.001082 |
| TA | 0 | `2c2bbd3f…` | `78805d15…` | 22 680 (+5 670 aux) | 2 291.0 s | 0.000995 |
| P3 | 1 | `655633c6…` | — | 22 680 | 583.6 s | 0.001077 |
| TA | 1 | `655633c6…` | `7a0dd83d…` | 22 680 (+5 670 aux) | 2 318.4 s | 0.001019 |

* Per seed, P3 and TA share the same SSL encoder hash; TA's head is the same SSL run's head; P3/TA batch-order hashes are identical per seed and match an independent recomputation. All arms verified the frozen code hash and the frozen config hash at launch.
* SSL quality (unchanged, as expected): atom acc 0.929 / 0.930, bond acc 0.9955 / 0.9944.
* Auxiliary task is alive throughout: 5 670/5 670 scheduled updates, 720 000 masked molecules seen per seed, atom CE 0.241→0.197 (seed 0) and 0.274→0.201 (seed 1), atom acc 0.913→0.925 / 0.913→0.925. Auxiliary head max gradient norm 0.075 / 0.070; clip events nearly identical to P3 (10 026 vs 9 754; 11 968 vs 11 651), so the property optimisation regime is essentially unchanged.
* **Extra compute cost:** TA-SSL supervised wall time is **≈4.2× P3** (2 291 s vs 539 s per seed); 1 672 s / 1 681 s (73%) of supervised time is inside the auxiliary branch (5 670 extra encoder forward/backward passes plus 720 000 masked-molecule rebuilds). GPU-hour total 2.537 h (limit 3.0).

## 6. Monitor results (split D, 1 000 molecules, one read, seed-paired)

Baseline P3 seed-mean Monitor MAE **0.104301** (0.103960 / 0.104642); TA **0.104384** (0.104176 / 0.104591).

| quantity | value |
|---|---|
| Δ = MAE(TA) − MAE(P3), seed-mean | **+0.000083** |
| per-seed Δ | +0.000216 / −0.000051 |
| paired molecule-bootstrap 95% CI | **[−0.005414, +0.004620]** |
| molecules better | 47.4% |
| ordinary (y > −3; n = 929) Δ | **+0.002797** (worse) |
| tail (y ≤ −3; n = 71) Δ | **−0.035439** (better) |
| RMSE Δ = TA − P3 | **−0.055235** (better) |
| top-10-error-removed Δ | +0.000705 (worse) |
| largest single-molecule gain share | 0.158 |

Frozen gate: seed-mean ≤ −0.004 **fail**, both seeds negative **fail**, CI upper < 0 **fail**, ordinary not worse **fail**, top-10-removed negative **fail**, RMSE within +0.005 **pass**, no single-molecule dominance **pass**, audits **pass** → **5/8 conditions fail**. Decision vocabulary: seed-mean Δ ≥ 0 → **`TASSL_NO_GAIN`**. Per the frozen purchase rules, **no compute control was run and Locked was not read**.

**Descriptive decomposition** (same single read, no new decisions): per seed the pattern is identical in sign — s0 ordinary +0.00412 / tail −0.05084 / RMSE −0.09095; s1 ordinary +0.00148 / tail −0.02004 / RMSE −0.02353. Ordinary MAE rises 0.08042→0.08321 (+3.5% relative) while tail MAE falls 0.41682→0.38138 (−8.5% relative). The large-error rows move strongly in TA's favour (the 10 removed rows Δ −0.202), but the effect is not limited to them: after removal the remaining tail still improves (−0.00747, n = 66) while the remaining ordinary rows still worsen (+0.00281). 239 molecules improve under both seeds vs 255 that worsen under both. So TA-SSL **trades bulk accuracy for robustness on extreme labels**, with a net seed-mean of ≈0.

## 7. Answers to the paper-facing questions

**(A) Does TA-SSL improve over P3, and by how much?** No, not in the pre-registered sense. The seed-mean Monitor Δ is **+0.000083** (TA slightly worse), the CI95 [−0.00541, +0.00462] includes zero and excludes any gain of the required size (≤−0.004), one of two seeds is worse, ordinary molecules are worse by +0.00280, and the top-10-removed contrast is also worse. The mechanism shifts error toward the tail rather than reducing total error.

**(B) Seed consistency.** The overall sign is seed-flipped (+0.00022 / −0.00005), i.e. **not seed-consistent as a gain**. The *subgroup* effect is seed-consistent: both seeds worsen ordinary molecules (+0.0041 / +0.0015) and improve the severe tail (−0.0508 / −0.0200) and RMSE (−0.0910 / −0.0235). So the honest reading is a seed-stable **trade**, not a seed-stable improvement.

**(C) Extra compute cost.** TA-SSL costs **≈4.2× the supervised wall time** of P3 per seed (2 291 s vs 539 s; 2 318 s vs 584 s), of which ~73% is the auxiliary branch; it does 5 670 extra encoder forward/backward passes and 720 000 masked-molecule rebuilds per seed. For a Δ ≈ 0 general result, this is a pure cost. Even had the gate passed, the compute control (P3-EXTRA, ≈0.19 GPU-h/seed) would have been the next required purchase.

**(D) Evidence strength.** This is a small-budget, single-mechanism, two-seed matched comparison on an internal re-partition that is **not an independent external test**; Monitor only (Locked unread). Within those limits the controls are strong (shared encoder/head/init, identical batch order and schedule, exact inference-parameter identity, CPU mask/gradient audits, single frozen read, no search after the freeze). The evidence is **strong against a ≥0.004 general gain** and against ordinary-molecule improvement; it is **insufficient to exclude small gains/losses within the ±0.005 CI**, and the tail/RMSE effect is descriptive and would need its own pre-registered hypothesis (tail metric, tail-matched baselines) to be treated as real.

**(E) Stop ZINC search and move to a second molecular task?** Within this round: **yes, stop** — the frozen rule says so and nothing in the data justifies another λ, frequency, mask-ratio, head-width or seed. Overall: the accumulated evidence now covers three mechanism families on the same BJOC/Masked-SSL/R3 system (shared-type residual, GLRC global gating, DLP distance products, and now task-adaptive SSL), and **none improves ordinary-molecule held-out error**. The remaining unexplained signal is the tail: 71 of 1 000 molecules carry ~0.42 MAE, and TA-SSL halves part of that at the cost of the bulk. The recommended next step is therefore **not** another ZINC variant but a **second molecular task** (different property/endpoint) under a fresh preregistration, which also directly tests whether the BJOC representation transfers. A ZINC tail-specific hypothesis (e.g. robust/quantile objective at matched compute) is the single listed alternative, but it is explicitly **not authorised or started** here.

## 8. Round decisions and artifacts

* Decision: **`TASSL_NO_GAIN`**; stop per preregistration. Not purchased: compute control `P3-EXTRA`, Locked read, second λ/frequency, third seed, full-10k refit.
* Key artifacts: `input_audit.json`, `split_D.json`, `objects_manifest_D.json`, `prep_fit_D.json`, `protocol_freeze.json`, `init_parity.json`, `stage_a.json`, `post_freeze_amendments.json`, `pretrain_D_s{0,1}.json`, `pretrain_head_D_s{0,1}.pt`, `gradients_ssl_real_s{0,1}.json`, `train_{P3,TA}_D_s{0,1}.json`, `soup_{P3,TA}_D_s{0,1}.pt`, `prediction_manifest_D.json`, `predictions/*.npz`, `monitor_read_D.lock`, `monitor_TA_D.json`, `monitor_rounds_D.json`, `decision.json`, `gpu_budget_ledger.json`.
* All cross-artifact references were re-verified after the run: freeze code hashes match the working tree, encoder/head/batch-order hashes are consistent per seed, all 8 prediction files match the manifest, the Monitor lock binds the manifest hash, and no Locked artifact exists.
