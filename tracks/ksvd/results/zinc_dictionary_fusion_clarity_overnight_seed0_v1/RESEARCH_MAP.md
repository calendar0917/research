# RESEARCH_MAP — ZINC dictionary responsibility split (as of 2026-10-06, round `zinc_dictionary_fusion_clarity_overnight_seed0_v1`)

## 1. The live architecture and where each dictionary sits

Body (identical in every arm of the current family; `prev.LocalTupleFull` /
`mlpmod.LocalTupleFullM` / this round's F arms):

```
phi65 (per root) ──┐  q60 (per tuple) ── local encoder (D_loc | A_raw | D_S+A?S/B) ── e(v) [64]
Sem108+size2 [110] ── fusion MLP (110→114→342) ──(+ W_loc @ (kappa*e))── hidden → fusion[1]/[2]
   → posterior bridge: matched MLP 144→288→144 (zw.MLPBridge; replaced the old D_L/V_L linear bridge)
   → reader 39→2 (COMP half-split) → hat_ell + hat_s = hat_g
node/edge encoded slots (3×48/6×32), static relations, topology25, C6 mask: frozen, unchanged
```

| dictionary object | input / shape | loss it trains under | status this round |
|---|---|---|---|
| old node/edge slot dictionaries + pair binding | frozen slots | historical; dead in this body family | closed (repeated rounds of evidence; not re-purchased) |
| posterior `D_L`/`V_L` linear bridge | 144-D h | replaced by matched MLP 144→288→144 in **both** arms of the source round | dead (verified in `src.build_arm_comp`, both arms `zw.MLPBridge`) |
| joint tuple dictionary `D_loc` [125,64] (J_D) | 125-D joint structure+semantics tuple, tied IHT top-8 | COMP (this round, contemporaneous control) | active mechanism, no advantage vs matched MLP (source round `DIRECTIONAL_NOT_CONFIRMED`) |
| structure dictionary `D_S` [65,64] (F_D, this round) | phi65 alone, tied IHT top-8, fused per tuple with explicit semantic branch `B` [64,60] | COMP | the round's new hypothesis — **not yet a located cause of anything** |
| `A_raw`/`A_S` SiLU projections | matched-parameter MLP controls | COMP | controls, not dictionaries |

Input facts (verified from source code, `parallel_evidence.json` → `responsibility_map`):
- **phi65 is genuinely pure topology** (`fsar_r2_ar0.phi_for_center`: root indicator, shell one-hots, log1p degree, per-shell neighbour counts, walk counts, node/edge basis mean/std, size logs; only adjacency + rooted distances).
- **Sem108 is structure-conditioned semantics, not pure global chemistry**: atom_shell (84 = shell-resolved atom counts) + bond_shell (24 = shell-resolved bond-type counts); the fusion input is [Sem108; size2] = 110.
- The only chemistry inside the joint 125-D tuple is q60 = [onehot28(root atom); onehot28(neighbour atom); onehot4(bond)].
- J/I incidence weights (`w_J = C/d`, `w_I = n_t·n_a/d²`) enter only the local tuple pooling; the static relations enter through the frozen node/edge slots + C6 mask; neither is touched this round.

## 2. Established facts (each with its evidence object)

1. Source round's numbers, recomputed independently from saved predictions
   (`parallel_evidence.json` → `unified_recomputation`; matches
   `…/zinc_local_dictionary_component_supervision_seed0_v1/dev_eval.json`):
   D_COMP dev G0 cal 0.090108 / raw 0.099279; M_COMP 0.088062 / 0.090340;
   gaps (M−D): G0 raw −0.008939, G0 cal −0.002045, overall raw −0.007946,
   overall cal −0.001138; raw CIs below zero, cal CIs cross zero.
2. **Retraction**: the earlier "the gap is not bias-related" phrasing was too
   strong. D's fit-median bias (−0.0428) is nearly twice M's (−0.0225); after
   each arm's own calibration the raw G0 gap shrinks ~4.4× (0.0089 → 0.0020).
   Part of the raw deficit is a fit-side median offset. The calibrated
   comparison (still MLP-favouring, CI crossing 0) is the honest endpoint.
3. The dictionary mechanism is genuinely alive in the source round
   (`mechanism_health.json`): 8.0 nnz/tuple exactly, 61/64 atoms used,
   root-code effective rank 61, injection RMS 0.415, live task gradients.
4. The dictionary changed shape, but moderately, not a reshuffle
   (`parallel_evidence.json` → `dictionary_change`): raw drift 71.5 vs init L2
   88.8 (relative 0.81 — a raw-matrix distance, **not** a unit-free progress
   metric); column-normalised same-index change 6.02; best-|cos| to *its own*
   init atoms mean 0.719 / median 0.756, none > 0.9, zero sign flips. Atoms
   keep moderate same-index shape similarity — the "drift 71.6" number alone
   overstates the qualitative change.
5. **ERRATA (this round)**: the source round's `cal_g_identity_max_abs`
   (6.60 / 4.11 in its `dev_eval.json`) was a sign-convention bug in the *check*
   (`err − (e_ell+e_s)` with e = hat−target). With the brief's convention
   (e = target−prediction) the identity `e_g_raw = e_ell + e_s` is exact
   (≤ 2.4e-7, float32). No historical number or conclusion changes; the check
   is fixed in this round's module. The `cancellation` diagnostics were
   sign-symmetric and are unaffected.
6. Code sparsity / reconstruction / effective rank / task gain are separate
   axes (health at 1/40/120/240 per arm, this round's `mechanism_health.json`):
   none of them implies any other.

## 3. Conditionally supported (bound to its exact comparison)

- "The matched MLP is better in this interface" — only within the source
  round's fixed 125-D joint tuple / J incidence / seed 0 / single fold / COMP
  interface; it does **not** license "all dictionaries lose to MLPs".
- "The joint 125-D fixed recipe is closed" — a resource decision about one
  fixed configuration, not a verdict on the dictionary direction (source
  DECISION.md, respected this round: no parametric rescue, no sweeps).
- The luyin19 hypothesis (separate structure coding + explicit semantic
  fusion) is *motivated but untested* until this round's Stage-1 dev scores
  land; the F−J difference is an information-division/function-class
  comparison, not a "pure fusion effect".

## 4. Undetermined (what this round can and cannot decide)

- Whether separating structure from semantics helps at all (Stage 1: C_D/C_M),
  and whether any dictionary increment exists in either construction
  (G_J/G_F).
- Whether the local interface even matters relative to the body channel
  (zero-injection collapses both arms to ~0.49–0.55 in the source round — the
  channel carries signal, but its share vs the body is unknown).
- Single seed, single fold per stage; Stage 3's split B checks fold
  robustness only; nothing here addresses training-seed uncertainty or
  unseen-chemistry transfer.

## 5. Closed configurations (do not re-purchase without a new preregistration)

Node revival; ring-tail weighting; posterior bridge/reader variants; the
fixed joint-IHT 125-D tuple recipe parametric family (K/s/steps/width/WD);
seed ensembles; dev-selected epochs; the old D_L/V_L linear bridge; joint
125-D parametric sweeps of any kind.

Evidence pointers: `tracks/ksvd/results/zinc_local_dictionary_component_supervision_seed0_v1/`
(REPORT/DECISION/EXECUTION/dev_eval/mechanism_health), this round's
`parallel_evidence.json`, `frozen_folds.json`, `init_identity.json`,
`kappa_fusion_A.json`, `PROTOCOL.md`.

## 6. Round outcome (2026-10-06 ~04:00; stages 1-3 complete, stage 4 not purchased)

1. **The joint J_D-vs-J_M gap is dominated by run-to-run training noise.**
   The source round had M better (G0 cal −0.002045, raw CI < 0); this round's
   contemporaneous J_D/J_M control under the identical recipe flipped the sign
   (G_J G0 cal +0.00304 [−0.00055, +0.0064]; **raw +0.0119 [+0.0081, +0.0157]**,
   CI entirely above zero). Two matched runs, two opposite sign verdicts →
   neither sign is a robust property of the encoder class in this interface.
2. **The separate-structure×semantics construction (F) produced no robust
   signal.** Stage-1 contrasts C_D/C_M/G_F all cross zero; the interaction
   I_derived = −0.0047 [−0.0119, +0.0025].
3. **Stage-2's two qualifying signals failed the pre-registered B-split
   replication (frozen gates require both raw gains > 0):**
   - F_D_RAND vs J_M (A: +0.0032 cal, +0.0101 raw) reversed on B to
     −0.0025 [−0.0061, +0.0011] cal with both raws negative.
   - F_M_I vs J_M (A: +0.0055 cal CI>0, +0.0087 raw CI>0 — a full "clear
     signal") kept its calibrated gain on B (+0.0051 [+0.0017, +0.0084]) but
     **reversed raw to −0.0176 [−0.0214, −0.0137]**: the replicated advantage
     rides entirely on the per-arm median bias (B_F_M_I bias −0.0695 vs
     B_J_M −0.0193), i.e. a calibration-offset effect, not per-molecule
     accuracy. Purchase rejected; the official-valid read never happened.
4. **Task adaptation of the structure dictionary directionally hurts**
   (A split, all CI-crossing): F_D < F_D_RAND (−0.0025), REC_TASK < REC
   (−0.0024), REC ≤ RAND (−0.0011). Frozen/random dictionaries are at least as
   good as any trained variant in this interface — but see 3: the whole
   family failed replication.
5. **ERRATA (material, historical)**: the fulltrain-round official-valid body
   inference attached *train* incidence structures to valid roots
   (`local_mol_id` 0..999 indexing the train payload's `root_base`/`pair_ptr`;
   exact reproduction of the frozen predictions confirmed, max_abs 0.0).
   Re-scoring the frozen COMP soup on the valid graphs' **own** incidence
   improves valid y_cal MAE **0.11741 → 0.11216** (h_raw mean |Δ| 0.0243, max
   0.1314). The same loader pattern was used for the cycle round's official
   **test** body inference — its terminal test numbers inherit the same
   defect and need one re-scoring pass before being quoted again
   (`valid_structure_check.py/.json/.npz` in this round's dir).
6. Interface health at every stage: zero-injection collapses G0 cal to
   0.46-0.63 (the local channel carries real signal); J↔I weight swap moves
   G0 cal by ≤0.01; dictionary codes stay exactly top-8 sparse; frozen
   dictionaries by construction drift 0.
