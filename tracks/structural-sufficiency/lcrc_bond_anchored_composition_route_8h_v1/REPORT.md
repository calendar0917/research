# Bond-Anchored S2 Joint Composition (BAJC) — 8 h route report

Round `lcrc_bond_anchored_composition_route_8h_v1` · branch `exp/lcrc-explanation-task-relevance-v1`
· wall clock `2026-11-18T00:33:38+08:00` → `2026-11-18T03:00+08:00` (≈ 2.45 h)
· 16 formal units · **2.66 GPU·h** of the 14 GPU·h soft budget · 2 × A100-SXM4-40GB (`res-gpu0`, `res-gpu1`, one process per card)

Every number below is **DEVELOPMENT**. Development Test was read exactly once per
frozen model, after the checkpoint was selected on Valid, and it never re-entered a
choice. No historical model was retrained. Nothing from `random122` / `random42` /
`public122` enters any comparison table here.

---

## 1. What was built

One new object and one new residual, on top of the **unchanged** C1 function:

* **Event** — an *undirected simple three-bond path* `P = (a,b,c,d)`: four distinct
  atoms, three real physical bonds `(a,b) (b,c) (c,d)`. The middle bond `(b,c)` is the
  **anchor bond**. `P` is exactly the pair of **real, adjacent frozen S2 occurrences**
  `p = (a,b,c)`, `q = (b,c,d)` (plus their reversals), verified item-by-item against
  the frozen enumeration. `closure4 = 1` iff `(a,d)` is additionally bonded.
* **Residual** — `u(r)` on the frozen 200-d per-occurrence typed-S2 record, `v(b)` on
  the frozen 96-d bond embedding,

  ```
  x(p,q,e)   = concat[u(r_p), u(r_q), u(r_p)*u(r_q), v(b_e), closure4]      (113)
  psi        = Linear(64,32) · SiLU · Linear(113,64)
  z_P        = ½ (psi(x_fwd) + psi(x_rev))
  c_P        = w·z_P / sqrt(max(1, n_paths3(G)))          w zero-initialised
  delta_comp = Σ_P c_P                                     y_new = y_C1 + delta_comp
  ```

  The nonlinearity sits **before** the path aggregation, both directions are averaged
  (so atom relabelling cannot change a number), and `w = 0` makes `delta_comp ≡ 0`
  at initialisation. **17 392 parameters = 7.61 %** of C1's 228 674.
* **MARGINAL control** — identical events, anchors, first path, `closure4` and `1/√n`
  normaliser; the second path's raw 200-d evidence is replaced by the *molecule-level
  mean* of that evidence per direction (`u(mean r)`, not `mean u(r)`). Same shapes,
  same parameter count, same initialisation stream, same event set.

Support (Stage-0 gate): 179 589 / 23 334 / 23 796 events on Train/Valid/Test, median
54 per molecule, `fraction_molecules_with_event = 1.000` on every split, and every
event resolvable to four frozen occurrences → `PAIRING_SUPPORT_OK`.

The 14 pre-registered gates plus one non-label numerical gate all pass
(`analysis/preflight.json`, ≈ 11 s, `failed gates: []`). Teacher replay reproduces the
frozen predictions to `max |Δraw| = 1.4e-6` (Train/Valid/Test). The protocol was
frozen (`analysis/protocol_freeze.json`) **before** the first formal unit.

---

## 2. The three questions

### Q1 — does the real-bond pairing carry task information beyond an equal-capacity control?

**No.** Lipo Valid, `gain_null = RMSE(MARGINAL) − RMSE(JOINT)` (positive ⇒ JOINT better):

| institution | seed | JOINT | MARGINAL | gain_null | 95 % CI | P(JOINT better) | pairing |
|---|---|---:|---:|---:|---|---:|---|
| FROZEN | 0 | 0.7047 | 0.7039 | −0.0007 | [−0.0034, +0.0018] | 0.28 | clean |
| E2E | 0 | 0.7172 | 0.6944 | −0.0227 | [−0.0559, +0.0103] | 0.09 | **control dead** |
| E2E | 1 | 0.7209 | 0.7066 | −0.0143 | [−0.0458, +0.0190] | 0.21 | **control dead** |
| E2E | 2 | 0.7063 | 0.7113 | **+0.0051** | [−0.0304, +0.0427] | 0.60 | clean |

E2E mean over the three pre-registered seeds: **−0.0106**. The only *clean* E2E
pairing (seed 2, both branches live) gives +0.0051 with a CI of width 0.073 — i.e.
nothing. Two of the three E2E seeds are not a clean null at all: the MARGINAL
branch stayed at `mean|Δ| = 0.0001 / 0.0000` for all 360 epochs, so at those seeds
"JOINT vs MARGINAL" degrades into "branch vs its own C1 trunk".

WC_LOGP (independent computed task) is the one row with a positive, highly probable
`gain_null`: 0.1106 vs 0.1119, +0.0013, P = 0.96. It is also a task with **no
headroom**: the frozen `L_WC_LOGP` teacher already has Valid R² = 0.9941
(0.1144 RMSE on a 1.49-std target), so a 0.0013 difference is 1 % of the residual and
cannot be distinguished from "the branch is doing nothing". Both WC branches are
2 orders of magnitude smaller than the Lipo E2E branches (`|Δ|` 0.020/0.018 raw vs
0.24).

### Q2 — does it improve real property prediction and close the C1 → GINE gap?

**No.** The C1 → GINE Valid gap is **0.0908** (0.7069 → 0.6161, paired CI
[+0.0206, +0.1678]).

| unit | Valid | vs C1 (0.7069) | vs GINE (0.6161) | Train |
|---|---:|---:|---:|---:|
| FROZEN JOINT s0 | 0.7047 | +0.0022 | 0.0886 | 0.2321 |
| FROZEN MARGINAL s0 | 0.7039 | +0.0030 | 0.0878 | 0.2294 |
| E2E JOINT s0 / s1 / s2 | 0.7172 / 0.7209 / 0.7063 | −0.0103 / −0.0140 / +0.0006 | 0.101 / 0.105 / 0.090 | 0.172 / 0.230 / 0.268 |
| E2E MARGINAL s0 / s1 / s2 | 0.6944 / 0.7066 / 0.7113 | +0.0125 / +0.0003 / −0.0044 | 0.078 / 0.091 / 0.095 | 0.266 / 0.236 / 0.182 |
| **best in round** | **0.6944** (the control) | +0.0125 | **0.0783** | |

The best Lipo Valid RMSE of the whole round is produced by the **control**, and it
still leaves 86 % of the C1 → GINE gap untouched. The real-pairing arm is worse than
the historical C1 anchor at two of three E2E seeds. And per the protocol: *"JOINT
being better than its own frozen teacher is never evidence for the composition"* —
the FROZEN +0.0022 (P = 0.77) is inside noise and the control is equally better.

Where does the branch spend its capacity? Live branches drive Train far below Valid
(JOINT s0: Train 0.172 / Valid 0.717; MARGINAL s2: Train 0.182 / Valid 0.711). The
branch is expressive enough to memorise Train — 0.17 RMSE on Train is *below* the
best Valid of any historical model — and that expressiveness is what costs Valid.
At E2E s0 the JOINT branch also cannibalises the trunk: `pred_base` (branch off) is
0.7959, so 0.079 of the final 0.717 is bought by the branch fixing a trunk that joint
training broke. That is a reparameterisation, not a structural gain.

### Q3 — does it recover externally-referenced connected structure, or only a construct-accountable contribution?

**Only the constructive ledger.** `delta_comp = Σ_P c_P` is an architectural identity
and is never counted as evidence. The three measurements that can fail:

| measurement | FROZEN Lipo JOINT (primary) | E2E Lipo JOINT s0 | reading |
|---|---|---|---|
| event deletion, top-10 % `\|c_P\|` vs matched random | 3.62 × | 3.39 × | *internally* faithful |
| event deletion, top vs bottom 10 % | 52.6 × | 22.4 × | monotone in `\|c_P\|` |
| direction asymmetry `\|c_P − c_P(reversed)\|` | **0.0** exact | **0.0** exact | invariance holds |
| branch-off identity `pred(branch=False) − pred_base` | **0.0** exact | **0.0** exact | zero residual holds |
| **anchor-bond gate vs matched random bond** (2 %/5 %/10 %/25 %) | **1.21 / 1.20 / 1.12 / 1.07 ×** | **1.03 / 1.12 / 1.00 / 1.00 ×** | the anchor is barely privileged |
| **anchor-bond gate vs the *peripheral* bonds of the same events** | first hop **1.20 ×**, last hop 1.11 × | first hop 1.17 ×, last hop 1.14 × | **the anchor is not special** |
| **WC_LOGP oracle capture**, top event vs matched random | — | — | **enrichment −0.0047 [−0.0089, −0.0003], P(>0) = 0.018** |
| contribution vs error reduction, Spearman | +0.008 (p = 0.87) | **−0.559 (p = 7e-36)** | no / anti-aligned |

Two details that matter:

* **Oracle capture is at chance and, on the primary task, slightly below it.** A
  random 4-atom draw captures 0.1563 of the Crippen contribution mass; the event the
  composer rates highest captures 0.1516. `capture ≈ 4/n_atoms` is exactly what
  "four arbitrary atoms" predicts. The MARGINAL control is at chance too
  (−0.0021, P = 0.15), so this is not a "control is better" artefact — neither arm
  localises. The Lipo transfer row is also at chance (+0.0024, P = 0.86).
* **The physical-bond intervention does move the prediction** (0.246 raw RMSE at 2 %
  gating, 0.924 at 25 %), and it is the validated `CONDITIONAL_S1_S2_EVIDENCE_MASK`
  semantics with bond-type-matched random controls. So the intervention is not inert;
  the *anchor* simply is not the bond that matters — the two peripheral hops of the
  same event are equally or more important.

---

## 3. Stage C confirmation (why it ran, and what it added)

The fourth pre-registered trigger ("a non-degenerate branch with non-reversed
Train/Valid gains") fired for E2E, so Stage C ran with the E2E institution. The three
numeric thresholds never came close: no seed reached +0.010 vs MARGINAL, +0.015 vs
C1, or P ≥ 0.80.

* **Seed 2** (pre-registered confirmation seed): JOINT 0.7063 / MARGINAL 0.7113. It is
  the *only* seed where both branches are live, and it is the one seed where JOINT
  wins — by +0.0051, CI [−0.030, +0.043], P = 0.60. Both readings ("the dead control
  was flattering JOINT's rivals" and "replication at 113–420 molecules is noisy") are
  consistent with the data; neither rescues the route.
* **ESOL transfer** (official `ogbg-molesol` 902/113/113, own Train-only scaler, E2E
  only — no teacher exists, no historical ESOL baseline was retrained or compared):
  gain_null **+0.0104 / +0.0119 / −0.0033**, mean **+0.0063** on Valid. All three
  pairings are clean (`|Δ|` 0.13–0.28). The sign reverses at seed 2. On Development
  Test the same three seeds give +0.0532 / +0.0249 / +0.0361 (mean +0.0381) — a real
  observation, but obtained on 113 molecules after the Valid readout, so it is
  reported and **not** used for selection or as confirmation.
* **Explanation evaluation**: the two tables above. FROZEN is the primary row (its
  branch was trained against a fixed, known teacher, so the composition is the only
  thing that moved); the E2E row is independent corroboration and is the one that
  turns from "no signal" into "anti-signal" (−0.56).

---

## 4. Stage D read-only integration — where the C1 → GINE gap actually lives

Pre-registered before the gap was inspected (`analysis/stage_d_plan.json`), read-only,
no GPU. Valid, 420 molecules, 2 000-draw paired molecule bootstrap:

| slice | bin | n | C1 | GINE | gap | 95 % CI | P(GINE better) |
|---|---|---:|---:|---:|---:|---|---:|
| atom count (quartiles) | Q1 / Q4 | 105 / 105 | 0.6432 / 0.7968 | 0.5520 / 0.6755 | +0.0912 / +0.1213 | [+0.0022,+0.1780] / [−0.0727,+0.3097] | 0.98 / 0.85 |
| paths3 count (quartiles) | Q1 / Q4 | 105 / 105 | 0.6255 / 0.8000 | 0.5637 / 0.6277 | +0.0618 / **+0.1723** | [−0.0207,+0.1479] / [−0.0300,+0.3567] | 0.93 / 0.95 |
| ring-atom fraction | (0,0.3] / >0.3 | 19 / 401 | 0.7120 / 0.7066 | 0.7025 / 0.6117 | +0.0095 / +0.0950 | [−0.2974,+0.3067] / [+0.0202,+0.1728] | 0.55 / 0.99 |
| C1-error decile | 0–5 | 252 | 0.041–0.448 | 0.271–0.523 | **−0.050 … −0.342** | — | 0.00–0.10 |
| C1-error decile | 6–9 | 168 | 0.561–1.704 | 0.514–1.165 | +0.047 / −0.012 / +0.181 / **+0.539** | — | 0.89 / 0.46 / 1.00 / 1.00 |

The **error-decile trade** is the robust part: GINE is *worse* than C1 on the 60 % of
molecules C1 already predicts well (deciles 0–5 all negative, P(GINE better) ≤ 0.10)
and better by **+0.539** on the worst decile. It is a tail trade — GINE buys the heavy
tail and pays on the body. **The same shape reproduces on Development Test**
(deciles 0–5: −0.055 … −0.360, P ≤ 0.14; decile 9: +0.339, P = 1.00).

The concentration by **paths3 count** is *not* robust and must not be quoted as a
finding: Valid Q4 gives +0.1723 (P = 0.95) but Development Test Q4 gives **−0.0142**
(P = 0.37), with Q2 also flipping sign. Read once, on a 105-molecule bin, that slice
does not survive. The only slice-level statement this round will carry forward is the
error-decile trade, and the atom-count Q1 gain (Valid +0.0912 → Test +0.0544, same
sign).

---

## 5. What this round rules out and what it does not

**Ruled out, with mechanism.** A real-physical-bond-anchored two-path interaction
term, added to a frozen or jointly trained C1 at 7.6 % parameter overhead with an
exactly-matched equal-capacity control, does not buy Valid performance on Lipo, does
not move towards GINE, and does not localise any externally referenced structure. The
branch is trainable and expressive (Train 0.17), it is direction-invariant to the bit,
it is exactly zero at initialisation — and its contribution is uncorrelated (FROZEN)
or anti-correlated (E2E) with where the error actually is.

**Not ruled out.**
* The **control is not yet trustworthy** at 2 of 3 E2E seeds: a zero-initialised
  readout over a molecule-mean input stays at zero for 360 epochs. Any future round on
  this route must first make the MARGINAL arm train (e.g. a non-zero readout init, or
  an explicit branch warm-up) — otherwise `gain_null` measures trainability, not
  structure.
* The WC_LOGP calibration task is **saturated** (teacher R² = 0.9941) and cannot
  discriminate anything at the 0.001 level. It should be replaced as an external
  reference by a task with residual headroom, or used only for the atom-additive
  oracle channel.
* ESOL Valid is 113 molecules; a ±0.04 CI cannot resolve a 0.01 effect. The ESOL Test
  column (+0.038 mean, 3/3 seeds) is the only place in the round where the sign is
  consistent, and it is Development Test — it needs a fresh pre-registration, not a
  reinterpretation of this one.

---

## 6. Compliance

| constraint | status |
|---|---|
| 8 h hard / 7.5 h soft wall clock | 2.4 h |
| 14 GPU·h soft / 16 hard | 2.66 GPU·h (16 units, `analysis/compute_ledger.json`) |
| `res-gpu0` + `res-gpu1`, one GPU process per card | yes, 21 `rr` runs, never two units on a card (`analysis/rr_runs.json`) |
| no retraining of C1 / GINE / D-MPNN / R1 / R2 / RR_REAL / RR_CONTROL / oracle baselines | none retrained (teachers loaded and SHA-256 verified) |
| no literature work, no self-invented architecture | the composition is the protocol's own object; no external architecture adopted |
| historical Test is DEVELOPMENT; each new frozen model reads Test once | yes, after min-Valid selection |
| protocol frozen before the first formal unit | `analysis/protocol_freeze.json` (`protocol_hash a722872b…`), written 00:45, first formal unit 01:02 |
| engineering fixes logged with affected units | `analysis/engineering_fixes.json` (FIX-01…06); no re-freeze, the science payload hash is unchanged |
| additions logged with the reason and the "cannot select" guarantee | `analysis/protocol_additions.json` (ADD-01 ESOL seeds 1/2, ADD-02 the E2E explanation row) |

Test was read once per frozen model. Everything reported is DEVELOPMENT.
