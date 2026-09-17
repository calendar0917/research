# FSAR-C1 — parameter-matched incidence processor (durable verdict)

Branch `exp/fsar-incidence-processor-zinc`.  Pre-registration:
`notes/fsar_incidence_processor_preregistration.md`, committed at `412b3bd`
**before** any formal run.  Training revision `412b3bd`; eval-only revision
`1f1871f` (`decide`/`report` include the C0 control).  **Official ZINC test
never loaded** (`official_test_loaded = false` everywhere).

This is a **processor hypothesis** round, not a feature-engineering round.  It
keeps the exact S / A / B primitives of FSAR-R2-AR0 / AR0-EDGE and only changes
*how* the assignment information is processed.

---

## 0. The one-line answer

At **identical 88,643 parameters**, the real node↔incident-edge **incidence
processor (C1)** reaches valid Top-5 soup **0.182050**, while the
**parameter-exact incidence-free bag processor (C0)** reaches **0.369359** —
a **+0.187309** MAE mechanism gap with every branch alive and high-rank in both
models.  So incidence composition is real and causally responsible for most of
the gain.  But C1 still sits **+0.062232 above B-Full seed0 (0.119818)** and
**+0.054668 above B-Bag seed0 (0.127382)**, so the processor hypothesis is
**partially supported**: incidence composition is necessary and highly
effective at this budget, but **not sufficient** to recover the strong mixed
backbone.  Pre-registered seed0 gate call: **INSPECT** (0.14 < 0.18205 ≤ 0.20).
**No seed1 was run, no official test was touched.**

---

## 1. Architecture (exact)

Shared by both variants; identical modules and parameter count.

| component | content | params |
|---|---|---:|
| node initializer | `f_s: 65→48`, `f_a: 28→48` | 4,560 |
| edge initializer | `f_se: 130→48`, `f_ae: 4→48` | 6,528 |
| binding | `U_s,U_a,W_b` (node), `U_se,U_ae,W_be` (edge), all 48×48 | 13,920 |
| incidence processor | `W_e1 144→96`, `W_e2 96→48`, `W_m1 96→96`, `W_m2 96→48`, `W_n1 96→96`, `W_n2 96→48`, 2 LayerNorms, 2 scalar residual scales | 46,994 |
| readout | `258→64→1`, SiLU | 16,641 |
| **total** | | **88,643** |

* `h_v^0 = f_s(phi_v) + f_a(q_v) + W_b(U_s f_s(phi_v) ⊙ U_a f_a(q_v))`
* `g_e^0 = f_se(psi_e) + f_ae(r_e) + W_be(U_se f_se(psi_e) ⊙ U_ae f_ae(r_e))`
* 4 **shared** recurrent rounds.  Edge update:
  `g_e ← g_e + α_e·W_e2(SiLU(W_e1(LN([h_u+h_v, |h_u-h_v|, g_e]))))` (endpoint
  symmetric).  Node update: degree-normalised mean over the true incident
  edges of `W_m2(SiLU(W_m1([g_e, h_w])))`, then
  `h_v ← h_v + α_v·W_n2(SiLU(W_n1(LN([h_v, m_v]))))`.  Dense-matmul incidence
  (no atomic `index_add_`), pre-LN, SiLU, scalar residual scales init `0.05`.
* readout input `[mean(h), std(h), mean(g), std(g), A(G), log1p(n), log1p(m)]`.
* no attention, no pair state, no triangles, no new feature family, radius 2,
  `dataset_dependent_vocabulary_params = 0`.

**C0 (parameter-exact incidence-free control).**  Same modules; the edge
update sees only its graph's global node mean and the node update receives the
global edge-message mean.  Real incidence is destroyed, every parameter stays
alive.  Because every parameter is shared, **C0 total == C1 total == 88,643**
(verified `parameter_accounting.json`).

**Parameter context.**  B-Full 84,495 / B-Bag 84,511 / cell A 85,763 (the
80–100k strong reference); AR0-EDGE BVE diagnostic 25,317.  C1 is +4,148
(+4.9 %) over B-Full.  No hidden padding was used to hit the band.

---

## 2. Protocol / execution regime

* canonical `OPTIMIZED_PROTOCOL`: Adam, lr 1e-3, batch 128, L1, grad-clip 5.0,
  no scheduler, max 240 epochs, patience 40, best official-valid checkpoint,
  fixed equal-weight Top-5 soup.
* single documented deviation (pre-registration §4): binding modules, the two
  LayerNorms and the two residual scales are in a `weight_decay = 0` group;
  everything else keeps `wd = 1e-5`.  Rationale: the repo's three prior
  Adam+L2 branch-annihilation replications.
* deterministic: `torch.use_deterministic_algorithms(True)`,
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`; NVIDIA A100-SXM4-40GB.
* GPU 0 held an unknown 35.8 GB / 100 % third-party task — **never touched,
  never co-tenanted**.  All jobs on GPU 1, which carried an unrelated
  ~561 MiB / 13 % process (recorded, never touched).  Our peak GPU memory:
  254.8 MB (smoke), 275.9 MB (C1 formal), 255.9 MB (C0 formal).
* official ZINC test never loaded.

---

## 3. Stage A — local implementation validation

* `tests/test_fsar_incidence.py`: **14/14 pass** (endpoint-swap invariance,
  `psi` chemistry purity, node relabel invariance, bond permutation changes
  C1, incidence used by C1 and not by C0, atom permutation sensitivity, batch
  invariance, C0/C1 parameter exactness, all-parameter non-zero gradient, one
  Adam step moves every binding module, parameter counts, no forbidden feature
  family / out-of-schema input, determinism, empty-edge safety).
* `gradient_audit --all` (CPU, deterministic): active prediction path
  **88,643 / 88,643**, all seven module groups non-zero (C1 and C0).
* forward/backward shape and finite checks on real valid batches.
* a 1-epoch CPU training smoke wrote a complete curve with all mechanism
  columns, then was deleted.

**Synthetic controls (must pass before any ZINC run).**

* *positive — incidence-only star task.*  Fixed 4-leaf star; exactly one leaf
  carries N and exactly one edge carries DOUBLE; label = 1 iff they coincide.
  The node multiset and edge multiset are identical for both labels, so any
  per-object + global-pooling model (C0) must be at chance.  Valid MAE
  (3 seeds, 250 epochs): **C1 0.00741** vs **C0 0.51087** → pass.
* *negative — marginal-only task.*  Label from topology/atom/bond counts, then
  assignment randomised.  Valid MAE **C1 0.00637** / **C0 0.00241** (within
  0.05) → pass.

This is the cleanest possible demonstration that C1 really routes the bond
type to its endpoint node and C0 cannot.

## 4. Stage B — remote GPU smoke

`smoke --device cuda --deterministic --steps 6`: finite losses, output not
constant, all module groups non-zero, 88,643 params, C1 and C0, peak 254.8 MB.
Remote `sanity` 14/14 and remote `synthetic` both pass.  No performance
conclusion is drawn from smoke.

## 5. Stage C — formal seed0 results

Two formal runs, one seed, canonical protocol.  Valid MAE:

| variant | best valid | best ep | epochs | early stop | Top-5 soup | Top-5 epochs | wall (s) | peak MB |
| --- | ---: | ---: | ---: | :---: | ---: | --- | ---: | ---: |
| **C1** incidence | 0.190477 | 219 | 240 | no | **0.182050** | 219/226/223/206/204 | 2079.9 | 275.9 |
| **C0** bag | 0.376120 | 166 | 206 | yes | **0.369359** | 166/198/199/185/194 | 2442.2 | 255.9 |

Paired deltas (positive = incidence better):

| comparison | delta |
| --- | ---: |
| C0 − C1 | **+0.187309** |
| C1 − AR0-EDGE BVE diagnostic (0.445958) | **−0.263908** |
| C1 − B-Bag seed0 (0.127382) | +0.054668 |
| C1 − B-Full seed0 (0.119818) | +0.062232 |
| fraction of the C0 → B-Full gap closed by C1 | **0.7506** |

Pre-registered gate (`gate.json`): `0.14 < 0.182050 ≤ 0.20` → **INSPECT**;
`c0_authorized = true` (so C0 was run); `seed1_authorized = false`
(`decision.json`).

**Learning curve (C1).**  Train MAE falls steadily to 0.1394 at epoch 240 while
valid oscillates 0.19–0.23 (last-20 mean 0.2096, last epoch 0.2217).  The best
epoch (219) and the soup (0.182) both sit on the low side of a noisy valid
curve.  The growing train/valid gap at the end is an **overfitting** signature,
**not** underfitting: the model had already fitted the training set below the
validation level.  C0 early-stopped (best 166, stopped 206), i.e. it converged
earlier and worse.

## 6. Mechanism diagnostics (frozen soup states)

| metric | C1 | C0 |
| --- | ---: | ---: |
| node binding output std | 0.2924 | 0.5090 |
| edge binding output std | 0.2488 | 0.4897 |
| node binding weight norm | 13.70 | 15.39 |
| edge binding weight norm | 13.63 | 14.80 |
| edge update residual std | 0.2602 | 0.1498 |
| node update residual std | 0.2069 | 0.1527 |
| edge residual scale (init 0.05) | 0.2614 | 0.2368 |
| node residual scale (init 0.05) | 0.1292 | 0.1384 |
| node rep effective rank (of 48) | 26.80 | 28.50 |
| edge rep effective rank (of 48) | 33.24 | 32.91 |
| all branch task gradients non-zero | yes | yes |

Both models are **fully alive**: no exact-zero / near-zero collapse anywhere,
large residual scales that grew from the `0.05` init, and high-rank object
representations.  C0's worse score is therefore a **representational limit** of
the incidence-free processor, **not** a collapse or optimization failure.  This
is what makes the C1 > C0 comparison a clean mechanism statement.

**Evaluation-only attribute permutation** (soup, 5 repeats, topology fixed):

| model | actual MAE | atom-perm MAE | atom mean abs Δpred | bond-perm MAE | bond mean abs Δpred |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 0.18205 | 1.2607 | 1.2326 | 1.5433 | 1.5179 |
| C0 | 0.36936 | 0.6769 | 0.5147 | 0.5168 | 0.3322 |

The trained C1 uses the real atom assignment and the real bond assignment
heavily; both permutation MAEs are ~7–8× the actual MAE.  C0 also responds
(its per-object `B_v` / `B_e` survive as pooled features), but much more
weakly — consistent with it lacking the node↔edge routing.

## 7. Exact verdict, and the most conservative interpretation

**Verdict: INSPECT — the processor hypothesis is partially supported.**

What is established:

1. At an **identical 88,643-parameter budget**, replacing the incidence-free
   bag processor with a real node↔incident-edge processor improves valid
   Top-5 soup by **+0.187309** MAE, with both models fully alive and high-rank.
   This is a clean, parameter-exact mechanism effect: **incidence composition
   is causally responsible for the bulk of the gain**, and it is not a capacity
   effect.
2. The processor recovers **75.1 %** of the C0 → B-Full gap and **~81 %** of
   the AR0-EDGE-diagnostic → B-Full gap at essentially the strong-model
   parameter budget, and is **−0.263908** better than the 25k factorized
   diagnostic.  Keeping persistent node/edge objects and composing over the
   real graph incidence is dramatically better than collapsing assignment
   into graph-level statistics.

What is **not** established / what must not be claimed:

1. C1 does **not** recover the strong backbone: it is **+0.062** above B-Full
   seed0 and **+0.055** above B-Bag seed0.  "Incidence composition recovers
   the strong backbone" is **false** at this budget.
2. This is a **single seed**.  The C1 valid curve is noisy and the best/soup
   numbers benefit from late dips; the +0.187 C1−C0 gap is much larger than
   the known ~0.007 per-seed soup spread, so the mechanism conclusion is
   robust in direction, but the exact values are single-seed.
3. The residual gap has at least two candidate explanations that this round
   **cannot separate**: (a) the strong backbone consumes information this round
   deliberately excludes (typed local tokens, radius-3 typed context), and
   (b) the tail overfitting / optimization at this budget.  The learning curve
   shows (b) is present; whether (a) also matters is open.
4. C0 is the **parameter-exact incidence-free processor**, not a pure bag: its
   per-object `B_v`/`B_e` survive as pooled features, which makes the C1 > C0
   test a *conservative* test of incidence (C0 is a stronger control than a
   plain bag).  It is still not a full information-matched non-incidence
   model.

**Failure taxonomy.**  No branch is dead; the label is **partial mechanism
support / capacity-or-information limited**, not `mechanism/optimization
failure`.

## 8. Gate and next action

* Pre-registered call: **INSPECT** (0.14 < 0.182050 ≤ 0.20).
* **No seed1 was run.**  **STOP** was not triggered.  **seed1-worthy** was not
  reached (0.182 > 0.14).  **STRONG** was not reached.
* Exactly **one** recommended next action: **a paired seed-1 confirmation of
  C1 and C0** (two runs, canonical protocol, same commit).  The headline
  scientific claim of this round is the parameter-matched mechanism delta
  (C1−C0 = +0.187), which is currently single-seed; replicating that delta at
  a second seed is cheaper and more decisive than chasing the absolute level
  by changing the architecture or the information set, both of which are out
  of scope for this round.  Do **not** auto-start it; it requires explicit
  authorisation.

Not to be done here: adding typed local context / radius-3 features, changing
the optimizer or regularization, adding attention / pair state / triangles,
sweeping width/depth/rounds, opening the official ZINC test, or buying 3+
seeds.

## 9. Reproduce

```
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence params
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence sanity
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence synthetic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence smoke   --device cuda --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence run     --variant C1 --seed 0 --device cuda --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence run     --variant C0 --seed 0 --device cuda --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence diagnostics --variant C1 --seed 0 --state soup --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence witness     --variant C1 --seed 0 --state soup --deterministic
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence gate
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence decide
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_incidence report
```

Artifacts: `results/fsar_incidence/` — `parameter_accounting.json`,
`sanity.json`, `synthetic_controls.json`, `smoke.json`,
`gradient_audit_{C1,C0}.json`, `runs/`, `curves/`, `states/`, `soup_states/`,
`soup_inc_{c1,c0}_seed0.json`, `diagnostics_inc_{c1,c0}_seed0_soup.json`,
`witness_inc_{c1,c0}_seed0_soup.json`, `gate.json`, `decision.json`,
`FORMAL_RESULTS.md`.  Remote run dir: `~/.research_runs/c1-seed0`,
`~/.research_runs/c0-seed0`.
