# E2E-DictEnv-T1 — analysis

Round **E2E-DictEnv-T1** · protocol `e2e_dictenv_t1` · study `zinc-context-gap`.
[preregistration](e2e_dictenv_t1_preregistration.md) ·
[prior-artifact audit](e2e_dictenv_t1_prior_artifact_audit.md) ·
[implementation](e2e_dictenv_t1_implementation.md) ·
[test read](e2e_dictenv_t1_test_read.md).

Frozen verdict:

```text
E2E_DICTENV_TUNED_MECHANISM_LOST
```

Formal-run commit `3c13c816cede53e5e03d5933fdada5e908561201`, remote
A100-SXM4-40GB, seed 0 tuning + frozen seed-1 confirmation, official test loaded
once at the very end (reporting only).  All numbers below come from
`results/e2e_dictenv_t1/`.

---

## 1. The question and the short answer

The pre-registered question was whether v0's weak absolute band
(`M_S = 0.145508`) was caused by a **weak dictionary core** or by an
**unnecessarily compressed dictionary→environment interface** (plus an
over-constrained reconstruction/task balance).

**Short answer — the ceiling was not a weak core.**  Relaxing only the three
authorised axes (interface, λ, horizon) moved the fixed Top-5 soup official-valid
MAE from `0.145508` to `0.125765`, a material gain of `+0.019743`, reaching the
pre-registered **Breakthrough** absolute band (`<= 0.130`) and beating the FEC-S1
static-composition anchor `0.130422`.  The dictionary core itself was never the
bottleneck — its structural/optimisation health was fine in v0 and stays fine
here.

**But the interface change that relieves the ceiling does so by making the
dictionary auxiliary.**  The authorised interface exposes the exact 146-D FEC-S0
coarse chemistry descriptor *alongside* the dictionary code,
`z_i = [x_i^146 ; m_i^D]`.  That second pathway lets the reader predict well
without the dictionary coordinate: neutralising the dictionary now costs almost
nothing (`G_dict-use = +0.017903`, versus `+0.854839` in v0), and the
pre-registered assignment-shuffle gate fails (`G_assign = +0.007628 < 0.010`).
Under the frozen decision table this is **case 4 (mechanism lost)**, which takes
precedence over the large absolute gain.

So the 0.145 ceiling was an **interface/architecture** property, not a property
of the dictionary core; but the cheapest interface relief found inside the
authorised budget relieves it by routing information around the core.

---

## 2. Stage A — dictionary→environment interface (3 candidates)

Seed 0, 240 epochs, `lambda_0 = 135.834928`, fixed Top-5 soup.

| cand | interface | soup valid | best valid | best ep | params | valid rec | health | failed gate |
|------|-----------|-----------:|-----------:|--------:|-------:|----------:|:------:|-------------|
| A1 | pooled64, z 210 | 0.128383 | 0.134283 | 201 | 66067 | 1.02e-4 | **FAIL** | `task_gradient_to_d` |
| A2 | slot48,   z 290 | **0.126436** | 0.131860 | 237 | 66132 | 1.09e-4 | PASS | — |
| A3 | slot64,   z 338 | 0.129966 | 0.135920 | 228 | 66219 | 2.34e-4 | PASS | — |
| A0 | v0 (reference)   | 0.145508 | 0.150982 | 228 | 66158 | 9.7e-5  | PASS | — |

Every candidate beats the v0 anchor by far more than the 0.002 material
threshold, so the pre-registered **`INTERFACE_UPGRADE_NO_MATERIAL_GAIN`** case is
not triggered.  The within-Stage-A ordering is A2 (0.126436) < A3 (0.129966) <
A1 (0.128383 excluded), i.e. **preserving root-relative shell localisation
(slot) beats the pooled sum**, consistent with the pre-registered interpretation
rule "A2/A3 clearly beat A1 → keeping root-relative shell localisation into the
nonlinear decoder is supported".

A1's failure is exact and interpretable, not a fluke: after one backward pass on
a real training batch, `‖∂L_task/∂D‖ = 0` **exactly** for the pooled interface,
while A2 and A3 have `0.0795` and `0.0339`.  Pooling the per-shell binding
`(α_v W_R) ⊙ (q_v W_C) ⊙ S_{s_iv}` into a single `m_i^D` makes the task loss
invariant to the dictionary coordinate at the aggregate, so `D` receives no task
gradient.  This is the first direct evidence in this line that the
*dictionary→environment interface*, not the dictionary, determines whether the
core is task-coupled.  Under the frozen rule only health-PASS candidates are
eligible, so A1 is out and parity tie-breaks are not needed.

Selected: **A2 = `A2_COARSE146_SLOT48`**.

---

## 3. Stage B — reconstruction/task balance (2 arms, on A2)

`B0` = A2 at `lambda_0 = 135.834928`; the two new arms are eligible.

| arm | λ_rec | soup valid | best valid | best ep | valid rec |
|-----|------:|-----------:|-----------:|--------:|----------:|
| B0 (A2) | 135.835 | 0.126436 | 0.131860 | 237 | 1.09e-4 |
| B1 (0.5×) | 67.917 | 0.125997 | 0.133258 | 226 | 1.39e-4 |
| B2 (0.25×) | **33.959** | **0.125765** | 0.132000 | 234 | 1.38e-4 |

Both new arms beat B0, and B2 is lowest: **the reconstruction term was
over-weighted**, and reducing it buys `+0.000671`.  All three arms pass the
Stage-B health condition (exact top-8, retention ≥ 0.80, effective count ≥ 8, no
atom > 50 %, task gradient nonzero, env rank > 1, usage stable, normalised valid
reconstruction ≤ 0.01).  The improvement is real but **sub-0.001**, i.e. the λ
axis is a minor effect relative to the interface axis — worth recording as
"over-constrained balance supported, but not the binding constraint".

---

## 4. Stage C — conditional horizon (triggered)

B2's `best_epoch = 234 ≥ 220` (and soup members reach 240), so the horizon
extension was triggered:

| horizon | soup valid | best valid | best ep |
|--------:|-----------:|-----------:|--------:|
| 240 | 0.125765 | 0.132000 | 234 |
| 320 | 0.125522 | 0.128983 | 310 |

`MAE_240 − MAE_320 = 0.000243 < 0.001`, so **horizon 240 is adopted**.  The run
is not wasted: it shows the model is still slowly improving at 240, and that the
preregistered 0.001 adoption bar is what keeps the budget honest.

**Frozen configuration** (`config_id = e2e_dictenv_t1_a2_lam0.25_h240`):

```text
architecture  A2_COARSE146_SLOT48           parameters  66132
lambda_rec    33.95873017865987  (0.25 x lambda_0)
horizon       240
tuning ceiling (seed-0 soup valid)          0.125765
band          Breakthrough (<=0.130)  [code field: "strong", the <=0.135 prefix]
improvement over v0  +0.019743
```

The tuning result clears the FEC-S1 static-composition anchor `0.130422` by
`0.004657`.

---

## 5. Confirmation — frozen seed-1 Sparse vs DenseTied

The config is frozen *before* seed 1; DenseTied shares the matrix, init, knobs,
data order, λ, horizon and soup protocol (`z = φ @ Dbar`).

| object | soup valid | best valid | best ep |
|--------|-----------:|-----------:|--------:|
| final Sparse seed 1 | **0.125496** | 0.133177 | 200 |
| final DenseTied seed 1 | 0.131315 | 0.135714 | 237 |

```text
G_sparse^seed1 = M_Dense - M_Sparse = +0.005819   (gate >= 0.003)  PASS
```

The dictionary-specific contrast replicates under the frozen tuned architecture
(and the seed-1 Sparse soup, 0.125496, reproduces the seed-0 ceiling to
`5e-4`).  So the *specificity* half of the v0 story survives the new interface;
the failure below is specifically about load-bearing, not about specificity.

---

## 6. Mechanism interventions on the final Sparse soup (the decisive layer)

| intervention | value | gate | result |
|--------------|------:|-----:|:------:|
| `M_S` (clean) | 0.125765 | — | — |
| `M_zero` (α→0, coarse kept) | 0.143668 | — | — |
| `G_dict-use = M_zero − M_S` | **+0.017903** | ≥ 0.010 | PASS (but see below) |
| `M_shuffle` (mean of 5 perms) | 0.133393 | — | — |
| `G_assign = M_shuffle − M_S` | **+0.007628** | ≥ 0.010 | **FAIL** |

Per-permutation shuffle MAE: 0.133423 / 0.134558 / 0.131568 / 0.134083 / 0.133335
— a tight, consistent ~+0.0076, not one bad draw.

Dictionary **health passes on every tag** (A2, B1, B2, C320, final Sparse seed 1):
`all_passed = true`, 27–29 / 32 atoms active, exact top-8 = 1.0, effective rank
8.7–12.1, top-1 share 0.125, valid reconstruction 1.1–1.9e-4, task gradient to
`D` 0.025–0.080, `D` moves 5.3–5.9 (Frobenius) from the K-SVD init.

The contrast with v0 is the whole finding:

| quantity | v0 (no coarse channel) | T1 (coarse channel) | ratio |
|----------|----------------------:|--------------------:|------:|
| `M_zero` | 1.000347 | 0.143668 | — |
| `G_dict-use` | +0.854839 | +0.017903 | 48× smaller |
| `G_assign` | +0.017289 | +0.007628 | 2.3× smaller |
| mean prediction shift when α→0 | 0.981 | 0.0621 | 16× smaller |

Because the environment input is `z = [x_i^146 ; m_i^D]` and
`E = env_mlp(z)`, zeroing `m_i^D` leaves `x_i^146` intact.  In v0 the dictionary
coordinate was the *only* fine structural coordinate into the decoder, so
removing it collapsed prediction to ~the mean; in T1 the reader still has the
coarse descriptor and only loses ~0.018 MAE.  The model is healthy and specific
but no longer **load-bearing through the dictionary**.

---

## 7. Verdict and reading

Frozen decision table (§18) precedence: case 4 (health / zero-code /
assignment-shuffle core mechanism fails) fires first, so

```text
E2E_DICTENV_TUNED_MECHANISM_LOST
```

and **no dictionary-core success may be claimed** for the tuned configuration,
despite the Breakthrough absolute band, the material gain over v0, and the
seed-1 specificity pass.  This is the honest reading: the shove that broke the
0.145 ceiling was a wider interface, and the widening worked by admitting a
dictionary-independent pathway.

Three durable, transferable conclusions:

1. **The 0.145 ceiling was an interface property, not a core-weakness property.**
   `K`, `s`, IHT steps, `D` init and the coding operator were untouched, yet the
   valid soup moved from 0.1455 to 0.1258 (+0.0197, into Breakthrough).  A weak
   core cannot explain that.
2. **Task-coupling of the dictionary is controlled at the interface, and can be
   destroyed by it.**  The pooled interface gives `∂L/∂D ≡ 0`; the slot interface
   keeps a nonzero task gradient.  Whether the dictionary is "load-bearing" is
   therefore a property of how the environment code is composed, observable with
   a single backward pass.
3. **Structural health is not behavioural load-bearing.**  The tuned model passes
   every dictionary-health gate (exact top-8, rank, retention, reconstruction,
   nonzero task gradient) yet its output barely depends on the dictionary;
   only the zero-code / assignment-shuffle interventions expose that.  Any future
   claim of a "dictionary core" must carry the behavioural interventions, not
   just health.

---

## 8. Caveats (recorded, not resolved)

* **Single tuning seed.**  Seed 0 tunes; seed 1 only confirms the frozen config.
  The 0.1258 ceiling is a one-draw valid estimate; the seed-1 Sparse soup
  (0.125496) reproduces it, which is reassuring but is still one extra draw.
* **GPU heterogeneity.**  A1 ran on a GPU0 contended by a foreign ~37 GB / ~100 %
  job, A2/A3/B/C/confirmation on GPU1 (foreign ~35 GB / ~0 %).  Wall-clock is not
  comparable across candidates; valid MAE is (the training math is per-epoch, not
  wall-clock).  The GPU is not bit-deterministic.
* **The coarse-146 channel is shared by all three Stage-A candidates**, exactly
  as pre-registered, so T1 cannot separately attribute the interface gain to
  "exposing coarse chemistry" versus "slot width/mode".  The within-Stage-A
  contrast only supports the shell-localisation conclusion; the coarse-vs-core
  attribution is an interpretation of the v0↔T1 mechanism difference, and is
  flagged as such.
* **The shuffle miss is marginal in absolute terms** (`0.0076` vs a `0.010`
  gate, −0.0024) and the GPU is not bit-deterministic.  But the *collapse
  relative to v0* (0.0173 → 0.0076) is far larger than execution noise and is
  corroborated by the independent 48× `G_dict-use` collapse, so the mechanism
  conclusion does not rest on the marginal gate alone.

---

## 9. Stop / next step

Per the frozen stop rule and §12/§20, this round authorises **no** rescue
(K/s/IHT/LISTA/dictionary-count/attention/MP/recurrence/reader/optimizer change,
no 400/500-epoch run, no new λ).  Two questions are left for a future
preregistration, to be decided by the user:

* **Can the ceiling be held while restoring load-bearing?**  The result says
  accuracy and dictionary-necessity came apart under this interface.  A future
  round could ask for an interface that is wide enough for the coarse signal but
  gates the *prediction* through the dictionary coordinate (e.g. an architecture
  in which the reader cannot see `x_i^146` without the dictionary code), with the
  zero-code / shuffle gates as the primary success criterion, not accuracy.
* **Is a dictionary-independent coarse channel simply a stronger backbone?**  If
  the honest goal is absolute ZINC MAE, the tuned T1 interface is a competent
  baseline (0.1258 valid, beats FEC-S1); whether the dictionary is worth keeping
  then becomes an explicit engineering question rather than a mechanism claim.

Neither is authorised by E2E-DictEnv-T1.
