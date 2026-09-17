# FSAR route recheck — binding null + explicit local S (durable split verdict)

Branch `exp/fsar-route-recheck-binding-explicit-zinc`.
Pre-registration: `notes/fsar_route_recheck_preregistration.md` (18 sections +
§19 addendum for Wave 6).
Training revision `50b5972` (architecture / runner / tests / pre-registration);
eval-only revisions `a4f5fde` (decide enrichment + explicit-basis diagnostics
path), `2146a24` (Wave 6 pre-registration), `77cdf84` (`gate()` audit-row fix),
plus the `checkpoint_audit` stage (eval-only).
**Official ZINC test never loaded** (`official_test_loaded = false` in every
JSON; `results/fsar_route_recheck/official_test*` does not exist).

This is **not** FSAR-v3 and **not** a continuation of FSAR-v2. It is an
independent route recheck that asks four separate questions and answers them
separately; no single combined verdict is issued.

---

## 0. The four answers, up front

| # | question | answer | status |
|---|---|---|---|
| **A** | does FSAR-v1 `SAB < SAM` replicate at seed 1? | `SAM−SAB` = +0.003270 (seed 0) / +0.001886 (seed 1) | **replicated**, but small and partly a dead-block artefact (§8) |
| **B** | `SAB < SABI` (aligned vs the operator-matched assignment-independent null) on paired seeds? | `Δ_B^latent` = **+0.014466 / −0.002719** (mean +0.005874) | **NOT supported** — sign flips; the only edge-matched pair is negative (§9) |
| **C** | what does explicit local `S` cost vs the latent topology-GNN `S`? | `SABE−SAB` = +0.012388 / −0.004595 (mean +0.003896); `SAE−SA`(v1) = **−0.007199** | explicit basis is a **viable replacement** (better than latent S at seed 0, more seed-stable); §10 |
| **D** | `SABE < SABEI` (explicit aligned vs explicit null)? | `Δ_B^explicit`(seed 0) = **−0.004190** | **NOT supported** — the null is better (§11) |

Plus one **mechanism finding** that changes how A and B must be read (§12): the
B channel's **edge sub-branch is annihilated** in most runs, and in FSAR-v1
itself it is alive at seed 0 but dead at seed 1.

---

## 1. What was held frozen

* FSAR-v1 `A` (56617-param weak mean/std attribute marginal), `R_S`
  (pair/path relation core) and `G_S` (graph head) — **unchanged, not retrained**.
* `A_exact` (FSAR-v2's 64-D exact marginal) — **not retrained, not used here**.
* The official ZINC test — never loaded.
* No A width sweep, no DeepSets / Transformer / attention variant, no auxiliary
  binding loss, no A-gate blocking of the explicit arms.

## 2. Mode grid (all seeds 0 and 1 unless noted)

| mode | local `S` | `B` construction |
|---|---|---|
| `A` | — | — |
| `SA` | latent topology-GNN | — |
| `SAB` | latent topology-GNN | **aligned** moments (`binding_kind="align"`, = FSAR-v1) |
| `SAM` | latent topology-GNN | marginal-only capacity MLP `[A,S]→B` (FSAR-v1 SAM; no aligned channel) |
| `SABI` | latent topology-GNN | **assignment-independent** moments (`binding_kind="indep"`) |
| `SAE` | **explicit** local structural basis | — |
| `SABE` | **explicit** | **aligned** |
| `SABEI` | **explicit** | **assignment-independent** |

`B_indep` (the operator-matched null) reuses the *same* `binding_fuse` module and
the *same* parameter count as `B_align`, but replaces the assignment-dependent
moments `m = [r_i + r_j, |r_i − r_j|]` by the assignment-free moments built from
the unordered marginals (`Σr`, `Σr²`, `(Σr)²`, cross terms). Theorem §5.2 of the
pre-registration: `m_indep` has no dependence on which attribute sits on which
node, so only the first-moment columns of `binding_fuse[0]` are structurally
zero-grad (2048 elements observed; §5).

## 3. GPU provenance, schedule and the co-tenant record

| item | value |
|---|---|
| GPU model | NVIDIA A100-SXM4-40GB |
| CUDA / PyTorch / Python | 12.4 / 2.5.1+cu124 / 3.12.14 |
| determinism | `torch.use_deterministic_algorithms(True)`, `CUBLAS_WORKSPACE_CONFIG=:4096:8` |
| GPU 0 | **unknown** co-tenant task, 35.8 GB / 91–98 % util for the whole session — **never touched, never co-tenanted, never killed** |
| GPU 1 | ours, plus an unrelated GSN process that grew from 0.54 GB to 5.6 GB mid-round — recorded, never touched |

| wave | jobs (all `--deterministic`, two jobs in parallel on GPU 1) | runner |
|---|---|---|
| 1 | `r1-sableg-s1-train` + `r1-samleg-s1-train`, then the two soups | **FSAR-v1** `zinc_fsar.py`, unmodified |
| 2 | `r2-sabi-s0`, `r2-sae-s0` (pids 441680, 441918) | route runner |
| 3 | `r3-sabi-s1`, `r3-sabe-s0` (pids 490986, 491114) | route runner |
| 4 | `r4-sabei-s0`, `r4-sabe-s1` (pids 712723, 712850) | route runner |
| 5 | **not launched** (pre-registered condition false; §11) | — |
| 6 | `r6-sablat-s0`, `r6-sablat-s1` (pids 785438, 785571) — harness control | route runner |

One process per job, no DDP. The co-tenant was re-checked immediately before
every launch. Our peak GPU memory: **288–451 MB**. Peak training provenance:

| run | runner | best valid | best ep | epochs | early stop | wall (s) | peak MB |
|---|---|---:|---:|---:|---|---:|---:|
| `route_sab` seed 0 | route @ `2146a24` | 0.1321601 | 222 | 240 | no | 3814.0 | 449.2 |
| `route_sab` seed 1 | route @ `2146a24` | 0.1508053 | 219 | 240 | no | 3803.3 | 450.4 |
| `SAB` seed 1 (legacy) | v1 | 0.1508053 | 219 | 240 | no | 3253.3 | 446.0 |
| `SAM` seed 1 (legacy) | v1 | 0.1507418 | 184 | 224 | **yes** | 2703.1 | 362.0 |
| `SABI` seed 0 / seed 1 | route @ `50b5972` | 0.1497976 / 0.1504690 | 240 / 235 | 240 | no | 4286.1 / 3831.5 | 448.8 / 450.8 |
| `SAE` seed 0 | route @ `50b5972` | 0.1460145 | 227 | 240 | no | 2886.6 | 288.1 |
| `SABE` seed 0 / seed 1 | route @ `50b5972` | 0.1468188 / 0.1432558 | 226 / 224 | 240 | no | 3673.6 / 4459.3 | 361.7 / 362.5 |
| `SABEI` seed 0 | route @ `50b5972` | 0.1450721 | 240 | 240 | no | 4460.5 | 361.7 |

Frozen protocol (identical for every run): Adam, lr 1e-3, weight decay 1e-5,
batch 128, max 240 epochs, patience 40, no scheduler, L1 loss, grad-clip 5.0,
79 steps/epoch, best-official-valid checkpoint selection, then a **fixed
equal-weight Top-5 soup** over the 5 best-valid epochs.

## 4. Harness equivalence — proved bit-for-bit (Wave 6)

The Waves 1–5 schedule used the *frozen FSAR-v1* soups as the latent `SAB`
reference. Wave 6 re-ran `SAB` inside the route harness for both seeds to test
whether that cross-harness reference is legitimate:

| seed | route `SAB` soup | frozen FSAR-v1 `SAB` soup | top-5 epochs |
|---|---:|---:|---|
| 0 | 0.13091256372159113 | 0.13091256372159113 | [222, 217, 181, 206, 180] |
| 1 | 0.1437008262788295 | 0.1437008262788295 | [219, 185, 198, 184, 227] |

Reproduced to the last float and the identical epoch list, with identical
`best_valid` (0.1321601 / 0.1508053). **The route harness is the FSAR-v1 harness
for latent modes**; the latent `SAB`/`SAM` references are therefore properly
paired and `Δ_B^latent` is not a harness artefact. (Also verified locally:
`fsar_route_collate` → `v2.fsar_v2_collate` → `v1.fsar_collate` + 2 extra
tensors, and `v2.build_fsar_v2_dataset` = `v1.build_fsar_dataset` + 2 fields, so
the latent-path tensors and batch order are identical.)

## 5. Parameter / active-path accounting (`gate`)

`matched_control_valid = true`; `gradient_audit_all_pass = true`;
`sanity 15/15` remote, `17/17` local suite.

| mode | total params | `B` params | zero-grad elements |
|---|---:|---:|---:|
| `A` | 56 617 | 0 | 12 798 |
| `SA` | 77 609 | 0 | 14 367 |
| `SAB` | 95 209 | 13 504 | 14 930 |
| `SAM` | 95 153 | 0 | 19 520 |
| `SABI` | **95 209** | **13 504** | 17 049 |
| `SAE` | 63 817 | 0 | 12 318 |
| `SABE` | 79 177 | 11 264 | 14 174 |
| `SABEI` | **79 177** | **11 264** | 16 356 |

* `SAB` vs `SABI`: **total and `B` identical** — the null has no extra MLP.
* `SABE` vs `SABEI`: **total and `B` identical**.
* `SABI − SAB` zero-grad = **+2119 = 2048** (the pre-registered first-moment
  `binding_fuse` columns) **+ 71** unused rows, confirming §5.2 exactly.
  `SABEI − SABE` = +2182.
* Module-granularity `active_prediction_path_params == total` for **every** mode
  (the FSAR-v2 "whole module outside the prediction path" failure does not
  recur).

## 6. Full results (valid MAE, fixed equal-weight Top-5 soup)

| mode | seed 0 | seed 1 | mean | seed range |
|---|---:|---:|---:|---:|
| `A` (v1) | 0.157917 | — | — | — |
| `SA` (v1) | 0.151617 | — | — | — |
| `SAB` (latent, aligned) | **0.130913** | 0.143701 | 0.137307 | **0.012788** |
| `SAM` (latent, marginal MLP) | 0.134182 | 0.145587 | 0.139885 | 0.011405 |
| `SABI` (latent null) | 0.145379 | 0.140982 | 0.143180 | 0.004397 |
| `SAE` (explicit, no B) | 0.144418 | — | — | — |
| `SABE` (explicit, aligned) | 0.143301 | **0.139105** | 0.141203 | 0.004195 |
| `SABEI` (explicit null) | 0.139111 | — | — | — |
| *B-Bag* (frozen external) | 0.127382 | 0.120229 | 0.123806 | — |
| *B-Full* (frozen external) | 0.119818 | 0.118126 | 0.118972 | — |
| *A2* (frozen external) | 0.121694 | 0.122134 | 0.121914 | — |

Reading notes:

* the **latent `S` route is strongly seed-unstable** (`SAB` range 0.0128,
  `SAM` 0.0114) while the **explicit `S` route is stable** (`SABE` 0.0042,
  `SABI` 0.0044);
* `SAB` seed 0 (0.130913) is the best number produced by this round but is
  **still worse than every frozen external reference** (B-Bag 0.127382,
  A2 0.121694, B-Full 0.119818); at seed 1 `SAB` (0.143701) is worse than
  B-Bag by **+0.023472**. There is **no competitive FSAR-SAB advantage** over
  the external baselines at either seed.

## 7. Answer A — historical replication (`SAB < SAM`)

| seed | `SAB` | `SAM` | `SAM − SAB` |
|---|---:|---:|---:|
| 0 | 0.130913 | 0.134182 | **+0.003270** |
| 1 | 0.143701 | 0.145587 | **+0.001886** |

Both seeds have `SAM > SAB`, so the *direction* replicates. Two caveats that
must travel with this answer:

1. the margin is small and **shrinks with the seed** (0.003270 → 0.001886; the
   seed-1 margin is below the soup's own seed-to-seed movement), and the
   absolute level of both modes degrades sharply from seed 0 to seed 1
   (`SAB` 0.1309 → 0.1437);
2. `SAM` is **not** a matched capacity control for `SAB`: `SAM` has **no aligned
   channel at all** (`edge_role_mlp` / `binding_fuse` / `node_role_projection`
   do not exist — `capacity_mlp` is a separate 663-|w| MLP; §12), and its total
   parameter count differs by 56. So `SAM − SAB` measures "aligned B present vs
   absent/marginal-only", not "aligned vs operator-matched null". Answer A is a
   *historical replication of a previously logged comparison*, not new evidence
   about binding.

**Status: replicated (direction), single-claim strength only.**

## 8. Answer B — clean binding (`SAB` vs `SABI`)

`Δ_B^latent = MAE(SABI) − MAE(SAB)` (positive ⇒ aligned better):

| seed | `SAB` | `SABI` | `Δ_B^latent` |
|---|---:|---:|---:|
| 0 | 0.130913 | 0.145379 | **+0.014466** |
| 1 | 0.143701 | 0.140982 | **−0.002719** |
| mean | 0.137307 | 0.143180 | **+0.005874** |

Pre-registered gate: *both seeds* `Δ_B > 0` **and** mean `≥ 0.001`.
**`both_seeds_positive = false` → the clean-binding claim is NOT supported.**

The seed 0 value looks large, but §12 shows it is confounded. The edge-matched
stratification (post-hoc, exploratory) is unambiguous:

| seed | `SAB` edge sub-branch | `SABI` edge sub-branch | edge-matched? | `Δ_B` |
|---|---|---|---|---|
| 0 | **alive** (\|w\| 92.4) | dead (0.0) | **no** | +0.014466 |
| 1 | dead (1.6e-36) | dead (0.0) | **yes** | **−0.002719** |

The only seed on which the two channels have the *same* functional form (seed 1)
gives a **negative** increment: the assignment-free null is *better*. Seed 0's
positive number contains the edge sub-branch's own contribution and is an
**upper bound** on the pairing effect.

**Status: NOT supported.** The aligned-moment increment beyond an
operator-matched, active-capacity-matched, assignment-independent null is
**not seed-stable**; the clean comparison is negative.

## 9. Answer C — structural explicitness (can the latent topology-GNN `S` be replaced?)

`S`-only comparison, no `B` anywhere — the cleanest apples-to-apples:

| pair | seed 0 | `Δ` |
|---|---:|---:|
| `SA` (latent `S`, FSAR-v1 frozen) | 0.151617 | — |
| `SAE` (explicit `S`) | **0.144418** | **−0.007199** |

The explicit local structural basis **beats** the latent topology-GNN `S` at
seed 0 by 0.0072 — i.e. the latent message-passing local `S` is doing **no
useful work** that the 11-D node / 15-D edge explicit basis cannot do better.

`S+B` comparison:

| seed | `SAB` | `SABE` | `SABE − SAB` | band |
|---|---:|---:|---:|---|
| 0 | 0.130913 | 0.143301 | +0.012388 | large cost |
| 1 | 0.143701 | 0.139105 | **−0.004595** | near parity (explicit better) |
| mean | 0.137307 | 0.141203 | +0.003896 | moderate |

The seed-0 "large cost" is **not** an explicit-basis defect: it is the seed-0
latent `SAB` being an unusually good *and rich-`B`* run (edge sub-branch alive,
`B` effective rank 15.94, norm 2.658) while every explicit run has a dead edge
sub-branch (`B` rank 4.25–6.18). Excluding that artefact, the explicit route is
competitive to better.

Seed stability: explicit `SABE` range **0.004195** vs latent `SAB` range
**0.012788** (~3× tighter), and `SABE` seed 1 is the best non-seed-0 number in
the whole round (0.139105).

**Status: the explicit local structural basis is a viable replacement for the
latent topology-GNN local `S` — better at seed 0 on the `S`-only comparison,
~+0.0039 mean on `S+B` (moderate band) with the cost concentrated in the one
seed where the latent `B` kept an extra sub-branch.**

## 10. Answer D — explicit binding (`SABE` vs `SABEI`)

`Δ_B^explicit = MAE(SABEI) − MAE(SABE)`:

| seed | `SABE` | `SABEI` | `Δ_B^explicit` | edge-matched |
|---|---:|---:|---:|---|
| 0 | 0.143301 | **0.139111** | **−0.004190** | **yes** (both edge-dead) |

The explicit route's aligned binding is **worse** than its operator-matched null
on the one seed that was run, and both members of the pair have the same
(dead) edge sub-branch, so this is a *clean, function-class-matched* negative.

Note the pre-registered Wave-4 trigger did fire
(`SABE 0.143301 < SAE − 0.001 = 0.143418`, by 0.000117), which is why `SABEI`
seed 0 exists at all. The Wave-5 trigger (`SABE < SABEI − 0.001`, i.e.
`0.143301 < 0.138111`) did **not** fire, so no seed-1 explicit-null / `SAE`
seed-1 runs were bought — consistent with §24 of the pre-registration
("explicit seed 0 already > implicit SAB + 0.005 with no special mechanism
advantage ⇒ do not buy Wave 5").

**Status: NOT supported.** The explicit aligned binding carries no increment over
its assignment-free, parameter-matched null; the null is better.

## 11. Mechanism finding — the `B` edge sub-branch is annihilated, seed-dependently

`checkpoint_audit` (new, reproducible stage) reports the |weight| sum of every
`B` sub-branch. Frozen states:

| state | `node_role_projection` | `edge_role_mlp` | `edge_attribute_mlp` | `edge_role_projection` | `binding_fuse` |
|---|---:|---:|---:|---:|---:|
| **FSAR-v1 `SAB` seed 0** | 105.33 | **92.43** | **55.99** | **11.78** | 342.31 |
| **FSAR-v1 `SAB` seed 1** | 81.96 | **1.6e−36** | **8.0e−37** | **5.2e−37** | 240.48 |
| FSAR-v1 `SAM` seed 0/1 | absent | absent | absent | absent | absent (`capacity_mlp` 661.8) |
| route `SABI` seed 0 / 1 | 56.16 / 37.70 | 0.0 | 0.0 | 0.0 | 150.84 / 123.97 |
| route `SABE` seed 0 / 1 | 54.60 / 47.13 | 0.0 | 0.0 | 0.0 | 293.02 / 249.72 |
| route `SABEI` seed 0 | 40.19 | 0.0 | 0.0 | 0.0 | 165.55 |
| route `SAE` seed 0 | absent | absent | absent | absent | absent |

Consequences:

1. This is the repo's documented **near-zero-gradient + Adam+L2 primitive
   annihilation** (`weight_decay=1e-5` on a parameter whose task gradient
   vanishes ⇒ the L2 term alone drives the Adam step to ≈ ±lr and the weight
   decays to underflow, ~1e-36…1e-40).
2. **It happens in FSAR-v1 itself, seed-dependently** (`SAB` seed 0 alive,
   seed 1 dead) — it is *not* introduced by this round's code. Bit-for-bit
   reproduction of both v1 `SAB` soups (§4) proves that.
3. `SAM`'s "binding" modules are **absent by design**, not annihilated; its 663
   `capacity_mlp` is alive. `SAM` is therefore not a capacity-matched control
   for `SAB`.
4. Interpretation rule used throughout: a `Δ_B` pair is only a clean
   aligned-vs-null comparison when both members have the **same** edge-sub-branch
   fate. Only two such pairs exist — latent seed 1 (negative) and explicit
   seed 0 (negative).

## 12. Witness — the null is *exactly* assignment-invariant

Fixed-centre attribute shuffle, eval-only, `witness` stage (soup states):

| mode | seed | Δ`A` | Δ`S` | Δ`B` | Δprediction |
|---|---|---:|---:|---:|---:|
| `SAB` | 0 | 0.0 | 0.0 | 0.3567 | **1.0178** |
| `SAB` | 1 | 0.0 | 0.0 | 0.1353 | 0.2535 |
| `SABE` | 0 | 0.0 | 0.0 | 0.1006 | 0.3864 |
| `SABE` | 1 | 0.0 | 0.0 | 0.0708 | 0.2637 |
| `SABI` | 0 / 1 | 0.0 | 0.0 | **0.0** | **0.0** |
| `SABEI` | 0 | 0.0 | 0.0 | **0.0** | **0.0** |

Correctness witnesses all hold: `A` and `S` are exactly stable under attribute
relabelling (assignment moves *only* `B`), and both assignment-free nulls are
**exactly** invariant (`Δprediction = 0.0`), which is the direct empirical
confirmation of the §5.2 operator construction.

## 13. Frozen interventions — dependency / reliance diagnostics, **not causal estimates**

Full 1,000-molecule official valid, soup states. `no_X` = the channel replaced
by a fixed non-informative value; `shuffle` = assignment relabelled. These are
**reliance diagnostics only** — the channels are not additively separable, so
the numbers are not additive causal contributions.

| mode | seed | true | `Δ` no-`A` | `Δ` no-`S` | `Δ` no-`B` | `Δ` shuffle |
|---|---|---:|---:|---:|---:|---:|
| `SAB` | 0 | 0.130913 | +2.2114 | +0.2596 | +0.4346 | +0.8421 |
| `SAB` | 1 | 0.143701 | +1.3591 | +0.1118 | +0.2426 | +0.1365 |
| `SABE` | 0 | 0.143301 | +1.3173 | +0.1988 | +0.1844 | +0.2089 |
| `SABE` | 1 | 0.139105 | +3.9470 | +0.1741 | +0.3955 | +0.1104 |
| `SABI` | 0 | 0.145379 | +3.3471 | +0.2163 | +0.0376 | **0.0000** |
| `SABI` | 1 | 0.140982 | +3.2240 | +0.2692 | +0.1205 | **−0.0000** |
| `SABEI` | 0 | 0.139111 | +2.1847 | +0.2120 | +0.0456 | **0.0000** |

* `A` remains by far the dominant reliance channel in every mode (the known
  "A starvation" is structural, not a bug), `S` is the weakest.
* The nulls' `B` reliance is small (`+0.038` / `+0.046` / `+0.121`) and their
  shuffle dependence is **exactly zero** — consistent with §12.
* `SAB` seed 0 is the most `B`-reliant mode (`Δ` no-`B` +0.4346, shuffle
  +0.8421), i.e. the seed where its `B` kept the edge sub-branch.

## 14. Channel diagnostics (soup)

| mode | seed | `B` norm / cross-mol std / eff. rank | `S` norm / eff. rank |
|---|---|---:|---:|
| `SAB` | 0 | 2.658 / 0.487 / **15.94** | 1.794 / 6.66 |
| `SAB` | 1 | 1.495 / 0.274 / 8.71 | 1.925 / 7.62 |
| `SABI` | 0 | 0.998 / 0.157 / 7.43 | 1.887 / 7.16 |
| `SABE` | 0 | 0.977 / 0.160 / 6.18 | 1.074 / 5.31 |
| `SABEI` | 0 | 0.708 / 0.101 / **4.25** | 1.097 / 5.29 |

`B` richness tracks the edge-sub-branch state (alive ⇒ rank ≈ 16; dead ⇒ rank
4–9). All `A` ranks 10.4–11.8.

## 15. Explicit basis diagnostics (soup)

* Node basis groups (11 coordinates: root indicator, 2 shell indicators,
  induced degree, neighbour-by-shell, rooted walks 1–3) are all non-degenerate;
  observed nonzero frequency: degree 1.000, `nbr1`/`w2` 0.653, `sh2` 0.486,
  `nbr2` 0.416, `w3` 0.416, `nbr0`/`sh1`/`w1` 0.351, root/`sh0` 0.163.
* Edge basis groups (15 coordinates): `common_neighbours` is **nearly constant**
  (group std 0.069) — a near-dead coordinate worth noting for any future basis
  revision, but not a performance claim.
* The learned first-layer column norms confirm the §11 finding independently:
  `edge_role_mlp_first_layer` per-group norms are **exactly 0.0** in `SABE`
  seed 0/1 and `SABEI` seed 0, while `structure_pool` (`S_explicit`) blocks are
  all alive (2.2–3.9).
* Unit-tested invariants: no message passing on the explicit path (AST check),
  chemistry purity, relabel equivariance, no mixed bypass.

## 16. Matched external references (frozen, never used as input)

| reference | seed 0 | seed 1 |
|---|---:|---:|
| B-Bag | 0.127382 | 0.120229 |
| B-Full | 0.119818 | 0.118126 |
| A2 | 0.121694 | 0.122134 |

`SAB` is worse than B-Bag by **+0.003531** (seed 0) and **+0.023472** (seed 1).
The round produces **no** result that beats the frozen external baselines.

## 17. Durable conclusion (split, no merged verdict)

* **Historical replication (A):** `SAB < SAM` replicates in direction on both
  seeds (+0.003270 / +0.001886), but `SAM` has no aligned channel at all
  (`capacity_mlp` only), so this is a replication of a logged comparison, not
  evidence about binding; the margin is small relative to the soup's seed
  spread.
* **Clean binding (B):** **not supported.** `Δ_B^latent` = +0.014466 / −0.002719
  (mean +0.005874); the pre-registered both-seeds-positive gate fails, and the
  only edge-sub-branch-matched pair is **negative**.
* **Structural explicitness (C):** **supported, with a caveat.** The explicit
  local structural basis beats the latent topology-GNN `S` on the `S`-only
  comparison (`SAE − SA` = **−0.007199**) and is ~3× more seed-stable; on `S+B`
  the mean cost is +0.003896 (moderate band) and is concentrated in the seed
  where the latent `B` retained an extra sub-branch. The latent message-passing
  local `S` is dispensable at this scale.
* **Explicit binding (D):** **not supported.** `Δ_B^explicit`(seed 0) =
  **−0.004190** on a function-class-matched pair.
* **Mechanism:** the `B` edge sub-branch is annihilated by Adam+L2 in most runs
  and is **seed-unstable in FSAR-v1 itself** (alive at seed 0, dead at seed 1),
  which is why answer B's seed 0 and the "large cost" of answer C are
  confounded. Any future binding claim on this code must either (a) report the
  edge-sub-branch state of every checkpoint, or (b) exclude the edge sub-branch
  from `B` so that aligned and null have identical function class by
  construction.
* **Competitiveness:** no mode in this round beats the frozen external
  baselines (B-Bag / B-Full / A2) on either seed.

## 18. What is explicitly *not* claimed

* No claim that aligned binding is harmful in general — only that **this**
  operator-matched, parameter-matched, assignment-independent null is not
  beaten by the aligned moment channel on any edge-matched seed.
* No additive causal decomposition from §13 (reliance diagnostics only).
* No official-test number, no generalization claim beyond official valid.
* No claim about the explicit basis at scales other than ZINC-12K / this
  protocol; no width, depth, attention, DeepSets, auxiliary-loss or
  architecture sweep.

## 19. Records and reproducibility

New / changed code (all committed):

* `tracks/ksvd/experiments/luyin16/fsar_route_recheck.py` — mode tables,
  `independent_moments`, `FSARLatentRouteEncoder`, `FSARExplicitRouteEncoder`,
  `PatchPathFSARRouteModel`, `module_groups`, `parameter_breakdown`,
  `active_path_from_grads`, `active_path_accounting`,
  `fsar_route_collate` / `make_fsar_route_loader`.
* `tracks/ksvd/experiments/luyin16/zinc_fsar_route_recheck.py` — stages
  `params preprocess sanity smoke gradient_audit train soup run diagnostics
  checkpoint_audit witness interventions explicit_basis_diagnostics gate decide
  report`.
* `tracks/ksvd/tests/test_fsar_route_recheck.py` — 17 tests.
* `tracks/ksvd/notes/fsar_route_recheck_preregistration.md` — 18 sections + §19
  Wave-6 addendum.

Bugs found and fixed during the round (all eval-only, no retraining affected):
`explicit_basis_diagnostics` `Path`/`str` concatenation; `gate()` indexing
`active_path` on an already-flat audit row; `decide()` missing the frozen
FSAR-v1 `SA` fallback and the four-way answer block.

Result artifacts: `tracks/ksvd/results/fsar_route_recheck/`
(`soup_route_*`, `diagnostics_route_*`, `checkpoint_audit_route_*`,
`witness_route_*`, `interventions_route_*`, `explicit_basis_diagnostics_route_*`,
`gradient_audit*.json`, `gate.json`, `decision.json`, `report.json`,
`sanity.json`, `smoke_*.json`, `runs/`, `curves/`, `states/`, `soup_states/`),
plus the Wave-1/6 FSAR-v1 outputs in `results/fsar/`
(`soup_fsar_sab_seed1.json`, `soup_fsar_sam_seed1.json`, and the seed-1
`soup_states` / `states`).

Reproduce:

```
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck preprocess
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck sanity
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck gradient_audit --all-modes
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck run --mode SABI --seed 0 --device cuda --deterministic   # etc.
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck checkpoint_audit --mode SAB --seed 1 --state soup
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck gate
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck decide
python -m tracks.ksvd.experiments.luyin16.zinc_fsar_route_recheck report
```

## 20. Revisit if

* a **seed-1 explicit-null** (`SABEI`) run is funded and its pair with `SABE` is
  edge-matched and again negative — that would make answer D two-seed;
* `B` is redefined to **exclude the edge sub-branch** (or the sub-branch is
  protected from annihilation), so that aligned and independent differ *only* in
  the moment construction — then answer B can be re-asked on truly
  function-class-matched pairs at both seeds;
* an explicit-basis revision addresses the near-constant `common_neighbours`
  coordinate and the 0.163-frequency `root` / `sh0` coordinates; not by widening
  the basis ad hoc;
* any claim wants to be competitive: the external B-Bag / B-Full / A2 baselines
  must be beaten first, at both seeds, with the edge-sub-branch state reported.

Not by: A width sweeps, DeepSets / Transformer / attention variants, auxiliary
binding losses, reopening `A_exact`, or reading the official test.
