# Analysis — ZINC No-Ring Dictionary × Generic Topology Cross (DTX-v0)

Round: **zinc-no-ring-dictionary-topology-cross-v0** (DTX-v0).
Preregistration: `tracks/ksvd/notes/zinc_no_ring_dictionary_topology_cross_v0_preregistration.md`
frozen at commit `7c28be1`.
Formal-run commit: `3810d73688805039f4ffccab1a90cf769d0e4639`
(`7c28be1` + one audit-only bugfix, see §1.3).
Device: remote `a100-2`, 2 × `A100-SXM4-40GB`, GPU 0 = Arm M, GPU 1 = Arm A.
Status: **two** seed-0 full runs (the round's whole budget); the causal gate
fires in the **negative** direction; round closed, **no** seed 1, no sweep, no
rescue.

Official ZINC **test was never loaded** in any stage
(`official_test_loaded=false` on every artifact).

## 0. What the round asked

> With the typed/parent identity channels permanently deleted, does an **aligned
> joint statistic** `J_align = (1/N) Σ_i alpha_i ⊗ s_i` — a dictionary
> assignment `alpha_i` crossed with a completely **generic, untyped, label-free**
> topology role `s_i` — beat a **matched marginal-only control**
> `J_indep = mu_alpha ⊗ mu_s`, **without ever feeding any explicit ring/cycle
> context** into the new branch?

One causal comparison (MARGINAL vs ALIGNED), one seed, fixed 240 epochs,
fixed Top-5 soup, no HPO.

## 1. Architecture actually implemented

Standalone module `tracks/ksvd/experiments/luyin16/zinc_no_ring_dictionary_topology_cross.py`
(not a `PatchPathModel` subclass), so the strict-static contract is structural.

```text
x_i        = [patch_cont | patch_context]              # runtime 146
h_i        = SiLU MLP(146 -> 64 -> 48)                 # 48D local state

x_dict_i   = patch_cont minus cycle-rank coord         # runtime 145
z_dict_i   = SiLU MLP(145 -> 64 -> 32)
q_i        = l2_normalize(z_dict_i)                    # 32D query
D_k        = l2_normalize(dictionary_atom_k)           # K = 64, rank = 32
tau        = 0.05 + 0.95*sigmoid(tau_logit)            # init 0.20
alpha_i    = softmax(q_i @ D_norm^T / tau)             # 64D

s_i        = generic_topology_role(patch_i)            # 8D, ring-blind

mu_alpha   = mean_i alpha_i                            # 64D
mu_s       = mean_i s_i                                # 8D
J_align    = mean_i (alpha_i outer s_i)                # 512D
J_indep    = mu_alpha outer mu_s                       # 512D
e_cross    = SiLU Linear(512 -> 32)

u_i        = pair_projection(h_i)                      # 48 -> 16
gate_ij    = 1 + tanh(distance_gate[bucket_ij])        # Embedding(5,16)
rel_ij     = relation_encoder(r_ij)                    # MLP(23 -> 32 -> 16)
pair_input = [u_i+u_j | |u_i-u_j| | (u_i*u_j)*gate | rel]   # 64
q_ij       = MLP(64 -> 64 -> 16)                       # ONCE per pair

R_final    = [unary 97 | pair 165 (5 buckets) | global 32 | topo 8
              | mu_alpha 64 | mu_s 8 | e_cross 32]     # 406
y_hat      = GenericReader(406 -> 16 -> 16 -> 1)
```

### 1.1 Identity channels removed (as in SRDA)

No typed 16D, no parent 8D, no corrected/historical/certificate/fixed-code
lookup. The only `nn.Embedding` is the 5-row distance-bucket table
(`identity_input_audit.only_distance_gate_embedding = true`,
`forbidden_key_hits = []`, `state_dict_key_count = 42`).

### 1.2 The new branch is ring-blind by construction

* Dictionary query drops exactly one coordinate: `patch_cont[143]`, the explicit
  **cycle rank / cyclomatic number** `max(E−V+1,0)/V` of the induced radius-2
  patch. Proven against the real `zpp._shell_descriptor` implementation with
  synthetic graphs (`tree_chain → 0`, `triangle → 1/3`, `square → 1/4`,
  `fused_triangles → 2/5`, `hexagon → 0`); no other coordinate is
  cycle-derived (`cycle_rank_coordinate_audit.passed = true`,
  `other_cycle_derived_coordinates = []`). All chemistry/size/degree/shell
  coordinates are retained.
* `s_i ∈ R^8` is computed from distances + the untyped induced edge set only:
  `root_degree_fraction, shell1_occupancy, shell2_occupancy,
  shell1_edge_density, shell1_shell2_edge_density, shell2_edge_density,
  shell2_multi_parent_fraction, same_shell_incidence_fraction`. Bounded
  `[0,1]`, permutation-invariant (bit-exact under node relabel), deterministic.
* AST source audit of the role extractor and `DTXModel.encode` finds **no**
  `ring|cycle|basis|aromatic|fused|spiro|cycle_rank` symbol and no
  typed/parent/structural token read (test gate G).
* `alpha_i` never modifies `h_i`; `s_i` never reads `alpha_i` or the global
  hinge; the cross statistic is a graph-level sufficient statistic that cannot
  write back to any local state.

### 1.3 Preregistration vs. formal-run commit

`7c28be1` froze architecture and gates. The single follow-up commit `3810d73`
touched **only the audit gate**: `static_contract_checks` had assumed the model
passed in was Arm A, so the marginal run aborted on the audit's own
`aligned_alignment_sensitive` assertion even though its real behaviour
(marginal invariance) was correct. The fix builds **both** an Arm-M and an
Arm-A probe from the trained weights and checks each independently. No frozen
mathematical object, width, hyperparameter, loss, protocol or threshold was
changed. Both formal runs use `3810d73`.

### 1.4 Parameter / width audit (runtime, `parameter_audit.json`)

| module | params |
|---|---|
| local encoder (146→64→48) | 12,528 |
| dictionary input encoder (145→64→32) | 11,424 |
| dictionary atoms + tau (64×32 + 1) | 2,049 |
| cross projection (512→32) | 16,416 |
| pair projection (48→16) | 768 |
| relation encoder (23→32→16) | 1,360 |
| pair encoder (64→64→16) | 5,328 |
| global encoder | 3,136 |
| topology hinge encoder | 552 |
| graph head (406→16→16→1) | 6,801 |
| distance gate embedding (5×16) | 80 |
| **total** | **60,442** |

Runtime widths, attested from real tensors at both arms
(`runtime_widths.widths_consistent = true`):
`local_in 146 → h 48 ; dict_in 145 → q 32 ; alpha 64 ; s 8 ; J 512 ;
e_cross 32 ; pair_input 64 ; relation_input 23 ; R 406`.
Arm-M and Arm-A parameter counts are equal (60,442), state-dict keys/shapes
equal and `shared_init_bit_identical = true`.

## 2. Strict-static hard gates (all pass)

CPU integrity stage and the A100 contract stage both pass
(`integrity_gates.json`, `smoke_gpu*.json`):

| gate | result |
|---|---|
| `center_context=False`, `center_update=None` | true |
| forward succeeds with `_pool_pairs_to_centres` raising | `forward_without_pair_to_center = true` |
| pair_encoder / relation_encoder / dict_encoder / dictionary calls per forward | 1 / 1 / 1 / 1 |
| `h`, `alpha`, dict embedding bit-identical under topology mutation | true (`max|Δalpha| = 0.0`) |
| Arm M invariant to within-graph `s` permutation | `marginal_joint_shift = 7.45e-09`, `marginal_pred_shift = 7.45e-09` |
| Arm A sensitive to within-graph `s` permutation | `aligned_joint_shift = 1.140e-02`, `aligned_pred_shift = 2.30e-05` (≫ M) |
| `J_indep` **is** the outer product `mu_alpha ⊗ mu_s` (Arm M) | true |
| `J_align` **is not** the outer product (Arm A) | true |
| pair-order / endpoint-swap / patch-relabel prediction invariance | `1.3e-07 / 0.0 / 1.5e-08` |
| repeat-forward noise floor / identity-mutation shift | `0.0 / 0.0` |
| all required branches receive gradient, finite | `required_positive_ok = true` |
| tau gradient finite | `1.1e-05` |

Gradient norms at the smoke stage (non-vacuous training):
head 0.838, pair_encoder 0.094, local 0.041, global 0.038, relation 0.028,
pair_projection 0.015, topology 0.0085, cross_projection 0.0062, dict_encoder
0.0031, dictionary atoms 0.0012.

## 3. Deterministic A100 smoke (GPU0 vs GPU1)

`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
Same seed, same 2-epoch / 256-train / 128-valid trace on both GPUs:

```text
parameter_init_sha256 GPU0 == GPU1 : d4568d5badf9f507…   (identical)
trace GPU0 == GPU1 : [[1, 1.467997, 1.395119], [2, 1.447125, 1.377680]]
best valid (both)  : 1.377679516808712
contract / gradient / widths : pass / pass / consistent
peak GPU memory    : 181.3 MB (both)
```

The deterministic regime is reproducible across devices.

## 4. Formal runs (two arms, seed 0, 240 epochs, no early stop)

| | Arm M (MARGINAL) GPU0 | Arm A (ALIGNED) GPU1 |
|---|---|---|
| best valid MAE | **0.148671** @238 | 0.154001 @232 |
| Top-5 soup MAE | **0.144330** | 0.150357 |
| soup members | 232, 233, 235, 236, 238 | 216, 224, 227, 231, 232 |
| wall clock | 1353.4 s | 1673.0 s |
| peak GPU memory | 191.0 MB | 192.5 MB |

```text
Delta_align (best) = M_best - A_best = -0.005330
Delta_align (soup) = M_soup - A_soup = -0.006027
```

The aligned cross is **worse than** the matched marginal control on both
checkpoint statistics.

## 5. Outcome — preregistered Case D

```text
case                     = D
verdict                  = NO_ALIGNED_CROSS_SIGNAL
Delta_align_soup         = -0.006027   (gate: >= +0.004 for Case A, > 0 for C)
Delta_align_best         = -0.005330   (gate: >= +0.003 for Case A)
base_regression_confound = false       (M_soup 0.144330 <= 0.1450)
mechanism_clear          = true        (see §6)
seed1_authorized         = false
```

The direction is negative, so neither Case B (`ALIGNMENT_SIGNAL_BUT_BASE_NOT_COMPETITIVE`)
nor Case C (`DIRECTIONAL_ONLY`) applies. The marginal control is itself healthy
(no base regression), so this is a clean causal negative, not a confound.

## 6. Mechanism — the aligned statistic is used, but it hurts

Inference-only interventions on the **Arm A soup** checkpoint (official valid,
`gradient_used=false`, repeat-noise floor `0.0`):

| intervention | valid MAE after | mean \|Δpred\| | max \|Δpred\| | clear |
|---|---|---|---|---|
| none (soup) | 0.150357 | — | — | — |
| `J_align → J_indep` (alignment removal) | 0.168766 | 0.06494 | 0.326 | true |
| within-graph shuffle of `s_i` | 0.184156 | 0.08716 | 0.480 | true |

Both interventions are far above the `0.010` mean-shift and `20×` noise gates
(`mechanism_clear = true`), and the Arm-A best checkpoint behaves identically
(0.174156 / 0.189081, mean shifts 0.0658 / 0.0882).

Interpretation: `J_align` is a genuinely **load-bearing** pathway in Arm A — the
aligned-trained weights depend on the `alpha_i ↔ s_i` pairing (removing it costs
`+0.018` MAE, shuffling `s` costs `+0.034` MAE). Yet Arm A still loses to the
Arm-M control whose only difference is that it discards exactly that pairing.
So the round's causal claim is **rejected in the strong form**: adding the
aligned dictionary×topology interaction is not merely neutral, it is harmful
at the matched-budget scale. The marginal control already extracts whatever
graph-level value the dictionary composition and topology composition carry;
the within-graph alignment term appears to add optimization burden / variance
without a compensating signal.

## 7. Dictionary health (report only, never a gate)

| metric | Arm M | Arm A |
|---|---|---|
| active atoms | 48 / 64 | 45 / 64 |
| argmax-used atoms | 51 | 41 |
| effective atom count | 2.47 | 2.88 |
| top-8 assignment mass | 0.979 | 0.969 |
| max average atom mass | 0.191 | 0.260 |
| mean assignment entropy | 0.903 | 1.058 |
| tau final | 0.0823 | 0.1029 |
| coherence (mean abs) | 0.240 | 0.306 |
| mean role coord (identical, precomputed) | — | — |
| mean role norm | 1.038 | 1.038 |

The dictionary is trained end-to-end and used (gradients flow, atoms move), but
in **both** arms it is **highly concentrated**: effective atom count ≈ 2.5–2.9
out of 64 and top-8 mass ≈ 0.97. This is a partial collapse, not a healthy
distributed code. It is reported as mixed dictionary health; the round does not
gate on it, but it is a caveat for interpreting the negative — the marginal
control reaches a competitive-ish MAE with a nearly degenerate dictionary, so
neither arm is a strong test of a *rich* dictionary.

## 8. Post-hoc ring analysis (explicit ring labels, explanation only)

`post_hoc_only = true`, `gradient_used = false`, `used_for_selection = false`.
Valid split, 23,083 patches. Coverage: `ring_any 0.636`, `ring5 0.186`,
`ring6 0.452`, `multi_fused 0.040`, `ring_boundary 0.166`, `non_ring 0.364`.

**The generic role vector is ring-relevant without being told about rings.**
Per-coordinate `Δ` (ring mean − complement mean), identical for both arms since
`s` is precomputed:

| role | ring_any | ring5 | ring6 | multi_fused | boundary |
|---|---|---|---|---|---|
| root_degree_fraction | +0.026 | −0.004 | +0.015 | −0.020 | −0.036 |
| shell1_occupancy | +0.026 | −0.004 | +0.015 | −0.020 | −0.036 |
| shell2_occupancy | −0.026 | +0.004 | −0.015 | +0.020 | +0.036 |
| shell1_edge_density | +0.012 | −0.009 | −0.014 | −0.006 | −0.009 |
| **shell1_shell2_edge_density** | **−0.271** | −0.138 | −0.179 | −0.193 | +0.126 |
| **shell2_edge_density** | **+0.096** | +0.336 | −0.103 | +0.042 | −0.070 |
| shell2_multi_parent_fraction | +0.001 | −0.001 | −0.001 | +0.002 | −0.001 |
| **same_shell_incidence_fraction** | +0.054 | +0.173 | −0.053 | +0.061 | −0.039 |

So `s` encodes ring structure (chiefly via S1–S2 edge density and within-shell
incidence) purely from untyped local topology — which is what made the
ring-blind cross plausible in the first place.

**The dictionary also learns ring-correlated environments without ring input.**
Largest per-atom ring enrichment (`Δ = mean mass on ring_any − on non_ring`):

* Arm A: atom 13 `+0.168` (ring6 `+0.197`, boundary `−0.130`), atom 24 `−0.154`,
  atom 43 `+0.074` (multi_fused `+0.165`).
* Arm M: atom 24 `−0.236`, atom 38 `+0.169` (ring6 `+0.252`, ring5 `−0.153`),
  atom 28 `−0.054`.

The two arms specialise **different** atoms onto ring strata, consistent with
the alignment term re-pairing dictionary atoms with topology coordinates — but
in neither case does that specialisation translate into better MAE.

Top aligned joint entries (`J_align[k,p]` mean, ring `Δ`): atom 13 ×
`shell2_occupancy` 0.152 (+0.087), atom 13 × `shell1_shell2_edge_density` 0.144
(+0.011), atom 13 × `shell1_occupancy` / `root_degree_fraction` 0.105 (+0.081),
atom 24 × `shell1_shell2_edge_density` 0.089 (−0.149). The strongest aligned
mass sits exactly on the `alpha × (S1–S2 edge density / shell occupancy)` ring
axis — the interaction is real and ring-correlated — it just does not help.

## 9. Direct answers to the round's questions

### A. Dictionary health / ring-blindness

1. **Are the typed/parent identity channels deleted?** Yes.
   `identity_input_audit.forbidden_key_hits = []`, the only embedding is the
   5-row distance gate, `state_dict_key_count = 42`.
2. **Is the dictionary query free of explicit cycle labels?** Yes. Exactly one
   coordinate is deleted: `patch_cont[143]`, the cycle rank/cyclomatic number;
   `dictionary_input_width = 145`, `cycle_rank_coordinate_audit.passed = true`,
   `other_cycle_derived_coordinates = []`.
3. **Does the new cross branch contain any cycle-enumeration / ring symbol?**
   No. AST audit of the role extractor and `encode` finds no ring/cycle/basis/
   aromatic/fused/spiro symbol and no enumeration call.
4. **Is the dictionary input only the static local descriptor, never topology?**
   Yes: `x_dict_i = patch_cont(146)[delete 143] → 145`; it never sees `s_i`.
5. **Does `alpha_i` ever modify `h_i` (residual / gamma)?** No. Under topology
   mutation `h`, `alpha` and the dict embedding are bit-identical
   (`max|Δalpha| = 0.0`); there is no dictionary residual path.
6. **Does the dictionary/roles read the explicit global hinge?** No. The global
   topology hinge (25→16→8) only enters the final graph readout; it is neither in
   `dict_encoder`, `alpha`, `s`, nor `J`.
7. **Is the dictionary healthy?** Mixed. Trained and used, but highly
   concentrated (effective atoms 2.5–2.9, top-8 mass ≈0.97, coherence 0.24–0.31);
   not collapsed to 1–2 atoms, not distributed. See §7.
8. **Does the dictionary learn ring-correlated environments without ring
   input?** Yes, post-hoc: atom 13 (Arm A) `+0.17` on ring_any, atom 38 (Arm M)
   `+0.17` ring_any / `+0.25` ring6. See §8.

### B. Aligned-cross mechanism

9. **Do both arms supply `mu_alpha` and `mu_s`?** Yes — they are in `R_final`
   for both arms; the arms differ *only* in `J_align` vs `J_indep`.
10. **Is `J_indep` exactly `mu_alpha ⊗ mu_s` and `J_align` exactly the mean
    outer product?** Yes: `marginal_joint_is_outer_product = true`,
    `aligned_joint_is_outer_product = false`.
11. **Is Arm M permutation-invariant and Arm A sensitive?** Yes. Within-graph
    `s` permutation shifts `J_indep`/prediction by `7.45e-09` (Arm M) but
    `J_align` by `1.140e-02` and prediction by `2.30e-05` (Arm A) at random init
    (CPU); the trained-model shuffle intervention is far larger (§6).
12. **Are the two arms parameter- and init-matched?** Yes: 60,442 params each,
    equal state-dict keys/shapes, `shared_init_bit_identical = true`.
13. **Does every required branch receive gradient?** Yes,
    `required_positive_ok = true`; all listed norms finite and > 0.
14. **Does the trained Arm A actually use `J_align`?** Yes. Replacing it with
    `J_indep` moves predictions by mean `0.0649` and raises valid MAE
    `0.1504 → 0.1688`.
15. **Does it use the `alpha↔s` alignment (not just the marginals)?** Yes.
    Shuffling `s_i` within graphs moves predictions by mean `0.0872` and raises
    valid MAE `0.1504 → 0.1842`; both clear the `0.010` / `20×` gates
    (`mechanism_clear = true`).
16. **Did the strict-static contract hold in training?** Yes (both arms, CUDA):
    pair/relation/dict encoders called once, no centre update, forward without
    pair→centre, invariance gates pass (§2).
17. **Does the generic role capture ring structure without ring labels?** Yes,
    strongly (S1–S2 edge density `−0.271`, shell2 edge density `+0.096`,
    same-shell incidence `+0.054` for ring_any; §8).
18. **Which aligned interactions carry the ring signal?** Atom 13 × shell
    occupancy / S1–S2 edge density and atom 24 × S1–S2 edge density; see §8.

### C. Absolute MAE / protocol / budget

19. **Best / soup MAE per arm and `Delta_align`?** M `0.148671 / 0.144330`,
    A `0.154001 / 0.150357`; `Delta_align = −0.005330 (best) / −0.006027 (soup)`.
20. **Which preregistered case fired?** Case **D** (`Delta_align ≤ 0`) →
    `NO_ALIGNED_CROSS_SIGNAL`; `seed1_authorized = false`.
21. **Absolute MAE vs references?** Arm M soup `0.144330` vs S0 seed0 soup
    `0.140794` (`+0.0035`), S0 seed1 `0.136423` (`+0.0079`), SDPK-v0 `0.139735`
    (`+0.0046`), SRDA-v0 `0.149873` (`−0.0055`). Arm A soup `0.150357` is worse
    than every reference except SRDA. The marginal control is a legitimate,
    non-degenerate strict-static model; the aligned variant is not competitive.
22. **Was there a base-regression confound?** No: `M_soup 0.144330 ≤ 0.1450`
    (`base_regression_confound = false`), so the negative is not caused by the
    shared backbone being broken.
23. **Budget / determinism / official test / seed 1?** 2 of 2 allowed full runs
    used (`full_runs_used = 2`, no HPO/sweep/seed1); deterministic algorithms on
    both GPUs with a bit-identical cross-device smoke; official ZINC test never
    loaded in any stage.

## 10. Verdict

```text
DTX_V0_NO_ALIGNED_CROSS_SIGNAL
```

The round's central hypothesis is **rejected**. A ring-blind dictionary, a
ring-blind generic topology role, and their within-graph alignment interaction
were all implemented and verified; the aligned interaction is genuinely
load-bearing in Arm A (removing or shuffling it degrades MAE by `+0.018` /
`+0.034`), and the dictionary and role both demonstrably pick up ring structure
from untyped local topology. But the matched marginal-only control is **better**
by `0.0060` soup / `0.0053` best MAE, so the aligned cross is not merely a
null — it is harmful at equal budget. No seed 1, no sweep, no rescue, no
control arm beyond MARGINAL, official test unread. The DTX-v0 route is closed.

## 11. Artifacts

* `results/zinc_no_ring_dictionary_topology_cross_v0/parameter_audit.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/parity_gates.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/integrity_gates.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/smoke_gpu0.json`, `…gpu1.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/runs/dtx_marginal_seed0.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/runs/dtx_aligned_seed0.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/curves/dtx_{marginal,aligned}_seed0_curve.csv`
* `results/zinc_no_ring_dictionary_topology_cross_v0/interventions_aligned_{soup,best}.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/posthoc_ring_{aligned,marginal}_soup.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/analysis.json`
* `results/zinc_no_ring_dictionary_topology_cross_v0/RESULTS_SUMMARY.md`
* `results/zinc_no_ring_dictionary_topology_cross_v0/states/*.pt` (remote, untracked)
