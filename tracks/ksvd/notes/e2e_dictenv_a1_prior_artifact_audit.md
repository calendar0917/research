# E2E-DictEnv-A1 — prior-artifact audit

Round **E2E-DictEnv-A1** · protocol `e2e_dictenv_a1` · subtitle *Invariant
Attributed Dictionary Core* · study `zinc-context-gap` · branch
`exp/e2e-dictenv-a1-attributed-dictionary`.

Written **before** any A1 code, cache, dictionary fit, training or GPU use.  Its
only job is to establish, from Git-tracked artifacts, which facts are already
durable, which representations are already refuted, and therefore which design
choices A1 is *not* free to make.  Every claim below cites a Git-tracked note /
claim / decision / result file; nothing here is from memory.

Audit revision: commit of this file.  Pre-registration:
[`e2e_dictenv_a1_preregistration.md`](e2e_dictenv_a1_preregistration.md).

---

## 0. Method

1. Read the current `tracks/ksvd/STATE.yaml`, then the notes, claims, decisions
   and (locally pulled) result JSONs of the rounds named in the A1 brief:
   E2E-DictEnv-P1, E2E-DictEnv-P2-ABS (H1), SDB-v0, FEC-D1, FSAR-R2-AR0,
   FSAR-R2-AR0-EDGE, TCCD-v0, FSAB, DTX-v0.
2. Read the mentor vendor snapshot
   `experiments/luyin16/mentor_upstream/v1_20260921/` (read-only, byte-exact,
   `README.md` sha256 table) to establish what the mentor's dictionary
   formation actually consumes.
3. Convert each fact into an explicit **constraint** on the A1 object and an
   explicit **reuse** decision.
4. Record the forbidden list (§10) that A1 may not re-enter.

---

## 1. H1 — the current ZINC clean dictionary-core winner (P2-ABS)

Source: `records/claims/claim-e2e-dictenv-p2-abs-h1-slot-decoder-reaches-target-band-20260924.yaml`,
`notes/e2e_dictenv_p2_abs_analysis.md`, `notes/e2e_dictenv_p2_abs_implementation.md`,
`results/e2e_dictenv_p2_abs/final_config.json`, `results/e2e_dictenv_p2_abs/k64_diagnostic.json`.

| fact | value |
|---|---|
| formal commit | `122db5dcd45e28f5a6dc3b6314a18e1ea328deae` |
| dictionary input | pure-topology `phi_i in R^65` (FSAR-R2 explicit basis) |
| dictionary | frozen **SDB-v0** K-SVD, `65 x 32`, sha256(float32) `b0c5da98aee5795450927fcd76606281aca947448f77ab186e11de09b33dfecd` |
| coding | tied IHT, `K=32`, `s=8`, 10 steps, exact top-8, column-normalized `Dbar` |
| decoder | H1 shared compact slot decoder (node `96->64->48`, edge `48->48->32`, anchor `62->32`, fusion `368->128->48`) |
| params | 97,487 (local 82,384 + backend 15,103) |
| horizon / λ | 320 epochs; `lambda_rec = 33.95873017865987` |
| result | official-valid Top-5 soup MAE **0.12354862861608853**, members `[293,309,313,316,317]`, 2,081 s, 207 MB peak |
| official test | never loaded |
| load-bearing | yes — the dictionary/code route is load-bearing (P1 mechanism: zero-code `+0.347`, node shuffle `+0.0131`, edge shuffle `+0.0595`, all-shuffle `+0.0730`); sparse **specificity not proven** (DenseTied `0.134534` vs sparse `0.131975` in P1, `G_specific = 0.002559 < 0.003`) |

**Constraint A1-C1.** A1 inherits H1 *unchanged* except for the dictionary-input
coordinate / code source.  No decoder, optimizer, horizon, λ or backend change
is available as a "rescue" axis.

**Constraint A1-C2.** Absolute MAE is **not** the A1 question.  `0.123549` is
already in the target band, and two independent rounds (FEC-D1 `G_dict_specific
= -0.001306`, P1 `G_specific = 0.002559`) showed that a lower absolute number
does *not* establish dictionary specificity.  A1 must therefore carry an
**exact parameter-matched assignment-independent attributed control** and let
that control decide the claim.

---

## 2. E2E-DictEnv-P1 — the clean core that H1 was tuned from

Source: `records/claims/claim-e2e-dictenv-p1-clean-load-bearing-but-not-specific-20260924.yaml`,
`notes/e2e_dictenv_p1_preregistration.md`, `notes/e2e_dictenv_p1_analysis.md`.

* P1 = primitive-only, chemistry-free **end-to-end** dictionary environment:
  `62-D` zeroth-order anchor (root one-hot 28 | patch atom mass 28 | patch bond
  mass 4 | size 2), node binding `u_v = (alpha_v W_A^S) * (q_v W_A^C)/sqrt(96)`
  summed per shell, edge binding `u_uv = ([a_u+a_v; |a_u-a_v|; a_u*a_v] W_E^S) *
  (b_uv W_E^C)/sqrt(48)` summed per shell-pair, `638 -> 102 -> 48` MLP, 15-D
  pure-topology relation, 97,865 params.
* Mechanism **strong and clean**: zero-coordinate `+0.347116`, node-assignment
  shuffle `+0.013124` (gate ≥ 0.010), combined shuffle `+0.072999` (gate ≥
  0.015).  Dictionary health passed every gate (`||dL/dD|| = 0.094`, 27/32
  active atoms, init-movement 5.07, train/valid usage Spearman 0.999).
* Absolute-strong: soup `0.131975`; DenseTied `0.134534`.
* **In P1 the α ↔ chemistry correspondence is broken *after* coding**: the
  dictionary input `phi_v` is chemistry-free by construction, and `q_v` only
  enters in `node_environment`/`edge_environment` after `alpha_v` exists.
  A1 asks the strictly new question: *what if the attribute assignment instead
  participates in forming `alpha` itself* (weakening constraint 3 below in a
  controlled way).

**Constraint A1-C3 (inherited and preserved).** The label-free purity contract
(no `patch_cont`, no `atom_shell`/`bond_shell` handcrafted histograms, no
message passing, no recurrence, no pair→centre, no attention, no ring/cycle
features) stays in force.  A1 adds attributes to the *dictionary input* only as
the explicit joint statistic of §3.3/§3.4 of the pre-registration, and keeps
the post-code binding untouched.

---

## 3. P2-ABS coding diagnostic — the frozen IHT-10 is a coder defect

Source: `notes/e2e_dictenv_p2_abs_analysis.md` §5, `results/e2e_dictenv_p2_abs/k64_diagnostic.json`.

On 47,588 official-train `phi65` rows:

| dictionary | s | OMP (exact) | IHT-10 | IHT-30 | IHT-100 |
|---|---|---|---|---|---|
| SDB K32 | 8 | **1.4518e-05** | 0.010084 | 0.002856 | 0.001655 |
| K64/s8 | 8 | 3.9066e-05 | 0.026816 | 0.007372 | 0.001896 |
| K64/s12 | 12 | 4.9789e-06 | 0.015776 | 0.002141 | 0.001007 |

The frozen 10-step tied IHT is ~700× worse than exact OMP on the winning
dictionary while still hitting exact top-8, and the K64S8 arm's λ recalibration
converted that coder deficit into a *weighting* change (`lambda 12.43`).

**Constraint A1-C4.** No A1 representation verdict may be produced by IHT-10.
A1 must (i) screen the attributed representations with **exact OMP on a frozen
dictionary** first (Stage 1), then (ii) run a **label-free IHT coder
qualification** (Stage 2) that selects *one* step count shared by every formal
arm — never a per-arm step count, never a per-arm λ.

---

## 4. TCCD-v0 — canonical raw attributed patch coordinates are refuted

Source: `notes/tccd_v0_analysis.md` (frozen verdict, commit `008b5e3`),
`results/tccd_v0/gate1.json`.

Frozen 714-D canonical fixed-coordinate attributed rooted patch
(topology + atom semantics + bond semantics + shell + mask, `M=14, A=21, B=3`),
`K=64`, `s=8`:

* permutation invariance, reuse and reconstruction were all **excellent**
  (dev `E_learned/E_random = 0.0491`, 64/64 atoms reused, `x`-space exact
  invariance);
* local-coordinate continuity **FAILED**: code-space AUC **0.4675**
  (`x`-space AUC 0.5019; graded tail code AUC 0.5494) against a pre-registered
  PASS threshold of 0.70, i.e. structurally near but non-isomorphic patches were
  *as far apart* in the frozen coordinate/code space as (patch-size, root-atom
  category)-matched random patches.  Three of the 1500-audit near-pair WL
  cosines were ~1.0: the hardest genuine near pairs are ~1-WL-equivalent.
* Atom-semantics concentration (8.1× random) was orthogonal: the dictionary
  latches onto *repeated exact* motifs only.

**Constraint A1-C5.** A1 may **not** use a canonical node-slot raw attributed
patch, may not use node-ID-dependent slots, may not use typed-WL slot ordering
as a continuous Euclidean coordinate, and may not mix `patch_cont146` or any
learned embedding into the dictionary input.  The A1 attribute channel must be a
**permutation-invariant joint statistic over already-audited topology primitives**.

**Constraint A1-C6.** A1 must re-run an adapted TCCD-style continuity audit on
its own coordinate *before* any predictor training, with code-space AUC ≥ 0.70
as a hard gate on the REAL object.  A FAIL stops the round.

---

## 5. FSAR-R2-AR0 — node role ↔ atom attribute assignment is real

Source: `notes/fsar_r2_ar0.md`, `notes/fsar_r2_ar0_node_binding_mechanism.md`,
`results/fsar_r2_ar0/*.json`.

* Object `C = Σ_v (φ_v − φ̄)(q_v − q̄)^T` on the **same** 65-D pure-topology
  `φ` that H1 codes, `q_v = onehot(atom_type_v) in R^28`.
* `MB` (assignment residual) beats `M0` (marginal baseline) **and** `MM`
  (parameter-matched marginal-capacity control on `P`) on **three paired seeds**:
  `Δ(M0−MB) = +0.0845/+0.0768/+0.0845`, `Δ(MM−MB) = +0.0591/+0.0845/+0.0625`;
  every paired per-molecule bootstrap 95 % CI excludes 0, `P(Δ>0)=1.0`.
* Assignment-sensitive: `A` and `S` are **exactly** invariant under attribute
  permutation while `C` and the prediction move materially (mean |Δpred| ≈ 0.79).
* Train-only per-coordinate RMS scaling, **no** dataset-mean subtraction; zero-RMS
  coordinates masked; sum-centred convention makes `E_π[C]=0` exact.
* The "effective rank 1.40" caveat in the original note was an audit-slicing bug
  (retracted in the mechanism note: full-`C` effective rank ≈ 56.6, model input
  `C̃` ≈ 116.3).

**Reuse decision A1-R1.** `phi65`, `q_v` (28 primitive atom categories), the
sum-centred joint statistic, the train-only RMS scaler (floor `1e-9`, eps
`1e-12`), and the relabel/shuffle audit machinery are adopted from this round
verbatim; A1 does **not** re-derive them.

---

## 6. FSAR-R2-AR0-EDGE — edge structural role ↔ bond type is real

Source: `notes/fsar_r2_ar0_edge_binding.md`, `results/fsar_r2_ar0_edge/*.json`.

* `psi_e = [φ_u+φ_v ; |φ_u−φ_v|] in R^130`, `r_e = onehot(bond_type_e) in R^4`,
  `C_E = Σ_e (ψ_e − ψ̄)(r_e − r̄)^T`; endpoint-swap invariant, chemistry-free
  topology role, undirected edges counted once.
* `BVE` beats the node baseline `BV` and the parameter-identical,
  assignment-independent marginal control `BVEM` on both completed paired seeds:
  `Δ(BV−BVE) = +0.023328/+0.014371`, `Δ(BVEM−BVE) = +0.017632/+0.016564`;
  both bootstrap CIs exclude 0 on seed 0; seed 1's lower bound is `+0.0038`
  (consistent direction, tighter on seed 0).
* Bond-type permutation leaves `S`, `A`, node `C_V` and `P_E` **exactly**
  invariant while `C_E` and the prediction move.
* Synthetic controls passed **before** any ZINC run: assignment-only positive
  control `BVE 0.0173` vs `BV 0.2531` / `BVEM 0.3228`; marginal-only negative
  control shows no repeatable `BVE` advantage.

**Reuse decision A1-R2.** The edge-side design pattern (undirected
canonicalization, endpoint-symmetric topology role, `onehot(bond_type) in R^4`,
sum-centred residual, matched `P`-style marginal control, positive/negative
synthetic controls first) is adopted for the A1 edge joint block.

---

## 7. FEC-D1 — "a localized structural coordinate helps" ≠ "the dictionary is specific"

Source: `notes/fec_d1_analysis.md`, `records/claims/claim-fec-d1-local-binding-supported-dictionary-not-specific-20260924.yaml`,
`STATE.yaml:fec_d1`.

* `G_D = M_B − M_D = +0.005245115` (gate ≥ 0.003, PASS): the localized
  binding refinement is real.
* `G_dict_specific = M_P − M_D = −0.001306051` (PASS needed ≥ +0.002, **FAIL**):
  train-fit affine **PCA32**(`phi_v`) is *better* than the sparse SDB K=32/s=8
  dictionary under an otherwise matched protocol.
* `G_assign = M_shuffle − M_D = +0.013131653` (gate ≥ 0.010, PASS): the gain is
  assignment-mediated.

**Constraint A1-C7.** "Attributed coordinate carries value" and
"sparsity/dictionary specificity carries value" must be **separated by design**,
not inferred.  A1 therefore needs (a) the primary `REAL vs INDEP` pairing
comparison, (b) the `REAL vs TOPO` secondary, and (c) a **conditional** Stage-4
DenseTied specificity control.  A1 must never report a dictionary-specificity
claim from an absolute-MAE improvement alone.

---

## 8. FSAB — an attribute/binding branch can be annihilated

Source: `notes/factorized_structure_attribute_binding.md`,
`records/claims/claim-factorized-structure-attribute-binding-collapse-20260916.yaml`,
`records/decisions/decision-factorized-structure-attribute-binding-stop-20260916.yaml`.

Verdict **Case E — the binding channel collapsed. STOP.**  The lecture for A1:
a binding/attribute route can die (near-zero task gradient + Adam/L2
contraction) while the harness still reports a plausible number.

**Constraint A1-C8.** A1 must carry explicit **branch/code/dictionary liveness**
monitoring (`||dL/dD||`, D movement from the K-SVD init, active/effective atoms,
effective rank, train↔valid usage correlation, code variance, decoder
slot-gradient statistics).  A "live but decorative" attributed code must not be
reported as a success.

---

## 9. DTX-v0 — a *graph-level* aligned cross fails

Source: `STATE.yaml:decision-zinc-dtx-v0-stop-no-aligned-cross-signal-20260923`
(prereg `7c28be1`, run `3810d73`).

Ring-blind softmax dictionary × explicit generic topology role, graph-level
aligned joint `J_align = mean_i α_i ⊗ s_i` vs matched marginal `J_indep`:
`Δ_align = −0.005330` (best) / `−0.006027` (soup) → **NO_ALIGNED_CROSS_SIGNAL**.
The aligned cross was *not* inert (permuting it moved MAE by 0.019), yet the
matched marginal control still won.

**Constraint A1-C9.** A graph-level aligned cross of marginals is refuted and
may not be re-entered.  A1's attribute channel is **occurrence/patch-level and
rooted** (per node), not a graph-level product of two marginals.  (Note
`P^V_i` in A1 is a *patch-level* marginal-null used as the control, not the DTX
graph-level statistic.)

---

## 10. Mentor upstream (read-only vendor) — what a typed/attributed K-SVD consumed

Source: `experiments/luyin16/mentor_upstream/v1_20260921/README.md` (sha256
table) plus direct reading of
`molhiv_online_structural_ksvd_full.py` and
`run_molhiv_ksvd_three_diagnostics.py`.

The mentor patch matrix is a **fixed-coordinate canonical slot** object
(`PATCH_RADIUS = 2`, capacity `M`): root slot, then shell-1 nodes, then shell-2
nodes each in *rooted typed-WL order*, with per-position channels for
"atom semantics", "shell", "mask", plus topology channels — i.e. attributes
enter dictionary formation through canonical slot positions.  `compact_atom_semantics`
and `compact_bond_semantics` one-hot-encode element group / degree / charge / H
count / valence / aromaticity / ring / chirality and bond type / conjugation /
ring, and ring/cycle context features are present (they were later studied
separately by the repo's own ring studies).

**Reading adopted by A1.**  A1 *restores the mentor's scientific idea* — the
local object that enters dictionary formation is attributed, not chemistry-free
— but **replaces the fixed-coordinate canonical slot representation** with the
permutation-invariant joint statistic of the pre-registration, precisely because
the canonical slot variant was already measured to fail the continuity gate
(TCCD-v0, §4).  A1 does **not** claim to reproduce the mentor pipeline, and does
not reuse the mentor slot layout, typed-WL slot ordering, ring features or the
`patch_cont`-style mixed descriptor.

---

## 11. Additional durable constraints pulled from the same ledger

| fact / source | constraint on A1 |
|---|---|
| SDB-v0 (`notes/sdb_v0.md`): the sparse `K=32, s=8` coordinate is a **non-lossy, causally used** replacement of the 65-D axis on a *weak* base (recovery 1.159, 3 seeds, shuffle 0.245) but adds only `+0.0024` (`< 0.003`) on the strong S0 backbone | A1 must keep the strong H1 backbone (not a weak base) and must still require `≥ 0.003` |
| SDB-v0 Stage 3: an early version selected on **train** error and inflated the effect; fixed before any durable record | A1 Stage 1/2/3 selection is on official-**valid** only; official train is never used to select an arm |
| P1: `G_specific = 0.002559 < 0.003` with a test-side reversal | A1 keeps exact parameter matching for `REAL vs INDEP` and treats sub-threshold as FAIL, not "close" |
| P2-ABS: `E64`, `K64S8`, `H2` all lost; the smallest interface won | A1 freezes `K=32, s=8`, `d_e=48`, decoder `h1`; no K/s/d_e/decoder sweep |
| P1/P2 runner: official test is refused by the loader; every artifact carries `official_test_loaded = false` | A1 reuses the same blocker and adds an explicit assertion |
| TCCD-v0: exact rooted canonicalization *is* invariant once the colouring includes root identity + shell, and tied-IHT step size must use a deterministic start vector | A1 reuses `v0.tied_iht_codes` / `tccd_v0.power_iter_sigma` (already deterministic) and makes no new coder |

---

## 12. Reuse inventory (exact functions / artifacts A1 will call)

| need | reuse target (no re-implementation) |
|---|---|
| `phi65` per node | `results/e2e_dictenv_p1/cache/env_{train,valid}.pt` field `phi` (bit-identical assert vs `fsar_r2_ar0.build_phi` on a sample) |
| node/edge pure-topology basis `b^V in R^11`, `b^E in R^15` | `fsar_v2._explicit_basis_for_patch` |
| root-relative patch incidence (which `v`, which induced edge, per root) | `e2e_dictenv_v0.env_incidence` (== `bfs_distances` radius 2) |
| atom category `q_v in R^28`, bond category `r_e in R^4` | P1 cache fields `atom` and `bond_type` (ZINC primitive categories) |
| sum-centred joint statistic + `P` marginal null | `fsar_r2_ar0.center_stats` pattern (numpy reference) |
| train-only RMS scaler / mask | `sdb_v0.fit_rms_scaler` (`SCALER_FLOOR=1e-9`, `SCALER_EPS=1e-12`, no mean subtraction) |
| K-SVD | `sdb_v0.fit_ksvd` → `tccd_v0.ksvd_fit` (10 epochs, seed 20260924) |
| exact top-`s` OMP | `sdb_v0.omp_codes` → `tccd_v0.omp_codes` |
| tied IHT | `e2e_dictenv_v0.tied_iht_codes` (+ `tccd_v0.hard_threshold_rows`, `power_iter_sigma`) |
| H1 decoder / backend / anchor / post-code binding / pair relation / reader | `e2e_dictenv_p2_abs.P2Model` (`decoder="h1"`, `d_e=48`, `K=32`, `s=8`) |
| batching / collate with occurrence offsets | `e2e_dictenv_p1.env_collate`, `e2e_dictenv_p1.make_env_loader` |
| model construction / evaluation / Top-5 soup / curve JSON | `zinc_e2e_dictenv_p2_abs.train_candidate` shape and `zinc_e2e_dictenv_p1.evaluate` |
| `phi` identity + relabel + batching + purity audits | `zinc_e2e_dictenv_p1/_g0.._g14` and `fsar_r2_ar0` audit helpers |
| continuity audit (typed-WL nearness, matched random pairs, paired AUC) | `code/run_tccd_v0.py:continuity_audit`, `_patch_graph_for_wl`, `_auc_paired`; `code/run_wholegraph_canonical_registration_audit.py:attributed_wl_fingerprint`, `_histogram_matrix` |
| official-test blocker | `zinc_e2e_dictenv_v0._g12_official_test_blocker` (via P1 `_g14`) |
| mixture/marginal matched controls, shuffle interventions | `e2e_dictenv_p1.shuffled_occ_node_for_molecule`, `shuffled_bond_endpoints_for_molecule` |

**Constraint A1-C10.** A second, approximate implementation of any row above is
forbidden; if a shared helper must be extended, the extension must be
backward-compatible and the existing tests of the owning round must still pass.

---

## 13. The forbidden list A1 inherits (any of these = a new round)

```
canonical node-slot raw attributed patch (TCCD-v0 714-D)
node-ID-dependent slot ordering / typed-WL slot ordering as a continuous coordinate
patch_cont146 or atom_shell/bond_shell mixed descriptor in the dictionary input
learned embedding / message-passing hidden state as dictionary input
ring / cycle / fused-ring handcrafted features (explicit or implicit)
target y anywhere in the dictionary fit, scaler, or selection
graph-level aligned cross of two marginals (DTX-v0)
K / s / radius / d_e / decoder / activation / reader / dropout / LR / wd / λ sweeps
per-arm IHT step counts, per-arm λ recalibration
LISTA / new sparse solver / new optimizer
seed hunting, validation-driven block reweighting
official ZINC test access in this round
```

---

## 14. Result of the audit

The A1 design is fully determined by the ledger: an invariant, permutation-safe,
rooted **joint** structure×attribute statistic (`11x28` node, `15x4` edge) added
to the already-audited `phi65`, dictionary `K=32/s=8` on official train only,
exact-OMP screening before any IHT-based training, one shared IHT step count
chosen label-free, H1 frozen downstream, and an exact parameter-matched
assignment-independent control (`P` marginal null) as the **primary** comparison.
The pre-registration freezes all of this, plus the gates and verdict table, at
the commit following this file.
