# Evidence carried into SOC-v1

All paths are under `tracks/structural-sufficiency/` unless stated. No official
test label was read for any item below.

## Prior-round results

| round | verdict | number that matters |
|---|---|---|
| `zinc_dual_channel_composer_v1` | champion | 494,193 params; official valid 0.075990 (4 seeds); test 0.069400 |
| `zinc_counterfactual_ownership_composer_v1` | STOP_NO_MATERIAL_GAIN | seed-mean Δ +0.007247 vs L on Monitor |
| `zinc_bond_anchored_marginal_field_v1` | STOP_NO_SIGNAL | Δ +0.003149 vs control I, +0.004509 vs L |
| `zinc_directed_bond_transformer_v1` | wash | test 0.069401 |
| `zinc_global_operator_search_v1` | closed route | broadcast bug; constant mass ≤ dynamic mass |
| `zinc_structural_token_attention_v1` | no gain | object-token attention |
| `zinc_structural_factor_contraction_v0` | no sample-efficiency evidence | COMPUTABLE_BUT_NO_SAMPLE_EFFICIENCY_EVIDENCE |
| `zinc_native_relation_mp_v1` | unstable/mixed | extra MP branch costs 0.004–0.006 |
| `zinc_dccl_frozen_transfer_audit_v1` | TRUNK_TRANSFER_SIGNAL | 77% of 4k→8k gain re-extractable from 513-d z, Δ +0.0297 |
| `zinc_dccl_bottleneck_v1` | closed | unseen radius-2 = 81.6% of error; seed/checkpoint lottery excluded |

## Error structure (DCB-v1)

* Support-tier MAE: 0.0365 (383 mols) / 0.0724 (366) / 0.1414 (251).
* Dose-response monotone; error is dispersion (bias²/MSE ≈ 0.001).
* Perfect repair of all 35 label-penalty rows: ΔMAE 0.01308, but only 0.00944 is
  cycle-attributable and 0.00509 is one row (valid:0172).
* After removing penalty rows, any-unseen radius-2 molecules (241, 25% of rows)
  carry 50.2% of no-penalty AE.

## CPU diagnostics run for this round (train labels only, no GPU)

1. **Composition-key Ridge probe** (`/tmp/diag_keys.py`, caches in
   `zinc_composition_transfer_v1/cache/keys.npz`). Molecule-level counts of
   radius-1 keys (489 types), radius-2 content keys (12,583 types), + global33.
   5-fold CV within train 10k:

   | features | MAE |
   |---|---:|
   | global33 | 1.040 |
   | k1 + global | 0.386 |
   | k2 + global | 0.393 |
   | k1 + k2 + global | 0.330 |

   Unseen molecules: 0.394 vs seen 0.303 (k1+k2+global). Linear key counts alone
   are far weaker than the neural trunk.

2. **Frozen z + key-count probes.** Export the 513-d frozen DCC-L reader input
   for all 10k train rows (`/tmp/z_fit4k_s0.npz`, `/tmp/z_fit8k_s0.npz`), then
   5-fold CV with a 128-hidden MLP head on the 6,000 rows outside Fit4k (2,000
   outside Fit8k):

   | features | Fit4k pool | Fit8k pool |
   |---|---:|---:|
   | z | 0.1390 | 0.1506 |
   | z + k1 | 0.1309 | **0.1399** |
   | z + k2 | 0.1265 | 0.1449 |
   | z + k1 + k2 | 0.1269 | 0.1475 |
   | z + k1 permuted | 0.1423 | — |
   | z + k2 permuted | worse | 0.1603 |

   Real k1 correspondence contributes only ~0.0024 beyond the capacity of the
   extra inputs; k2 overfits. Explicit coarse counts are **not** the main lever.

3. **Iterative MP on frozen Fit8k encoders** (`/tmp/probe_iter.py`). Manually
   unrolled alpha-blended junction→bond→atom updates, recomputed the 513-d z,
   trained a 64-hidden head on 4,000 rows, monitored on 1k:

   | setting | Monitor MAE |
   |---|---:|
   | alpha 0 (no iteration) | **0.13384** |
   | alpha 0.1, 1 step | 0.13400 |
   | alpha 0.2, 1 step | 0.13512 |
   | alpha 0.5, 1 step | 0.16317 |
   | alpha 1.0, 3 steps | 0.43316 |

   Naive iteration on a frozen trunk degrades monotonically. Any iteration must
   be trained end-to-end; SOC-v1 does that and starts at DCC-L.

## Consequence for the design

The only surviving, honest lever is a **small end-to-end iterative operator**
that (a) starts at the champion, (b) uses explicit incidence statistics rather
than attention/relations, and (c) can be falsified at the head/monitor level
before the full 10k confirmation. SOC-v1 is that operator.

## Workspace at round start

* repo `calendar0917/research`, branch `main`, worktree clean at `c864d4d`.
* `uv run research context` → phase `zinc-upstream-portfolio-v1-stop-no-purchased-candidate`.
* `rr doctor res` → 10 ok / 0 fail, host `a100-2`, 2× A100-SXM4-40GB.
* GPU spend this round at protocol freeze: **0 A100·h**.

## External boundary (verified in `notes/prior_work_boundary.md`)

The round did not re-audit external source code; it relies on the track's frozen
catalog and flags the distinction explicitly:

* **ESA/ESA+** (Buterez et al., *Nature Communications* 16, 2025, DOI
  `10.1038/s41467-025-60252-z`): edge-set masked self-attention + global
  self-attention; ZINC-12k 0.052±0.001 with RWSE + ring features. Table 5 does
  not contain an all-S (no-mask) matched control. This is global attention, not
  a local moment closure.
* **GRIT** (ICML 2023, `LiamMa/GRIT`): RRWP positional encoding + node-token
  global self-attention; ZINC ~0.059 at ~473k params. The official config was
  verified byte-identical in
  `zinc_grit_reference_preparation_v1/COST_REPORT.md`.
* **CSA / CSA-rings** (`inria-thoth/csa`), **GPM/NGPM**
  (`Zehong-Wang/GPM`), **MoSE**, **SPSE**: catalogued with code URLs and
  paper-level protocol notes in `notes/prior_work_boundary.md`. Their structural
  information is static (rings, edge colours, homomorphism/path counts, pattern
  tokens) and their interaction is global attention; none is a repeated local
  moment closure.
* **Boundary claim used here**: SOC-v1 is *not* a variant of these. It uses no
  attention, no positional-encoding table, no pattern/path tokens; it applies
  one shared first/second-moment operator to the existing bond/junction
  incidence graph. The comparison is therefore a new operator family, not a
  re-run of a known static-relation model. The 0.060 target is at the level of
  the known no-MP attention/positional-encoding frontier, which this round did
  not attempt to replicate.
