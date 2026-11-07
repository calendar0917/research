# REPORT — `lcrc-pcba-atom-bond-ssl-pilot-v1`

Masked atom/bond self-supervised initialisation of the original no-message-passing LCRC-A (`A_SSL_INIT_S0`), seed 0, one pretrain and one 40-epoch supervised fine-tune. Generated from `analysis/` by `code/ssl_report.py`.

| field | value |
| --- | --- |
| status | COMPLETED |
| decision label | **NO_CLEAR_GAIN** |
| Validation AP40 | 0.2457260613 |
| historical A0 AP40 | 0.2463560961 |
| Delta AP40 | **-0.0006300348** |
| pretext gate | PRETEXT_LEARNED |
| git commit | `d7b4be8e9eb1120ac6bccbf9b5007e700209c6f0` |
| Test evaluated | no |

## 1. 结论摘要 (required summary)

1. **Masked atom/bond pretext 是否学超过类别先验？** 是。Probe atom CE 0.3290 vs 先验 1.5086（margin +1.1795 nats）；bond CE 0.0894 vs 先验 0.9224（margin +0.8330 nats）。
2. **预训练是否完成全部 10000 updates？** 是，实际 10,000/10,000 updates，status `COMPLETED`。
3. **Encoder 权重是否确实迁移？** 是：8/8 个 encoder tensor 被覆盖，其余 4 组模块（`local.`/`f.`/`rho.`/`readout.`）与历史 seed-0 scratch 初始化逐 tensor 相等（`True`）。两个 encoder 在预训练中都发生了变化（`True`）。
4. **SSL 模型 epoch40 Validation AP？** `0.2457260613`。
5. **相对历史 A0 = 0.2463560956？** 下降 `-0.0006300348`（NO_CLEAR_GAIN）。
6. **Train−Valid gap 如何？** Train AP40 0.4306467894，Valid AP40 0.2457260613，gap +0.1849207281；Train BCE 0.0303283001，Valid BCE 0.0456266720。

## 2. Protocol compliance

| constraint | observed |
| --- | --- |
| seed | 0 |
| pretrain runs | 1 |
| supervised runs | 1 (fixed epoch 40) |
| inference parameters | 287,644 |
| base + new parameters | 245,184 + 42,460 |
| historical A0 retrained | no |
| other arms (A1/A2/A3/G/V) | none |
| validation checkpoints | 10, 20, 30, 40 (diagnostics only) |
| official Test labels | never read, never scored |
| pretrain updates | 10,000 (export at update 10000) |
| decision bands frozen | yes (`ssl_common.DECISION_BANDS`) |

## 3. Pretext learning (probe, visit 0)

Probe = 4,096 hash-selected **official Train** molecules, never used for a parameter update. Prior CE is computed from SSL-fit class frequencies on the probe's own masked targets.

| update | atom CE | atom prior CE | atom acc | bond CE | bond prior CE | bond acc |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 2.7906 | 1.5086 | 0.00% | 1.6009 | 0.9224 | 47.67% |
| 2000 | 0.3721 | 1.5086 | 86.68% | 0.1110 | 0.9224 | 96.35% |
| 4000 | 0.3478 | 1.5086 | 87.45% | 0.0938 | 0.9224 | 96.79% |
| 6000 | 0.3376 | 1.5086 | 87.69% | 0.0930 | 0.9224 | 96.93% |
| 8000 | 0.3325 | 1.5086 | 87.61% | 0.0912 | 0.9224 | 97.04% |
| 10000 | 0.3290 | 1.5086 | 88.16% | 0.0894 | 0.9224 | 97.09% |

| gate item | value | required | passed |
| --- | --- | --- | --- |
| atom margin (nats) | +1.1795 | >= 0.05 | True |
| bond margin (nats) | +0.8330 | >= 0.05 | True |
| encoders changed | +22.999306 | L2 > 0 and all tensors changed | True |
| gate label | **PRETEXT_LEARNED** |  |  |

Masked targets scored on the probe at the final update: 15,808 atoms, 17,094 bonds. Atom class counts `0;29;4445;7052;130;161;2;1061;709;1801;84;4;244;85;1;0`, bond class counts `7570;1311;16;8197;0`.

## 4. Encoder transfer audit

| item | value |
| --- | --- |
| transferred prefixes | `atom_encoder.*`, `bond_encoder.*` |
| frozen prefixes | `local.*`, `f.*`, `rho.*`, `readout.*` |
| tensors transferred | 8 |
| encoder tensors changed vs scratch | 8/8 |
| non-transferred tensors == historical seed-0 | True |
| pretext heads transferred | [] |
| SSL optimizer state transferred | False |
| state-dict keys unchanged | True |
| inference parameters | 287,644 |
| forward probe | [8, 128] |
| SSL updates used | 10,000 |
| SSL encoder fingerprint | `5b664f5f3f3e4217…` |

Transferred tensor SHA256:

| tensor | sha256 |
| --- | --- |
| `atom_encoder.lins.0.bias` | `e6587ff3d326ed59…` |
| `atom_encoder.lins.0.weight` | `92c0fa8ef33a268e…` |
| `atom_encoder.lins.1.bias` | `dd30cb39612f043a…` |
| `atom_encoder.lins.1.weight` | `a1a19941a1a7b277…` |
| `bond_encoder.lins.0.bias` | `b89b14aa505f0be6…` |
| `bond_encoder.lins.0.weight` | `179ec6a54ed03ede…` |
| `bond_encoder.lins.1.bias` | `ba643b987a3f0ad4…` |
| `bond_encoder.lins.1.weight` | `d393d0f6836d3045…` |

Encoder L2 change vs scratch init: `22.999306` (relative `1.813125`); all tensors changed: `True`.

## 5. Supervised fine-tune (official Validation)

| epoch | Validation AP | Valid BCE | tasks scored |
| --- | --- | --- | --- |
| 10 | 0.2208235608 | 0.0468888976 | 127 |
| 20 | 0.2399283334 | 0.0458543941 | 127 |
| 30 | 0.2453049040 | 0.0456163854 | 127 |
| 40 | 0.2457260613 | 0.0456266720 | 127 |

| epoch-40 quantity | value |
| --- | --- |
| Validation macro AP | 0.2457260613 |
| historical A0 AP | 0.2463560961 |
| Delta AP40 | -0.0006300348 |
| decision label | **NO_CLEAR_GAIN** |
| Train AP40 | 0.4306467894 |
| Train BCE40 | 0.0303283001 |
| Valid BCE40 | 0.0456266720 |
| Train − Valid AP gap | +0.1849207281 |
| epochs completed | 40 |
| global steps | 109,520 |
| gradient-clip events | 21 |
| non-finite events | 0 |

Batch-order parity with the historical seed-0 plan:

| plan field | value |
| --- | --- |
| global plan sha256 | `06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6` |
| prefix sha256 | `1a24dfca54898dd2b0e7c58d2b2f1956df67b233fdceaa7a376f81a2377a5f58` |
| epochs consumed | 40 |
| batch size | 128 |

## 6. Per-task breadth

| metric | value |
| --- | --- |
| tasks compared | 127 |
| improved | 66 |
| regressed | 61 |
| unchanged | 0 |
| mean delta | -0.0006300348 |
| median delta | +0.0007018383 |
| min / max delta | -0.2225858008 / +0.1878429247 |

Support strata (Validation positives per task):

| bucket | tasks | mean AP | historical mean AP | mean delta | improved | regressed |
| --- | --- | --- | --- | --- | --- | --- |
| support_ge20 | 107 | 0.2705106837 | 0.2687158061 | +0.0017948777 | 58 | 49 |
| support_5_19 | 18 | 0.1235678817 | 0.1338348377 | -0.0102669560 | 7 | 11 |
| support_lt5 | 2 | 0.0191723764 | 0.0628029387 | -0.0436305623 | 1 | 1 |

Largest regressions:

| task | AP | historical AP | delta | positives |
| --- | --- | --- | --- | --- |
| 31 | 0.0043708536 | 0.2269566544 | -0.2225858008 | 6 |
| 102 | 0.0916441463 | 0.2154453880 | -0.1238012416 | 37 |
| 108 | 0.0370370370 | 0.1250000000 | -0.0879629630 | 1 |
| 9 | 0.1041125315 | 0.1765226088 | -0.0724100773 | 14 |
| 77 | 0.0400462258 | 0.1044174928 | -0.0643712671 | 7 |

Largest improvements:

| task | AP | historical AP | delta | positives |
| --- | --- | --- | --- | --- |
| 34 | 0.1977951381 | 0.0099522134 | +0.1878429247 | 13 |
| 19 | 0.3868710372 | 0.2915001258 | +0.0953709113 | 79 |
| 13 | 0.2742120555 | 0.1885599930 | +0.0856520624 | 20 |
| 41 | 0.4030235125 | 0.3356341822 | +0.0673893303 | 249 |
| 17 | 0.2110372872 | 0.1540922761 | +0.0569450111 | 48 |

Breadth note: 66/127 tasks improved and 61 regressed, with a median delta of +0.0007018383 but task-level moves from -0.2225858008 to +0.1878429247: the aggregate delta is a wash of offsetting per-task changes, not a uniform shift. Tasks with <5 Validation positives contribute 0.05% of the summed positive per-task deltas; a gain concentrated there is not a stable generalisation claim.

## 7. Compute and provenance

| item | value |
| --- | --- |
| parameters | 287,644 |
| forward FLOPs (128 molecules) | 1,441,206,336 (torch.utils.flop_counter.FlopCounterMode (forward only, FP32)) |
| pretrain GPU hours | 0.1034 |
| supervised GPU hours | 1.3980 |
| total GPU hours | 1.5151 |
| budget target / hard | 5.0 / 7.0 |
| within target | True |
| pretrain peak allocated | 82,868,736 B |
| supervised peak allocated | 128,225,792 B |
| host / device | a100-2 / NVIDIA A100-SXM4-40GB |
| torch / CUDA | 2.5.1+cu124 / cuda 12.4 |
| git commit | `d7b4be8e9eb1120ac6bccbf9b5007e700209c6f0` |
| cache fingerprint | `7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee` |
| cache sha256 | `caad3e90dd76d1f1483d2a01501eba23413072a7b5a2fc4b62780c315aa59035` |
| feature spec hash | `7270f7677b349a46bb193174381abcb84f14d1b4f83bd1362d4aa20389a515ee` |
| split partition sha256 | `c872bd6311796826db9365c5c739c4cee94c3baa78fbd18006616feefc67bf4f` |
| seed-0 norm fingerprint | `6ce50f989fc0789d9573f7f787a44a49a282531ac5a2db09c6600c67dc3293fb` |
| pretrain checkpoint | `tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/runs/pretrain/checkpoint_final.pt` |
| pretrain checkpoint sha256 | `136ba95307360a67d04017e854492b88952200c36eaa5997fbc6f5d6a0b1c9de` |
| epoch-40 checkpoint | `tracks/structural-sufficiency/lcrc_pcba_atom_bond_ssl_pilot_v1/runs/supervised/checkpoint_epoch40.pt` |
| epoch-40 checkpoint sha256 | `b79678946be97c710b2d58775042b56042e6b0ce0ece5667b9e338e099beafb0` |
| per-task CSV sha256 | `2801ca8fabb514fca670b30e463f5a71145234cc1c99584f6e995e3d79eed03f` |

## 8. Label and Test access audit

| audit item | value |
| --- | --- |
| label artifact opened by pretraining | False |
| labels read | False |
| Test rows used for pretraining | 0 |
| Test labels read | False |
| Valid rows used for pretraining | 0 (probe and fit are official Train only) |
| mask/leakage audit all_passed | True |
| audit checks | loader_label_isolation, row_partition, targets_not_in_input, atom_view_visibility_invariance, bond_view_visibility_invariance, no_forbidden_fields, atom_renumbering, bond_endpoint_swap, no_message_passing, gradients_reach_all_groups, original_a_geometry, reproducibility, gpu_smoke |
| run registry Test evaluated | False |
| supervised labels | official Train/Valid targets only; Test rows are NaN in the frozen artifact and asserted NaN at load, never indexed |
| supervised test_policy | — |
| run record Test evaluated | False |

The epoch-40 checkpoint contains no Test logits: validation and Train reports are the only scored splits in `runs/supervised/`.

## 9. Decision and interpretation

**NO_CLEAR_GAIN** — Delta AP40 = `-0.0006300348` against the frozen historical A0 `0.2463560961`.

Pre-registered bands:

| band | rule |
| --- | --- |
| PROMISING_SINGLE_SEED | Delta >= +0.005 |
| WEAK_SIGNAL | +0.002 <= Delta < +0.005 |
| NO_CLEAR_GAIN | -0.002 < Delta < +0.002 |
| NEGATIVE_TRANSFER | Delta <= -0.002 |

Mandatory caveats:

- no same-round A0 run: implementation/environment drift cannot be excluded
- SSL pretraining adds compute, so any gain cannot be attributed to self-supervision alone
- a well-learned pretext with no validation gain is a valid negative transfer result
- the historical Test-facing A0 number is not a target of this round
- the exploratory bands are resource-allocation labels, not significance tests

## 10. Artifacts

| artifact | path |
| --- | --- |
| data audit | `analysis/data_audit.json` |
| mask/leakage audit | `analysis/mask_leakage_audit.json` |
| pretrain probe | `analysis/pretrain_probe.json` |
| pretrain run | `analysis/pretrain_run.json` |
| encoder transfer | `analysis/encoder_transfer_audit.json` |
| epoch-40 report | `analysis/supervised_epoch40.json` |
| per-task AP | `analysis/per_task_ap.csv` |
| run registry | `analysis/run_registry.json` |
| GPU ledger | `analysis/gpu_ledger.json` |
| decision | `analysis/decision.json` |
| pretrain checkpoint (local) | `runs/pretrain/checkpoint_final.pt` |
| pretrain curve (local) | `runs/pretrain/curve.csv` |
| supervised checkpoint (local) | `runs/supervised/checkpoint_epoch40.pt` |
| supervised curve (local) | `runs/supervised/train_curve.csv` |
| per-task AP (local) | `runs/supervised/per_task_ap.csv` |

