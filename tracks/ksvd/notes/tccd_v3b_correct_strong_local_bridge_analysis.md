# TCCD-v3b — Correct Pre-Pair Local-State Bridge Analysis

Preregistration: `notes/tccd_v3b_correct_strong_local_bridge_preregistration.md`.
Formal implementation revision: `f942a00febbd54b9a3b0f1dfdb301eb48d0293a2`.
Preregistration commit: `f46cd63`.

## Scope correction

Historical TCCD-v3 used the isolated 16-D `e_struct` channel. The B-full
forward audit showed that this is only one input block to the fused local
`patch_encoder`. TCCD-v3b therefore selects the **output** of the complete
B-full `patch_encoder`, before pair projection, pair encoding, centre-context,
global, or topology operations.

Selected tensor:

```text
PatchPathRecurrentPairCentreModel._encode_core -> patch_encoder output
```

Confirmed widths:

- fused pure-local patch-encoder input: `170D`
- selected pre-pair patch state: `64D`
- next operation: `pair_projection` (`64 -> 16`)
- matched trainable adapter in both arms: `Linear(64,64)`

The historical TCCD-v3 files and numbers are preserved unchanged. This round
is the corrective audit and supersedes any broader interpretation of TCCD-v3.

## Execution integrity

- Device: remote A100 **GPU1** only.
- Seed: `0`.
- Official train / valid: `10,000 / 1,000`.
- Official test: never loaded (`official_test_loaded: false` in every formal artifact).
- Frozen source checkpoint:
  `tracks/ksvd/results/shared_structural_patch_encoder/soup_states/sspe_seed0_top5_soup.pt`.
- Source checkpoint SHA-256:
  `785dff866fa8d952d6e9764549761a7019a4e46ffd70d625110b618f51e6a2c4`.
- Source training commit: `0aa71c81e845fd334c8ecd2bf7904c149277c289`.
- Formal implementation commit: `f942a00`.
- The source B-full model is frozen; only the registered bridge parameters train.
- The source checkpoint was selected using official-valid performance in its
  source study, so this remains a representation-capacity diagnostic bridge,
  not an unbiased final-model comparison.

## Gate 0

The final GPU1 Gate 0 passed all registered checks:

- selected tensor is the `patch_encoder` output, not an input block;
- output width is `64D`; input width is `170D`;
- `pair_projection` consumes the selected `64D` state;
- pure-local exterior invariance: max delta `0.0`;
- node-relabel invariance: max delta `0.0`;
- batching invariance: max delta `1.6019e-7`;
- deterministic checkpoint reload: max delta `0.0`;
- pair/global/topology intervention independence: max delta `0.0`;
- frozen B-full parameters receive no gradient;
- DenseREL and PrototypeREL adapter/reader/prototype/temperature gradients pass;
- PrototypeREL has no raw continuous-state bypass;
- official test remains blocked;
- cache freshness: train/valid max absolute delta `2.3842e-7`.

Cache sizes:

- train: `231,664` patch states across `10,000` graphs;
- valid: `23,083` patch states across `1,000` graphs;
- cache raw width: `64D`.

## Formal results

| arm / control | best valid MAE | Top-5 soup MAE |
|---|---:|---:|
| CorrectStrongLocal-DenseREL | `0.327528` | `0.312805` |
| CorrectStrongLocal-PrototypeREL | `0.315263` | `0.312643` |
| PrototypeREL shuffle, evaluation-only | `0.592378` | — |

The preregistered primary bridge metric is the best-checkpoint
CorrectStrongLocal-PrototypeREL value. Relative to the frozen TCCD-v2
PrototypeREL reference `0.287337`:

```text
Delta_local = 0.287337 - 0.315263 = -0.027926
```

This is **L3** because `MAE > 0.24` and `Delta_local < 0.04`.

The prototype gap is:

```text
Delta_proto,strong = 0.315263 - 0.327528 = -0.012265
```

The negative gap is within the registered `<= 0.015` preservation band. The
prototype vocabulary therefore preserves, and slightly improves on, the dense
composition arm; it is not the reason the bridge misses the absolute target.

The evaluation-only composition shuffle gap is:

```text
Delta_comp,strong = 0.592378 - 0.315263 = 0.277115
```

This is far above the registered `0.01` sanity threshold. Assignment-sensitive
`C^T R C` composition remains materially useful.

## Vocabulary diagnostics

The PrototypeREL vocabulary is healthy rather than collapsed:

- active prototypes: `64 / 64`;
- dead prototypes: `0`;
- effective prototype count: `63.8328`;
- normalized global usage entropy: `0.999690`;
- top-8 usage mass: `0.135370`;
- learned temperature: `0.133860`;
- mean top-50 semantic concentration: `0.402813`;
- random baseline concentration: `0.057188`.

Therefore the L3 outcome is not explained by locality failure, frozen-encoder
failure, cache mismatch, prototype collapse, or composition insensitivity.

## Interpretation

The complete fused B-full pre-pair local state is a materially better bridge
than the isolated `e_struct` channel: the corrected PrototypeREL reaches
`0.315263` best valid MAE rather than the historical isolated-channel
`0.982502`. However, replacing only the local representation still does not
recover the TCCD-v2 target or the native B-full reference `0.119818`.

The corrected result rejects the broad claim that a strong local representation
alone is sufficient to rescue TCCD through the frozen prototype/composition/
reader interface. It does **not** prove that local representations are weak or
that the B-full encoder is unnecessary. It localizes the remaining problem to
the interface/inductive-bias mismatch between the fused local state and the
frozen TCCD-v2 graph-level composition/readout path.

## Decision

Stop after the preregistered DenseREL, PrototypeREL, shuffle evaluation, and
vocabulary diagnostics. Do not run rescue seeds, K/temperature sweeps, reader
capacity changes, relation changes, pair/topology additions, higher-order
features, or source-encoder unfreezing. Any interface redesign requires a new
preregistration. No official test evaluation is authorized.

Evidence:

- `tracks/ksvd/notes/tccd_v3b_correct_strong_local_bridge_preregistration.md`
- `tracks/ksvd/results/tccd_v3b/gate0.json`
- `tracks/ksvd/results/tccd_v3b/cache.json`
- `tracks/ksvd/results/tccd_v3b/dense_seed0.json`
- `tracks/ksvd/results/tccd_v3b/prototype_seed0.json`
- `tracks/ksvd/results/tccd_v3b/diagnostics.json`
- `tracks/ksvd/tests/test_tccd_v3b.py`
