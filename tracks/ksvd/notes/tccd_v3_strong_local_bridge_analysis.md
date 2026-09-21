# TCCD-v3 — Strong-Local Bridge Analysis

Pre-registration: `notes/tccd_v3_strong_local_bridge_preregistration.md`.
Formal implementation revision: `18753a1622dba61fba770372da8e192df3095ea1`.
Preregistration commit: `dead011`. Implementation began at `dee4f36`; the final
cache/provenance fixes are included in `18753a1`.

## Execution integrity

- Formal device: remote A100 **GPU1** only.
- Seed: `0`.
- Official train / valid: `10,000 / 1,000`.
- Official test: **never loaded** (`official_test_loaded: false` in every formal artifact).
- Frozen source checkpoint:
  `results/shared_structural_patch_encoder/soup_states/sspe_seed0_top5_soup.pt`.
- Source checkpoint SHA-256:
  `785dff866fa8d952d6e9764549761a7019a4e46ffd70d625110b618f51e6a2c4`.
- Source training commit: `0aa71c81e845fd334c8ecd2bf7904c149277c289`.
- Strong encoder remains frozen; only the matched `Linear(16,64)` adapter and
  the registered downstream bridge parameters are trainable.
- Embedding cache schema: `tccd_v3_strong_local_embeddings_v1`.
- Cache size: `231,664` train patches and `23,083` valid patches.
- The cached embeddings were verified against the native frozen encoder output
  on a fresh batch (max absolute numerical difference `1.79e-7`, attributable
  to float32 serialization).

## Gate 0

The final GPU1 Gate 0 rerun at commit `18753a1` passed all checks:

- graph-level pure-local exterior test: max delta `0.0`;
- synthetic exterior test: max delta `0.0`;
- patch relabel invariance: max delta `2.98e-8`;
- batching invariance: max delta `5.96e-8`;
- deterministic reload: max delta `0.0`;
- frozen encoder: passed;
- output width: `16`;
- official test loaded: `false`.

Local targeted tests also pass: `20 passed` across the TCCD-v3, TCCD-v2,
and shared-structural encoder suites; Ruff and byte-compilation pass.

## Formal results

| arm / control | best valid MAE | Top-5 soup MAE |
|---|---:|---:|
| StrongLocal-DenseREL | `1.107023` | `1.102820` |
| StrongLocal-PrototypeREL | `0.982502` | `0.971857` |
| PrototypeREL shuffle, evaluation-only | `1.242556` | — |

The preregistered primary bridge metric is the best-checkpoint PrototypeREL
value. Relative to TCCD-v2 PrototypeREL best valid `0.287337`:

```text
Delta_local = 0.287337 - 0.982502 = -0.695165
```

This fails L3 decisively because `MAE_StrongProto > 0.24` and
`Delta_local < 0.04`.

The prototype-vocabulary gap under the stronger frozen local representation is
negative:

```text
Delta_proto,strong = 0.982502 - 1.107023 = -0.124521
```

The Top-5 soup gap is also negative (`0.971857 - 1.102820 = -0.130963`).
Thus the prototype vocabulary still beats the matched dense composition arm.

The evaluation-only shuffle control remains materially worse:

```text
Delta_comp,strong = 1.242556 - 0.982502 = 0.260054
```

This preserves the preregistered conclusion that assignment-sensitive
`C^T R C` composition is active and materially useful.

## Vocabulary diagnostics

The trained PrototypeREL vocabulary is healthy rather than collapsed:

- active prototypes: `64 / 64`;
- dead prototypes: `0`;
- effective prototype count: `62.6835`;
- normalized global usage entropy: `0.997451`;
- top-8 usage mass: `0.158809`;
- learned temperature: `0.050159`;
- mean top-50 semantic concentration: `0.31625` versus random baseline
  `0.05719`.

Therefore the L3 result is not explained by dead prototypes, a frozen encoder
failure, a locality violation, or a composition-insensitivity artifact.

## Interpretation

The strong-local representation is individually strong in its native B-full
model (`0.119818` official-valid MAE), but that strength does not transfer
through the frozen TCCD-v2 prototype/composition interface plus lightweight
reader. The bridge therefore rejects the narrow hypothesis that TCCD-v2's
absolute gap is mainly caused by the simple `714 -> 64` local encoder.

The result does **not** prove that local representation quality is irrelevant;
it shows only that replacing the local encoder alone, while freezing the
TCCD-v2 vocabulary/composition/reader protocol, is insufficient to recover the
strong standalone performance. The remaining bottleneck is consequently a
broader interface or inductive-bias mismatch between the strong local features
and the fixed TCCD-v2 composition/readout path.

## Decision

Stop after the registered DenseREL, PrototypeREL, shuffle evaluation, and
vocabulary diagnostics. Do not unfreeze or retrain the strong encoder. Do not
add reader capacity, new relations, higher-order moments, topology, pair
machinery, K/temperature sweeps, or rescue seeds. Any follow-up architecture
requires a new preregistration.

Evidence pointers:

- `results/tccd_v3/gate0.json`
- `results/tccd_v3/cache.json`
- `results/tccd_v3/dense_seed0.json`
- `results/tccd_v3/prototype_seed0.json`
- `results/tccd_v3/diagnostics.json`
- `tests/test_tccd_v3.py`
- `notes/tccd_v3_strong_local_bridge_preregistration.md`
