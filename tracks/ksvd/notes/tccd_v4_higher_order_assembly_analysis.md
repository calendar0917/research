# TCCD-v4 — Higher-Order Assembly Sufficiency Audit

Preregistration: `notes/tccd_v4_higher_order_assembly_preregistration.md`.
Implementation commit: `2e0218a121d26737a5fff9e45bfaf4cbad4a4b25`.
Formal result: `results/tccd_v4/stageA_seed0.json`.

## Verdict

**STOP at Stage A.** The frozen TCCD-v2 representation does not show a
practically useful incremental signal from the registered normalized two-hop
assembly moment `M2 = C^T S^2 C`.

The failure is decisive under the preregistered Stage-A rule:

```text
delta_topo = MAE_MIS2 - MAE_REAL2 = 0.0000068843
 delta_add = MAE_BASE  - MAE_REAL2 = -0.0000134408
```

Both are below the `0.005` failure threshold. No paired reader seed is
allowed. Stage B, Stage C, and the official-valid full run were not run.

## Execution integrity

- GPU: remote A100 **GPU1** only.
- Seed: `0`.
- Split: fixed internal `8000/2000`, split seed `20260922`.
- Base: exact TCCD-v2 PrototypeREL local encoder / K=64 vocabulary / existing
  relation set / reader / temperature / regularizers.
- Reused TCCD-v2 best checkpoint SHA-256:
  `093e0e46d4d5dab7e5558b807d993d1136df58d8f531d6b9b3621e8ab05c0f42`.
- Formal commit: `2e0218a`.
- Runtime: `163.07 s`, peak GPU memory `498.1 MB`.
- Official test: never loaded (`official_test_loaded: false`).
- K-SVD, OMP/IHT, local encoder changes, prototype changes, temperature
  changes, relation changes, reader changes, and architecture sweeps: none.

## Gate 0

Gate 0 passed locally and on GPU1. It verified:

- `S = D^-1/2 A D^-1/2` from native untyped adjacency with no self-loop;
- safe zero inverse degree for isolated nodes;
- exact `R2 = S @ S` algebra;
- permutation invariance of `C^T S^2 C`;
- single/batched computation equality;
- two-hop sensitivity on a synthetic structure pair;
- material REAL2/MIS2 intervention difference;
- no target access in permutation or relation construction;
- official-test block.

The GPU1 Gate-0 maximum permutation delta was `9.54e-7`; the synthetic
REAL2/MIS2 maximum feature delta was `5.0764`.

## Stage A — frozen representation screen

A single exact TCCD-v2 PrototypeREL checkpoint was frozen. BASE, REAL-2 and
MIS-2 were evaluated with the same lightweight linear reader family.
REAL-2 and MIS-2 had exactly matched feature width `12,544`; BASE had width
`10,464`. The primary matched comparison is REAL-2 versus MIS-2.

| arm | best internal-dev MAE | Top-5 soup MAE | feature width |
|---|---:|---:|---:|
| BASE | `0.287915707` | `0.279730588` | `10,464` |
| REAL-2 | `0.287929147` | `0.278037608` | `12,544` |
| MIS-2 | `0.287936032` | `0.279558271` | `12,544` |

Best-checkpoint deltas used for the preregistered decision:

```text
delta_topo = 0.2879360318 - 0.2879291475 = 0.0000068843
delta_add  = 0.2879157066 - 0.2879291475 = -0.0000134408
```

Diagnostic soup deltas point in the same direction but remain tiny:

```text
soup delta_topo = 0.2795582712 - 0.2780376077 = 0.0015206636
soup delta_add  = 0.2797305882 - 0.2780376077 = 0.0016929805
```

The topology-mismatch control therefore does not reveal a meaningful
assignment-to-two-hop-topology alignment signal in the frozen TCCD-v2 codes,
and appending REAL-2 does not improve over BASE in a practically relevant way.

## Decision

This round does **not** support the claim that adding one normalized two-hop
walk moment materially overcomes the TCCD-v2 ceiling. The correct conclusion
is narrower than “topology is unimportant”:

> A parameter-free normalized two-hop aggregated moment does not materially
> add predictive information beyond the current TCCD-v2 relation set under the
> frozen representation and matched reader control.

Because Stage A failed, the round forbids:

- end-to-end +2hop training;
- any paired Stage-A seed;
- three-hop `S^3`;
- official-valid full training;
- official-test access.

The leading next hypothesis is now stronger: the main loss may occur before or
at the occurrence-level global contraction itself. In particular, a family of
moments `C^T R C` may be too lossy because it removes occurrence identity
before the reader can use it. Any occurrence-preserving representation is a
new research round and requires a new preregistration.

Evidence:

- `tracks/ksvd/results/tccd_v4/gate0.json`
- `tracks/ksvd/results/tccd_v4/stageA_seed0.json`
- `tracks/ksvd/tests/test_tccd_v4.py`
- `tracks/ksvd/code/tccd_v4.py`
- `tracks/ksvd/code/run_tccd_v4.py`
