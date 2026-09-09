# CPU Determinism & Serial-Execution Policy (reproducibility note)

> Status: active policy — 2026-09-09
> Applies to: all numeric runs of the ZINC compact family (`zinc_patch_path_pooling`
> runner, protocol `zinc-context-gap`) and, by extension, any promoted numeric run
> on shared CPU machines.
> Related evidence: `notes/compact_v4_global_topology_channel.md` §12 (execution
> discipline); `notes/compact_v4_multiseed_protocol_confirmation.md` (this stage).

## Observation

Same code + same seed + same machine + same environment, but with
**>= 3 concurrent CPU training processes** on a 16-logical-thread machine, produce
run-to-run numerical divergence **from epoch 3 onward** (epochs 1–2 stay
bit-identical, then valid-MAE traces diverge; e.g. the 4-way concurrent hinge run
`20260909-154106-7f962608` reached valid 0.1692115 @54 while the canonical solo
hinge run is 0.1700656 @53; traces compared epoch-by-epoch: first divergent epoch 3,
58/60 epochs differ).

Concrete A/B evidence (2026-09-09, all `zinc-context-gap`, same day):

| concurrency | run(s) | valid best MAE | bit-identical with canonical? |
|---|---|---|---|
| 4-way (15:41:06 batch) | `...154106-7f962608` (hinge) | 0.1692115 @54 | **no** (diverges from epoch 4) |
| 2-way (16:13:02 batch) | `...161302-4c318efd` (v4-none) | 0.18415821571176638 @56 | **yes** (== v2 canonical) |
| 2-way | `...161302-97c62f91` (capacity) | 0.18425278290727876 @60 | yes (== its own solo rerun) |
| solo reruns | `...164306/164801/165340/165917/171303/172752/174717` | see notes | yes (bit-identical among repeats) |

Interpretation: thread oversubscription perturbs BLAS/OpenMP reduction order inside
torch CPU kernels; the perturbation is seed- and process-scheduling-dependent and is
**not** a code or data problem (identical inputs, identical code).

## Control

Serial isolated run (one training process on an otherwise idle machine, the repo's
`runtime.torch_threads: 4` setting unchanged) gives **bit-identical reproduction**:

- v2 canonical: `20260907-193612-46c1a12f` (screen) / `20260907-194818-604fa0f5`
  (terminal) / `20260909-174717-2b498e06` (terminal, with state dicts): all valid
  0.18415821571176638 @56, refit test 0.1353615188403055.
- v4-hinge: `20260909-165340-96be4349` (solo screen) == `20260909-171303-95a4f542`
  (terminal): valid 0.17006561887910357 @53, refit test 0.13944621286727488.
- Multi-seed confirmation stage (2026-09-09, `compact_v4_multiseed_protocol_confirmation.md`):
  seed-0 guard runs re-produced the canonical traces epoch-by-epoch.

## Canonical policy (from now on)

1. **All promoted / canonical runs are executed serially**: one training process at a
   time, same machine, same environment, no concurrent model training of any kind
   during a run (including other researchers' processes when detectable).
2. Runs that must be compared bit-wise or ranked scientifically are executed under the
   same `runtime.torch_threads` (4), same process settings, and recorded with
   Python/PyTorch versions, git commit, code-state hash and config hash in the run
   manifest.
3. Concurrency variance must never be mixed into seed variance: multi-seed statistics
   are computed only from serial runs.
4. If concurrency cannot be guaranteed, runs are marked non-canonical
   (purpose must say so) and are not promoted into records/ without a bit-identity
   guard against a serial rerun.

## Environment baseline (this machine, 2026-09-09)

- CPU: AMD Ryzen 7 8845H (8 cores / 16 threads), machine otherwise idle.
- Python 3.12.14 (repo `.venv`), torch 2.5.1+cu124 (CPU execution), numpy 2.1.3,
  networkx 3.4.2, sklearn 1.5.2, PyG 2.6.1.
- `torch.set_num_threads(4)` via config `runtime.torch_threads: 4` for every run.
