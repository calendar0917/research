# METHOD_CONTRACT — zinc-task-dictionary-and-cycle-witness-seed0-v1

## 1. Objects, never to be confused

| object | what it is | provenance |
| --- | --- | --- |
| **new fold** | `rng(20261004).permutation(10000)` split of the 10000 official-train rows into 8000 fit / 2000 dev, sorted indices | generated once by the round runner; hashes in `protocol.json` / `EXECUTION.md` |
| **old fold** | the previous 8000/2000 split at seed 20261003 (`fit 165e87ef…`, `dev fb8b7806…`) | `zinc_joint_dictionary_decision_v1/prep/fold_objects.npz`; used **only** by the CPU witness task |
| **arm `D`** | fresh compressed body + current 144-d / 288-atom task dictionary bridge (16-step ISTA) | `build_deploy_model(state=None)`, no old state |
| **arm `M`** | identical fresh body + bias-free `144 -> 288 -> 144` SiLU MLP bridge, `W1=D_L_init.T`, `W2=V_L_init.T` | same constructor; bridge frame copied from the fresh D frame |
| **old checkpoints** | `R_SJ` / `R_DJ` raw soups from the previous round | used only for the section-10 J/M flip clarification; never for initialization |

## 2. Freshness contract

* `build_deploy_model(None)` runs `DeployFull.__init__` only: it seeds torch,
  builds a dummy dictionary/subspace, deletes the structural path and creates a
  **new** 110-d fusion first layer. It reads no checkpoint and no fitted object.
* No main-arm tensor is loaded from S_M/A0/C/T/O/H/Y, and no old bridge tensor
  is copied. The only saved-state reads are the replay checks *within* each
  arm's own run.
* `M` differs from `D` only by replacing `local_dictionary_bridge` with
  `MLPBridge`. Its `fc1/fc2` are copied from the fresh D-arm bridge frame; no
  trained D/V ever enters `M`.
* Shared non-bridge tensors are compared item-for-item by SHA-256 in the smoke
  test and in `{arm}_meta.json` hashes; a mismatch raises and aborts.

## 3. Input objects: refit vs reused

Only the official-train cache is loaded: `encoded_train.pt` + the train env
cache (`load_train_only`); `load_split("train")`, which also deserializes
valid, is deliberately not called. Official-valid and official-test are never
deserialized.

| input | fitted? | this round |
| --- | --- | --- |
| raw graph / endpoint occurrence caches (`dict_atom`, `env_occ_*`, `env_bond_*`, pair index/relation/bucket) | no | reused unchanged |
| `patch_cont` 146-d | per-column mean/scale | inverted through the frozen all-train cache standardizer (recovery constants only), then **refit on the new 8000 fit roots** (`floor=1e-6`) and applied to all rows |
| `global_context` 62-d | per-column mean/scale | same, refit on new fit roots |
| `anchor` 62-d | per-column mean/scale | same, refit on new fit roots |
| `topology_features` 25-d | per-column mean/scale | same, refit on new fit roots |
| `dict_phi` / `x175` / KSVD `D_fit` / common subspace | yes | **not used at all** by the compressed model (`code()` returns a zero placeholder); no KSVD refit performed |
| scalar targets `y` | no (labels) | reused; `k`/`c` only for post-freeze grouped analysis |

Every fitted vector's SHA-256 and the `fit_root_rows` count are recorded in the
arm meta. The old *fit* scalers in `fold_objects.npz` are never applied.

## 4. Bridge semantics (single insertion point)

`DeployFull.environments_masked` is the only producer of the environment tensor
used by unary and pair consumers:

```
interface = Sem108(coord, data)
h = fusion(interface)               # [*, 144]
E = local_dictionary_bridge(h)      # D: ISTA task dictionary; M: matched MLP
```

`D` keeps the exact source semantics: detached solver step
`1 / (1.05 * eigmax(Dbar Dbar^T) + lambda2)`, 16 unrolled soft-threshold steps,
`E = scale * (alpha @ V_L)`. `M` computes `E = scale * W2 SiLU(W1 x)` with the
same `scale = sqrt(mean(h^2) + 1e-12)` and `x = h/scale`.

No graph id, adjacency, node/edge state or readout bypass enters either bridge.
Loss is exactly mean `L1(y)`; there is no reconstruction term, no g/c target,
no severity weighting, no per-group sampler.

## 5. What is matched and what is not

Matched: parameter count (82,944), input normalization, shared body and data
protocol, optimizer/loss/schedule, seed.

Not matched by design: `D` runs 16 unrolled ISTA steps with a matrix dictionary;
`M` is one hidden SiLU layer with no sparsity. Parameter and FLOP/time counts
are reported; the comparison is between these two function classes, not a
purified "with vs without dictionary/sparsity" experiment.

## 6. CPU witness semantics

* `T25` is the exact float32 input of the released old CPU head
  (`T25_all.npz`), evaluated on the original 8000 fit / 2000 internal dev
  indices. Exact classes are float32 equality classes; `-0.0 == 0.0` is the
  old comparison semantics and the signed-zero count is reported.
* `global_min_l1` uses all-fit class medians; `group_only_min_l1` recomputes
  medians inside each severity group; the severity-weighted number is
  descriptive only and is never deployed as a new head.
* The repo cycle helper is the `_cycle_basis_stats` semantics
  (`nx.cycle_basis`, `cycle_score = -max(0, max_basis_length - 6)`); the
  official label process is the upstream canonical-SMILES + RDKit route whose
  per-row results are read from the audit table (`cycle_score_gvae_order`,
  `label_effective_cycle_snapped`). Renumbering changes only node insertion
  order; graph and attributes are kept isomorphic.

## 7. Replay and determinism

* Per-arm training uses a pre-hashed shared batch schedule; no global RNG
  decides batch order.
* After training, the raw-soup state is reloaded into a fresh arm and both fit
  and dev predictions are computed in eval mode; raw-soup predictions are saved
  per graph.
* A fixed 128-molecule fit batch is replayed CPU vs GPU; max abs difference must
  be <= 1e-5 or the arm fails.
* The smoke test verifies: shared body hashes, both bridge parameter counts,
  consumer use of the bridge output, `y` non-read, endpoint-offset correctness,
  real backward/optimizer updates, no auxiliary loss, and schedule match.
