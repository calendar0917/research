# PROTOCOL — task_response_composition_s0_v1

Round brief: `/home/calendar/Downloads/结构响应组合_8小时研究任务与AgentPrompt.md`.
This file is written **before the first formal training run** and only records
the frozen plan; later commits may append timing facts, actual conditions and
explained deviations. It is never rewritten to look better than the plan.

## Question

Can the task effect of an explicit fragment be carried by a small shared
response state, with the environment acting through that state — under an
explicit sharing restriction that is checked against a real matched
replacement directory — and does that improve whole-molecule prediction and
data efficiency? Main task: whole-molecule property prediction
(`ogbg-mollipo`). No GNN backbone, no pretraining, no distillation, no strong
model residual. GINE is an external read-only prediction coordinate only.

## Fixed structure

* objects: every `(root, atom_set)`, `1 <= |atom_set| <= 4`, root in the set,
  induced subgraph connected; all internal bonds keep the raw 9/3 attributes.
* parse: a **bridge** bond `b=(u,v)`; each side with `1 <= |S| <= 8` is a parse
  (`S` word fragment, `C` environment); both directions kept when both fit.
* fallback: a molecule with **no** valid bridge parse contributes one parse per
  atom: `S = whole G`, `C = empty`, independent learned null cut.
* every prediction averages over **all** parses of the molecule.
* fragment input: 6 within-fragment port-distance buckets
  `0,1,2,3,>=4,unreachable`, each `mean_d(32) + density_d(32) + log1p(n_d)`,
  plus `log1p(n_atoms)`, `log1p(n_bonds)`, `log1p(n_components)`, `1` → 394.
* exact keys: canonical SMILES of a colour-encoded auxiliary graph (RDKit,
  disjoint isotope ranges for ATOM/BOND/PORT, no sanitisation to chemistry).
  context key = `canon(C,p_C) + cut bond raw 3-tuple`; word key = `canon(S,p_S)`.

## Models (R0)

`phi`: 9 atom categoricals × emb4 → 36 → AtomMLP 36-32-16; 3 bond categoricals
× emb4 → 12 → BondMLP 12-16-8; frame input 132 (4×16 + 6×8 + 4 + 6 + 6 + 4) →
FrameMLP 132-128-32-LN32; object phi = mean over all winning root-fixed frames.

`E` (shared on S and C): 394-192-128-LN128, SiLU, final dropout 0.10.

| arm | cut head |
|---|---|
| `FACTOR` | `beta + a(h_C) + a(h_S) + b(e) + <q(h_C,e), r(h_S,e)>/sqrt(K)`, `K=32`, `q`,`r` tanh |
| `DIRECT` | `beta + head([h_C,h_S,e])`, 264-96-48-1 |
| `ADD` | `beta + a_add(h_C) + a_add(h_S) + b(e)`, `a_add` 128-128-96-1 |
| `GLOBAL` | no cut; per object-size bucket (1..4) `mean phi, sum phi/n_atoms, log1p(n)` → 263 → 256-128-1 |

Empty `C`: `E` is not run; `FACTOR`/`DIRECT` use a learned `h_empty[128]`,
`a(empty)=0`. Null cut uses a learned 8-vector, distinct from a real all-zero
bond record. No whole-molecule bag bypass, no teacher residual, no zero-init
join, no pretrained weights.

## Losses

`L_main` (cut arms) = `mean_G mean_{c in parses(G)} (f_c - y_std)^2` — averaged
**inside** a molecule first. `GLOBAL`: plain molecule MSE.
Aux (`CUTRESP`, `MOLPAIR`) = `mean((delta - delta_y/sigma_y)^2)/tau2`,
`L = L_main + 0.25 L_aux`, `tau2 = max(0.25, E_sampler[(delta_y/sigma_y)^2])`
computed exactly on the finite sampling distribution, training labels only.
Pair batch 16/step; 80% from transforms present in >= 2 Train pair-components,
20% from all transforms; uniform transform → component → pair.

## Recipe

AdamW LR 1e-3, wd 1e-4, clip 5, batch 64 molecules + 16 aux pairs, 5 epoch
linear warmup, cosine to 1e-5, seed 0, single seed. Horizon `E` **240 if**
`240 × 1.25 × median_epoch_seconds <= 50 min`, else 160 if it fits, else 160 and
later units are reduced. `E` is decided from epoch-3..5 steady-state timing
(colours of the loss curve are never used) and written to
`analysis/timing.json` before epoch 6. Every epoch evaluates full Train/Valid/
Test; checkpoint selection is Valid RMSE only; Train RMSE is always reported on
whole-molecule `f_bar`, cut-level Train error separately.

## Queue (max 20 units)

A. core 7: `FACTOR_CUTRESP`, `DIRECT_CUTRESP`, `FACTOR_NONE`, `DIRECT_NONE`,
`GLOBAL_NONE`, `FACTOR_MOLPAIR`, `ADD_CUTRESP` (runs 1/2 first, in parallel).
B. at most one triggered diagnostic (`R_CONSIST` / `R_GEN` / `R_CAP` /
`R_RANK`) chosen by the frozen priority.
C. ESOL: `GLOBAL_NONE`, `DIRECT`, `FACTOR`, `GINE_RAW_REF`.
D. at most 3 mechanism units (`DIRECT_MOLPAIR` slot, `ERASE_DISTANCE`, `K8`/
`K128`, matched `ADD_*`, lambda_var=0).
E. at most 4 small-sample units (Train 50%/25%, nested sha256 subset).

Resource stop: after the core 7 and one triggered correction, if every FACTOR is
> 0.76 Valid, worse than GLOBAL by > 0.04, and no FACTOR beats the matched
DIRECT by >= 0.01, stop the neural extension of this family.

## Engineering checks (before formal training)

Input/scaler/split/row_id; frame fields and masks; cut recombination restores
the original attributed graph; renumbering invariance of objects/keys/buckets/
parses and whole-molecule prediction (FP64 on CPU, FP32 on GPU); identical
context features across both pair endpoints and the FACTOR algebraic delta
identity; shuffled-batch equals single-molecule forward; sparse aggregation
equals an explicit loop; duplicate-frame compression is exact; one real batch
produces finite main/aux losses and real gradients on phi/E/q/r/heads.

## Deviations log

(appended during execution; each entry needs a reason)

## Deviations log (appended during execution)

1. **2026-11-21T23:50 environment.** `rdkit` is not present in the `res`
   virtualenv (`uv sync --frozen` installs default groups only; rdkit lives in
   the optional `lipo` group). Preparation is CPU data work, so the prepared
   caches are built **locally** (where rdkit exists) and transferred with
   `rsync` to the compute checkout. All experiment execution still goes through
   `rr`. `analysis/timing.json` records the remote epochs; `data/` is
   gitignored on both sides.
2. **2026-11-21T23:56 engineering rerun #1 (invalid runs discarded).** The
   first `LIPO_FACTOR_CUTRESP_R0` / `LIPO_DIRECT_CUTRESP_R0` launches were
   cancelled at epochs 78 / 24: `C` was being set to the *other side of the cut*
   instead of `V(G) \ S`, which silently dropped other connected components of
   multi-component molecules (one molecule). Fixed in `trc_data.cut_parses`;
   cache rebuilt; both units restarted from the common initial state. The
   invalid runs are kept in `results/LIPO_*_R0_invalid_2026-11-21/` and are not
   reported as results.
3. **2026-11-22T00:20 engineering rerun #2 (same restarts).** Phase-2 CSR
   assembly indexed `obj_frame_id` by object index instead of by
   `obj_frame_offset`, which lost the multiplicity of multi-frame rooted
   objects and attached wrong frames to a minority of fragments (76/29283).
   Fixed; cache rebuilt (`sha256 c53fee5d...`). Because (2) and (3) are both
   correctness defects found before any reported result, they are treated as a
   single repair cycle of the two affected units.
4. **2026-11-22T00:05 `res` GPU0 is shared** with an unrelated process already
   resident (22 GB, foreign PID). It is not touched; the second core unit runs
   concurrently on GPU0 as instructed and is correspondingly slower.
