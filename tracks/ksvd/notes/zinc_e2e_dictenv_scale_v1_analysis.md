# Analysis — ZINC E2E-DictEnv-Scale-v1 (unified Small/Full shared task dictionary, Full m=3 seed 0, local CPU)

Round: `zinc-e2e-dictenv-scale-v1` (task package
`/home/calendar/Downloads/zinc_dictionary_scale_plan.md` +
`zinc_dictionary_scale_agent_prompt.md`, handoff
`zinc_dictionary_scaling_handoff.zip` vendored as
`experiments/luyin16/dictionary_scaling/v1_20261001/`, audit revision
`00c203624682a9ca46d54a225721e25015d8b28c`).

Formal runs: training `20261002-004025-2196abdb` (320/320 epochs at
formal-run commit `5335301d0e8f36c67b1acd8fb192d92c79769e3e`), completion
(probes + analysis on the cached frozen training state)
`20261002-023535-07a8b584` at commit `638d6a3` (promoted record
`records/runs/20261002-023535-07a8b584.json`), preregistration sha `c65d6e223aa9`
enforced by preflight.  Official test never instantiated
(`official_test_loaded = false`).

## 1. Verdict

`M_S = 0.119154` (soup valid MAE) → band
**SCALE_LIMITED_SIGNAL_CLOSE_CAPACITY_ROUTE** (`0.115 < M_S <= 0.120`, case C:
limited signal, this round closes without further widening), single Full seed-0
trajectory, no matched control, no official-test read.

- best single epoch `0.123307` @ 299, Top-5 soup members [268, 276, 289, 299,
  317] with member valid MAE 0.123804 / 0.125532 / 0.125689 / 0.123307 /
  0.123545.
- first/last valid `0.526117` / `0.125832`; training-loop final train `0.059272`,
  minimum `0.058894`.
- same-protocol (eval-mode, no-shuffle) soup train MAE `0.045397`, train→valid
  gap `+0.073757`.
- wall `6274.4 s` train (`19.608 s/epoch`) + `58.9 s` interventions/analysis;
  peak RSS `3053.4 MB`.

Unmatched backgrounds: latent-bridge `0.121058` (difference `-0.001904`),
Sem108 `0.123705` (difference `-0.004551`).  Both differences are single-seed
context, not increments.

## 2. What this round tested

The previous round showed that a 9,216-parameter shared task dictionary at the
Sem108 fusion output is trainable, load-bearing and near-dense, but its
borderline endpoint (`0.121058`) was never compared against a stronger capacity
regime.  This round built the **single unified Small/Full model**: one code
path where a scale multiplier `m` widens only the task-local path — fusion
output `h`, the shared task dictionary, the pair encoder, the distance gate,
the relation encoder and the reader input — while every Sem108 module (local
object, static relations that are functions of the readout, readout topology)
is left byte-identical.  `m = 1` reproduces the previous round's
`LatentBridgeSEM108` exactly (106,925 parameters, state dict bit-identical);
`m = 2` is audit-only (233,203); `m = 3` is the single authorised Full
candidate (408,651).  No graph index, no message passing, no residual bypass,
no new raw features, no dense/random control, no second seed, no ensembles.

Deliverables: scale core `experiments/luyin16/e2e_dictenv_scale_v1.py`, stage
driver `experiments/luyin16/zinc_e2e_dictenv_scale_v1.py`, control-plane runner
`runners/zinc_e2e_dictenv_scale_v1.py`, config
`configs/luyin16/zinc_e2e_dictenv_scale_v1.yaml`, vendored handoff package
`experiments/luyin16/dictionary_scaling/v1_20261001/` (sha256s in its README),
9 unit tests, and the pre-registration
`notes/zinc_e2e_dictenv_scale_v1_preregistration.md`.

## 3. A — zero-training acceptance (correctness stage, all gates passed)

Seven gates on the frozen revision, one real official-train batch of 64
molecules, no training, test never loaded (`correctness.json`):

- **S1 Small identity**: `m = 1` has exactly `106925` parameters and its state
  dict matches the frozen `LatentBridgeSEM108` constructor with zero key/value
  mismatches; plain and masked Small predictions differ by `0.0` on the real
  batch.
- **S2 Full parameter/width contract**: exactly `408651` parameters
  (body `325707` + task dictionary `82944`), inclusive per-module inventory
  exact (`fusion 202266`, `reader 33385`, `pair_encoder 46704`, ...), widths
  `h/E/alpha/u/pair/reader_input = 144/144/288/48/48/814`, all 25 fixed Sem108
  tensors identical, within the `420000` budget.  `m = 2` audits to exactly
  `233203`.
- **S3 containment witness** (eval mode only, never a training initialiser):
  embedding the exact Small state into the Full state and block-replicating it
  reproduces Small predictions with max delta `1.07e-06` plain / `1.19e-06`
  masked (tolerance `1e-4`).
- **S4 identity mode and formal init**: with `D_L = [I;0]`, `V_L = [I;0]^T`,
  `lambda1 = lambda2 = 0` the readout equals the Sem108 parent (relative L2
  `1.13e-07`, tolerance `1e-6`; repeat code exact); the frozen unit-column init
  maps `h` to itself with relative squared error `0.00237` (threshold `0.05`).
- **S5 gradients and step**: MAE-only task gradients at init are non-zero
  (`D_L 0.669`, `V_L 0.704`; fusion `1.20/1.23`, reader `25.69`), one Adam step
  moves `D_L`/`V_L` by `0.204` each, and one small SGD step lowers the loss
  `1.116 -> 1.030`.
- **S6 wiring / no bypass**: exactly one task-dictionary call on each path,
  unary and pair inputs match the dictionary output, there is no `h` bypass
  (forcing one changes predictions by `0.0229`), within-molecule relabel
  `2.4e-05` (tolerance `1e-4`), batch offset `3.0e-08`, endpoint swap `0`,
  relation no-writeback `0`, zero code gives `E = 0` exactly.
- **S7 official-test blocker**: constructing the test path raises.

**Timing gate** (pre-registered resource decision; measured on the frozen code
tree, see section 8 for the stamp note): steady train step `0.2320 s`, eval
step `0.0730 s` → `18.913 s/epoch` → `6052 s` for 320 epochs × `1.10` safety =
`6657 s` against a `14400 s` budget → `AUTHORISED`.
**Smoke**: 3 epochs on 1,024 molecules (24 steps), finite loss `0.904365`,
non-zero gradients, `6.3 s`.

## 4. B — formal 320-epoch Full screen

320/320 epochs, one trajectory, seed 0, batch 128, Adam `lr 1e-3` /
`wd 1e-5` / `clip 5.0`, fixed LR, Top-5 soup by official valid, no early stop.
Curve anchors (valid MAE): epoch 1 `0.526117`, 80 `0.150633`, 160 `0.153065`,
240 `0.135208`, 320 `0.125832`, best `0.123307` @ 299; the endpoint is the Top-5
soup `0.119154`.

Training cost: `6274.4 s` (`19.608 s/epoch`, `3053.4 MB` peak RSS) versus the
Small reference `2434.8 s` (`7.609 s/epoch`, `2313.1 MB`) — the 3.82× parameter
increase cost 2.58× wall time and finished well inside the authorised budget
(predicted `6657 s` with margin, actual `6274 s`).

Same-protocol soup train MAE `0.045397` versus valid `0.119154`
(gap `+0.073757`).  Compared with the Small reference (soup train `0.055133`,
valid `0.121058`, gap `+0.065926`), the Full candidate lowered both the train
fit and the valid endpoint, but the train improvement (`-0.009736`) is 5.1×
larger than the valid improvement (`-0.001904`), and the gap widened: the extra
capacity mostly bought fitting, not generalisation, at this horizon and data
size.

## 5. C — task-dictionary mechanism (frozen soup state, inference only)

All probes are inference-only on the frozen soup state
(`mechanism/bridge_probes.json`):

| probe | delta MAE | delta pred RMS |
|---|---|---|
| zero the task-dictionary code (`E = 0`) | +1.436962 | 1.940718 |
| within-molecule code permutation (5 seeds) | +0.394184 | 0.607477 |
| reset `D_L`/`V_L` to init (encoder/head kept) | +1.329684 | 1.746171 |

- learned: `D_L` / `V_L` Frobenius movement `16.4566` / `12.8033` from the
  frozen frame, with non-zero MAE-only task gradients `0.0963` / `0.2431` at
  the endpoint (movement already `1.25` / `1.39` at epoch 1; gradients
  `0.0664` / `0.0490`).
- code density at the endpoint: mean non-zero `250.89 / 288`
  (fraction `0.871`, p50 255, p95 275, max 286, n = 2998 objects) — again
  variable and near-dense: the soft-threshold regime barely zeros atoms in the
  wider dictionary, so this round also does not claim a sparse code.
- diagnostic low-dimensional reconstruction `h ≈ E`: mean relative `0.0588`,
  p95 relative `0.1238` (diagnostic only, never optimised; the Small reference
  measured `0.0048`/`0.0109`).
- readout block layout (frozen, measured): reader input `814` = unary
  `288+289` + pair `96+97` + topology `806..814` - `806` + global `774..806`
  - `774`; relation readout dim `485`.

Interpretation.  The dictionary is again demonstrably learned and
load-bearing, and the capacity is real (3.82× parameters, 9× dictionary), but
the intervention deltas only establish that the channel carries prediction —
zeroing it removes the whole local encoding by construction.  They do not show
that dictionary learning beats a matched trained alternative, and no such
control was run.  The near-dense code and the wider train→valid gap both point
the same way: the added capacity was used, but not in a way that moved the
valid endpoint by more than single-seed resolution.

## 6. What this round does and does not claim

Answering the plan's four questions explicitly:

1. **Did Full enter `M_S <= 0.115`?**  No: `0.119154`, inside the `0.115–0.120`
   limited-signal band, so the capacity route closes without further widening.
2. **Did train and valid improve simultaneously?**  Both values are lower than
   the Small reference (soup train `0.045397` vs `0.055133`, soup valid
   `0.119154` vs `0.121058`), but the valid change is `-0.001904` — an
   unmatched, single-seed difference — while the train fit improved 5.1× more
   and the gap widened.  No causal or significance claim follows.
3. **Was effective capacity actually added and used?**  Added: exact
   `408651` = `325707` body + `82944` dictionary, inventory and widths audited,
   containment witness passed.  Used: movement `16.46` / `12.80`, MAE-only
   gradients `0.0963` / `0.2431`, zero-code `+1.437` and permutation `+0.394`
   MAE.  Not demonstrated: that this use yields a validated gain over any
   matched trained alternative.
4. **CPU cost?**  `6274.4 s` (`1.74 h`, `19.608 s/epoch`, `3053.4 MB` peak
   RSS) versus Small `2434.8 s` (`7.609 s/epoch`, `2313.1 MB`); the
   pre-registered gate authorised the run (`6657 s` predicted with margin vs
   `14400 s` budget) and the actual wall time confirms it.

Does **not** claim any improvement over the latent-bridge background
(`0.121058`) or Sem108 (`0.123705`): different parameter counts, single seed,
unmatched protocols.  Does not claim sparse coding (measured density
contradicts it), any causal dictionary-vs-dense statement, or any generalisation
benefit (wider gap than Small).  The band is a resource-decision threshold, not
a significance test.

## 7. Round outcome and next step

Per the pre-registration, case C closes the capacity route: no wider `m`, no
`K`/`d`/`lambda`/steps/lr rescue, no seed 1, no post-hoc matched control, no
official-test read.  The two honest readings are: the unified Small/Full
construction works exactly as specified (bit-identical Small, exact Full
budget, contained semantics), and the shared task dictionary scales up as a
trainable, used component without producing a validated generalisation gain at
`m = 3` on this horizon.  A future continuation requires a new pre-registration
and must be either a confirmation design (paired seeds plus a parameter-matched
dense/random adapter, band and density pre-declared) or a change of the local
object / learning objective; it must not be another widening of `K` or `d`.

## 8. Provenance and limitations

- **Two-revision chain (explicit)**: training ran to completion at the
  formal-run commit `5335301` (run `20261002-004025-2196abdb`, 320/320 epochs,
  `run_seed0.json`/`soup.json`/checkpoints stamped `5335301`).  That run then
  crashed in the probe **reporting** line (`per_object.mean()` on a `Long`
  support count) before writing `bridge_probes.json`.  The one-line dtype fix
  `638d6a3` changes no model parameter, loss, data order, selection or probe
  definition; the completion run `20261002-023535-07a8b584` cache-hit the
  frozen training artifacts (identical state hashes: best
  `8fda5c4d97f2...`, soup `ef5c45a260d8...`) and produced the probes, summary,
  REPORT and DECISION at `638d6a3`, which the report states separately.
- **Timing stamp note**: `timing.json` is stamped `git_commit 00c2036`
  because the timing measurement was taken on the working tree minutes before
  that tree was committed byte-identically as `5335301` (the commit added no
  changes to the measured code).  The stronger confirmation is the actual
  training wall time: `6274 s < 14400 s` budget.
- `references` reconfirmed the latent-bridge background `0.121058` from the
  previous round's durable artifacts (read-only, never rerun); `audit` reused
  the frozen-interface audits byte-identically with decision `PROCEED` and the
  recorded `T1_BLOCK_AUDIT_UNAVAILABLE` limitation (no T1 retraining);
  `preflight` enforced the preregistration sha and the exact parameter counts
  (`small 106925`, `full 408651`, fixed modules identical).
- Historical numbers (T1 `0.125765`, FINAL-CLEAN-C6 `0.128499`, CSSD-q1
  `0.130028`, P1-sparse `0.131975`, Sem108 `0.123705`, latent-bridge
  `0.121058`) are background only: different architectures, parameter counts
  and protocols.
- CPU regime (`runtime.device=cpu`, 8 threads), one geometry only
  (`d = 144`, `K = 288`, 16 unrolled ISTA steps, `lambda1 = 0.05`,
  `lambda2 = 0.01`, tied `V_L`), one seed, one authoring machine; no matched
  arms; the official test split was never instantiated; `m = 2` is a parameter
  audit only and was never trained.
