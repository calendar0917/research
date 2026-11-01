# LCRC Sprint 04 — from LCRC-A to a possible LCRC-B

**Round:** `lcrc-performance-longrun-v1`  
**Protocol:** `tracks/structural-sufficiency/lcrc_performance_longrun_v1/PROTOCOL.md`  
**Protocol freeze:** `analysis/protocol_freeze.json` (code hashes re-verified after every run)  
**Test exposure:** every number below is `DEVELOPMENT_TEST`; Test never selected a checkpoint, epoch, learning rate, dropout, width or seed.

## Verdict

```text
NO_LCRC_B_SIGNAL_THIS_ROUND
```

No single arm and no combination improves **both Validation and Test on at least two
anchors** with a stable, non-trivial effect. The strongest result is a
**Lipophilicity-specific** signal from `combo_chem_contrib` (five runs: three seeds
plus `random123/124`; mean Test RMSE **0.5429 vs 0.5620 BASE**, all five Test
deltas negative). It is not promoted to LCRC-B because it is a single-dataset
gain and it regresses BACE, ESOL and HIV. It is the one result worth carrying
forward as a **Lipophilicity specialist**, not as the next main model.

The most interesting single-arm direction is `rhoshape` on ESOL/aqueous
solubility (seed-stable Test gain), but it consistently worsens Lipophilicity
Test and its ESOL gain reverses at `SCALE-M`. That is a capacity-specific
effect, not a general architecture win.

## 1. What was built and frozen

Frozen BASE: `tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code/lcrc_model.py`
(S1/S2, path types/endpoints, one-shot local encoder, additive readout, no
iterative hidden-state message passing). No attention, no new message passing,
no SSL, no prototype dictionary, no DIST/PATH/G2L, no ZINC architecture search.

Three single-change arms, mapped 1:1 to the three hypotheses:

| arm | hypothesis | change | params |
|---|---|---|---:|
| `chem8` | A. local chemistry incomplete | 8 label-free atom channels (CIP_R/S/UNASSIGNED_TETRA, HBD, HBA, HYB_SP/SP2/SP3); first `atom_encoder` linear 51→59, new columns zero | 248,130 |
| `rhoshape` | B. global statistics lossy | add `max_v h_v` and `mean_v |h_v − mean h|` (128 dims) to `rho`; new first-layer columns zero | 255,298 |
| `multicontrib` | C. single additive contribution too simple | `Q: 64→64→8`, graph sum, `delta=MLP([Q_graph, log1p(n)])`; last delta layer zero | 252,139 |
| `combo_chem_contrib` | A+C | both extensions | 253,163 |
| `combo_rho_contrib` | B+C | both extensions | 260,331 |

All extensions are zero-initialised and every run records
`model.extension_zero_init.passed == true`; the shared tensors are copied
bit-for-bit from a same-seed BASE model. Step-0 equality was tested to ≤2.4e-7
and is enforced by the unit tests.

**CHEMLOCAL8 artifact.** The 8 channels are a committed, label-free `.npz`
artifact (`analysis/source/chem8/*.npz`, 2.2 MB total) with a sha256 manifest.
The GPU host never re-parses SMILES and does not need RDKit. Local RDKit is used
only to build the artifact; the remote runtime suite passes without it
(17 passed, 2 skipped — the two hand-chemistry tests are local-only).

### P0 checks

- `uv run pytest -q .../tests` locally: **19 passed**.
- Remote (`res`, A100): **17 passed, 2 skipped** (RDKit-only build tests).
- Verified: row/cache alignment, Train-only normalisation, single-vs-batch
  consistency, atom-permutation invariance, segment statistics batch isolation,
  effective gradients in every new path, multi-task shapes, checkpoint reload,
  and step-0 BASE recovery.

## 2. P1 screening (9 runs, seed 0)

Deltas are vs the **same-round BASE parity** run (same commit, same harness,
same GPU host); historical BASE references are in `analysis/p1_table.csv`.
For RMSE, negative is better; for ROC-AUC, positive is better.

| arm | dataset | Valid Δ | Test Δ | both? | best epoch |
|---|---|---:|---:|:---:|---:|
| chem8 | Lipophilicity | −0.0130 | −0.0045 | yes | 240 |
| chem8 | ESOL | +0.0057 | −0.0172 | no | 136 |
| chem8 | BACE | +0.0073 | +0.0025 | yes | 99 |
| rhoshape | Lipophilicity | −0.0357 | +0.0140 | no | 166 |
| rhoshape | ESOL | −0.0067 | −0.0498 | yes | 266 |
| rhoshape | BACE | +0.0201 | +0.0011 | yes | 163 |
| multicontrib | Lipophilicity | −0.1154 | +0.0035 | no | 232 |
| multicontrib | ESOL | +0.0011 | +0.0103 | no | 187 |
| multicontrib | BACE | +0.0036 | −0.0078 | no | 98 |

Reading:

- `chem8` is the most **consistent** single arm: two anchors both improve, but
  the effects are small (Lipo Test −0.0045, BACE AUC +0.0025) and ESOL is mixed.
- `rhoshape` has the **largest single effect** (ESOL Test −0.0498, ~9.5%), but
  Lipophilicity Test worsens (+0.0140) while Validation improves (−0.0357).
- `multicontrib` produces a huge Lipophilicity **Validation** gain
  (−0.1154) with essentially flat Test (+0.0035); that is a Valid/Test
  disagreement, not a performance win.

No arm is a clean multi-anchor winner. The two representation directions
(`chem8`, `rhoshape`) were carried into P2; `multicontrib` was the readout
candidate.

## 3. P2 combinations

The protocol says: one combo, strongest representation + strongest readout.
The first choice used the largest single effect (`rhoshape` + `multicontrib`).
After the P1 consistency audit showed `chem8` was the only arm with **two**
anchors both improved, the pre-registered alternative
(`chem8` + `multicontrib`) was also run as a corrected P2. Both are reported;
no hyperparameter was changed and Test never selected anything.

| combo | dataset | Valid Δ | Test Δ | both? |
|---|---|---:|---:|:---:|
| rho+contrib | Lipophilicity | −0.0200 | +0.0023 | no |
| rho+contrib | ESOL | +0.0070 | −0.0227 | no |
| rho+contrib | BACE | +0.0048 | −0.0062 | no |
| chem+contrib | Lipophilicity | −0.0146 | −0.0194 | **yes** |
| chem+contrib | ESOL | +0.0154 | +0.0066 | no |
| chem+contrib | BACE | −0.0059 | −0.0159 | no |

`rho+contrib` fails on all three anchors. `chem+contrib` is the best
Lipophilicity model of the round but regresses BACE and ESOL. It is a
single-task specialist, not a multi-task LCRC-B.

## 4. P3 capacity check (`SCALE-M`)

BASE-SCALE-M = 320,230 params; rhoshape-SCALE-M = 333,030 params. Both inside
the pre-registered 320k–450k band. Same recipe, seed 0.

| dataset | arm | Valid | Test | Δ Valid | Δ Test |
|---|---|---:|---:|---:|---:|
| Lipophilicity | BASE-M | 0.7301 | 0.5917 | — | — |
| Lipophilicity | rho-M | 0.6409 | 0.5632 | −0.0891 | −0.0285 |
| ESOL | BASE-M | 0.5191 | 0.5159 | — | — |
| ESOL | rho-M | 0.5234 | 0.5315 | +0.0043 | +0.0157 |

The rhoshape gain is **capacity-dependent**: it wins Lipophilicity at SCALE-M
but loses ESOL at SCALE-M, the reverse of the base-capacity ESOL result. This
is not a stable architectural improvement.

## 5. P4 transfer

### Rhoshape (seed 0)

| dataset/split | Valid Δ | Test Δ | both? |
|---|---:|---:|:---:|
| FreeSolv | −0.0406 | −0.0682 | yes |
| Lipophilicity random123 | −0.0342 | −0.0137 | yes |
| Lipophilicity random124 | −0.0293 | +0.0007 | no |
| BBBP | +0.0015 | +0.0101 | yes |
| Tox21 | −0.0033 | +0.0068 | no |
| ogbg-molhiv (matched short BASE) | +0.0129 | −0.0060 | no |

Transfer is mixed: real gains on FreeSolv and random123, flat/mixed on
random124/BBBP/Tox21, and a Test regression on HIV after the matched
short-protocol BASE is used. The historical HIV comparison would have looked
positive; the matched BASE shows that the apparent gain was mostly the shorter
training protocol, not the architecture.

### combo chem+contrib (seed 0)

| dataset/split | Valid Δ | Test Δ | both? |
|---|---:|---:|:---:|
| Lipophilicity random123 | +0.0044 | −0.0324 | no |
| Lipophilicity random124 | −0.0032 | −0.0062 | yes |
| BBBP | +0.0012 | +0.0039 | yes |
| FreeSolv | +0.0597 | −0.0456 | no |
| Tox21 | −0.0013 | +0.0005 | no |
| ogbg-molhiv (matched short BASE) | −0.0095 | −0.0256 | no |

## 6. P5 seed stability

### Rhoshape on Lipophilicity — the primary metric

| seed | BASE Test | rho Test | Test Δ | Valid Δ |
|---:|---:|---:|---:|---:|
| 0 (P1) | 0.57045 | 0.58449 | +0.0140 | −0.0357 |
| 1 | 0.56337 | 0.63239 | +0.0690 | −0.0354 |
| 2 | 0.57794 | 0.58727 | +0.0093 | −0.0854 |
| **mean** | **0.5706** | **0.6014** | **+0.0308** | **−0.0522** |

Validation improves on all three seeds; Test worsens on all three. This is the
clearest Valid/Test divergence in the round and disqualifies rhoshape as a
Lipophilicity main model.

### Rhoshape on ESOL / FreeSolv — seed-stable but capacity-fragile

| task | seed 0 Test Δ | seed 1 Test Δ | seed 2 Test Δ | mean Test Δ |
|---|---:|---:|---:|---:|
| ESOL | −0.0498 | −0.0411 | −0.0026 | **−0.0312** |
| FreeSolv | −0.0682 | +0.0015 | −0.1518 | **−0.0729** |

ESOL is the only **seed-stable Test gain** in the round (all three negative),
but it disappears at SCALE-M (+0.0157) and does not generalize to the primary
Lipophilicity metric. FreeSolv has a positive mean but high seed variance.

### Combo chem+contrib on Lipophilicity — robust specialist

| run | BASE Test | combo Test | Test Δ | Valid Δ |
|---|---:|---:|---:|---:|
| seed 0 | 0.57045 | 0.55104 | −0.0194 | −0.0146 |
| seed 1 | 0.56337 | 0.54722 | −0.0161 | −0.0688 |
| seed 2 | 0.57794 | 0.55675 | −0.0212 | −0.1602 |
| random123 | 0.56982 | 0.53739 | −0.0324 | +0.0044 |
| random124 | 0.52815 | 0.52191 | −0.0062 | −0.0032 |
| **mean** | **0.5620** | **0.5429** | **−0.0191** | **−0.0485** |

All five Test deltas are negative; four of five are both-split improvements.
This is a genuine, repeatable Lipophilicity gain. It is still a single-dataset
gain, and the same model regresses BACE, ESOL and HIV, so it does not meet the
LCRC-B bar.

## 7. Attribution of the strongest signal

Diagnostic on the frozen `combo_chem_contrib` Lipophilicity checkpoint
(`analysis/attribution_combo_lipo.json`). Ablations are at fixed trained
weights; they measure pathway usage, not a retrained counterfactual.

- Full Test RMSE: **0.5510**.
- Zero the `delta` pathway: RMSE 0.6196; mean |prediction change| 0.235,
  max 1.315.
- Zero the 8 chem8 input columns: RMSE 0.7858; mean |prediction change| 0.367,
  max 2.164.
- Both pathways are actively used; neither is collapsed.
- Chem8 new-column weight norm is 0.216× the original-column norm.
  Chirality/HBD-containing molecules have larger mean absolute ablation
  effects (0.53/0.47) than molecules without (0.39/0.27), but the sign is
  uniformly negative, so this is usage intensity rather than a clean chemical
  mechanism.
- MULTICONTRIB8: all 8 channels are alive (std 2.42–3.63), no channel collapse;
  maximum off-diagonal channel correlation 0.706, so the channels are correlated
  but not redundant.

The attribution supports the interpretation that the Lipophilicity gain comes
from **both** changes, not from one pathway alone. It does not rescue the
multi-task case.

## 8. Decision and what it means

| candidate | evidence | verdict |
|---|---|---|
| CHEMLOCAL8 | 2/3 anchors both, but small effects | weak consistent direction; not LCRC-B |
| RHOSHAPE | ESOL seed-stable; Lipo Test worse on all seeds; SCALE-M reverses ESOL | task/capacity-specific; not LCRC-B |
| MULTICONTRIB8 | Lipo Valid huge, Test flat; BACE tiny; ESOL worse | readout alone not enough |
| COMBO rho+contrib | no anchor both | rejected |
| COMBO chem+contrib | robust Lipophilicity, regresses BACE/ESOL/HIV | **best specialist, not LCRC-B** |

Per the pre-registered decision rules this is **Case E**:
`NO_LCRC_B_SIGNAL_THIS_ROUND`.

The result is not "nothing happened". It says:

1. The global-statistics hypothesis (B) has a real, seed-stable effect on
   aqueous solubility (ESOL), and FreeSolv mean improves, but the effect is
   capacity-fragile and does not transfer to Lipophilicity.
2. The local-chemistry hypothesis (A) gives small, consistent gains on
   Lipophilicity and BACE, and is a useful component of the Lipophilicity
   specialist.
3. The readout hypothesis (C) alone is not supported: the huge Lipophilicity
   Validation gain does not reach Test.
4. A+C is the best Lipophilicity combination found, but it is a specialist,
   not a multi-task main model.

### Recommended next step (not executed this round)

If the programme wants a **Lipophilicity-first** model, carry
`combo_chem_contrib` forward as a specialist and test it on more Lipophilicity
repeats/splits before naming it. If the programme wants a **multi-task**
LCRC-B, the next round should look for a combination that preserves the
Lipophilicity gain while not regressing BACE/ESOL/HIV — e.g. a less aggressive
readout than `multicontrib`, or a chem8-only model with a small capacity
increase. Do not add more descriptors or attention mechanisms on the basis of
this round.

## 9. Budget, provenance, artifacts

- GPU: **3.50 A100 GPU·h / 15.0** (49 one-GPU runs; `analysis/gpu_ledger.json`).
- Wall time: ~3 h 10 min of the 8 h window, leaving >4 h for sync/report/Git.
- Test reads: 49, all `DEVELOPMENT_TEST`, ledger in
  `analysis/test_access_ledger.json`; checkpoints in
  `analysis/checkpoint_manifest.json`.
- Every formal run records the commit, code hashes, GPU/torch/driver, row
  hashes and zero-init audit in its `train.json`.
- `rr` registry: `analysis/rr_run_registry.json`.
- Tables: `analysis/p1_table.csv`, `p2_table.csv`, `p2b_table.csv`,
  `p3_table.csv`, `p4_table.csv`, `p4b_table.csv`, `p5_table.csv`,
  `p5b_table.csv`.
- Protocol deviation, stated explicitly: after the first P2 combo was
  rejected, the pre-registered alternative `chem8+contrib` was run (3 runs);
  after the combo showed a robust Lipophilicity signal, extra seed/repeat runs
  were added to test its stability (6 runs). These are reported, not hidden.
  Total GPU use remained far below budget.

## 10. Reproduction

```bash
# local correctness
uv run pytest -q tracks/structural-sufficiency/lcrc_performance_longrun_v1/tests

# protocol freeze / code-hash check
uv run python - <<'PY'
import sys; sys.path.insert(0, "tracks/structural-sufficiency/lcrc_performance_longrun_v1/code")
import cl_common as C; print(C.verify_protocol_freeze())
PY

# rebuild tables/ledgers from committed run artifacts
uv run python tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_report.py

# attribution of the Lipophilicity specialist
uv run python tracks/structural-sufficiency/lcrc_performance_longrun_v1/code/cl_attrib.py
```

All formal runs were launched through `rr` on `res` (`res-gpu0` / `res-gpu1`),
one training per GPU pool at a time. The remote checkout was kept at the
committed revision; run artifacts were pulled and committed after the campaign.
