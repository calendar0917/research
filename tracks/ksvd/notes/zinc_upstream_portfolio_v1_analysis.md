# ZINC upstream portfolio v1 — analysis (decision-first)

Round: `zinc-upstream-portfolio-v1`  ·  study `zinc-context-gap`
Code revision: `4c2667ee0ace` (branch `upstream-portfolio-v1`); CODE stage-0 proxy revision `574cc091`.
Parent: `E2E-DictEnv-Scale-v1` Full `LatentScaleSEM108`, 408,651 params, published soup valid
`0.1191540920053958`, checkpoint sha256 `17f5fcc3cdf32462657006ec89ebf2ecc72ab6fbc954b926af248aca374574eb`,
split fingerprint `58c69506df857bd8452481c6dcdd608ad230ba1a1aa90624850068fdd8faf28a`.
Official test never instantiated in any run of this round.

## Verdict

**本轮没有值得完整训练的候选。三臂全部关闭，不购买任何新鲜初始化的 320 轮 screen。**

- **CODE** — closed at its *frozen stage-0 conditional proxy* (`CODE_CONDITIONAL_STOP`).
- **DROP** — 80-epoch warm pilot `0.115925` vs control `0.114399`; gain **−0.001526**, groups 1/5 → gate FAIL.
- **R3** — 80-epoch warm pilot `0.113868` vs control `0.114399`; gain **+0.000531**, groups 4/5, ex-172 **+0.000442** → gate FAIL (both `≤0.111` and `gain ≥ 0.004` missed).

No follow-up seed / K / s / λ / dropout-p / horizon / LR rescue is authorized. This is a
short-screen non-purchase result; it is **not** a claim about the ceiling of a fresh-initialized
end-to-end model.

## The table (protocols kept separate)

| stage | arm | protocol / init | matcher (same renderer) | raw valid | cal. valid | gain vs matcher | groups | status |
|---|---|---|---|---|---|---|---|---|
| cached conditional | CODE | frozen parent + fixed convex H39+z head | replayed parent H39 `0.115023784` | — | `0.115068616` | **−0.000045** | 3/5 | **closed** `CODE_CONDITIONAL_STOP` |
| warm80 | CONTROL | parent soup, 80 ep, lr 1e-4, last-5 soup | *is* the matcher | `0.114414` | `0.114399` | — | — | reference |
| warm80 | DROP | control + train-only atom mask p=0.10 | CONTROL cal `0.114399` | `0.117198` | `0.115925` | **−0.001526** | 1/5 | **closed** |
| warm80 | R3 | control + 36-D exact shell-3 block | CONTROL cal `0.114399` | `0.113842` | `0.113868` | **+0.000531** | 4/5 | **closed** |
| fresh320 | — | — | — | — | — | — | — | **not run** (no gate passed) |

The cached conditional and the warm80 rows are **different protocols** and are not combined into
a single improvement figure: cached CODE is a frozen-parent linear-probe on the historical
`[1, H39]` target, while the warm80 arms are full-model short training from the same parent soup
under the new runner. Within each row, the comparison is matched (calibrated-vs-calibrated,
same renderer, same split).

## What was actually run

### CODE stage-0 conditional (local CPU, run `20261002-150804-a1c659d2`, revision `574cc091`)
The `H39_PHYSICAL` reference is a fit on the **second** reader hidden `[1, H2]` (post-ReLU), not
the first; refitting it reproduces the historical baseline exactly (`0.11502378425375083` →
refit `0.11502413546485733`). Appending the code moments `z = [Σ c, Σ c²]`, `c = rho*alpha`
(K=288, width 576, two train-only RMS scales) gives `0.11506861603035269`, gain
`-4.483e-05`, groups 3/5, solver `CONVERGED` (gap `6.165e-07`). Gate required
`valid ≤ 0.112` **and** `gain ≥ 0.003`; neither held → `CODE_CONDITIONAL_STOP`. The proxy is
deterministic and certified; the negative closes this fixed proxy only.

### Shared warm-80 pilot (res-2 Slurm, node c05, A100-PCIE-40GB, driver 525.85.12, torch 2.5.1+cu124)
All three arms warm-start the *same* frozen parent soup, train the full model 80 epochs
(train 10000, batch 128, MAE, Adam lr 1e-4, wd 1e-5, clip 5), freeze the last 5 epochs
(76..80) into a weight soup, fold each arm's own train-median output bias, and re-evaluate all
1000 official-valid rows. Control is shared, bought once.

| run | arm | params | soup sha256 | s/epoch | wall s | peak MB | bias Δ |
|---|---|---|---|---|---|---|---|
| `20261002-152611-e30fc31a` | control | 408651 | `ad93338e…` | 10.52 | 855.0 | 745.5 | −0.003617 |
| `20261002-152626-c036a7b8` | drop | 408651 | `6fb948a9…` | 10.70 | 869.0 | 746.3 | −0.016984 |
| `20261002-154105-9f1bd08f` | r3 | 420963 | `8cb931a3…` | 10.28 | 835.4 | 745.1 | −0.003256 |

R3's learned train-only block RMS scales were `[0.483, 1.495, 0.165]` (atom / shell2-3 bond /
within-shell3 bond), i.e. the shell-2→3 bond block carried the most scale.

Two short smokes (`up-smoke-control`, `up-smoke2`) failed on a `Tensor is not JSON serializable`
reporting bug; the fix is `_jsonable`; `up-smoke3` (`20261002-152330-2e7250e6`) passed and gave
the steady-state estimate (~11.9 s/epoch, 724 MB) used to buy the pilots. Two concurrent Slurm
jobs on quota 2 overlapped the pilots; r3 additionally built its shell-3 raw cache on first run.
No budget was exhausted, no task left running.

## Gate arithmetic (pre-registered, all four required)

Candidate passes iff cal `≤0.111` **and** gain vs same-round CONTROL cal `≥0.004` **and**
`≥4/5` fixed `id%5` groups improve **and** non-`id172` gain `≥0.002`.

| arm | cal ≤0.111 | gain ≥0.004 | groups ≥4/5 | ex172 ≥0.002 | verdict |
|---|---|---|---|---|---|
| DROP | 0.115925 ✗ | −0.001526 ✗ | 1/5 ✗ | −0.001496 ✗ | FAIL |
| R3 | 0.113868 ✗ | +0.000531 ✗ | 4/5 ✓ | +0.000442 ✗ | FAIL |

Fixed-group deltas (control − arm, positive = arm better):
- DROP: `[−0.00093, +0.00049, −0.00216, −0.00267, −0.00237]` (1 positive)
- R3:   `[−0.00042, +0.00050, +0.00178, +0.00013, +0.00068]` (4 positive)

## Interpretation

- **CODE's moment readout adds no information** beyond the frozen `[1, H39]` linear probe:
  a 257→576-column convex head on certified features cannot beat the parent's own output at the
  same checkpoint. The code moments are computed correctly (`c @ V_L == E`, verified per batch),
  so this is a genuine "the moments are already linearly spanned / not predictive" result for this
  proxy, not a wiring bug.
- **DROP is a small, direction-consistent regularizer loss.** Training-only atom dropout (`p=0.10`)
  slightly *worsens* the calibrated valid MAE and moves only 1/5 groups down. It does not improve
  the real train–valid gap on this short warm horizon.
- **R3 is the only directionally positive arm** (+0.000531, 4/5 groups) and it is also the most
  expensive (420,963 params, +36 fusion columns), but its effect is ~8× below the purchase gate
  and its ex-172 gain is ~4.5× below the descriptive floor. This is consistent with the prior
  radius-3 VQ proxy (`0.115479`, worse than radius-2): exact shell-3 local counts carry a small
  amount of usable signal but not enough to buy a full run.
- Because **no** arm cleared the gate, the fresh-320 stage is correctly **not run** — spending it
  would be exactly the "buy a success story" behaviour the round forbids.

## Reproduction

```bash
# local focused checks (new code only)
uv run pytest -q tracks/ksvd/tests/test_upstream_portfolio_v1.py

# CODE stage-0 conditional (local CPU, frozen parent + convex head)
uv run research run zinc_upstream_portfolio_v1 --mode stage0-code \
  --set runtime.device=cpu

# warm-80 pilots (remote A100, parallel, one shared control)
rr run res-2 up-pilot-control --gpus 1 --cpus 4 \
  --result tracks/ksvd/results/e2e_dictenv_upstream_portfolio_v1 -- \
  uv run --no-sync research run zinc_upstream_portfolio_v1 --mode pilot \
  --purpose "shared control" --set model.stage=pilot --set model.arm=control --set runtime.device=cuda:0
# … identical for model.arm=drop (up-pilot-drop) and model.arm=r3 (up-pilot-r3)
rr status up-pilot-r3 ; rr logs up-pilot-r3 ; rr pull up-pilot-r3
```

Parents are loaded by **sha256** (`17f5fcc3…`), never re-trained; `data/**` and
`tracks/*/results/**` caches are git-ignored and were copied to the res-2 checkout as untracked
files. `H39_PHYSICAL.npz` needed a force-add (`.npz` is git-ignored).

## Artifacts

- results: `tracks/ksvd/results/e2e_dictenv_upstream_portfolio_v1/`
  (`pilot_summary.json`, `pilot_{control,drop,r3}.json`, `valid_predictions.npz`,
  `stage0_code.json`, `REPORT.md`, `DECISION.md`)
- promoted runs: `records/runs/20261002-150804-a1c659d2.json`,
  `20261002-152611-e30fc31a.json`, `20261002-152626-c036a7b8.json`, `20261002-154105-9f1bd08f.json`
- rr jobs: `up-pilot-control-20261002-152756-17b74fbe`,
  `up-pilot-drop-20261002-152811-0fbb708a`, `up-pilot-r3-20261002-152829-fae8154f`
- claim: `claim-zinc-upstream-portfolio-v1-no-purchased-candidate-20261002`
- decision: `decision-zinc-upstream-portfolio-v1-stop-no-candidate-20261002`
- preregistration: `tracks/ksvd/notes/zinc_upstream_portfolio_v1_preregistration.md`