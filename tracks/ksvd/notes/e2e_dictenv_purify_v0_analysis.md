# E2E-DictEnv-Purify-v0 — analysis and frozen verdict

**Verdict: `REFERENCE_REPRODUCTION_FAILURE` (pre-registration §10.2), round stopped
before the candidate was interpreted.**  Official ZINC test was never loaded.

* preregistration `e2e_dictenv_purify_v0_preregistration.md` (commit `bbcaf58`),
  amendment A1 `e2e_dictenv_purify_v0_amendment_a1.md`,
  architecture audit `e2e_dictenv_purify_v0_architecture_audit.md`
* implementation `tracks/ksvd/experiments/luyin16/e2e_dictenv_purify_v0.py`,
  runner `.../zinc_e2e_dictenv_purify_v0.py` (commit `8657ffc`, amendment `6ec6218`,
  reproducibility probe `79fb460`), 24 CPU tests green
* run commit `6ec6218`, remote `hxy@a100-2:/home/hxy/cy/research`, **GPU1 only**
  (GPU0 stayed foreign-contended, never touched), official test never opened
* artifacts: `results/e2e_dictenv_purify_v0/` (`decision.json`, `REPORT.md`,
  `DECISION.md`, `paired_performance.json`, `reproducibility_probe.json`,
  `semantic_refactor_equivalence.json`, `purity_audit.json`,
  `parameter_ledger.json`, `local_analysis.json`, curves, states)

Independent local recomputation (`code/analyze_e2e_dictenv_purify_v0.py --write`):
**38 checks, 0 errors**, verdict reproduced exactly, including the STOP enforcement
(no seed-1 artifact, no mechanism/ablation/health artifact, no interpreted candidate).

## 1. What the round did deliver

| item | result |
|---|---|
| Stage 0 semantic refactor (four concept modules) | **PASS**, CPU path bit-identical on all 12 comparison points (max abs diff `0.0`; frozen tolerance `1e-6`) |
| Stage 0 on CUDA | prediction `8.345e-07` vs a measured same-implementation rerun noise floor of `5.722e-06`; gate passed (amendment A1) |
| parameter ledger | reference **97,487** → purified **95,711** (`−1,776`, `−1.822 %`), confirmed against live parameter counts |
| purity audit | local raw-chemistry bypasses **3 → 0**; local chemistry entry points **5 → 2**; chemistry-specific parameter mass 7,570.6 → 4,498.6 |
| smoke gates (both arms, GPU1) | all pass, incl. non-zero gradient into the constant channel rows |
| matched seed-0 arms | reference soup **0.1298781834495603** (best 0.13576880361413352 @274), purified soup **0.13549722206493606** (best 0.14364955635269872 @246), both complete 320-epoch curves, ~2,140 s each, ≤ 219 MB peak |

So the *architectural* goal of the round is realised and verified: the local
zeroth-order chemistry no longer bypasses the attributed measure into the anchor,
the anchor is size-only, the dictionary/reader/composition semantics are
unchanged (bit-identical reference path), and the parameter count went **down**.

## 2. Why the round stopped

```text
Delta_purification(seed 0) = MAE_purified - MAE_reference
                           = 0.13549722206493606 - 0.1298781834495603
                           = +0.005619038615375743      -> seed-0 case `performance_failure`

reference reproduction drift = 0.1298781834495603 - 0.12354862861608853
                             = +0.006329554833471779   > tolerance 0.005
                             -> REFERENCE_REPRODUCTION_FAILURE, STOP
```

Section 10.2 is unambiguous and one-sided in exactly this direction: while the
fresh baseline drifts, **the candidate is not interpreted**.  `run_all` enforced
it in-process (the job raised by itself, exit 1): no seed 1 was bought, no
mechanism intervention, no constant-channel ablation, no dictionary health.

`delta_seed0 = +0.005619` is recorded as measured-but-**not interpreted**: it is a
single-seed difference whose baseline had already drifted by more than its own
magnitude, and re-reading it after the fact is precisely the goalpost move the
round forbids.

## 3. Diagnosis (§10.2 explicitly requires one)

The drift is **not** a data, dictionary, initialization, evaluation or code change:

| candidate cause | evidence | verdict |
|---|---|---|
| data | `results/e2e_dictenv_p1/cache/env_{train,valid}.pt` written 2026-09-24 16:10, i.e. **before** H1's run 2026-09-24 21:34; both splits 10000/1000 | ruled out |
| dictionary | fresh reference `dictionary_sha256 = b0c5da98…dfecd` = H1's recorded sha, bit-for-bit | ruled out |
| code | `git log 122db5dc..HEAD -- e2e_dictenv_p2_abs.py e2e_dictenv_p1.py` is **empty** | ruled out |
| initialization | `p2.build_model` calls `torch.manual_seed(0)`; P2 has **no dropout**; no `_seed_everything` in either runner | ruled out |
| evaluation | this round's evaluator, run on the stored `H1_soup_state.pt`, returns **0.12354863** with `rec = 1.4615e-04` — H1's recorded soup value and rec exactly; the fresh state returns 0.12987817 = its recorded value | ruled out |
| batch order | `make_env_loader` uses an explicitly seeded `torch.Generator` | ruled out |

What remains is the **execution regime**, and it is decisive
(`code/diag_e2e_dictenv_purify_v0_reproducibility.py`, GPU1, two independent
executions, artifact `reproducibility_probe.json`):

* **repeated identical forward passes on CUDA** (same weights, same batch, eval
  mode): IHT codes bit-identical (`codes_max_abs_diff = 0.0`, 0 support flips),
  but the pooled read-outs are not — `relation_readout` / `unified` differ by
  `1.07e-04 … 1.37e-04` and the prediction by `2.4e-06 … 2.9e-06`.  On CPU the
  same comparison is **exactly 0.0** everywhere.  Cause: the shared pooling
  helpers use `index_add_`, whose CUDA float accumulation order is not
  deterministic.
* **repeated runs of the identical protocol** (same code, seed, GPU, data, two
  epochs each, epoch body copied verbatim from the formal trainer):

  | execution | epoch-1 valid MAE diff | epoch-2 valid MAE diff | epoch-1 train MAE diff | epoch-2 train MAE diff |
  |---|---:|---:|---:|---:|
  | 1 | `+0.0414` | `+0.0291` | `+0.0056` | `+0.0163` |
  | 2 | `-0.0089` | `-0.0422` | `+0.0013` | `-0.0026` |

  The fresh reference's own epoch-1 train MAE (`1.011811`) sits between the two
  probe runs (`1.011238`, `1.012584`): it is an *ordinary draw* of this protocol,
  not a differently configured run.

**Conclusion.**  The protocol is not bit-reproducible on this GPU regime, and its
two-epoch dispersion in official-valid MAE (`0.009 … 0.042`) already exceeds the
`0.005` reproduction tolerance, let alone the `0.002` non-inferiority band.
Therefore (a) the historical H1 point `0.12354862861608853` is **not a usable
decision baseline** — it carries an unquantified O(0.01) run-to-run uncertainty —
and (b) the historical-anchor sanity check of §10.2 cannot in general be passed by
any fresh run, however faithful.  This is a property of the *measurement*, not of
the purified architecture.

## 4. What is and is not claimed

* **Claimed (structural):** the four-module consolidation is functionally
  equivalent to the H1 reference on the deterministic path (bit-identical), it
  removes all three local raw-chemistry bypasses at 1.8 % *fewer* parameters,
  and it trains end-to-end under the frozen protocol (both arms complete, healthy
  mechanisms, gradient into the constant channel).
* **Not claimed:** anything about the predictive quality of the purified
  architecture.  `Delta_purification` is recorded but must not be read as
  evidence for or against non-inferiority: the round's own reference baseline
  failed its reproducibility check, and there is no seed-1 replicate.
* **Not run:** `reference_seed1.json`, `purified_seed1.json`,
  `mechanism_interventions.json`, `constant_channel_ablation.json`,
  `dictionary_health.json` (all NOT RUN by the frozen stop rule).

## 5. Consequences for the programme

1. Historical ZINC numbers in this line must be treated as **point estimates with
   O(0.01) protocol dispersion**; they are context, never gates.
2. Any future non-inferiority claim needs a **same-round paired baseline and ≥ 2
   seeds**, and its tolerance must exceed the measured protocol dispersion of the
   quantity it gates (soup-level dispersion here is of the same order as the
   epoch-level one).
3. Cheaper and much stronger than more seeds: make the *measurement* deterministic
   — `torch.use_deterministic_algorithms(True)` plus a deterministic
   `index_add_` replacement (or CPU pooling) — and re-measure the dispersion;
   only then can a `0.002` band be tested at all.
4. The purified architecture itself remains an open, *unmeasured* question; a
   fresh round may re-ask it, but it must be pre-registered and it must not reuse
   this round's `delta_seed0`.

## 6. Reproduction

```bash
# local, no GPU
uv run pytest -q tracks/ksvd/tests/test_e2e_dictenv_purify_v0.py            # 24 tests
uv run python -m tracks.ksvd.code.analyze_e2e_dictenv_purify_v0 --write     # 38 checks, 0 errors
uv run python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_purify_v0 report

# remote, GPU1 only (remote-research-runner skill launchers)
R=/home/calendar/.pi/agent/skills/remote-research-runner/scripts
bash $R/deploy.sh
bash $R/run_remote.sh 1 python -m tracks.ksvd.experiments.luyin16.zinc_e2e_dictenv_purify_v0 all --device cuda
bash $R/run_remote.sh 1 python -m tracks.ksvd.code.diag_e2e_dictenv_purify_v0_reproducibility --write
```

`run_remote.sh` exposes exactly physical GPU1 (`CUDA_VISIBLE_DEVICES=1`), and the
round's `resolve_device` additionally refuses any other device mask.
