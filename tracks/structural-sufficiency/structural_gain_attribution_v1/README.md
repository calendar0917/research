# structural_gain_attribution_v1

Progressive study on the predecessor round `molhiv_multigranular_structure_sharing_v0`
(MSDS-v0): **where does the explicit-subgraph prediction gain come from**, and can a
different readout (MLP) use the same structure information?

* Frozen protocol: [PROTOCOL.md](PROTOCOL.md) (+ `configs/protocol.json`)
* Results and the belief-update table: [REPORT.md](REPORT.md)
* Display-layer correction of MSDS-v0: [ERRATA.md](ERRATA.md)
* Provenance (commit / data identity / environment / resources): `analysis/provenance.json`

## Phases

| phase | what | trains |
|---|---|---|
| 0.1 | corrected OGB attribute decoding (display only) + ERRATA | nothing |
| 0.2 | three-way "out of vocabulary" statistics from the frozen matrices | nothing |
| A | six new XGBoost attribution arms (`X-MC`, `X-MU`, `X-WL`, `X-C`, `X-F`, `X-PRES`) | 6 XGB |
| B | MLP readout on the identical inputs (`N0`, `N-WL`, `N-CF`) | 3 MLP |
| C | minimal ZINC transfer (`ZX0`, `ZX1`, `ZN0`, `ZN1`) | 2 XGB + 2 MLP |
| D | structure / molecule / case analysis of the frozen predictions | nothing |

`X0` (historical XGBoost) and `X-CF` (MSDS-v0 `XGB_STRUCT`) are **reused**, never refitted.

## Run

```bash
cd tracks/structural-sufficiency/structural_gain_attribution_v1
PYTHONPATH=code uv run python -m build_inputs          # bundle/sav1_inputs.npz (committed)
PYTHONPATH=code uv run python -m run_p0_chem_errata    # ERRATA + registry_v1
PYTHONPATH=code uv run python -m run_p0_oov            # three-way OOV statistics
# remote (rr) — committed revision only
rr run res sga1-pA-X-C  --pool res-local --cpus 16 --mem 64G --result <round>/runs -- \
   python -m run_phaseA_xgb --arm X-C
rr run res sga1-pB-N0   --pool res-gpu1  --gpus 1 --result <round>/runs -- \
   python -m run_phaseB_mlp --arm N0
rr run res sga1-pC-prep --pool res-local --cpus 16 --mem 32G --result <round>/data -- \
   python -m run_phaseC_zinc_prep
# after `rr pull`, copy the JSON/npz back into runs/ and data/, then:
PYTHONPATH=code uv run python -m run_analysis_abc   # metrics, contrasts, bootstraps, curves
PYTHONPATH=code uv run python -m run_phaseD         # structure classes, subgroups, cases, figures
PYTHONPATH=code uv run python -m record_provenance  # analysis/provenance.json
PYTHONPATH=code uv run python -m write_configs      # configs/protocol.json
PYTHONPATH=code uv run pytest -q tests              # 18 CPU invariants
```

## Deliverables

| path | what |
|---|---|
| `PROTOCOL.md` / `configs/protocol.json` | frozen protocol (narrative + machine-readable) |
| `REPORT.md` | six answers, belief-update table, per-phase numbers and boundaries |
| `ERRATA.md` + `analysis/registry_v1/` | display-layer correction of MSDS-v0 |
| `analysis/predictions/*.npz` | per-molecule predictions, every arm, every split |
| `analysis/curves/` | per-epoch validation curves (Phase B / Phase C MLPs) |
| `analysis/*.json`, `analysis/*.csv` | metrics, contrasts, bootstraps, OOV, Phase D |
| `analysis/figures/` | `d1_family_gain.png`, `d2_coverage_gain.png` |
| `analysis/provenance.json` | commit, data identity, environment, jobs, GPU hours |
| `bundle/sav1_inputs.npz` | label-free input bundle (all MolHIV design blocks) |
| `data/zinc_design.npz` | label-free ZINC design (bit-identical local/remote build) |
| `tests/test_sga_core.py` | 18 CPU invariants (decoding, PRES, MC↔binary, WL, MLP init, bootstrap, OOV, ZINC) |

`runs/` (booster dumps, checkpoints, raw remote payloads) stays local by repository
convention; everything needed to re-derive a reported number is under `analysis/`.
