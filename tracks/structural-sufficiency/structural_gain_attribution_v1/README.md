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
uv run pytest -q tracks/structural-sufficiency/structural_gain_attribution_v1/tests
```
