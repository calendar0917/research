# Typed-cycle static object into the shared task dictionary (ZINC, CPU)

- Round `e2e_dictenv_typed_cycle_v1`; seed 0; 320 epochs; official test never loaded.
- Verdict: `TYPED_CYCLE_STOP`; soup valid MAE `0.120779`.
- best valid `0.127393` @ epoch 269; soup members `[269, 273, 305, 306, 318]`.
- same-protocol soup train MAE `0.054857`; train->valid gap `+0.065922`.
- parameters `133002`; ring encoder `24816`; reader increment `1261`.
- wall `2962.0s`; `9.256s/epoch`; peak RSS `2418 MB`.

## Absolute band
- band boundaries: `<= 0.115` promising / `0.115-0.120` limited / `> 0.120` stop; observed `0.120779` -> `TYPED_CYCLE_STOP`.
- unmatched backgrounds (different params / single seed): Small latent-bridge soup `0.121058`, Full `0.119154`; these are context, not a matched comparison.

## Inference-only diagnostics
- zero_ring: {"delta_mae": 0.2793113961815834, "delta_pred_rms": 0.6662854552268982, "mae": 0.4000905454158783}
- cross-molecule same-length ring permutation: [{"base_mae": 0.12077914923429489, "delta_mae": 0.21575994044542313, "permuted_mae": 0.336539089679718, "prediction_rms": 0.3854864835739136, "seed": 11, "swapped_fraction": 0.6184493898061737, "swapped_objects_per_side": 1723, "total_objects": 2786}, {"base_mae": 0.12077914923429489, "delta_mae": 0.2080044522881508, "permuted_mae": 0.3287836015224457, "prediction_rms": 0.37268733978271484, "seed": 22, "swapped_fraction": 0.6184493898061737, "swapped_objects_per_side": 1723, "total_objects": 2786}]
- ring code density: {"fraction": 0.9497974537037037, "mean_nonzero": 91.18055555555556, "n_objects": 360, "p50": 91.0, "p95": 94.0}
- node code density: {"fraction": 0.9163887035801644, "mean_nonzero": 87.97331554369579, "n_objects": 2998, "p50": 88.0, "p95": 93.0}
- ring encoder task gradient `0.3386647514998913`; shared bridge task gradient `{"D_L": 0.24007552862167358, "V_L": 0.24419596791267395}`.

These interventions establish dependence, not a training gain, and the absolute band is a resource decision, not a significance test.
