# Quarantined Monitor execution (invalid normalization)

This directory preserves the **first** Monitor execution. It is invalid and
was not used for any H1–H4 verdict.

- Prediction manifest: `ec043a7fe92ba479d7cc8c8d109df37253bec589b1b7cab5cdfa47e988b8f817`
- `v2_predict.predict_nn` returned the model's standardized target output
  without applying the frozen Fit2352 target normalization
  (mean 2.157857, scale 1.224198).
- Consequence: NN predictions were centred near 0 while Monitor labels have
  mean 2.230913. NN RMSE was ~2.30–2.39, R² ~−3; the CPU baseline (which was
  already on the raw scale) had RMSE 0.778777.
- The error is a prediction-code bug, not a model result. The bug was fixed by
  `pred = denormalise(predict_view(...), target_norm)` with a fail-closed
  mean check.
- Archived: the old prediction directory/manifest, `monitor_metrics_v2.json`,
  `decision_v2.json`, `tables_v2.json`.
- Do not cite these numbers as Monitor performance.
