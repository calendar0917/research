# Decision - E2E-DictEnv-RoleCorr-Increment-v2 (route 1)

- verdict: **INCREMENT_NO_MATERIAL_GAIN**
- `M_A = 0.126393070`, `M_B = 0.125661825`, `M_C = 0.123292468`

The frozen absolute gate did not fire: appending the correspondence coordinate on top of the full structural budget does not improve official-valid Top-5 soup MAE by >= 0.003 over the matched extra-structural control. Per the pre-registration route 1 stops here; no K/s/LR/horizon rescue and no seed purchase. Route 1 answers the increment question negatively under this budget.

## Forbidden without a new pre-registration

- seed 1 of any arm;
- any K / sparsity / K-SVD-epoch / scaler / PCA-rank / width / horizon change;
- end-to-end (unfrozen-dictionary) fine-tuning of any arm;
- touching the official ZINC test split;
- re-using these numbers as a replacement for the historical Sem108 result.
