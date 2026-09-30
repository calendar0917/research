# Decision - E2E-DictEnv-RoleCorr-v1

- verdict: **ROLE_CORR_NO_MATERIAL_GAIN**
- `M_A (TOPO) = 0.123253705`, `M_B (CORR) = 0.126447222`, `relative = -2.5910%`

The 2% relative screening gate did not fire: the role/attribute correspondence dictionary coordinate gives no material task gain over the frozen pure-topology coordinate under the identical frozen-dictionary protocol. Per the pre-registration the round stops here; controls C/D are not trained. No width / horizon / seed / regularisation rescue is authorised; a future step needs a new pre-registration. This does not falsify every correspondence object, and does not replace the historical end-to-end Sem108 number.

## Forbidden without a new pre-registration

- seed 1 of either arm;
- any K / sparsity / K-SVD-epoch / scaler / PCA-rank / width / horizon change;
- end-to-end (unfrozen-dictionary) fine-tuning of either arm;
- touching the official ZINC test split;
- re-using these numbers as a replacement for the historical Sem108 end-to-end result.
