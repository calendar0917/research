# E2E-DictEnv-v0 — Amendment A1 (user-authorized Stage-1 gate waiver)

Date: 2026-09-24
Applies to: `notes/e2e_dictenv_v0_preregistration.md` §16 (Stage-1 mechanism
smoke) and the frozen verdict table §21 case S1.

## Change

The Stage-1 train-only mechanism smoke failed exactly one sub-gate:

```text
atoms_active >= 24/32 :  measured 23/32  (official train and 512-subset,
                                          after the 3-epoch smoke; 25/32 and
                                          24/32 respectively at initialisation)
```

The user has **explicitly authorized proceeding with this sub-gate treated as
passed**, because the failure is a coverage/sample-size artefact and the
mechanism is demonstrably not collapsed:

* exact top-8 support preserved (`max l0 = 8`);
* train MAE falls 1.34700 → 1.24918 and reconstruction falls 0.00844 → 0.00532;
* task gradients reach `D` (norm > 0);
* no single atom dominates (top-1 share 0.125);
* effective atom count 12.83 (≥ 8);
* environment effective rank 2.30 (> 1);
* 25/32 atoms active on official train (24/32 on the 512 subset) at
  initialisation.

## What is unchanged

* All Gate-0 correctness gates G0..G12 remain binding (all PASS).
* All other Stage-1 sub-gates remain binding (all PASS).
* The architecture, K=32 / s=8 / 10 IHT steps, frozen SDB K-SVD
  initialisation, parameter budget, λ_rec, optimizer, schedule, seed, data
  splits, absence of official-test access.
* Every formal threshold and the frozen verdict table §21 cases A–E and §19/§20
  bands are unchanged.
* No forbidden rescue (K/s/IHT/LISTA/dictionary-count/decoder/attention/
  LayerNorm/λ-sweep/extra-epochs/seed) is introduced.

## Consequence

The formal arms (SparseDictEnv GPU0, DenseTiedEnv GPU1), the zero-code and
assignment-shuffle interventions, dictionary health and the full analysis are
executed. The Stage-1 `atoms_active` value (23/32) is reported verbatim in every
downstream artifact; nothing is hidden. The frozen verdict classifier's S1
branch is bypassed only through this explicit, recorded user authorization, and
the `smoke_gate.json` retains the original `atoms_active: false` sub-gate so the
provenance is auditable.
