# ZINC strict-static dictionary-pair v0 — results summary

Question: on a modern, strongly-trained **strict-static** patch-pair baseline, is a residual learnable dictionary more valuable than a parameter-matched generic dense residual adapter?

Official ZINC **test was never loaded** in any stage.

## Primary metric — best official-valid MAE

| arm | best valid MAE | best epoch | epochs | soup MAE | params | branch params |
|---|---|---|---|---|---|---|
| S0 | 0.145674 | 164 | 204 | 0.140794 | 66228 | 0 |
| S-Dense | 0.145958 | 170 | 210 | 0.142660 | 71417 | 5189 |
| S-Dict | 0.136050 | 225 | 240 | 0.133829 | 71429 | 5201 |

## Pre-registered gains and case

* `M0 - MK` (gain_vs_base_dict) = +0.009625
* `M0 - MD` (gain_vs_base_dense) = -0.000284
* `MD - MK` (dict_specific_gain) = +0.009908
* primary case: **A_DICT_SPECIFIC_SIGNAL**
* soup case (corroborating): **A_DICT_SPECIFIC_SIGNAL** (M0 0.140794, MD 0.142660, MK 0.133829)

## Control integrity

* strict-static contract passed for every arm: True
* shared tensors bit-identical (S0 vs Dense vs Dict): True (n=40)
* Dense/Dict branch parameter mismatch: 0.23%

## Dictionary diagnostics (report only; never used for selection)

* mean_assignment_entropy: 2.0950422286987305
* effective_prototype_count: 8.125783920288086
* active_prototype_count: 64
* used_prototype_count_argmax: 12
* top8_assignment_mass: 0.8447091579437256
* max_average_assignment_mass: 0.34823086857795715
* dictionary_coherence_mean_abs: 0.45274215936660767
* dictionary_coherence_max_abs: 1.0000001192092896
* mean_residual_norm: 0.06920712441205978
* mean_h0_norm: 1.885798692703247
* tau_final: 0.21008212864398956

## Inference ablation — dictionary residual forced to zero

* mean |prediction shift|: 0.30436395554244516
* max |prediction shift|: 0.9831738471984863
* fraction of predictions shifted > 1e-6: 1.0

## Budget

* full training runs: 3 (S0 seed0, S-Dense seed0, S-Dict seed0)
* total wall clock: 47.5 min
* seed1/2/3: not purchased; HPO: none; dictionary-pair kernel: not implemented
* official test accessed = false

