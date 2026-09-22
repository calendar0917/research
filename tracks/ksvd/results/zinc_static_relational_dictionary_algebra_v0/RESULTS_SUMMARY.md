# SRDA-v0 — static relational dictionary algebra: results summary

Question: with the typed-token (16D) and parent (8D) identity channels deleted, does making the end-to-end dictionary assignment *define* the occurrence-level relation algebra open a new absolute strict-static ZINC MAE band?

Official ZINC **test was never loaded** in any stage.

## Primary metric — fixed Top-5 weight soup (official-valid MAE)

| run | best MAE | best epoch | epochs | soup MAE | params | wall s |
|---|---|---|---|---|---|---|
| S0 seed0 (reference) | 0.145674 | - | - | 0.140794 | 66228 (typed+parent present) | (historical) |
| SDPK-v0 seed0 (reference) | 0.142193 | 209 | 240 | 0.139735 | 74996 | 888.1 |
| **SRDA-v0 seed0** | 0.155449 | 196 | 240 | 0.149873 | 49970 | 909.3 |

## Seed-0 performance gate (both must hold)

* Top-5 soup <= 0.134: measured 0.149873
* best valid <= 0.1385: measured 0.155449
* improvement vs S0 seed0: best -0.009775, soup -0.009079
* improvement vs SDPK-v0 seed0: best -0.013255, soup -0.010138
* verdict: **SRDA_V0_NO_STRONG_PERFORMANCE_SIGNAL**

## Runtime widths (attested)

* local_input_width: 146
* local_state_width: 64
* dictionary_assignment_width: 64
* prototype_factor_width: 32
* residual_input_width: 32
* residual_width: 8
* pair_input_width: 72
* relation_input_width: 23
* unary_width: 129
* pair_pool_width: 325
* global_width: 32
* topology_width: 8
* graph_width: 494
* identity-channel params: 0 (only embedding: True)

## Dictionary / decomposition diagnostics (report only, never a gate)

* mean_assignment_entropy: 2.672219753265381
* effective_atom_count: 14.472058296203613
* active_atom_count: 64
* argmax_used_atoms: 51
* top8_assignment_mass: 0.7497097849845886
* dictionary_coherence_mean_abs: 0.171961709856987
* dictionary_coherence_max_abs: 0.7810962200164795
* tau_final: 0.17179708182811737
* mean_residual_norm: 0.5797919631004333
* mean_norm_p_dict: 2.2554714679718018
* mean_norm_p_eps: 0.6593062281608582
* mean_norm_rel_feat: 0.8348535299301147
* pair encoder first-layer block weight norms: {'p_dict': 7.040309429168701, 'p_eps': 4.532189846038818, 'rel_feat': 3.1178300380706787}

## Cheap inference interventions

| intervention | mean |dpred| | max |dpred| | valid MAE after |
|---|---|---|---|
| neutral_dict | 1.2017923780977726 | 2.8373329639434814 | 1.2319464431191445 |
| zero_eps | 0.2462456296160817 | 0.5006635189056396 | 0.28605216084694257 |
| mean_alpha | 0.6124557076841592 | 2.5659177005290985 | 0.6436459120092332 |

## Budget

* full training runs used: 1 / 2
* seed 1 purchased: False
* matched dense/raw control: none (not purchased this round)
* HPO / sweep: none
* official test accessed = false

