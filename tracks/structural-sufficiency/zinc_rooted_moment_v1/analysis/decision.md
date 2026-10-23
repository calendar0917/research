# Frozen decision (RMC)

```json
{
  "arms": [
    "G",
    "M",
    "P"
  ],
  "seeds": [
    0,
    1
  ],
  "endpoint": "T100 (epochs 501..600)",
  "T100_gain_vs_P": {
    "M": {
      "0": -0.008007745529990645,
      "1": -0.007355897735571479
    },
    "G": {
      "0": -0.008633595059800436,
      "1": -0.006296848495607257
    },
    "M_minus_G": {
      "0": 0.000625849529809791,
      "1": -0.0010590492399642215
    }
  },
  "mean_T100_gain_vs_P": {
    "M": -0.007681821632781062,
    "G": -0.007465221777703847
  },
  "both_seeds_T100_M_better_P": false,
  "both_seeds_T100_G_better_P": false,
  "both_seeds_T100_M_better_G": false,
  "ordinary_best_M_minus_P": {
    "0": 0.007618621615848517,
    "1": 0.006096483994459348
  },
  "ordinary_final_M_minus_P": {
    "0": 0.006187136651296929,
    "1": 0.0061851773678153615
  },
  "unseen_ordinary_final_M_minus_P": {
    "0": 0.017554021752229365,
    "1": 0.016471615254119942
  },
  "h1_supported": false,
  "h2_supported": false,
  "h1_rule": "both seeds T100(M) < T100(P) and seed-mean gain >= 0.003",
  "h2_rule": "both seeds T100(M) < T100(G) and seed-mean gain >= 0.0",
  "subgroup_reading": {
    "ordinary_best_and_final_both_better": false,
    "unseen_ordinary_final_both_better": false
  },
  "verdict": "NO_STABLE_GAIN_OVER_P",
  "u_trigger": {
    "candidate_arm": "G",
    "mean_T100_gain_vs_P": -0.007465221777703847,
    "both_seeds_T100_positive": false,
    "ordinary_best_seed_mean_gain": -0.005911983386401497,
    "ordinary_final_seed_mean_gain": -0.004688331907310578,
    "unseen_ordinary_final_per_seed_gain": {
      "0": -0.013848133263115375,
      "1": -0.013865480996695878
    },
    "unseen_ordinary_final_seed_mean_gain": -0.013856807129905627,
    "rule": "both seeds T100>0, mean>=0.003, ordinary best+final seed-mean>0, unseen+ordinary final both seeds not worse and mean>=0.005, budget allows 2x600 epochs U",
    "c1": false,
    "c2": false,
    "c3": false,
    "group_rule_satisfied": false
  }
}
```
