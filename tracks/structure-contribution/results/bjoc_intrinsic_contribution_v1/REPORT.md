# bjoc-intrinsic-contribution-feasibility-v1 — report

Protocol: `cscl-bjoc-intrinsic-contribution-feasibility-v1`.  This report is assembled from the frozen JSON artifacts in this directory; official valid/test were never loaded.

## Part 1 — absolute prediction capacity

- Historical BJOC-Null official-test raw-y MAE (context only): seed-mean 0.092233, two-seed ensemble 0.085233; user target band 0.06–0.07.

- Internal split-A locked (already-exposed jdrop split, seed-mean): MAE 0.143290; ordinary (y>-3) MAE 0.092005 on 931 molecules; tail (y<=-3) MAE 0.835272 on 69 molecules carrying 40.2% of the error mass.

- The held-out floor on ordinary molecules is already ~0.09; the target band requires a broad improvement, not tail repair.  Top-10 removed locked MAE 0.090631.


## Part 2 — predictive cost of the interpretable model

| arm | fit | monitor | locked |
| --- | ---: | ---: | ---: |
| B_A_s0 | 0.037033 | 0.132840 | 0.118042 |
| S_A_s0 | 0.105686 | 0.181423 | 0.165671 |


Stage-1 monitor stop-loss: **C_too_costly** (Δ(S−B) = +0.048583, purchase seed 1 = False).  monitor cost exceeds the stop-loss; no further S training, pivot to prediction research.


## Part 3 — structural contributions

- Synthetic A identifiability: design rank 263/268, null-space dimension 5.

- Synthetic B co-occurrence trap: A and B are exactly collinear in every molecule; fitted allocation moves between seeds (A sd 0.025, B sd 0.025, mean |B| 0.139). Type-level co-occurrence cannot be resolved by the additive model; the uncertainty must be exported.

- Synthetic C additive limit: The interaction is not additive in the count basis: additive OLS floor 0.1507, interaction oracle 0.0194; 32% of the interaction variance is orthogonal to the additive design. Additive IJCM reaches 0.0923: better than the count-additive floor (local radius-2/bond context leaks some adjacency information) but still short of the oracle, and its residual is strongly anti-correlated with the true interaction count (-0.951). An explicit pairwise interaction would be needed for the residual gap; this round does not add one.


## Part 4 — stop or advance

- Final label: **ATTRIBUTION_WORKS_BUT_PREDICTIVE_COST_TOO_HIGH**.
- Rationale: synthetic identifiability works, but the held-out prediction cost of the additive head is too large


## Part 5 — the next GPU purchase

- Highest-value prediction direction: **One pre-registered pairwise junction-interaction channel inside the frozen one-shot BJOC family: bond-adjacent junction-pair bilinear products added to the joint reader, with a capacity/wall-time-matched control and a single monitor gate.**

- Why: The locked read shows the additive head is not a prediction route: locked MAE 0.165671 vs 0.118042 (delta +0.047629, CI95 [+0.034763,+0.063349]) and its fit MAE is 0.105686 vs 0.037033, i.e. it cannot even represent the training relation. Synthetic C shows why: with a true pairwise junction interaction the additive fit error (0.1507) is far above the interaction oracle (0.0194), 32% of the interaction variance is orthogonal to the additive design, and the additive residual is -0.95 correlated with the true interaction count. The 0.06-0.07 band therefore needs a composition mechanism, not a wider additive vocabulary.


**Research judgement (≤500 characters):** the current BJOC-class predictor is ~0.09 on ordinary molecules; reaching 0.06–0.07 is not a calibration problem but a missing capability: *reusable cross-molecule local-relation composition that survives distribution shift*. The next GPU purchase should test one pre-registered capacity/regularisation-controlled change that alters how multi-scale local relations are composed inside the frozen one-shot family — not another feature dump, and not more additive structure.
