# TCCD-v2 — Task-Learned Prototype Vocabulary

Pre-registration: `notes/tccd_v2_preregistration.md`.
Formal result files: `results/tccd_v2/gateA_seed0.json`,
`results/tccd_v2/vocabulary_seed0.json`, `results/tccd_v2/absolute_seed0.json`.

## Final verdict

TCCD-v2 establishes a healthy shared task-learned prototype vocabulary under
the matched `714 -> 64` local encoder. Prototype-REL beats Prototype-BAG and
its assignment-shuffle control, and it is stronger than the GPU1 matched
Dense-REL on the internal best-checkpoint metric. Usage remains broad: 64/64
active, effective count 62.58, top-8 mass 0.153, and normalized global usage
entropy 0.997.

The first full-data attempt was invalidated because the runner accidentally
used official-valid indices against the training-record list. After a targeted
code fix, the corrected GPU1 train-to-official-valid run reaches best valid MAE
`0.287337` / Top-5 soup `0.261988`, versus the canonical GPU1 baseline
`0.119818`. The preregistered best-checkpoint absolute gap is `0.167519`, so
TCCD-v2 is **NOT VIABLE** on the absolute gate. Official test was never opened.

## Execution provenance

* preregistration commit: `6773866`;
* internal Gate A / vocabulary commit: `69a985a`;
* invalidated first full-data attempt: `8ad255c` — excluded because of the
  validation-index bug;
* corrected full-data commit: `030b3c5`;
* GPU: A100 GPU1 only;
* seed: 0;
* split: internal `8000/2000`, then official train `10000` / valid `1000`;
* K-SVD refit: NO;
* OMP/IHT: NO;
* raw reconstruction loss: NO;
* official test: NO.

## Decision

Stop this prototype/dictionary bottleneck round. The supported claim is
mechanistic and representational: task-learned shared local prototypes plus
explicit composition can preserve assignment-sensitive predictive information
without vocabulary collapse. The unsupported claim is absolute competitiveness
with the strong ZINC baseline under the simple local encoder class.

Any future continuation requires a new preregistration for a genuinely
stronger permutation-invariant local encoder. No seed-1 rescue, temperature/K
sweep, reader expansion, relation redesign, or official-test access is
authorized by this round.
