# AMENDMENTS — cscl-bjoc-sac-v1

Every entry is written *before* the corresponding formal GPU trajectory and
*before* any Monitor/Locked label read of this round.

## Amendment 1 — 2026-10-17T09:05Z (post-freeze, pre-formal-run)

**Trigger.** The post-freeze smoke on `res-gpu0` failed after the 20-update SSL
probe with `TypeError: 'float' object is not iterable` in `cmd_smoke`:
`projected = dict(out["ssl"]["projected_full_seconds"])` attempted to build a
dict from a scalar instead of seeding the projection dict with
`{"SSL": <seconds>}`.

**Change.** One line in `run_sac_v1.py`:

```python
- projected = dict(out["ssl"]["projected_full_seconds"])
+ projected = {"SSL": float(out["ssl"]["projected_full_seconds"])}
```

and the `freeze` stage gained `--amendment N --amendment-reason TEXT` so a
re-freeze can be recorded instead of silently overwritten.

**No protocol semantics changed.** The bug is confined to the *reporting* of the
measured-throughput budget projection in the smoke stage; it fired after the SSL
probe and **before** any arm was trained, so no supervised trajectory, soup, or
prediction exists from the failed smoke. Arms, relations, projections, reader,
initialisation, recipe, seeds, GPU assignment, thresholds, decision rule and
stop rules are unchanged. `sac_v1.py` (the model/relations/trainer) is
byte-identical to the original freeze.

**Cost already incurred.** The failed smoke consumed one 20-update SSL probe
(~21 s wall on one A100). It is not entered in the GPU ledger because the ledger
counts only SSL/train trajectories, but it is disclosed here and in
`DEVIATIONS.md`.

**Re-freeze.** `protocol_freeze.json` was regenerated with `amendment: 1` after
re-running the 12 CPU gates on the amended code (`cpu_gates.json` unchanged in
verdict, 12/12). All other frozen inputs (split S, objects/relations manifests,
relation audit, input audit) are unchanged.
