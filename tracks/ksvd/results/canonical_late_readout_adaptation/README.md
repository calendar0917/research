# Canonical Late-Readout Adaptation Confirmation

**Verdict:** FULL CONTINUATION BETTER (Case C) — HEAD-ONLY FREEZING NO-GO.

Does freezing the final compact-v4-hinge graph representation and continuing to
train only the existing MAE head improve official-valid MAE, beyond a matched
full-model continuation with the same fresh optimizer budget?

| seed | B0 stop | C full continue | E head-only | ΔA=B0−E | ΔC=B0−C | ΔF=C−E |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.170066 | 0.160976 | 0.173042 | **−0.002977** | **+0.009090** | **−0.012067** |

`ΔA < +0.003` → head-only adaptation does not advance; `MAE(C) < MAE(E) − 0.0015`
→ full continuation is clearly better. Seed 1 was **not** run (clear negative),
so the OOF late-refit signal did **not** transfer to the leak-free canonical
protocol.

## Protocol (frozen in `adaptation_protocol_lock.json`)

- Starts from the canonical compact-v4-hinge **selected validation checkpoint**
  (seed 0 run `20260909-194445-182c7021`); no baseline retrained.
- official **train** only for updates; official **valid** only for a single
  post-budget evaluation; official **test never loaded**.
- Fixed adaptation horizon `K* = 94` epochs = `round_half_up(median(L-warm OOF
  best-selection epochs))`, chosen from the completed OOF audit before any
  canonical-valid number was seen.
- Fixed standardisation `z = (R − μ)/max(σ, 1e-6)` fit on official-train `R`
  once; exact first-layer reparameterisation makes step 0 equivalent to B0.
- Adam(lr=1e-3, wd=0), batch 512, deterministic mini-batch order, global torch
  seed 0, L1 loss, same budget/reset for C and E.

## Files

| file | content |
|---|---|
| `checkpoint_inventory.json` | canonical selected checkpoints + SHAs |
| `adaptation_protocol_lock.json` | preregistration (K*, optimiser, thresholds) |
| `standardization_stats.json` | train-only standardisation statistics |
| `starting_equivalence.json` | B0/C/E step-0 function equivalence |
| `trainable_parameter_audit.json` | C and E trainable parameter sets |
| `seed0_results.json` | B0/C/E metrics, curves, diagnostics |
| `seed0_training_curves.csv` | train-only adaptation curves (C, E) |
| `seed0_representation_drift.json` | C drift / E frozen check |
| `pooled_seed_results.csv` | primary results table |
| `final_decision.json` | preregistered decision mapping |
| `figures/` | valid MAE, train curves, C drift (≤ 3 figures) |

## Reproduce

```bash
uv run python -m tracks.ksvd.experiments.luyin16.zinc_canonical_late_readout_adaptation all
uv run pytest tracks/ksvd/tests/test_canonical_late_readout_adaptation.py
```

Runs must be **serial** (`torch_threads=4`), per
`notes/reproducibility_cpu_determinism.md`; the two branches are bit-reproducible
across independent serial reruns.

See `tracks/ksvd/notes/canonical_late_readout_adaptation.md` for the full note.
