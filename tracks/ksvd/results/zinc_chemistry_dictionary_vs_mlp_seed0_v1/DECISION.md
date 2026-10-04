# DECISION — zinc-chemistry-dictionary-vs-mlp-seed0-v1

## Final class

**Dictionary gate: `INCONCLUSIVE`.**  **Target-interference label:
`RELIEF_UNCONFIRMED`.**  The round is complete: the comparison exists, is
verifiable and has a clear interpretation boundary.  No third arm, no second
seed, no rescue configuration, no official split.

## Direct answers to the six closing questions

**1. How were the `g/c` labels formed, was current-dev fitting used, and how
large is the correction versus the old definition?**

`y = g + c` exactly.  `c = (k - mu_cycle)/sigma_cycle` with the snapped
effective cycle `k` obtained from the label-side formula
`eff_norm = y - [(logP - mu_logP)/sigma_logP + (SA - mu_SA)/sigma_SA]`.
All constants were **refit on the 8000 fit rows only** (OLS on the cycle-free
fit bulk + Nelder-Mead snap on fit rows) and frozen in `fit_only_targets.npz`;
the old round had fitted them on the whole 12000-row audit table, which
includes the current dev rows.  No current-dev label enters any fit, input
standardizer, sampler or model selection.  The refit changes the constants by
at most `0.00124`, changes **no `k` label** (0/10000 rows) and changes `c` (hence
`g`) by at most `0.033654`; on G0 `c0` moves from `0.0004627` to `0.0005791`.

**2. After removing ring supervision, who is better on G0 / overall `g`, and
does the evidence pass the threshold?**

Neither claim passes.  G0 cal `D_g 0.103614` vs `M_g 0.101875`:
`G_g = -0.001739`, CI `[-0.005604, +0.002716]`, below the `0.003` threshold
and crossing zero; the raw G0 gain is `+0.002262` in the other direction.
Overall cal `D_g 0.105486` vs `M_g 0.103717`: gain `-0.001769`, CI
`[-0.005804, +0.002341]`; overall raw is `+0.002122` for `D_g`.  No gate
category fires (`INCONCLUSIVE`), and a CI crossing zero is not equivalence.

**3. What are the `D_y -> D_g` / `M_y -> M_g` subject changes and the
interaction, raw and calibrated?**

`B_D` cal `+0.000686` (CI `[-0.003722, +0.004978]`), raw `+0.000461`;
`B_M` cal `+0.001430` (CI `[-0.003186, +0.005689]`), raw `-0.000776`;
`I` cal `-0.000744` (CI `[-0.006229, +0.005203]`), raw `+0.001237`.
All points are below `0.003`, all CIs cross zero, and `B_M`/`I` are not
raw/cal consistent.  The per-row identity `I = B_D - B_M = G_g - G_y` holds
exactly in every bootstrap sample.  Relief is `RELIEF_UNCONFIRMED`.

**4. If fit improves while dev does not, which facts support a
fit/generalisation difference and what remains unattributable?**

Facts supporting a fit-side difference: `M_g` fits overall `g` better in
sample (`0.029280` vs `0.033291`; fit gain `-0.004011`, CI separated) and wins
the frequent groups (G0, `k=-1`) on dev; `D_g` is relatively better on the rare
groups (`k=-2`, `k<=-3`) and the previous y round's extreme-tail failure of `D`
vanished under `g` supervision.  Fit->dev gaps are large for both
(`~3.2-3.5x`).  Unattributable with one seed, one fold and 2/10 rare dev rows:
capacity vs optimisation vs rare-target coverage vs a shared-representation
limit; the soup-vs-last anchor difference shows the result is also sensitive to
which fixed soup is averaged.

**5. What is actually supported about the current task dictionary, and which
luyin19 claims remain unfulfilled?**

Supported: the fused task-dictionary bridge is statistically competitive with
the matched MLP under `g` supervision and is not dominated; its previous tail
weakness was coupled to the ring target rather than being a pure capacity
verdict; the bridge is a **dense** post-fusion code (~92% non-zero atoms) and
not a sparse structural dictionary.  Unfulfilled: luyin19's claim that a shared
dictionary learns a transferable **structure-attribute relation** is not
tested by this compressed skeleton (no structural `D`, no `U`/IHT, no slot
binding), and the structure-semantic fusion question is untouched.  A
dictionary win here would not have proven luyin19; its absence does not refute
all dictionaries.

**6. The single next design (design only, never executed in this round).**

**No-coding reference + pre-bridge information probe, fit-only, on this fold.**
Hold the body frozen at the `g` soup and compare, under the identical recipe,
(a) the current `D_g` bridge, (b) the matched `M_g`, (c) a no-coding linear
`144 -> 144` reference, and (d) a fit-only readout from the pre-bridge fused
interface (and from the raw `Sem108+size2` 110-d input).  It excludes the
explanation "the difference between the dictionary and the MLP is about how
each codes `g`": if the no-coding reference or the pre-bridge readout already
matches both, the coding bridge is not where the remaining frequent-chemistry
error is created, and the next investment must be the shared local environment
representation and the static relational information — not another bridge
tweak and not another five-tail fit.  No automatic start; it needs its own
pre-registration and budget.

## Stop statement

All this-round compute is stopped: the two formal Slurm jobs completed with
exit 0, no job is running or queued, and no official-valid/test split was
loaded.  No 10k confirmation stage starts from this result.  The branch
`task/zinc-chemistry-dictionary-vs-mlp-seed0-v1` is left isolated (no push, no
merge); old result directories are byte-preserved.

## Provenance

* training revision `7c0224a` (`7c0224a37141` on res-2), branch
  `task/zinc-chemistry-dictionary-vs-mlp-seed0-v1`, commit `8bf8b4e` holds the
  pre-registered protocol and frozen targets;
* `D_g` = `zcdm-d` (Slurm 55897), `M_g` = `zcdm-m` (Slurm 55898), both on c05
  (A100-PCIE-40GB, driver 525.85.12, torch 2.5.1+cu124);
* target npz hash `e2adf5f2…`; prediction alignment uses the frozen stable-ID
  order; `G_y` reproduced the previous round exactly (`-0.0009945985`).
