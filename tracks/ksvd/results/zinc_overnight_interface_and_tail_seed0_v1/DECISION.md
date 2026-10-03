# DECISION — zinc-overnight-interface-and-tail-seed0-v1

Status: **closed, negative for the joint-correspondence interface**.
No Phase 3, no official-valid read, no write-back.  This file is the frozen
decision record; `final_decision.json` carries the machine-readable version.

## Answers to the five closing questions

**1. On top of the compressed `B` (G0 endpoint), does the new interface buy
generalisation?  Where does the gain come from?**

On the internal dev, yes — small.  `R_DM` (dense code, marginal statistics)
gets dev G0 cal `0.092664` vs `0.097347` for `B` (`+0.004683`, CI
`[+0.003269, +0.005923]`) and dev overall cal `+0.004395`
(`[+0.002915, +0.005798]`).  But the decomposition says the gain is *not* the
joint correspondence the hypothesis is about, and it is also not primarily the
structure x category statistics:

* a Sem-only adapter (`R_CS`) already delivers `+0.002273` overall /
  `+0.002630` G0;
* only dense-marginal adds a CI-positive structural increment over that
  (`R_DM - R_CS = +0.002122` overall, `+0.002053` G0);
* joint vs marginal is null in the sparse path (`+0.000365`, CI crosses 0) and
  **negative** in the dense path (`-0.001825`, CI excludes 0);
* trained joint arms are nearly invariant to destroying the pairing at
  inference (J->M flip: `R_SJ +0.000692`, `R_DJ +0.000330` dev cal, vs
  marginal-trained arms `R_SM +0.006720`, `R_DM +0.008989` for M->J);
* once the body is unfrozen (matched `A0`), the interface adds nothing
  measurable: `T +0.000205` overall cal, `-0.000170` G0 (both CIs cross 0),
  while unfreezing alone gives `A0 - B = +0.006011`.

The frozen-body gain is therefore a mix of (a) adapter capacity on the frozen
body (largest single component), (b) a small dense-marginal readout effect,
(c) essentially no joint-pairing effect, and (d) a frozen-body limitation:
the same interface loses all measurable value once the body can adapt.

**2. Extreme ring tail: can the severity-balanced `q` head pull rare-ring
targets (`q_U` vs `q_B`)?**

No.  `q_U` reproduces the released `P_seed0` exactly (replay diff `0.0`) and
keeps dev cal `0.113180`; `q_B` (max weight 400, ESS 68.29) drops to
`0.152442` (`-0.039262`, CI `[-0.074767, -0.010800]`).  It fits the 5 fit rows
at `k<=-3` (`14.24 -> 0.097`) but the 2 dev rows worsen (`7.25 -> 21.01`), and
`k=0` worsens (`0.1003 -> 0.1307`).  Gate failed on overall cal, overall raw
and G0 worsening.  This head-only weighting scheme is closed.

**3. Write back to the main line?  Next experiment?**

No write-back; the branch is isolated and unfused.  If the question is
reopened, the only candidate worth a fresh independent split is `R_DM` vs `B`
(with `R_CS` as the capacity control), and the J-vs-M contrast should be
pre-registered as expected-null.  Do not buy more arms on this reused dev, do
not reweight the tail head on the same 5 rows, do not tune `H1_LAMBDA`,
`kappa` or width on dev, and do not open official-valid to rescue the gate.

**4. What is truly reproducible vs exploratory?**

Reproducible from committed artifacts: anchor/identity checks (all `<= 4.8e-6`,
CPU heads exact), the main/gain/group tables and CIs (`analyze.py --stage 1/2`,
seed 20261003), the J/M switch, and the conflict analysis.  Exploratory:
every performance delta — one seed, reused internal dev, `T` chosen by
Phase-1 dev performance without selection correction; the `~0.003`-scale
effects are at the fold's noise scale.  No official-valid confirmation exists
in this round (not performed by design).

**5. Budget and data boundaries?**

Kept: seed 0 only; 8000/2000 fixed folds; 8 formal A100 trajectories
(5 Phase 1 + 3 Phase 2), max 2 concurrent GPUs, total `1.593` GPU-hours
(<= 6); all training started well before the 420-minute cutoff and stopped;
official-test never touched; official-valid read `0` times (the single gated
read was not performed because no candidate passed); local CPU <= 8 threads
(CPU runner/replay 4); all self-created Slurm jobs completed — none left
running.  Failed/aborted attempts (5 pre-training submission failures, one
CPU scheduling bug) are listed in `REPORT.md` section 6 and consumed no
trained steps.

## Decision rules applied (pre-registered)

* Phase-1 `T`: among `R_SM/R_SJ/R_DM/R_DJ`, best dev G0 cal if better than
  `B`, else `R_SJ` -> `R_DM` (dense, marginal), frozen before Phase 2.
* Phase-2 gate (vs matched `A0`): G0 cal gain `>= 0.003` with CI lower `> 0`,
  overall cal gain `>= 0.003`, overall raw gain `> 0` -> `C` and `T` fail.
* CPU gate (`P_B` vs `P_U`): overall cal `>= 0.003`, raw `> 0`, G0 worsening
  `<= 0.001` -> `P_B` fails on all clauses.
* Therefore: no Phase 3, no official-valid read, report negative.
