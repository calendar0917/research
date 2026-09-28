# REPORT — e2e_dictenv_clean_mechanism_v1

CPU-only, multi-seed mechanism round on the ZINC dictionary-environment line.
Official test never loaded. No GPU, no SSH, no architecture/hyperparameter search.
Preregistration: `notes/e2e_dictenv_clean_mechanism_v1_preregistration.md`.
Full numbers: `analysis_tables.md`, `summary.json`, `notes/e2e_dictenv_clean_mechanism_v1_analysis.md`.

## A. Bottom line

1. **C6 is adopted as the clean base** (Q1). Matched-pair deltas C6−BASE are negative on
   3/3 seeds (−0.014799, −0.000116, −0.009229; mean −0.008048, max −0.000116, range
   0.014683 < 0.015) → frozen gate `STRONG_SUPPORT`. The honest reading: removing the
   bundle costs nothing anywhere and the C6 regime is more stable; the mean gain is driven
   by the BASE seed-0 outlier, so this is a *safety* adoption, not a claim that C6 is
   reliably 0.008 better.
2. **The clean model is structurally blind to the removed shortcut** (Q1 mechanism):
   distribution-preserving graph-chemistry row shuffle changes C6 MAE by exactly 0.0000 on
   all three seeds, with the mask active *and* with the channels re-opened (BASE:
   +0.055…+0.079).
3. **The binding mechanism is fine per-occurrence correspondence, not shell marginals**
   (Q2), and it is *more* load-bearing in C6 than in BASE. Edge correspondence is 4–6×
   stronger than node correspondence (frozen Δ: edge +0.0401 mean, node +0.0093 mean, C6).
   From-scratch independence-null controls: node +0.00434 mean (max +0.00925), edge +0.00755
   mean → neither simplification is adopted; the "marginals suffice" hypothesis is rejected
   for edges and unconfirmed for nodes.
4. **Distance-only relation is not supported** (Q3): 320-epoch Δ = +0.005661 →
   `INCONCLUSIVE_KEEP_FULL`; the full distance+overlap+boundary bundle stays.
5. **Sparse dictionary specificity is not established** (Q4): matched DenseTied is
   marginally *better*, `G_dict = −0.002936` (gate needs ≥ +0.003). The coordinate code is
   load-bearing but the sparse encoder is not shown to be specific.

## B. Protocol and provenance

| item | value |
|---|---|
| device | CPU only (`CUDA_VISIBLE_DEVICES=""`), 16 cores, 3 procs × 4 threads |
| data | official ZINC 10 000 train / 1 000 valid; official test never loaded |
| protocol | Adam 1e-3 / wd 1e-5, batch 128, clip 5.0, 320 epochs, Top-5 soup, matched init & data order per seed |
| seed 0 | reused from `results/e2e_dictenv_h1_clarity_audit/matched_cpu/` (bit-exact re-verified) |
| revisions | `bb004bf` → `21fc45d` → `fa5ea5f` → `84a188f` → `38dda2a` → `b21cf91` → `f6a8c57`; every artifact stores its commit; no training-path change after the formal wave started |
| cost | 320-epoch run ≈ 3 930 s at concurrency 3 (≈ 12.3 s/epoch), ≈ 1.93 GiB RSS/proc |
| formal runs | 12 × 320 epochs (A: 4, C: 6, D: 1, F: 1) + 5 × 20-epoch screens/adaptations + 18 frozen-diagnostic variant evaluations (6 checkpoints x 3 operators); ~14 h CPU wall total |

Harness defects found by hand-checking the gates (all fixed, regression-tested, decisions
unchanged): seed-suffixed artifact lookup, `mask=None` bypassing binding overrides,
seed-0 baselines in gates C/F, missing `delta_mae` in stage B, mis-defined dictionary
sparsity statistics. Details in the analysis note §1.

## C. Q1 — C6 stability (Stage A)

| seed | BASE | C6 | Δ | relative |
|---|---|---|---|---|
| 0 (reused) | 0.143298 | 0.128499 | −0.014799 | −10.33 % |
| 1 | 0.130622 | 0.130505 | −0.000116 | −0.09 % |
| 2 | 0.131494 | 0.122265 | −0.009229 | −7.02 % |

Gate `STRONG_SUPPORT` (mean −0.008048, 3/3 negative, max −0.000116 ≤ +0.010, range
0.014683 ≤ 0.015). BASE std 0.0071 / range 0.0127 vs C6 std 0.0043 / range 0.0082.
Stage B: GS1 = 0.0000, EG2 = 0.0000, RS4 = 0.0000 on all C6 seeds (BASE non-zero).

## D. Q2 — node/edge binding: correspondence vs marginals (Stage C)

Frozen replacement of the paired statistic by the analytic independence null (same
weights): C6 node Δ +0.0068/+0.0093/+0.0120, edge Δ +0.0571/+0.0308/+0.0324; BASE node
+0.0124/+0.0047/+0.0020, edge +0.0365/+0.0408/+0.0263.

From-scratch C6+indep controls and 20-epoch warm adaptation:

| role | from-scratch Δ (seeds 0,1,2) | mean | max | warm Δ | adopted |
|---|---|---|---|---|---|
| node | +0.00008 / +0.00368 / +0.00925 | +0.00434 | +0.00925 | +0.00261 | no |
| edge | +0.00900 / +0.00471 / +0.00895 | +0.00755 | +0.00901 | +0.00833 | no |

Per-slot residual structure: n=1 groups are exactly paired = indep (shell0 23 083 slots;
shellpair3 70 slots) and are excluded from evidence; multi-slot residual/paired ratios grow
with range — node shell1 0.10–0.13 (≈19 % exact zeros from constant chemistry), shell2
0.14–0.22; edge shellpair1 0.10–0.20, shellpair4 0.22–0.28 (p90 0.52–0.68). Stage B
assignment shuffs agree: N4 (edge) +0.064…+0.105 vs N3 (node) +0.015…+0.022 on C6.

## E. Q3 — minimal relation bundle (Stage D)

Provenance: 9/9 checks on 200 000 pairs. Screen (20-epoch warm, control 0.128782):
REL-DIST +0.01488, REL-DIST-BOUNDARY +0.00518. Simplest survivor REL-DIST got the
prescribed 320-epoch seed-0 run: soup 0.134159 → Δ +0.005661 → `INCONCLUSIVE_KEEP_FULL`.
No relation simplification adopted; distance-only rejected; full bundle retained.

## F. Q4 — sparse dictionary specificity (Stage F)

FINAL-CLEAN = C6 + paired/paired + sparse. Matched DenseTied (only coding differs):
0.125563 vs sparse 0.128499 → `G_dict = −0.002936` → `SPECIFICITY_NOT_ESTABLISHED`
(single seed; no extension per gate). Dictionary diagnostics: 8/32 nnz per row (sparsity
fraction 0.25), 27–28 active atoms, effective atoms 13.7–15.3, top-5 atoms carry 4.4–5.0
of the 8 activations per row, movement ≈ 5.5 Frobenius.

## G. Stage B core-probe revalidation (frozen registry)

Highlight deltas (own-mask view): PS1 unary first-moment shuffle +1.45/+1.43/+1.36 (C6)
vs +1.34/+1.25/+1.30 (BASE) — the dominant path; PS3/PS6 count rows 0.0000; N6 code zero
+0.27…+0.38; N1/N2 slot/role zero +0.15…+0.29; T1 topology25 zero +0.134…+0.199;
A2/A4/A5 anchor chemistry fill +0.57…+0.75 / +0.35…+0.53 / +0.21…+0.29; EB3 pair
projection zero +0.45…+1.10; GS4 topology shuffle +0.25…+0.27 (both regimes). Full table
in `analysis_tables.md`; identity-view diagnostics in `stage_b_mechanism/`.

## H. Gate discipline and deviations

* No post-hoc threshold changes; gates were applied as frozen. No architecture search.
* Gates that did not fire (correctly): C1 fallback, seed-3 instability extension, node/edge
  independence adoption, relation extension, dense extension, combined FINAL-CLEAN-SPARSE
  run, Stage E (its precondition is unmet).
* Deviations: only harness fixes listed in §B; they changed numbers that were demonstrably
  wrong (seed-0 baselines, unapplied overrides) and did not change any decision.
* `official_test_loaded: false` in every artifact.

## I. Adopted architecture

```text
FINAL-CLEAN = C6 mask (no graph atom/bond histogram, no path_count, no unary/pair counts;
              anchor chemistry retained)
              + paired node binding + paired edge binding
              + sparse tied-IHT coding (s = 8, IHT-10)
```

## J. Claims and status

| claim | status |
|---|---|
| C6 removal is safe under the frozen gate | supported (Q1, 3 seeds) |
| graph-chemistry shortcut is structurally unused in C6 | supported (GS1 = 0 in both views) |
| fine node correspondence is replaceable by marginals | not supported (max Δ +0.00925) — kept |
| fine edge correspondence is replaceable by marginals | rejected (mean Δ +0.00755) — kept |
| distance-only relation is sufficient | rejected (Δ +0.005661, inconclusive-keep-full) |
| sparse coding is dictionary-specific | not established (G_dict −0.002936) |
| the code is load-bearing as information | supported (N6 +0.27…+0.38; PS/RS/N probes) |

## K. Limitations

Three seeds only; BASE seed 0 is an internal outlier, so the C6 *mean* improvement overstates
a performance claim. The C6 gate range is close to the instability threshold. Q4 rests on a
single seed by protocol. Identity-view absolute MAEs are OOD diagnostics. No Stage E, no
PCA32, no shell/radius/shellpair experiment. CPU-matched regime numbers do not transfer to
the historical GPU H1 number.

## L. Next step

Do not run a sparse-specificity confirmatory round: the matched dense control is at least as
good. The open mechanism question this round exposes is the *concentration* of the sparse
code (one coordinate active in nearly every row, top-5 carrying ≈60 % of activations). A new
preregistration would be required to ask whether that concentration is a learned necessity
or a regularizer artefact, and whether rebalancing it changes Q4 — not a gate-tuning rerun.
