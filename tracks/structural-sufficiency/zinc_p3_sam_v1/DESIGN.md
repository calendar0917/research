# DESIGN — zinc_p3_sam_v1 (pre-registered before any arm was trained)

Track `structural-sufficiency`. Two single-model arms, seed 0, no third arm.

## Question

With P3's representation, capacity, initialisation, SSL start and training
schedule held fixed, does **SAM training** improve generalisation relative to a
**same-round AdamW control**? This is a training-method exploration. It is not a
new architecture, not a sharpness causal diagnosis, and not an ensemble /
averaging / routing / distillation round. The 0.05x ZINC raw-y MAE target is
unchanged; both arms may be far from it.

## Arms

| | C — P3-Control | S — P3-SAM |
|---|---|---|
| model | `ad2_models.P3NullModel` (P3N), 413,505 params | identical object, same initial `state_dict` |
| init | frozen train-only masked SSL encoders (`pretrain_encoder_s0.pt`, sha `006fabc1…fa97`) + seed-0 random composer/reader | identical |
| objective | raw-y L1 | raw-y L1 |
| optimizer | AdamW, peak `1e-3`, min `1e-5`, wd `1e-5`, betas `(0.9,0.999)`, eps `1e-8` | same AdamW |
| schedule | 79 updates/epoch × 600 epochs = 47,400 outer updates, 500-step warmup + per-step cosine, clip 5 | same |
| training rule | one forward / one backward / one optimizer step | **vanilla global-L2 SAM**, fixed non-adaptive `rho=0.05`: raw unclipped `g1` → `eps_p = rho·g1_p/(‖g1‖+1e-12)` → second forward/backward at `θ+eps` on the same batch → restore θ → clip `g2` at 5 → one AdamW step |
| selection | full valid every epoch, strict `<`, ties keep earlier | identical |
| test | one descriptive read after the freeze | identical |

Both arms share: official PyG ZINC subset (10,000 / 1,000 / 1,000), train-only
normalisation (frozen P3 `norm.json`), the unchanged v1 collate / batch layout,
`drop_last=False`, the same row-order RNG (seed+101) and therefore the same
molecule order, dropout 0, no BN, fp32, no AMP/compile, no frozen module, no
EMA/SWA/soup, no early stopping, no seed 1, no rho grid.

## Explicit boundaries of this round

- `SAM` here means the author implementation's vanilla two-step ascent
  (arXiv 2010.01412, google-research/sam). The ascent loss at `θ+eps` is *not*
  the inner maximum and is not called “true sharpness”.
- The first-pass gradient is never clipped; only the second-pass gradient is
  clipped at 5, exactly as specified for this round.
- Same outer-update count (47,400) for both arms; S costs 94,800 gradient
  calculations vs C's 47,400. This is **not** an equal-compute comparison.
- `perturbed_train_mae` (S only) is a separate column; it is not the normal
  train MAE.
- Existing four-group analysis (audited radius-2 `signature_unseen` ×
  `y>-3` / `y<=-3`) is descriptive only; no new thresholds, no gate, row 172
  stays in the main numbers.
- Official test has been exposed in earlier rounds; this round's read is
  descriptive and single-pass, not a new blind test.
- Historical P3B / L / DBT predictions are reused as reference rows and
  checked for row identity/labels; they are not retrained and are not part of
  the same-round contrast.
- Stop rule: if the timing epochs project either arm past 60 minutes, save the
  implementation + checks + timing and report the execution limit instead of
  shortening a formal arm. If timing is fine, both arms run the full 600 epochs
  regardless of early-epoch valid values.
- Decision framing is pre-committed in the task brief: ≥~0.003 same-direction
  valid+test gain with a supportive tail is a training-method signal (still not
  the 0.05x target); ~0.001 / one-split / within-noise is a weak or split-
  dependent signal; worse normal fit and no valid/test gain closes this fixed
  rho/recipe/horizon intervention.
