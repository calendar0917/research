# DESIGN — COC-v1: counterfactual ownership composer

**Round:** `zinc_counterfactual_ownership_composer_v1`
**Protocol:** `PROTOCOL.md` (`ssrc-zinc-counterfactual-ownership-composer-v1`)
**Frozen config:** `configs/coc_v1.json`
**Status:** design + CPU-gate specification, written before any GPU run.
**Access boundary:** official valid and official test are never loaded in this round.

---

## 0. One sentence

Keep every DCC-L local structural input and every parameter, but replace the absolute
local composer output by the **difference between the same nonlinear local composer
evaluated on true junction-to-bond ownership and on the molecule mean-field reference**:

```
c_loc = f(x_real) - f(x_null),      f = local_composer, same parameters in both calls
```

The reader then receives the first and second graph moments of this counterfactual
difference, exactly where DCC-L currently receives the first and second moments of
`f(x_real)`.

---

## 1. Why this is a different object from prior rounds

### 1.1 DCC-L

DCC-L computes

```
c_glob = g(x_null),      x_null = [b, Wg, Vg, log1p(n), bond_feat]
c_loc  = f(x_real),      x_real = [b, Wl, Vl, log1p(n), bond_feat]
z      = [Σc_glob, Σa, Σa², global33, Σc_loc, Σc_loc²]
```

`Wg,Vg` are the molecule-level mean-field counts (`n_b * graph_mean(w)`), and
`Wl,Vl` are the true incidence sums over the junctions touching each bond. DCC-L
learns two separate nonlinear functions of the two absolute inputs. It never asks
what part of `f(x_real)` is specifically due to the *difference* between real
ownership and the mean field.

### 1.2 DCC-L C arm / SCSB

The C arm replaces `x_real` by `x_null` everywhere and then trains a separate
absolute composer; it measures whether real ownership exists at all. SCSB mixes
the two inputs by a label-free support scalar before a single composer:

```
x_mix = alpha * x_real + (1-alpha) * x_null
c     = f(x_mix)
```

That is a convex **input shrinkage** with a hand-designed alpha. It does not share
a composer across a real/null counterfactual and it does not subtract the two
outputs. SCSB's negative result is about the support-conditioned mixing operator,
not about this operator.

### 1.3 DCC information audit / RCC-v0 / block ownership

- The DCC information audit tested whether pooled `z` retains correspondence-aware
  joint statistics, and whether adding frozen joint statistics to a probe improves
  prediction. It did not change the DCC-L forward and did not construct a
  counterfactual composer difference.
- RCC-v0 tested a graph-level quadratic relation kernel `Zᵀ(K_R−K_D)Z` and stopped
  at a CPU redundancy gate. It is a global second-order operator, not a per-bond
  shared nonlinear composer evaluated on two ownership assignments.
- Block ownership rounds changed the grouping object (biconnected blocks) and added
  a new block channel/parameters. They did not keep the DCC-L local inputs fixed
  while subtracting a mean-field reference inside the same composer.

Therefore the exact computation

```
f(x_real) - f(x_null),   f shared, no detach, no new parameters
```

has not been materially tested by the prior negative results. This is the only
claim of novelty the round makes; it is not a claim that the operator is useful.

---

## 2. Mathematical reading

Let `f` be the shared two-layer SiLU MLP of the DCC-L local composer. For a bond
`b`, write

```
u_b = (b_b, Wl_b, Vl_b, log1p(n_b), bond_feat_b)
v_b = (b_b, Wg_b, Vg_b, log1p(n_b), bond_feat_b)
d_b = f(u_b) - f(v_b)
```

If `f` were affine, `d_b` would collapse to a fixed linear functional of
`(Wl_b-Wg_b, Vl_b-Vg_b)`. The reader already receives `Σc_glob`, `Σa`, `Σa²` and
`Σc_loc²`; a purely linear difference would be a much weaker object. The mechanism
being tested is specifically the **nonlinear** difference: `f` can amplify or
suppress ownership deviations depending on the local context (`b_b`, `n_b`,
`bond_feat_b`), and the same parameters must explain both the absolute mean-field
response and the deviation.

This is a conditional centering operation in function space, not in input space:

- input-space centering would be `f(x_real - x_null)`;
- output-space centering of a linear model would be equivalent to a linear input
  difference;
- COC is `f(x_real) - f(x_null)`, which retains the nonlinear interaction between
  the mean-field level and the ownership deviation.

The reader still sees the absolute global channel `Σc_glob`; only the local channel
is replaced by the difference. Thus the model does not lose the mean-field
representation, and the atom/global channels are untouched.

---

## 3. Why sharing parameters is the definition of the mechanism

Two independent composers would be a different, larger model: the null branch could
drift to an arbitrary function and the difference would no longer be a counterfactual
of the same function. The shared-parameter constraint is what makes the object
"the part of the learned local response that is attributable to the ownership
assignment, relative to the molecule mean field." It also keeps the parameter count
at exactly 494,193 and makes the initial state bitwise identical to DCC-L.

The second call must remain differentiable. If it were detached, the null reference
would be a constant per batch and the gradient would not train the shared function
to represent the mean-field response; the operator would degenerate into a residual
against a fixed baseline. We explicitly check the gradient path in C5.

---

## 4. Initialization and cold start

The construction follows DCC-L exactly:

1. `torch.manual_seed(seed)`;
2. build `P3NullModel(norm)`, load the seed-matched SSL encoder checkpoint;
3. capture the P3 state;
4. `torch.manual_seed(seed + 90731)`;
5. build the dual-channel model, load all P3 weights;
6. copy the old 321 reader columns and zero the new 192 columns.

Because `COCDualChannel` adds no submodules and inherits the DCC-L constructor, the
RNG consumption and every shared tensor are identical. The initial prediction is
therefore the P3 prediction for both COC and DCC-L: the new reader columns are zero,
so the nonzero `c_loc` difference cannot affect the output at step 0.

This cold start has a known consequence: at the first backward pass, the local
composer receives zero gradient (the reader columns multiplying its moments are
zero), while the reader's new columns receive a nonzero gradient from `Σc_loc` and
`Σc_loc²`. After the first AdamW step, the reader columns become nonzero and the
local composer path is trained. C5 records both the expected zero first-step local
gradient and the nonzero post-step gradients; this is a property of the inherited
zero-column initialization, not a bug.

---

## 5. What the difference representation should look like if alive

For each bond we log:

- `rms(c_loc_b)` and `rms(c_abs_b)` where `c_abs = f(x_null)`;
- the ratio `median_b rms(c_loc_b) / median_b rms(c_abs_b)`;
- the fraction of entries with `|c_loc| > 1e-8`;
- the within-molecule variance of `c_loc` across bonds;
- the participation ratio of the pooled within-molecule covariance;
- the RMS ratio to `c_real = f(x_real)`.

If `c_loc` is numerically zero, the reader's local columns carry no information and
the round must stop before GPU (`STOP_MECHANISM`). If `c_loc` is nonzero but the
reader ignores it, the diagnostics will show a live representation with no
generalization gain; that is a different failure mode and is reported as such.
A higher participation ratio is not treated as evidence of performance.

---

## 6. Training and evaluation discipline

- Fit8k only, 47,400 updates, batch 128, AdamW, warmup 500, cosine 1e-3→1e-5,
  wd 1e-5, clip 5, dropout 0, raw-y L1.
- The historical L checkpoints are reused, not retrained. Their Fit8k rows,
  seed-matched SSL, initialization hash, batch generator and update budget are
  verified before any comparison.
- Monitor1k selects the best checkpoint and is the only set used for the
  pre-registered Monitor decision. Locked1k is read once, only after a Monitor PASS
  and a hash freeze.
- Official valid/test are never loaded.

The primary scientific contrast is

```
Delta = MAE(COC) - MAE(L)
```

on the same molecules and the same checkpoint semantics. Best-Monitor, W20, T100 and
final are reported separately; best is never compared with another model's final.

---

## 7. Failure modes the design can distinguish

1. **Representation degeneracy:** `c_loc` collapses to zero or near zero. Detected by
   C8 and post-training diagnostics; action `STOP_MECHANISM`.
2. **Optimization failure:** `c_loc` is alive but the model cannot fit or diverges.
   Detected by non-finite values, Monitor tail vs best, and Fit8k train MAE ratio;
   action `STOP_NO_MATERIAL_GAIN`.
3. **Learned but non-generalizing difference:** `c_loc` is alive, training fits, but
   Monitor overall/any-unseen does not improve; action `STOP_NO_MATERIAL_GAIN`.
4. **Monitor gain but Locked non-confirmation:** the effect is a Monitor/window
   artifact; action `STOP_LOCKED_NOT_CONFIRMED`.
5. **Gain concentrated in the penalty tail or one row:** excluded by the Locked
   drop-all-penalty / drop-top1-penalty / drop-worst10 gates.

The report only assigns a failure cause when the recorded diagnostics distinguish it.

---

## 8. Non-claims

- This design does not claim that input-space centering, all mean-field methods, or
  all ownership methods are useless.
- A positive result would not prove originality; it would only support this specific
  operator under the frozen Fit8k/Monitor/Locked protocol.
- The Locked1k split has been used in prior work; this round's read is a controlled
  holdout evaluation, not a pristine blind test.
- No official valid/test result is produced, and no full10k run is authorized by this
  design.
