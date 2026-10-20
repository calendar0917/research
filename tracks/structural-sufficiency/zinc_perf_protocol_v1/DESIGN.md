# DESIGN — ZSRT (ZINC Structural Relation Transformer), zinc-perf-protocol-v1

## 0. Bounded literature/code check (before implementation)

The check was limited to the two direct references named in the brief and to
the already-run in-repo negative rounds. No broad literature sweep was done.

**GRIT** — vendored pristine snapshot at `LiamMa/GRIT`
`6c988ea600a606fbb49a2246c64a2d37396b3ab5`, official
`configs/GRIT/zinc-GRIT-RRWP.yaml` (SHA256
`e60b425b…`). Verified values: 473,473 parameters, 10 layers, 8 heads,
hidden 64, RRWP `ksteps=21`, batch 32, 2,000 epochs, 50 warm-up epochs,
AdamW, cosine to 1e-6, L1 loss, sum pooling, no local MP (full attention).
The previously measured A100 cost in this repository is **35.2 s/epoch
train-only** (GRIT reference preparation, 2026-10-19), i.e. ~20.5 A100-h for
one seed at the paper recipe. That is far outside this round's 8 h cap, so the
paper recipe is not purchased and the paper's 0.059 test MAE is not a local
target.

**GPM** — checked at `Zehong-Wang/GPM`
`2ed3ebf7df8d8401520e6ec3bdbd1c2b291b5784` (2025-05-28), `config/main.yaml`.
The ZINC entries (`zinc` and `zinc_full` are distinct keys) use batch 1,024,
LR 0.01, dropout 0.1, weight decay 0, random-walk node PE (dim 8) with a GRU
encoder, p=q=1; the default schedule is 1,000 epochs with early stopping.
The model samples random-walk **path patterns** in a preprocessing step and
quantises them through a VQ codebook before a Transformer. This is a real
alternative, but it adds a stochastic sampled-pattern pipeline and a codebook
whose cost/behaviour would have to be validated inside this round; it also
does not fit the "one family, four cheap configs" budget. It is documented as
the fallback direction, not implemented here.

**Already-run in-repo alternatives** — structural-token attention (STA),
bond-conditioned context attention (BCCA), and exact-type residual (ESR) are
negative results under the old soup protocol. The new candidate must not be a
rename of those. The differences are stated in §2.

## 1. Computation graph

```
raw PyG ZINC graph (atom type a_v in 0..20, bond type t_uv in {1,2,3})
  |
  +-- RRWP (deterministic, label-free): P = D^-1 A,
  |     node diag  r_v = [ (P^k)_vv ]_{k=0..K},  K=21
  |     pair value r_uv = [ (P^k)_uv ]_{k=0..K}
  |
  +-- TypeDictNode(a_v) -> h_v^0  (explicit atom identity)
  +-- TypeDictEdge(t_uv) -> e_uv^0 (explicit bond identity)
  +-- h_v^0 += FC(r_v);  e_uv^0 += FC(r_uv) on the full pair graph
  |
  L x GritTransformerLayer (full attention over all node pairs):
      score_uv = (Q h_u)·(K h_v)  modulated by  E e_uv
      e_uv'    = O_e(wE_uv)        (edge state update, update_e=True)
      h_v'     = [ sum_u softmax_u(score) V h_u ] + edge-enhancement row term
      h_v'     = deg-scaled residual + FFN + residual (BatchNorm)
  |
  z = sum_v h_v^L                      (explicit additive graph pooling)
  y_hat = MLP(z)                       (64 -> 32 -> 16 -> 1)
```

All atom/bond identities and the exact edge multiplicities are preserved as
categorical embeddings until the attention layers mix them. There is no
hand-built junction/bond-incidence summary and no iterative neighbour-only
message passing: every layer attends over the whole graph, and RRWP supplies
the global relational geometry from step 0.

## 2. Why this is not P3 / STA / BCCA

| | P3 (BJOC-Null) | STA | BCCA | **ZSRT (this round)** |
|---|---|---|---|---|
| structural objects | atom→bond→junction MLPs, hand-built 2-hop junctions | P3 atom/bond/junction tokens | P3 bond + junction context | atoms + real bonds only, typed embeddings |
| relation input | none; fixed incidence sums `W_b,V_b` | none (learned token attention) | learned per-bond junction weights | **RRWP k-step pair + node encodings in every layer** |
| composition | one-shot `sum_b c_b`, `sum_i a_i`, `sum_i a_i^2` | global token attention | reweighted Null moments | global full attention + edge-state updates + degree scaler |
| P3 reader input | 321-D mixed reader vector | kept P3 substrate | kept P3 substrate | **not used** |
| iteration | none | attention layers on P3 tokens | none | attention layers, not local MP |
| parameters | 413,505 | 483k | 419,649 | 380,161 (A/B/C), 377,473 (D control) |

STA and BCCA kept the frozen P3 object encoder and added attention/reweighting
on top of it; ESR added an exact-type lookup into the atom embedding. ZSRT
replaces the representation and the composition with a typed atom/bond +
RRWP full-attention backbone. The only borrowed part is the mature GRIT
attention/edge-update block; that borrowing is disclosed and is not claimed as
novelty.

## 3. Where a gain could come from

1. **Pair structure before prediction.** P3's only graph-level signal is an
   extensive sum of per-bond vectors and atom first/second moments. Two
   molecules with the same bond/junction marginals but different long-range
   organisation are hard to separate. RRWP pair values expose exactly that
   geometry to each attention layer.
2. **Multiplicities are not averaged away.** The P3 Null grouping deliberately
   replaces true junction ownership by graph means. ZSRT keeps each real bond
   and its order as a typed edge, so double/triple bonds and ring closure are
   first-class signals.
3. **Task-adaptive global routing.** The attention can learn which pairs matter
   for solubility instead of committing to a fixed sum/sum-of-squares pooling.
4. **Better inductive bias per parameter.** GRIT's published ZINC result at
   473k parameters shows the family can reach ~0.059 test MAE at the paper
   budget; the open question here is what it does at a matched 28,440-update
   budget.

## 4. Main risks

* **Budget mismatch.** 360 epochs at batch 128 is 28,440 updates, ~1/5 of
  GRIT's 2000×313 updates. If valid is still falling at epoch 360, the result
  is budget-limited, not an architecture verdict.
* **Small-graph inefficiency.** Full attention over 9–37 nodes is cheap but
  launch/BN/edge-state overhead can dominate; measured epoch time decides the
  affordable variant.
* **Overfitting.** 10k molecules, ~380k parameters, dropout 0 and only weight
  decay 1e-5. Best-valid selection plus the LR/WD screen is the mitigation;
  no extra regulariser sweep is opened.
* **RRWP cost/benefit.** RRWP may add little over typed attention at this
  budget. The `D_no_rrwp` control measures that directly; if D ties or beats A,
  the round reports that the relation encoding did not pay for itself.
* **BatchNorm with small batches.** Batch 128 keeps BN statistics reasonable.
  Validation uses `model.eval()` with running statistics and never updates
  them.

## 5. Parameter and compute budget

| variant | layers | RRWP | params | measured 360-epoch cost |
|---|---:|---|---:|---|
| A_main | 8 | k=21 | 380,161 | 0.954 A100-h (9.5 s/epoch) |
| B_lr_low | 8 | k=21 | 380,161 | 0.933 A100-h (9.3 s/epoch) |
| C_wd_high | 8 | k=21 | 380,161 | 1.007 A100-h (10.1 s/epoch) |
| D_no_rrwp | 8 | off | 377,473 | 0.825 A100-h (8.3 s/epoch) |

The original CPU-smoke counts (378,753 / 375,937) were from a GRIT
configuration without the final edge-state/degree-scaler settings; the
train-time counts above are authoritative and are asserted in
`code/tests/test_zpp_core.py`. The projection was replaced by the measured
smoke epoch time before the formal screen (see `analysis/smoke.json`). Budget guard: screen 4 configs × seed 0, then 2 additional seeds for
the selected config, total new-model GPU time target ≤ 5.5 A100-h; P3
calibration ≤ 0.5 h; reserve ≥ 0.5 h of the 8.0 h cap.

## 6. What would count as a result

* **Signal:** selected ZSRT beats P3 best-valid by more than ~0.003 seed-mean
  with the same two/three seeds, and the paired per-seed differences have the
  same sign. A ~0.001 difference or a difference smaller than the seed spread
  is reported as a small/ambiguous signal, not a breakthrough.
* **No signal:** all four screen configs are at or above P3 best-valid; the
  round stops at the development conclusion, no test read for ZSRT, and the
  next decision is a new pre-registration (e.g. a longer-budget GRIT run or the
  GPM path-pattern direction), not another small module on P3.
