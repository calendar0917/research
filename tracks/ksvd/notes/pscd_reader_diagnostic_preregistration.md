# Pre-registration — PSCD-R-v0: Frozen-Dictionary Motif Readability Audit

Round name: **PSCD-R-v0**. Written **before** the formal A100 run, following the
`remote-research-runner` skill (local code → remote compute → local analysis).

This round keeps the entire PSCD-v0 representation object **frozen** and changes
only **how a tiny reader represents a dictionary atom**, in order to separate two
hypotheses about why the lossless composition code is hard to read:

* **H1 (opaque-atom)** — giving each motif only an independent ID embedding
  destroys atom/bond-level parameter sharing.
* **H2 (port-bottleneck)** — the composition reader collapses all attachment
  ports of an occurrence into one state too early, so inter-motif relational
  information cannot propagate.

Official ZINC `test` is **never** loaded, instantiated or referenced. `y` is used
only in the readers. This note does not modify any historical record.

---

## 0. Context this round does **not** re-litigate

PSCD-v0 (2026-09-20, commit `cdae8b0`, A100) established, on ZINC train 10 000 /
valid 1 000:

* `Decode(D, C_G) == G` for 100 % of train + valid (exact colored canonical key);
* permutation invariance (0/10 000 relabel mismatches);
* `mean(M_G / n_G) = 0.320`, MDL ratio `L_PSCD / L_atom = 0.215`;
* 64/64 learned motif types reused; healthy port complexity (0.471 port fraction
  for size ≥ 4); full-PSCD collision = 0.

Stage-B (frozen dictionary, matched tiny readers `d = 32, L = 2`, seed 0) gave
valid MAE **composition 0.5429** vs **raw atom graph 0.3113** (motif bag 0.5609).
This round does **not** re-prove losslessness and does **not** imply the object is
wrong. It asks whether the lossless structured code has a low-capacity,
shared-parameter, task-friendly decoder.

---

## 1. Pre-run defect finding (reader plumbing, not the object)

While inspecting the frozen PSCD-v0 Stage-B code we found that the **as-shipped
composition reader is not batch-invariant**. In `collate(kind="comp")` the node
array is laid out interleaved per graph (`[occ_g0, conn_g0, occ_g1, conn_g1, …]`)
and `off += M + Ec`, while the reader forward assumes

```
h[:M_total] = E_M(occ_ids)     # all occurrence nodes first
h[M_total:] = E_B(conn_ids)    # then all connection nodes
```

For any batch with more than one graph whose graphs have connection nodes the two
layouts disagree, so the motif/bond features land on the wrong nodes. Verified
directly: with a fixed random init, batch-of-4 predictions differ from the
per-graph single predictions by up to **1.36 MAE**, whereas the bag and raw
readers reproduce single predictions with batch error `< 1.4e-6`.

Consequence: the historical number `composition valid 0.5429` was produced by a
**corrupted** reader and cannot be interpreted as a property of the frozen
dictionary or of the opaque representation. This round therefore runs two opaque
controls:

* `R0ship` — the **as-shipped** PSCD-v0 composition collate, unchanged, to
  reproduce / quantify the historical behaviour;
* `R0` — the **same module** with only the collate node order fixed
  (`[all occurrence nodes | all connection nodes]`, edges remapped). This is the
  honest, batch-invariant implementation of the opaque-ID hypothesis and is the
  primary opaque baseline for H1/H2.

The fix is confined to node/collate ordering; the module, its parameters and its
message passing are untouched. `bag` and `raw` readers need no change.

---

## 2. Frozen artifacts (asserted, never relearned)

Loaded from `tracks/ksvd/results/pscd_compositional/stageA.pkl` (PSCD-v0, commit
`cdae8b0`; local and remote copies are byte-identical, md5
`b180e9695fb4f0a2cb366338bd58f459`):

* the 64 learned merge rules + 21 singleton atom types (85 motif types);
* occurrence partitions for train (10 000) and valid (1 000);
* canonical motif graphs, automorphism orbits, lex-min port maps;
* composition edges `(i, p_i, t, j, p_j)`.

**Freeze gate.** Before any training the run re-computes `Decode(D, C_G)` for all
train + valid and requires **100 %** exact colored-canonical reconstruction. If
any graph fails, the run aborts with **no training**.

Forbidden this round: relearning the vocabulary, frequency re-tallying,
vocabulary / motif-size sweeps, task-aware BPE, attention / transformer / virtual
node, depth/width sweep, multiple seeds, soup, official test, long training.

---

## 3. The four readers (+2 opaque controls)

All readers: hidden width 32, outer composition depth 2, seed 0, identical
optimizer (Adam, lr 1e-3, wd 0, clip 5), identical MAE loss, batch 128, identical
tiny final head `Linear(32,32) → SiLU → Linear(32,1)`, sum pooling.

| id | reader | dictionary-atom representation |
|---|---|---|
| `R0ship` | opaque-ID composition (as-shipped) | `E_M(k)`, historical (defective) collate |
| `R0` | opaque-ID composition (corrected) | `E_M(k)`, corrected collate |
| `R1` | structured-static | shared intra-motif encoder; `h_k = Σ_v z_{k,v}^{(2)}`, port `r_{k,p} = z_{k,p}^{(2)}` |
| `R2` | port-resolved structured | per-active-port dynamic state + occurrence core |
| `R3` | matched raw atom graph | atom + bond embeddings, 2 MP layers, sum pool |

### R1 — shared structural dictionary-atom encoder

One shared encoder over all dictionary motifs; `z_{k,v}^0 = E_V(x_v)`,
`b_{uv} = E_E(t_{uv})`, `L_intra = 2` sum message passing
`m_{k,v} = Σ_{u∈N(v)} ψ_l(z_{k,u}, b_{uv})`,
`z_{k,v}' = SiLU(W_s z_{k,v} + W_m m_{k,v} + b)`. **No** attention, norm,
residual tower, or per-motif encoder. Motif state uses **sum** (not mean) so
motif size is part of the composition mass:

```
h_k^{dict} = Σ_{v∈V_k} z_{k,v}^{(2)} ,      r_{k,p}^{dict} = z_{k,p}^{(2)} .
```

Occurrence init `h_i^0 = h_{k_i}^{dict}`. The outer reader is kept **as close to
R0 as possible**: same connection-node bipartite message passing, same `W_self`,
`W_msg`, `W_edge`, same `E_B(t)` connection feature; only the motif/port features
are replaced (`E_M(k) → h_k^{dict}`, `E_P(slot) → r_{k,p}^{dict}`). This is the
key isolation control.

### R2 — port-resolved structured reader

Same encoder. Each occurrence `i` keeps a separate dynamic state for every
**active** port `p ∈ P_i^{active}`: `r_{i,p}^0 = r_{k_i,p}^{dict}`, plus the
occurrence core `h_i^0 = h_{k_i}^{dict}`. Per outer layer:

```
m_{i,p_i ← j,p_j} = ψ_l(r_{i,p_i}, r_{j,p_j}, E_B(t))      (summed over a port's edges)
s_i               = h_i + Σ_p W_P r_{i,p}
r_{i,p}'          = SiLU(W_R r_{i,p} + W_S s_i + W_M m_{i,p})
h_i'              = SiLU(W_H h_i + W_A Σ_p r_{i,p}')
```

`L_outer = 2` (no third layer). No port attention / all-pairs port transformer /
motif-specific matrices / depth tuning. Readout `z_G = Σ_i h_i^{(2)}`.

### R3 — raw reader

The matched PSCD-v0 raw atom-graph reader, unchanged: atom embedding 32, bond
embedding 32, 2 MP layers, sum pool, `32 → 32 → 1` head. No added advantage.

---

## 4. Training budget and stopping (screening, not convergence)

* Optimizer-step budget, not epochs. Primary screening: **≤ 3000 steps** per
  model (~38 epochs at batch 128 over 10 000).
* Evaluation every **100 steps**; checkpoints recorded at **500, 1000, 2000,
  3000** (each evaluation records train MAE, valid MAE, best valid, wall-clock,
  steps/s).
* **Early termination**: after step 1000, if **5** consecutive evaluations do not
  improve best valid MAE by **≥ 0.003**, stop that model.
* **Hard failure stop** [structured models only]: at step 2000, if
  `valid MAE > 0.48` **and** `train MAE > 0.43`, stop — "clearly not closing the
  PSCD readability gap".
* **Extension gate** (best structured reader only, and R3 if its curve is still
  moving): allow continuation to **6000 total steps** only if any of
  * E1 `R_recover^valid ≥ 0.50`,
  * E2 `MAE_structured^valid ≤ MAE_raw^valid + 0.10`,
  * E3 last-1000-step valid improvement `≥ 0.02` with train still improving.
* **Absolute ceiling 6000 steps** for any single model. No 100/200-epoch runs, no
  "just run more", no soup, no seed sweep, no LR sweep. If 6000 steps are
  inconclusive, the round reports **inconclusive**, not more budget.

Matched gap metric (train and valid):

```
R_recover = (M_opaque − M_structured) / (M_opaque − M_raw)
```

with `M_opaque = R0` (corrected opaque), `M_structured ∈ {R1, R2}`,
`M_raw = R3`, all at the **same step**. `R0ship` is reported separately.

---

## 5. Reporting

* Learning curves `MAE vs optimizer step` (train figure and valid figure) with
  R0, R0ship, R1, R2, R3 overlaid.
* Parameter breakdown per reader: dictionary structural encoder params (shared),
  per-motif lookup params, outer reader params, head params, total.
* `R_recover` at each matched checkpoint (train + valid).
* **Mechanism audit** on the trained dictionary encoder of the best structured
  reader: (A) motif-embedding diversity (pairwise cosine / Euclidean, effective
  rank, no collapse); (B) structural-similarity sanity (motifs sharing a
  connected induced substructure ≤ 4 atoms vs unrelated); (C) port
  differentiation (within-motif pairwise cosine of `r_{k,p}`).
* **Evaluation-only ablation** (no retraining): replace every dictionary port
  vector by the motif mean `\bar r_k` at inference and report the change in valid
  and train MAE. Near-zero change ⇒ ports not really used; clear degradation ⇒
  port-resolved dictionary semantics are used.
* Parameter discipline: because R2 has more parameters than R0/R3, report
  **MAE gain per parameter increase**; do not announce success from a larger
  model.

---

## 6. Interpretation and decision (frozen before the run)

Primary comparison uses **R0 (corrected opaque)**, R1, R2, R3 at matched steps.
Cases (from the brief):

* **Case A** `R_recover^valid ≥ 0.60` and train close to raw ⇒ frequency motifs
  are task-usable; the dominant failure was treating structured dictionary atoms
  as opaque symbols. → `develop the structured dictionary reader`.
* **Case B** `R2 ≫ R1 ≫ R0` ⇒ port resolution specifically required. → keep the
  frequency dictionary, develop a compact port-structured reader.
* **Case C** `0.2 < R_recover^valid < 0.5` and structured train floor still
  clearly above raw ⇒ opaque parameterisation was one problem, but
  frequency-only abstraction remains task-hostile. → `proceed to task-aware
  MDL-constrained motif discovery`.
* **Case D** `R_recover^valid ≤ 0.20` and R1/R2 train MAE still far above raw ⇒
  frequency partition places abstraction boundaries in task-hostile locations.
  → `proceed to task-aware MDL-constrained motif discovery`.

Additional pre-registered possibility: if **R0 (corrected)** already jumps to near
the raw reader, the historical gap was dominated by the reader defect, not by the
opaque representation or ports; the round then reports that the historical
`composition 0.5429` was not a valid measurement of the frozen object, and
`R0ship` quantifies the defect.

Final conclusion first line is one of the brief's four frozen sentences; next
decision is one of `develop the structured dictionary reader` /
`proceed to task-aware MDL-constrained motif discovery` /
`run one targeted reader diagnostic`. This round does **not** start the next
phase.

## 7. Data-free correctness tests

`tracks/ksvd/tests/test_pscd_reader.py` locks: the corrected composition collate
is batch-invariant; the as-shipped collate is not; R1/R2 collates and readers are
batch-invariant and finite; the shared encoder exposes `h_k = Σ_v z_v` so the
port-mean ablation is well defined.
