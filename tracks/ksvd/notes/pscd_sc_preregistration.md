# Pre-registration — PSCD-SC-v0: Strong-Capacity Port-Operator Reader Audit

Round name: **PSCD-SC-v0** (*Strong-Capacity*). Written **before** the formal
Arm-C (and Arm-B) A100 runs, following the `remote-research-runner` skill
(local code → remote compute → local analysis). This note does **not** modify
any historical record.

Official ZINC `test` is **never** loaded, instantiated or referenced. `y` is used
only inside the strong readers / decode oracle. Splits are the repository's
canonical PyG ZINC `subset=True`: **train 10 000 / valid 1 000**.

---

## 0. The single question

PSCD-v0 froze a frequency-derived 64-motif / max-size-8 composition dictionary
`(D, C_G)`. The object is healthy (100 % exact decode, permutation invariance,
`mean(M_G/n_G)=0.320`, 64/64 reuse, port complexity 0.471, MDL ratio 0.215).
PSCD-R-v0 then showed the historical Stage-B failure was dominated by a **reader**
defect (a non-batch-invariant collate) plus an opaque/port-collapsed reader:
with a corrected structured reader the frozen code is read as well as the raw
atom graph at a **tiny** budget (`d=32, L=2, ≤6000` steps; `R2` best-valid
0.3246 ≈ raw 0.3363). But the *converged* regime was never reached, and no
reader ever left the ~0.32 band.

This round therefore asks exactly one question, with the dictionary **completely
frozen**:

> Under a **strong-reader regime** (full canonical training budget, `d=64`,
> `L_outer=4`, alternating inter/intra-motif operator), does the fixed PSCD
> representation follow the raw graph down into a `0.1–0.2` MAE band?

$$
\boxed{\mathcal D \text{ completely frozen};\ \text{study } C_G \rightarrow y.}
$$

## 1. What this round does **not** re-litigate / forbids

* The PSCD dictionary: merge sequence, tokenization, port assignments,
  composition graphs, `stageA.pkl`, exact decoder are **frozen**.
* Forbidden this round: task-aware motif search, merge-rule / boundary
  changes, vocabulary-size sweep, `max motif size` sweep, official test,
  Gumbel / differentiable tokenizer, learned dictionary boundaries, more than
  one mechanism-oriented reader revision.

This is a **capacity diagnostic of the frozen object**, not a new dictionary
experiment; it is not "task-coupled dictionary learning" (that is a *later*
round, only if this one passes).

## 2. Frozen artifacts and baselines

* **Frozen PSCD object** — `tracks/ksvd/results/pscd_compositional/stageA.pkl`
  (PSCD-v0, commit `cdae8b0`; md5 `b180e9695fb4f0a2cb366338bd58f459`). Re-decoded
  for **100 %** exact colored-canonical reconstruction of train + valid *before*
  any training; abort with no training otherwise.
* **Arm A canonical strong raw baseline = `zinc-b-full`** — the repo's current
  canonical strong architecture (`tracks/ksvd/experiments/luyin16/zinc_shared_structural_patch_encoder.py`),
  i.e. the shared connectivity-aware patch encoder (vocab-free) built by
  mirroring `capacity_decomposition.build_cell("A", seed)`; **84 495 params**,
  historical 2-seed Top-5 soup valid **0.118972** (raw 0.124859). Reused builder,
  protocol, checkpoint/soup, seed handling; nothing hand-written.

| Arm | what | model | budget |
|---|---|---|---|
| **A** | canonical strong **raw** | B-Full (reused builder) | canonical protocol, seed 0 |
| **B** | decode oracle | **same Arm-A weights** on `G` vs `Decode(C_G)` | inference only |
| **C** | strong **PSCD port-operator** | frozen `(D, C_G)` → MAE | canonical protocol, seed 0 |

## 3. Arm A — canonical strong raw (execution regime)

Re-run B-Full **seed 0** under the canonical `OPTIMIZED_PROTOCOL`
(Adam, lr 1e-3, wd 1e-5, batch 128, clip 5, L1, no scheduler, max_epochs 240,
patience 40, equal-weight Top-5 soup), on the A100 host, reusing the cached
patch graphs and the exact builder/config. Purpose: a real strong ceiling *in
this execution regime*, not architecture search.

**Stop rule.** If Arm A valid MAE is clearly stuck `> 0.16`, the round reports a
baseline-reproduction problem and stops interpreting performance; a failed raw
baseline is never used as a PSCD ceiling.

## 4. Arm B — decode oracle (no training)

With the **same Arm-A model instance and weights**, feed `G` and
`Decode(C_G)`, for **≥ 512 train + 512 valid** graphs, and check
`f_θ(G) ≈ f_θ(Decode(C_G))` with `max|Δŷ| < 1e-5`. A CPU deterministic
reference must pass; if GPU nondeterminism makes the threshold unrealistic, the
GPU error *distribution* is reported instead. The decoded graph is processed
through the exact same Arm-A pipeline (records + encoding), not a hand-written
clone. This only proves the artifact hides no data loss; no model is trained.

## 5. Arm C — strong compressed port-operator reader (primary)

The reader may **not** reconstruct/maintain all original atom states. Dynamic
state exists only at **motif occurrence centers** and **active motif ports**:

```
dynamic state count = |O_G| + |{active ports}|
```

* **Shared motif structural encoder** `E_motif` (2 edge-aware message-passing
  layers over each ≤8-atom dictionary motif, 64 motifs, one parameter set):
  outputs motif descriptor `g_k` and canonical port descriptors `s_{k,p}`.
  **No** free per-motif embedding table is the primary identity.
* Occurrence init: `c_o^(0) = g_{k_o}`,
  `h_{o,p}^(0) = s_{k_o,p} + W_C c_o^(0)`.
* **Alternating operator, `L_outer = 4`**, each layer:
  1. **inter-motif** (sum-aggregated, no attention):
     `m_ext = φ_ext^(ℓ)(h_{o,p}, h_{o',q}, E_B(t))`,
     `\tilde h_{o,p} = h_{o,p} + Σ m_ext`;
  2. **intra-motif transfer** (the round's core difference from R2): shared
     transfer descriptor `τ_{k,pq} = ψ(s_{k,p}, s_{k,q}, g_k)` (computed once,
     no layer index), then
     `m_int = φ_int^(ℓ)(\tilde h_{o,p}, τ_{k,pq})`,
     `h_{o,q}' = SiLU(W_S h_{o,q} + Σ_p m_int + W_G c_o)`,
     `c_o' = SiLU(W_C c_o + Σ_p W_P h_{o,p}' + W_D g_{k_o})`.
  `φ_ext`, `φ_int` are small `Linear→SiLU→Linear` MLPs, **layer-specific**;
  `W_C, W_S, W_G, W_P, W_D` are the shared (no layer index) state maps.
* Readout pools **occurrence centers only**: `z_G = Σ_o c_o^(L)`, then
  `Linear(d,d) → SiLU → Linear(d,1)`. Port states are *not* pooled directly.
* Occurrences with **no** active port are kept (center self-updates only).

**Fixed capacity, no sweep:** `d = 64`, `L_outer = 4`, `L_intra = 2`,
`L_motif = 2`. Hidden sizes are not tuned. Parameter accounting is reported in
five buckets (motif encoder / inter / intra / head / total); the ideal band is
`0.5 P_raw ≤ P_PSCD ≤ 1.5 P_raw`, and any overshoot is **reported, not fixed by
hidden-size tuning**.

## 6. Compression and compute must be *measured*

Not just `M_G/n_G=0.32`. For train + valid report

```
N_raw    = |V_G| ,                    N_PSCD = |O_G| + Σ_o |P_o^active| ,
r_state  = N_PSCD / N_raw ,
```

plus inter-motif message count, intra-motif port-pair message count, raw edge
message count, peak GPU memory, graphs/s and wall time/epoch. If the port
operator is actually *more* dynamic compute than the raw graph, that is stated
explicitly ("3.2× compressed" is not claimed from occurrence count alone).

## 7. Training protocol

Arm C reuses Arm A's protocol: same target transform (none; raw MAE), L1 loss,
Adam family, no scheduler, clip 5, **full canonical budget** (≤240 epochs,
patience 40, batch 128 → ~18 960 optimizer steps), evaluation each epoch,
equal-weight **Top-5 checkpoint soup**, seed 0. The previous tiny reader at 6000
steps had not converged; this round reports **converged capacity**, with
train/valid MAE and LR curves sampled every fixed number of epochs. If a
memory-driven batch-size change is ever needed, the optimizer-step budget is held
and recorded. No LR / architecture tuning from official valid.

## 8. Seed discipline

Primary seed **0** for Arms A and C. If `|MAE_PSCD − MAE_raw| ≤ 0.04` (ambiguous
/ potentially-successful), run **seed 1** as paired confirmation. If the gap is
`> 0.08`, no extra seeds are run merely to confirm an obvious failure.

## 9. Data-free tests (must pass before the formal training)

* **A. Batch invariance** — a molecule's prediction is identical individually vs
  batched (`< 1e-5`), avoiding a repeat of the PSCD-R collate defect.
* **B. Molecule permutation invariance** — random atom relabel + refrozen code ⇒
  same prediction.
* **C. Motif automorphism invariance** — canonical-equivalent motif relabeling
  leaves the port-operator output unchanged.
* **D. Empty / single-port / multi-port** synthetic motifs are all finite.
* **E. Port-routing sensitivity** — same motif ID with different port
  configuration can change the output (the reader must not degenerate to a port
  mean).

## 10. Mechanism ablation (only if Arm C is promising)

Only if seed-0 Arm C valid `≤ 0.20`, one extra training with all weights/config
fixed but **no intra-motif transfer** (external messages → center pooling only).
If full ≫ no-transfer, this supports that dynamic intra-motif computation
matters. No further ablations are queued in advance.

## 11. Interpretation gates (frozen before the run)

* **Gate A — strong success:** `MAE_PSCD ≤ 0.16` **and**
  `MAE_PSCD − MAE_raw ≤ 0.04`, with still-reasonable dynamic-state / compute
  compression ⇒ the fixed frequency dictionary preserves most strong-model
  predictive capacity. Do **not** further strengthen the reader; next round is
  task-coupled dictionary learning.
* **Gate B — intermediate:** `0.16 < MAE_PSCD ≤ 0.22` with a clear monotone drop
  from the tiny regime and no capacity-driven gap growth ⇒ representation still
  possibly viable, compressed computation not yet sufficient; allow **one**
  mechanism-oriented reader revision.
* **Gate C — abstraction ceiling:** strong raw reaches ~0.12–0.15 but PSCD stays
  `> 0.22` or `MAE_PSCD − MAE_raw > 0.08` with clearly higher train MAE ⇒
  frequency decomposition imposes a real computational/task abstraction
  bottleneck; stop reader tuning, move to task+MDL motif discovery.
* **Gate D — optimization failure:** train MAE still falling / no plateau in the
  canonical budget ⇒ *optimization inconclusive*; only budget extensions inside
  the canonical range are allowed, no valid-based architecture tuning.
* **Gate E — compression failure:** good MAE but `N_PSCD ≳ N_raw` or higher
  message count / wall time ⇒ "predictive representation works, but
  computational compression is not yet demonstrated".

## 12. Final report must answer

Q1 canonical strong raw (execution regime) · Q2 decode equality under the same
strong model · Q3 fixed-PSCD strong-reader ceiling · Q4 tiny→strong: do raw and
PSCD improve together · Q5 does the PSCD/raw gap shrink / hold / grow with
capacity · Q6 is dynamic intra-motif transfer necessary (only if the ablation
gate fires) · Q7 are PSCD dynamic state / message counts actually smaller ·
Q8 is there now enough evidence to enter task-coupled dictionary learning.

The final conclusion first line is one of the five frozen sentences (strong
capacity preserved / scales but compression incomplete / real abstraction
ceiling / predictive-but-no-compression / inconclusive), and the next decision
is one of: `proceed to task-coupled compositional dictionary learning`,
`perform one final port-operator mechanism revision`,
`replace frequency motif discovery with task+MDL discovery`,
`stop claiming computational compression`, `fix baseline/optimization before
interpretation`. This round does **not** implement the next stage.
