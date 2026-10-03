# METHOD_CONTRACT — zinc-overnight-interface-and-tail-seed0-v1

## 1. What the reference `B` keeps

`B` is the compressed `S_M` deployment model (`S_M_deploy_state.pt`,
267,611 parameters, verified here to reproduce the full `S_M` raw prediction on
all 2000 dev rows to `1.9e-06`).  It **keeps**:

* the shared task dictionary bridge `D_L [144,288]`, `V_L [288,144]`, `rho`
  thresholds and 16-step ISTA code;
* `Sem108` (atom-shell `3x28` + bond-shell `6x4`) and `size2`;
* the static relation path (`pair_projection`, relation encoder, distance
  gate, pair encoder), unary/pair moments, global C6 context and topology25;
* the C6 mask (chemistry histograms zeroed) exactly as in the source arms.

It **removes** the structural slot path: `D`, `U`, IHT coding, the four
binding projections, both slot encoders and the 446-D fusion input.  Removing
the path is not "no dictionary": the task dictionary and all statistical
inputs remain.

## 2. What the new interface reads

* the real structural code from `S_M`'s trained `D_star`/`U` (never the
  compressed zero placeholder): sparse tied-IHT `alpha` or dense
  `kappa * (r @ Dbar)` with a re-estimated `kappa` that matches the sparse
  alpha RMS on the 8000 fit rows;
* direct structure `(x)` category joint tensors built from the *existing*
  node/bond occurrences (`env_occ_*`, `env_bond_*`), root/shell and
  root/shellpair buckets, atom one-hot (28) and bond one-hot (4):
  `K_A` 3x32x28 and `K_E` 6x3x32x4 per molecule, from the formulas in
  PROTOCOL.md section 1;
* `K^M` is the *marginal* control: it keeps both multisets but breaks the
  within-bucket pairing `a_v <-> t_v` (node) or `g_uv <-> bond type` (edge).
  It does **not** remove the pairing of structure codes between a bond's two
  endpoints (`g_uv` keeps `u+v`, `|u-v|`, `u*v`), and it does not remove
  Sem108's own structural/statistical content.  `n=1` gives `J=M`, `n=0`
  gives zeros (verified on synthetic and real batches).
* all 5102 inputs enter one linear layer `Linear(5102,24) -> SiLU ->
  Linear(24,144)`; the two matrices are the only new trainable parameters.
  There is no per-occurrence trainable projection and no graph-level head.

## 3. Where the adapter sits and what can be trained

* Insertion is after `fusion` (the same `h0` that `B` computes from Sem110) and
  before `local_dictionary_bridge`; the bridge and the whole static aggregation
  run exactly once.
* Phase 1: parent and `D` frozen and eval; only the 126,072 adapter parameters
  are in the optimizer; loss is mean `L1(y)` only, so the reconstruction term
  is a constant and is not in the loss.
* Phase 2/3: body, `D` and adapter are registered in the same Adam; loss is
  `L1(y) + H1_LAMBDA * reconstruction` (the source formula:
  residual numerator / raw-`phi` denominator; dense uses the physical
  coefficient `alpha/kappa`).  `A0` multiplies the adapter output by zero (not
  detach/removal) so its parameters keep zero task-grad tensors and the same
  coupled-L2/clip treatment as the other arms.
* `tau_A/tau_E/kappa`, `U` and the input transforms stay frozen in all phases;
  `D_star` initialises every arm.  Phase 3 retrains from the same compressed
  8k start on all 10,000 train rows (the 8k-fit input transforms are kept
  deliberately; this is not a fresh all-data preprocessing).

## 4. Relation to luyin19 and what is *not* delivered

luyin19's strongest route is a structure-attribute joint statistic whose
transfer value comes from a shared dictionary: structure codes learned once
and reused to code unseen graphs, with structure and attribute paired (not
separately aggregated).  This round tests exactly the *inference increment* of
that idea in the smallest concrete form: a direct linear readout of
`struct_code (x) category` joint statistics, inserted into the existing
reference **without** touching Sem108 or the task dictionary.

Not delivered / not comparable to luyin19:

* no learnable per-occurrence structure/attribute projection — the readout is
  one linear layer over the fixed joint tensors;
* no OMP/reconstruction transfer objective on the structural dictionary in
  Phase 1 (it is frozen there); Phase 2 uses the source reconstruction loss
  only as an auxiliary;
* no claim about "structure-attribute alignment" mechanisms beyond the
  measured J-vs-M and sparse-vs-dense contrasts;
* luyin19 contains no performance claim for this interface, so no numeric
  target from luyin19 is used; the only performance bar is the
  pre-registered one.

## 5. Extra supervision disclosure (`g/c`)

The CPU tail branch uses the training-side decomposition labels
`c = y - g` as its head target (`q_U` unweighted, `q_B` severity-balanced).
`g` supervises nothing else here; the frozen `h_raw` is the released
`O_seed0` g-predictor output and is not retrained.  Inference is
`P = h_raw + q(T25) + b_P` and **never reads `c`**.  Results on this branch
are not single-scalar-`y` supervision and are reported separately from the
main interface arms (which use `y` only).

## 6. Parameters, gradients and identity checks

* measured: body 267,611 (`B`) + `D` 2,080 + adapter 126,072 = 395,763;
  original Full was 408,651.
* Phase 0 verifies: adapter output exactly zero at init and B reproduction
  (`<= 1e-05`), node/edge `J != M` on real batches, `M` invariant to category
  permutation while `J` changes, `n=1 J=M`, `n=0` zero, graph-order map and
  label-shuffle invariance, adapter-only Phase-1 optimizer (126,072) with
  step-1 upstream gradient zero and step-2 alive, and all 395,763 parameters
  registered in the Phase-2 optimizer with live `D` reconstruction gradient.
* Phase 1 also checks the parent state hash is byte-identical after smoke
  training.  `A0` keeps `D`'s reconstruction gradient even though its task
  path is inactive; this coupling to global clipping is recorded.
