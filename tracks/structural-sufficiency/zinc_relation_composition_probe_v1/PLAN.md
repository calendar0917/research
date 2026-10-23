# PLAN — `zinc-relation-composition-probe-v1`

Track: `structural-sufficiency`. Branch: `probe/relation-composition-v1`.
Base commit: `3f3cf0a62486be789e325e0c72ce412885170074` (main, unchanged so far).

**Official test is never constructed, loaded or scored in this round.**

## 0. The one question

Under the *same* explicit structural input, the *same* local encoding, the
*same* parameter scale and the *same* training recipe:

> does letting the **relation state** depend on the current bond-fragment
> representation buy anything over a relation state that only ever depends on
> the fixed structure? And is this new body worth continuing, relative to the
> confirmed `B-R0` baseline?

Two new supervised arms, `D` and `S`, differ in exactly one equation. There is
no third model, no recipe grid, no EMA, no SSL project and no RMC repair in this
round.

## 1. What is new here (and what is not)

| arm | local encoding | token interaction | relation memory | read-out |
|---|---|---|---|---|
| `B` (baseline) | P3 encoders + real-ownership composers, one shot | none | none | molecule-level summary of `c_g`/`c_l` |
| old full-pair / triad | shared one contraction through an intermediate atom | none | static | molecule summary |
| DBT (prior) | directed-bond tokens | 4 layers global attention | relation is a *function of the input structure*; it does **not** become per-token state | pooled tokens |
| **`S` (this round)** | P3 encoders + `c_g`/`c_l` | 4 layers global attention, real `xi` | multi-layer, learnable, `R`-only: `e^{l+1}=BN(e^l+O_e xi_static)` | last `h` + atom/global |
| **`D` (this round)** | same | same, real `xi` | multi-layer, learnable, **`h`-conditioned**: `e^{l+1}=BN(e^l+O_e xi)` | same |

The capability under test is *"current context written into the relation state,
then read back by later layers"* — not "more attention layers", not "higher-order
relations", not "longer range". This is a research adaptation borrowing the
relation-channel idea from GRIT/EGT/GraphiT; no novelty is claimed.

The `D` vs `B` comparison is a **whole-model** comparison (tokenisation,
read-out, normalisation and depth all change together), so it cannot be
attributed to a single change. `D` vs `S` is the mechanism comparison of record.

## 2. Model

Per undirected physical bond `b` — one token, not two directed ones:

```
c_g[b] = group_composer([b, Wg, Vg, log1p n_b, bond_feat])   # 96, B's module
c_l[b] = local_composer([b, Wl, Vl, log1p n_b, bond_feat])   # 96, B's module
h_b^0  = SiLU(W_h [c_g_b, c_l_b] + b_h) + W_abs R_bb
e_bc^0 = W_rel R_bc                     (W_rel, W_abs: 21 -> 48, no bias)
```

The relation comes from `rc_data.py` (see §3 of `rc_data.py` for the proof
obligations):

```
P = diag(deg)^-1 A                     (raw unweighted graph, zero rows zero)
M_{b,i} = 1/sqrt(2) if i is an endpoint of bond b else 0
R_bc[k] = (M P^k M^T)_bc,  k = 0..20
```

`R[0]` is 1 on the diagonal, 0.5 for a single shared endpoint and 0 otherwise;
it is invariant under endpoint flip and equivariant under relabelling (checked).
It is *not* a line-graph random-walk probability.

Each of the 4 layers, block-diagonal inside each molecule, no adjacency mask:

```
t_bc    = Q_l h_b + K_l h_c + c_l                      (c_l learnable, init 1)
E_w/E_b = split(E_l(e_bc))                             (d -> 2d -> per head)
xi_bc   = ReLU( signed_sqrt(t_bc * E_w_bc) + E_b_bc )
alpha   = softmax_c( clamp(Aw_l . xi, -5, 5) )         (attention dropout 0.2)
u_b     = sum_c alpha [ V_l h_c + Ve_l xi_bc ]
h       -> deg scaler [1, log1p(line_deg_b)] -> O_h -> +residual -> BN
h       -> FFN(d -> 2d -> d, SiLU) -> +residual -> BN
signed_sqrt(x) = sign(x) * sqrt(|x| + 1e-6)
```

The **only** D/S difference is the write into the next layer's relation state;
both arms use the *real* `xi` for the token update:

```
D:  e^{l+1} = BN_e( e^l + O_e,l xi_bc )
S:  e^{l+1} = BN_e( e^l + O_e,l xi_static_bc ),
    xi_static_bc = ReLU( signed_sqrt(c_l * E_w_bc) + E_b_bc )     # R-only
```

`S` is not "the relation is fixed": it keeps 4 attention layers, a learnable
relation encoder, a learnable per-layer update and the full edge value path; its
state evolves with depth but, at fixed weights, is a function of `R` alone.
`S` does **not** use `h.detach()`, which would keep the forward context.
Only the first 3 layers register `O_e`/`BN_e`; layer 4 updates `h` only.

One head, no per-graph LayerNorm anywhere:

```
z = [ sum_b h_b / mean_bonds_train, sum_b h_b^2 / mean_bonds_train,
      sum_i a_i / mean_atoms_train, sum_i a_i^2 / mean_atoms_train,
      (global33 - mean)/scale, log1p n_bonds ]         # 48+48+96+96+33+1 = 322
y = MLP(z; 128, 128, SiLU; out 1)
```

`mean_bonds_train = 24.9279`, `mean_atoms_train = 23.1664` (fixed train means,
not per-molecule counts).

Initialisation: encoders, `group_composer` and `local_composer` are copied
tensor-by-tensor from `dcc_models.build_initial_state(norm, seed)` — i.e. B's own
seed-matched shared initial state (verified exact: max diff 0.0 in
`analysis/checks.json`). The encoders equal the train-only masked-SSL checkpoint
(`006fabc1…`, s0 for seed 0, s1 for seed 1). New modules are drawn from the
seed's fixed second RNG stream (`seed + 90731`). The arm flag is the only
difference, so D and S have identical keys, shapes and tensors.

## 3. Parameters

Measured on the real modules (`analysis/param_audit.json`):

| module | params |
|---|---:|
| P3 encoders (fine-tuned, SSL init) | 202,048 |
| `group_composer` (B's) | 62,976 |
| `local_composer` (B's) | 31,536 |
| relation core (`w_h` 9,264 + `w_abs` 1,008 + `w_rel` 1,008 + 4 layers 104,496) | 115,776 |
| reader (322→128→128→1) | 57,985 |
| **total** | **470,321** |

≤ 500,000 and identical for D and S. The brief's design arithmetic said
470,513; the difference is **−192 = 4 × d**, i.e. the estimate counted one extra
d-wide per-layer vector that the declared stack does not instantiate (the last
layer has no relation-write module, and the three `c_l` vectors plus `deg_coef`
are counted exactly once). No width was reduced and no encoder was omitted; the
full ledger is the table above, and `rc_checks.check_numerical` confirms that all
118 parameter tensors are registered, optimiser-covered and receive non-zero,
finite gradients.

`d=40` fallback (declared, used only if timing demands): 434,897 params
(−160 = 4×d behind the brief's 435,057 for the same reason).

## 4. Pre-declared training schedule

One complete, pre-declared schedule; the same for both arms and both seeds:

| item | value |
|---|---|
| data | official ZINC subset, 10,000 train / 1,000 valid (test never loaded) |
| batch | 128 |
| epochs | 600 (79 updates/epoch, 47,400 updates) |
| loss | raw-y L1 |
| optimiser | AdamW, lr 1e-3 → 1e-5 cosine, warmup 500 updates, wd 1e-5, clip 5 |
| dropout | encoders/composers 0, core attention 0.2, rest 0 |
| selection | full valid every epoch, strict best-valid, ties → earliest; final also saved |
| ckpt | best + final only |

Identical init, identical batch order (same generator seed), and the D/S-specific
deterministic work consumes no dropout RNG. This is **not** GRIT's native
2000-epoch / 625k-update recipe and does not claim that the body is fully
trained; it is a declared exploration horizon.

## 5. Cost rule (decided after measurement, before any formal run)

`rc_runner.py measure` runs complete epochs on the full 10k/1k splits and
records train / valid / collate / transfer / save separately. The decision rule,
fixed in advance:

1. d=48 / 4 heads / 4 layers if 4 complete runs (2 seeds × D/S, 2 GPUs) fit the
   3 h wall / 6 GPU·h cap with margin;
2. otherwise d=40 / 4 heads / 4 layers, re-timed once, re-declared;
3. otherwise seed 0's D/S pair only (single-seed boundary stated explicitly);
4. otherwise `COST_NOT_TESTED` — a budget stop, not a mechanism result.

## 6. Decisions after the probe (report answers each)

* was the relation genuinely generated, transported and read by the output?
* is D > S under this schedule, and on how many seeds, with best/W20/T100 agreeing?
* is there any D/S signal against B, and is it in ordinary or tail molecules?
* if not, is it under-training, an unused core, an unprofitable body, or no
  detectable D/S difference — and which of those remain indistinguishable?
* one fixed-weight intervention on the trained D (rule switched to S, full
  re-forward) as a dependence check on the trained model only.
* never a test claim; the 0.05× target is not evaluated in this round.
