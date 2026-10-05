# METHOD_CONTRACT — zinc_local_dictionary_component_supervision_seed0_v1

Frozen before the two formal training runs. Execution module:
`tracks/ksvd/experiments/luyin16/zinc_local_dictionary_component_supervision_seed0_v1.py`
(importing the verified production sources `prev`=joint_vs_independent,
`mlpmod`=vs_mlp, `zfr`=fresh_fold_replication, `zcs`=chemistry_component_supervision,
`zw`/`zftd`/`cm`/`zcdm`/`zbn`/`zjd`). Any deviation is recorded in EXECUTION.md;
frozen scientific definitions are never changed after dev scores are seen.

## Contracted quantities

| item | contract |
|---|---|
| fold | `rng(20261006).permutation(10000)`, fit = first 8000 sorted (sha `734d27f2…`), dev = rest (sha `cb5f49dc…`); fixed once |
| targets | constants refit on new fit only (MU_LOGP = 2.4570953396190123 fixed); `c=(k−mu_cycle)/sigma_cycle`, `g=y−c`, `ell=(logP−MU_LOGP)/sigma_logP`, `s=g−ell`; float64 identities ≤1e-12 |
| tuple input | `x(v,a,t)=[phi_v(65) std; onehot28(a_v); onehot28(a); onehot4(t)]` (125-D); real J incidence `pair_wJ`; per-tuple encode → incidence pool (root offsets/zero-degree handling per production path) |
| D_COMP | `D_loc_raw[125,64]`, tied IHT 10 steps, hard top-8/64, `eta=1/(1.05·σ(Dbar)²+1e-12)` detached power-iteration σ (frozen start vector, no RNG); init `init_d_loc` frame seed 20261004 |
| M_COMP | `A_raw[64,125]`, `SiLU(x@A_bar.T)`, row-L2; init = `D_init.T` element-wise; no bias/gain/second layer/norm/dropout |
| injection | `fusion0(Sem110) + W_loc @ (kappa·e)`, `W_loc[342,64]=0` init; posterior bridge = matched MLP (144→288→144) in BOTH arms |
| scales | kappa_D = `1/RMS(e_J,e_I)` (fit roots, ≤8192, seed 20261004) = 2.042991; kappa_M = r_D/r_M = 1.983953 (r_D = RMS(kappa_D·e_D_init) = RMS over the frozen sample, not assumed 1) |
| reader | 39→2 half-split of the original 39→1 head (fork_rng); `hat_g = hat_ell + hat_s`; initial sum function = original head (verified 3e-8) |
| loss | `MAE(hat_g,g) + 0.5·(MAE(hat_ell,ell) + MAE(hat_s,s))`, both arms, coefficient fixed |
| params | 297,539 per arm (body 184,707 incl. reader output 80; bridge 82,944; local 8,000 + W_loc 21,888) |
| recipe | seed 0; Adam 1e-3 / wd 1e-5 coupled / clip 5.0; batch 128; 240 epochs = 15,120 steps; schedule seed 101 (hash `7b11a529…`, equal position + gid streams across arms); soup 236–240; checkpoints 1/40/120/240 |
| calibration | `b_g = median(g_fit − raw_soup_fit)` per arm, fit-only; dev never used |
| endpoint | dev G0(k=0) cal g-MAE main; overall cal common gate; raw parallel; gain = MAE(M_COMP) − MAE(D_COMP) |
| bootstrap | paired, 1000×, seed 20261006, same per-graph indices both arms; witnesses same→0/swap→mirror |
| gates | dictionary: G0 cal ≥.003 ∧ overall cal ≥.003 ∧ G0 cal CI lower >0 ∧ G0 raw >0 ∧ overall raw >0 (and no mechanism failure); swapped roles for M; [−.003,+.003] both CIs for LOCAL_EQUIVALENCE |
| interventions | zero injection; fit-roots mean code (same kappa/W_loc); J→existing marginal I incidence; native fit bias; sensitivity-only |
| data | 10,000 official-train only; train-only loader; official valid/test never loaded; no full-train-soup warm start |

## Forbidden

Width/loss-coefficient/sparsity/IHT-step/init/WD/optimizer/fold/seed search;
epoch or last-vs-soup selection by dev; dev calibration; extra arms or seeds;
rescue rules for the dictionary on a failed gate; new features for the J→I
operator check; test/valid access of any kind; cross-regime or cross-fold gain
splicing; "not significant" rewritten as "useless/equivalent"; discarding and
re-running trajectories for a bad loss; pushing/merging.
