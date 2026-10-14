# Synthetic witnesses — cscl-bond-random-walk-organization-v1

- protocol `cscl-bond-random-walk-organization-v1` · commit `10496f29243c3a55ca7cb123d00c8322f1c55e4f`
- all witnesses are CPU-only and label-blind; no molecule target y is used

## C1 — Junction-moment counterexample

- legal molecule: **False** (abstract incidence systems, not realisable as molecules)
- equal raw marginals: bond multiset / junction multiset / per-bond junction counts = [4, [0, 0, 1, 1], [2, 2, 2, 2]]
- K max abs difference between the two organisations: **0.2500**
- line-graph edges: A 4, B 2

The historical (Null) Junction summary is unchanged while the bond random-walk
relation input differs; this is the requested controlled counterexample and it
is explicitly **not** a legal molecule.

## C1 — Synthetic relation task

- target: `sum_ef K_ef (b_e.b_f) + 0.1*sum_e b_e ; b fixed random 8-D`
- test MAE, real-K model: **0.3755**
- same model evaluated with uniform K: 23.9834
- uniform-K trained model: 3.8388
- linear-summary model: 7.6431

if the real-K model beats both the uniform-K control and the linear summary, the pair channel uses actual pair organisation; this is a controlled relation target, not evidence about chemistry

## C2 — Pair MLP joint information

- random-phi block ablation: zero product rel change 0.507; zero diff rel 0.682; zero sum rel 0.783
- linear-summary relative residual (trained phi): **0.5226**
- trained-phi block ablation: zero product rel 0.762, zero diff rel 2.196
- relation reassignment mean prediction delta (uniform / row-perm / bond-perm): 0.2715 / 0.1569 / 0.1533

The linear-surrogate fit `r_e ~ W (K b)_e + c*rowsum_e` captures the entire
K-dependence of any linear reweighting of `b_f`; a clearly positive relative
residual means the trained pair channel is not reducible to that form.
