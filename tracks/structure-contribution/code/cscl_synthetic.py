"""Synthetic ground-truth task for CSCL-v0 (v0_protocol §6.3).

Real RINGCHAIN-v0 structures, artificial targets with **known** effects:

    y_syn(G) = Σ_t θ_t · count_t(G) + Σ_{(t,t')∈P+} θ_{tt'} · 1[t,t' both in G]
               + ε,      ε ~ N(0, σ²)

* θ_t: per-type unary effects (fit types only, sampled with fixed seed),
  mixed signs.
* P+: a fixed set of type pairs with **real interactions** (nonzero θ).
* Trap: pairs that strongly co-occur but have **zero** interaction are the
  false-interaction control — a model may only score them if it confuses
  co-occurrence with interaction.

Diagnostics returned: direction recovery of α vs θ, interaction detection
(|γ| on P+ vs co-occurring-zero pairs), and transfer to dev molecules whose
pair combination is unseen at fit time.
"""

from __future__ import annotations

import numpy as np

N_INTERACTING_PAIRS = 12
N_TRAP_PAIRS = 12
NOISE_STD = 0.1
UNARY_SCALE = 0.35
INTERACT_SCALE = 0.5


def build_synthetic_effect(
    fit_mols,  # list[MolUnits]: fit_inner molecules (effect/pair selection)
    all_mols,  # list[MolUnits]: all molecules to score (fit+monitor+dev)
    fit_stats,  # FitStats (vocab fitted)
    seed: int = 20261010,
) -> dict:
    """Draw θ_t / θ_tt' over fit-known types; compute y_syn for every molecule.

    θ and the real/trap pair sets are selected on **fit molecules only**; the
    returned y is then scored for all molecules with the frozen effects.
    """
    rng = np.random.RandomState(seed)
    known = sorted(fit_stats.vocab.known_sigs)
    tids = [fit_stats.vocab.sig_to_id[s] for s in known]

    theta = {t: float(v) for t, v in zip(tids, rng.randn(len(tids)) * UNARY_SCALE)}

    # candidate pairs: top-frequency types (present in enough FIT molecules)
    presence: dict[int, int] = {t: 0 for t in tids}
    cooc: dict[tuple[int, int], int] = {}
    for m in fit_mols:
        tset = {fit_stats.type_id(m, k) for k in range(len(m.unit_sigs))}
        for t in tset:
            if t in presence:
                presence[t] += 1
        tl = sorted(t for t in tset if t in presence)
        for i in range(len(tl)):
            for j in range(i + 1, len(tl)):
                cooc[(tl[i], tl[j])] = cooc.get((tl[i], tl[j]), 0) + 1

    frequent = [t for t in tids if presence[t] >= 30]
    freq_sorted = sorted(frequent, key=lambda t: -presence[t])
    pool = [t for t in freq_sorted[:40]]
    pair_pool = sorted({tuple(sorted((a, b))) for i, a in enumerate(pool) for b in pool[i + 1 :]})
    # real interactions: highest-co-occurrence pairs (hardest case: strong
    # co-occurrence AND true interaction), traps: also high co-occurrence but
    # zero effect
    by_cooc = sorted(pair_pool, key=lambda p: -cooc.get(p, 0))
    real_pairs = [p for p in by_cooc[:N_INTERACTING_PAIRS]]
    trap_pairs = [p for p in by_cooc[N_INTERACTING_PAIRS : N_INTERACTING_PAIRS + N_TRAP_PAIRS]]
    theta_pair = {p: float(v) for p, v in zip(real_pairs, rng.randn(len(real_pairs)) * INTERACT_SCALE)}

    y = np.zeros((len(all_mols),), dtype=np.float32)
    for i, m in enumerate(all_mols):
        tset = [fit_stats.type_id(m, k) for k in range(len(m.unit_sigs))]
        acc = sum(theta.get(t, 0.0) for t in tset)
        ts = set(tset)
        for p, w in theta_pair.items():
            if p[0] in ts and p[1] in ts:
                acc += w
        y[i] = acc + rng.randn() * NOISE_STD

    return {
        "theta": theta,
        "theta_pair": theta_pair,
        "trap_pairs": trap_pairs,
        "cooc_fit": {f"{a}_{b}": c for (a, b), c in cooc.items() if c >= 20},
        "y": y,
        "noise_std": NOISE_STD,
    }
