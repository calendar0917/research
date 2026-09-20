#!/usr/bin/env python
"""PSCD-TMDL-v0 — Task + MDL Motif Discovery Diagnostic.

Question
--------
Frequency graph-BPE selects each merge rule by raw occurrence frequency, which
optimises recurrence/compression but not downstream prediction.  PSCD-SC-v0
showed that a *strong* reader on the resulting frozen frequency dictionary does
NOT follow the raw graph (valid MAE 0.263 vs raw 0.136) while fitting train
better -- a representation-induced abstraction ceiling.  This round changes
**only the dictionary-selection rule** and asks whether task signal can select
more task-relevant, still reusable/compressive motif boundaries:

    D_freq : c* = argmax_c f_occ(c)                      (matched control)
    D_TM   : MDL/reuse defines feasibility, task residual selects among them
    D_task : task residual alone (cheap diagnostic, no strong reader)

Discipline
----------
* Only official ZINC ``train`` (10 000) / ``valid`` (1 000) are loaded.  The
  official ``test`` split is never read, instantiated or referenced.
* Dictionary discovery never sees the official valid split: canonical train is
  split deterministically (seed 0) into 9 000 discovery + 1 000 internal-monitor
  graphs; only the 9 000 discovery graphs drive **every** merge choice.
* All dictionaries start from the same singleton partition, max motif size 8,
  exactly 64 merge steps, no K / max-size / port-definition sweep.
* Motif identity is the exact attributed-graph canonical key (pynauty); no
  handcrafted chemistry, no overlapping motifs, no Gumbel tokenizer.
* Reader architecture / ports / vocabulary budget are frozen in Stage B; the
  only variable is the dictionary-selection rule.

Stages
------
* Stage A: cheap audit of D_freq / D_task / D_TM (Ridge probe, reuse,
  compression, boundary complexity, exactness, dictionary similarity).
* Stage B: two strong port-operator readers (B-FREQ, B-TM) trained on the full
  10 000 canonical train graphs with the frozen dictionaries.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (  # noqa: E402
    MAX_MOTIF,
    MotifType,
    Vocabulary,
    _q,
    aggregate_motif_size,
    apply_rule,
    atom_to_token,
    build_structured_code,
    compression_metrics,
    decode,
    enumerate_candidates,
    graph_bonds,
    init_tokens,
    learn_vocabulary,
    load_mols_and_y,
    mdl_proxy,
    mol_iso_key,
    mol_rank,
    motif_orbits,
    port_complexity,
    tokenize_frozen,
    vocabulary_reuse,
)

# ---------------------------------------------------------------------------
# Frozen configuration (no sweep)
# ---------------------------------------------------------------------------
ROUND = "PSCD-TMDL-v0"
SEED = 0
N_TRAIN = 10_000
N_DISCOVERY = 9_000
N_MONITOR = 1_000
MERGE_STEPS = 64
MAX_SIZE = MAX_MOTIF  # 8
MIN_GRAPH_SUPPORT = 45          # ~0.5% of discovery-9000
MDL_TOP = 32                    # max eligible candidates kept for D_TM
TASK_ONLY_TOP = 256             # D_task candidate shortlist by graph support
RIDGE_ALPHA = 1.0
STD_EPS = 1e-8
TIE_REL = 0.01                  # 1% relative tie window
MONITOR_EVERY = 8

RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pscd_tmdl"


def _torch():
    import torch

    return torch


def _sklearn_ridge():
    from sklearn.linear_model import Ridge

    return Ridge


# ===========================================================================
# 0. Deterministic discovery / monitor split
# ===========================================================================
def discovery_split(n: int = N_TRAIN, n_disc: int = N_DISCOVERY, seed: int = SEED):
    """Deterministic split of the canonical train set (seed-fixed permutation)."""
    n_disc = min(n_disc, n - 1)
    perm = np.random.default_rng(seed).permutation(n)
    disc = np.sort(perm[:n_disc])
    mon = np.sort(perm[n_disc:])
    return disc.tolist(), mon.tolist()


# ===========================================================================
# 1. First-order two-part description-length (MDL) proxy
# ===========================================================================
def build_symbol_tables(disc_codes, disc_mols) -> dict[str, Any]:
    """Empirical edge/port/atom symbol code-lengths from the *current* corpus.

    The PSCD-v0 ``mdl_proxy`` uses an empirical-entropy *symbol* model that
    assigns near-zero cost to the dominant atom symbols; as a per-merge
    objective it cannot express a positive merge gain (a new mesoscale motif is
    charged the unseen-symbol length while the atoms it replaces are near-free).
    For a first-order *merge* description-length change we instead use a
    two-part code in the BPE/two-part tradition: a uniform token-symbol code
    ``log2(K+1)`` plus the empirical edge/port code, with the same dictionary
    cost decomposition as ``mdl_proxy``.  The corpus-level empirical-entropy
    proxy is still reported in Stage A as the compression measure.
    """
    atom_c = Counter(int(x) for mol in disc_mols for x in mol.node_types.tolist())
    bond_c = Counter(int(x) for mol in disc_mols for x in mol.bond_types.tolist())
    atom_lp = {k: -math.log2(v / sum(atom_c.values())) for k, v in atom_c.items()}
    bond_lp = {k: -math.log2(v / sum(bond_c.values())) for k, v in bond_c.items()}
    port_c: dict[int, Counter] = defaultdict(Counter)
    for c in disc_codes:
        for (i, pi, t, j, pj) in c.comp_edges:
            port_c[c.occ_motif_ids[i]][pi] += 1
            port_c[c.occ_motif_ids[j]][pj] += 1
    port_lp = {k: {s: -math.log2(v / sum(cnt.values())) for s, v in cnt.items()}
               for k, cnt in port_c.items()}
    return {"atom_lp": atom_lp, "bond_lp": bond_lp, "port_lp": port_lp}


def graph_cost_uniform(code, tables, sym_bits: float) -> float:
    """Two-part graph code length: uniform token symbols + empirical edges/ports."""
    cost = sym_bits * len(code.occ_motif_ids)
    for (i, pi, t, j, pj) in code.comp_edges:
        cost += tables["bond_lp"].get(int(t), 8.0)
        cost += tables["port_lp"].get(code.occ_motif_ids[i], {}).get(pi, 3.0)
        cost += tables["port_lp"].get(code.occ_motif_ids[j], {}).get(pj, 3.0)
    return cost


def dict_cost_single(size: int, canon_nt, canon_bonds, tables, size_counter,
                     n_learned: int) -> float:
    """Marginal dictionary cost of one new motif (add-one smoothed size prior)."""
    cost = -math.log2((size_counter.get(int(size), 0) + 1) / (n_learned + 1))
    for c in canon_nt:
        cost += tables["atom_lp"].get(int(c), 8.0)
    for (_a, _b, t) in canon_bonds:
        cost += tables["bond_lp"].get(int(t), 8.0)
    cost += size * (size - 1) // 2
    return cost


def _register_temp(vocab: Vocabulary, key: bytes, size: int, cn, cb, mid: int) -> None:
    mt = MotifType(mid=mid, key=key, size=size, canon_nt=tuple(cn),
                   canon_bonds=tuple(tuple(b) for b in cb),
                   orbits=motif_orbits(cn, cb), singleton=False)
    vocab.by_key[key] = mt
    vocab.by_id[mid] = mt


def mdl_delta_candidate(key: bytes, size: int, cn, cb, *, sym_bits: float,
                        vocab: Vocabulary, disc_tokens, disc_mols, disc_ranks,
                        per_graph, key_graphs, base_cost, tables,
                        size_counter, n_learned) -> tuple[float, int]:
    """First-order merge description-length change ``L_after - L_before``.

    Only graphs actually touched by the merge contribute.  ``ΔL < 0`` means the
    merge genuinely shortens the dictionary+corpus code.
    """
    mid = len(vocab)
    _register_temp(vocab, key, size, cn, cb, mid)
    try:
        delta = dict_cost_single(size, cn, cb, tables, size_counter, n_learned)
        merges = 0
        for gi in sorted(set(key_graphs[key])):
            toks2, nm = apply_rule(list(disc_tokens[gi]), per_graph[gi], key, disc_ranks[gi])
            merges += nm
            if nm == 0:
                continue
            code2 = build_structured_code(disc_mols[gi], toks2, vocab, disc_ranks[gi])["code"]
            delta += graph_cost_uniform(code2, tables, sym_bits) - base_cost[gi]
    finally:
        del vocab.by_key[key]
        del vocab.by_id[mid]
    return float(delta), int(merges)


# ===========================================================================
# 2. Candidate enumeration + Ridge task residual
# ===========================================================================
def enumerate_step(disc_tokens, disc_mols, disc_bonds, vocab, introduced):
    """All adjacent-token candidates: per-graph list, f_occ, f_graph, examples."""
    f_occ: Counter[bytes] = Counter()
    key_graphs: dict[bytes, list[int]] = defaultdict(list)
    example: dict[bytes, tuple[int, Any, Any]] = {}
    per_graph = []
    for gi, mol in enumerate(disc_mols):
        idx = atom_to_token(disc_tokens[gi])
        cands = enumerate_candidates(disc_tokens[gi], disc_bonds[gi], mol.node_types, idx,
                                     max_motif=MAX_SIZE)
        per_graph.append(cands)
        for (_ta, _tb, atoms, key, cn, cb) in cands:
            if key in introduced:
                continue
            f_occ[key] += 1
            key_graphs[key].append(gi)
            if key not in example:
                example[key] = (len(atoms), cn, cb)
    f_graph = {k: len(set(v)) for k, v in key_graphs.items()}
    return per_graph, f_occ, f_graph, key_graphs, example


def enumerate_monitor(mon_tokens, mon_mols, mon_bonds):
    per_graph = []
    for gi, mol in enumerate(mon_mols):
        idx = atom_to_token(mon_tokens[gi])
        per_graph.append(enumerate_candidates(mon_tokens[gi], mon_bonds[gi],
                                              mol.node_types, idx, max_motif=MAX_SIZE))
    return per_graph


def motif_count_features(tokens_list, vocab: Vocabulary) -> np.ndarray:
    """``x_G[k] = # occurrences of current motif/atom type k``."""
    K = len(vocab)
    X = np.zeros((len(tokens_list), K), dtype=np.float64)
    by_key = vocab.by_key
    for i, toks in enumerate(tokens_list):
        row = X[i]
        for t in toks:
            row[by_key[t.key].mid] += 1.0
    return X


def features_from_codes(codes, vocab: Vocabulary) -> np.ndarray:
    K = len(vocab)
    X = np.zeros((len(codes), K), dtype=np.float64)
    for i, c in enumerate(codes):
        for k in c.occ_motif_ids:
            X[i, k] += 1.0
    return X


def fit_ridge(X: np.ndarray, y: np.ndarray):
    Ridge = _sklearn_ridge()
    model = Ridge(alpha=RIDGE_ALPHA)
    model.fit(X, y)
    pred = model.predict(X)
    return model, y - pred


def ridge_mae(model, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(model.predict(X) - y).mean())


def task_score(key: bytes, key_graphs, residual: np.ndarray, n_graphs: int) -> float:
    z = np.zeros(n_graphs, dtype=np.float64)
    np.add.at(z, np.asarray(key_graphs[key], dtype=np.int64), 1.0)
    zt = (z - float(z.mean())) / (float(z.std()) + STD_EPS)
    return abs(float(np.mean(residual * zt)))


# ===========================================================================
# 3. Dictionary discovery (three arms)
# ===========================================================================
def discover(mode: str, disc_mols, disc_bonds, disc_ranks,
             mon_mols, mon_bonds, mon_ranks,
             y_disc: np.ndarray, y_mon: np.ndarray,
             steps: int = MERGE_STEPS, log=print) -> dict[str, Any]:
    """Learn one dictionary under ``mode`` in ``{"task", "tmdl"}``."""
    vocab = Vocabulary()
    disc_tokens = [init_tokens(vocab, m) for m in disc_mols]
    mon_tokens = [init_tokens(vocab, m) for m in mon_mols]
    introduced: set[bytes] = set()
    rounds: list[dict[str, Any]] = []
    monitor_curve: list[dict[str, Any]] = []
    n_disc = len(disc_mols)

    for r in range(1, steps + 1):
        disc_per_graph, f_occ, f_graph, key_graphs, example = enumerate_step(
            disc_tokens, disc_mols, disc_bonds, vocab, introduced)
        mon_per_graph = enumerate_monitor(mon_tokens, mon_mols, mon_bonds)

        # --- current-dictionary Ridge probe + residual (discovery only) ----
        Xd = motif_count_features(disc_tokens, vocab)
        Xm = motif_count_features(mon_tokens, vocab)
        ridge, residual = fit_ridge(Xd, y_disc)
        disc_mae = float(np.abs(residual).mean())
        mon_mae = ridge_mae(ridge, Xm, y_mon)

        if not f_occ:
            log(f"[{mode}] r{r:02d}: no candidates left; stopping at {r - 1} merges")
            break

        mdl_gain: dict[bytes, float] = {}
        support_ok = [k for k in f_occ if f_graph.get(k, 0) >= MIN_GRAPH_SUPPORT]
        forced_fallback = False

        if mode == "task":
            pool = sorted(support_ok, key=lambda k: (-f_graph[k], -f_occ[k], k))[:TASK_ONLY_TOP]
        elif mode == "tmdl":
            disc_codes = [build_structured_code(m, disc_tokens[gi], vocab, disc_ranks[gi])["code"]
                          for gi, m in enumerate(disc_mols)]
            tables = build_symbol_tables(disc_codes, disc_mols)
            sym_bits = math.log2(len(vocab) + 1)
            base_cost = [graph_cost_uniform(c, tables, sym_bits) for c in disc_codes]
            learned = [mt for mt in vocab.by_id.values() if not mt.singleton]
            size_counter = Counter(mt.size for mt in learned)
            n_learned = len(learned)
            deltas: list[tuple[float, bytes]] = []
            for k in support_ok:
                size, cn, cb = example[k]
                d, _merges = mdl_delta_candidate(
                    k, size, cn, cb, sym_bits=sym_bits, vocab=vocab,
                    disc_tokens=disc_tokens, disc_mols=disc_mols,
                    disc_ranks=disc_ranks, per_graph=disc_per_graph,
                    key_graphs=key_graphs, base_cost=base_cost, tables=tables,
                    size_counter=size_counter, n_learned=n_learned)
                mdl_gain[k] = -d
                deltas.append((-d, k))
            compressive = [e for e in deltas if e[0] > 0.0]
            forced_fallback = not compressive
            use = compressive if compressive else deltas
            use = sorted(use, key=lambda e: (-e[0], -f_graph[e[1]], -f_occ[e[1]], e[1]))[:MDL_TOP]
            pool = [e[1] for e in use]
            if forced_fallback:
                log(f"[{mode}] r{r:02d}: no ΔL<0 candidate; forced MDL-best fallback")
        else:
            raise ValueError(mode)

        scores = {k: task_score(k, key_graphs, residual, n_disc) for k in pool}
        smax = max(scores.values())
        near = [k for k, s in scores.items() if s >= smax * (1.0 - TIE_REL) - 1e-15]
        if mode == "tmdl":
            key = min(near, key=lambda k: (-mdl_gain[k], -f_graph[k], -f_occ[k], k))
        else:
            key = min(near, key=lambda k: (-f_graph[k], -f_occ[k], k))

        size, cn, cb = example[key]
        merges_disc = 0
        for gi in range(n_disc):
            disc_tokens[gi], nm = apply_rule(disc_tokens[gi], disc_per_graph[gi],
                                             key, disc_ranks[gi])
            merges_disc += nm
        for gi in range(len(mon_mols)):
            mon_tokens[gi], _ = apply_rule(mon_tokens[gi], mon_per_graph[gi],
                                           key, mon_ranks[gi])
        vocab.add(key, size, cn, cb, singleton=False)
        introduced.add(key)

        rec = {
            "round": r, "key": key.hex(), "size": int(size),
            "node_types": list(cn), "bonds": [list(b) for b in cb],
            "occurrence_freq": int(f_occ[key]), "graph_support": int(f_graph[key]),
            "merges": int(merges_disc),
            "task_score": float(scores[key]),
            "mdl_gain": (None if mode != "tmdl" else float(mdl_gain.get(key, 0.0))),
            "delta_mdl": (None if mode != "tmdl" else float(-mdl_gain.get(key, 0.0))),
            "n_support_ok": int(len(support_ok)),
            "n_compressive": (None if mode != "tmdl" else int(len(
                [1 for k in support_ok if mdl_gain.get(k, -1.0) > 0.0]))),
            "forced_fallback": bool(forced_fallback),
            "disc_ridge_mae": float(disc_mae), "mon_ridge_mae": float(mon_mae),
        }
        rounds.append(rec)
        if r % MONITOR_EVERY == 0 or r == steps:
            monitor_curve.append({"round": r, "disc_ridge_mae": float(disc_mae),
                                  "mon_ridge_mae": float(mon_mae)})
        log(f"[{mode}] r{r:02d} size={size} occ={f_occ[key]} sup={f_graph[key]} "
            f"merges={merges_disc} task={rec['task_score']:.4f} "
            f"disc={disc_mae:.4f} mon={mon_mae:.4f}")

    sequence = [bytes.fromhex(r["key"]) for r in rounds]
    disc_codes = [build_structured_code(m, tokenize_frozen(m, vocab, sequence, rk), vocab, rk)["code"]
                  for m, rk in zip(disc_mols, disc_ranks)]
    mon_codes = [build_structured_code(m, tokenize_frozen(m, vocab, sequence, rk), vocab, rk)["code"]
                 for m, rk in zip(mon_mols, mon_ranks)]
    return {"mode": mode, "vocab": vocab, "sequence": sequence, "rounds": rounds,
            "monitor_curve": monitor_curve, "disc_codes": disc_codes,
            "mon_codes": mon_codes}


# ===========================================================================
# 4. Stage A audit
# ===========================================================================
def exact_reconstruction(vocab, mols, codes) -> dict[str, Any]:
    exact = sum(1 for mol, code in zip(mols, codes)
                if mol_iso_key(decode(code, vocab)) == mol_iso_key(mol))
    return {"exact": int(exact), "n": len(mols), "frac": exact / max(len(mols), 1)}


def atom_bond_composition(codes, vocab) -> dict[str, Any]:
    atoms: Counter[int] = Counter()
    bonds: Counter[int] = Counter()
    for c in codes:
        for k in c.occ_motif_ids:
            mt = vocab.by_id[k]
            if mt.singleton:
                continue
            atoms.update(int(x) for x in mt.canon_nt)
            bonds.update(int(t) for (_a, _b, t) in mt.canon_bonds)
    return {"atom_category_counts": {int(k): int(v) for k, v in sorted(atoms.items())},
            "bond_category_counts": {int(k): int(v) for k, v in sorted(bonds.items())}}


def reuse_summary(codes, vocab) -> dict[str, Any]:
    reuse = vocabulary_reuse(codes, vocab)
    supports = sorted((int(v) for v in reuse["motif_support"].values()), reverse=True)
    occs = sorted((int(v) for v in reuse["motif_occ"].values()), reverse=True)
    return {
        "learned_types": reuse["learned_types"], "used_types": reuse["used_types"],
        "unused_types": reuse["unused_types"], "low_freq_lt10": reuse["low_freq_lt10"],
        "low_support_lt45": int(sum(1 for s in supports if s < MIN_GRAPH_SUPPORT)),
        "support_mean": float(np.mean(supports)) if supports else 0.0,
        "support_median": float(np.median(supports)) if supports else 0.0,
        "support_min": int(supports[-1]) if supports else 0,
        "support_max": int(supports[0]) if supports else 0,
        "occ_mean": float(np.mean(occs)) if occs else 0.0,
        "support_q": _q([float(s) for s in supports]),
        "occ_q": _q([float(o) for o in occs]),
        "top_coverage": reuse["top_coverage"], "support_hist": reuse["support_hist"],
    }


def motif_size_summary(codes, vocab) -> dict[str, Any]:
    agg = aggregate_motif_size(codes, vocab)
    sizes = Counter(mt.size for mt in vocab.by_id.values() if not mt.singleton)
    return {"occ_size_q": agg["occ_size"], "occ_size_hist": agg["hist"],
            "type_size_hist": {int(k): int(v) for k, v in sorted(sizes.items())},
            "atom_frac_size_ge2": agg["atom_frac_size_ge2"]}


def stage_a_audit(name, vocab, disc_mols, mon_mols, disc_codes, mon_codes,
                  y_disc, y_mon, n_discovery) -> dict[str, Any]:
    Xd = features_from_codes(disc_codes, vocab)
    Xm = features_from_codes(mon_codes, vocab)
    ridge, residual = fit_ridge(Xd, y_disc)
    ports = port_complexity(list(disc_codes) + list(mon_codes), vocab)
    active_ports = int(sum(len(c.port_slots()) for c in disc_codes))
    return {
        "arm": name, "n_types": len(vocab),
        "n_learned": sum(1 for mt in vocab.by_id.values() if not mt.singleton),
        "exactness": {"discovery": exact_reconstruction(vocab, disc_mols, disc_codes),
                      "monitor": exact_reconstruction(vocab, mon_mols, mon_codes)},
        "ridge": {"discovery_mae": float(np.abs(residual).mean()),
                  "monitor_mae": ridge_mae(ridge, Xm, y_mon),
                  "discovery_monitor_gap": float(ridge_mae(ridge, Xm, y_mon)
                                                 - np.abs(residual).mean())},
        "reuse": reuse_summary(disc_codes, vocab),
        "compression": {
            "discovery": compression_metrics(disc_codes),
            "monitor": compression_metrics(mon_codes),
            "motif_atom_ratio_discovery": compression_metrics(disc_codes)["mean_ratio"],
            "mdl_ratio_total_over_atom": float(
                mdl_proxy(disc_codes, mon_codes, disc_mols, mon_mols, vocab)["ratio_total_over_atom"]),
        },
        "motif_size": motif_size_summary(disc_codes, vocab),
        "ports": {
            "active_ports_total_discovery": active_ports,
            "active_ports_per_occ_mean": ports["p_mean"],
            "active_ports_p95": ports["p_p95"],
            "port_over_size_mean": ports["p_over_size"]["mean"],
            "port_over_size_ge4_mean": ports["p_over_size_ge4_mean"],
            "near_all_ports_frac_ge4": ports["near_all_ports_frac_ge4"],
        },
        "composition": atom_bond_composition(disc_codes, vocab),
        "n_discovery": int(n_discovery),
    }


# ===========================================================================
# 5. Dictionary similarity audit (D_freq vs D_TM)
# ===========================================================================
def dictionary_similarity(freq_vocab, freq_rounds, tm_vocab, tm_rounds) -> dict[str, Any]:
    freq_rules = [r["key"] for r in freq_rounds]
    tm_rules = [r["key"] for r in tm_rounds]
    freq_types = {mt.key.hex() for mt in freq_vocab.by_id.values() if not mt.singleton}
    tm_types = {mt.key.hex() for mt in tm_vocab.by_id.values() if not mt.singleton}
    return {
        "n_rules_freq": len(freq_rules), "n_rules_tm": len(tm_rules),
        "identical_positional_rules": int(sum(1 for a, b in zip(freq_rules, tm_rules) if a == b)),
        "shared_rule_keys": int(len(set(freq_rules) & set(tm_rules))),
        "identical_final_types": int(len(freq_types & tm_types)),
        "freq_only_types": int(len(freq_types - tm_types)),
        "tm_only_types": int(len(tm_types - freq_types)),
        "jaccard_types": float(len(freq_types & tm_types) / max(len(freq_types | tm_types), 1)),
    }


def top_by(rounds, field, top=20, reverse=True):
    return sorted(rounds, key=lambda r: ((-r[field]) if reverse else r[field], r["round"]))[:top]


# ===========================================================================
# 6. Stage B — strong port-operator reader (frozen dictionary)
# ===========================================================================
def build_full_codes(vocab, sequence, mols, ranks):
    return [build_structured_code(
        m, tokenize_frozen(m, vocab, sequence, rk), vocab, rk)["code"]
        for m, rk in zip(mols, ranks)]


def stage_b_train(arm: str, vocab, sequence, device: str, data_root: Path,
                  out: Path, max_epochs: int | None = None,
                  patience: int | None = None, limit: int | None = None,
                  log=print) -> dict[str, Any]:
    from tracks.ksvd.code.run_pscd_strong_reader import (
        MAX_EPOCHS, PATIENCE, category_index, train_seed,
    )
    max_epochs = MAX_EPOCHS if max_epochs is None else max_epochs
    patience = PATIENCE if patience is None else patience
    train_mols, y_train = load_mols_and_y(data_root, "train", limit)
    valid_mols, y_valid = load_mols_and_y(
        data_root, "valid", None if limit is None else max(50, limit // 5))
    log(f"[B:{arm}] train={len(train_mols)} valid={len(valid_mols)} "
        f"(official test never loaded)")
    train_codes = build_full_codes(vocab, sequence, train_mols, [mol_rank(m) for m in train_mols])
    valid_codes = build_full_codes(vocab, sequence, valid_mols, [mol_rank(m) for m in valid_mols])
    rec_tr = exact_reconstruction(vocab, train_mols, train_codes)
    rec_va = exact_reconstruction(vocab, valid_mols, valid_codes)
    if rec_tr["frac"] != 1.0 or rec_va["frac"] != 1.0:
        raise SystemExit(f"[B:{arm}] exactness changed: {rec_tr} {rec_va}")
    atom_index, bond_index = category_index(vocab, train_mols)
    result = train_seed(vocab, atom_index, bond_index, train_codes, valid_codes,
                        np.asarray(y_train), np.asarray(y_valid), device=device,
                        tag=f"B-{arm}", seed=SEED, intra_transfer=True,
                        max_epochs=max_epochs, patience=patience, log=log)
    torch = _torch()
    write_json(out / f"history_B{arm}.json", result["curve"])
    if result["soup_state"] is not None:
        torch.save(result["soup_state"], out / f"B{arm}_top5_soup.pt")
    if result["best_state"] is not None:
        torch.save(result["best_state"], out / f"B{arm}_best_state.pt")
    payload = {"arm": arm, "reconstruction": {"train": rec_tr, "valid": rec_va},
               **result["summary"]}
    write_json(out / f"summary_B{arm}.json", payload)
    return payload


# ===========================================================================
# 7. IO
# ===========================================================================
def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n",
                    encoding="utf-8")


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def provenance(stage, device, args) -> dict[str, Any]:
    return {
        "round": ROUND, "stage": stage,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "device": device, "seed": SEED,
        "config": {
            "n_train": N_TRAIN, "n_discovery": args.n_discovery, "n_monitor": args.n_monitor,
            "merge_steps": args.steps, "max_size": MAX_SIZE,
            "min_graph_support": MIN_GRAPH_SUPPORT, "mdl_top": MDL_TOP,
            "task_only_top": TASK_ONLY_TOP, "ridge_alpha": RIDGE_ALPHA,
            "tie_rel": TIE_REL, "monitor_every": MONITOR_EVERY,
        },
    }


# ===========================================================================
# 8. CLI
# ===========================================================================
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["stageA", "stageB"])
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=MERGE_STEPS)
    parser.add_argument("--n-discovery", type=int, default=N_DISCOVERY)
    parser.add_argument("--n-monitor", type=int, default=N_MONITOR)
    parser.add_argument("--arms", type=str, default="freq,task,tmdl")
    parser.add_argument("--arm", type=str, default=None, choices=["FREQ", "TM", "TASK"])
    parser.add_argument("--artifact", type=Path, default=None,
                        help="stageB frozen dictionary pkl (default: <out>/dictionary_<ARM>.pkl)")
    parser.add_argument("--max-epochs", type=int, default=None, help="stageB smoke override only")
    parser.add_argument("--patience", type=int, default=None, help="stageB smoke override only")
    parser.add_argument("--limit", type=int, default=None, help="stageB smoke override only")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    if args.smoke:
        args.n_discovery = min(args.n_discovery, 400)
        args.n_monitor = min(args.n_monitor, 60)
        args.steps = min(args.steps, 6)

    if args.stage == "stageB":
        arm = args.arm or "TM"
        artifact = args.artifact or (out / f"dictionary_{arm}.pkl")
        if not artifact.exists():
            raise SystemExit(f"missing frozen dictionary artifact: {artifact}")
        with artifact.open("rb") as fh:
            data = pickle.load(fh)
        payload = stage_b_train(arm, data["vocab"], data["sequence"], args.device,
                                args.data_root, out, max_epochs=args.max_epochs,
                                patience=args.patience, limit=args.limit, log=log)
        write_json(out / f"provenance_B{arm}.json", provenance("stageB", args.device, args))
        log(f"[B:{arm}] {json.dumps(payload, default=float)}")
        return 0

    # ---- Stage A -------------------------------------------------------
    log(f"=== {ROUND} Stage A ===")
    n_load = (args.n_discovery + args.n_monitor) if args.smoke else N_TRAIN
    train_mols, y_all = load_mols_and_y(args.data_root, "train", n_load)
    n = len(train_mols)
    disc_idx, mon_idx = discovery_split(n, args.n_discovery)
    disc_mols = [train_mols[i] for i in disc_idx]
    mon_mols = [train_mols[i] for i in mon_idx]
    y_disc = np.asarray([y_all[i] for i in disc_idx], dtype=np.float64)
    y_mon = np.asarray([y_all[i] for i in mon_idx], dtype=np.float64)
    log(f"[A] loaded {n} canonical train; discovery={len(disc_mols)} monitor={len(mon_mols)} "
        f"(official test never loaded)")
    write_json(out / "split_indices.json",
               {"seed": SEED, "n_train": n, "discovery_idx": disc_idx, "monitor_idx": mon_idx})
    disc_bonds = [graph_bonds(m) for m in disc_mols]
    mon_bonds = [graph_bonds(m) for m in mon_mols]
    disc_ranks = [mol_rank(m) for m in disc_mols]
    mon_ranks = [mol_rank(m) for m in mon_mols]

    summary: dict[str, Any] = {"provenance": provenance("stageA", args.device, args)}
    arms = [a for a in args.arms.split(",") if a]
    loaded: dict[str, Any] = {}

    for arm in arms:
        t0 = time.time()
        log(f"[A:{arm}] learning dictionary on discovery-{len(disc_mols)}")
        if arm == "freq":
            vocab = Vocabulary()
            _tokens, rounds = learn_vocabulary(disc_mols, vocab, args.steps, log=log)
            sequence = [bytes.fromhex(r["key"]) for r in rounds]
            common_rounds = [{
                "round": r["round"], "key": r["key"], "size": r["size"],
                "occurrence_freq": r["occurrence_freq"], "graph_support": r["graph_support"],
                "merges": r["merges"], "task_score": None, "mdl_gain": None,
                "delta_mdl": None, "n_support_ok": None, "n_compressive": None,
                "forced_fallback": False, "disc_ridge_mae": None, "mon_ridge_mae": None,
            } for r in rounds]
            data = {"mode": "freq", "vocab": vocab, "sequence": sequence,
                    "rounds": common_rounds, "monitor_curve": []}
        else:
            data = discover(arm, disc_mols, disc_bonds, disc_ranks,
                            mon_mols, mon_bonds, mon_ranks, y_disc, y_mon,
                            steps=args.steps, log=log)
            vocab = data["vocab"]
            sequence = data["sequence"]

        disc_codes = data.get("disc_codes") or [
            build_structured_code(m, tokenize_frozen(m, vocab, sequence, rk), vocab, rk)["code"]
            for m, rk in zip(disc_mols, disc_ranks)]
        mon_codes = data.get("mon_codes") or [
            build_structured_code(m, tokenize_frozen(m, vocab, sequence, rk), vocab, rk)["code"]
            for m, rk in zip(mon_mols, mon_ranks)]
        audit = stage_a_audit(arm, vocab, disc_mols, mon_mols, disc_codes, mon_codes,
                              y_disc, y_mon, len(disc_mols))
        audit["wall_s"] = time.time() - t0
        audit["monitor_curve"] = data.get("monitor_curve", [])
        summary[f"stage_a_{arm}"] = audit
        log(f"[A:{arm}] n_learned={audit['n_learned']} ridge={audit['ridge']}")

        with (out / f"dictionary_{arm.upper()}.pkl").open("wb") as fh:
            pickle.dump({"vocab": vocab, "sequence": sequence, "rounds": data["rounds"]}, fh)
        write_json(out / f"rounds_{arm}.json", data["rounds"])
        write_json(out / f"audit_{arm}.json", audit)
        loaded[arm] = data

    if "freq" in loaded and "tmdl" in loaded:
        sim = dictionary_similarity(loaded["freq"]["vocab"], loaded["freq"]["rounds"],
                                    loaded["tmdl"]["vocab"], loaded["tmdl"]["rounds"])
        sim["top_task_motifs"] = top_by(loaded["tmdl"]["rounds"], "task_score", 20)
        sim["top_frequency_motifs"] = top_by(loaded["freq"]["rounds"], "occurrence_freq", 20)
        summary["dictionary_similarity"] = sim
        write_json(out / "dictionary_similarity.json", sim)
        log(f"[A] similarity jaccard={sim['jaccard_types']} "
            f"shared_rules={sim['shared_rule_keys']} identical_types={sim['identical_final_types']}")

    write_json(out / "SUMMARY_A.json", summary)
    log(f"=== {ROUND} Stage A done ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
