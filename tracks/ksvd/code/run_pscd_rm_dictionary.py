#!/usr/bin/env python
"""PSCD-RM-v0 — Rate-Matched Task-Driven Dictionary Audit.

Question
--------
PSCD-TMDL-v0 changed ONLY the dictionary-selection rule (frequency -> task+MDL)
and improved the identical strong reader by ``Delta_dictionary`` ~ 0.058-0.067,
but the task+MDL dictionary was also materially LESS compressive (dynamic-state
ratio 1.392x vs 0.888x raw; occurrence/atom 0.555 vs 0.315).  Two explanations
remained entangled:

    (a) task supervision selects *better* graph boundaries;
    (b) task supervision simply retains *more* fine-grained structure.

This round separates them by running two new dictionary controls **at matched
representation rate**:

* ``FREQ-LOOSE`` — an early prefix of the *frozen frequency BPE trajectory*
  whose ``(occurrences + active ports) / atoms`` budget matches the task+MDL
  dictionary.  Pure decompression control: same frequency algorithm, stopped at
  a lower compression strength (no motif added/removed, no merge reordered).
* ``TM-RATE`` — a fresh task+MDL dictionary learned from singletons under a
  **hard representation-rate constraint** that holds it to the frequency-64
  occurrence / dynamic-state budget.

The four comparison points are FREQ, TM (reused from PSCD-TMDL-v0), FREQ-LOOSE
and TM-RATE.

Discipline (frozen, unchanged from PSCD-TMDL-v0)
------------------------------------------------
* Official ZINC ``test`` is never loaded, instantiated or referenced.
* Canonical train 10 000 / official valid 1 000; deterministic 9 000 discovery +
  1 000 internal-monitor split (seed 0); only the 9 000 discovery graphs drive
  every merge.  The official valid split is touched only after the dictionary is
  frozen.
* Reader architecture / port definition / composition representation / vocabulary
  budget / max motif size / optimizer protocol / checkpoint rule are frozen.
  Exactly 64 merge steps, max motif size 8, no K / max-size / beta / port sweep.
* Motif identity = exact attributed-graph canonical key (pynauty), no hand-coded
  chemistry, no overlapping motifs, no Gumbel / differentiable tokenizer.

Rate definition (matches PSCD-SC-v0 ``r_state``)
------------------------------------------------
For a frozen dictionary tokenisation of the discovery corpus,

    R_occ   = (total occurrences) / (total atoms)
    R_state = (total occurrences + total distinct active ports) / (total atoms)

where an *active port* is a distinct ``(occurrence, canonical slot)`` pair
(identical to ``precompute_graph``'s ``n_port``).

Stages
------
* Stage A: rate curve of the frozen frequency trajectory; ``k*`` selection and
  the FREQ-LOOSE truncation; the rate-constrained TM-RATE discovery; structural
  audits (exactness, invariance, reuse, ports, sizes) for both new arms.
* Stage B: two strong port-operator readers (B-FREQLOOSE, B-TMRATE) trained on
  the full 10 000 canonical train graphs with the frozen dictionaries, using the
  IDENTICAL protocol as PSCD-TMDL-v0 (192 257 params each).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import platform
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (  # noqa: E402
    Mol,
    Vocabulary,
    apply_rule,
    atom_to_token,
    build_structured_code,
    decode,
    enumerate_candidates,
    graph_bonds,
    init_tokens,
    load_mols_and_y,
    mol_iso_key,
    mol_rank,
    tokenize_frozen,
)
from tracks.ksvd.code.run_pscd_tmdl_dictionary import (  # noqa: E402
    MAX_SIZE,
    MERGE_STEPS,
    MIN_GRAPH_SUPPORT,
    MONITOR_EVERY,
    N_DISCOVERY,
    N_MONITOR,
    N_TRAIN,
    ROUND as PARENT_ROUND,
    SEED,
    TIE_REL,
    _register_temp,
    build_symbol_tables,
    dict_cost_single,
    discovery_split,
    enumerate_monitor,
    enumerate_step,
    fit_ridge,
    graph_cost_uniform,
    motif_count_features,
    ridge_mae,
    stage_a_audit,
    stage_b_train,
    task_score,
    write_json,
)

# ---------------------------------------------------------------------------
# Frozen configuration (no sweep)
# ---------------------------------------------------------------------------
ROUND = "PSCD-RM-v0"
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pscd_rm"
PARENT_RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pscd_tmdl"
FREQ_ARTIFACT = PARENT_RESULTS_DIR / "dictionary_FREQ.pkl"
TM_ARTIFACT = PARENT_RESULTS_DIR / "dictionary_TM.pkl"

RATE_EPS = 1e-12


# ===========================================================================
# 0. Rate statistics (discovery corpus)
# ===========================================================================
def corpus_rates(codes) -> dict[str, Any]:
    """Global (corpus-level) occurrence / dynamic-state ratios.

    ``R_occ = M / N`` and ``R_state = (M + P) / N`` with ``P`` the number of
    distinct ``(occurrence, canonical slot)`` port pairs.
    """
    N = int(sum(int(c.n) for c in codes))
    M = int(sum(len(c.occ_motif_ids) for c in codes))
    P = int(sum(sum(len(s) for s in c.port_slots()) for c in codes))
    return {"N": N, "M": M, "P": P,
            "R_occ": M / max(N, 1), "R_state": (M + P) / max(N, 1)}


def codes_for(vocab: Vocabulary, tokens, mols, ranks):
    return [build_structured_code(m, tokens[i], vocab, ranks[i])["code"]
            for i, m in enumerate(mols)]


def _merge_once(vocab, tokens, bonds, mols, ranks, key, gi):
    idx = atom_to_token(tokens[gi])
    cands = enumerate_candidates(tokens[gi], bonds[gi], mols[gi].node_types, idx,
                                 max_motif=MAX_SIZE)
    tokens[gi], _ = apply_rule(tokens[gi], cands, key, ranks[gi])


def frequency_prefix_rate_curve(vocab: Vocabulary, sequence: Sequence[bytes],
                                mols, bonds, ranks, log=print) -> list[dict]:
    """``R_occ`` / ``R_state`` after every prefix ``k = 0..len(sequence)``.

    Replays the FROZEN frequency merge sequence incrementally on the discovery
    corpus (prefix ``k`` is exactly ``tokenize_frozen`` with ``sequence[:k]``).
    """
    tokens = [init_tokens(vocab, m) for m in mols]
    rows: list[dict[str, Any]] = []

    def measure(k: int):
        st = corpus_rates(codes_for(vocab, tokens, mols, ranks))
        return {"k": int(k), **st}

    rows.append(measure(0))
    for k, key in enumerate(sequence, start=1):
        for gi in range(len(mols)):
            _merge_once(vocab, tokens, bonds, mols, ranks, key, gi)
        rows.append(measure(k))
        if k % 8 == 0 or k <= 8:
            log(f"[rateF] k={k:02d} R_occ={rows[-1]['R_occ']:.4f} "
                f"R_state={rows[-1]['R_state']:.4f}")
    return rows


def rates_for_sequence(vocab: Vocabulary, sequence: Sequence[bytes],
                       mols, bonds, ranks) -> dict[str, Any]:
    """Final rate statistics for a frozen (vocab, sequence) on the corpus."""
    tokens = [tokenize_frozen(m, vocab, sequence, rk) for m, rk in zip(mols, ranks)]
    return corpus_rates(codes_for(vocab, tokens, mols, ranks))


def truncate_vocabulary(vocab: Vocabulary, sequence: Sequence[bytes],
                        n_rules: int) -> Vocabulary:
    """Vocabulary of a frozen frequency dictionary stopped after ``n_rules``.

    Singletons keep their original mid ordering; the first ``n_rules`` learned
    motifs are re-added in rule order, so the resulting mid layout is identical
    to the full dictionary restricted to the prefix (no motif is invented,
    removed or reordered).
    """
    new = Vocabulary()
    for mid in range(len(vocab)):
        mt = vocab.by_id[mid]
        if mt.singleton:
            new.add_singleton(int(mt.canon_nt[0]))
    for key in sequence[:n_rules]:
        mt = vocab.by_key[key]
        new.add(key, mt.size, mt.canon_nt, mt.canon_bonds, singleton=False)
    return new


# ===========================================================================
# 1. Rate-constrained task+MDL discovery (TM-RATE)
# ===========================================================================
def candidate_mdl_and_rates(key: bytes, size: int, cn, cb, *, sym_bits: float,
                            vocab: Vocabulary, disc_tokens, disc_mols, disc_ranks,
                            per_graph, key_graphs, base_cost, tables,
                            size_counter, n_learned, cur_ports
                            ) -> tuple[float, int, int]:
    """First-order merge ``ΔL`` plus its *real corpus-level* rate gains.

    Returns ``(delta_mdl, merges_total, port_reduction)`` where the merge is
    simulated on exactly the discovery graphs the rule touches:

        g_occ   = merges_total / N
        g_state = (merges_total + port_reduction) / N

    with ``port_reduction = Σ_g (ports_before(g) - ports_after(g))`` over the
    touched graphs.  This is the true corpus-level rate gain, not a motif-size
    or frequency proxy.
    """
    mid = len(vocab)
    _register_temp(vocab, key, size, cn, cb, mid)
    try:
        delta = dict_cost_single(size, cn, cb, tables, size_counter, n_learned)
        merges = 0
        port_red = 0
        for gi in sorted(set(key_graphs[key])):
            toks2, nm = apply_rule(list(disc_tokens[gi]), per_graph[gi], key,
                                   disc_ranks[gi])
            merges += nm
            if nm == 0:
                continue
            code2 = build_structured_code(disc_mols[gi], toks2, vocab,
                                          disc_ranks[gi])["code"]
            delta += graph_cost_uniform(code2, tables, sym_bits) - base_cost[gi]
            port_red += cur_ports[gi] - sum(len(s) for s in code2.port_slots())
        return float(delta), int(merges), int(port_red)
    finally:
        del vocab.by_key[key]
        del vocab.by_id[mid]


def discover_rate(disc_mols, disc_bonds, disc_ranks,
                  mon_mols, mon_bonds, mon_ranks,
                  y_disc: np.ndarray, y_mon: np.ndarray,
                  r_state_target: float, r_occ_target: float,
                  steps: int = MERGE_STEPS, log=print) -> dict[str, Any]:
    """Task+MDL motif discovery under a hard representation-rate constraint.

    At merge step ``t`` (with ``m = steps - t + 1`` remaining) a candidate is
    admissible iff

        f_graph(c) >= MIN_GRAPH_SUPPORT
        ΔL_MDL(c) < 0
        g_occ(c)   >= max(0, R_occ_current   - R_occ_target)   / m
        g_state(c) >= max(0, R_state_current - R_state_target) / m

    among which the task residual selector picks ``argmax S_task`` (deterministic
    tie-break: larger state rate gain, larger MDL gain, larger graph support,
    larger occurrence frequency, canonical key).  If NO candidate is admissible
    the construction is declared **infeasible** under the frozen 64-rule budget
    and the run stops (no support / beta / K / max-size relaxation).
    """
    vocab = Vocabulary()
    disc_tokens = [init_tokens(vocab, m) for m in disc_mols]
    mon_tokens = [init_tokens(vocab, m) for m in mon_mols]
    introduced: set[bytes] = set()
    rounds: list[dict[str, Any]] = []
    monitor_curve: list[dict[str, Any]] = []
    n_disc = len(disc_mols)
    N = int(sum(int(m.n) for m in disc_mols))
    infeasible: dict[str, Any] | None = None

    for r in range(1, steps + 1):
        disc_per_graph, f_occ, f_graph, key_graphs, example = enumerate_step(
            disc_tokens, disc_mols, disc_bonds, vocab, introduced)
        mon_per_graph = enumerate_monitor(mon_tokens, mon_mols, mon_bonds)

        Xd = motif_count_features(disc_tokens, vocab)
        Xm = motif_count_features(mon_tokens, vocab)
        ridge, residual = fit_ridge(Xd, y_disc)
        disc_mae = float(np.abs(residual).mean())
        mon_mae = ridge_mae(ridge, Xm, y_mon)

        if not f_occ:
            infeasible = {"round": r, "reason": "no_candidates_left",
                          "completed_rounds": r - 1}
            log(f"[rate] r{r:02d}: no candidates left; stopping at {r - 1} merges")
            break

        disc_codes = codes_for(vocab, disc_tokens, disc_mols, disc_ranks)
        cur_ports = [sum(len(s) for s in c.port_slots()) for c in disc_codes]
        M_cur = int(sum(len(c.occ_motif_ids) for c in disc_codes))
        P_cur = int(sum(cur_ports))
        r_occ_cur = M_cur / max(N, 1)
        r_state_cur = (M_cur + P_cur) / max(N, 1)

        m_rem = steps - r + 1
        D_occ = max(0.0, r_occ_cur - r_occ_target)
        D_state = max(0.0, r_state_cur - r_state_target)
        need_occ = D_occ / m_rem
        need_state = D_state / m_rem

        tables = build_symbol_tables(disc_codes, disc_mols)
        sym_bits = math.log2(len(vocab) + 1)
        base_cost = [graph_cost_uniform(c, tables, sym_bits) for c in disc_codes]
        learned = [mt for mt in vocab.by_id.values() if not mt.singleton]
        size_counter = Counter(mt.size for mt in learned)
        n_learned = len(learned)

        support_ok = [k for k in f_occ if f_graph.get(k, 0) >= MIN_GRAPH_SUPPORT]
        adm: list[dict[str, Any]] = []
        n_mdl_ok = 0
        n_rate_ok = 0
        best_g_occ = 0.0
        best_g_state = 0.0
        for k in support_ok:
            size, cn, cb = example[k]
            delta, merges, port_red = candidate_mdl_and_rates(
                k, size, cn, cb, sym_bits=sym_bits, vocab=vocab,
                disc_tokens=disc_tokens, disc_mols=disc_mols,
                disc_ranks=disc_ranks, per_graph=disc_per_graph,
                key_graphs=key_graphs, base_cost=base_cost, tables=tables,
                size_counter=size_counter, n_learned=n_learned, cur_ports=cur_ports)
            g_occ = merges / max(N, 1)
            g_state = (merges + port_red) / max(N, 1)
            mdl_ok = (-delta) > 0.0
            rate_ok = (g_occ >= need_occ - RATE_EPS) and (g_state >= need_state - RATE_EPS)
            if mdl_ok:
                n_mdl_ok += 1
                best_g_occ = max(best_g_occ, g_occ)
                best_g_state = max(best_g_state, g_state)
            if rate_ok:
                n_rate_ok += 1
            if mdl_ok and rate_ok:
                adm.append({"key": k, "size": int(size), "mdl_gain": float(-delta),
                            "g_occ": float(g_occ), "g_state": float(g_state),
                            "merges": int(merges)})

        if not adm:
            infeasible = {
                "round": r, "reason": "rate_constraint_infeasible",
                "completed_rounds": r - 1,
                "R_occ_current": float(r_occ_cur),
                "R_state_current": float(r_state_cur),
                "R_occ_target": float(r_occ_target),
                "R_state_target": float(r_state_target),
                "D_occ": float(D_occ), "D_state": float(D_state),
                "need_occ": float(need_occ), "need_state": float(need_state),
                "n_support_ok": int(len(support_ok)),
                "n_mdl_ok": int(n_mdl_ok), "n_rate_ok": int(n_rate_ok),
                "best_g_occ_among_mdl_ok": float(best_g_occ),
                "best_g_state_among_mdl_ok": float(best_g_state),
            }
            log(f"[rate] r{r:02d}: INFEASIBLE (support={len(support_ok)} "
                f"mdl_ok={n_mdl_ok} rate_ok={n_rate_ok}); "
                f"need_occ={need_occ:.5f} need_state={need_state:.5f} "
                f"best_g_occ={best_g_occ:.5f} best_g_state={best_g_state:.5f}")
            break

        scores = {e["key"]: task_score(e["key"], key_graphs, residual, n_disc)
                  for e in adm}
        smax = max(scores.values())
        near = [e for e in adm if scores[e["key"]] >= smax * (1.0 - TIE_REL) - 1e-15]
        chosen = min(near, key=lambda e: (-e["g_state"], -e["mdl_gain"],
                                          -f_graph[e["key"]], -f_occ[e["key"]],
                                          e["key"]))
        key = chosen["key"]
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
            "mdl_gain": float(chosen["mdl_gain"]),
            "delta_mdl": float(-chosen["mdl_gain"]),
            "g_occ": float(chosen["g_occ"]), "g_state": float(chosen["g_state"]),
            "R_occ_before": float(r_occ_cur), "R_state_before": float(r_state_cur),
            "need_occ": float(need_occ), "need_state": float(need_state),
            "D_occ": float(D_occ), "D_state": float(D_state),
            "n_support_ok": int(len(support_ok)), "n_mdl_ok": int(n_mdl_ok),
            "n_rate_ok": int(n_rate_ok), "n_admissible": int(len(adm)),
            "forced_fallback": False,
            "disc_ridge_mae": float(disc_mae), "mon_ridge_mae": float(mon_mae),
        }
        rounds.append(rec)
        if r % MONITOR_EVERY == 0 or r == steps:
            monitor_curve.append({"round": r, "disc_ridge_mae": float(disc_mae),
                                  "mon_ridge_mae": float(mon_mae)})
        log(f"[rate] r{r:02d} size={size} occ={f_occ[key]} sup={f_graph[key]} "
            f"merges={merges_disc} task={rec['task_score']:.4f} "
            f"g_occ={chosen['g_occ']:.5f} g_state={chosen['g_state']:.5f} "
            f"adm={len(adm)}/{len(support_ok)} disc={disc_mae:.4f} mon={mon_mae:.4f}")

    sequence = [bytes.fromhex(x["key"]) for x in rounds]
    disc_codes = [build_structured_code(
        m, tokenize_frozen(m, vocab, sequence, rk), vocab, rk)["code"]
        for m, rk in zip(disc_mols, disc_ranks)]
    mon_codes = [build_structured_code(
        m, tokenize_frozen(m, vocab, sequence, rk), vocab, rk)["code"]
        for m, rk in zip(mon_mols, mon_ranks)]
    return {"vocab": vocab, "sequence": sequence, "rounds": rounds,
            "monitor_curve": monitor_curve, "disc_codes": disc_codes,
            "mon_codes": mon_codes, "infeasible": infeasible,
            "completed": infeasible is None}


# ===========================================================================
# 2. Structural invariants
# ===========================================================================
def invariance_check(vocab: Vocabulary, sequence, mols, ranks, n_graphs: int = 200,
                     n_relabels: int = 5, seed: int = 20260921) -> dict[str, Any]:
    """Random-relabel invariance of the frozen tokenisation (0 mismatch target)."""
    rng = np.random.default_rng(seed)
    mismatch = 0
    checked = 0
    for mol in mols[:n_graphs]:
        base = build_structured_code(
            mol, tokenize_frozen(mol, vocab, sequence, mol_rank(mol)),
            vocab, mol_rank(mol))["code"]
        base_key = mol_iso_key(decode(base, vocab))
        for _ in range(n_relabels):
            perm = rng.permutation(mol.n)
            inv = np.empty_like(perm)
            inv[perm] = np.arange(mol.n)
            # relabelling: old atom ``a`` -> new atom ``perm[a]``
            new_node = np.asarray(mol.node_types, dtype=np.int64)[inv]
            edges = []
            bts = []
            for i, (a, b) in enumerate(mol.bonds):
                u, v = int(perm[a]), int(perm[b])
                if u > v:
                    u, v = v, u
                edges.append((u, v))
                bts.append(int(mol.bond_types[i]))
            relab = Mol(mol.n, edges, np.asarray(bts, dtype=np.int64), new_node)
            code_r = build_structured_code(
                relab, tokenize_frozen(relab, vocab, sequence, mol_rank(relab)),
                vocab, mol_rank(relab))["code"]
            if mol_iso_key(decode(code_r, vocab)) != base_key:
                mismatch += 1
            checked += 1
    return {"checked": checked, "mismatch": mismatch}


# ===========================================================================
# 3. Provenance
# ===========================================================================
def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def _md5(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(stage: str, device: str, args) -> dict[str, Any]:
    return {
        "round": ROUND, "parent_round": PARENT_ROUND, "stage": stage,
        "git_commit": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short"),
        "python": platform.python_version(), "numpy": np.__version__,
        "platform": platform.platform(), "device": device, "seed": SEED,
        "inputs": {
            "freq_artifact": str(args.freq_artifact),
            "freq_md5": _md5(Path(args.freq_artifact)),
            "tm_artifact": str(args.tm_artifact),
            "tm_md5": _md5(Path(args.tm_artifact)),
        },
        "config": {
            "n_train": N_TRAIN, "n_discovery": args.n_discovery,
            "n_monitor": args.n_monitor, "merge_steps": args.steps,
            "max_size": MAX_SIZE, "min_graph_support": MIN_GRAPH_SUPPORT,
            "tie_rel": TIE_REL, "monitor_every": MONITOR_EVERY,
        },
    }


# ===========================================================================
# 4. Stage A
# ===========================================================================
def _load_dict(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def stage_a(args, log) -> int:
    out = args.out
    data_root = args.data_root
    out.mkdir(parents=True, exist_ok=True)

    train_mols, y_all = load_mols_and_y(data_root, "train",
                                        args.n_discovery + args.n_monitor
                                        if args.smoke else N_TRAIN)
    n = len(train_mols)
    disc_idx, mon_idx = discovery_split(n, args.n_discovery, seed=SEED)
    disc_mols = [train_mols[i] for i in disc_idx]
    mon_mols = [train_mols[i] for i in mon_idx]
    y_disc = np.asarray([y_all[i] for i in disc_idx], dtype=np.float64)
    y_mon = np.asarray([y_all[i] for i in mon_idx], dtype=np.float64)
    log(f"[A] loaded {n} canonical train; discovery={len(disc_mols)} "
        f"monitor={len(mon_mols)} (official test never loaded)")
    write_json(out / "split_indices.json",
               {"seed": SEED, "n_train": n, "discovery_idx": disc_idx,
                "monitor_idx": mon_idx})
    disc_bonds = [graph_bonds(m) for m in disc_mols]
    mon_bonds = [graph_bonds(m) for m in mon_mols]
    disc_ranks = [mol_rank(m) for m in disc_mols]
    mon_ranks = [mol_rank(m) for m in mon_mols]

    steps = args.steps
    summary: dict[str, Any] = {"provenance": provenance("stageA", "cpu", args)}

    # ---- FREQ prefix rate curve -------------------------------------------
    freq = _load_dict(args.freq_artifact)
    fvocab, fseq = freq["vocab"], list(freq["sequence"])[:steps]
    curve = frequency_prefix_rate_curve(fvocab, fseq, disc_mols, disc_bonds,
                                        disc_ranks, log=log)
    write_json(out / "freq_rate_curve.json", curve)
    target = {kk: float(curve[-1][kk]) for kk in ("R_occ", "R_state")}
    log(f"[A] FREQ-{steps} discovery budget: R_occ={target['R_occ']:.4f} "
        f"R_state={target['R_state']:.4f}")

    # ---- TM rates (for the k* match) --------------------------------------
    tm = _load_dict(args.tm_artifact)
    tm_rates = rates_for_sequence(tm["vocab"], list(tm["sequence"])[:steps],
                                  disc_mols, disc_bonds, disc_ranks)
    write_json(out / "tm_rates.json", tm_rates)
    log(f"[A] TM discovery rates: R_occ={tm_rates['R_occ']:.4f} "
        f"R_state={tm_rates['R_state']:.4f}")

    # ---- k* selection (state is the primary matching quantity) ------------
    k_star = int(min(range(len(curve)),
                     key=lambda i: abs(curve[i]["R_state"] - tm_rates["R_state"])))
    d_state = abs(curve[k_star]["R_state"] - tm_rates["R_state"])
    d_occ = abs(curve[k_star]["R_occ"] - tm_rates["R_occ"])
    match = {
        "k_star": k_star,
        "R_state_freqloose": float(curve[k_star]["R_state"]),
        "R_occ_freqloose": float(curve[k_star]["R_occ"]),
        "R_state_tm": float(tm_rates["R_state"]),
        "R_occ_tm": float(tm_rates["R_occ"]),
        "abs_state_diff": float(d_state),
        "abs_occ_diff": float(d_occ),
        "state_matched": bool(d_state <= 0.03),
        "strong_rate_match": bool(d_state <= 0.03 and d_occ <= 0.05),
    }
    write_json(out / "rate_match.json", match)
    log(f"[A] k*={k_star} R_state^FL={match['R_state_freqloose']:.4f} "
        f"R_occ^FL={match['R_occ_freqloose']:.4f} "
        f"|state|={d_state:.4f} |occ|={d_occ:.4f} "
        f"strong_match={match['strong_rate_match']}")

    # ---- FREQ-LOOSE dictionary + audit ------------------------------------
    fl_vocab = truncate_vocabulary(fvocab, list(freq["sequence"]), k_star)
    fl_seq = list(freq["sequence"])[:k_star]
    fl_rounds = [r for r in freq["rounds"] if int(r["round"]) <= k_star]
    fl_disc_codes = codes_for(
        fl_vocab, [tokenize_frozen(m, fl_vocab, fl_seq, rk)
                   for m, rk in zip(disc_mols, disc_ranks)], disc_mols, disc_ranks)
    fl_mon_codes = codes_for(
        fl_vocab, [tokenize_frozen(m, fl_vocab, fl_seq, rk)
                   for m, rk in zip(mon_mols, mon_ranks)], mon_mols, mon_ranks)
    fl_audit = stage_a_audit("FREQ-LOOSE", fl_vocab, disc_mols, mon_mols,
                             fl_disc_codes, fl_mon_codes, y_disc, y_mon,
                             len(disc_mols))
    fl_audit["rate"] = corpus_rates(fl_disc_codes)
    fl_audit["invariance"] = invariance_check(fl_vocab, fl_seq, disc_mols, disc_ranks)
    fl_audit["k_star"] = k_star
    summary["stage_a_FREQ-LOOSE"] = fl_audit
    with (out / "dictionary_FREQLOOSE.pkl").open("wb") as fh:
        pickle.dump({"vocab": fl_vocab, "sequence": fl_seq, "rounds": fl_rounds,
                     "k_star": k_star, "provenance": provenance("stageA", "cpu", args)}, fh)
    write_json(out / "audit_freqloose.json", fl_audit)
    log(f"[A:FREQ-LOOSE] n_learned={fl_audit['n_learned']} "
        f"rate={fl_audit['rate']} ridge={fl_audit['ridge']}")

    # ---- TM-RATE discovery + audit ----------------------------------------
    t0 = time.time()
    log(f"[A:TM-RATE] rate-constrained task+MDL discovery on discovery-{len(disc_mols)}")
    tr = discover_rate(disc_mols, disc_bonds, disc_ranks, mon_mols, mon_bonds,
                       mon_ranks, y_disc, y_mon, target["R_state"], target["R_occ"],
                       steps=steps, log=log)
    write_json(out / "rounds_tmrate.json", tr["rounds"])
    tr_audit: dict[str, Any] = {
        "arm": "TM-RATE", "completed": bool(tr["completed"]),
        "infeasible": tr["infeasible"], "n_learned": int(len(tr["rounds"])),
        "wall_s": time.time() - t0, "target": target,
    }
    if tr["completed"]:
        tr_audit.update(stage_a_audit("TM-RATE", tr["vocab"], disc_mols, mon_mols,
                                      tr["disc_codes"], tr["mon_codes"],
                                      y_disc, y_mon, len(disc_mols)))
        tr_audit["rate"] = corpus_rates(tr["disc_codes"])
        tr_audit["monitor_rate"] = corpus_rates(tr["mon_codes"])
        tr_audit["invariance"] = invariance_check(tr["vocab"], tr["sequence"],
                                                  disc_mols, disc_ranks)
        tr_audit["rate_gate"] = {
            "R_state": float(tr_audit["rate"]["R_state"]),
            "R_occ": float(tr_audit["rate"]["R_occ"]),
            "state_le_target_plus_002": bool(
                tr_audit["rate"]["R_state"] <= target["R_state"] + 0.02),
            "occ_le_target_plus_002": bool(
                tr_audit["rate"]["R_occ"] <= target["R_occ"] + 0.02),
            "abs_state_diff": float(abs(tr_audit["rate"]["R_state"] - target["R_state"])),
            "abs_occ_diff": float(abs(tr_audit["rate"]["R_occ"] - target["R_occ"])),
        }
        with (out / "dictionary_TMRATE.pkl").open("wb") as fh:
            pickle.dump({"vocab": tr["vocab"], "sequence": tr["sequence"],
                         "rounds": tr["rounds"],
                         "provenance": provenance("stageA", "cpu", args)}, fh)
        summary["dictionary_similarity_freqloose_vs_tmrate"] = dictionary_like(
            fl_vocab, fl_rounds, tr["vocab"], tr["rounds"])
        summary["dictionary_similarity_tm_vs_tmrate"] = dictionary_like(
            tm["vocab"], tm["rounds"], tr["vocab"], tr["rounds"])
        log(f"[A:TM-RATE] completed rate={tr_audit['rate']} "
            f"gate={tr_audit['rate_gate']} ridge={tr_audit['ridge']}")
    else:
        log(f"[A:TM-RATE] INFEASIBLE: {tr['infeasible']}")
    summary["stage_a_TM-RATE"] = tr_audit
    write_json(out / "audit_tmrate.json", tr_audit)

    summary["rate_match"] = match
    summary["freq_budget_target"] = target
    summary["tm_rates"] = tm_rates
    write_json(out / "SUMMARY_A.json", summary)
    log(f"=== {ROUND} Stage A done ===")
    return 0


def dictionary_like(v1: Vocabulary, r1, v2: Vocabulary, r2) -> dict[str, Any]:
    """Dictionary similarity between two (vocab, rounds) dictionaries."""
    rules1 = [r["key"] for r in r1]
    rules2 = [r["key"] for r in r2]
    types1 = {mt.key.hex() for mt in v1.by_id.values() if not mt.singleton}
    types2 = {mt.key.hex() for mt in v2.by_id.values() if not mt.singleton}
    return {
        "n_rules_a": len(rules1), "n_rules_b": len(rules2),
        "identical_positional_rules": int(sum(1 for a, b in zip(rules1, rules2) if a == b)),
        "shared_rule_keys": int(len(set(rules1) & set(rules2))),
        "identical_final_types": int(len(types1 & types2)),
        "a_only_types": int(len(types1 - types2)),
        "b_only_types": int(len(types2 - types1)),
        "jaccard_types": float(len(types1 & types2) / max(len(types1 | types2), 1)),
    }


# ===========================================================================
# 5. Stage B
# ===========================================================================
def stage_b(args, log) -> int:
    arm = args.arm
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    artifact = args.artifact or (out / f"dictionary_{arm}.pkl")
    if not artifact.exists():
        raise SystemExit(f"missing frozen dictionary artifact: {artifact}")
    data = _load_dict(artifact)
    payload = stage_b_train(arm, data["vocab"], data["sequence"], args.device,
                            args.data_root, out, max_epochs=args.max_epochs,
                            patience=args.patience, limit=args.limit, log=log)
    write_json(out / f"provenance_B{arm}.json", provenance("stageB", args.device, args))
    log(f"[B:{arm}] {json.dumps(payload, default=float)}")
    return 0


# ===========================================================================
# 6. CLI
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
    parser.add_argument("--freq-artifact", type=Path, default=FREQ_ARTIFACT)
    parser.add_argument("--tm-artifact", type=Path, default=TM_ARTIFACT)
    parser.add_argument("--arm", type=str, default=None, choices=["FREQLOOSE", "TMRATE"])
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
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
        if args.max_epochs is None:
            args.max_epochs = 2
        if args.patience is None:
            args.patience = 2

    log(f"=== {ROUND} {args.stage} ===")
    if args.stage == "stageB":
        return stage_b(args, log)
    return stage_a(args, log)


if __name__ == "__main__":
    raise SystemExit(main())
