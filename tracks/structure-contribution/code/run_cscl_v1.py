"""cscl-correctness-v1 driver: audit / screen / train.

Version semantics
-----------------
* Fixes confirmed cscl-v0 correctness defects (signature relabel variance +
  chemistry erasure; doubled physical bonds) — see
  ``notes/correctness_information_audit_v1.md``.
* Owns its own cache (``data/cache/cscl_v1_units.pt``) guarded by
  ``cscl_features.load_or_build_units_cache``; the pre-fix
  ``cscl_v0_units.pt`` can never be silently reused.
* The published cscl-v0 REPORT (commit 6549c04) is a historical record of the
  pre-fix code and is not re-interpreted here.

Modes
-----
``audit``   label-blind representation audit + v0→v1 diff + fake-sharing
            (signature-collision) sampling check + SMILES/PyG chemical-graph
            alignment over all 10000 rows (task §6.2/§6.1/3.3).
``screen``  CPU prediction-information screening (task §6.3): Ridge /
            HistGradientBoosting / XGBoost on (a) v0-legacy features,
            (b) fixed RINGCHAIN-v1 features, (c) rich static ksvd features
            (base573, label-free).  Same fit/dev split; no hyperparameter
            search (fixed screening configs, recorded verbatim).
``train``   O-unit / O-rich arms (task §7): OpaqueModel-style non-GNN
            regressor on the fixed unit representation vs an equal-capacity
            MLP on the rich static features.  GPU round: seed 0 only.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
import cscl_models as cm  # noqa: E402
import cscl_smiles_graph as csg  # noqa: E402
import cscl_units_v0_reference as v0ref  # noqa: E402
from cscl_units import KIND_CHAIN, KIND_RING, unique_undirected_bonds  # noqa: E402

PROTOCOL_ID = "cscl-correctness-v1"
RESULTS_SUBDIR = "tracks/structure-contribution/results"
V1_CACHE = REPO_ROOT / "data/cache/cscl_v1_units.pt"

ARMS = ("ounit", "orich")


# ---------------------------------------------------------------------------
# v0-semantics unit extraction (legacy, audit-only)
# ---------------------------------------------------------------------------


def v0_molecule_units(mol_index: int, x: np.ndarray, edge_index: np.ndarray, edge_attr: np.ndarray, smiles: str = "") -> cf.MolUnits:
    """Bit-exact reproduction of cscl-v0 ``molecule_units`` behaviour.

    Feeds PyG's double-stored directed edges into the v0 partition and signs
    units with the legacy dense-remap signature.  Used only by the audit to
    quantify what the fix changed.  Descriptors here keep v0's defective bond
    counting (double-counted intra/inter bonds, cyclomatic inflated,
    ``is_terminal`` broken) so old-vs-new statistics are comparable.
    """
    from cscl_units import build_partition as _v0_style_partition_compat  # noqa: F401

    atom_types = [int(v) for v in x.reshape(-1)]
    edges = [(int(edge_index[0, k]), int(edge_index[1, k]), int(edge_attr.reshape(-1)[k])) for k in range(edge_index.shape[1])]
    # v0 passed raw directed edges into partition_units + v0 signature
    part = _partition_units_direct(atom_types, edges)
    n_units = part.n_units
    nb_ring = [0] * n_units
    nb_chain = [0] * n_units
    rel_map = part.inter_relations()
    attach_count = [0] * n_units
    for (a, b), bts in rel_map.items():
        attach_count[a] += len(bts)
        attach_count[b] += len(bts)
        for target in (a, b):
            if part.units[target].kind == KIND_RING:
                nb_ring[target] += 1
            else:
                nb_chain[target] += 1
    desc = np.zeros((n_units, cf.DESC_DIM), dtype=np.float32)
    for unit in part.units:
        i = unit.unit_id
        for a in unit.atoms:
            desc[i, int(atom_types[a])] += 1.0
        for _, _, bt in unit.intra_bonds:
            desc[i, cf.ATOM_CATEGORIES + int(bt)] += 1.0
        cyc = len(unit.intra_bonds) - len(unit.atoms) + 1
        scalars = [
            np.log1p(len(unit.atoms)),
            np.log1p(len(unit.intra_bonds)),
            float(max(cyc, 0)),
            1.0 if unit.kind == KIND_RING else 0.0,
            float(attach_count[i]),
            float(nb_ring[i]),
            float(nb_chain[i]),
            1.0 if attach_count[i] <= 1 else 0.0,
        ]
        desc[i, cf.D_HIST:] = np.asarray(scalars, dtype=np.float32)
    rel_pairs = sorted(rel_map.keys())
    rel_feat = np.zeros((len(rel_pairs), cf.REL_DIM), dtype=np.float32)
    for k, (a, b) in enumerate(rel_pairs):
        for bt in rel_map[(a, b)]:
            rel_feat[k, int(bt)] += 1.0
        rel_feat[k, cf.BOND_CATEGORIES] = np.log1p(len(rel_map[(a, b)]))
    return cf.MolUnits(
        mol_index=mol_index,
        unit_sigs=[v0ref.v0_unit_signature(u.kind, u.atoms, atom_types, u.intra_bonds) for u in part.units],
        unit_kinds=[u.kind for u in part.units],
        unit_atoms=[list(u.atoms) for u in part.units],
        desc=desc,
        rel_pairs=rel_pairs,
        rel_feat=rel_feat,
        n_atoms=part.n_atoms,
        smiles=smiles,
    )


def _partition_units_direct(atom_types: list[int], edges: list[tuple[int, int, int]]):
    """v0's partition path: raw (possibly doubled) directed edges, no dedup."""
    from cscl_units import partition_units

    return partition_units(atom_types, edges)


def extract_v0_units() -> tuple[list[cf.MolUnits], np.ndarray, list[str]]:
    import tracks.ksvd.experiments.luyin16.zinc_long_range_proxy as proxy

    dataset = cf.load_official_train()
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    assert len(dataset) == len(smiles) == 10000
    mols, ys = [], np.zeros((len(dataset), 1), dtype=np.float32)
    for i, data in enumerate(dataset):
        mols.append(v0_molecule_units(i, data.x.numpy(), data.edge_index.numpy(), data.edge_attr.numpy(), smiles[i]))
        ys[i, 0] = float(data.y.reshape(-1)[0])
    return mols, ys, smiles


# ---------------------------------------------------------------------------
# rich static features (ksvd typed_cycle_probe base573; label-free)
# ---------------------------------------------------------------------------


def rich_features_573(dataset) -> np.ndarray:
    """Per-molecule base573 (Sem110 sums/sumsq, phi65 sums/sumsq, global62,
    topology25 zeros, untyped ring counts, edge-type counts).

    Provenance audit (task 7.2): every block is a deterministic function of
    the raw attributed graph (``typed_cycle_probe_v1.graph_feature_row``);
    no trained object, no label, no auxiliary target, no frozen predictor and
    no fit-split-dependent statistic enters.  Whole-feature standardization,
    if any, is fit on fit_inner only (done by the trainers, not here).
    """
    from tracks.ksvd.experiments.luyin16 import typed_cycle_probe_v1 as tcp

    rows = []
    for data in dataset:
        base, _typed, _census = tcp.graph_feature_row(data)
        rows.append(base)
    return np.stack(rows).astype(np.float32)


def standardize_rich(R: np.ndarray, fit_rows: np.ndarray, clip: float = 10.0) -> tuple[np.ndarray, dict]:
    """Fit-only standardization of the rich feature matrix with bounded tails.

    Several base573 columns are (near-)constant on fit_inner (e.g. rare
    ring-length counts) while dev molecules can deviate far from the fit
    mean; naive z-scoring then produces ~1e9-magnitude inputs (measured:
    max|Rz| = 2e9 unclipped), which destroys any linear/nonlinear head.
    Rule (label-free, fixed protocol constant — the clip bound is not a fit
    statistic): z-score with fit_inner mean/std (std floored at 1e-9), then
    clip to ``[-clip, +clip]``.
    """
    mu = R[fit_rows].mean(axis=0)
    sd = R[fit_rows].std(axis=0) + 1e-9
    Rz = np.clip((R - mu) / sd, -clip, clip)
    return Rz.astype(np.float32), {"mu": mu, "sd": sd, "clip": clip}


# ---------------------------------------------------------------------------
# audit mode
# ---------------------------------------------------------------------------


def _units_group_stats(sigs: list[str]) -> dict:
    from collections import Counter

    counts = Counter(sigs)
    return counts


def run_audit(out_dir: Path, log=print, n_iso_samples: int = 2000, seed: int = 0) -> dict:
    """Label-blind audit; includes v0→v1 diff and alignment checks."""
    # --- v1 units (fixed code) -------------------------------------------
    mols_v1, _y, smiles = cf.load_or_build_units_cache(V1_CACHE, cf.extract_all)
    idx = cf.build_split_indices(smiles)

    # --- v0 units (legacy semantics, recomputed audit-only) ---------------
    log("recomputing v0-legacy units for the diff ...")
    mols_v0, _, _ = extract_v0_units()

    fit_inner = idx["fit_inner"]
    dev = idx["dev"]

    def summarize_v1(group_rows: np.ndarray) -> dict:
        mols = [mols_v1[i] for i in group_rows]
        sizes = [len(a) for m in mols for a in m.unit_atoms]
        n_units = [len(m.unit_sigs) for m in mols]
        n_rel = [len(m.rel_pairs) for m in mols]
        sigs = [s for m in mols for s in m.unit_sigs]
        cycl = []
        for m in mols:
            for k, u_atoms in enumerate(m.unit_atoms):
                n_intra = int(round(float(m.desc[k, cf.ATOM_CATEGORIES : cf.ATOM_CATEGORIES + cf.BOND_CATEGORIES].sum())))
                cycl.append(n_intra - len(u_atoms) + 1)
        counts = _units_group_stats(sigs)
        top = counts.most_common()
        return {
            "n_mols": len(mols),
            "unit_total": int(sum(n_units)),
            "units_per_mol": {"mean": float(np.mean(n_units)), "p95": float(np.percentile(n_units, 95)), "max": int(np.max(n_units))},
            "unit_size": {"mean": float(np.mean(sizes)), "p95": float(np.percentile(sizes, 95)), "max": int(np.max(sizes))},
            "rels_per_mol": {"mean": float(np.mean(n_rel)), "p95": float(np.percentile(n_rel, 95)), "max": int(np.max(n_rel))},
            "kind_ring_vs_chain": [sum(1 for m in mols for k in m.unit_kinds if k == KIND_RING), sum(1 for m in mols for k in m.unit_kinds if k == KIND_CHAIN)],
            "distinct_types": len(counts),
            "top10_type_unit_coverage": float(sum(c for _, c in top[:10]) / max(len(sigs), 1)),
            "singleton_type_frac_of_units": float(sum(1 for _, c in counts.items() if c == 1) / max(len(sigs), 1)),
            "min_cyclomatic": int(min(cycl)) if cycl else None,
            "max_cyclomatic": int(max(cycl)) if cycl else None,
        }

    def summarize_v0(group_rows: np.ndarray) -> dict:
        mols = [mols_v0[i] for i in group_rows]
        sizes = [len(a) for m in mols for a in m.unit_atoms]
        n_units = [len(m.unit_sigs) for m in mols]
        sigs = [s for m in mols for s in m.unit_sigs]
        cyc = []
        for m in mols:
            for k, u_atoms in enumerate(m.unit_atoms):
                n_intra = int(round(float(m.desc[k, cf.ATOM_CATEGORIES : cf.ATOM_CATEGORIES + cf.BOND_CATEGORIES].sum())))
                cyc.append(n_intra - len(u_atoms) + 1)
        counts = _units_group_stats(sigs)
        top = counts.most_common()
        return {
            "n_mols": len(mols),
            "unit_total": int(sum(n_units)),
            "units_per_mol": {"mean": float(np.mean(n_units)), "p95": float(np.percentile(n_units, 95)), "max": int(np.max(n_units))},
            "unit_size": {"mean": float(np.mean(sizes)), "p95": float(np.percentile(sizes, 95)), "max": int(np.max(sizes))},
            "distinct_types": len(counts),
            "top10_type_unit_coverage": float(sum(c for _, c in top[:10]) / max(len(sigs), 1)),
            "singleton_type_frac_of_units": float(sum(1 for _, c in counts.items() if c == 1) / max(len(sigs), 1)),
            "min_cyclomatic": int(min(cyc)) if cyc else None,
            "max_cyclomatic": int(max(cyc)) if cyc else None,
        }

    # vocab sizes on fit_inner (the fit-only vocabulary)
    stats = cf.FitStats(min_count=3).fit([mols_v1[i] for i in fit_inner], np.zeros((10000, 1), dtype=np.float32))
    unk = 0
    tot = 0
    for i in dev:
        m = mols_v1[i]
        for k in range(len(m.unit_sigs)):
            tot += 1
            if not stats.vocab.is_known(m.unit_sigs[k]):
                unk += 1

    report: dict = {
        "protocol": PROTOCOL_ID,
        "feature_version": cf.FEATURE_VERSION,
        "signature_version": cf.SIGNATURE_VERSION,
        "code_fingerprint": cf.code_fingerprint(),
        "label_blind": True,
        "official_test_loaded": False,
        "fit_inner": summarize_v1(fit_inner),
        "dev": summarize_v1(dev),
        "v0_fit_inner": summarize_v0(fit_inner),
        "v0_dev": summarize_v0(dev),
        "vocab_known_types_v1": stats.vocab.n_known,
        "vocab_total_ids_v1": stats.vocab.n_total,
        "dev_unk_unit_frac_v1": float(unk / max(tot, 1)),
    }

    # --- v0 -> v1 type-group split / merge quantification ------------------
    log("quantifying type split/merge ...")
    all_inst = []
    for i in range(10000):
        m0 = mols_v0[i]
        m1 = mols_v1[i]
        # unit counts are identical by construction (same partition algorithm)
        assert len(m0.unit_sigs) == len(m1.unit_sigs)
        for k in range(len(m0.unit_sigs)):
            all_inst.append((m0.unit_sigs[k], m1.unit_sigs[k]))
    from collections import Counter, defaultdict

    g0 = defaultdict(list)
    g1 = defaultdict(list)
    for j, (s0, s1) in enumerate(all_inst):
        g0[s0].append(j)
        g1[s1].append(j)
    # split: instances sharing v0 type but separated in v1 types
    split_instances = 0
    for s0, insts in g0.items():
        c = Counter(all_inst[j][1] for j in insts)
        split_instances += len(insts) - max(c.values())
    merge_instances = 0
    for s1, insts in g1.items():
        c = Counter(all_inst[j][0] for j in insts)
        merge_instances += len(insts) - max(c.values())
    report["type_groups"] = {
        "unit_instances": len(all_inst),
        "v0_distinct_types": len(g0),
        "v1_distinct_types": len(g1),
        "split_unit_instances": int(split_instances),
        "split_unit_frac": float(split_instances / len(all_inst)),
        "merged_unit_instances": int(merge_instances),
        "merged_unit_frac": float(merge_instances / len(all_inst)),
        "exact_purity_note": "split/merge/purity are exact counts over all unit instances (no sampling); a v0 group is pure iff all its members share one v1 type",
    }
    # exact v0-group purity (instance-weighted): a v0 group is chemically pure
    # iff it maps to exactly one v1 type
    pure_instances = 0
    n_pure_groups = 0
    for s0, insts in g0.items():
        if len({all_inst[j][1] for j in insts}) == 1:
            n_pure_groups += 1
            pure_instances += len(insts)
    report["type_groups"]["v0_pure_groups"] = int(n_pure_groups)
    report["type_groups"]["v0_groups_total"] = len(g0)
    report["type_groups"]["v0_pure_group_instances"] = int(pure_instances)
    report["type_groups"]["v0_pure_instance_frac"] = float(pure_instances / len(all_inst))

    # --- fake-sharing sampling (attributed isomorphism via VF2) ------------
    log(f"sampling {n_iso_samples} same-type unit pairs for attributed isomorphism ...")
    rng = np.random.default_rng(seed)

    dataset = cf.load_official_train()

    def unit_graph(mol_row: int, k: int):
        data = dataset[mol_row]
        atom_types = [int(v) for v in data.x.reshape(-1).tolist()]
        ei = data.edge_index.numpy()
        ea = data.edge_attr.reshape(-1).numpy()
        bonds = unique_undirected_bonds([(int(ei[0, j]), int(ei[1, j])) for j in range(ei.shape[1])], [int(v) for v in ea])
        atoms = set(mols_v1[mol_row].unit_atoms[k])
        g = nx.Graph()
        for a in atoms:
            g.add_node(a, t=atom_types[a])
        for u, v, bt in bonds:
            if u in atoms and v in atoms:
                g.add_edge(u, v, t=bt)
        return g

    # sample same-v0-type pairs and same-v1-type pairs (both units from distinct mols)
    pairs_v0 = []
    keys0 = [s for s, insts in g0.items() if len(insts) >= 2]
    while len(pairs_v0) < n_iso_samples:
        s = keys0[rng.integers(len(keys0))]
        j1, j2 = rng.choice(g0[s], 2, replace=False)
        pairs_v0.append((int(j1), int(j2)))
    keys1 = [s for s, insts in g1.items() if len(insts) >= 2]
    pairs_v1 = []
    while len(pairs_v1) < n_iso_samples:
        s = keys1[rng.integers(len(keys1))]
        j1, j2 = rng.choice(g1[s], 2, replace=False)
        pairs_v1.append((int(j1), int(j2)))

    # build instance index map (mol, unit_k)
    inst_index = []
    for i in range(10000):
        for k in range(len(mols_v1[i].unit_sigs)):
            inst_index.append((i, k))

    def iso_rate(pairs) -> float:
        ok = 0
        for j1, j2 in pairs:
            (i1, k1) = inst_index[j1]
            (i2, k2) = inst_index[j2]
            g1 = unit_graph(i1, k1)
            g2 = unit_graph(i2, k2)
            if nx.is_isomorphic(g1, g2, node_match=lambda a, b: a["t"] == b["t"], edge_match=lambda a, b: a["t"] == b["t"]):
                ok += 1
        return ok / len(pairs)

    report["fake_sharing_check"] = {
        "n_samples_per_side": n_iso_samples,
        "seed": seed,
        "note": "attributed-graph isomorphism (VF2) on unit-internal graphs; equality of signatures is NOT claimed to prove isomorphism",
        "v0_same_type_iso_rate": iso_rate(pairs_v0),
        "v1_same_type_iso_rate": iso_rate(pairs_v1),
    }

    # --- SMILES <-> PyG chemical-graph alignment (all 10000) ---------------
    log("running SMILES/PyG attributed-isomorphism alignment over 10000 molecules ...")
    tier1 = tier2 = tier3 = parse_fail = atom_count_mismatch = kek_fail = 0
    failures = []
    for i in range(10000):
        data = dataset[i]
        try:
            res = csg.check_alignment(smiles[i], data.x.numpy(), data.edge_index.numpy(), data.edge_attr.numpy())
        except csg.SmilesParseError as exc:
            parse_fail += 1
            failures.append({"row": i, "error": str(exc)})
            continue
        if res["n_atoms_smiles"] != res["n_atoms_pyg"] or res["n_bonds_smiles"] != res["n_bonds_pyg"]:
            atom_count_mismatch += 1
            failures.append({"row": i, "tier": "counts", **{k: v for k, v in res.items() if k.startswith("n_")}})
        if res.get("tier1_isomorphic"):
            tier1 += 1
        elif res.get("tier2_isomorphic"):
            tier2 += 1
        elif res.get("tier3_isomorphic"):
            tier3 += 1
            failures.append({"row": i, "tier": "t3_only"})
        if res.get("kekulization_failed"):
            kek_fail += 1
        if i % 2000 == 0:
            log(f"  alignment {i}/10000 (t1={tier1} t2={tier2} t3={tier3})")
    report["smiles_pyg_alignment"] = {
        "rows": 10000,
        "tier1_exact_isomorphic": tier1,
        "tier2_arom_relaxed_only": tier2,
        "tier3_element_topology_only": tier3,
        "atom_or_bond_count_mismatch": atom_count_mismatch,
        "smiles_parse_errors": parse_fail,
        "kekulization_ambiguous": kek_fail,
        "failures": failures[:50],
        "method": "hand-written strict SMILES parser + kekulization + VF2 attributed isomorphism (no RDKit in env; limitations in module docstring)",
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audit.json").write_text(json.dumps(report, indent=2))
    log(json.dumps({k: report[k] for k in ("fit_inner", "v0_fit_inner", "type_groups", "fake_sharing_check")}, indent=2))
    log(f"alignment: t1={tier1} t2={tier2} t3={tier3} count_mismatch={atom_count_mismatch} parse_err={parse_fail}")
    return report


# ---------------------------------------------------------------------------
# screen mode (CPU, task §6.3)
# ---------------------------------------------------------------------------


def run_screen(out_dir: Path, log=print) -> dict:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor

    mols_v1, y, smiles = cf.load_or_build_units_cache(V1_CACHE, cf.extract_all)
    idx = cf.build_split_indices(smiles)
    stats = cf.FitStats(min_count=3).fit([mols_v1[i] for i in idx["fit_inner"]], y)

    # (b) v1 features via the v0 feature layout (fixed code path)
    from run_cscl_v0 import build_tensors, xgb_features  # shared layout

    fit_in_mols = [mols_v1[i] for i in idx["fit_inner"]]
    mon_mols = [mols_v1[i] for i in idx["monitor"]]
    dev_mols = [mols_v1[i] for i in idx["dev"]]
    ts_in = build_tensors(fit_in_mols, stats)
    ts_mon = build_tensors(mon_mols, stats)
    ts_dev = build_tensors(dev_mols, stats)
    X_v1 = {"fit": xgb_features(ts_in, stats.vocab, fit_in_mols), "mon": xgb_features(ts_mon, stats.vocab, mon_mols), "dev": xgb_features(ts_dev, stats.vocab, dev_mols)}

    # (a) v0-legacy features (recomputed with legacy signature + doubled bonds)
    log("building v0-legacy feature matrix ...")
    mols_v0, _, _ = extract_v0_units()
    stats0 = cf.FitStats(min_count=3).fit([mols_v0[i] for i in idx["fit_inner"]], y)
    ts_in0 = build_tensors([mols_v0[i] for i in idx["fit_inner"]], stats0)
    ts_mon0 = build_tensors([mols_v0[i] for i in idx["monitor"]], stats0)
    ts_dev0 = build_tensors([mols_v0[i] for i in idx["dev"]], stats0)
    X_v0 = {"fit": xgb_features(ts_in0, stats0.vocab, [mols_v0[i] for i in idx["fit_inner"]]), "mon": xgb_features(ts_mon0, stats0.vocab, [mols_v0[i] for i in idx["monitor"]]), "dev": xgb_features(ts_dev0, stats0.vocab, [mols_v0[i] for i in idx["dev"]])}

    # (c) rich573
    log("building rich573 feature matrix ...")
    dataset = cf.load_official_train()
    R = rich_features_573(dataset)
    Rz, _ = standardize_rich(R, idx["fit_inner"])
    X_rich = {"fit": Rz[idx["fit_inner"]], "mon": Rz[idx["monitor"]], "dev": Rz[idx["dev"]]}

    y_in = y[idx["fit_inner"], 0]
    y_mon = y[idx["monitor"], 0]
    y_dev = y[idx["dev"], 0]

    def eval_all(name, X):
        rows = {}
        # Ridge: two fixed screening strengths (documented; not a search)
        for alpha in (1.0, 10.0):
            r = Ridge(alpha=alpha).fit(X["fit"], y_in)
            rows[f"ridge_a{alpha:g}"] = {
                "fit_mae": float(np.abs(r.predict(X["fit"]) - y_in).mean()),
                "monitor_mae": float(np.abs(r.predict(X["mon"]) - y_mon).mean()),
                "dev_mae": float(np.abs(r.predict(X["dev"]) - y_dev).mean()),
            }
        h = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.1, early_stopping=False, random_state=0)
        h.fit(X["fit"], y_in)
        rows["hgb"] = {
            "fit_mae": float(np.abs(h.predict(X["fit"]) - y_in).mean()),
            "monitor_mae": float(np.abs(h.predict(X["mon"]) - y_mon).mean()),
            "dev_mae": float(np.abs(h.predict(X["dev"]) - y_dev).mean()),
        }
        try:
            import xgboost as xgb

            m = xgb.XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=0, n_jobs=8, early_stopping_rounds=30)
            m.fit(X["fit"], y_in, eval_set=[(X["mon"], y_mon)], verbose=False)
            rows["xgb"] = {
                "fit_mae": float(np.abs(m.predict(X["fit"]) - y_in).mean()),
                "monitor_mae": float(np.abs(m.predict(X["mon"]) - y_mon).mean()),
                "dev_mae": float(np.abs(m.predict(X["dev"]) - y_dev).mean()),
            }
        except ImportError:
            rows["xgb"] = None
        return {k: v for k, v in rows.items() if v is not None}

    out = {
        "protocol": PROTOCOL_ID,
        "split": "cscl-v0 fit_inner/monitor/dev (unchanged)",
        "v0_legacy_features": eval_all("v0", X_v0),
        "v1_fixed_ringchain_features": eval_all("v1", X_v1),
        "rich573_features": eval_all("rich", X_rich),
        "dims": {"v0": int(X_v0["fit"].shape[1]), "v1": int(X_v1["fit"].shape[1]), "rich": int(X_rich["fit"].shape[1])},
        "screening_config": "Ridge alpha in {1,10}; HGB lr=0.1 max_iter=300; XGB 600x0.05 depth6 (fixed; no search)",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screen.json").write_text(json.dumps(out, indent=2))
    log(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# train mode (O-unit / O-rich; GPU round seed 0 only)
# ---------------------------------------------------------------------------


def train_rich_arm(seed: int, device: str, data: dict, epochs: int = 300, patience: int = 30, batch_size: int = 128, lr: float = 1e-3, wd: float = 1e-5, log=print) -> dict:
    """O-rich: MLP on standardized rich573; identical training protocol to
    train_torch_arm (AdamW 1e-3/1e-5, batch 128, ≤300 epochs, monitor
    early-stop patience 30, top-5 soup + best-epoch)."""
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    dev = torch.device(device)
    R = data["rich"]  # [N, D] standardized (fit_inner stats)
    y = data["y"]
    idx = data["idx"]
    y_in = y[idx["fit_inner"], 0]
    y_mon = y[idx["monitor"], 0]
    y_dev = y[idx["dev"], 0]
    y_mean, y_std = float(y_in.mean()), float(y_in.std() + 1e-12)

    def T(rows):
        return torch.from_numpy(R[rows]).float()

    fit_rows, mon_rows, dev_rows = idx["fit_inner"], idx["monitor"], idx["dev"]
    fit_X = T(fit_rows)
    mon_X, dev_X = T(mon_rows).to(dev), T(dev_rows).to(dev)
    mon_ys = torch.from_numpy((y_mon - y_mean) / y_std).float().to(dev)
    dev_ys = torch.from_numpy((y_dev - y_mean) / y_std).float().to(dev)

    dim = fit_X.shape[1]
    hidden = 48  # ~29.8k params, matched to O-unit opaque (27k)
    model = torch.nn.Sequential(
        torch.nn.Linear(dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, 1)
    ).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    history, ckpts = [], []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(fit_X), generator=torch.Generator().manual_seed(seed * 1000 + epoch))
        for lo in range(0, len(fit_X), batch_size):
            sel = perm[lo : lo + batch_size]
            xb = fit_X[sel].to(dev)
            ysb = torch.from_numpy(((y_in[sel.numpy()] - y_mean) / y_std)).float().to(dev)
            pred = model(xb).squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred, ysb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            mon_mae = float((model(mon_X).squeeze(-1) - mon_ys).abs().mean()) * y_std
        history.append(mon_mae)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        ckpts.append((mon_mae, state))
        ckpts.sort(key=lambda x: x[0])
        if len(ckpts) > 5:
            ckpts.pop()
        if epoch % 25 == 0 or epoch == 1:
            log(f"[orich s{seed}] epoch {epoch} monitor_mae {mon_mae:.5f}")
        best_epoch = int(np.argmin(history))
        if epoch - (best_epoch + 1) >= patience:
            break
    avg = {k: torch.stack([s[1][k].float() for s in ckpts]).mean(0) for k in ckpts[0][1]}
    best_state = dict(ckpts[0][1])
    model.load_state_dict(avg)
    model.eval()
    wall = time.time() - t0
    with torch.no_grad():
        dev_mae = float((model(dev_X).squeeze(-1) - dev_ys).abs().mean()) * y_std
        model.load_state_dict(best_state)
        model.eval()
        best_dev_mae = float((model(dev_X).squeeze(-1) - dev_ys).abs().mean()) * y_std
        # fit MAE
        fit_preds = []
        for lo in range(0, len(fit_X), 512):
            fit_preds.append(model(fit_X[lo : lo + 512].to(dev)).squeeze(-1).cpu())
        fit_mae = float((torch.cat(fit_preds).numpy() * y_std + y_mean - y_in).astype(np.float64).__abs__().mean())
    n_params = int(sum(p.numel() for p in model.parameters()))
    # per-molecule dev predictions (soup model) for paired comparisons
    with torch.no_grad():
        model.load_state_dict(avg)
        model.eval()
        dev_pred = (model(dev_X).squeeze(-1) * y_std + y_mean).detach().cpu().numpy().astype(np.float32)
    return {
        "arm": "orich",
        "seed": seed,
        "device": str(dev),
        "soup_dev_mae": dev_mae,
        "best_epoch_dev_mae": best_dev_mae,
        "fit_inner_mae": fit_mae,
        "best_monitor_mae": min(history),
        "best_epoch": int(np.argmin(history)) + 1,
        "epochs_run": len(history),
        "n_params": n_params,
        "input_dim": int(dim),
        "wall_seconds": wall,
        "_dev_pred": dev_pred,
        "_dev_row": idx["dev"].astype(np.int64),
    }


def run_train(arm: str, seed: int, device: str, out_dir: Path, smoke: bool, log=print) -> dict:
    from run_cscl_v0 import build_tensors, train_torch_arm  # shared training machinery

    data = cf.load_or_build_units_cache(V1_CACHE, cf.extract_all)
    mols, y, smiles = data
    idx = cf.build_split_indices(smiles)
    stats = cf.FitStats(min_count=3).fit([mols[i] for i in idx["fit_inner"]], y)
    payload_common = {
        "protocol": PROTOCOL_ID,
        "seed": seed,
        "official_test_loaded": False,
        "official_valid_loaded": False,
        "supervision": "raw y only",
        "feature_version": cf.FEATURE_VERSION,
        "code_fingerprint": cf.code_fingerprint(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if device.startswith("cuda") and torch.cuda.is_available() else "cpu",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    if arm == "ounit":
        dt = {
            "fit_inner": build_tensors([mols[i] for i in idx["fit_inner"]], stats),
            "monitor": build_tensors([mols[i] for i in idx["monitor"]], stats),
            "dev": build_tensors([mols[i] for i in idx["dev"]], stats),
            "stats": stats,
            "y": y,
            "fit_inner_idx": idx["fit_inner"],
            "monitor_idx": idx["monitor"],
            "dev_idx": idx["dev"],
        }
        res = train_torch_arm("opaque", seed, device, dt, epochs=3 if smoke else 300, patience=30, log=log)
    elif arm == "orich":
        dataset = cf.load_official_train()
        R = rich_features_573(dataset)
        Rz, _ = standardize_rich(R, idx["fit_inner"])
        res = train_rich_arm(seed, device, {"rich": Rz, "y": y, "idx": idx}, epochs=3 if smoke else 300, log=log)
    else:
        raise ValueError(arm)
    payload = {**payload_common, **{k: v for k, v in res.items() if not k.startswith("_")}}
    if "_dev_pred" in res:
        np.savez_compressed(
            out_dir / f"dev_preds_{arm}_s{seed}.npz",
            row=res["_dev_row"],
            pred=res["_dev_pred"],
            y=y[res["_dev_row"], 0].astype(np.float32),
        )
    (out_dir / f"train_{arm}_s{seed}.json").write_text(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["audit", "screen", "train"])
    ap.add_argument("--arm", default="ounit", choices=ARMS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-iso-samples", type=int, default=2000)
    args = ap.parse_args()
    print(f"[cscl-correctness-v1] feature_version={cf.FEATURE_VERSION} code_fp={cf.code_fingerprint()}", flush=True)
    if args.mode == "audit":
        run_audit(args.out, n_iso_samples=args.n_iso_samples)
    elif args.mode == "screen":
        run_screen(args.out)
    else:
        run_train(args.arm, args.seed, args.device, args.out, args.smoke)


if __name__ == "__main__":
    main()
