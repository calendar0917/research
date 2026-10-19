"""Phase 1d CPU gates (run before any Phase 1d GPU trajectory).

Checks frozen data hashes, the information guard, parameter/FLOP matching, the
additive mechanism, permutation/direction invariance, a valid local-multiset
rewiring, gradient liveness, data isolation, and the structure-only feature
table.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PHASE1B = HERE.parent / "phase1b"
PHASE1C = HERE.parent / "phase1c"
for p in (HERE, PHASE1B, PHASE1C):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ssrc_data import (  # noqa: E402
    CACHE_DIR,
    LabelVault,
    MolGraph,
    build_cache,
    gather_batch,
    load_cache,
    row_sha256,
    sha256_file,
)
from ssrc_models import (  # noqa: E402
    ModelConfig,
    build_model,
    parameter_count,
)
from ssrc_models_p1c import P1CConfig, register_phase1c_models  # noqa: E402
from ssrc_models_p1d import (  # noqa: E402
    AdditiveIncidenceRelation,
    ModelRIncidence,
    base_incidence_config,
    incidence_capacity_config,
    r1_size_capacity_config,
    r1_size_edge_aggregate,
    register_phase1d_models,
)
from ssrc_train import MODEL_KEYS, TrainConfig, _measure_flops, _measure_step_time, _model_batch  # noqa: E402

RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RESULTS_1C = REPO / "tracks/structural-sufficiency/results/phase1c"
RESULTS_1D = REPO / "tracks/structural-sufficiency/results/phase1d"
RUNS_1B = REPO / "tracks/structural-sufficiency/runs/phase1b"
RUNS_1C = REPO / "tracks/structural-sufficiency/runs/phase1c"
CACHE_PATH = CACHE_DIR / "zinc_phase1b.pt"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
LABELS_PATH = RESULTS_1B / "zinc_labels_fit_monitor.json"
SIZE_PATH = RESULTS_1C / "zinc_size_features.pt"
SIZE_MANIFEST = RESULTS_1C / "prep_size_manifest.json"
FEATURES_PATH = RESULTS_1D / "structural_features.npz"
FEATURES_MANIFEST = RESULTS_1D / "structural_features_manifest.json"
OUT = RESULTS_1D / "cpu_gates_phase1d.json"

EXPECTED = {
    "split_sha256": "fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49",
    "labels_sha256": "631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78",
    "cache_sha256": "76c1749277f1c4019f8b94b5a3a9bc7b9f83abb8b11b7feed60449f47a9ed84c",
    "row_sha256": {
        "fit": "8351cbee52e06e504a7113a4649afbfcd840321e057c554811b4aec910d786d0",
        "monitor": "2e4801c644acece7431d892ecf8fe6588335c079a5a300da0cd8f8fe7133b790",
        "reserved": "cf1e8c6c662add9a6c38e34df49c5bae7d0ede8e35a92693f481a82d3c21477d",
    },
    "params": {"R1_size": 115_977, "R_incidence": 115_952, "R1_size_cap": 453_129, "R_incidence_cap": 436_841},
    "gref_params": 439_559,
    "anchors": {"S0_seed0": 0.317303, "R1_seed0": 0.260161, "M2_seed0": 0.252220, "G_ref_seed0": 0.165290},
}

records: list[dict] = []


def ok(name: str, condition: bool, detail: dict | None = None) -> None:
    records.append({"check": name, "pass": bool(condition), **(detail or {})})
    print(("PASS " if condition else "FAIL ") + name + (f" {detail}" if detail else ""))


def mol_chain(n: int = 6, d: int = 4, cycle: bool = False) -> MolGraph:
    af = np.zeros((n, d), dtype=np.float32)
    af[np.arange(n), np.arange(n) % d] = 1.0
    if cycle:
        bu = np.arange(n, dtype=np.int64)
        bv = (np.arange(n) + 1) % n
    else:
        bu = np.arange(n - 1, dtype=np.int64)
        bv = np.arange(1, n, dtype=np.int64)
    bf = np.zeros((len(bu), 2), dtype=np.float32)
    bf[:, 0] = 1.0
    return MolGraph(af, bu, bv, bf)


def permute_mol(mol: MolGraph, perm: np.ndarray) -> MolGraph:
    n = mol.n_atoms
    old_to_new = np.empty(n, dtype=np.int64)
    old_to_new[perm] = np.arange(n)
    af = mol.atom_feat[perm].copy()
    bonds = []
    for k in range(mol.n_bonds):
        u = int(old_to_new[int(mol.bond_u[k])])
        v = int(old_to_new[int(mol.bond_v[k])])
        bonds.append((min(u, v), max(u, v), k))
    bonds.sort()
    return MolGraph(
        af,
        np.asarray([b[0] for b in bonds], dtype=np.int64),
        np.asarray([b[1] for b in bonds], dtype=np.int64),
        mol.bond_feat[[b[2] for b in bonds]].copy(),
    )


def one_mol_batch(mol: MolGraph) -> dict:
    cache = build_cache([mol])
    b = gather_batch(cache, [0], torch.zeros(1, dtype=torch.float32), include_size=True)
    return _model_batch("R_incidence", b)


def rewiring_pair() -> tuple[MolGraph, MolGraph]:
    """Two graphs with equal node attributes and per-node incident type multisets."""
    d_atom, d_edge = 4, 2
    af = np.zeros((6, d_atom), dtype=np.float32)
    af[0, 0] = 1.0
    af[1, 0] = 1.0
    af[2, 1] = 1.0
    af[3, 2] = 1.0
    af[4, 3] = 1.0
    af[5, 1] = 1.0
    bf_single = np.asarray([1.0, 0.0], dtype=np.float32)
    bf_double = np.asarray([0.0, 1.0], dtype=np.float32)
    g1 = MolGraph(
        af,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([2, 3, 4, 5], dtype=np.int64),
        np.stack([bf_single, bf_double, bf_single, bf_double]),
    )
    g2 = MolGraph(
        af,
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        np.asarray([4, 5, 2, 3], dtype=np.int64),
        np.stack([bf_single, bf_double, bf_single, bf_double]),
    )
    return g1, g2


def incident_signature(mol: MolGraph) -> list[tuple]:
    out: list[list[tuple[int, ...]]] = [[] for _ in range(mol.n_atoms)]
    for k in range(mol.n_bonds):
        u, v = int(mol.bond_u[k]), int(mol.bond_v[k])
        t = int(np.argmax(mol.bond_feat[k]))
        out[u].append((t,))
        out[v].append((t,))
    return [(tuple(mol.atom_feat[i].astype(float)), tuple(sorted(out[i]))) for i in range(mol.n_atoms)]


def edge_batch(mol: MolGraph) -> dict:
    src, dst, attrs = [], [], []
    d_edge = mol.bond_feat.shape[1]
    for k in range(mol.n_bonds):
        u, v = int(mol.bond_u[k]), int(mol.bond_v[k])
        src.extend([u, v])
        dst.extend([v, u])
        attrs.extend([mol.bond_feat[k], mol.bond_feat[k]])
    return {
        "root_edge_index": torch.as_tensor([src, dst], dtype=torch.long),
        "root_edge_attr": torch.as_tensor(np.stack(attrs), dtype=torch.float32),
        "root_batch": torch.zeros((mol.n_atoms,), dtype=torch.long),
        "n_molecules": 1,
    }


def gate_frozen_data() -> dict:
    split = json.loads(SPLIT_PATH.read_text())
    rows = {k: [int(x) for x in v] for k, v in split["rows"].items()}
    ok(
        "zinc_split_and_labels_frozen",
        sha256_file(SPLIT_PATH) == EXPECTED["split_sha256"]
        and sha256_file(LABELS_PATH) == EXPECTED["labels_sha256"]
        and all(row_sha256(rows[k]) == v for k, v in EXPECTED["row_sha256"].items())
        and split["counts"] == {"fit": 8000, "monitor": 1000, "reserved": 1000},
        {"split_sha256": sha256_file(SPLIT_PATH), "labels_sha256": sha256_file(LABELS_PATH)},
    )
    vault = LabelVault(LABELS_PATH, rows["fit"], rows["monitor"], rows["reserved"])
    vault.load()
    audit = vault.audit()
    ok(
        "label_vault_fit_monitor_only",
        audit["labels_retained"] == 9000
        and audit["reserved_labels_retained"] == 0
        and audit["official_valid_test_labels_retained"] == 0,
        audit,
    )
    ok("zinc_cache_sha256_frozen", sha256_file(CACHE_PATH) == EXPECTED["cache_sha256"], {"sha256": sha256_file(CACHE_PATH)})
    return {"split": rows, "vault": vault}


def gate_capacity_and_flops(data: dict) -> dict:
    cache = load_cache(CACHE_PATH)
    cache["mol_size_feat"] = torch.load(SIZE_PATH, map_location="cpu", weights_only=False)["size_feat"].float()
    rows_fit = data["split"]["fit"][:64]
    y_all = torch.full((10000,), float("nan"), dtype=torch.float32)
    for rows, purpose in ((data["split"]["fit"], "train"), (data["split"]["monitor"], "monitor")):
        y_all[torch.as_tensor(rows, dtype=torch.long)] = data["vault"].labels(rows, purpose)
    register_phase1c_models()
    register_phase1d_models()
    configs = {
        "R1_size": P1CConfig(),
        "R_incidence": base_incidence_config(),
        "R1_size_cap": r1_size_capacity_config(),
        "R_incidence_cap": incidence_capacity_config(),
    }
    counts: dict[str, int] = {}
    flops: dict[str, int | None] = {}
    steps: dict[str, float | None] = {}
    for name, cfg in configs.items():
        torch.manual_seed(0)
        model = build_model(name, int(cache["meta"]["atom_dim"]), int(cache["meta"]["bond_dim"]), cfg)
        counts[name] = parameter_count(model)
        tcfg = TrainConfig(epochs=1, patience=1, batch_size=64, model=cfg)
        kwargs = {"include_size": True}
        flops[name] = _measure_flops(model, name, rows_fit, y_all, cache, tcfg, torch.device("cpu"), batch_kwargs=kwargs).get("flops_forward")
        steps[name] = _measure_step_time(model, name, rows_fit, y_all, cache, tcfg, torch.device("cpu"), batch_kwargs=kwargs)
    ok(
        "base_parameter_match",
        counts["R_incidence"] == EXPECTED["params"]["R_incidence"]
        and counts["R1_size"] == EXPECTED["params"]["R1_size"]
        and abs(counts["R_incidence"] - counts["R1_size"]) / counts["R1_size"] < 0.01,
        {"params": counts, "relative_delta": (counts["R_incidence"] - counts["R1_size"]) / counts["R1_size"]},
    )
    cap_ok = all(
        abs(counts[k] - EXPECTED["gref_params"]) / EXPECTED["gref_params"] < 0.05
        for k in ["R1_size_cap", "R_incidence_cap"]
    )
    ok(
        "capacity_arms_within_5pct_of_gref",
        cap_ok and counts["R1_size_cap"] == EXPECTED["params"]["R1_size_cap"] and counts["R_incidence_cap"] == EXPECTED["params"]["R_incidence_cap"],
        {"params": {k: counts[k] for k in ["R1_size_cap", "R_incidence_cap"]}, "gref": EXPECTED["gref_params"]},
    )
    ok(
        "forward_flops_measured",
        all(flops[k] and flops[k] > 0 for k in configs),
        {"flops_batch64": flops},
    )
    ratio = flops["R_incidence"] / flops["R1_size"]
    ok("incidence_flops_close_to_r1_size", 0.8 < ratio < 1.25, {"ratio": ratio, "flops": flops})
    return {"params": counts, "flops": flops, "step_time_ms": steps, "capacity_target": EXPECTED["gref_params"]}


def gate_information_and_invariance() -> dict:
    register_phase1c_models()
    register_phase1d_models()
    ok(
        "incidence_and_r1_size_use_identical_batch_keys",
        MODEL_KEYS["R_incidence"] == MODEL_KEYS["R1_size"]
        and MODEL_KEYS["R1_size_cap"] == MODEL_KEYS["R1_size"]
        and MODEL_KEYS["R_incidence_cap"] == MODEL_KEYS["R1_size"],
        {"keys": sorted(MODEL_KEYS["R_incidence"])},
    )
    cfg_i = P1CConfig(hidden=16, ffn_hidden=32, rel_hidden=32, edge_readout_hidden=16)
    cfg_i_d = __import__("ssrc_models_p1d", fromlist=["P1DConfig"]).P1DConfig(
        hidden=16, ffn_hidden=32, rel_hidden=32, edge_readout_hidden=16, incidence_rel_hidden=31
    )
    torch.manual_seed(0)
    inc = ModelRIncidence(4 + 4, 2, cfg_i_d).eval()
    torch.manual_seed(0)
    r1s = build_model("R1_size", 4, 2, cfg_i).eval()

    # same module classes and parameter shapes for the shared readout rule
    class_names = {
        "trunk": (type(inc.trunk), type(r1s.trunk)),
        "readout": (type(inc.readout), type(r1s.readout)),
        "edge_readout": (type(inc.edge_readout), type(r1s.edge_readout)),
        "head": (type(inc.head), type(r1s.head)),
    }
    shared_shapes = all(
        list(inc_state.shape) == list(r1_state.shape)
        for inc_state, r1_state in zip(
            [p for n, p in inc.named_parameters() if n.startswith(("trunk", "readout", "edge_readout", "head"))],
            [p for n, p in r1s.named_parameters() if n.startswith(("trunk", "readout", "edge_readout", "head"))],
        )
    )
    ok(
        "shared_module_classes_and_shapes",
        all(a is b for a, b in class_names.values()) and shared_shapes,
        {"classes": {k: (a.__name__, b.__name__) for k, (a, b) in class_names.items()}},
    )

    mol = mol_chain(6, d=4)
    b0 = one_mol_batch(mol)
    with torch.no_grad():
        y0_i = inc(b0)
        y0_r = r1s(b0)
        e0_i = inc.edge_aggregate(inc.trunk(b0), b0)
        e0_r = r1_size_edge_aggregate(r1s, r1s.trunk(b0), b0)
        # node renumbering
        perm = np.asarray([3, 0, 5, 1, 4, 2], dtype=np.int64)
        bp = one_mol_batch(permute_mol(mol, perm))
        yp_i = inc(bp)
        yp_r = r1s(bp)
        # reverse every directed edge
        b_rev = dict(b0)
        b_rev["root_edge_index"] = b0["root_edge_index"].flip(0)
        yr_i = inc(b_rev)
        yr_r = r1s(b_rev)
    ok(
        "node_renumbering_invariance",
        torch.allclose(y0_i, yp_i, atol=1e-6) and torch.allclose(y0_r, yp_r, atol=1e-6),
        {"incidence_absdiff": float((y0_i - yp_i).abs().max()), "r1_size_absdiff": float((y0_r - yp_r).abs().max())},
    )
    ok(
        "edge_direction_swap_invariance",
        torch.allclose(y0_i, yr_i, atol=1e-6) and torch.allclose(y0_r, yr_r, atol=1e-6),
        {"incidence_absdiff": float((y0_i - yr_i).abs().max()), "r1_size_absdiff": float((y0_r - yr_r).abs().max())},
    )

    # Direct edge-aggregate test on a valid local-incidence-preserving rewiring.
    g1, g2 = rewiring_pair()
    ok(
        "rewiring_preserves_node_attributes_and_incident_bond_multisets",
        incident_signature(g1) == incident_signature(g2) and g1.n_bonds == g2.n_bonds,
        {"signature_0": incident_signature(g1)[0], "signature_1": incident_signature(g1)[1]},
    )
    b1, b2 = edge_batch(g1), edge_batch(g2)
    inv_diffs, r1_diffs = [], []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        inc2 = ModelRIncidence(8, 2, cfg_i_d).eval()
        torch.manual_seed(seed)
        r1s2 = build_model("R1_size", 4, 2, cfg_i).eval()
        tok = torch.randn(6, 16)
        with torch.no_grad():
            em1, es1, _, _ = inc2.edge_aggregate(tok, b1)
            em2, es2, _, _ = inc2.edge_aggregate(tok, b2)
            rm1, rs1, _, _ = r1_size_edge_aggregate(r1s2, tok, b1)
            rm2, rs2, _, _ = r1_size_edge_aggregate(r1s2, tok, b2)
        inv_diffs.append(float(max((em1 - em2).abs().max(), (es1 - es2).abs().max())))
        r1_diffs.append(float(max((rm1 - rm2).abs().max(), (rs1 - rs2).abs().max())))
    ok(
        "incidence_edge_aggregate_invariant_under_valid_rewiring",
        max(inv_diffs) < 1e-6,
        {"max_abs_diffs_by_seed": inv_diffs},
    )
    ok(
        "r1_size_edge_aggregate_changes_under_rewiring",
        min(r1_diffs) > 1e-5,
        {"min_abs_diff_by_seed": r1_diffs},
    )
    return {"rewiring_incidence_diffs": inv_diffs, "rewiring_r1_size_diffs": r1_diffs}


def gate_relation_form_and_gradients(data: dict) -> dict:
    cache = load_cache(CACHE_PATH)
    cache["mol_size_feat"] = torch.load(SIZE_PATH, map_location="cpu", weights_only=False)["size_feat"].float()
    rows = data["split"]["fit"][:8]
    y = data["vault"].labels(rows, "train")
    b = _model_batch("R_incidence", gather_batch(cache, rows, y, include_size=True))
    register_phase1d_models()
    cfg = base_incidence_config()
    torch.manual_seed(0)
    model = build_model("R_incidence", 21, 4, cfg)
    ok(
        "incidence_relation_has_no_joint_endpoint_input",
        isinstance(model.rel, AdditiveIncidenceRelation)
        and model.rel.net[0].in_features == cfg.hidden + 4
        and model.rel.net[0].in_features != 3 * cfg.hidden + 4,
        {"relation_in_features": int(model.rel.net[0].in_features), "expected": int(cfg.hidden + 4)},
    )
    model.train()
    pred = model(b)
    loss = torch.nn.functional.l1_loss(pred, b["y"])
    loss.backward()
    wanted = {
        "rel": ["rel.net.0.weight", "rel.net.3.weight"],
        "trunk": ["trunk.local.input_proj.weight", "trunk.blocks.0.net.0.weight"],
        "readout": ["readout.net.0.weight", "readout.net.3.weight"],
        "edge_readout": ["edge_readout.net.0.weight", "edge_readout.net.3.weight"],
        "head": ["head.net.0.weight", "head.net.3.weight"],
    }
    grad_state = {}
    for group, names in wanted.items():
        vals = []
        for n, p in model.named_parameters():
            if n in names:
                vals.append(None if p.grad is None else float(p.grad.detach().abs().max()))
        grad_state[group] = vals
    ok(
        "incidence_forward_backward_gradients_live",
        all(v is not None and np.isfinite(v) and v > 0 for vals in grad_state.values() for v in vals),
        grad_state,
    )
    # both endpoint slots contribute to the relation output
    with torch.no_grad():
        tok = model.trunk(b)
        src, dst = b["root_edge_index"]
        e = b["root_edge_attr"]
        base_r = model.rel(tok[src], e) + model.rel(tok[dst], e)
        zero_src = torch.zeros_like(tok[src])
        zero_dst = torch.zeros_like(tok[dst])
        r_src = model.rel(zero_src, e) + model.rel(tok[dst], e)
        r_dst = model.rel(tok[src], e) + model.rel(zero_dst, e)
    ok(
        "incidence_both_endpoint_slots_are_live",
        float((base_r - r_src).abs().max()) > 0 and float((base_r - r_dst).abs().max()) > 0,
        {"src_absdiff": float((base_r - r_src).abs().max()), "dst_absdiff": float((base_r - r_dst).abs().max())},
    )
    return {"grad_state": grad_state}


def gate_feature_table() -> dict:
    if not FEATURES_PATH.exists() or not FEATURES_MANIFEST.exists():
        ok("structure_feature_table_present", False, {"missing": str(FEATURES_PATH)})
        return {}
    z = np.load(FEATURES_PATH, allow_pickle=False)
    manifest = json.loads(FEATURES_MANIFEST.read_text())
    cache = load_cache(CACHE_PATH)
    rows = z["rows"]
    n_atoms = z["n_atoms"]
    n_bonds = z["n_bonds"]
    ok(
        "structure_feature_table_is_frozen_structure_only",
        rows.size == 10000
        and manifest["scope"]["structure_only"] is True
        and manifest["scope"]["labels_read"] is False
        and manifest["scope"]["official_valid_test_read"] is False
        and manifest["scope"]["internal_reserved_read"] is False
        and manifest["feature_sha256"] == sha256_file(FEATURES_PATH)
        and np.array_equal(n_atoms.astype(np.int64), cache["mol_n_atoms"].numpy()[rows])
        and np.array_equal(n_bonds.astype(np.int64), (cache["root_n_edges"].numpy() // 2)[rows]),
        {
            "n_rows": int(rows.size),
            "n_features": int(len(z.files) - 1),
            "feature_sha256": manifest["feature_sha256"],
        },
    )
    return {"feature_sha256": manifest["feature_sha256"], "n_rows": int(rows.size), "n_features": int(len(z.files) - 1)}


def gate_frozen_anchors() -> dict:
    paths = {
        "S0_seed0": RUNS_1B / "zinc/S0_seed0.json",
        "R1_seed0": RUNS_1B / "zinc/R1_seed0.json",
        "M2_seed0": RUNS_1B / "zinc/M2_seed0.json",
        "G_ref_seed0": RUNS_1B / "zinc/G_ref_seed0.json",
    }
    values = {}
    for name, path in paths.items():
        d = json.loads(path.read_text())
        values[name] = float(np.mean(np.abs(np.asarray(d["monitor_pred_best"]) - np.asarray(d["monitor_y"]))))
    ok(
        "phase1b_anchors_unchanged",
        all(abs(values[k] - EXPECTED["anchors"][k]) < 1e-6 for k in values),
        {"anchor_mae": values, "expected": EXPECTED["anchors"]},
    )
    return {"anchor_mae": values}


def main() -> int:
    t0 = time.perf_counter()
    data = gate_frozen_data()
    cap = gate_capacity_and_flops(data)
    inv = gate_information_and_invariance()
    grad = gate_relation_form_and_gradients(data)
    feat = gate_feature_table()
    anchors = gate_frozen_anchors()
    payload = {
        "protocol_id": "ssrc-phase1d-cpu-gates",
        "scope": {
            "official_valid_test_labels_read": False,
            "internal_reserved_labels_read": False,
            "labels_used": "Fit/Monitor only",
        },
        "expected": EXPECTED,
        "capacity_and_flops": cap,
        "invariance": inv,
        "gradients": grad,
        "features": feat,
        "anchors": anchors,
        "checks": records,
        "n_checks": len(records),
        "n_passed": sum(1 for r in records if r["pass"]),
        "wall_seconds": time.perf_counter() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(OUT), "n_checks": len(records), "n_passed": payload["n_passed"], "wall_seconds": payload["wall_seconds"]}, indent=2))
    return 0 if payload["n_passed"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
