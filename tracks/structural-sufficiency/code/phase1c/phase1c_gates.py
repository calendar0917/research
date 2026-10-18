"""Phase 1c CPU gates (run before any Phase 1c GPU trajectory).

Checks: frozen Phase 1b data hashes, size-feature sidecar, connectivity,
parameter matching, real forward FLOPs, shared-trunk initialisation,
permutation/batch invariance, information guards, endpoint-correspondence
ablation, mean-vs-size readout separation, gradient liveness, label isolation,
and the optional M_keep_edge matched-compute ratio.
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
for p in (HERE, PHASE1B):
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
    MARGINAL_SIZE_KEYS,
    SIZE_ONLY_KEYS,
    assert_marginal_size_batch,
    assert_size_only_batch,
    build_model,
    parameter_count,
)
from ssrc_models_p1c import P1CConfig, register_phase1c_models  # noqa: E402
from ssrc_train import TrainConfig, _measure_flops, _model_batch  # noqa: E402

RESULTS = REPO / "tracks/structural-sufficiency/results/phase1c"
RESULTS_1B = REPO / "tracks/structural-sufficiency/results/phase1b"
RUNS_1B = REPO / "tracks/structural-sufficiency/runs/phase1b"
CACHE_PATH = CACHE_DIR / "zinc_phase1b.pt"
SPLIT_PATH = RESULTS_1B / "split_zinc_phase1b.json"
LABELS_PATH = RESULTS_1B / "zinc_labels_fit_monitor.json"
SIZE_PATH = RESULTS / "zinc_size_features.pt"
SIZE_MANIFEST = RESULTS / "prep_size_manifest.json"
OUT = RESULTS / "cpu_gates_phase1c.json"

EXPECTED = {
    "split_sha256": "fbca5fd5ac9c82c457bf4d78781fbe9936dd7068082377ff8cc2bf2309d72b49",
    "labels_sha256": "631dae5fe93f543599c5cdfe92057fa8ee0fcb3e3faaf4e51aceed1e8312ab78",
    "cache_sha256": "76c1749277f1c4019f8b94b5a3a9bc7b9f83abb8b11b7feed60449f47a9ed84c",
    "row_sha256": {
        "fit": "8351cbee52e06e504a7113a4649afbfcd840321e057c554811b4aec910d786d0",
        "monitor": "2e4801c644acece7431d892ecf8fe6588335c079a5a300da0cd8f8fe7133b790",
        "reserved": "cf1e8c6c662add9a6c38e34df49c5bae7d0ede8e35a92693f481a82d3c21477d",
    },
    "params": {"S_cap_size": 115_883, "R1_size": 115_977, "R1_marginal_size": 115_977},
    "anchors": {
        "S0_seed0": 0.317303,
        "R1_seed0": 0.260161,
        "M2_seed0": 0.252220,
        "G_ref_seed0": 0.165290,
    },
}

records: list[dict] = []


def ok(name: str, condition: bool, detail: dict | None = None) -> None:
    records.append({"check": name, "pass": bool(condition), **(detail or {})})
    print(("PASS " if condition else "FAIL ") + name + (f" {detail}" if detail else ""))


def mol_chain(n: int, d: int = 4, cycle: bool = False) -> MolGraph:
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


def mol_typed_cycle(n: int) -> MolGraph:
    af = np.zeros((n, 4), dtype=np.float32)
    af[np.arange(n), np.arange(n) % 4] = 1.0
    bu = np.arange(n, dtype=np.int64)
    bv = (np.arange(n) + 1) % n
    bf = np.zeros((n, 2), dtype=np.float32)
    bf[::2, 0] = 1.0
    bf[1::2, 1] = 1.0
    return MolGraph(af, bu, bv, bf)


def disjoint_union(mols: list[MolGraph]) -> MolGraph:
    atoms, bu, bv, bf = [], [], [], []
    off = 0
    for mol in mols:
        atoms.append(mol.atom_feat)
        bu.append(mol.bond_u + off)
        bv.append(mol.bond_v + off)
        bf.append(mol.bond_feat)
        off += mol.n_atoms
    return MolGraph(np.concatenate(atoms), np.concatenate(bu), np.concatenate(bv), np.concatenate(bf))


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


def make_batch(name: str, cache, rows, y=None):
    marginal = name == "R1_marginal_size"
    b = gather_batch(cache, rows, y, include_size=True, include_edge_mol=marginal)
    return _model_batch(name, b)


def gate_frozen_data() -> None:
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


def gate_size_sidecar() -> None:
    obj = torch.load(SIZE_PATH, map_location="cpu", weights_only=False)
    cache = load_cache(CACHE_PATH)
    n = obj["n_atoms"].numpy()
    m = obj["n_bonds"].numpy()
    comp = obj["n_components"].numpy()
    rings = obj["rings"].numpy()
    feat = obj["size_feat"].numpy()
    expected_feat = np.stack([n / 10.0, m / 10.0, rings], axis=1).astype(np.float32)
    manifest = json.loads(SIZE_MANIFEST.read_text())
    ok(
        "size_features_match_cache_and_formula",
        np.array_equal(n, cache["mol_n_atoms"].numpy())
        and np.array_equal(m, cache["root_n_edges"].numpy() // 2)
        and bool((comp == 1).all())
        and np.array_equal(rings, m - n + comp)
        and np.allclose(feat, expected_feat, atol=1e-6)
        and manifest["size_features"]["sha256"] == sha256_file(SIZE_PATH),
        {
            "n_molecules": int(n.size),
            "all_connected": bool((comp == 1).all()),
            "ring_range": [int(rings.min()), int(rings.max())],
            "sha256": sha256_file(SIZE_PATH),
        },
    )


def gate_params_and_flops(rows_fit, y_fit) -> None:
    cfg = P1CConfig()
    cache = load_cache(CACHE_PATH)
    cache["mol_size_feat"] = torch.load(SIZE_PATH, map_location="cpu", weights_only=False)["size_feat"].float()
    tcfg = TrainConfig(epochs=1, patience=1, batch_size=64, model=cfg)
    counts = {}
    flops = {}
    for name in ["S0", "R1", "S_cap_size", "R1_size", "R1_marginal_size", "M_keep_edge"]:
        torch.manual_seed(0)
        model = build_model(name, int(cache["meta"]["atom_dim"]), int(cache["meta"]["bond_dim"]), cfg)
        counts[name] = parameter_count(model)
        if name in {"S_cap_size", "R1_marginal_size"}:
            kwargs = {"include_size": True, "include_edge_mol": name == "R1_marginal_size"}
        elif name in {"R1_size", "M_keep_edge"}:
            kwargs = {"include_size": True}
        else:
            kwargs = None
        flops[name] = _measure_flops(
            model, name, rows_fit, y_fit, cache, tcfg, torch.device("cpu"), batch_kwargs=kwargs
        ).get("flops_forward")
    ratio = counts["S_cap_size"] / counts["R1_size"]
    ok(
        "parameter_matching_reported",
        counts["R1_size"] == counts["R1_marginal_size"] == EXPECTED["params"]["R1_size"]
        and counts["S_cap_size"] == EXPECTED["params"]["S_cap_size"]
        and abs(counts["S_cap_size"] - counts["R1_size"]) / counts["R1_size"] < 0.005,
        {"params": counts, "s_cap_over_r1_size": ratio},
    )
    ok(
        "forward_flops_measured",
        all(flops[k] and flops[k] > 0 for k in ["S_cap_size", "R1_size", "R1_marginal_size"]),
        {"flops_batch64": flops},
    )
    # Pre-registered optional M_keep_edge compute match: R1_size gets a longer
    # cosine schedule so its forward-FLOP budget matches M_keep_edge at 300 ep.
    if flops.get("R1_size") and flops.get("M_keep_edge"):
        r1_long_epochs = int(round(300.0 * flops["M_keep_edge"] / flops["R1_size"]))
    else:
        r1_long_epochs = None
    ok(
        "m_keep_edge_compute_match_preregistered",
        r1_long_epochs is not None and 300 < r1_long_epochs < 900,
        {
            "r1_size_flops": flops.get("R1_size"),
            "m_keep_edge_flops": flops.get("M_keep_edge"),
            "ratio": (flops["M_keep_edge"] / flops["R1_size"]) if flops.get("R1_size") else None,
            "r1_size_long_epochs_for_matched_flops": r1_long_epochs,
        },
    )


def gate_invariance_and_info() -> None:
    cfg = P1CConfig()
    torch.manual_seed(0)
    r1s = build_model("R1_size", 4, 2, cfg)
    torch.manual_seed(0)
    r1m = build_model("R1_marginal_size", 4, 2, cfg)
    same_state = list(r1s.state_dict()) == list(r1m.state_dict()) and all(
        torch.equal(r1s.state_dict()[k], r1m.state_dict()[k]) for k in r1s.state_dict()
    )
    ok(
        "r1_size_and_marginal_bitwise_identical_init",
        same_state and parameter_count(r1s) == parameter_count(r1m),
        {"params": parameter_count(r1s)},
    )
    mols = [mol_chain(7), mol_chain(6, cycle=True), mol_chain(8)]
    cache = build_cache(mols)
    y = torch.tensor([0.0, 1.0, -1.0])
    trunks = {}
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(0)
        model = build_model(name, 4, 2, cfg)
        trunks[name] = {k: v.clone() for k, v in model.trunk.state_dict().items()}
    ok(
        "shared_trunk_initialisation_identical",
        all(all(torch.equal(trunks["S_cap_size"][k], trunks[n][k]) for k in trunks["S_cap_size"]) for n in ["R1_size", "R1_marginal_size"]),
    )

    perm = np.array([3, 0, 6, 1, 5, 2, 4], dtype=np.int64)
    cache_p = build_cache([permute_mol(mols[0], perm)])
    deltas = {}
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(1)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        b = make_batch(name, cache, [0], torch.tensor([0.0]))
        bp = make_batch(name, cache_p, [0], torch.tensor([0.0]))
        with torch.no_grad():
            deltas[name] = abs(float(model(b).item() - model(bp).item()))
    ok("permutation_invariance", max(deltas.values()) <= 1e-5, deltas)

    batch_deltas = {}
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(2)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        b1 = make_batch(name, cache, [0], torch.tensor([0.0]))
        b3 = make_batch(name, cache, [0, 1, 2], y)
        with torch.no_grad():
            batch_deltas[name] = abs(float(model(b1)[0].item() - model(b3)[0].item()))
    ok("batch_composition_invariance", max(batch_deltas.values()) <= 1e-5, batch_deltas)

    raw = gather_batch(cache, [0, 1, 2], y, include_size=True)
    b = _model_batch("S_cap_size", raw)
    blocked_root = False
    try:
        assert_size_only_batch({**b, "root_edge_index": torch.zeros((2, 0), dtype=torch.long)})
    except AssertionError:
        blocked_root = True
    ok("scap_size_guard_rejects_root_edges", blocked_root and "root_edge_index" not in SIZE_ONLY_KEYS)

    bm = gather_batch(cache, [0, 1, 2], y, include_size=True, include_edge_mol=True)
    bm = {k: v for k, v in bm.items() if k in MARGINAL_SIZE_KEYS}
    assert_marginal_size_batch(bm)
    blocked_endpoint = False
    try:
        assert_marginal_size_batch({**bm, "root_edge_index": torch.zeros((2, 0), dtype=torch.long)})
    except AssertionError:
        blocked_endpoint = True
    ok(
        "marginal_guard_has_no_endpoint_keys",
        blocked_endpoint and "root_edge_index" not in MARGINAL_SIZE_KEYS and "root_batch" not in MARGINAL_SIZE_KEYS,
    )

    # Endpoint-correspondence ablation: permute physical bond types.
    cache_t = build_cache([mol_typed_cycle(8), mol_typed_cycle(7)])
    yt = torch.tensor([0.0, 1.0])
    ablation = {}
    for name in ["R1_size", "R1_marginal_size"]:
        b = make_batch(name, cache_t, [0, 1], yt)
        torch.manual_seed(3)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        mol_ids = b["root_edge_mol"] if name == "R1_marginal_size" else b["root_batch"][b["root_edge_index"][0]]
        unique_mask = b["root_edge_unique"] if "root_edge_unique" in b else b["root_edge_index"][0] < b["root_edge_index"][1]
        unique = np.flatnonzero(unique_mask.numpy())
        rng = np.random.default_rng(7)
        attr = b["root_edge_attr"].clone()
        mol_np = mol_ids.numpy()
        for mol in np.unique(mol_np):
            idx = unique[mol_np[unique] == mol]
            perm_idx = rng.permutation(idx.size)
            for target, source in zip(idx, idx[perm_idx]):
                value = b["root_edge_attr"][source]
                attr[target] = value
                attr[target ^ 1] = value
        b_perm = dict(b)
        b_perm["root_edge_attr"] = attr
        with torch.no_grad():
            ablation[name] = float((model(b) - model(b_perm)).abs().max())
    ok(
        "endpoint_correspondence_ablation_active",
        ablation["R1_size"] > 1e-5 and ablation["R1_marginal_size"] <= 1e-6,
        ablation,
    )

    # Mean-only S0 cannot separate G from its disjoint duplicate; size-aware
    # arms can.
    g = mol_chain(8, cycle=True)
    g2 = disjoint_union([g, g])
    cache_dup = build_cache([g, g2])
    ydup = torch.tensor([0.0, 0.0])
    torch.manual_seed(4)
    s0 = build_model("S0", 4, 2, cfg)
    s0.eval()
    b0 = _model_batch("S0", gather_batch(cache_dup, [0, 1], ydup))
    with torch.no_grad():
        s0_delta = abs(float(s0(b0)[0].item() - s0(b0)[1].item()))
    dup = {"S0": s0_delta}
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(4)
        model = build_model(name, 4, 2, cfg)
        model.eval()
        b = make_batch(name, cache_dup, [0, 1], ydup)
        with torch.no_grad():
            dup[name] = abs(float(model(b)[0].item() - model(b)[1].item()))
    ok("size_readout_separates_duplicate_graph", dup["S0"] <= 1e-6 and all(dup[n] > 1e-5 for n in ["S_cap_size", "R1_size", "R1_marginal_size"]), dup)

    grads = {}
    for name in ["S_cap_size", "R1_size", "R1_marginal_size"]:
        torch.manual_seed(5)
        model = build_model(name, 4, 2, cfg)
        b = make_batch(name, cache, [0, 1, 2], y)
        model(b).sum().backward()
        if name == "S_cap_size":
            grads[name] = float(model.graph_ffns[0].net[0].weight.grad.abs().sum())
        else:
            grads[name] = {
                "relation": float(model.rel.net[0].weight.grad.abs().sum()),
                "edge_readout": float(model.edge_readout.net[0].weight.grad.abs().sum()),
            }
    ok(
        "forward_and_gradients_live",
        grads["S_cap_size"] > 0
        and all(grads[n]["relation"] > 0 and grads[n]["edge_readout"] > 0 for n in ["R1_size", "R1_marginal_size"]),
        grads,
    )


def gate_frozen_anchors() -> None:
    vals = {}
    for name, fname in [("S0_seed0", "S0_seed0.json"), ("R1_seed0", "R1_seed0.json"), ("M2_seed0", "M2_seed0.json"), ("G_ref_seed0", "G_ref_seed0.json")]:
        d = json.loads((RUNS_1B / "zinc" / fname).read_text())
        vals[name] = float(d["best_metrics"]["monitor"]["mae"])
    ok(
        "phase1b_frozen_anchors_unchanged",
        all(abs(vals[k] - EXPECTED["anchors"][k]) < 1e-6 for k in vals),
        vals,
    )


def main() -> int:
    t0 = time.time()
    register_phase1c_models()
    split = json.loads(SPLIT_PATH.read_text())
    rows_fit = [int(x) for x in split["rows"]["fit"]]
    vault = LabelVault(LABELS_PATH, rows_fit, [int(x) for x in split["rows"]["monitor"]], [int(x) for x in split["rows"]["reserved"]])
    vault.load()
    y_fit = torch.full((10000,), float("nan"))
    y_fit[torch.as_tensor(rows_fit, dtype=torch.long)] = vault.labels(rows_fit, "train")

    gate_frozen_data()
    gate_size_sidecar()
    gate_params_and_flops(rows_fit, y_fit)
    gate_invariance_and_info()
    gate_frozen_anchors()

    payload = {
        "protocol_id": "ssrc-phase1c-cpu-gates",
        "n_checks": len(records),
        "n_pass": sum(1 for r in records if r["pass"]),
        "all_pass": all(r["pass"] for r in records),
        "records": records,
        "seconds": time.time() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\n{payload['n_pass']}/{payload['n_checks']} PASS in {payload['seconds']:.1f}s -> {OUT}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
