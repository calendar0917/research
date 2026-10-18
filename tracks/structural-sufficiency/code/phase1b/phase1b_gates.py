"""Phase 1b CPU gates (run before any GPU trajectory).

Covers: frozen split provenance, label isolation, patch equivalence with
Phase 1a, permutation / batch-composition invariance, endpoint-swap
invariance, edge-use sensitivity, tied-iteration identity, collision boundary,
parameter counts and cost measurements.
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

from ssrc_data import (  # noqa: E402
    LIPO_LABELS,
    LIPO_RAW,
    LIPO_SPLIT,
    MolGraph,
    LabelVault,
    build_cache,
    extract_patch_arrays,
    gather_batch,
    load_lipo_mols,
    load_lipo_split,
    row_sha256,
    zinc_internal_split,
)
from ssrc_models import (  # noqa: E402
    GRAPH_ONLY_KEYS,
    PATCH_ONLY_KEYS,
    RELATIONAL_KEYS,
    ModelConfig,
    build_model,
    parameter_count,
    filter_batch,
)
from ssrc_train import TrainConfig, _measure_flops, _measure_step_time, _model_batch, predict_rows  # noqa: E402

OUT = REPO / "tracks/structural-sufficiency/results/phase1b/cpu_gates_phase1b.json"


def _ok(name: str, condition: bool, detail: dict | None = None) -> dict:
    rec = {"check": name, "pass": bool(condition), **(detail or {})}
    print(("PASS " if condition else "FAIL ") + name + (f" {detail}" if detail else ""))
    return rec


def synthetic_mol(n: int, cycle: bool = False, d: int = 4) -> MolGraph:
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
    """New atom i is old atom perm[i]."""
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
    bu = np.asarray([b[0] for b in bonds], dtype=np.int64)
    bv = np.asarray([b[1] for b in bonds], dtype=np.int64)
    bf = mol.bond_feat[[b[2] for b in bonds]].copy()
    return MolGraph(af, bu, bv, bf)


def edge_batch(cache, rows, y=None):
    return gather_batch(cache, rows, y)


def gate_split_and_labels() -> list[dict]:
    out = []
    try:
        from tracks.ksvd.code.run_tccd_v0 import internal_split  # type: ignore

        tr, dev = internal_split(10000)
        s = zinc_internal_split()
        out.append(_ok(
            "zinc_split_matches_tccd_v0",
            np.array_equal(np.asarray(s["fit"]), tr)
            and np.array_equal(np.asarray(s["monitor"] + s["reserved"]), dev),
            {"fit_first": s["fit"][:3], "dev_first": (s["monitor"] + s["reserved"])[:3]},
        ))
    except Exception as exc:  # pragma: no cover
        out.append(_ok("zinc_split_matches_tccd_v0", False, {"error": repr(exc)}))

    lipo = load_lipo_split(LIPO_SPLIT)
    expected = {
        "fit": "c474e38cdecf10071965527409901403ec8b670b96fe4a3c42452531dd0210e9",
        "monitor": "fe81cc7904a6da9378f5bf5ae7452ac71629adad83a2a98b917f87299a85f39f",
        "reserved": "bc28c5d37c2fe2fbe39c01d5bf2ab9c56630eee2d0a4454a108d150ab2f42d86",
    }
    out.append(_ok(
        "lipo_split_matches_phase1a_hashes",
        all(lipo["sha256"][k] == v for k, v in expected.items()),
        lipo["sha256"],
    ))

    split = lipo["split"]
    vault = LabelVault(LIPO_LABELS, split["fit"], split["monitor"], split["reserved"])
    vault.load()
    audit = vault.audit()
    out.append(_ok(
        "lipo_label_vault_isolation",
        audit["reserved_labels_retained"] == 0 and audit["labels_retained"] == 2856,
        audit,
    ))
    try:
        vault.labels(split["reserved"][:2], "monitor")
        reserved_blocked = False
    except PermissionError:
        reserved_blocked = True
    out.append(_ok("reserved_label_request_blocked", reserved_blocked))
    return out


def gate_patch_equivalence() -> list[dict]:
    from phase1a_data import extract_patches as phase1a_extract

    raws = load_lipo_mols(LIPO_RAW)
    # Phase 1a Patch objects are easiest to compare from the raw reader.
    from phase1a_data import load_raw_graphs

    raw_objs = load_raw_graphs(LIPO_RAW)
    max_node_diff = 0.0
    max_edge_diff = 0.0
    max_attr_diff = 0.0
    root_mismatch = 0
    count_mismatch = 0
    for idx in range(20):
        ours = extract_patch_arrays(raws[idx])
        ref = phase1a_extract(raw_objs[idx], radius=2)
        if len(ref) != len(ours["n_nodes"]):
            count_mismatch += 1
            continue
        for pi, p in enumerate(ref):
            if int(ours["roots"][pi]) != int(p.root) or int(ours["n_nodes"][pi]) != p.n_nodes:
                root_mismatch += 1
            a = ours["node_feat"][pi]
            b = p.node_feat
            if a.shape != b.shape:
                count_mismatch += 1
                continue
            max_node_diff = max(max_node_diff, float(np.max(np.abs(a - b))))
            if p.n_edges:
                es = ours["edge_src"][pi]
                ed = ours["edge_dst"][pi]
                ei = np.stack([es, ed], axis=0)
                max_edge_diff = max(max_edge_diff, float(np.max(np.abs(ei - p.edge_index))))
                max_attr_diff = max(max_attr_diff, float(np.max(np.abs(ours["edge_attr"][pi] - p.edge_attr))))
    return [
        _ok("patch_node_features_match_phase1a", max_node_diff <= 1e-7 and root_mismatch == 0 and count_mismatch == 0,
            {"max_abs_diff": max_node_diff, "root_mismatch": root_mismatch, "count_mismatch": count_mismatch}),
        _ok("patch_edge_index_match_phase1a", max_edge_diff == 0.0, {"max_abs_diff": max_edge_diff}),
        _ok("patch_edge_attr_match_phase1a", max_attr_diff <= 1e-7, {"max_abs_diff": max_attr_diff}),
    ]


def gate_invariance_and_edges() -> list[dict]:
    cfg = ModelConfig()
    out = []
    mols = [synthetic_mol(7), synthetic_mol(6, cycle=True), synthetic_mol(8)]
    cache = build_cache(mols)
    y = torch.tensor([0.0, 1.0, -1.0])

    # shared trunk initialisation across the four SSRC arms
    states = {}
    for name in ["S0", "R1", "M1", "M2"]:
        torch.manual_seed(0)
        m = build_model(name, 4, 2, cfg)
        states[name] = {k: v.clone() for k, v in m.trunk.state_dict().items()}
    same_trunk = all(
        all(torch.equal(states["S0"][k], states[n][k]) for k in states["S0"])
        for n in ["R1", "M1", "M2"]
    )
    out.append(_ok("shared_trunk_initialisation_identical", same_trunk))

    # M2 is M1 with the same parameters applied twice
    torch.manual_seed(0)
    m1 = build_model("M1", 4, 2, cfg)
    torch.manual_seed(0)
    m2 = build_model("M2", 4, 2, cfg)
    m2.load_state_dict(m1.state_dict())
    same_m_params = parameter_count(m1) == parameter_count(m2) and all(
        torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters())
    )
    out.append(_ok("m2_parameter_identical_to_m1", same_m_params, {"params": parameter_count(m1)}))

    # permutation invariance
    perm = np.array([3, 0, 6, 1, 5, 2, 4], dtype=np.int64)
    mol = mols[0]
    mol_p = permute_mol(mol, perm)
    cache_p = build_cache([mol_p])
    b = _model_batch("S0", edge_batch(cache, [0], torch.tensor([0.0])))
    bp = _model_batch("S0", edge_batch(cache_p, [0], torch.tensor([0.0])))
    max_perm = 0.0
    for name in ["S0", "R1", "M1", "M2", "G_ref"]:
        torch.manual_seed(1)
        m = build_model(name, 4, 2, cfg)
        m.eval()
        b = _model_batch(name, edge_batch(cache, [0], torch.tensor([0.0])))
        bp = _model_batch(name, edge_batch(cache_p, [0], torch.tensor([0.0])))
        with torch.no_grad():
            max_perm = max(max_perm, float(abs(m(b).item() - m(bp).item())))
    out.append(_ok("permutation_invariance", max_perm <= 1e-5, {"max_abs_diff": max_perm}))

    # batch composition invariance
    max_batch = 0.0
    for name in ["S0", "R1", "M1", "M2", "G_ref"]:
        torch.manual_seed(2)
        m = build_model(name, 4, 2, cfg)
        m.eval()
        b1 = _model_batch(name, edge_batch(cache, [0], torch.tensor([0.0])))
        b3 = _model_batch(name, edge_batch(cache, [0, 1, 2], y))
        with torch.no_grad():
            max_batch = max(max_batch, float(abs(m(b1)[0].item() - m(b3)[0].item())))
    out.append(_ok("batch_composition_invariance", max_batch <= 1e-5, {"max_abs_diff": max_batch}))

    # endpoint swap: r_ij == r_ji
    torch.manual_seed(3)
    r1 = build_model("R1", 4, 2, cfg)
    r1.eval()
    b = _model_batch("R1", edge_batch(cache, [0, 1, 2], y))
    with torch.no_grad():
        token = r1.trunk(b)
        r = r1.rel(token, b["root_edge_index"], b["root_edge_attr"])
        src, dst = b["root_edge_index"]
        lookup = {}
        for k in range(src.numel()):
            lookup[(int(src[k]), int(dst[k]))] = k
        max_swap = 0.0
        for (u, v), k in list(lookup.items()):
            if (v, u) in lookup:
                max_swap = max(max_swap, float((r[k] - r[lookup[(v, u)]]).abs().max()))
    out.append(_ok("edge_relation_endpoint_swap_invariant", max_swap <= 1e-6, {"max_abs_diff": max_swap}))

    # edge use: S0 unchanged, R1/M changed when root endpoints are shuffled
    torch.manual_seed(4)
    b = _model_batch("R1", edge_batch(cache, [0, 1, 2], y))
    b_shuf = dict(b)
    g = torch.Generator().manual_seed(123)
    perm_dst = torch.randperm(b["root_edge_index"].shape[1], generator=g)
    b_shuf["root_edge_index"] = torch.stack([b["root_edge_index"][0], b["root_edge_index"][1][perm_dst]], dim=0)
    deltas = {}
    for name in ["S0", "R1", "M1", "M2"]:
        torch.manual_seed(5)
        m = build_model(name, 4, 2, cfg)
        m.eval()
        with torch.no_grad():
            if name == "S0":
                p0 = m(b)
                p1 = m(b_shuf)
            else:
                p0 = m(b)
                p1 = m(b_shuf)
            deltas[name] = float((p0 - p1).abs().max())
    out.append(_ok("s0_ignores_root_edges", deltas["S0"] == 0.0, {"max_abs_diff": deltas["S0"]}))
    out.append(_ok("r1_m_use_root_edges", min(deltas["R1"], deltas["M1"], deltas["M2"]) > 1e-4, deltas))

    # relation / update gradients are live
    grad_info = {}
    for name in ["R1", "M1", "M2"]:
        torch.manual_seed(6)
        m = build_model(name, 4, 2, cfg)
        b = _model_batch(name, edge_batch(cache, [0, 1, 2], y))
        outv = m(b).sum()
        outv.backward()
        rel_g = float(m.rel.net[0].weight.grad.abs().sum()) if m.rel.net[0].weight.grad is not None else 0.0
        grad_info[name] = {"rel_grad": rel_g}
        if name in {"M1", "M2"}:
            grad_info[name]["update_grad"] = float(m.update[0].weight.grad.abs().sum())
    out.append(_ok(
        "relation_and_update_gradients_live",
        all(v["rel_grad"] > 0 for v in grad_info.values())
        and all(v.get("update_grad", 1.0) > 0 for k, v in grad_info.items() if k in {"M1", "M2"}),
        grad_info,
    ))

    # E2 collision boundary: all-identical C12 vs C14 have the same patch multiset
    def uniform_cycle(n: int) -> MolGraph:
        af = np.zeros((n, 1), dtype=np.float32)
        af[:, 0] = 1.0
        bu = np.arange(n, dtype=np.int64)
        bv = (np.arange(n) + 1) % n
        bf = np.ones((n, 1), dtype=np.float32)
        return MolGraph(af, bu, bv, bf)

    c12 = uniform_cycle(12)
    c14 = uniform_cycle(14)
    cc = build_cache([c12, c14])
    bcc = edge_batch(cc, [0, 1], torch.tensor([0.0, 0.0]))
    coll = {}
    for name in ["S0", "R1", "M1", "M2"]:
        torch.manual_seed(7)
        m = build_model(name, 1, 1, ModelConfig(hidden=32, ffn_hidden=64, rel_hidden=64, g_hidden=32, g_layers=2, g_ffn_hidden=32))
        m.eval()
        bb = _model_batch(name, bcc)
        with torch.no_grad():
            coll[name] = float(abs(m(bb)[0].item() - m(bb)[1].item()))
    out.append(_ok("e2_collision_regular_cycles_unresolved", max(coll.values()) <= 1e-5, coll))
    return out


def gate_phase1a_equivalence() -> list[dict]:
    """Our S0 must be bitwise the Phase 1a ModelS after a weight-name map."""
    from phase1a_models import ModelS

    mols = load_lipo_mols(LIPO_RAW)[:8]
    cache = build_cache(mols)
    b = _model_batch("S0", edge_batch(cache, list(range(8)), torch.arange(8.0)))
    b["patch_mask"] = torch.ones((8, int(cache["mol_n_atoms"][:8].max())), dtype=torch.bool)
    torch.manual_seed(0)
    ref = ModelS(hidden=64, local_layers=2, ffn_layers=2, ffn_hidden=128, dropout=0.1)
    torch.manual_seed(0)
    ours = build_model("S0", 37, 12, ModelConfig())
    mapping = {}
    for k in ours.state_dict():
        if k.startswith("trunk.local.input_proj"):
            nk = "local.input_proj" + k[len("trunk.local.input_proj"):]
        elif k.startswith("trunk.local.layers."):
            parts = k.split(".")
            i, rest = parts[3], ".".join(parts[4:])
            if rest.startswith("conv."):
                nk = f"local.convs.{i}." + rest[5:]
            elif rest.startswith("norm."):
                nk = f"local.norms.{i}." + rest[5:]
            else:
                raise ValueError(k)
        elif k.startswith("trunk.blocks."):
            nk = "blocks." + k[len("trunk.blocks."):]
        elif k.startswith("head."):
            nk = k
        else:
            raise ValueError(k)
        mapping[k] = nk
    ref.load_state_dict({mapping[k]: v for k, v in ours.state_dict().items()})
    ref.eval()
    ours.eval()
    with torch.no_grad():
        a, c = ref(b), ours(b)
    return [_ok("s0_bitwise_matches_phase1a_modelS", float((a - c).abs().max()) <= 1e-7,
                {"max_abs_diff": float((a - c).abs().max()), "params": parameter_count(ours)})]


def gate_cache_metadata() -> list[dict]:
    from ssrc_data import CACHE_DIR, load_cache

    out = []
    expected = {"zinc": (10000, 21, 4, 14), "lipo": (4200, 37, 12, 13)}
    for name, (n_mol, atom_dim, bond_dim, max_patch) in expected.items():
        path = CACHE_DIR / f"{name}_phase1b.pt"
        if not path.exists():
            out.append(_ok(f"{name}_cache_present", False, {"path": str(path)}))
            continue
        cache = load_cache(path)
        meta = cache["meta"]
        no_labels = not any(("label" in k.lower() or k == "y") for k in cache)
        out.append(_ok(
            f"{name}_cache_metadata",
            meta["n_molecules"] == n_mol and meta["atom_dim"] == atom_dim
            and meta["bond_dim"] == bond_dim and meta["max_patch_nodes"] == max_patch
            and meta["radius"] == 2 and no_labels,
            {"meta": meta, "no_label_keys": no_labels},
        ))
    return out


def gate_guards() -> list[dict]:
    cfg = ModelConfig()
    mols = [synthetic_mol(6)]
    cache = build_cache(mols)
    b = edge_batch(cache, [0], torch.tensor([0.0]))
    out = []
    try:
        from ssrc_models import assert_patch_only_batch

        assert_patch_only_batch(b)
        blocked = False
    except AssertionError:
        blocked = True
    out.append(_ok("s0_guard_rejects_root_edges", blocked))
    try:
        from ssrc_models import assert_relational_batch

        assert_relational_batch(filter_batch(b, PATCH_ONLY_KEYS))
        blocked = False
    except AssertionError:
        blocked = True
    out.append(_ok("r1_guard_requires_root_edges", blocked))
    try:
        from ssrc_models import assert_graph_only_batch

        assert_graph_only_batch(b)
        blocked = False
    except AssertionError:
        blocked = True
    out.append(_ok("gref_guard_rejects_patch_fields", blocked))
    return out


def gate_cost() -> list[dict]:
    cfg = ModelConfig()
    mols = [synthetic_mol(8), synthetic_mol(9), synthetic_mol(10, cycle=True)]
    cache = build_cache(mols)
    y = torch.tensor([0.0, 1.0, -1.0])
    tcfg = TrainConfig(epochs=1, patience=1, batch_size=3)
    out = []
    counts = {}
    for name in ["S0", "R1", "M1", "M2", "G_ref"]:
        torch.manual_seed(0)
        m = build_model(name, 4, 2, cfg)
        n = parameter_count(m)
        flops = _measure_flops(m, name, [0, 1, 2], y, cache, tcfg, torch.device("cpu"))
        step = _measure_step_time(m, name, [0, 1, 2], y, cache, tcfg, torch.device("cpu"), n_steps=2)
        counts[name] = {"params": n, "flops": flops.get("flops_forward"), "step_ms": step}
    out.append(_ok(
        "parameter_and_cost_measured",
        all(v["params"] > 0 and v["flops"] for v in counts.values()),
        counts,
    ))
    return out


def main() -> int:
    t0 = time.time()
    records = []
    for fn in [
        gate_split_and_labels,
        gate_patch_equivalence,
        gate_phase1a_equivalence,
        gate_cache_metadata,
        gate_invariance_and_edges,
        gate_guards,
        gate_cost,
    ]:
        records.extend(fn())
    payload = {
        "protocol_id": "ssrc-phase1b-cpu-gates",
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
