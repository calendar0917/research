#!/usr/bin/env python
"""Driver for protocol ``cscl-task-adaptive-structure-units-v1``.

Two-stage, label-disciplined workflow:

  CPU feasibility (no GPU):
    verify-inputs, audit-blocks, cpu-gates   (S1 boundary learning, S2 known truth)

  One frozen GPU pass (after ``freeze``):
    train --arm {F-A,L-A,F-O,L-O} --seed {0,1}
    predict (monitor only), freeze-predictions, score-monitor

  Label-blind structural audits (after training, no labels):
    audit-partition, audit-contribution, audit-sharing, finalize

The Monitor labels of split D are read exactly once, inside ``score-monitor``,
after every prediction file is frozen.  The locked side is never touched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_arch_breakthrough_v1 as ab  # noqa: E402
import bjoc_perfmech_long_v1 as bp  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402
import task_adaptive_structure_units_v1 as tasu  # noqa: E402

REPO_ROOT = _CODE_DIR.parents[2]
TRACK = _CODE_DIR.parent
OUT = TRACK / "results" / "cscl_task_adaptive_structure_units_v1"
PRIOR = TRACK / "results" / "bjoc_task_adaptive_ssl_v1"
ENCODED_TRAIN = REPO_ROOT / "tracks/ksvd/results/zinc_static_dictionary_pair/cache/encoded_train.pt"

PROTOCOL_ID = tasu.PROTOCOL_ID
SPLIT_TAG = "D"
SEEDS = (0, 1)
ARMS = tuple(tasu.ARMS)

#: frozen by the task prompt: pooled monitor delta MAE of L-A vs F-A
H1_DELTA_THRESHOLD = -0.01
#: ordinary rows tolerance (delta in raw-y MAE must not be worse than this)
GATE_ORDINARY_TOL = 0.0
#: RMSE must not degrade by more than this (structural change tolerance)
GATE_RMSE_TOL = 0.02
#: no single molecule may own more than this share of the positive gain
GATE_MAX_SINGLE_GAIN_FRACTION = 0.25
#: opaque reader is capacity matched to the additive heads (+-2%)
OPAQUE_CAPACITY_TOL = 0.02
#: synthetic S1 acceptance
S1_TEST_BRIDGE_ACC_MIN = 0.85
S1_TEST_EXACT_MIN = 0.50
#: split D sizes
N_FIT = 8000
N_MONITOR = 1000
#: budgets
GPU_BUDGET_CAP_HOURS = 2.5
#: read-only reference (BJOC-Null P3 seed-mean monitor MAE, raw y)
BJOC_NULL_REF = {
    "source": "bjoc_task_adaptive_ssl_v1/monitor_TA_D.json",
    "arm": "P3 (BJOC-Null)",
    "monitor_mae_seed_mean": 0.10430100391665473,
    "per_seed": {"0": 0.10408494917506238, "1": 0.10451705865824707},
}
#: decision vocabulary (frozen)
DECISIONS = {
    "STRUCTURE_UNITS_LEARNING_PROMISING": "L-A beats F-A on the frozen H1 gate",
    "STRUCTURE_UNITS_READOUT_BOTTLENECK": "L-A ~ F-A but opaque arms beat additive arms",
    "STRUCTURE_UNITS_NO_GAIN": "no additive boundary gain and no readout gain",
    "STRUCTURE_UNITS_NONIDENTIFIABLE": "boundary learning works on known truth but monitor partition degenerates",
    "STRUCTURE_UNITS_STOP": "no learnable boundaries / boundary scorer dead on known truth",
}


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------


def log(msg: str) -> None:
    print(msg, flush=True)


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=1, sort_keys=False, default=_json_default)
        fh.write("\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def read_json(path: Path | str) -> Any:
    with Path(path).open() as fh:
        return json.load(fh)


def file_sha256(path: Path | str) -> str:
    return bp.file_sha256(Path(path))


def rows_sha256(rows: Sequence[int]) -> str:
    return bp.rows_sha256(rows)


def array_sha256(a: np.ndarray) -> str:
    return bp.array_sha256(np.asarray(a))


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise SystemExit(msg)


def parse_seeds(text: str | None) -> tuple[int, ...]:
    if text is None:
        return SEEDS
    return tuple(int(x) for x in text.split(","))


def parse_arms(text: str | None) -> tuple[str, ...]:
    if text is None:
        return ARMS
    arms = tuple(x.strip() for x in text.split(","))
    for arm in arms:
        require(arm in ARMS, f"unknown arm {arm!r}")
    return arms


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def fit_objects_path() -> Path:
    return PRIOR / "objects_fit_D.pt"


def monitor_objects_path() -> Path:
    return PRIOR / "objects_monitor_D.pt"


def ssl_path() -> Path:
    return PRIOR / "pretrain_D_s0.pt"


def split_path() -> Path:
    return PRIOR / "split_D.json"


def topos_cache_path(zone: str) -> Path:
    return OUT / f"topos_{zone}_{SPLIT_TAG}.pt"


def load_split() -> Mapping[str, Any]:
    return read_json(split_path())


def object_bonds(obj: sro.MolObjects) -> list[tuple[int, int, int]]:
    return [(int(u), int(v), int(bt)) for (u, v), bt in zip(obj.bond_index, obj.bond_type)]


def topology_for(obj: sro.MolObjects) -> tasu.StructureTopology:
    return tasu.build_structure_topology(int(obj.mol_index), object_bonds(obj))


def load_zone(zone: str) -> tuple[list[sro.MolObjects], list[tasu.StructureTopology]]:
    path = fit_objects_path() if zone == "fit" else monitor_objects_path()
    objs = torch.load(path, map_location="cpu", weights_only=False)
    cache = topos_cache_path(zone)
    if cache.exists():
        topos = torch.load(cache, map_location="cpu", weights_only=False)
        require(len(topos) == len(objs), f"cached topology count mismatch for {zone}")
        return objs, topos
    topos = [topology_for(obj) for obj in objs]
    torch.save(topos, cache)
    return objs, topos


def load_fit_y() -> np.ndarray:
    rows = [int(i) for i in load_split()["fit_rows"]]
    require(len(rows) == N_FIT, "split D fit row count changed")
    return bp.load_y_rows(ENCODED_TRAIN, rows)


def load_monitor_y_once() -> np.ndarray:
    lock = OUT / "monitor_read_D.lock"
    require(lock.exists(), "monitor labels are not unlocked yet")
    payload = read_json(lock)
    y = np.asarray(np.load(OUT / "monitor_y_D.npy"), dtype=np.float64).reshape(-1)
    require(array_sha256(y) == payload["y_sha256"], "monitor y drifted from its lock")
    return y


def load_ssl_state() -> Mapping[str, Mapping[str, torch.Tensor]]:
    return torch.load(ssl_path(), map_location="cpu", weights_only=False)


def prediction_dir(zone: str) -> Path:
    return OUT / "predictions" / zone


def prediction_path(zone: str, arm: str, seed: int) -> Path:
    return prediction_dir(zone) / f"{arm}_s{int(seed)}.npz"


def soup_path(arm: str, seed: int) -> Path:
    return OUT / f"soup_{arm}_{SPLIT_TAG}_s{int(seed)}.pt"


def train_json_path(arm: str, seed: int) -> Path:
    return OUT / f"train_{arm}_{SPLIT_TAG}_s{int(seed)}.json"


# ---------------------------------------------------------------------------
# verify-inputs
# ---------------------------------------------------------------------------


def cmd_verify_inputs(_args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "checked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "inputs": {},
    }
    for name, path in (
        ("objects_fit_D.pt", fit_objects_path()),
        ("objects_monitor_D.pt", monitor_objects_path()),
        ("pretrain_D_s0.pt", ssl_path()),
        ("split_D.json", split_path()),
        ("encoded_train.pt", ENCODED_TRAIN),
    ):
        payload["inputs"][name] = {"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path), "exists": path.exists()}
    split = load_split()
    payload["split"] = {
        "n_fit": split["n_fit"],
        "n_monitor": split["n_monitor"],
        "n_locked": split["n_locked"],
        "fit_sha256_rows": split["fit_sha256_rows"],
        "monitor_sha256_rows": split["monitor_sha256_rows"],
        "locked_sha256_rows": split["locked_sha256_rows"],
    }
    payload["prior_art"] = {
        "bjoc_null_monitor_mae": BJOC_NULL_REF,
        "prior_lock_files": sorted(p.name for p in PRIOR.glob("*read_*.lock")),
    }
    payload["guards"] = {
        "monitor_labels_read": (OUT / "monitor_read_D.lock").exists(),
        "locked_labels_read": False,
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "input_audit.json", payload)
    log(f"[verify-inputs] wrote {OUT / 'input_audit.json'}")
    return 0


# ---------------------------------------------------------------------------
# block audit (ZINC applicability, label free)
# ---------------------------------------------------------------------------


def cmd_audit_blocks(args: argparse.Namespace) -> int:
    summary: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "zones": {}}
    ok = True
    for zone in ("fit", "monitor"):
        objs, topos = load_zone(zone)
        n_mols = len(objs)
        n_blocks = np.asarray([int(t.n_blocks) for t in topos])
        n_bridges = np.asarray([len(t.bridge_bond) for t in topos])
        ring_bond_masks = [obj.bond_feat[:, 4] > 0.5 for obj in objs]
        n_ring_bonds = np.asarray([int(np.sum(m)) for m in ring_bond_masks])
        block_sizes: list[int] = []
        n_ring_blocks = 0
        for topo in topos:
            for atoms in topo.block_atom_ids:
                block_sizes.append(len(atoms))
                if len(atoms) > 1:
                    n_ring_blocks += 1
        bs = np.asarray(block_sizes, dtype=np.int64)
        dist = {str(k): int((bs == k).sum()) for k in range(1, int(bs.max()) + 1)}
        pair_counts = np.asarray([int(t.pair_edge_ids.size) for t in topos], dtype=np.int64)
        path_len = np.asarray(
            [int(np.diff(t.pair_edge_offsets).max()) if t.pair_edge_ids.size else 0 for t in topos], dtype=np.int64
        )
        # ring bonds must never be bridges; every bridge must be a raw bond
        ring_bridge_overlap = int(sum(int(np.sum(t.is_bridge & m)) for t, m in zip(topos, ring_bond_masks)))
        bridge_not_bond = int(sum(len(set(t.bridge_bond.tolist()) - set(range(t.n_bonds))) for t in topos))
        zone_payload = {
            "n_molecules": n_mols,
            "n_blocks": {
                "mean": float(n_blocks.mean()),
                "median": float(np.median(n_blocks)),
                "p90": float(np.percentile(n_blocks, 90)),
                "max": int(n_blocks.max()),
            },
            "n_bridges": {
                "mean": float(n_bridges.mean()),
                "median": float(np.median(n_bridges)),
                "p90": float(np.percentile(n_bridges, 90)),
                "max": int(n_bridges.max()),
                "total": int(n_bridges.sum()),
            },
            "single_block_fraction": float((n_blocks == 1).mean()),
            "frac_molecules_with_boundary": float((n_bridges >= 1).mean()),
            "frac_molecules_with_2plus_boundaries": float((n_bridges >= 2).mean()),
            "block_size_histogram": dist,
            "frac_singleton_blocks": float((bs == 1).mean()),
            "frac_ring_blocks": float(n_ring_blocks / max(len(block_sizes), 1)),
            "max_path_length": int(path_len.max()) if path_len.size else 0,
            "mean_pair_entries_per_molecule": float(pair_counts.mean()),
            "max_pair_entries_per_molecule": int(pair_counts.max()),
            "ring_bridge_overlap": ring_bridge_overlap,
            "bridge_not_raw_bond": bridge_not_bond,
            "ring_bond_fraction": float(n_ring_bonds.sum() / max(int(sum(t.n_bonds for t in topos)), 1)),
        }
        if zone == "fit":
            ok = bool(
                zone_payload["frac_molecules_with_boundary"] >= 0.5
                and zone_payload["n_bridges"]["mean"] >= 1.0
                and ring_bridge_overlap == 0
                and bridge_not_bond == 0
            )
        summary["zones"][zone] = zone_payload
    summary["applicability_ok"] = ok
    summary["criterion"] = "fit zone: >=50% molecules have a bridge candidate and no ring bond is a bridge"
    write_json(OUT / "block_audit_D.json", summary)
    log(f"[audit-blocks] applicability_ok={ok}")
    for zone, z in summary["zones"].items():
        log(
            f"  {zone}: blocks={z['n_blocks']['mean']:.2f} bridges={z['n_bridges']['mean']:.2f} "
            f"single-block={z['single_block_fraction']:.4f} ring-blocks={z['frac_ring_blocks']:.3f}"
        )
    return 0 if ok or args.allow_fail else 1


# ---------------------------------------------------------------------------
# structural correctness + forward gates + synthetic known truth
# ---------------------------------------------------------------------------


def structural_correctness(zone: str, n_molecules: int, seed: int) -> dict[str, Any]:
    import networkx as nx

    objs, _ = load_zone(zone)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(objs), size=min(int(n_molecules), len(objs)), replace=False)
    checks: dict[str, Any] = {"n_molecules": int(len(indices))}
    mismatches = 0
    ring_bridge_overlap = 0
    tree_ok = 0
    partition_ok = 0
    path_product_ok = 0
    for i in indices:
        obj = objs[int(i)]
        bonds = object_bonds(obj)
        topo = topology_for(obj)
        ref = tasu.reference_bridges(topo.n_atoms, bonds)
        brute = tasu.brute_force_bridges(topo.n_atoms, bonds)
        g = nx.Graph()
        g.add_nodes_from(range(topo.n_atoms))
        g.add_edges_from((u, v) for u, v, _ in bonds)
        nx_bridges = {tuple(sorted(e)) for e in nx.bridges(g)}
        bridge_atom_pairs = {tuple(sorted(bonds[int(k)][:2])) for k in topo.bridge_bond.tolist()}
        if not (bridge_atom_pairs == ref == brute == nx_bridges):
            mismatches += 1
        ring_mask = obj.bond_feat[:, 4] > 0.5
        ring_bridge_overlap += int(np.sum(topo.is_bridge & ring_mask))
        if len(topo.tree_edge) == int(topo.n_blocks) - 1:
            tree_ok += 1
        if sorted(set(topo.block_of_atom.tolist())) == list(range(int(topo.n_blocks))):
            partition_ok += 1
        # path-product membership must agree with union-find on the block tree
        rng2 = np.random.default_rng(seed * 1000 + int(i))
        gates = rng2.random(len(topo.bridge_bond)) < 0.5
        reps, _ = topo.hard_components(gates)
        good = True
        offsets = topo.pair_edge_offsets
        for e, (u, v) in enumerate(zip(topo.pair_u.tolist(), topo.pair_v.tolist())):
            ids = topo.pair_edge_ids[offsets[e] : offsets[e + 1]]
            prod = float(np.prod(gates[ids])) if len(ids) else 1.0
            same = reps[int(u)] == reps[int(v)]
            if abs(prod - (1.0 if same else 0.0)) > 1e-9:
                good = False
                break
        path_product_ok += int(good)
    checks.update(
        {
            "bridge_set_mismatches": int(mismatches),
            "ring_bridge_overlap": int(ring_bridge_overlap),
            "tree_edge_ok": int(tree_ok),
            "block_partition_ok": int(partition_ok),
            "path_product_ok": int(path_product_ok),
            "all_ok": bool(
                mismatches == 0
                and ring_bridge_overlap == 0
                and tree_ok == len(indices)
                and partition_ok == len(indices)
                and path_product_ok == len(indices)
            ),
        }
    )
    return checks


def _bond_uv(bonds: Sequence[tuple[int, int, int]]):
    for u, v, _ in bonds:
        yield (u, v)


def forward_gates(zone: str, n_molecules: int, seed: int, device: torch.device) -> dict[str, Any]:
    objs, topos = load_zone(zone)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(objs), size=min(int(n_molecules), len(objs)), replace=False)
    objs_sel = [objs[int(i)] for i in idx]
    topo_sel = [topos[int(i)] for i in idx]
    out: dict[str, Any] = {}
    init_hashes: dict[str, str] = {}
    for arm in ARMS:
        part, read = tasu.arm_modes(arm)
        model = tasu.build_model(
            tasu.TASUConfig(), seed=0, partition_mode=part, readout_mode=read,
            global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
            global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
        )
        init_hashes[arm] = tasu.state_dict_hash(model)
        model.eval()
        batch = tasu.batch_atom_bond(objs_sel)
        batch.update(tasu.batch_structure_topology(objs_sel, topo_sel))
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            pred, feats = model(batch, return_features=True)
        n_bridges = int(batch["bridge_bond"].shape[0])
        n_blocks = int(batch["blk_batch"].shape[0])
        row_arm: dict[str, Any] = {
            "n_bridges": n_bridges,
            "n_blocks": n_blocks,
            "init_state_sha256": init_hashes[arm],
            "forbidden_modules": tasu.forbidden_module_scan(model),
        }
        if read == tasu.READOUT_ADDITIVE:
            alpha_sum = _scatter_sum(feats["alpha"].view(-1).cpu().numpy(), batch["blk_batch"].cpu().numpy(), int(batch["n_graphs"]))
            gamma = feats["gamma"].view(-1).cpu().numpy()
            gamma_sum = _scatter_sum(gamma, batch["bridge_graph"].cpu().numpy(), int(batch["n_graphs"]))
            bias = float(model.bias.detach().cpu().item())
            recon = bias + alpha_sum + gamma_sum
            err = float(np.max(np.abs(recon - pred.cpu().numpy())))
            row_arm["decomposition_max_abs_err"] = err
            row_arm["decomposition_ok"] = bool(err < 1e-4)
            row_arm["relation_share"] = float(np.abs(gamma).sum() / max(np.abs(feats["alpha"].view(-1).cpu().numpy()).sum() + np.abs(gamma).sum(), 1e-12))
        if part == tasu.PARTITION_FIXED:
            row_arm["fixed_gates_all_zero"] = bool(float(feats["gates"].abs().max()) == 0.0)
        else:
            logits = feats["merge_logits"]
            row_arm["scorer_init_prob_mean"] = float(torch.sigmoid(logits).mean()) if logits.numel() else None
            row_arm["scorer_init_prob_lt_0p5"] = float((torch.sigmoid(logits) < 0.5).float().mean()) if logits.numel() else None
        out[arm] = row_arm
    out["init_parity_all_arms_equal"] = len(set(init_hashes.values())) == 1
    out["init_parity_sha256"] = init_hashes
    return out


def _scatter_sum(values: np.ndarray, index: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(int(n), dtype=np.float64)
    np.add.at(out, index.astype(np.int64), values.astype(np.float64))
    return out


def gradient_activity(zone: str, n_molecules: int, device: torch.device) -> dict[str, Any]:
    objs, topos = load_zone(zone)
    objs_sel, topo_sel = objs[: int(n_molecules)], topos[: int(n_molecules)]
    batch = tasu.batch_atom_bond(objs_sel)
    batch.update(tasu.batch_structure_topology(objs_sel, topo_sel))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    y = torch.randn(len(objs_sel), device=device)
    out: dict[str, Any] = {}
    for arm in ("L-A", "L-O"):
        part, read = tasu.arm_modes(arm)
        model = tasu.build_model(
            tasu.TASUConfig(), seed=0, partition_mode=part, readout_mode=read,
            global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
            global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
        )
        model.train()
        pred = model(batch)
        loss = torch.nn.functional.l1_loss(pred.view(-1), y)
        model.zero_grad()
        loss.backward()
        out[arm] = {
            "loss": float(loss.detach().cpu()),
            "scorer_last_weight_grad_norm": float(model.merge_scorer.lins[-1].weight.grad.detach().norm()),
        }
        if read == tasu.READOUT_ADDITIVE:
            out[arm]["unit_contrib_grad_norm"] = float(model.unit_contrib.lins[-1].weight.grad.detach().norm())
            out[arm]["relation_contrib_grad_norm"] = float(model.relation_contrib.lins[-1].weight.grad.detach().norm())
        else:
            out[arm]["reader_last_weight_grad_norm"] = float(model.reader.lins[-1].weight.grad.detach().norm())
    return out


def invariance_suite(zone: str, n_molecules: int, seed: int, device: torch.device, n_trials: int = 8) -> dict[str, Any]:
    objs, topos = load_zone(zone)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(objs), size=min(int(n_molecules), len(objs)), replace=False)
    objs_sel = [objs[int(i)] for i in idx]
    topo_sel = [topos[int(i)] for i in idx]
    out: dict[str, Any] = {"n_molecules": len(objs_sel), "trials": int(n_trials)}
    for arm in ("L-A", "F-O", "L-O"):
        part, read = tasu.arm_modes(arm)
        model = tasu.build_model(
            tasu.TASUConfig(), seed=0, partition_mode=part, readout_mode=read,
            global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
            global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
        )
        model.eval()
        base = _predict_objects(model, objs_sel, topo_sel, device)
        max_relabel = 0.0
        max_bond_perm = 0.0
        for _ in range(int(n_trials)):
            p_objs = [_permute_object(o, rng.permutation(int(o.n_atoms))) for o in objs_sel]
            p_topos = [topology_for(o) for o in p_objs]
            pred_p = _predict_objects(model, p_objs, p_topos, device)
            max_relabel = max(max_relabel, float(np.max(np.abs(pred_p - base))))
            b_objs = [_shuffle_bonds(o, rng) for o in objs_sel]
            b_topos = [topology_for(o) for o in b_objs]
            pred_b = _predict_objects(model, b_objs, b_topos, device)
            max_bond_perm = max(max_bond_perm, float(np.max(np.abs(pred_b - base))))
        out[arm] = {
            "max_abs_diff_atom_relabel": max_relabel,
            "max_abs_diff_bond_permutation": max_bond_perm,
            "within_tolerance": bool(max_relabel < 1e-4 and max_bond_perm < 1e-4),
        }
    return out


def _predict_objects(model, objs, topos, device) -> np.ndarray:
    preds = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(objs), 64):
            batch = tasu.batch_atom_bond(list(objs[i : i + 64]))
            batch.update(tasu.batch_structure_topology(list(objs[i : i + 64]), list(topos[i : i + 64])))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            preds.append(model(batch).detach().cpu().numpy())
    return np.concatenate(preds, 0) if preds else np.zeros(0)


def _permute_object(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    new_bonds = np.stack([inv[obj.bond_index[:, 0]], inv[obj.bond_index[:, 1]]], axis=1)
    swap = new_bonds[:, 0] > new_bonds[:, 1]
    new_bonds[swap] = new_bonds[swap][:, ::-1]
    atoms = perm.astype(np.int64)
    return sro.MolObjects(
        mol_index=obj.mol_index,
        smiles=obj.smiles,
        n_atoms=int(obj.n_atoms),
        atom_feat=obj.atom_feat[atoms],
        bond_index=new_bonds,
        bond_type=obj.bond_type.copy(),
        bond_feat=obj.bond_feat.copy(),
        junc_index=obj.junc_index.copy(),
        junc_bond=obj.junc_bond.copy(),
        junc_feat=obj.junc_feat.copy(),
        global_feat=obj.global_feat.copy(),
    )


def _shuffle_bonds(obj: sro.MolObjects, rng: np.random.Generator) -> sro.MolObjects:
    p = rng.permutation(int(obj.bond_index.shape[0]))
    return sro.MolObjects(
        mol_index=obj.mol_index,
        smiles=obj.smiles,
        n_atoms=int(obj.n_atoms),
        atom_feat=obj.atom_feat.copy(),
        bond_index=obj.bond_index[p],
        bond_type=obj.bond_type[p],
        bond_feat=obj.bond_feat[p],
        junc_index=obj.junc_index.copy(),
        junc_bond=obj.junc_bond.copy(),
        junc_feat=obj.junc_feat.copy(),
        global_feat=obj.global_feat.copy(),
    )


def _corr(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"pearson": None, "spearman": None, "n": int(len(x))}
    pearson = float(np.corrcoef(x, y)[0, 1])
    rx = _rank(x)
    ry = _rank(y)
    spearman = float(np.corrcoef(rx, ry)[0, 1])
    return {"pearson": pearson, "spearman": spearman, "n": int(len(x))}


def _rank(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    return ranks


def cmd_cpu_gates(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    result: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "device": "cpu"}
    torch.manual_seed(0)
    np.random.seed(0)

    result["structural_correctness"] = structural_correctness("fit", int(args.struct_molecules), 0)
    result["forward_gates"] = forward_gates("fit", int(args.forward_molecules), 0, device)
    result["invariance"] = invariance_suite("fit", int(args.invariance_molecules), 0, device)
    result["gradient_activity"] = gradient_activity("fit", int(args.grad_molecules), device)
    param_counts = {}
    for arm in ARMS:
        part, read = tasu.arm_modes(arm)
        model = tasu.build_model(
            tasu.TASUConfig(), seed=0, partition_mode=part, readout_mode=read,
            global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
            global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
        )
        audit = tasu.parameter_audit(tasu.TASUConfig(), model)
        active = sum(p.numel() for p in model.parameters() if _param_active(model, p))
        param_counts[arm] = {
            "allocated": audit["actual_parameters"],
            "active_forward": int(active),
            "named": audit["named"],
            "parameter_exact": audit["parameter_exact"],
            "forbidden_modules": tasu.forbidden_module_scan(model),
        }
    result["parameters"] = param_counts
    result["opaque_capacity_ratio_L"] = float(
        param_counts["L-O"]["active_forward"] / max(param_counts["L-A"]["active_forward"], 1)
    )
    result["opaque_capacity_ratio_F"] = float(
        param_counts["F-O"]["active_forward"] / max(param_counts["F-A"]["active_forward"], 1)
    )
    result["opaque_capacity_matched"] = bool(
        abs(result["opaque_capacity_ratio_L"] - 1.0) <= OPAQUE_CAPACITY_TOL
        and abs(result["opaque_capacity_ratio_F"] - 1.0) <= OPAQUE_CAPACITY_TOL
    )

    # ---- S1: direct boundary supervision ---------------------------------
    train = tasu.synthetic_dataset(int(args.synth_train), seed=101)
    test = tasu.synthetic_dataset(int(args.synth_test), seed=202)
    s1 = tasu.train_synthetic_s1(
        seed=0, train_mols=train, test_mols=test, epochs=int(args.s1_epochs), batch_size=64, lr=3e-3
    )
    s1_payload = {
        "epochs": s1["epochs"],
        "final_train": s1["trace"][-1],
        "test_stats": s1["test_stats"],
        "trace_every_10": s1["trace"][9::10],
        "test_bridge_accuracy_min": S1_TEST_BRIDGE_ACC_MIN,
        "test_exact_partition_min": S1_TEST_EXACT_MIN,
        "passed": bool(
            s1["test_stats"]["bridge_accuracy"] >= S1_TEST_BRIDGE_ACC_MIN
            and s1["test_stats"]["exact_partition_accuracy"] >= S1_TEST_EXACT_MIN
        ),
    }
    write_json(OUT / "synthetic_s1.json", s1_payload)
    result["synthetic_s1"] = {k: v for k, v in s1_payload.items() if k != "trace_every_10"}

    # ---- S2: y-only, fixed vs learned vs all-merge vs oracle --------------
    s2_train = tasu.synthetic_dataset(int(args.synth_train), seed=7)
    s2_test = tasu.synthetic_dataset(int(args.synth_test), seed=8)
    s2: dict[str, Any] = {}
    s2_models: dict[str, Any] = {}
    s2_seeds = (0, 1, 2)
    for label, partition, forced in (
        ("F-A", "fixed", None),
        ("L-A", "learned", None),
        ("M-A", "learned", "all"),
        ("O-A", "learned", "oracle"),
    ):
        per_seed = {}
        for seed in s2_seeds:
            res = tasu.train_synthetic_s2(
                seed=seed, train_mols=s2_train, test_mols=s2_test, partition_mode=partition,
                forced=forced, epochs=int(args.s2_epochs), batch_size=64, lr=2e-3,
            )
            per_seed[str(seed)] = {
                "test_mae": res["test_mae"],
                "train_mae_final": res["curve"][-1]["train_mae"],
                "scorer_partition_stats": res["partition_stats"],
            }
            s2_models[f"{label}_s{seed}"] = res["model"]
        s2[label] = {
            "per_seed": per_seed,
            "test_mae_seed_mean": float(np.mean([per_seed[str(s)]["test_mae"] for s in s2_seeds])),
            "test_mae_seed_std": float(np.std([per_seed[str(s)]["test_mae"] for s in s2_seeds])),
            "mean_units_seed_mean": float(
                np.mean([per_seed[str(s)]["scorer_partition_stats"]["mean_units"] for s in s2_seeds])
            ),
        }
    s2["seeds"] = list(s2_seeds)
    s2["oracle_units_mean"] = float(np.mean([m.n_units_oracle for m in s2_train]))
    s2["fixed_units_mean"] = float(np.mean([len(m.true_merge) + 1 for m in s2_train]))
    for label, actual in (("F-A", s2["fixed_units_mean"]), ("M-A", 1.0), ("O-A", s2["oracle_units_mean"])):
        s2[label]["actual_mean_units_seed_mean"] = float(actual)
    s2["L-A"]["actual_mean_units_seed_mean"] = float(s2["L-A"]["mean_units_seed_mean"])
    s2["oracle_merge_fraction"] = float(np.mean([np.mean(m.true_merge) for m in s2_train]))
    s2["grouping_helps"] = bool(s2["O-A"]["test_mae_seed_mean"] < s2["F-A"]["test_mae_seed_mean"])
    s2["learned_uses_grouping"] = bool(s2["L-A"]["test_mae_seed_mean"] < s2["F-A"]["test_mae_seed_mean"])
    s2["pass_criterion"] = (
        "3-seed mean test MAE: O-A < F-A (task rewards grouping) and L-A < F-A (learned uses grouping)"
    )
    s2["passed"] = bool(s2["grouping_helps"] and s2["learned_uses_grouping"])
    s2["oracle_headroom"] = float(s2["F-A"]["test_mae_seed_mean"] - s2["O-A"]["test_mae_seed_mean"])
    s2["learned_vs_all_merge"] = float(s2["L-A"]["test_mae_seed_mean"] - s2["M-A"]["test_mae_seed_mean"])
    s2["learned_overtake_of_all_merge"] = bool(s2["learned_vs_all_merge"] < 0)
    s2["learned_partition_degenerate_seeds"] = [
        s
        for s in s2_seeds
        if s2["L-A"]["per_seed"][str(s)]["scorer_partition_stats"]["mean_units"] < 1.5
    ]

    # alpha / gamma direction recovery on the oracle-partition model
    rec = synthetic_recovery_units(s2_models["O-A_s0"], s2_test, "oracle", device)
    rec_l = synthetic_recovery_units(s2_models["L-A_s0"], s2_test, "learned", device)
    s2["recovery_oracle_partition"] = rec
    s2["recovery_learned_partition"] = rec_l
    write_json(OUT / "synthetic_s2.json", s2)
    result["synthetic_s2"] = {k: v for k, v in s2.items()}
    result["synthetic_s2_models_not_serialised"] = True

    # ---- smoke reproducibility -------------------------------------------
    smoke_objs, smoke_topos = load_zone("fit")
    smoke_objs, smoke_topos = smoke_objs[:64], smoke_topos[:64]
    y = torch.randn(64)
    s1_state = _smoke_train_state(smoke_objs, smoke_topos, y, device, seed=0)
    s2_state = _smoke_train_state(smoke_objs, smoke_topos, y, device, seed=0)
    max_diff = max(float((s1_state[k] - s2_state[k]).abs().max()) for k in s1_state)
    result["smoke_reproducibility"] = {
        "max_abs_parameter_diff": max_diff,
        "within_noise": bool(max_diff < 1e-6),
        "hash_1": _state_sha(s1_state),
        "hash_2": _state_sha(s2_state),
    }

    checks = {
        "structural_correctness": bool(result["structural_correctness"]["all_ok"]),
        "init_parity_all_arms": bool(result["forward_gates"]["init_parity_all_arms_equal"]),
        "additive_decomposition": all(
            result["forward_gates"][arm].get("decomposition_ok", True) for arm in ("F-A", "L-A")
        ),
        "fixed_arm_zero_gates": bool(result["forward_gates"]["F-A"]["fixed_gates_all_zero"]),
        "scorer_init_cuts": bool(
            result["forward_gates"]["L-A"]["scorer_init_prob_lt_0p5"] is not None
            and result["forward_gates"]["L-A"]["scorer_init_prob_lt_0p5"] > 0.95
        ),
        "invariance": all(
            result["invariance"][arm]["within_tolerance"] for arm in ("L-A", "F-O", "L-O")
        ),
        "scorer_receives_gradient": all(
            result["gradient_activity"][arm]["scorer_last_weight_grad_norm"] > 0 for arm in ("L-A", "L-O")
        ),
        "parameters_exact": all(p["parameter_exact"] for p in param_counts.values()),
        "no_forbidden_modules": all(not p["forbidden_modules"] for p in param_counts.values()),
        "opaque_capacity_matched": bool(result["opaque_capacity_matched"]),
        "synthetic_s1": bool(s1_payload["passed"]),
        "synthetic_s2": bool(s2["passed"]),
        "smoke_reproducible": bool(result["smoke_reproducibility"]["within_noise"]),
    }
    result["checks"] = checks
    result["ok"] = bool(all(checks.values()))
    write_json(OUT / "cpu_gates.json", result)
    log(f"[cpu-gates] ok={result['ok']}")
    for name, val in checks.items():
        log(f"  {'PASS' if val else 'FAIL'} {name}")
    return 0 if result["ok"] or args.allow_fail else 1


def _param_active(model, param) -> bool:
    arm_active = {
        "merge_scorer": model.partition_mode == tasu.PARTITION_LEARNED,
        "unit_contrib": model.readout_mode == tasu.READOUT_ADDITIVE,
        "relation_contrib": model.readout_mode == tasu.READOUT_ADDITIVE,
        "reader": model.readout_mode == tasu.READOUT_OPAQUE,
        "bias": model.readout_mode == tasu.READOUT_ADDITIVE,
    }
    for name, module in model.named_children():
        for p in module.parameters(recurse=True):
            if p is param:
                return bool(arm_active.get(name, True))
    return True


def _smoke_train_state(objs, topos, y, device, seed: int) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    model = tasu.build_model(
        tasu.TASUConfig(), seed=seed, partition_mode="learned", readout_mode="additive",
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = tasu.batch_atom_bond(list(objs))
    batch.update(tasu.batch_structure_topology(list(objs), list(topos)))
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    for _ in range(3):
        pred = model(batch)
        loss = torch.nn.functional.l1_loss(pred.view(-1), y.to(device))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return tasu.state_dict_to_cpu(model)


def _state_sha(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state.items()):
        h.update(k.encode())
        h.update(np.ascontiguousarray(v.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def synthetic_recovery_units(
    model: tasu.TaskAdaptiveStructureUnits,
    mols: Sequence[tasu.SynthMolecule],
    partition: str,
    device: torch.device,
) -> dict[str, Any]:
    """Correlate per-unit alpha and per-cut-bridge gamma with oracle values."""
    dev = torch.device(device)
    model.eval()
    objs, topos, _ = tasu.synthetic_objects_topos(list(mols))
    alpha_vals: list[float] = []
    alpha_target: list[float] = []
    gamma_vals: list[float] = []
    gamma_target: list[float] = []
    n_units_total = 0
    n_relations_total = 0
    for i in range(0, len(objs), 64):
        chunk_objs, chunk_topos = objs[i : i + 64], topos[i : i + 64]
        chunk_mols = list(mols[i : i + 64])
        batch = tasu.synthetic_batch(chunk_objs, chunk_topos)
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            _, feats = model(batch, return_features=True)
        alpha = feats["alpha"].view(-1).cpu().numpy()
        gamma = feats["gamma"].view(-1).cpu().numpy()
        rel_weight = feats["rel_weight"].cpu().numpy()
        blk_off = batch["blk_off"].cpu().numpy()
        base_bridge = 0
        for g, mol in enumerate(chunk_mols):
            unit_w: dict[int, float] = {}
            for a, u in enumerate(mol.unit_of_atom):
                unit_w[int(u)] = unit_w.get(int(u), 0.0) + float(mol.atom_weight[a])
            for u, w in unit_w.items():
                alpha_vals.append(float(alpha[int(blk_off[g]) + int(u)]))
                alpha_target.append(
                    tasu.SYNTH_SCALE_UNARY_LINEAR * w + tasu.SYNTH_SCALE_UNARY_SQUARE * w * w / 10.0
                )
                n_units_total += 1
            for k, merged in enumerate(mol.true_merge):
                if merged:
                    continue
                bu, bv = mol.tree_edge[k]
                wu = sum(float(mol.atom_weight[a]) for a in mol.block_atom_ids[bu])
                wv = sum(float(mol.atom_weight[a]) for a in mol.block_atom_ids[bv])
                gamma_vals.append(float(gamma[base_bridge + k]))
                gamma_target.append(tasu.SYNTH_SCALE_RELATION * wu * wv / 10.0)
                n_relations_total += 1
            base_bridge += len(mol.true_merge)
    return {
        "partition": partition,
        "n_units": int(n_units_total),
        "n_relations": int(n_relations_total),
        "alpha_vs_oracle_unit_term": _corr(alpha_vals, alpha_target),
        "gamma_vs_oracle_relation_term": _corr(gamma_vals, gamma_target),
    }


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def frozen_config_hash() -> str:
    cfg = tasu.TASUConfig()
    payload = {k: getattr(cfg, k) for k in sorted(cfg.__dataclass_fields__.keys())}
    payload["merge_init_bias"] = tasu.MERGE_INIT_BIAS
    payload["arms"] = list(ARMS)
    payload["recipe"] = tasu.RECIPE_R3
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def cmd_freeze(args: argparse.Namespace) -> int:
    gates = read_json(OUT / "cpu_gates.json")
    require(bool(gates.get("ok")), "cpu-gates not passing; refusing to freeze")
    blocks = read_json(OUT / "block_audit_D.json")
    require(bool(blocks.get("applicability_ok")), "block audit not applicable; refusing to freeze")
    code_files = [_CODE_DIR / "task_adaptive_structure_units_v1.py", _CODE_DIR / "run_task_adaptive_structure_units_v1.py"]
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": {p.name: file_sha256(p) for p in code_files},
        "config": {k: getattr(tasu.TASUConfig(), k) for k in sorted(tasu.TASUConfig().__dataclass_fields__.keys())},
        "config_sha256": frozen_config_hash(),
        "merge_init_bias": tasu.MERGE_INIT_BIAS,
        "recipe": tasu.RECIPE_R3,
        "arms": list(ARMS),
        "seeds": list(SEEDS),
        "split_tag": SPLIT_TAG,
        "init_seed": 0,
        "ssl_init": {"path": str(ssl_path().relative_to(REPO_ROOT)), "sha256": file_sha256(ssl_path()), "encoders_only": True},
        "inputs": {
            "objects_fit_D.pt": file_sha256(fit_objects_path()),
            "objects_monitor_D.pt": file_sha256(monitor_objects_path()),
            "split_D.json": file_sha256(split_path()),
        },
        "monitor": {
            "gate": "paired seed-mean raw-y MAE",
            "delta_threshold": H1_DELTA_THRESHOLD,
            "ordinary_tol": GATE_ORDINARY_TOL,
            "rmse_tol": GATE_RMSE_TOL,
            "max_single_gain_fraction": GATE_MAX_SINGLE_GAIN_FRACTION,
            "two_seeds": True,
            "reference": BJOC_NULL_REF,
        },
        "budget": {"gpu_cap_hours": GPU_BUDGET_CAP_HOURS, "runs_planned": len(ARMS) * len(SEEDS)},
        "decisions": DECISIONS,
        "read_discipline": {
            "monitor": "read once inside score-monitor after predictions are frozen",
            "locked": "never read by this protocol",
            "official_valid_test": "never loaded",
        },
        "notes": [
            "F-A/L-A use the additive readout; F-O/L-O use a capacity-matched opaque reader",
            "the merge scorer receives the straight-through surrogate from the raw-y L1 loss",
            "learned boundaries are audited label-blind before monitor labels are read",
        ],
    }
    write_json(OUT / "protocol_freeze.json", payload)
    (OUT / "CHECKS.md").write_text(_render_checks_md(gates), encoding="utf-8")
    (OUT / "RUNBOOK.md").write_text(_render_runbook_md(), encoding="utf-8")
    log(f"[freeze] config_sha256={payload['config_sha256'][:12]}")
    return 0


def _render_checks_md(gates: Mapping[str, Any]) -> str:
    lines = [f"# {PROTOCOL_ID} CPU feasibility gates", ""]
    lines.append(f"- git commit: `{gates['git_commit']}`")
    lines.append(f"- overall: **{'PASS' if gates['ok'] else 'FAIL'}**")
    lines.append("")
    lines.append("| check | result |")
    lines.append("| --- | --- |")
    for name, val in gates["checks"].items():
        lines.append(f"| {name} | {'PASS' if val else 'FAIL'} |")
    lines.append("")
    lines.append("## structural correctness")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(gates["structural_correctness"], indent=1))
    lines.append("```")
    lines.append("")
    lines.append("## synthetic S1 (explicit boundary supervision)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(gates["synthetic_s1"], indent=1))
    lines.append("```")
    lines.append("")
    lines.append("## synthetic S2 (y-only, known truth)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(gates["synthetic_s2"], indent=1))
    lines.append("```")
    return "\n".join(lines) + "\n"


def _render_runbook_md() -> str:
    return f"""# {PROTOCOL_ID} runbook

Single frozen pass, four arms, two seeds:

```
python tracks/structure-contribution/code/run_task_adaptive_structure_units_v1.py verify-inputs
python ... audit-blocks
python ... cpu-gates          # S1 + S2 + forward/invariance/gradient gates
python ... freeze
# remote (res, GPU):
python ... train --arm F-A --seed 0
python ... train --arm F-A --seed 1
... 8 runs total ...
python ... predict --arm F-A --seed 0   # monitor only, no labels
... 8 predictions ...
python ... freeze-predictions
python ... score-monitor      # single monitor label read
python ... audit-partition    # label blind
python ... audit-contribution
python ... audit-sharing
python ... finalize
```

`audit-*` and `finalize` must run before any interpretation of monitor scores.
"""


def verify_freeze(require_exact_code: bool = True) -> Mapping[str, Any]:
    freeze_path = OUT / "protocol_freeze.json"
    require(freeze_path.exists(), "protocol_freeze.json missing; run freeze first")
    payload = read_json(freeze_path)
    require(payload["protocol"] == PROTOCOL_ID, "freeze protocol id mismatch")
    if require_exact_code:
        for name, sha in payload["code"].items():
            require(file_sha256(_CODE_DIR / name) == sha, f"code drift detected in {name}; re-freeze is forbidden")
    require(frozen_config_hash() == payload["config_sha256"], "model config drifted from freeze")
    return payload


# ---------------------------------------------------------------------------
# train / predict
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=True)
    arm = args.arm
    seed = int(args.seed)
    device = torch.device(args.device)
    objs, topos = load_zone("fit")
    y = load_fit_y()
    ssl = load_ssl_state() if not args.no_ssl else None
    probe_objs, probe_topos = objs[:1024], topos[:1024]
    recipe = dict(tasu.RECIPE_R3)
    if args.epochs is not None:
        require(args.smoke, "epochs override is only allowed with --smoke (smoke runs are never scored)")
        recipe["epochs"] = int(args.epochs)
    log(f"[train {arm} s{seed}] device={device} epochs={recipe['epochs']} ssl={'yes' if ssl else 'no'}")
    model, info = tasu.train_supervised_tasu(
        arm=arm, seed=seed, train_objs=objs, train_topos=topos, y_fit=y, device=device,
        ssl_state=ssl, recipe=recipe, probe_objs=probe_objs, probe_topos=probe_topos,
        probe_every=int(args.probe_every), log_fn=log,
    )
    info["smoke"] = bool(args.smoke)
    if args.smoke:
        info["note"] = "smoke run: excluded from the frozen prediction manifest"
        write_json(OUT / f"smoke_train_{arm}_s{seed}.json", info)
        return 0
    torch.save(tasu.state_dict_to_cpu(model), soup_path(arm, seed))
    write_json(train_json_path(arm, seed), info)
    _append_ledger({
        "arm": arm,
        "seed": seed,
        "wall_seconds": info["wall_seconds"],
        "device": info["device"],
        "epochs": info["epochs"],
    })
    log(f"[train {arm} s{seed}] soup_fit_mae={info['soup_fit_mae']:.6f} wall={info['wall_seconds']:.0f}s")
    return 0


def _append_ledger(entry: Mapping[str, Any]) -> None:
    path = OUT / "gpu_budget_ledger.json"
    payload = read_json(path) if path.exists() else {"protocol": PROTOCOL_ID, "entries": []}
    payload["entries"].append({**entry, "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    total = float(sum(float(e["wall_seconds"]) for e in payload["entries"]))
    payload["total_wall_hours"] = total / 3600.0
    payload["cap_hours"] = GPU_BUDGET_CAP_HOURS
    payload["within_cap"] = bool(total / 3600.0 <= GPU_BUDGET_CAP_HOURS)
    write_json(path, payload)
    log(f"[ledger] total {payload['total_wall_hours']:.3f} h / cap {GPU_BUDGET_CAP_HOURS} h within={payload['within_cap']}")


def cmd_predict(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    arm, seed = args.arm, int(args.seed)
    require(soup_path(arm, seed).exists(), f"missing soup for {arm} s{seed}")
    device = torch.device(args.device)
    objs, topos = load_zone("monitor")
    part, read = tasu.arm_modes(arm)
    model = tasu.build_model(
        tasu.TASUConfig(), seed=0, partition_mode=part, readout_mode=read,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
    )
    model.load_state_dict({k: v.to(device) for k, v in torch.load(soup_path(arm, seed), map_location="cpu").items()})
    model.eval()
    preds = []
    merge_frac = []
    n_units = []
    rel_counts = []
    with torch.no_grad():
        for i in range(0, len(objs), 128):
            chunk_objs, chunk_topos = list(objs[i : i + 128]), list(topos[i : i + 128])
            batch = tasu.batch_atom_bond(chunk_objs)
            batch.update(tasu.batch_structure_topology(chunk_objs, chunk_topos))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            pred, feats = model(batch, return_features=True)
            preds.append(pred.detach().cpu().numpy())
            gates = feats["gates"].detach().cpu().numpy()
            rel_weight = feats["rel_weight"].detach().cpu().numpy()
            off = 0
            for topo in chunk_topos:
                nb = len(topo.bridge_bond)
                g = gates[off : off + nb]
                off += nb
                merge_frac.append(float(g.mean()) if nb else 0.0)
                _, nu = topo.hard_components(g > 0.5)
                n_units.append(int(nu))
                rel_counts.append(int(np.sum(rel_weight >= 0.5)))
    preds = np.concatenate(preds, 0) if preds else np.zeros(0)
    out_path = prediction_path("monitor", arm, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        pred=preds.astype(np.float32),
        merge_fraction=np.asarray(merge_frac, dtype=np.float32),
        n_units=np.asarray(n_units, dtype=np.int32),
        n_retained_relations=np.asarray(rel_counts, dtype=np.int32),
        arm=arm,
        seed=seed,
        n_molecules=len(preds),
    )
    meta = {
        "arm": arm, "seed": seed, "n_molecules": int(len(preds)),
        "pred_sha256": array_sha256(preds), "file_sha256": file_sha256(out_path),
        "mean_merge_fraction": float(np.mean(merge_frac)) if merge_frac else 0.0,
        "mean_units": float(np.mean(n_units)) if n_units else 0.0,
        "labels_used": "none",
    }
    manifest_path = OUT / "prediction_manifest_D.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"protocol": PROTOCOL_ID, "records": {}}
    manifest["records"][f"{arm}_s{seed}"] = meta
    write_json(manifest_path, manifest)
    log(f"[predict {arm} s{seed}] n={len(preds)} mean_merge={meta['mean_merge_fraction']:.3f} mean_units={meta['mean_units']:.2f}")
    return 0


def cmd_freeze_predictions(args: argparse.Namespace) -> int:
    manifest_path = OUT / "prediction_manifest_D.json"
    manifest = read_json(manifest_path)
    expected = {f"{arm}_s{seed}" for arm in ARMS for seed in SEEDS}
    missing = sorted(expected - set(manifest["records"]))
    require(not missing, f"missing predictions: {missing}")
    manifest["complete"] = True
    manifest["n_predictions"] = len(manifest["records"])
    manifest["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps({k: v["pred_sha256"] for k, v in sorted(manifest["records"].items())}, sort_keys=True).encode()
    ).hexdigest()
    write_json(manifest_path, manifest)
    log(f"[freeze-predictions] {manifest['n_predictions']} files frozen sha={manifest['manifest_sha256'][:12]}")
    return 0


def _verify_prediction_manifest() -> Mapping[str, Any]:
    manifest = read_json(OUT / "prediction_manifest_D.json")
    require(bool(manifest.get("complete")), "prediction manifest is not complete")
    for key, rec in manifest["records"].items():
        p = OUT / "predictions" / "monitor" / f"{key}.npz"
        require(p.exists(), f"prediction file missing: {p}")
        require(file_sha256(p) == rec["file_sha256"], f"prediction file drifted: {key}")
    return manifest


# ---------------------------------------------------------------------------
# monitor scoring (single label read)
# ---------------------------------------------------------------------------


def cmd_score_monitor(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    require(not (OUT / "monitor_read_D.lock").exists(), "monitor labels were already read")
    manifest = _verify_prediction_manifest()
    rows = [int(i) for i in load_split()["monitor_rows"]]
    require(len(rows) == N_MONITOR, "monitor row count changed")
    y = bp.load_y_rows(ENCODED_TRAIN, rows)
    np.save(OUT / "monitor_y_D.npy", y)
    write_json(
        OUT / "monitor_read_D.lock",
        {
            "protocol": PROTOCOL_ID,
            "read_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_rows": len(y),
            "y_sha256": array_sha256(y),
            "rows_sha256": rows_sha256(rows),
            "prediction_manifest_sha256": manifest["manifest_sha256"],
        },
    )
    preds = {
        arm: {int(s): np.asarray(np.load(prediction_path("monitor", arm, s))["pred"], dtype=np.float64) for s in SEEDS}
        for arm in ARMS
    }
    scores = {arm: {str(s): float(np.abs(preds[arm][s] - y).mean()) for s in SEEDS} for arm in ARMS}
    for arm in ARMS:
        scores[arm]["seed_mean"] = float(np.mean([scores[arm][str(s)] for s in SEEDS]))
    contrasts = {
        "L-A_vs_F-A": ab.compute_contrast(preds["L-A"], preds["F-A"], y),
        "F-O_vs_F-A": ab.compute_contrast(preds["F-O"], preds["F-A"], y),
        "L-O_vs_L-A": ab.compute_contrast(preds["L-O"], preds["L-A"], y),
        "L-O_vs_F-O": ab.compute_contrast(preds["L-O"], preds["F-O"], y),
        "L-A_vs_L-O": ab.compute_contrast(preds["L-A"], preds["L-O"], y),
    }
    h1 = _h1_gate(contrasts["L-A_vs_F-A"])
    readout = {
        "opaque_better_than_additive_learned": bool(contrasts["L-O_vs_L-A"]["delta_seed_mean"] < 0),
        "opaque_better_than_additive_fixed": bool(contrasts["F-O_vs_F-A"]["delta_seed_mean"] < 0),
        "learned_boundary_gain_under_opaque": bool(contrasts["L-O_vs_F-O"]["delta_seed_mean"] < 0),
    }
    decision = "STRUCTURE_UNITS_NO_GAIN"
    if h1["passed"] and not readout["opaque_better_than_additive_learned"]:
        decision = "STRUCTURE_UNITS_LEARNING_PROMISING"
    elif readout["opaque_better_than_additive_learned"] and not h1["passed"]:
        decision = "STRUCTURE_UNITS_READOUT_BOTTLENECK"
    payload = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "split_tag": SPLIT_TAG,
        "seeds": list(SEEDS),
        "arm_mae": scores,
        "bjoc_null_reference": BJOC_NULL_REF,
        "contrasts": contrasts,
        "h1_gate": h1,
        "readout_gate": readout,
        "decision": decision,
        "prediction_manifest_sha256": manifest["manifest_sha256"],
        "monitor_read_utc": read_json(OUT / "monitor_read_D.lock")["read_utc"],
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "monitor_score_D.json", payload)
    write_json(OUT / "decision.json", {
        "protocol": PROTOCOL_ID,
        "decision": decision,
        "h1_passed": h1["passed"],
        "h1_delta_seed_mean": contrasts["L-A_vs_F-A"]["delta_seed_mean"],
        "readout": readout,
        "locked_read_authorized": bool(h1["passed"]),
        "official_test_loaded": False,
    })
    log(f"[score-monitor] decision={decision} h1={'PASS' if h1['passed'] else 'FAIL'}")
    for arm in ARMS:
        log(f"  {arm}: seed-mean MAE {scores[arm]['seed_mean']:.6f}")
    return 0


def _h1_gate(contrast: Mapping[str, Any]) -> dict[str, Any]:
    per_seed_ok = all(v < 0 for v in contrast["per_seed_delta"].values())
    conditions = {
        "seed_mean_delta_le_threshold": bool(contrast["delta_seed_mean"] <= H1_DELTA_THRESHOLD),
        "per_seed_all_negative": bool(per_seed_ok),
        "bootstrap_ci_upper_negative": bool(contrast["bootstrap"]["ci_hi"] < 0),
        "ordinary_not_worse": bool(contrast["ordinary_delta"] <= GATE_ORDINARY_TOL),
        "top10_removed_negative": bool(contrast["top10_removed_delta"] < 0),
        "rmse_within_tolerance": bool(contrast["rmse_new"] - contrast["rmse_ref"] <= GATE_RMSE_TOL),
        "no_single_row_dominance": bool(contrast["max_single_gain_fraction"] <= GATE_MAX_SINGLE_GAIN_FRACTION),
    }
    return {
        "two_seeds": True,
        "delta_threshold": H1_DELTA_THRESHOLD,
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }


# ---------------------------------------------------------------------------
# label-blind structural audits
# ---------------------------------------------------------------------------


def _unit_induced_graph(obj: sro.MolObjects, atoms: Sequence[int]):
    import networkx as nx

    atoms = [int(a) for a in atoms]
    index = {a: i for i, a in enumerate(atoms)}
    g = nx.Graph()
    for a in atoms:
        g.add_node(index[a], label=tuple(np.round(obj.atom_feat[a], 4).tolist()))
    for (u, v), bt in zip(obj.bond_index, obj.bond_type):
        u, v = int(u), int(v)
        if u in index and v in index:
            g.add_edge(index[u], index[v], type=int(bt))
    return g


def _unit_wl_hash(g) -> str:
    import networkx as nx

    return nx.weisfeiler_lehman_graph_hash(g, node_attr="label", edge_attr="type", iterations=4)


def unit_records_for_zone(zone: str, arm: str, seed: int, device: torch.device, limit: int | None = None) -> list[dict[str, Any]]:
    """Per-molecule retained-unit records (labels never touched)."""
    objs, topos = load_zone(zone)
    if limit is not None:
        objs, topos = objs[: int(limit)], topos[: int(limit)]
    # rebuild the arm model from its soup (learned arms) or init (fixed arms)
    part, read = tasu.arm_modes(arm)
    model = tasu.build_model(
        tasu.TASUConfig(), seed=0, partition_mode=part, readout_mode=read,
        global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
        global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
    )
    if soup_path(arm, seed).exists():
        model.load_state_dict({k: v.to(device) for k, v in torch.load(soup_path(arm, seed), map_location="cpu").items()})
    model.eval()
    records: list[dict[str, Any]] = []
    with torch.no_grad():
        for i in range(0, len(objs), 64):
            chunk_objs, chunk_topos = list(objs[i : i + 64]), list(topos[i : i + 64])
            batch = tasu.batch_atom_bond(chunk_objs)
            batch.update(tasu.batch_structure_topology(chunk_objs, chunk_topos))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            _, feats = model(batch, return_features=True)
            u = feats["unit_repr"].detach().cpu().numpy()
            alpha = feats["alpha"].detach().cpu().numpy().reshape(-1) if "alpha" in feats else np.zeros(u.shape[0])
            s = feats["s"].detach().cpu().numpy()
            blk_off = batch["blk_off"].cpu().numpy()
            n_blocks = int(batch["blk_batch"].shape[0])
            for g, topo in enumerate(chunk_topos):
                base = int(blk_off[g])
                selector = s[g] > 0.5
                units = []
                for k in range(int(topo.n_blocks)):
                    if not bool(selector[k]):
                        continue
                    atoms = topo.block_atom_ids[k]
                    graph = _unit_induced_graph(chunk_objs[g], atoms)
                    units.append(
                        {
                            "block": int(k),
                            "atom_ids": [int(a) for a in atoms],
                            "n_atoms": len(atoms),
                            "wl_hash": _unit_wl_hash(graph),
                            "repr": [float(x) for x in u[base + k]],
                            "alpha": float(alpha[base + k]),
                        }
                    )
                records.append(
                    {
                        "mol_index": int(chunk_objs[g].mol_index),
                        "smiles": chunk_objs[g].smiles,
                        "n_blocks": int(topo.n_blocks),
                        "n_bridges": int(len(topo.bridge_bond)),
                        "n_units": len(units),
                        "units": units,
                    }
                )
    return records


def cmd_audit_partition(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    device = torch.device(args.device)
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit(), "zones": {}}
    for zone in ("monitor",):
        zone_payload: dict[str, Any] = {}
        for arm in ("F-A", "L-A"):
            records = unit_records_for_zone(zone, arm, int(args.seed), device, limit=args.limit)
            merge_fractions = []
            n_units = []
            all_merge = 0
            all_cut = 0
            for rec in records:
                if rec["n_bridges"]:
                    merge_fractions.append(1.0 - rec["n_units"] / max(rec["n_bridges"] + 1, 1))
                n_units.append(rec["n_units"])
                if rec["n_bridges"] and rec["n_units"] == 1:
                    all_merge += 1
                if rec["n_bridges"] and rec["n_units"] == rec["n_bridges"] + 1:
                    all_cut += 1
            zone_payload[arm] = {
                "n_molecules": len(records),
                "mean_units": float(np.mean(n_units)) if n_units else 0.0,
                "mean_bridges": float(np.mean([r["n_bridges"] for r in records])) if records else 0.0,
                "mean_merge_fraction": float(np.mean(merge_fractions)) if merge_fractions else 0.0,
                "all_merge_fraction": float(all_merge / max(len(records), 1)),
                "all_cut_fraction": float(all_cut / max(len(records), 1)),
                "unit_size_histogram": _hist([u["n_atoms"] for r in records for u in r["units"]]),
                "examples_all_merge": [r["smiles"] for r in records if r["n_bridges"] and r["n_units"] == 1][:5],
                "examples_all_cut": [r["smiles"] for r in records if r["n_bridges"] and r["n_units"] == r["n_bridges"] + 1][:5],
            }
        payload["zones"][zone] = zone_payload
    if "L-A" in payload["zones"]["monitor"]:
        la = payload["zones"]["monitor"]["L-A"]
        payload["learned_partition_degenerate"] = bool(
            la["all_merge_fraction"] > 0.5 or la["all_cut_fraction"] > 0.5
        )
    write_json(OUT / "partition_audit_D.json", payload)
    log(f"[audit-partition] degenerate={payload.get('learned_partition_degenerate')}")
    return 0


def _hist(values: Sequence[float]) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[str(int(v))] = out.get(str(int(v)), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: int(kv[0])))


def cmd_audit_contribution(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    device = torch.device(args.device)
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit()}
    # 1) decomposition identity on monitor
    objs, topos = load_zone("monitor")
    identity: dict[str, Any] = {}
    for arm in ("F-A", "L-A"):
        part, read = tasu.arm_modes(arm)
        model = tasu.build_model(
            tasu.TASUConfig(), seed=0, partition_mode=part, readout_mode=read,
            global_mean=np.zeros(sro.GLOBAL_DIM, dtype=np.float32),
            global_scale=np.ones(sro.GLOBAL_DIM, dtype=np.float32), device=device,
        )
        model.load_state_dict({k: v.to(device) for k, v in torch.load(soup_path(arm, int(args.seed)), map_location="cpu").items()})
        model.eval()
        max_err = 0.0
        alpha_abs = 0.0
        gamma_abs = 0.0
        with torch.no_grad():
            for i in range(0, len(objs), 128):
                chunk_objs, chunk_topos = list(objs[i : i + 128]), list(topos[i : i + 128])
                batch = tasu.batch_atom_bond(chunk_objs)
                batch.update(tasu.batch_structure_topology(chunk_objs, chunk_topos))
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                pred, feats = model(batch, return_features=True)
                alpha = feats["alpha"].view(-1).cpu().numpy()
                gamma = feats["gamma"].view(-1).cpu().numpy()
                bias = float(model.bias.detach().cpu().item())
                recon = bias + _scatter_sum(alpha, batch["blk_batch"].cpu().numpy(), int(batch["n_graphs"])) + _scatter_sum(
                    gamma, batch["bridge_graph"].cpu().numpy(), int(batch["n_graphs"])
                )
                max_err = max(max_err, float(np.max(np.abs(recon - pred.cpu().numpy()))))
                alpha_abs += float(np.abs(alpha).sum())
                gamma_abs += float(np.abs(gamma).sum())
        identity[arm] = {
            "decomposition_max_abs_err": max_err,
            "decomposition_ok": bool(max_err < 1e-4),
            "sum_abs_alpha": alpha_abs,
            "sum_abs_gamma": gamma_abs,
            "relation_absolute_share": float(gamma_abs / max(alpha_abs + gamma_abs, 1e-12)),
        }
    payload["decomposition_identity"] = identity
    # 2) exact-isomorphism consistency of unit representations / alpha
    iso = iso_consistency(unit_records_for_zone("monitor", "L-A", int(args.seed), device, limit=args.limit))
    payload["isomorphism_consistency"] = iso
    write_json(OUT / "contribution_audit_D.json", payload)
    log(f"[audit-contribution] iso_pairs={iso.get('n_exact_iso_pairs')} max_repr_diff={iso.get('max_repr_abs_diff')}")
    return 0


def iso_consistency(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import networkx as nx

    buckets: dict[str, list[tuple[int, int]]] = {}
    meta: list[tuple[int, int, Mapping[str, Any], Mapping[str, Any]]] = []
    for ri, rec in enumerate(records):
        for ui, unit in enumerate(rec["units"]):
            buckets.setdefault(unit["wl_hash"], []).append((ri, ui))
            meta.append((ri, ui, rec, unit))
    n_pairs = 0
    n_exact = 0
    repr_diffs: list[float] = []
    alpha_diffs: list[float] = []
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for a, b in zip(bucket, bucket[1:]):
            n_pairs += 1
            ra, rb = records[a[0]], records[b[0]]
            ga = _unit_induced_graph_from_rec(ra, a[1])
            gb = _unit_induced_graph_from_rec(rb, b[1])
            same = nx.is_isomorphic(
                ga, gb, node_match=lambda x, y: x["label"] == y["label"], edge_match=lambda x, y: x["type"] == y["type"]
            )
            if not same:
                continue
            n_exact += 1
            ua, ub = ra["units"][a[1]], rb["units"][b[1]]
            repr_diffs.append(float(np.max(np.abs(np.asarray(ua["repr"]) - np.asarray(ub["repr"])))))
            alpha_diffs.append(abs(float(ua["alpha"]) - float(ub["alpha"])))
    return {
        "n_units": int(sum(len(r["units"]) for r in records)),
        "n_wl_buckets": int(len(buckets)),
        "n_wl_candidate_pairs": int(n_pairs),
        "n_exact_iso_pairs": int(n_exact),
        "max_repr_abs_diff": float(max(repr_diffs)) if repr_diffs else 0.0,
        "mean_repr_abs_diff": float(np.mean(repr_diffs)) if repr_diffs else 0.0,
        "max_alpha_abs_diff": float(max(alpha_diffs)) if alpha_diffs else 0.0,
        "within_float_tolerance": bool((max(repr_diffs) if repr_diffs else 0.0) < 1e-5),
    }


def _unit_induced_graph_from_rec(rec: Mapping[str, Any], unit_index: int):
    obj = _object_lookup(rec["mol_index"])
    atoms = rec["units"][unit_index]["atom_ids"]
    return _unit_induced_graph(obj, atoms)


_OBJECT_CACHE: dict[int, sro.MolObjects] = {}


def _object_lookup(mol_index: int) -> sro.MolObjects:
    if not _OBJECT_CACHE:
        objs, _ = load_zone("monitor")
        for obj in objs:
            _OBJECT_CACHE[int(obj.mol_index)] = obj
    return _OBJECT_CACHE[int(mol_index)]


def cmd_audit_sharing(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    device = torch.device(args.device)
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "git_commit": git_commit()}
    for zone in ("monitor",):
        records = unit_records_for_zone(zone, "L-A", int(args.seed), device, limit=args.limit)
        type_map: dict[str, list[dict[str, Any]]] = {}
        for rec in records:
            for unit in rec["units"]:
                type_map.setdefault(unit["wl_hash"], []).append({**unit, "mol_index": rec["mol_index"], "smiles": rec["smiles"]})
        counts = np.asarray([len(v) for v in type_map.values()], dtype=np.int64)
        cross = [v for v in type_map.values() if len({u["mol_index"] for u in v}) > 1]
        alphas_per_type = [
            float(np.std([u["alpha"] for u in v])) for v in type_map.values() if len(v) >= 5
        ]
        payload[zone] = {
            "n_units": int(sum(len(r["units"]) for r in records)),
            "n_types": int(len(type_map)),
            "reuse_rate": float(1.0 - len(type_map) / max(sum(len(r["units"]) for r in records), 1)),
            "types_reused_across_molecules": int(len(cross)),
            "max_type_count": int(counts.max()) if counts.size else 0,
            "mean_type_count": float(counts.mean()) if counts.size else 0.0,
            "top_types": [
                {
                    "wl_hash": h,
                    "count": len(v),
                    "molecules": len({u["mol_index"] for u in v}),
                    "examples": [u["smiles"] for u in v[:3]],
                }
                for h, v in sorted(type_map.items(), key=lambda kv: -len(kv[1]))[:10]
            ],
            "alpha_within_type_mean_std": float(np.mean(alphas_per_type)) if alphas_per_type else 0.0,
        }
    write_json(OUT / "sharing_audit_D.json", payload)
    log(f"[audit-sharing] types={payload['monitor']['n_types']} reuse={payload['monitor']['reuse_rate']:.3f}")
    return 0


# ---------------------------------------------------------------------------
# finalize / render
# ---------------------------------------------------------------------------


def cmd_finalize(args: argparse.Namespace) -> int:
    verify_freeze(require_exact_code=False)
    decision = read_json(OUT / "decision.json")
    monitor = read_json(OUT / "monitor_score_D.json")
    partition = read_json(OUT / "partition_audit_D.json")
    contribution = read_json(OUT / "contribution_audit_D.json")
    sharing = read_json(OUT / "sharing_audit_D.json")
    gates = read_json(OUT / "cpu_gates.json")
    blocks = read_json(OUT / "block_audit_D.json")
    if decision["decision"] == "STRUCTURE_UNITS_LEARNING_PROMISING" and partition.get("learned_partition_degenerate"):
        decision["decision"] = "STRUCTURE_UNITS_NONIDENTIFIABLE"
        decision["downgrade_reason"] = "monitor partition degenerates to a trivial all-merge/all-cut solution"
        write_json(OUT / "decision.json", decision)
    report = _render_report(decision, monitor, partition, contribution, sharing, gates, blocks)
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    manifest = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "finalised_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision["decision"],
        "files": {
            p.name: file_sha256(p)
            for p in sorted(OUT.glob("*"))
            if p.is_file() and p.name not in {"REPORT.md"}
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    write_json(OUT / "evidence_manifest.json", manifest)
    log(f"[finalize] decision={decision['decision']}")
    return 0


def _render_report(decision, monitor, partition, contribution, sharing, gates, blocks) -> str:
    lines = [f"# {PROTOCOL_ID} — task-adaptive molecular structure units", ""]
    lines.append(f"- git commit: `{monitor['git_commit']}`")
    lines.append(f"- decision: **{decision['decision']}**")
    lines.append("- Monitor labels read exactly once, after prediction freeze; locked side untouched.")
    lines.append("")
    lines.append("## Answers to the five protocol questions")
    lines.append("")
    la = monitor["arm_mae"]["L-A"]["seed_mean"]
    fa = monitor["arm_mae"]["F-A"]["seed_mean"]
    lo = monitor["arm_mae"]["L-O"]["seed_mean"]
    fo = monitor["arm_mae"]["F-O"]["seed_mean"]
    lines.append(
        f"1. **Can boundaries be learned from task loss?** CPU S1 direct supervision reaches "
        f"{gates['synthetic_s1']['test_stats']['bridge_accuracy']:.3f} held-out bridge accuracy; "
        f"S2 y-only: L-A {gates['synthetic_s2']['L-A']['test_mae_seed_mean']:.3f} vs F-A "
        f"{gates['synthetic_s2']['F-A']['test_mae_seed_mean']:.3f} (oracle {gates['synthetic_s2']['O-A']['test_mae_seed_mean']:.3f}, "
        f"all-merge {gates['synthetic_s2']['M-A']['test_mae_seed_mean']:.3f}). "
        f"Monitor: L-A {la:.6f} vs F-A {fa:.6f}."
    )
    lines.append(
        f"2. **Connected / non-duplicated / reusable?** Units are merged base blocks of the block tree, so "
        f"connectedness is structural; partition degeneracy on monitor: "
        f"all-merge={partition['zones']['monitor']['L-A']['all_merge_fraction']:.3f}, "
        f"all-cut={partition['zones']['monitor']['L-A']['all_cut_fraction']:.3f}; "
        f"type reuse={sharing['monitor']['reuse_rate']:.3f} over {sharing['monitor']['n_types']} exact-WL types."
    )
    lines.append(
        f"3. **Does learned partition beat fixed?** H1 gate {'PASS' if monitor['h1_gate']['passed'] else 'FAIL'}; "
        f"L-A-F-A delta {monitor['contrasts']['L-A_vs_F-A']['delta_seed_mean']:+.6f} "
        f"(threshold {monitor['h1_gate']['delta_threshold']})."
    )
    lines.append(
        f"4. **Cost of explicit additive decomposition vs opaque readout?** "
        f"L-O {lo:.6f} vs L-A {la:.6f}; F-O {fo:.6f} vs F-A {fa:.6f}; "
        f"capacity matched within {100*abs(gates['opaque_capacity_ratio_L']-1):.2f}%."
    )
    lines.append(
        f"5. **Worth pursuing vs BJOC-Null?** BJOC-Null reference monitor MAE "
        f"{monitor['bjoc_null_reference']['monitor_mae_seed_mean']:.6f}; best arm here "
        f"{min(fa, la, fo, lo):.6f}."
    )
    lines.append("")
    lines.append("## Monitor arms (raw-y MAE, seeds " + ", ".join(map(str, monitor["seeds"])) + ")")
    lines.append("")
    lines.append("| arm | seed 0 | seed 1 | seed mean |")
    lines.append("| --- | --- | --- | --- |")
    for arm in ARMS:
        row = monitor["arm_mae"][arm]
        lines.append(f"| {arm} | {row['0']:.6f} | {row['1']:.6f} | {row['seed_mean']:.6f} |")
    lines.append("")
    lines.append("## Label-blind audits before monitor read")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(
        {
            "partition_learned_degenerate": partition.get("learned_partition_degenerate"),
            "partition_monitor": partition["zones"]["monitor"],
            "isomorphism_consistency": contribution["isomorphism_consistency"],
            "sharing_monitor": sharing["monitor"],
            "decomposition_identity": contribution["decomposition_identity"],
        },
        indent=1,
    ))
    lines.append("```")
    lines.append("")
    lines.append("## Synthetic known-truth evidence")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps({"s1": gates["synthetic_s1"], "s2": gates["synthetic_s2"]}, indent=1))
    lines.append("```")
    lines.append("")
    lines.append("## ZINC block applicability")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(blocks, indent=1))
    lines.append("```")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("verify-inputs")
    s.set_defaults(func=cmd_verify_inputs)

    s = sub.add_parser("audit-blocks")
    s.add_argument("--allow-fail", action="store_true")
    s.set_defaults(func=cmd_audit_blocks)

    s = sub.add_parser("cpu-gates")
    s.add_argument("--allow-fail", action="store_true")
    s.add_argument("--struct-molecules", type=int, default=200)
    s.add_argument("--forward-molecules", type=int, default=256)
    s.add_argument("--invariance-molecules", type=int, default=32)
    s.add_argument("--grad-molecules", type=int, default=256)
    s.add_argument("--synth-train", type=int, default=1500)
    s.add_argument("--synth-test", type=int, default=500)
    s.add_argument("--s1-epochs", type=int, default=120)
    s.add_argument("--s2-epochs", type=int, default=60)
    s.set_defaults(func=cmd_cpu_gates)

    s = sub.add_parser("freeze")
    s.set_defaults(func=cmd_freeze)

    s = sub.add_parser("train")
    s.add_argument("--arm", required=True, choices=list(ARMS))
    s.add_argument("--seed", required=True, type=int)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.add_argument("--probe-every", type=int, default=20)
    s.add_argument("--epochs", type=int, default=None)
    s.add_argument("--smoke", action="store_true")
    s.add_argument("--no-ssl", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("predict")
    s.add_argument("--arm", required=True, choices=list(ARMS))
    s.add_argument("--seed", required=True, type=int)
    s.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    s.set_defaults(func=cmd_predict)

    s = sub.add_parser("freeze-predictions")
    s.set_defaults(func=cmd_freeze_predictions)

    s = sub.add_parser("score-monitor")
    s.set_defaults(func=cmd_score_monitor)

    s = sub.add_parser("audit-partition")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_audit_partition)

    s = sub.add_parser("audit-contribution")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, default=1000)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_audit_contribution)

    s = sub.add_parser("audit-sharing")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default="cpu")
    s.set_defaults(func=cmd_audit_sharing)

    s = sub.add_parser("finalize")
    s.set_defaults(func=cmd_finalize)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    torch.set_num_threads(4)
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
