"""E2E-DictEnv-Typed-Cycle-v1 driver + control-plane stages (ZINC, CPU).

Round ``zinc_e2e_dictenv_typed_cycle_v1`` (Workstream Z, ZINC).
Pre-registration:
``tracks/ksvd/notes/zinc_e2e_dictenv_typed_cycle_v1_preregistration.md``.
Core: ``e2e_dictenv_typed_cycle_v1.py``; ring cache: ``typed_cycle_data_v1.py``.

Stage order: ``data correctness timing smoke train interventions analysis``.
Phase A (``typed_cycle_probe_v1``) must already have returned
``BUY_ONE_TYPED_CYCLE_SCREEN`` before the formal run.  Exactly one seed-0
320-epoch CPU trajectory.  The official ZINC **test** split is never
instantiated.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_clean_mechanism_v1 as cm
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_common_subspace_dictionary_v1 as cssd
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_typed_cycle_v1 as tc
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0
from tracks.ksvd.experiments.luyin16 import typed_cycle_data_v1 as tcd
from tracks.ksvd.experiments.luyin16 import typed_cycle_probe_v1 as probe
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p2_abs as p2run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_sem108_v1 as base
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = tc.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/zinc_e2e_dictenv_typed_cycle_v1"
AUDIT_DIR = RESULTS_DIR / "audit"
MECHANISM_DIR = RESULTS_DIR / "mechanism"
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CACHE_DIR = RESULTS_DIR / "cache"
NOTES_DIR = TRACK_ROOT / "notes"
PREREG_PATH = NOTES_DIR / "zinc_e2e_dictenv_typed_cycle_v1_preregistration.md"
PROBE_STOP_PATH = TRACK_ROOT / "results/zinc_e2e_dictenv_typed_cycle_v1/screen.json"

TAG = "TYPED-CYCLE"
THREADS = 8
SEED = 0
RING_SEED = 1
TRAIN_EPOCHS = 320
SMOKE_EPOCHS = 3
SMOKE_SUBSET = 1024
BATCH_SIZE = int(p2run.BATCH_SIZE)
SHUFFLE_SEEDS = (11, 22)
FORMAL_BUDGET_S = 4.0 * 3600.0
TIMING_SAFETY = 1.15
TIMING_STEPS = 20
ANALYSIS_OVERHEAD_S = 15.0 * 60.0

#: acceptance tolerances (pre-registration section 2.1).
GATE_CONTAINMENT_MAX = 1.0e-5
GATE_RELABEL_TOL = 1.0e-4
GATE_BATCH_TOL = 1.0e-5

BANDS: tuple[tuple[float, str], ...] = (
    (0.115, "TYPED_CYCLE_PROMISING_ABSOLUTE_SIGNAL"),
    (0.120, "TYPED_CYCLE_LIMITED_SIGNAL"),
    (float("inf"), "TYPED_CYCLE_STOP"),
)

_write_json = base._write_json
_read_json = base._read_json
_read_csv = getattr(base, "_read_csv", None)
_git_commit = base._git_commit
_sha256_file = base._sha256_file


def _read_json_or(path: Path) -> dict[str, Any]:
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def _band(mae: float) -> str:
    for threshold, label in BANDS:
        if float(mae) <= threshold:
            return label
    return BANDS[-1][1]


def _ensure_dirs() -> None:
    for path in (RESULTS_DIR, AUDIT_DIR, MECHANISM_DIR, CHECKPOINT_DIR, CACHE_DIR):
        path.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_typed_split(split: str, subset: int | None = None) -> list[Any]:
    data_list = p1run.load_split(split, subset=subset)
    meta = _read_json(CACHE_DIR / "ring_cache.json") if (CACHE_DIR / "ring_cache.json").exists() else {}
    key = meta.get(split, {}).get("key")
    if key is None:
        raise RuntimeError("ring cache missing; run stage_data first")
    path = CACHE_DIR / f"ring_{split}_{key}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    tcd.attach_rings(data_list, payload, subset=subset)
    return data_list


def stage_data(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    root = REPO_ROOT / "data/ZINC"
    meta_path = CACHE_DIR / "ring_cache.json"
    meta = _read_json_or(meta_path)
    for split in ("train", "valid"):
        if split in meta and not force:
            continue
        info = tcd.build_cache(root, split, CACHE_DIR, force=force)
        payload = torch.load(Path(info["path"]), map_location="cpu", weights_only=False)
        info["total_cycles"] = int(sum(row["ring_cycle_anchor"].shape[0] for row in payload))
        info["total_views"] = int(sum(row["ring_view_atom"].shape[0] for row in payload))
        info["n_graphs"] = int(len(payload))
        info["acyclic_graphs"] = int(sum(1 for row in payload if row["ring_cycle_anchor"].shape[0] == 0))
        meta[split] = info
        print(f"[typed-cycle] ring cache {split}: {info}", flush=True)
    _write_json(meta_path, meta)
    return meta


# ---------------------------------------------------------------------------
# model / batch helpers
# ---------------------------------------------------------------------------
def _dictionary() -> np.ndarray:
    return base._dictionary_tensor()


def _subspace() -> Any:
    return base._load_parent_subspace()


def model_factory(dictionary: np.ndarray, seed: int, subspace: Any) -> tc.TypedCycleSEM108:
    return tc.build_typed_cycle_model(dictionary, int(seed), subspace, ring_seed=RING_SEED)


def _small_model(seed: int = SEED) -> lb.LatentBridgeSEM108:
    return lb.build_latent_bridge_model(_dictionary(), int(seed), _subspace())


def _typed_batch(data_list: Sequence[Any]) -> Any:
    return tc.typed_cycle_collate(data_list)


def _view_rows(payload_row: Mapping[str, np.ndarray]) -> list[tuple]:
    atom = payload_row["ring_view_atom"]
    bond = payload_row["ring_view_bond"]
    mask = payload_row["ring_view_mask"]
    length = payload_row["ring_view_length"]
    rows = []
    for index in range(atom.shape[0]):
        rows.append(
            (
                tuple(int(v) for v in atom[index]),
                tuple(int(v) for v in bond[index]),
                tuple(float(v) for v in mask[index]),
                int(length[index]),
            )
        )
    return rows


def _view_signature(payload_row: Mapping[str, np.ndarray]) -> tuple:
    return tuple(sorted(_view_rows(payload_row)))


def _capture_bridge_calls(model: tc.TypedCycleSEM108, batch: Any, mask: Any) -> dict[str, Any]:
    calls: list[tuple[int, int]] = []
    handle = model.local_dictionary_bridge.register_forward_hook(
        lambda _m, inputs, _o: calls.append((int(inputs[0].shape[0]), int(inputs[0].shape[1])))
    )
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(batch, mask=mask)
    finally:
        handle.remove()
        model.train(was)
    return {"n_calls": len(calls), "calls": calls}


# ---------------------------------------------------------------------------
# stage: preflight
# ---------------------------------------------------------------------------
def stage_preflight() -> dict[str, Any]:
    _ensure_dirs()
    screen = _read_json_or(PROBE_STOP_PATH)
    if screen.get("verdict") != "BUY_ONE_TYPED_CYCLE_SCREEN":
        raise RuntimeError("Phase A did not buy this candidate; refusing preflight")
    audit = tc.typed_cycle_parameter_audit(model_factory(_dictionary(), SEED, _subspace()))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "stage": "screen",
        "seed": SEED,
        "ring_seed": RING_SEED,
        "epochs": TRAIN_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": float(p2run.LEARNING_RATE),
        "weight_decay": float(p2run.WEIGHT_DECAY),
        "grad_clip": float(p2run.GRAD_CLIP),
        "soup_k": int(p2run.SOUP_K),
        "absolute_bands": [
            {"max": 0.115, "label": "TYPED_CYCLE_PROMISING_ABSOLUTE_SIGNAL"},
            {"max": 0.120, "label": "TYPED_CYCLE_LIMITED_SIGNAL"},
            {"max": None, "label": "TYPED_CYCLE_STOP"},
        ],
        "preregistration_path": str(PREREG_PATH.relative_to(REPO_ROOT)),
        "preregistration_sha256": _sha256_file(PREREG_PATH),
        "parameter_audit": audit,
        "screen": screen,
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "preflight.json", payload)
    _write_json(
        RESULTS_DIR / "preregistration_snapshot.json",
        {
            "path": str(PREREG_PATH.relative_to(REPO_ROOT)),
            "sha256": _sha256_file(PREREG_PATH),
            "git_commit": _git_commit(),
        },
    )
    print(f"[preflight] prereg sha={payload['preregistration_sha256'][:12]} params={audit['total_parameters']}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: correctness (8 acceptance gates)
# ---------------------------------------------------------------------------
def _gate_containment(audit: Mapping[str, Any]) -> dict[str, Any]:
    root = REPO_ROOT / "data/ZINC"
    data = p1run.load_split("train", subset=32)
    payload = tcd.build_ring_payload(root, "train", subset=32)
    tcd.attach_rings(data, payload, subset=32)
    batch = _typed_batch(data)
    mask = cm.C6_MASK
    small = _small_model().eval()
    typed = model_factory(_dictionary(), SEED, _subspace()).eval()
    sstate, tstate = small.state_dict(), typed.state_dict()
    shared_mismatch = [
        key for key, value in sstate.items() if key in tstate and tstate[key].shape == value.shape and not torch.equal(tstate[key], value)
    ]
    with torch.no_grad():
        plain = float((small(batch).view(-1).double() - typed(batch).view(-1).double()).abs().max())
        masked = float(
            (small(batch, mask=mask).view(-1).double() - typed(batch, mask=mask).view(-1).double()).abs().max()
        )
    passed = bool(
        audit["parameter_exact"]
        and not shared_mismatch
        and plain <= GATE_CONTAINMENT_MAX
        and masked <= GATE_CONTAINMENT_MAX
    )
    return {
        "parameter_exact": bool(audit["parameter_exact"]),
        "total_parameters": int(audit["total_parameters"]),
        "shared_parameter_mismatch": shared_mismatch,
        "plain_max_abs_diff": plain,
        "masked_max_abs_diff": masked,
        "tolerance": GATE_CONTAINMENT_MAX,
        "passed": passed,
    }


def _gate_invariance() -> dict[str, Any]:
    root = REPO_ROOT / "data/ZINC"
    from tracks.ksvd.experiments.luyin16 import zinc_long_range_proxy as zlr

    # 1) integer view set == reference dihedral views.
    raw = list(zlr._load_zinc(root, "train"))[:4]
    import typed_cycle_reference as tcr

    view_set_equal = True
    for data in raw:
        graph, node_types, edge_types = zlr._data_to_graph(data)
        ref = probe.to_reference_graph(graph, node_types, edge_types)
        row = tcd.ring_fields(ref)
        for cycle in tcr.chordless_cycles(ref, max_length=tc.MAX_RING_LENGTH):
            ref_views = tcr.dihedral_views(ref, cycle)
            ell = len(cycle)
            found = []
            for index in range(row["ring_view_atom"].shape[0]):
                if int(row["ring_view_length"][index]) != ell:
                    continue
                found.append(
                    (
                        tuple(int(v) for v in row["ring_view_atom"][index]),
                        tuple(int(v) for v in row["ring_view_bond"][index]),
                    )
                )
            ref_set = {
                (
                    tuple(int(np.argmax(view[t * 28 : t * 28 + 28])) for t in range(ell)),
                    tuple(int(np.argmax(view[280 + t * 4 : 280 + t * 4 + 4])) for t in range(ell)),
                )
                for view in ref_views
            }
            # filter the row views to this cycle by membership in the reference set
            if not ref_set.issubset({(a[:ell], b[:ell]) for a, b in found}):
                view_set_equal = False
    # 2) node relabel -> identical integer view set
    relabel_equal = True
    rng = np.random.default_rng(20261002)
    for data in raw[:2]:
        graph, node_types, edge_types = zlr._data_to_graph(data)
        ref = probe.to_reference_graph(graph, node_types, edge_types)
        perm = rng.permutation(len(ref.atoms)).tolist()
        relabelled = tcr.relabel(ref, perm)
        if _view_signature(tcd.ring_fields(ref)) != _view_signature(tcd.ring_fields(relabelled)):
            relabel_equal = False
    # 3) graph order / batch invariance
    data = load_typed_split("train", subset=8)
    typed = model_factory(_dictionary(), SEED, _subspace()).eval()
    mask = cm.C6_MASK
    with torch.no_grad():
        forward = typed(_typed_batch(data), mask=mask).view(-1)
        reverse = typed(_typed_batch(list(reversed(data))), mask=mask).view(-1).flip(0)
        per_mol = torch.cat([typed(_typed_batch([item]), mask=mask).view(-1) for item in data])
    batch_delta = float((forward - reverse).abs().max())
    per_mol_delta = float((forward - per_mol).abs().max())
    # 4) view-order permutation within each batch preserves the ring pool.
    batch = _typed_batch(data)
    with torch.no_grad():
        base_pool = typed.ring_pool(batch)
    perm = torch.randperm(int(batch.ring_view_atom.shape[0]), generator=torch.Generator().manual_seed(7))
    permuted = batch.clone()
    for name in tc.RING_VIEW_FIELDS:
        permuted[name] = batch[name][perm]
    with torch.no_grad():
        perm_pool = typed.ring_pool(permuted)
    view_perm_delta = float((base_pool - perm_pool).abs().max())
    passed = bool(
        view_set_equal
        and relabel_equal
        and batch_delta <= GATE_BATCH_TOL
        and per_mol_delta <= GATE_RELABEL_TOL
        and view_perm_delta <= GATE_RELABEL_TOL
    )
    return {
        "reference_view_set_equal": view_set_equal,
        "relabel_integer_view_set_equal": relabel_equal,
        "graph_order_max_abs_diff": batch_delta,
        "per_molecule_batch_max_abs_diff": per_mol_delta,
        "view_permutation_pool_max_abs_diff": view_perm_delta,
        "passed": passed,
    }


def _gate_counterexample() -> dict[str, Any]:
    import cheap_probe_features as cpf
    import typed_cycle_reference as tcr

    from tracks.ksvd.code.graph import from_edges

    parallel, mixed = tcr.witness_pair()

    def _repo_graph(ref):
        edges = [(u, v) for u, v, _ in ref.edges]
        return from_edges(len(ref.atoms), edges)

    def _old_blocks(ref):
        sem = tcr.semantic110(ref)
        untyped, _typed = cpf.ring_features(ref)
        edge_counts = cpf.edge_type_counts(ref)
        phi = fsar_r2_ar0.build_phi(_repo_graph(ref))
        return sem, untyped, edge_counts, phi

    old_equal = {
        "sem110_equal": bool(np.array_equal(_old_blocks(parallel)[0], _old_blocks(mixed)[0])),
        "untyped_cycles_equal": bool(np.array_equal(_old_blocks(parallel)[1], _old_blocks(mixed)[1])),
        "edge_type_counts_equal": bool(np.array_equal(_old_blocks(parallel)[2], _old_blocks(mixed)[2])),
        "phi65_equal": bool(np.array_equal(_old_blocks(parallel)[3], _old_blocks(mixed)[3])),
    }

    typed = model_factory(_dictionary(), SEED, _subspace()).eval()

    def _ring_pool(row):
        from types import SimpleNamespace

        obj = SimpleNamespace()
        obj.ring_view_atom = torch.as_tensor(row["ring_view_atom"], dtype=torch.long)
        obj.ring_view_bond = torch.as_tensor(row["ring_view_bond"], dtype=torch.long)
        obj.ring_view_mask = torch.as_tensor(row["ring_view_mask"], dtype=torch.float32)
        obj.ring_view_length = torch.as_tensor(row["ring_view_length"], dtype=torch.long)
        obj.ring_view_cycle = torch.as_tensor(row["ring_view_cycle"], dtype=torch.long)
        obj.ring_cycle_anchor = torch.zeros(int(row["ring_cycle_anchor"].shape[0]), dtype=torch.long)
        obj.ring_cycle_length = torch.as_tensor(row["ring_cycle_length"], dtype=torch.long)
        obj.global_context = torch.zeros(1, p2.GLOBAL_WIDTH)
        obj.batch = torch.zeros(1, dtype=torch.long)
        with torch.no_grad():
            return typed.ring_pool(obj)

    pool_parallel = _ring_pool(tcd.ring_fields(parallel))
    pool_mixed = _ring_pool(tcd.ring_fields(mixed))
    ring_delta = float((pool_parallel - pool_mixed).abs().max())

    # six-ring composition-equal / cyclic-order-different example.
    def _hexacycle(sequence):
        n = len(sequence)
        edges = [(min(i, (i + 1) % n), max(i, (i + 1) % n), 0) for i in range(n)]
        ref = tcr.Graph(tuple(int(a) for a in sequence), tuple(sorted(edges)))
        return ref

    ring_a = _hexacycle([0, 0, 1, 1, 2, 2])
    ring_b = _hexacycle([0, 1, 0, 1, 2, 2])
    order_delta = float((_ring_pool(tcd.ring_fields(ring_a)) - _ring_pool(tcd.ring_fields(ring_b))).abs().max())

    old_collides = all(old_equal.values())
    passed = bool(ring_delta > 1e-6 and order_delta > 1e-6)
    return {
        "old_feature_blocks": old_equal,
        "old_feature_blocks_collide": bool(old_collides),
        "new_ring_pool_max_abs_diff": ring_delta,
        "six_ring_order_max_abs_diff": order_delta,
        "passed": passed,
        "note": (
            "Old feature blocks are built by the repository Sem110 / phi65 / untyped-ring / "
            "atom-pair x bond-type builders; the new ring summary uses the real PyTorch "
            "ring encoder + shared bridge.  A feature-block collision is necessary but not a "
            "proof of full-model prediction collision."
        ),
    }


def _gate_wiring() -> dict[str, Any]:
    data = load_typed_split("train", subset=8)
    batch = _typed_batch(data)
    typed = model_factory(_dictionary(), SEED, _subspace()).eval()
    mask = cm.C6_MASK
    calls = _capture_bridge_calls(typed, batch, mask)
    # ring-object change must not change node E at a fixed state.
    with torch.no_grad():
        _, aux0 = typed(batch, mask=mask, return_aux=True)
        changed = batch.clone()
        for name in ("ring_view_atom", "ring_view_bond", "ring_view_mask", "ring_view_length"):
            if name == "ring_view_length":
                changed[name] = torch.clamp(batch[name] + 1, max=tc.MAX_RING_LENGTH)
            elif name == "ring_view_mask":
                changed[name] = torch.zeros_like(batch[name])
            elif name == "ring_view_atom":
                changed[name] = (batch[name] + 1) % 28
            else:
                changed[name] = (batch[name] + 1) % 4
        _, aux1 = typed(changed, mask=mask, return_aux=True)
    e_delta = float((aux0["E"] - aux1["E"]).abs().max())
    # node path uses exactly one bridge call; ring path exactly one -> two total.
    passed = bool(calls["n_calls"] == 2 and e_delta <= 1e-6)
    return {
        "bridge_calls_per_forward": calls["n_calls"],
        "bridge_call_shapes": calls["calls"],
        "expected_calls": 2,
        "node_E_under_changed_rings_max_abs_diff": e_delta,
        "passed": passed,
        "note": "one frozen node environment bridge call + one independent ring bridge call; no ring->node write-back",
    }


def _gate_cold_start_gradient() -> dict[str, Any]:
    data = load_typed_split("train", subset=64)
    batches = list(tc.typed_cycle_loader(data, 32, False, 0))
    assert len(batches) >= 2
    mask = cm.C6_MASK
    model = model_factory(_dictionary(), SEED, _subspace())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY))

    def _grads(batch):
        model.eval()
        model.zero_grad(set_to_none=True)
        prediction = model(batch, mask=mask)
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        loss.backward()
        out = {
            "ring_encoder_grad": float(
                sum(float(p.grad.norm()) for p in model.ring_encoder.parameters() if p.grad is not None)
            ),
            "new_reader_columns_grad": float(
                model.reader.net[0].weight.grad[:, tc.SMALL_READER_IN :].norm()
                if model.reader.net[0].weight.grad is not None
                else 0.0
            ),
            "bridge_D_L_grad": float(
                model.local_dictionary_bridge.D_L.grad.norm()
                if model.local_dictionary_bridge.D_L.grad is not None
                else 0.0
            ),
            "bridge_V_L_grad": float(
                model.local_dictionary_bridge.V_L.grad.norm()
                if model.local_dictionary_bridge.V_L.grad is not None
                else 0.0
            ),
            "loss": float(loss.detach()),
        }
        return out

    step0 = _grads(batches[0])
    model.train()
    model.zero_grad(set_to_none=True)
    prediction = model(batches[0], mask=mask)
    loss = F.l1_loss(prediction.view(-1), batches[0].y.view(-1))
    loss.backward()
    optimizer.step()
    step1 = _grads(batches[1])
    model.zero_grad(set_to_none=True)
    passed = bool(
        step0["ring_encoder_grad"] == 0.0
        and step0["new_reader_columns_grad"] > 0.0
        and step1["ring_encoder_grad"] > 0.0
        and (step1["bridge_D_L_grad"] > 0.0 or step1["bridge_V_L_grad"] > 0.0)
    )
    return {"step0": step0, "step1_after_one_adam_step": step1, "passed": passed}


def _gate_finite() -> dict[str, Any]:
    data = load_typed_split("train", subset=64)
    batch = _typed_batch(data)
    mask = cm.C6_MASK
    model = model_factory(_dictionary(), SEED, _subspace())
    prediction = model(batch, mask=mask)
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    model.zero_grad(set_to_none=True)
    loss.backward()
    grads_finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
    )
    with torch.no_grad():
        pool = model.ring_pool(batch)
    passed = bool(torch.isfinite(prediction).all() and torch.isfinite(pool).all() and grads_finite)
    return {
        "prediction_finite": bool(torch.isfinite(prediction).all()),
        "ring_pool_finite": bool(torch.isfinite(pool).all()),
        "gradients_finite": bool(grads_finite),
        "passed": passed,
    }


def _gate_zero_ring() -> dict[str, Any]:
    data = load_typed_split("train", subset=8)
    batch = _typed_batch(data)
    mask = cm.C6_MASK
    model = model_factory(_dictionary(), SEED, _subspace()).eval()
    with torch.no_grad():
        _, aux0 = model(batch, mask=mask, return_aux=True)
        pool0 = model.ring_pool(batch)
        model.set_zero_ring(True)
        _, aux1 = model(batch, mask=mask, return_aux=True)
        pool1 = model.ring_pool(batch)
        model.set_zero_ring(False)
    e_same = bool(torch.equal(aux0["E"], aux1["E"]))
    pool_zero = bool(float(pool1.abs().max()) == 0.0)
    return {
        "node_E_identical": e_same,
        "ring_pool_exact_zero": pool_zero,
        "ring_pool_nonzero_before": float(pool0.abs().max()),
        "passed": bool(e_same and pool_zero),
    }


def _gate_test_blocker() -> dict[str, Any]:
    raised = False
    try:
        tc.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        raised = True
    try:
        probe.official_test_blocker({"official_test_loaded": True})
    except RuntimeError:
        raised = True
    # stage_data / loaders reference only train/valid caches.
    caches = list(CACHE_DIR.glob("ring_test_*.pt"))
    return {
        "payload_guard_raises": bool(raised),
        "test_cache_files_present": len(caches),
        "passed": bool(raised and not caches),
    }


def stage_correctness() -> dict[str, Any]:
    _ensure_dirs()
    base.sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    screen = _read_json_or(PROBE_STOP_PATH)
    if screen.get("verdict") != "BUY_ONE_TYPED_CYCLE_SCREEN":
        raise RuntimeError("Phase A did not buy this candidate; refusing to run Phase B")
    typed = model_factory(_dictionary(), SEED, _subspace())
    audit = tc.typed_cycle_parameter_audit(typed)
    gates = {
        "G1_containment": _gate_containment(audit),
        "G2_invariance": _gate_invariance(),
        "G3_counterexample": _gate_counterexample(),
        "G4_wiring": _gate_wiring(),
        "G5_cold_start_gradient": _gate_cold_start_gradient(),
        "G6_finite": _gate_finite(),
        "G7_zero_ring": _gate_zero_ring(),
        "G8_test_blocker": _gate_test_blocker(),
    }
    all_passed = all(bool(gate["passed"]) for gate in gates.values())
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "parameter_audit": audit,
        "gates": gates,
        "all_passed": all_passed,
        "git_commit": _git_commit(),
    }
    _write_json(AUDIT_DIR / "correctness.json", payload)
    _write_json(AUDIT_DIR / "parameter_audit.json", audit)
    print(f"[correctness] all_passed={all_passed}", flush=True)
    for name, gate in gates.items():
        print(f"  {name}: {gate['passed']}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: timing
# ---------------------------------------------------------------------------
def stage_timing(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = RESULTS_DIR / "timing.json"
    if out.exists() and not force:
        payload = _read_json(out)
        print("[timing] cache hit", flush=True)
        return payload
    correctness = _read_json_or(AUDIT_DIR / "correctness.json")
    if not correctness.get("all_passed"):
        raise RuntimeError("acceptance gates not passed; refusing to time")
    base.sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    train_data = load_typed_split("train")
    model = model_factory(_dictionary(), SEED, _subspace())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY))
    mask = cm.C6_MASK
    loader = tc.typed_cycle_loader(train_data, BATCH_SIZE, True, SEED + int(p2run.TRAIN_SHUFFLE_OFFSET))
    iterator = iter(loader)
    model.train()
    for _ in range(3):  # warmup
        batch = next(iterator).to("cpu")
        prediction = model(batch, mask=mask)
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    started = time.perf_counter()
    for _ in range(TIMING_STEPS):
        batch = next(iterator).to("cpu")
        prediction = model(batch, mask=mask)
        loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    step_seconds = (time.perf_counter() - started) / TIMING_STEPS
    steps_per_epoch = math.ceil(len(train_data) / BATCH_SIZE)
    epoch_seconds = step_seconds * steps_per_epoch
    predicted = TRAIN_EPOCHS * epoch_seconds * TIMING_SAFETY + ANALYSIS_OVERHEAD_S
    authorized = bool(predicted < FORMAL_BUDGET_S)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "steady_step_seconds": step_seconds,
        "steps_per_epoch": steps_per_epoch,
        "seconds_per_epoch": epoch_seconds,
        "predicted_total_seconds_with_margin": predicted,
        "budget_seconds": FORMAL_BUDGET_S,
        "formal_run_authorized": authorized,
        "git_commit": _git_commit(),
    }
    _write_json(out, payload)
    print(f"[timing] step={step_seconds:.4f}s epoch={epoch_seconds:.2f}s predicted={predicted:.0f}s authorized={authorized}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: smoke
# ---------------------------------------------------------------------------
def stage_smoke(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = RESULTS_DIR / "smoke.json"
    if out.exists() and not force:
        return _read_json(out)
    if not _read_json_or(RESULTS_DIR / "timing.json").get("formal_run_authorized"):
        payload = {"protocol_version": PROTOCOL_VERSION, "official_test_loaded": False, "status": "NOT_RUN_RESOURCE_BUDGET"}
        _write_json(out, payload)
        return payload
    base.sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    train_data = load_typed_split("train", subset=SMOKE_SUBSET)
    model = model_factory(_dictionary(), SEED, _subspace())
    optimizer = torch.optim.Adam(model.parameters(), lr=float(p2run.LEARNING_RATE), weight_decay=float(p2run.WEIGHT_DECAY))
    mask = cm.C6_MASK
    loader = tc.typed_cycle_loader(train_data, BATCH_SIZE, True, SEED + int(p2run.TRAIN_SHUFFLE_OFFSET))
    started = time.perf_counter()
    losses = []
    steps = 0
    model.train()
    for _epoch in range(SMOKE_EPOCHS):
        for batch in loader:
            prediction = model(batch, mask=mask)
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(p2run.GRAD_CLIP))
            optimizer.step()
            losses.append(float(loss.detach()))
            steps += 1
    elapsed = time.perf_counter() - started
    passed = bool(len(losses) == 24 and all(math.isfinite(value) for value in losses))
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "steps": steps,
        "epochs": SMOKE_EPOCHS,
        "subset": SMOKE_SUBSET,
        "final_train_mae": float(losses[-1]),
        "seconds": float(elapsed),
        "seconds_per_epoch": float(elapsed / SMOKE_EPOCHS),
        "passed": passed,
        "git_commit": _git_commit(),
    }
    _write_json(out, payload)
    print(f"[smoke] steps={steps} final={losses[-1]:.6f} passed={passed}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: train (frozen parent loop, independent collate)
# ---------------------------------------------------------------------------
def _install_typed_loader() -> Any:
    original = p1.make_env_loader
    p1.make_env_loader = tc.typed_cycle_loader
    return original


def stage_train(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    final_path = RESULTS_DIR / "run_seed0.json"
    if final_path.exists() and not force:
        print("[train] cache hit", flush=True)
        return _read_json(final_path)
    correctness = _read_json_or(AUDIT_DIR / "correctness.json")
    timing = _read_json_or(RESULTS_DIR / "timing.json")
    smoke = _read_json_or(RESULTS_DIR / "smoke.json")
    if not correctness.get("all_passed"):
        raise RuntimeError("acceptance gates not passed; refusing to train")
    if not timing.get("formal_run_authorized"):
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "status": "NOT_RUN_RESOURCE_BUDGET",
            "timing": timing,
        }
        _write_json(RESULTS_DIR / "train_skipped.json", payload)
        return payload
    if not smoke.get("passed"):
        raise RuntimeError("smoke not passed; refusing to train")
    if _read_json_or(RESULTS_DIR / "preflight.json").get("preregistration_sha256") not in (None, _sha256_file(PREREG_PATH)):
        raise RuntimeError("preregistration changed after preflight; refusing to train")
    base.sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    train_data = load_typed_split("train")
    valid_data = load_typed_split("valid")
    original = _install_typed_loader()
    started = time.perf_counter()
    try:
        result = cssd.train_cssd(
            tag=f"{TAG}-seed0",
            epochs=TRAIN_EPOCHS,
            threads=THREADS,
            out_dir=CHECKPOINT_DIR,
            subspace=_subspace(),
            train_data=train_data,
            valid_data=valid_data,
            seed=SEED,
            callback=None,
            model_factory=model_factory,
            save_states=True,
            log=True,
        )
    finally:
        p1.make_env_loader = original
    model = model_factory(_dictionary(), SEED, _subspace())
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / f"{TAG}-seed0_soup_state.pt", map_location="cpu", weights_only=False)
    )
    result.update(
        {
            "git_commit": _git_commit(),
            "device": "cpu",
            "threads": THREADS,
            "seed": SEED,
            "ring_seed": RING_SEED,
            "official_test_loaded": False,
            "round_seconds": float(time.perf_counter() - started),
            "parameter_audit": tc.typed_cycle_parameter_audit(model),
        }
    )
    _write_json(final_path, result)
    print(
        f"[train] epochs={result['epochs_run']} best={result['best_valid_mae']:.6f}@{result['best_epoch']} "
        f"soup={result['soup']['soup_valid_mae']:.6f} wall={result['wall_clock_s']:.1f}s",
        flush=True,
    )
    return result


def _soup_model() -> tc.TypedCycleSEM108:
    model = model_factory(_dictionary(), SEED, _subspace())
    state = torch.load(CHECKPOINT_DIR / f"{TAG}-seed0_soup_state.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model


def _evaluate(model: tc.TypedCycleSEM108, data_list: Sequence[Any], mask: Any) -> dict[str, Any]:
    loader = tc.typed_cycle_loader(data_list, BATCH_SIZE, False, SEED + int(p2run.EVAL_SHUFFLE_OFFSET))
    model.eval()
    predictions, targets = [], []
    with torch.no_grad():
        for batch in loader:
            predictions.append(model(batch, mask=mask).view(-1).cpu())
            targets.append(batch.y.view(-1).cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return {"mae": float((prediction - target).abs().mean()), "prediction": prediction}


# ---------------------------------------------------------------------------
# stage: interventions
# ---------------------------------------------------------------------------
def _cross_molecule_ring_permutation(model: tc.TypedCycleSEM108, data_list: Sequence[Any], mask: Any, seed: int) -> dict[str, Any]:
    """Swap same-length ring objects between molecules (content exchange).

    Per-graph cycle counts and the length distribution are preserved, and the
    global multiset of ring objects is unchanged (each exchange swaps both
    sides).  Graph-internal ring-row permutation is a sum-pooling invariance
    and is therefore not a liveness probe; this cross-molecule exchange is.
    """
    loader = tc.typed_cycle_loader(data_list, BATCH_SIZE, False, 0)
    model.eval()
    rng = torch.Generator().manual_seed(int(seed))
    base_pred, perm_pred, targets = [], [], []
    swapped = 0
    total_cycles = 0
    fields = ("ring_view_atom", "ring_view_bond", "ring_view_mask", "ring_view_length")
    with torch.no_grad():
        for batch in loader:
            base_pred.append(model(batch, mask=mask).view(-1).cpu())
            changed = batch.clone()
            n_graphs = int(batch.global_context.shape[0])
            cycle_graph = batch.batch[batch.ring_cycle_anchor]
            total_cycles += int(batch.ring_cycle_anchor.numel())
            lengths = batch.ring_cycle_length.unique().tolist()
            for graph in range(n_graphs):
                mine = (cycle_graph == graph).nonzero(as_tuple=False).view(-1)
                if mine.numel() == 0:
                    continue
                donor = int(torch.randint(0, n_graphs, (1,), generator=rng))
                if donor == graph:
                    continue
                donor_cycles = (cycle_graph == donor).nonzero(as_tuple=False).view(-1)
                if donor_cycles.numel() == 0:
                    continue
                for length in lengths:
                    mine_l = mine[batch.ring_cycle_length[mine] == int(length)]
                    donor_l = donor_cycles[batch.ring_cycle_length[donor_cycles] == int(length)]
                    k = min(int(mine_l.numel()), int(donor_l.numel()))
                    if k == 0:
                        continue
                    pick_m = mine_l[torch.randperm(int(mine_l.numel()), generator=rng)[:k]]
                    pick_d = donor_l[torch.randperm(int(donor_l.numel()), generator=rng)[:k]]
                    for target, source in zip(pick_m.tolist(), pick_d.tolist()):
                        rows_t = (batch.ring_view_cycle == target).nonzero(as_tuple=False).view(-1)
                        rows_s = (batch.ring_view_cycle == source).nonzero(as_tuple=False).view(-1)
                        if int(rows_t.numel()) != int(rows_s.numel()):
                            raise RuntimeError("same-length cycles must have equal view counts")
                        for name in fields:
                            changed[name][rows_t] = batch[name][rows_s]
                            changed[name][rows_s] = batch[name][rows_t]
                        swapped += 2
            perm_pred.append(model(changed, mask=mask).view(-1).cpu())
            targets.append(batch.y.view(-1).cpu())
    base_prediction = torch.cat(base_pred)
    perm_prediction = torch.cat(perm_pred)
    target = torch.cat(targets)
    base_mae = float((base_prediction - target).abs().mean())
    perm_mae = float((perm_prediction - target).abs().mean())
    return {
        "base_mae": base_mae,
        "permuted_mae": perm_mae,
        "delta_mae": float(perm_mae - base_mae),
        "prediction_rms": float(torch.sqrt(((perm_prediction - base_prediction) ** 2).mean())),
        "swapped_objects": int(swapped),
        "total_objects": int(total_cycles),
        "swapped_fraction": float(swapped / max(total_cycles, 1)),
        "seed": int(seed),
    }


def _code_stats(alpha: torch.Tensor) -> dict[str, Any]:
    alpha = alpha.detach().double()
    l0 = (alpha != 0).sum(dim=1).to(torch.float64)
    return {
        "n_objects": int(alpha.shape[0]),
        "mean_nonzero": float(l0.mean()) if alpha.numel() else 0.0,
        "fraction": float(l0.mean() / alpha.shape[1]) if alpha.numel() else 0.0,
        "p50": float(torch.quantile(l0, 0.5)) if alpha.numel() else 0.0,
        "p95": float(torch.quantile(l0, 0.95)) if alpha.numel() else 0.0,
    }


def stage_interventions(force: bool = False) -> dict[str, Any]:
    _ensure_dirs()
    out = MECHANISM_DIR / "typed_cycle_probes.json"
    if out.exists() and not force:
        return _read_json(out)
    run = _read_json_or(RESULTS_DIR / "run_seed0.json")
    if int(run.get("epochs_run", 0)) < TRAIN_EPOCHS:
        payload = {"protocol_version": PROTOCOL_VERSION, "official_test_loaded": False, "status": "NOT_RUN"}
        _write_json(out, payload)
        return payload
    base.sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    data = load_typed_split("valid")
    mask = cm.C6_MASK
    model = _soup_model().eval()

    baseline = _evaluate(model, data, mask)
    model.set_zero_ring(True)
    zero = _evaluate(model, data, mask)
    model.set_zero_ring(False)
    zero_row = {
        "mae": zero["mae"],
        "delta_mae": float(zero["mae"] - baseline["mae"]),
        "delta_pred_rms": float(torch.sqrt(((zero["prediction"] - baseline["prediction"]) ** 2).mean())),
    }

    # code density and MAE-only gradient on one real batch
    batch = next(iter(tc.typed_cycle_loader(data, BATCH_SIZE, False, 0)))
    aux_capture: dict[str, torch.Tensor] = {}
    handle = model.local_dictionary_bridge.register_forward_hook(
        lambda _m, _i, output: aux_capture.__setitem__("alpha", output[1]["alpha"].detach())
        if isinstance(output, tuple) and len(output) > 1 and isinstance(output[1], dict)
        else None
    )
    model.zero_grad(set_to_none=True)
    prediction = model(batch, mask=mask)
    loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    loss.backward()
    handle.remove()
    ring_alpha = aux_capture.get("alpha")
    ring_params = [p for p in model.ring_encoder.parameters()]
    ring_grad = float(sum(float(p.grad.norm()) for p in ring_params if p.grad is not None))
    bridge_grad = {
        "D_L": float(model.local_dictionary_bridge.D_L.grad.norm() if model.local_dictionary_bridge.D_L.grad is not None else 0.0),
        "V_L": float(model.local_dictionary_bridge.V_L.grad.norm() if model.local_dictionary_bridge.V_L.grad is not None else 0.0),
    }
    model.zero_grad(set_to_none=True)

    permutations = [_cross_molecule_ring_permutation(model, data, mask, seed) for seed in SHUFFLE_SEEDS]
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "baseline_valid_mae": float(baseline["mae"]),
        "zero_ring": zero_row,
        "cross_molecule_same_length_ring_permutation": permutations,
        "ring_code_density_all_objects": _code_stats(ring_alpha) if ring_alpha is not None else None,
        "ring_encoder_task_grad": ring_grad,
        "shared_bridge_task_grad": bridge_grad,
        "note": (
            "ring code density is measured on the shared bridge code; a near-dense code is a "
            "soft-threshold task dictionary, not a sparse-dictionary advantage"
        ),
    }
    _write_json(out, payload)
    print(f"[interventions] zero_ring delta={zero_row['delta_mae']:.6f}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# stage: analysis
# ---------------------------------------------------------------------------
def stage_analysis() -> dict[str, Any]:
    _ensure_dirs()
    run = _read_json_or(RESULTS_DIR / "run_seed0.json")
    probe_payload = _read_json_or(MECHANISM_DIR / "typed_cycle_probes.json")
    screen = _read_json_or(PROBE_STOP_PATH)
    timing = _read_json_or(RESULTS_DIR / "timing.json")
    correctness = _read_json_or(AUDIT_DIR / "correctness.json")
    if not run or "soup" not in run:
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "official_test_loaded": False,
            "status": "NOT_TRAINED",
            "verdict": "NOT_TRAINED",
            "screen": screen,
            "timing": timing,
            "correctness": {"all_passed": correctness.get("all_passed", False)},
        }
        _write_json(RESULTS_DIR / "summary.json", summary)
        return summary
    soup_valid = float(run["soup"]["soup_valid_mae"])
    # same-protocol soup train MAE
    model = _soup_model().eval()
    data = load_typed_split("valid", subset=None)
    mask = cm.C6_MASK
    valid_eval = _evaluate(model, data, mask)
    # soup train evaluation (no shuffle)
    train_data = load_typed_split("train")
    train_eval = _evaluate(model, train_data, mask)
    verdict = _band(soup_valid)
    summary = {
        "protocol_version": PROTOCOL_VERSION,
        "official_test_loaded": False,
        "status": "completed",
        "verdict": verdict,
        "M_S": soup_valid,
        "soup_valid_mae": soup_valid,
        "soup_train_mae": float(train_eval["mae"]),
        "train_valid_gap": float(valid_eval["mae"] - train_eval["mae"]),
        "best_valid_mae": float(run["best_valid_mae"]),
        "best_epoch": int(run["best_epoch"]),
        "soup_members": run["soup"]["members"],
        "member_valid_mae": run["soup"]["member_valid_mae"],
        "epochs": int(run["epochs_run"]),
        "wall_clock_s": float(run["wall_clock_s"]),
        "seconds_per_epoch": float(run["seconds_per_epoch"]),
        "peak_rss_mb": float(run.get("peak_rss_mb", float("nan"))),
        "parameters": run["parameter_audit"]["total_parameters"],
        "ring_encoder_parameters": run["parameter_audit"]["ring_encoder_parameters"],
        "reader_increment": run["parameter_audit"]["reader_increment"],
        "zero_ring": probe_payload.get("zero_ring"),
        "cross_molecule_permutations": probe_payload.get("cross_molecule_same_length_ring_permutation"),
        "ring_code_density": probe_payload.get("ring_code_density_all_objects"),
        "ring_encoder_task_grad": probe_payload.get("ring_encoder_task_grad"),
        "shared_bridge_task_grad": probe_payload.get("shared_bridge_task_grad"),
        "screen": screen,
        "timing": timing,
        "correctness_all_passed": bool(correctness.get("all_passed", False)),
        "git_commit": _git_commit(),
    }
    _write_json(RESULTS_DIR / "summary.json", summary)
    _write_report(summary, run)
    _write_decision(summary)
    print(f"[analysis] soup_valid={soup_valid:.6f} verdict={verdict}", flush=True)
    return summary


def _write_report(summary: Mapping[str, Any], run: Mapping[str, Any]) -> None:
    lines = [
        "# Typed-cycle static object into the shared task dictionary (ZINC, CPU)",
        "",
        f"- Round `{PROTOCOL_VERSION}`; seed 0; 320 epochs; official test never loaded.",
        f"- Verdict: `{summary['verdict']}`; soup valid MAE `{summary['M_S']:.6f}`.",
        f"- best valid `{summary['best_valid_mae']:.6f}` @ epoch {summary['best_epoch']}; "
        f"soup members `{summary['soup_members']}`.",
        f"- same-protocol soup train MAE `{summary['soup_train_mae']:.6f}`; "
        f"train->valid gap `{summary['train_valid_gap']:+.6f}`.",
        f"- parameters `{summary['parameters']}`; ring encoder `{summary['ring_encoder_parameters']}`; "
        f"reader increment `{summary['reader_increment']}`.",
        f"- wall `{summary['wall_clock_s']:.1f}s`; `{summary['seconds_per_epoch']:.3f}s/epoch`; peak RSS `{summary['peak_rss_mb']:.0f} MB`.",
        "",
        "## Inference-only diagnostics",
        f"- zero_ring: {json.dumps(summary.get('zero_ring'))}",
        f"- cross-molecule same-length ring permutation: {json.dumps(summary.get('cross_molecule_permutations'))}",
        f"- ring code density: {json.dumps(summary.get('ring_code_density'))}",
        f"- ring encoder task gradient `{summary.get('ring_encoder_task_grad')}`; "
        f"shared bridge task gradient `{json.dumps(summary.get('shared_bridge_task_grad'))}`.",
        "",
        "These interventions establish dependence, not a training gain, and the absolute band is a "
        "resource decision, not a significance test.",
        "",
    ]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_decision(summary: Mapping[str, Any]) -> None:
    lines = [
        "# DECISION - typed-cycle static object",
        "",
        f"`{summary['verdict']}` (soup valid MAE `{summary['M_S']:.6f}`).",
        "",
        "This is a single seed-0 trajectory with no matched control; the official test is never read.",
        "",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------
def configure(*, epochs: int | None = None, threads: int | None = None, device: str = "cpu") -> dict[str, Any]:
    if str(device) != "cpu":
        raise ValueError("typed-cycle round is CPU-only")
    if threads is not None:
        torch.set_num_threads(int(threads))
    return {"device": "cpu", "threads": int(threads or THREADS), "epochs": int(epochs or TRAIN_EPOCHS)}


def run_stages(stage: str, *, force: bool = False) -> dict[str, Any]:
    base.sem.cpu_only_guard(torch.device("cpu"))
    torch.set_num_threads(THREADS)
    if stage == "all":
        stage_data(force=force)
        stage_preflight()
        stage_correctness()
        stage_timing(force=force)
        stage_smoke(force=force)
        stage_train(force=force)
        stage_interventions(force=force)
        return stage_analysis()
    table = {
        "data": lambda: stage_data(force=force),
        "preflight": stage_preflight,
        "correctness": stage_correctness,
        "timing": lambda: stage_timing(force=force),
        "smoke": lambda: stage_smoke(force=force),
        "train": lambda: stage_train(force=force),
        "interventions": lambda: stage_interventions(force=force),
        "analysis": stage_analysis,
    }
    if stage not in table:
        raise ValueError(f"unknown stage {stage!r}")
    return table[stage]()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = run_stages(args.stage, force=args.force)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())