"""E2E-DictEnv-Purify-v0 runner — matched reference / purified arms on ZINC.

Round ``e2e_dictenv_purify_v0`` · study ``zinc-context-gap``.
Pre-registration: ``tracks/ksvd/notes/e2e_dictenv_purify_v0_preregistration.md``.
Core module: ``tracks/ksvd/experiments/luyin16/e2e_dictenv_purify_v0.py``.

Official ZINC **test is never loaded** in this round (hard blocker in
:func:`load_split`).  All CUDA work addresses physical **GPU1** only; GPU0 is
never touched and no DDP / multi-GPU path exists.

Stages::

    audit equiv smoke train mechanism health ablate paired decision all

``all`` is resumable: every stage skips work whose artifact already exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p1 as p1
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_p2_abs as p2
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_purify_v0 as pur
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_v0 as v0
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_p1 as p1run
from tracks.ksvd.experiments.luyin16 import zinc_e2e_dictenv_v0 as v0run
from tracks.ksvd.experiments.luyin16 import zinc_sdb_v0 as zsdb
from tracks.ksvd.experiments.luyin16.zinc_compact_v4_smallhead_e2e import REPO_ROOT

PROTOCOL_VERSION = pur.PROTOCOL_VERSION
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/e2e_dictenv_purify_v0"
STATE_DIR = RESULTS_DIR / "states"
CURVE_DIR = RESULTS_DIR / "curves"

BATCH_SIZE = int(p1run.BATCH_SIZE)
LEARNING_RATE = float(p1run.LEARNING_RATE)
WEIGHT_DECAY = float(p1run.WEIGHT_DECAY)
GRAD_CLIP = float(p1run.GRAD_CLIP)
TRAIN_SHUFFLE_OFFSET = int(p1run.TRAIN_SHUFFLE_OFFSET)
EVAL_SHUFFLE_OFFSET = int(p1run.EVAL_SHUFFLE_OFFSET)
SOUP_K = 5
SMOKE_MOLECULES = int(p1run.SMOKE_MOLECULES)  # 512
SMOKE_EPOCHS = int(p1run.SMOKE_EPOCHS)  # 3
SHUFFLE_SEEDS = tuple(p1run.SHUFFLE_SEEDS)  # (101, 202, 303, 404, 505)
MECHANISM_PERMUTATIONS = len(SHUFFLE_SEEDS)

#: historical H1 context only (never a decision input)
H1_HISTORICAL_SOUP = 0.12354862861608853
REFERENCE_REPRODUCTION_TOLERANCE = 0.005

#: frozen decision thresholds (pre-registration sections 10.3-10.5, 11)
SEED0_PASS = 0.002
SEED0_AMBIGUOUS_MAX = 0.004
MEAN_PASS = 0.002
WORST_SEED_TOLERANCE = 0.004
GATE_ZERO = 0.030
GATE_NODE = 0.010
GATE_ALL = 0.015

#: mechanism reference values (P1 continuity context only)
P1_MECHANISM_CONTEXT = {
    "G_zero": 0.3471155649620341,
    "G_node": 0.013123559097200616,
    "G_edge": 0.05952847671797498,
    "G_all": 0.07299894420839845,
}

ARMS = ("reference", "purified")
SPLITS = ("train", "valid")

OFFICIAL_TEST_BLOCKED = True

_write_json = v0run._write_json
_read_json = v0run._read_json
_write_csv = v0run._write_csv
_git_commit = v0run._git_commit
_n_params = v0run._n_params


# ---------------------------------------------------------------------------
# device / split discipline
# ---------------------------------------------------------------------------


def resolve_device(spec: str = "cuda") -> torch.device:
    """Resolve the compute device under the round's hard GPU1-only constraint.

    * ``cpu`` is allowed (audits / tests).
    * under ``CUDA_VISIBLE_DEVICES`` exactly one entry is permitted and it must
      be the physical device ``1`` (the launch wrapper's ``1`` argument);
    * without a mask the physical device is addressed explicitly as ``cuda:1``.

    There is no fallback to GPU0 and no multi-device path.
    """
    spec = str(spec)
    if spec == "cpu":
        return torch.device("cpu")
    if not (spec == "cuda" or spec.startswith("cuda:")):
        raise ValueError(f"unsupported device {spec!r}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        entries = [entry.strip() for entry in visible.split(",") if entry.strip() != ""]
        if len(entries) != 1:
            raise RuntimeError(f"exactly one visible GPU is authorized (GPU1 only), got {visible!r}")
        if entries[0] != "1":
            raise RuntimeError(f"only physical GPU1 is authorized, got CUDA_VISIBLE_DEVICES={visible!r}")
        return torch.device("cuda:0")
    return torch.device("cuda:1")


def load_split(split: str, subset: int | None = None) -> list[Any]:
    """Load official train / valid; the official test split is hard-blocked."""
    if split == "test" or split not in SPLITS:
        raise PermissionError(
            f"official ZINC test is never loaded in {PROTOCOL_VERSION} (requested {split!r})"
        )
    return p1run.load_split(split, subset=subset)


def env_cache_path(split: str) -> Path:
    return p1run._env_cache_path(split)


def dictionary() -> tuple[np.ndarray, str]:
    """Frozen SDB-v0 K-SVD dictionary ``sdb32`` (K=32) plus its tensor sha256."""
    D, _rand, _pca = zsdb.load_dictionary()
    D = np.asarray(D, dtype=np.float32)
    return D, hashlib.sha256(D.tobytes()).hexdigest()


def model_path(arm: str, seed: int, kind: str = "soup") -> Path:
    return STATE_DIR / f"{arm}_seed{int(seed)}_{kind}_state.pt"


def run_path(arm: str, seed: int) -> Path:
    return RESULTS_DIR / f"{arm}_seed{int(seed)}.json"


# ---------------------------------------------------------------------------
# stage 0 — audit artifacts
# ---------------------------------------------------------------------------


def audit() -> dict[str, Any]:
    """Write the frozen reference/architecture/purity ledgers."""
    D, sha = dictionary()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ledger = pur.architecture_ledger()
    ledger["dictionary_sha256"] = sha
    ledger["git_commit"] = _git_commit()

    actual: dict[str, int] = {}
    for arm in ARMS:
        config = pur.reference_config() if arm == "reference" else pur.purified_config()
        model = pur.build_model(config, D, seed=0)
        actual[arm] = int(_n_params(model))
        expected = int(ledger[arm]["whole_model"])
        if actual[arm] != expected:
            raise RuntimeError(f"parameter ledger mismatch for {arm}: {actual[arm]} != {expected}")
    ledger["actual_parameters"] = actual

    purity = {arm: pur.purity_audit(pur.reference_config() if arm == "reference" else pur.purified_config()) for arm in ARMS}
    inventory = pur.reference_inventory(dictionary_sha256=sha)
    flow = pur.information_flow()

    _write_json(RESULTS_DIR / "parameter_ledger.json", ledger)
    _write_json(RESULTS_DIR / "parameter_ledger_reference.json", ledger["reference"])
    _write_json(RESULTS_DIR / "purity_audit.json", purity)
    _write_json(RESULTS_DIR / "information_flow.json", flow)
    _write_json(RESULTS_DIR / "reference_inventory.json", inventory)
    print(
        f"[audit] reference={ledger['reference']['whole_model']} purified={ledger['purified']['whole_model']} "
        f"delta={ledger['delta']['absolute']} ({ledger['delta']['percent']:.3f}%)",
        flush=True,
    )
    return ledger


# ---------------------------------------------------------------------------
# stage 0 — semantic refactor equivalence
# ---------------------------------------------------------------------------


def _reference_p2_model() -> p2.P2Model:
    D, _sha = dictionary()
    config = p2.P2Config("REF", "h1", 48, 32, 8, 0.25, 320, "sdb32")
    return p2.build_model(config, D, seed=0)


def _reference_checkpoint() -> tuple[dict[str, torch.Tensor], str] | None:
    soup = TRACK_ROOT / "results/e2e_dictenv_p2_abs" / "states" / "H1_soup_state.pt"
    if soup.exists():
        return torch.load(soup, map_location="cpu", weights_only=False), str(soup)
    return None


def semantic_refactor_equivalence(n_molecules: int = 16, device: str = "cpu", write: bool = True) -> dict[str, Any]:
    """Old implementation vs refactored reference, intermediates and prediction.

    Uses the trained H1 soup checkpoint when present, otherwise the deterministic
    matched initial state; both go through the same comparison.
    """
    device_obj = resolve_device(device) if device != "cpu" else torch.device("cpu")
    D, sha = dictionary()
    old = _reference_p2_model().to(device_obj).eval()
    checkpoint = _reference_checkpoint()
    checkpoint_meta: dict[str, Any]
    if checkpoint is not None:
        state, path = checkpoint
        old.load_state_dict(state)
        checkpoint_meta = {"loaded": True, "path": path, "kind": "trained_H1_soup"}
    else:
        checkpoint_meta = {"loaded": False, "kind": "deterministic_initial_state"}
    new = pur.build_model(pur.reference_config(), D, seed=0).to(device_obj).eval()
    mapping = pur.load_p2_state(new, old.state_dict())
    params_identical = all(
        torch.equal(old.state_dict()[key].float(), new.state_dict()[pur.p2_key_to_purify(key)].float())
        for key in old.state_dict()
    )

    data = load_split("valid", subset=int(n_molecules))
    loader = p1.make_env_loader(data, int(n_molecules), False, EVAL_SHUFFLE_OFFSET)
    batch = next(iter(loader)).to(device_obj)

    with torch.no_grad():
        old_prediction, old_edges = _old_intermediates(old, batch)
        new_prediction, new_aux = new(batch, return_aux=True)

    comparisons = {
        "alpha": _max_abs(old_edges["coord"], new_aux["coord"]),
        "node_slots": _max_abs(old_edges["node_slots"], new_aux["node_slots"]),
        "edge_slots": _max_abs(old_edges["edge_slots"], new_aux["edge_slots"]),
        "anchor": _max_abs(batch.anchor.to(new_aux["anchor"].dtype), new_aux["anchor"]),
        "pair_value": _max_abs(old_edges["pair_value"], new_aux["pair_value"]),
        "unary": _max_abs(old_edges["unary"], new_aux["unary"]),
        "relation_readout": _max_abs(old_edges["relation_readout"], new_aux["relation_readout"]),
        "global_out": _max_abs(old_edges["global_out"], new_aux["global_out"]),
        "topology_out": _max_abs(old_edges["topology_out"], new_aux["topology_out"]),
        "unified": _max_abs(old_edges["unified"], new_aux["unified"]),
        "environment": _max_abs(old_edges["E"], new_aux["E"]),
        "prediction": _max_abs(old_prediction, new_prediction),
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "device": str(device_obj),
        "n_molecules": int(n_molecules),
        "n_nodes": int(batch.dict_phi.shape[0]),
        "checkpoint": checkpoint_meta,
        "state_mapping": mapping,
        "state_bit_identical": bool(params_identical),
        "comparisons_max_abs": comparisons,
        "max_abs": float(max(comparisons.values())),
        "tolerance": 1.0e-6,
        "passed": bool(max(comparisons.values()) <= 1.0e-6),
        "bit_identical": bool(all(value == 0.0 for value in comparisons.values())),
        "official_test_loaded": False,
    }
    if write:
        _write_json(RESULTS_DIR / "semantic_refactor_equivalence.json", payload)
    print(f"[equiv] max_abs={payload['max_abs']:.3e} passed={payload['passed']} bit_identical={payload['bit_identical']}", flush=True)
    if not payload["passed"]:
        raise RuntimeError("semantic refactor equivalence FAILED; purification must stop")
    return payload


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.float() - right.float()).abs().max().item())


def _old_intermediates(model: p2.P2Model, batch: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Recompute the legacy P2 forward step by step (same op order as P2Model)."""
    coord = model.code(batch.dict_phi)
    anchor = batch.anchor.to(coord.dtype)
    n = int(coord.shape[0])
    q = F.one_hot(batch.dict_atom, num_classes=p2.ATOM_CATEGORIES).to(coord.dtype)
    c = coord[batch.env_occ_node]
    qc = q[batch.env_occ_node]
    u = (c @ model.W_A_S) * (qc @ model.W_A_C) / math.sqrt(float(p2.D_A))
    flat = torch.zeros((n * p2.N_SHELLS, p2.D_A), device=u.device, dtype=u.dtype)
    flat.index_add_(0, batch.env_occ_root * p2.N_SHELLS + batch.env_occ_shell, u)
    node_slots = flat.view(n, p2.N_SHELLS, p2.D_A)
    d_e = int(model.config.d_e)
    cu = coord[batch.env_bond_u]
    cv = coord[batch.env_bond_v]
    g = torch.cat([cu + cv, torch.abs(cu - cv), cu * cv], dim=1)
    b = F.one_hot(batch.env_bond_type, num_classes=p2.BOND_CATEGORIES).to(coord.dtype)
    ue = (g @ model.W_E_S) * (b @ model.W_E_C) / math.sqrt(float(d_e))
    flat_e = torch.zeros((n * p2.SHELLPAIR_CLASSES, d_e), device=ue.device, dtype=ue.dtype)
    flat_e.index_add_(
        0, batch.env_bond_root * p2.SHELLPAIR_CLASSES + batch.env_bond_shellpair, ue
    )
    edge_slots = flat_e.view(n, p2.SHELLPAIR_CLASSES, d_e)
    node_out = model.node_encoder(node_slots)
    edge_out = model.edge_encoder(edge_slots)
    anchor_out = model.anchor_encoder(anchor)
    E = model.fusion(torch.cat([anchor_out, node_out.reshape(n, -1), edge_out.reshape(n, -1)], dim=1))

    n_graphs = int(batch.global_context.shape[0])
    unary = v0.pool_moments(E, batch.batch, n_graphs)
    source, target = batch.pair_index[0], batch.pair_index[1]
    projected = model.pair_projection(E)
    left, right = projected[source], projected[target]
    relation = model.relation_encoder(batch.pair_relation[:, list(p1.P1_RELATION_INDICES)])
    gate = 1.0 + torch.tanh(model.distance_gate(batch.pair_bucket))
    pair_input = torch.cat([left + right, torch.abs(left - right), (left * right) * gate, relation], dim=1)
    pair_value = model.pair_encoder(pair_input)
    relation_readout = v0.pool_pair_moments(pair_value, batch.batch[source], batch.pair_bucket, n_graphs)
    global_out = model.global_encoder(batch.global_context)
    topology_out = model.topology_encoder(batch.topology_features)
    unified = torch.cat([unary, relation_readout, global_out, topology_out], dim=1)
    prediction = model.reader(unified).view(-1)
    return prediction, {
        "coord": coord,
        "node_slots": node_slots,
        "edge_slots": edge_slots,
        "E": E,
        "pair_value": pair_value,
        "unary": unary,
        "relation_readout": relation_readout,
        "global_out": global_out,
        "topology_out": topology_out,
        "unified": unified,
    }


# ---------------------------------------------------------------------------
# training / evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: pur.PurifyV0Model,
    loader: Any,
    device: torch.device,
    mode: str = "clean",
    *,
    constant_node_scale: float = 1.0,
    constant_edge_scale: float = 1.0,
    return_environment: bool = False,
) -> dict[str, Any]:
    """Evaluate one intervention mode.

    ``mode`` mirrors the frozen mechanism semantics: ``clean`` / ``zero`` /
    ``node`` / ``edge`` / ``all``.  Shuffle indices are attached to the graphs
    (``env_occ_coord_node`` / ``env_bond_u_shuffled`` / ``env_bond_v_shuffled``)
    so that the shared collate applies the per-graph node offsets.
    """
    if mode not in ("clean", "zero", "node", "edge", "all"):
        raise ValueError(f"unknown evaluation mode {mode!r}")
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    environments: list[np.ndarray] = []
    rec_sum = 0.0
    node_count = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            interventions = pur.Interventions(
                coord_zero=(mode == "zero"),
                occ_coord_node=batch.env_occ_coord_node if mode in ("node", "all") else None,
                bond_u=batch.env_bond_u_shuffled if mode in ("edge", "all") else None,
                bond_v=batch.env_bond_v_shuffled if mode in ("edge", "all") else None,
                constant_node_scale=float(constant_node_scale),
                constant_edge_scale=float(constant_edge_scale),
            )
            prediction, aux = model(batch, interventions=interventions, return_aux=True)
            predictions.append(prediction.view(-1).cpu().numpy())
            targets.append(batch.y.view(-1).cpu().numpy())
            phi = aux["phi"]
            phi_hat = model.reconstruct(aux["coord"])
            numerator = ((phi - phi_hat) ** 2).sum(dim=1)
            denominator = (phi ** 2).sum(dim=1) + v0.EPS
            rec_sum += float((numerator / denominator).sum().item())
            node_count += int(phi.shape[0])
            if return_environment:
                environments.append(aux["E"].cpu().numpy())
    target = np.concatenate(targets).astype(np.float64)
    pred = np.concatenate(predictions).astype(np.float64)
    payload = {
        "mae": float(np.mean(np.abs(target - pred))),
        "rec": float(rec_sum / max(node_count, 1)),
        "n_molecules": int(target.shape[0]),
        "n_nodes": int(node_count),
        "targets": target,
        "predictions": pred,
    }
    if return_environment:
        payload["environments"] = np.concatenate(environments, axis=0)
    return payload


def _build_arm(arm: str, seed: int, D: np.ndarray) -> pur.PurifyV0Model:
    if arm == "reference":
        model = pur.build_model(pur.reference_config(), D, seed=seed)
    elif arm == "purified":
        reference, model, _report = pur.build_matched_pair(D, seed=seed)
        del reference
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return model


def train_arm(arm: str, seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    """One matched formal arm (resumable: skips when the artifact exists)."""
    path = run_path(arm, seed)
    if path.exists():
        return _read_json(path)
    device_obj = resolve_device(device)
    audit()
    semantic_refactor_equivalence(device=str(device_obj))
    D, dict_sha = dictionary()
    config = pur.reference_config() if arm == "reference" else pur.purified_config()
    ledger = pur.parameter_ledger(config)
    train_data = load_split("train")
    valid_data = load_split("valid")
    model = _build_arm(arm, seed, D).to(device_obj)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loader = p1.make_env_loader(train_data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    eval_loader = p1.make_env_loader(valid_data, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    lam = float(config.lambda_rec)
    if device_obj.type == "cuda":
        torch.cuda.manual_seed_all(int(seed))
        torch.cuda.reset_peak_memory_stats(device_obj)
    best_mae = float("inf")
    best_epoch = 1
    best_state: dict[str, torch.Tensor] | None = None
    epoch_states: dict[int, dict[str, torch.Tensor]] = {}
    curve: list[dict[str, Any]] = []
    gradient_probe: dict[str, float] = {}
    started = time.perf_counter()
    for epoch in range(1, int(config.horizon) + 1):
        model.train()
        task_sum = rec_sum = 0.0
        n_mol = n_nodes = 0
        for batch in loader:
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            if epoch == 1 and not gradient_probe:
                gradient_probe = _gradient_probe(model, config)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum().item())
            n_mol += int(batch.y.numel())
            phi = aux["phi"]
            phi_hat = model.reconstruct(aux["coord"]).detach()
            rec_sum += float((((phi - phi_hat) ** 2).sum(dim=1) / ((phi ** 2).sum(dim=1) + v0.EPS)).sum().item())
            n_nodes += int(phi.shape[0])
        train_mae = float(task_sum / max(n_mol, 1))
        train_rec = float(rec_sum / max(n_nodes, 1))
        valid = evaluate(model, eval_loader, device_obj)
        curve.append(
            {
                "epoch": int(epoch),
                "train_mae": train_mae,
                "train_rec": train_rec,
                "valid_mae": float(valid["mae"]),
                "valid_rec": float(valid["rec"]),
                "d_norm": float(model.D.detach().norm().item()),
            }
        )
        epoch_states[int(epoch)] = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        keep = {i + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:SOUP_K]}
        for cached in list(epoch_states):
            if cached not in keep:
                del epoch_states[cached]
        if float(valid["mae"]) < best_mae:
            best_mae = float(valid["mae"])
            best_epoch = int(epoch)
            best_state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 20 == 0 or epoch == int(config.horizon):
            print(
                f"[{arm}:{seed}] epoch={epoch:03d} train={train_mae:.6f} rec={train_rec:.2e} "
                f"valid={float(valid['mae']):.6f} best={best_mae:.6f}@{best_epoch}",
                flush=True,
            )
    wall = float(time.perf_counter() - started)
    assert best_state is not None
    members = sorted(int(i) + 1 for i in sorted(range(len(curve)), key=lambda i: float(curve[i]["valid_mae"]))[:SOUP_K])
    soup_state = {k: torch.stack([epoch_states[e][k].float() for e in members]).mean(0) for k in epoch_states[members[0]]}
    soup_model = _build_arm(arm, seed, D)
    soup_model.load_state_dict(soup_state)
    soup_valid = evaluate(soup_model.to(device_obj), eval_loader, device_obj)
    dbar_init = np.asarray(v0.normalized_dictionary(torch.as_tensor(D)))
    dbar_soup = np.asarray(v0.normalized_dictionary(soup_state["basis.D"].detach().float()))
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, model_path(arm, seed, "raw"))
    torch.save(soup_state, model_path(arm, seed, "soup"))
    _write_csv(CURVE_DIR / f"{arm}_seed{seed}_curve.csv", curve)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "seed": int(seed),
        "config": config.as_dict(),
        "parameters": ledger,
        "actual_params": int(_n_params(model)),
        "dictionary_sha256": dict_sha,
        "lambda_rec": lam,
        "horizon": int(config.horizon),
        "data": {"train": int(len(train_data)), "valid": int(len(valid_data)), "official_test_loaded": False},
        "device": str(device_obj),
        "best_valid_mae": float(best_mae),
        "best_epoch": int(best_epoch),
        "soup": {
            "members": members,
            "member_valid_mae": [float(curve[e - 1]["valid_mae"]) for e in members],
            "soup_valid_mae": float(soup_valid["mae"]),
        },
        "train_min_mae": float(min(row["train_mae"] for row in curve)),
        "train_mae_at_best": float(curve[best_epoch - 1]["train_mae"]),
        "valid_rec_soup": float(soup_valid["rec"]),
        "dictionary_movement": {
            "soup_vs_init_fro": float(np.linalg.norm(dbar_soup - dbar_init)),
            "soup_vs_init_relative": float(np.linalg.norm(dbar_soup - dbar_init) / (np.linalg.norm(dbar_init) + v0.EPS)),
        },
        "gradient_probe_epoch1": gradient_probe,
        "epochs_run": int(len(curve)),
        "wall_clock_s": wall,
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    print(f"[{arm}:{seed}] soup={float(soup_valid['mae']):.6f} best={best_mae:.6f}@{best_epoch} wall={wall:.1f}s", flush=True)
    return payload


def _gradient_probe(model: pur.PurifyV0Model, config: pur.PurifyConfig) -> dict[str, float]:
    """Non-zero-gradient probe for the core mechanisms (real MAE backward)."""
    probe: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        probe[name] = float(parameter.grad.norm().item()) if parameter.grad is not None else 0.0
    probe["_dictionary"] = probe.get("basis.D", 0.0)
    probe["_atom_valuation"] = probe.get("measure.W_A_C", 0.0)
    probe["_bond_valuation"] = probe.get("measure.W_E_C", 0.0)
    if config.constant_node_channel:
        probe["_constant_node_row"] = float(model.measure.W_A_S.grad[0].norm().item())
        probe["_constant_edge_row"] = float(model.measure.W_E_S.grad[0].norm().item())
    return probe


def load_soup(arm: str, seed: int, device: torch.device) -> pur.PurifyV0Model:
    D, _sha = dictionary()
    model = _build_arm(arm, seed, D)
    state = torch.load(model_path(arm, seed, "soup"), map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    return model.to(device).eval()


def codes_for_split(model: pur.PurifyV0Model, split: str, device: torch.device, chunk: int = 65536) -> np.ndarray:
    blob = torch.load(env_cache_path(split), map_location="cpu", weights_only=False)
    phi = blob["phi"]
    chunks: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, int(phi.shape[0]), int(chunk)):
            chunks.append(model.code(phi[start : start + int(chunk)].to(device)).cpu().numpy())
    return np.concatenate(chunks, axis=0)


# ---------------------------------------------------------------------------
# smoke
# ---------------------------------------------------------------------------


def smoke(arm: str, device: str = "cuda", seed: int = 0) -> dict[str, Any]:
    path = RESULTS_DIR / f"smoke_{arm}.json"
    if path.exists():
        return _read_json(path)
    device_obj = resolve_device(device)
    D, dict_sha = dictionary()
    config = pur.reference_config() if arm == "reference" else pur.purified_config()
    model = _build_arm(arm, seed, D).to(device_obj)
    data = load_split("train", subset=SMOKE_MOLECULES)
    loader = p1.make_env_loader(data, BATCH_SIZE, True, int(seed) + TRAIN_SHUFFLE_OFFSET)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    lam = float(config.lambda_rec)
    losses: list[float] = []
    recs: list[float] = []
    gradient_probe: dict[str, float] = {}
    started = time.perf_counter()
    epochs = int(SMOKE_EPOCHS)
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(loader):
            batch = batch.to(device_obj)
            prediction, aux = model(batch, return_aux=True)
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = F.l1_loss(prediction.view(-1), batch.y.view(-1)) + lam * rec
            optimizer.zero_grad()
            loss.backward()
            if epoch == 0 and step == 0:
                gradient_probe = _gradient_probe(model, config)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            losses.append(float(loss.detach().item()))
            recs.append(float(rec.detach().item()))
    model.eval()
    with torch.no_grad():
        check_batch = next(iter(p1.make_env_loader(data, BATCH_SIZE, False, 0))).to(device_obj)
        alpha = model.code(check_batch.dict_phi)
        _pred, aux = model(check_batch, return_aux=True)
    l0 = (alpha.abs() > 0).sum(dim=1)
    active = int((alpha.abs() > 0).any(dim=0).sum())
    environment_rank = p1run._environment_rank(aux["E"].cpu().numpy())
    gates = {
        "task_loss_finite": bool(all(math.isfinite(value) for value in losses)),
        "reconstruction_loss_finite": bool(all(math.isfinite(value) for value in recs)),
        "task_loss_decreased": bool(losses[-1] < losses[0]),
        "task_gradient_to_dictionary": bool(gradient_probe.get("_dictionary", 0.0) > 0.0),
        "atom_valuation_gradient": bool(gradient_probe.get("_atom_valuation", 0.0) > 0.0),
        "bond_valuation_gradient": bool(gradient_probe.get("_bond_valuation", 0.0) > 0.0),
        "exact_top_s": bool(int(l0.max()) <= int(config.s)),
        "dictionary_atoms_alive": bool(active >= 8),
        "environment_effective_rank_gt_1": bool(environment_rank["participation_ratio"] > 1.0),
        "no_nan_or_inf": bool(all(math.isfinite(value) for value in losses + recs)),
    }
    if config.constant_node_channel:
        gates["constant_node_row_gradient"] = bool(gradient_probe.get("_constant_node_row", 0.0) > 0.0)
        gates["constant_edge_row_gradient"] = bool(gradient_probe.get("_constant_edge_row", 0.0) > 0.0)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "device": str(device_obj),
        "seed": int(seed),
        "molecules": int(len(data)),
        "epochs": int(SMOKE_EPOCHS),
        "official_valid_read": False,
        "lambda_rec": lam,
        "parameters": pur.parameter_ledger(config),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "reconstruction_first": recs[0],
        "reconstruction_last": recs[-1],
        "gradient_probe": gradient_probe,
        "max_l0": int(l0.max()),
        "active_atoms": active,
        "environment_rank": environment_rank,
        "gates": gates,
        "all_passed": bool(all(gates.values())),
        "wall_clock_s": float(time.perf_counter() - started),
        "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated(device_obj) / (1024 ** 2)) if device_obj.type == "cuda" else None,
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    print(f"[smoke:{arm}] passed={payload['all_passed']} loss {losses[0]:.4f}->{losses[-1]:.4f} l0max={int(l0.max())}", flush=True)
    if not payload["all_passed"]:
        raise RuntimeError(f"smoke FAILED for {arm}: {payload['gates']}")
    return payload


# ---------------------------------------------------------------------------
# paired performance
# ---------------------------------------------------------------------------


def paired_performance() -> dict[str, Any]:
    """Frozen performance comparison (pre-registration sections 10.3-10.5)."""
    seeds = [seed for seed in (0, 1) if run_path("reference", seed).exists() and run_path("purified", seed).exists()]
    if 0 not in seeds:
        raise RuntimeError("seed-0 arms must both exist before the paired comparison")
    reference_soup = {seed: float(_read_json(run_path("reference", seed))["soup"]["soup_valid_mae"]) for seed in seeds}
    purified_soup = {seed: float(_read_json(run_path("purified", seed))["soup"]["soup_valid_mae"]) for seed in seeds}
    deltas = {seed: float(purified_soup[seed] - reference_soup[seed]) for seed in seeds}
    delta0 = deltas[0]
    reference_reproduction = {
        "historical_h1_soup": H1_HISTORICAL_SOUP,
        "fresh_reference_seed0_soup": reference_soup[0],
        "drift": float(reference_soup[0] - H1_HISTORICAL_SOUP),
        "tolerance": REFERENCE_REPRODUCTION_TOLERANCE,
        "passed": bool(abs(reference_soup[0] - H1_HISTORICAL_SOUP) <= REFERENCE_REPRODUCTION_TOLERANCE),
    }
    if delta0 <= SEED0_PASS:
        seed0_case = "non_inferior"
    elif delta0 <= SEED0_AMBIGUOUS_MAX:
        seed0_case = "ambiguous"
    else:
        seed0_case = "performance_failure"
    seed1_authorized = bool(seed0_case in ("non_inferior", "ambiguous"))
    mean_delta = float(np.mean([deltas[seed] for seed in seeds]))
    worst_delta = float(max(deltas.values()))
    if len(seeds) < 2:
        verdict = None
        accepted = False
    elif mean_delta <= MEAN_PASS and worst_delta <= WORST_SEED_TOLERANCE:
        verdict = "PURIFIED_ARCHITECTURE_ACCEPTED"
        accepted = True
    elif mean_delta <= SEED0_AMBIGUOUS_MAX:
        verdict = "PURIFICATION_INCONCLUSIVE"
        accepted = False
    else:
        verdict = "PURIFICATION_REJECTED"
        accepted = False
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "primary_quantity": "Delta_purification = MAE_purified - MAE_reference (Top-5 soup, official valid)",
        "seeds_available": seeds,
        "reference_soup": {str(seed): reference_soup[seed] for seed in seeds},
        "purified_soup": {str(seed): purified_soup[seed] for seed in seeds},
        "delta": {str(seed): deltas[seed] for seed in seeds},
        "delta_seed0": delta0,
        "reference_reproduction": reference_reproduction,
        "seed0_case": seed0_case,
        "seed1_authorized": seed1_authorized,
        "seed1_run": bool(1 in seeds),
        "thresholds": {
            "seed0_pass": SEED0_PASS,
            "seed0_ambiguous_max": SEED0_AMBIGUOUS_MAX,
            "mean_pass": MEAN_PASS,
            "worst_seed_tolerance": WORST_SEED_TOLERANCE,
        },
        "mean_delta": mean_delta if len(seeds) > 1 else None,
        "worst_seed_delta": worst_delta if len(seeds) > 1 else None,
        "verdict": verdict,
        "accepted": accepted,
        "official_test_loaded": False,
    }
    _write_json(RESULTS_DIR / "paired_performance.json", payload)
    print(
        f"[paired] seeds={seeds} delta={ {k: round(v, 6) for k, v in deltas.items()} } "
        f"case={seed0_case} verdict={verdict}",
        flush=True,
    )
    return payload


# ---------------------------------------------------------------------------
# mechanism interventions
# ---------------------------------------------------------------------------


def _permute_node_shuffle(data_list: Sequence[Any], seed: int) -> None:
    for data in data_list:
        data.env_occ_coord_node = p1.shuffled_occ_node_for_molecule(
            data.env_occ_node, data.env_occ_root, data.env_occ_shell, seed
        )


def _permute_edge_shuffle(data_list: Sequence[Any], seed: int) -> None:
    for data in data_list:
        u, v = p1.shuffled_bond_endpoints_for_molecule(
            data.env_bond_root, data.env_bond_shellpair, data.env_bond_u, data.env_bond_v, seed
        )
        data.env_bond_u_shuffled = u
        data.env_bond_v_shuffled = v


def _clear_shuffles(data_list: Sequence[Any]) -> None:
    for data in data_list:
        data.env_occ_coord_node = None
        data.env_bond_u_shuffled = None
        data.env_bond_v_shuffled = None


def mechanism_interventions(arm: str = "purified", seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    path = RESULTS_DIR / "mechanism_interventions.json"
    if path.exists():
        return _read_json(path)
    device_obj = resolve_device(device)
    valid = load_split("valid")
    loader = p1.make_env_loader(valid, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)

    def _clean() -> pur.PurifyV0Model:
        _clear_shuffles(valid)
        return load_soup(arm, seed, device_obj)

    aggregates: dict[str, Any] = {}
    for tag in ("zero", "node", "edge", "all"):
        per_permutation: list[dict[str, Any]] = []
        if tag == "zero":
            model = _clean()
            per_permutation.append(_intervention_record(evaluate(model, loader, device_obj, "zero")))
        else:
            for shuffle_seed in SHUFFLE_SEEDS:
                _clear_shuffles(valid)
                if tag in ("node", "all"):
                    _permute_node_shuffle(valid, shuffle_seed)
                if tag in ("edge", "all"):
                    _permute_edge_shuffle(valid, shuffle_seed)
                model = load_soup(arm, seed, device_obj)
                result = evaluate(model, loader, device_obj, tag)
                per_permutation.append(_intervention_record(result, shuffle_seed))
        maes = [record["mae"] for record in per_permutation]
        aggregates[tag] = {
            "mae_mean": float(np.mean(maes)),
            "mae_min": float(np.min(maes)),
            "mae_max": float(np.max(maes)),
            "permutations": per_permutation,
        }
    model = _clean()
    clean = evaluate(model, loader, device_obj)
    clean_mae = float(clean["mae"])
    degradations = {tag: float(aggregates[tag]["mae_mean"] - clean_mae) for tag in aggregates}
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "seed": int(seed),
        "device": str(device_obj),
        "clean_soup_valid_mae": clean_mae,
        "clean_valid_rec": float(clean["rec"]),
        "interventions": aggregates,
        "degradation": degradations,
        "gates": {
            "zero": {"value": degradations["zero"], "threshold": GATE_ZERO, "passed": bool(degradations["zero"] >= GATE_ZERO)},
            "node": {"value": degradations["node"], "threshold": GATE_NODE, "passed": bool(degradations["node"] >= GATE_NODE)},
            "all": {"value": degradations["all"], "threshold": GATE_ALL, "passed": bool(degradations["all"] >= GATE_ALL)},
            "edge": {"value": degradations["edge"], "threshold": None, "reported_only": True},
        },
        "mechanism_preserved": bool(
            degradations["zero"] >= GATE_ZERO and degradations["node"] >= GATE_NODE and degradations["all"] >= GATE_ALL
        ),
        "p1_mechanism_context": P1_MECHANISM_CONTEXT,
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    print(f"[mechanism:{arm}] degradations={ {k: round(v, 6) for k, v in degradations.items()} }", flush=True)
    return payload


def _intervention_record(result: Mapping[str, Any], shuffle_seed: int | None = None) -> dict[str, Any]:
    record = {"mae": float(result["mae"]), "rec": float(result["rec"]), "n_molecules": int(result["n_molecules"])}
    if shuffle_seed is not None:
        record["shuffle_seed"] = int(shuffle_seed)
    return record


def constant_channel_ablation(seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    """Inference-only constant-channel ablation on the purified soup (section 12)."""
    path = RESULTS_DIR / "constant_channel_ablation.json"
    if path.exists():
        return _read_json(path)
    device_obj = resolve_device(device)
    valid = load_split("valid")
    loader = p1.make_env_loader(valid, BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    model = load_soup("purified", seed, device_obj)
    clean = evaluate(model, loader, device_obj)
    clean_pred = clean["predictions"]
    records: dict[str, Any] = {}
    for tag, node_scale, edge_scale in (
        ("node_off", 0.0, 1.0),
        ("edge_off", 1.0, 0.0),
        ("both_off", 0.0, 0.0),
    ):
        result = evaluate(
            model,
            loader,
            device_obj,
            "clean",
            constant_node_scale=node_scale,
            constant_edge_scale=edge_scale,
        )
        shift = np.abs(result["predictions"] - clean_pred)
        records[tag] = {
            "mae": float(result["mae"]),
            "mae_delta": float(result["mae"] - clean["mae"]),
            "prediction_shift_mean": float(shift.mean()),
            "prediction_shift_median": float(np.median(shift)),
            "prediction_shift_p90": float(np.quantile(shift, 0.9)),
            "prediction_shift_max": float(shift.max()),
        }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "seed": int(seed),
        "device": str(device_obj),
        "clean_soup_valid_mae": float(clean["mae"]),
        "ablations": records,
        "not_an_architecture_selection_gate": True,
        "official_test_loaded": False,
    }
    _write_json(path, payload)
    print(f"[ablate] { {k: round(v['mae_delta'], 6) for k, v in records.items()} }", flush=True)
    return payload


# ---------------------------------------------------------------------------
# dictionary health
# ---------------------------------------------------------------------------


def dictionary_health(arm: str, seed: int = 0, device: str = "cuda") -> dict[str, Any]:
    path = RESULTS_DIR / "dictionary_health.json"
    if path.exists() and str(arm) in _read_json(path):
        return _read_json(path)
    device_obj = resolve_device(device)
    model = load_soup(arm, seed, device_obj)
    alpha_train = codes_for_split(model, "train", device_obj)
    alpha_valid = codes_for_split(model, "valid", device_obj)
    usage_train = p1run._atom_usage(alpha_train)
    usage_valid = p1run._atom_usage(alpha_valid)
    l0_train = (np.abs(alpha_train) > 0).sum(axis=1)
    l0_valid = (np.abs(alpha_valid) > 0).sum(axis=1)
    D, _sha = dictionary()
    dbar = np.asarray(v0.normalized_dictionary(model.D.detach().float().cpu()))
    dbar_init = np.asarray(v0.normalized_dictionary(torch.as_tensor(D)))
    singular = np.linalg.svd(dbar, compute_uv=False)
    energy = singular ** 2
    gram = dbar.T @ dbar
    off = gram - np.diag(np.diag(gram))
    blob_train = torch.load(env_cache_path("train"), map_location="cpu", weights_only=False)
    blob_valid = torch.load(env_cache_path("valid"), map_location="cpu", weights_only=False)

    def _split_rec(alpha: np.ndarray, blob: Mapping[str, Any]) -> float:
        phi = blob["phi"].numpy().astype(np.float64)
        phi_hat = alpha.astype(np.float64) @ dbar.astype(np.float64).T
        return float((((phi - phi_hat) ** 2).sum(axis=1) / ((phi ** 2).sum(axis=1) + v0.EPS)).mean())

    valid_loader = p1.make_env_loader(load_split("valid"), BATCH_SIZE, False, int(seed) + EVAL_SHUFFLE_OFFSET)
    environment_stats = evaluate(model, valid_loader, device_obj, return_environment=True)
    rank = p1run._environment_rank(environment_stats["environments"])
    batch = p1run._first_batch(load_split("train", subset=64), device_obj, 64)
    model.zero_grad(set_to_none=True)
    prediction = model(batch)
    task_loss = F.l1_loss(prediction.view(-1), batch.y.view(-1))
    task_loss.backward()
    task_grad = float(model.D.grad.norm().item()) if model.D.grad is not None else 0.0
    nonzero = np.abs(alpha_valid[np.abs(alpha_valid) > 0])
    entry = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "arm": arm,
        "seed": int(seed),
        "device": str(device_obj),
        "train": {
            "n_nodes": int(alpha_train.shape[0]),
            **usage_train,
            "max_l0": int(l0_train.max()),
            "exact_top_s_fraction": float((l0_train == int(model.config.s)).mean()),
        },
        "valid": {
            "n_nodes": int(alpha_valid.shape[0]),
            **usage_valid,
            "max_l0": int(l0_valid.max()),
            "exact_top_s_fraction": float((l0_valid == int(model.config.s)).mean()),
            "top8_share": usage_valid["top8_share"],
        },
        "usage_spearman_train_valid": p1run._spearman(np.asarray(usage_train["usage"]), np.asarray(usage_valid["usage"])),
        "code_variance": {
            "all_entries": float(np.var(alpha_valid)),
            "nonzero_entries": float(np.var(nonzero)) if nonzero.size else 0.0,
            "mean_abs_nonzero": float(nonzero.mean()) if nonzero.size else 0.0,
            "max_abs": float(np.abs(alpha_valid).max()),
        },
        "dictionary": {
            "effective_rank": float((energy.sum() ** 2) / float((energy ** 2).sum())),
            "coherence_max_abs_cosine": float(np.abs(off).max()),
            "movement_fro_from_ksvd_init": float(np.linalg.norm(dbar - dbar_init)),
            "movement_relative": float(np.linalg.norm(dbar - dbar_init) / (np.linalg.norm(dbar_init) + v0.EPS)),
        },
        "reconstruction": {"train": _split_rec(alpha_train, blob_train), "valid": _split_rec(alpha_valid, blob_valid)},
        "task_gradient_to_D": task_grad,
        "environment_effective_rank": rank,
        "official_test_loaded": False,
    }
    existing = _read_json(path) if path.exists() else {}
    existing[str(arm)] = entry
    _write_json(path, existing)
    print(
        f"[health:{arm}] active={usage_valid['active_atoms']} eff={usage_valid['effective_atom_count']:.2f} "
        f"valid_rec={entry['reconstruction']['valid']:.2e} gradD={task_grad:.4f}",
        flush=True,
    )
    return existing


# ---------------------------------------------------------------------------
# decision / report
# ---------------------------------------------------------------------------


def decision() -> dict[str, Any]:
    paired = _read_json(RESULTS_DIR / "paired_performance.json")
    purity = _read_json(RESULTS_DIR / "purity_audit.json")
    ledger = _read_json(RESULTS_DIR / "parameter_ledger.json")
    equivalence = _read_json(RESULTS_DIR / "semantic_refactor_equivalence.json")
    smokes = {arm: _read_json(RESULTS_DIR / f"smoke_{arm}.json") for arm in ARMS}
    mechanism = _read_json(RESULTS_DIR / "mechanism_interventions.json") if (RESULTS_DIR / "mechanism_interventions.json").exists() else None
    ablation = _read_json(RESULTS_DIR / "constant_channel_ablation.json") if (RESULTS_DIR / "constant_channel_ablation.json").exists() else None
    health = _read_json(RESULTS_DIR / "dictionary_health.json") if (RESULTS_DIR / "dictionary_health.json").exists() else None

    if not equivalence.get("passed"):
        verdict = "SEMANTIC_REFACTOR_EQUIVALENCE_FAILURE"
    elif not paired["reference_reproduction"]["passed"]:
        verdict = "REFERENCE_REPRODUCTION_FAILURE"
    elif not all(smoke["all_passed"] for smoke in smokes.values()):
        verdict = "SMOKE_FAILURE"
    elif paired["seed0_case"] == "performance_failure":
        verdict = "PURIFICATION_PERFORMANCE_FAILURE"
    elif paired.get("verdict") == "PURIFIED_ARCHITECTURE_ACCEPTED":
        verdict = "PURIFIED_ARCHITECTURE_ACCEPTED" if (mechanism or {}).get("mechanism_preserved", False) else "PERFORMANCE_PRESERVED_BUT_MECHANISM_CHANGED"
    elif paired.get("verdict") == "PURIFICATION_INCONCLUSIVE":
        verdict = "PURIFICATION_INCONCLUSIVE"
    elif paired.get("verdict") == "PURIFICATION_REJECTED":
        verdict = "PURIFICATION_REJECTED"
    else:
        verdict = "PURIFICATION_PENDING_SEED1"

    not_run = []
    if not paired["seed1_authorized"]:
        not_run.append({"artifact": "reference_seed1.json", "status": "NOT RUN", "reason": "seed-0 Delta > +0.004 (PURIFICATION_PERFORMANCE_FAILURE)"})
        not_run.append({"artifact": "purified_seed1.json", "status": "NOT RUN", "reason": "seed-0 Delta > +0.004 (PURIFICATION_PERFORMANCE_FAILURE)"})
    if mechanism is None:
        not_run.append({"artifact": "mechanism_interventions.json", "status": "NOT RUN", "reason": "candidate not accepted"})
    if ablation is None:
        not_run.append({"artifact": "constant_channel_ablation.json", "status": "NOT RUN", "reason": "candidate not accepted"})
    if health is None:
        not_run.append({"artifact": "dictionary_health.json", "status": "NOT RUN", "reason": "candidate not accepted"})

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "verdict": verdict,
        "paired_performance": paired,
        "purity_audit": purity,
        "parameter_ledger": {"reference": ledger["reference"]["whole_model"], "purified": ledger["purified"]["whole_model"], "delta": ledger["delta"]},
        "semantic_refactor_equivalence": {"passed": equivalence.get("passed"), "max_abs": equivalence.get("max_abs"), "bit_identical": equivalence.get("bit_identical")},
        "smoke": {arm: {"all_passed": smoke["all_passed"]} for arm, smoke in smokes.items()},
        "mechanism": None if mechanism is None else {
            "degradation": mechanism["degradation"],
            "mechanism_preserved": mechanism["mechanism_preserved"],
        },
        "constant_channel_ablation": None if ablation is None else ablation["ablations"],
        "dictionary_health": None if health is None else {arm: {k: health[arm][k] for k in ("train", "valid", "reconstruction", "task_gradient_to_D", "environment_effective_rank", "dictionary")} for arm in health},
        "not_run": not_run,
        "official_test_loaded": False,
        "scope_of_claim": (
            "local zeroth-order chemistry can be consolidated into the attributed structural measure "
            "without material loss under the frozen H1 architecture"
        ),
        "not_claimed": ["all chemistry bypasses are unnecessary", "global composition is redundant"],
    }
    _write_json(RESULTS_DIR / "decision.json", payload)
    return payload


def report() -> dict[str, Any]:
    payload = decision()
    paired = payload["paired_performance"]
    ledger = payload["parameter_ledger"]
    lines = [
        "# E2E-DictEnv-Purify-v0 — report",
        "",
        f"Frozen verdict: **{payload['verdict']}**",
        "",
        "Structure basis -> attribute valuation -> static composition.  Official ZINC test was **never** loaded.",
        "",
        "## Stage 0 — semantic refactor equivalence",
        "",
        f"* reference checkpoint comparison max |old - new| = {payload['semantic_refactor_equivalence']['max_abs']:.3e} "
        f"(tolerance 1e-6, bit-identical: {payload['semantic_refactor_equivalence']['bit_identical']})",
        f"* parameter ledger: reference {ledger['reference']} -> purified {ledger['purified']} "
        f"({ledger['delta']['absolute']:+d}, {ledger['delta']['percent']:+.3f} %)",
        "",
        "## Primary result (Top-5 soup, official valid)",
        "",
        "| seed | reference | purified | Delta |",
        "|---|---:|---:|---:|",
    ]
    for seed in paired["seeds_available"]:
        lines.append(
            f"| {seed} | {paired['reference_soup'][str(seed)]:.6f} | {paired['purified_soup'][str(seed)]:.6f} | "
            f"{paired['delta'][str(seed)]:+.6f} |"
        )
    lines += [
        "",
        f"* seed-0 case: `{paired['seed0_case']}`; seed 1 authorized: {paired['seed1_authorized']}; seed 1 run: {paired['seed1_run']}",
        f"* fresh reference reproduction drift vs historical H1 context = {paired['reference_reproduction']['drift']:+.6f} "
        f"(tolerance {paired['reference_reproduction']['tolerance']})",
    ]
    if paired["mean_delta"] is not None:
        lines.append(f"* mean Delta = {paired['mean_delta']:+.6f}; worst seed Delta = {paired['worst_seed_delta']:+.6f}")
    if payload["mechanism"] is not None:
        lines += ["", "## Mechanism (inference only, purified soup)", "", "```"]
        lines += [f"{key}: {value:+.6f}" for key, value in payload["mechanism"]["degradation"].items()]
        lines += ["```", f"* mechanism preserved: {payload['mechanism']['mechanism_preserved']}"]
    if payload["constant_channel_ablation"] is not None:
        lines += ["", "## Constant-channel ablation", ""]
        for tag, record in payload["constant_channel_ablation"].items():
            lines.append(
                f"* `{tag}`: MAE delta {record['mae_delta']:+.6f}, shift mean {record['prediction_shift_mean']:.4f}, "
                f"median {record['prediction_shift_median']:.4f}, p90 {record['prediction_shift_p90']:.4f}"
            )
    if payload["dictionary_health"] is not None:
        lines += ["", "## Dictionary health", ""]
        for arm, health in payload["dictionary_health"].items():
            lines.append(
                f"* {arm}: active {health['valid']['active_atoms']}/32, effective {health['valid']['effective_atom_count']:.2f}, "
                f"valid rec {health['reconstruction']['valid']:.2e}, ||dL/dD|| {health['task_gradient_to_D']:.4f}, "
                f"environment rank {health['environment_effective_rank']['participation_ratio']:.2f}"
            )
    lines += ["", "## Not run", ""]
    if payload["not_run"]:
        lines += [f"* `{record['artifact']}` — {record['status']}: {record['reason']}" for record in payload["not_run"]]
    else:
        lines.append("* none")
    lines += ["", "## Claim scope", "", f"> {payload['scope_of_claim']}", "", "Not claimed: " + "; ".join(payload["not_claimed"]), ""]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    _write_decision(payload)
    return payload


def _write_decision(payload: Mapping[str, Any]) -> None:
    paired = payload["paired_performance"]
    lines = [
        "# E2E-DictEnv-Purify-v0 — decision",
        "",
        "```",
        f"delta_purification = MAE_purified - MAE_reference",
        f"delta_seed0        = {paired['delta_seed0']:+.6f}",
        f"seed0_case         = {paired['seed0_case']}",
        f"seed1_authorized   = {paired['seed1_authorized']}",
        f"verdict            = {payload['verdict']}",
        "```",
        "",
        f"* reference seed 0 soup {paired['reference_soup']['0']:.6f} (historical context {H1_HISTORICAL_SOUP:.6f}, "
        f"drift {paired['reference_reproduction']['drift']:+.6f})",
        f"* purified seed 0 soup {paired['purified_soup']['0']:.6f}",
        f"* parameter ledger {payload['parameter_ledger']['reference']} -> {payload['parameter_ledger']['purified']}",
        f"* official test never loaded",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def run_all(device: str = "cuda") -> None:
    audit()
    semantic_refactor_equivalence()
    for arm in ARMS:
        smoke(arm, device=device)
    for arm in ARMS:
        train_arm(arm, seed=0, device=device)
    paired = paired_performance()
    if not paired["reference_reproduction"]["passed"]:
        raise RuntimeError("REFERENCE_REPRODUCTION_FAILURE: diagnose before interpreting the candidate")
    if paired["seed0_case"] == "performance_failure":
        report()
        return
    if paired["seed1_authorized"]:
        for arm in ARMS:
            train_arm(arm, seed=1, device=device)
        paired = paired_performance()
    if paired["accepted"]:
        mechanism_interventions(device=device)
        constant_channel_ablation(device=device)
        dictionary_health("purified", device=device)
    dictionary_health("reference", device=device)
    report()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["audit", "equiv", "smoke", "train", "mechanism", "health", "ablate", "paired", "decision", "report", "all"],
    )
    parser.add_argument("--arm", default="purified", choices=list(ARMS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.stage == "audit":
        print(json.dumps(audit(), indent=2))
    elif args.stage == "equiv":
        print(json.dumps(semantic_refactor_equivalence(device=args.device), indent=2))
    elif args.stage == "smoke":
        print(json.dumps(smoke(args.arm, device=args.device, seed=args.seed), indent=2))
    elif args.stage == "train":
        print(json.dumps(train_arm(args.arm, seed=args.seed, device=args.device), indent=2))
    elif args.stage == "mechanism":
        print(json.dumps(mechanism_interventions(args.arm, seed=args.seed, device=args.device), indent=2))
    elif args.stage == "health":
        print(json.dumps(dictionary_health(args.arm, seed=args.seed, device=args.device), indent=2))
    elif args.stage == "ablate":
        print(json.dumps(constant_channel_ablation(seed=args.seed, device=args.device), indent=2))
    elif args.stage == "paired":
        print(json.dumps(paired_performance(), indent=2))
    elif args.stage == "decision":
        print(json.dumps(decision(), indent=2))
    elif args.stage == "report":
        print(json.dumps(report(), indent=2))
    else:
        run_all(device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
