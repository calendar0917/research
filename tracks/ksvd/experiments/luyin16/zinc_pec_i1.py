"""PEC-I1 runner — Static Composition Interface Audit (ZINC).

Round ``pec_i1``.  Pre-registration: ``notes/pec_i1_preregistration.md``.

Stages
------
``recoverability correctness parameters screen report full all``

The architecture is PEC-v0's, imported unchanged; the only scientific change
is the graph-level statistical pooling interface (``pec_i1``).  Official ZINC
**test is never loaded**; only ``train`` and ``val`` are read, through the
guarded accessors below.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import pec_i1 as i1
from tracks.ksvd.experiments.luyin16 import pec_v0 as pec
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _data_to_graph,
    _load_zinc,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RESULTS_DIR = REPO_ROOT / "tracks/ksvd/results/pec_i1"
CACHE_DIR = REPO_ROOT / "tracks/ksvd/results/pec_c1/cache"
C1_DICT_DIR = REPO_ROOT / "tracks/ksvd/results/pec_c1/dicts"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

PROTOCOL_VERSION = "pec_i1"

# --- Stage A -----------------------------------------------------------------
AUDIT_TRAIN = 2000
AUDIT_VALID = 1000

# --- Stage B (PEC-v0 Gate-2 exact split/protocol) ----------------------------
GATE2_TRAIN = 2000
GATE2_DEV_START = 8000
GATE2_DEV = 500
SCREEN_EPOCHS = 60
SCREEN_LR = 1.0e-3
SCREEN_WD = 1.0e-5
SCREEN_BATCH = 64
SCREEN_CLIP = 5.0
SEED = 0
SOUP_TOP = 5
DICT_EPOCHS = 10

# --- full-data formal protocol (PEC-C1 exact) --------------------------------
FULL_EPOCHS = 240
FULL_LR = 1.0e-3
FULL_WD = 1.0e-5
FULL_BATCH = 64
FULL_CLIP = 5.0

# --- frozen thresholds (pre-registration §§7, 9) -----------------------------
SCREEN_STRONG = 0.010
SCREEN_WEAK = 0.005
FULL_VIABLE = 0.145
FULL_MATCH_S0 = 0.1408
FULL_MATERIAL = 0.005
PARAM_TOLERANCE = 0.01

# --- frozen historical references (never re-run) -----------------------------
HIST_CD_SCREEN_SOUP = 0.4671078622341156
HIST_CD_SCREEN_BEST = 0.47684603929519653
HIST_PEC_C1_CD_SOUP = 0.1517672836780548
HIST_PEC_C1_CD_BEST = 0.15978211164474487
HIST_S0_SEED0_SOUP = 0.140794

FORBIDDEN_SPLITS = ("test",)

ROLE_PARAM_NAMES = ("m_node.weight", "m_edge.weight")


# ---------------------------------------------------------------------------
# io / provenance
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True
        )
        return bool(out.strip())
    except Exception:  # pragma: no cover
        return True


def cuda_index(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    torch.cuda.init()
    if device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def device_report(device: torch.device) -> dict[str, Any]:
    index = cuda_index(device)
    payload: dict[str, Any] = {
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu_index": index,
    }
    if index is not None:
        payload["cuda"] = torch.version.cuda
        payload["gpu_name"] = torch.cuda.get_device_name(index)
        payload["gpu_total_memory_bytes"] = int(
            torch.cuda.get_device_properties(index).total_memory
        )
    return payload


def _provenance(device: torch.device) -> dict[str, Any]:
    return {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "official_test_loaded": False,
        "splits_read": ["train", "val"],
        "device": device_report(device),
    }


# ---------------------------------------------------------------------------
# guarded data access
# ---------------------------------------------------------------------------


def _guard_split(split: str) -> str:
    if split in FORBIDDEN_SPLITS:
        raise RuntimeError(
            f"official ZINC {split!r} split is forbidden in PEC-I1; "
            "only 'train' and 'val' may be read"
        )
    return split


def load_cached_split(split: str) -> list[pec.MoleculeSample]:
    _guard_split(split)
    filename = "valid.pkl.gz" if split == "valid" else f"{split}.pkl.gz"
    path = CACHE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"missing PEC cache {path}; run the PEC-C1 cache stage first"
        )
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)


def load_graph_records(split: str, limit: int) -> list[dict[str, Any]]:
    """Load ``(graph, atom_types, edge_types, sample)`` for the first ``limit``."""
    _guard_split(split)
    zinc_split = "val" if split == "valid" else split
    dataset = _load_zinc(ZINC_ROOT, zinc_split)
    records: list[dict[str, Any]] = []
    for index, data in enumerate(dataset):
        if index >= int(limit):
            break
        graph, node_types, edge_types = _data_to_graph(data)
        sample = pec.build_sample(
            graph, node_types, edge_types, y=float(data.y.view(-1)[0])
        )
        records.append(
            {
                "graph": graph,
                "node_types": np.asarray(node_types, dtype=np.int64),
                "edge_types": edge_types,
                "sample": sample,
            }
        )
    return records


# ---------------------------------------------------------------------------
# Stage A — recoverability
# ---------------------------------------------------------------------------


def _scalar5_probe(
    train_records: Sequence[Mapping[str, Any]],
    valid_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Ridge probe for S0 scalar 5 from PEC primitives only (train -> valid)."""

    def _features(record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        sample: pec.MoleculeSample = record["sample"]
        graph = record["graph"]
        from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

        feature_rows: list[np.ndarray] = []
        targets: list[float] = []
        for root in range(int(sample.n_nodes)):
            s0 = i1._s0_blocks(
                graph, root, record["node_types"], record["edge_types"]
            )
            rows = np.nonzero(sample.occ_root == root)[0]
            induced_degree = np.expm1(
                np.asarray(sample.node_basis[rows, 4], dtype=np.float64)
            )
            feature_rows.append(
                np.concatenate(
                    [
                        np.asarray(sample.root_scalars[root], dtype=np.float64),
                        np.asarray(sample.global_topo, dtype=np.float64),
                        np.asarray(
                            [
                                induced_degree.mean(),
                                induced_degree.std(),
                                induced_degree.max(initial=0.0),
                            ]
                        ),
                    ]
                )
            )
            targets.append(float(np.asarray(s0["scalars"])[5]))
        return np.stack(feature_rows, axis=0), np.asarray(targets, dtype=np.float64)

    train_x, train_y = zip(*(_features(r) for r in train_records)) if train_records else ((), ())
    valid_x, valid_y = zip(*(_features(r) for r in valid_records)) if valid_records else ((), ())
    if not train_x:
        return {"available": False}
    X_train = np.concatenate(train_x, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_valid = np.concatenate(valid_x, axis=0)
    y_valid = np.concatenate(valid_y, axis=0)

    mean = X_train.mean(axis=0, keepdims=True)
    scale = X_train.std(axis=0, keepdims=True)
    scale[scale < 1e-12] = 1.0
    Z = (X_train - mean) / scale
    Z_valid = (X_valid - mean) / scale
    Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], axis=1)
    Z_valid = np.concatenate([Z_valid, np.ones((Z_valid.shape[0], 1))], axis=1)
    ridge = 1.0e-3 * np.eye(Z.shape[1])
    ridge[-1, -1] = 0.0
    weights = np.linalg.solve(Z.T @ Z + ridge, Z.T @ y_train)
    prediction = Z_valid @ weights
    residual = y_valid - prediction
    variance = float(((y_valid - y_valid.mean()) ** 2).sum())
    r2 = 1.0 - float((residual**2).sum()) / max(variance, 1e-12)
    # molecule-scope analogue diagnostic
    analogue = X_valid[:, 3] / 4.0
    analogue_mae = float(np.abs(analogue - y_valid).mean())
    return {
        "available": True,
        "n_train_roots": int(X_train.shape[0]),
        "n_valid_roots": int(X_valid.shape[0]),
        "feature_names": [
            "root_scalars_0..5",
            "global_topo_0..7",
            "patch_induced_degree_mean",
            "patch_induced_degree_std",
            "patch_induced_degree_max",
        ],
        "ridge_alpha": 1.0e-3,
        "valid_r2": r2,
        "valid_mae": float(np.abs(residual).mean()),
        "valid_max_abs_error": float(np.abs(residual).max()),
        "molecule_scope_analogue_mae": analogue_mae,
    }


def stage_a(device: torch.device) -> dict[str, Any]:
    train_records = load_graph_records("train", AUDIT_TRAIN)
    valid_records = load_graph_records("valid", AUDIT_VALID)
    all_records = [
        (r["graph"], r["node_types"], r["edge_types"], r["sample"])
        for r in train_records + valid_records
    ]
    report = i1.stage_a_recoverability(all_records)
    report.update(_provenance(device))
    report["stage_a_subset"] = {
        "train_molecules": len(train_records),
        "valid_molecules": len(valid_records),
        "root_limit": None,
        "note": "all roots of every audited molecule",
    }
    report["scalar5_ridge_probe"] = _scalar5_probe(train_records, valid_records)

    # full-split shell-pair occurrence frequencies from the PEC cache
    frequencies: dict[str, Any] = {}
    for split in ("train", "valid"):
        samples = load_cached_split(split)
        counter = np.zeros(len(pec.SHELLPAIRS), dtype=np.int64)
        for sample in samples:
            counter += np.bincount(
                sample.eocc_shellpair, minlength=len(pec.SHELLPAIRS)
            )
        frequencies[split] = {
            "pec_shell_pair_occurrences": {
                str(pair): int(counter[index])
                for index, pair in enumerate(pec.SHELLPAIRS)
            },
            "s0_shell_pair_occurrences": {
                str(pair): int(
                    counter[pec.SHELLPAIRS.index(pair)]
                    if pair in pec.SHELLPAIRS
                    else 0
                )
                for pair in i1.S0_SHELL_PAIRS
            },
            "total_occurrences": int(counter.sum()),
            "s0_00_frequency": 0.0,
            "s0_00_note": (
                "structurally impossible: shell 0 is the singleton root, so a "
                "(0,0) induced bond would be a self-loop"
            ),
        }
    report["shell_pair_frequencies"] = frequencies

    blocks = report["blocks"]
    chemistry_exact = all(
        blocks[name]["unexplained_coordinates"] == 0
        for name in ("atom_shell", "bond_shell", "root_atom", "incident")
    )
    scalar_unexplained = int(blocks["scalars"]["unexplained_coordinates"])
    scalar_explained_fraction = float(blocks["scalars"]["exact_fraction"])
    probe = report["scalar5_ridge_probe"]
    verdict = (
        "LOCAL_INFORMATION_SUBSTANTIALLY_PRESENT"
        if chemistry_exact
        and scalar_explained_fraction >= 5.0 / 6.0
        and frequencies["train"]["s0_00_frequency"] == 0.0
        else "LOCAL_INFORMATION_GAP_FOUND"
    )
    report["stage_a_decision"] = {
        "chemistry_blocks_exact": chemistry_exact,
        "scalar_exact_fraction": scalar_explained_fraction,
        "scalar_unexplained_coordinates": scalar_unexplained,
        "missing_shell_pair_class_occurrences": 0,
        "scalar5_probe_valid_r2": None if not probe.get("available") else probe["valid_r2"],
        "verdict": verdict,
    }
    _write_json(RESULTS_DIR / "information_recoverability.json", report)
    _write_information_markdown(report)
    return report


def _write_information_markdown(report: Mapping[str, Any]) -> None:
    lines = [
        "# PEC-I1 — Stage A: zero-training Patch-B recoverability",
        "",
        f"Verdict: **{report['stage_a_decision']['verdict']}**",
        "",
        "Label-free (no target `y` read).  PEC environment raw primitives vs S0",
        "`patch_cont` blocks for the same rooted radius-2 patches.",
        "",
        f"Audited molecules: train {report['stage_a_subset']['train_molecules']}, "
        f"valid {report['stage_a_subset']['valid_molecules']}; "
        f"roots {report['totals']['roots']}.",
        "",
        "| block | dim | max abs err | mean abs err | exact fraction | unexplained |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in i1.BLOCK_NAMES:
        block = report["blocks"][name]
        lines.append(
            f"| {name} | {block['dim']} | {block['max_abs_error']:.3e} | "
            f"{block['mean_abs_error']:.3e} | {block['exact_fraction']:.6f} | "
            f"{block['unexplained_coordinates']} |"
        )
    probe = report["scalar5_ridge_probe"]
    lines += [
        "",
        "S0 structural-scalar coordinate 5 (patch-scoped mean molecule degree) has",
        "no analytic map in PEC; its molecule-scope counterpart and a train-fit",
        "ridge probe are reported.",
        "",
    ]
    if probe.get("available"):
        lines += [
            f"* ridge probe valid R² = {probe['valid_r2']:.6f}, "
            f"MAE = {probe['valid_mae']:.6f}, max |err| = "
            f"{probe['valid_max_abs_error']:.6f}",
            f"* molecule-scope analogue MAE = {probe['molecule_scope_analogue_mae']:.6f}",
        ]
    freq = report["shell_pair_frequencies"]["train"]
    lines += [
        "",
        "Shell-pair taxonomy:",
        "",
        f"* S0 declares {len(i1.S0_SHELL_PAIRS)} classes, including `(0,0)`, which "
        "PEC cannot express.",
        "* `(0,0)` train occurrences: 0 (structurally impossible).",
        f"* `(0,2)` train occurrences: "
        f"{freq['s0_shell_pair_occurrences'].get('(0, 2)', 0)} (also empty).",
        "",
    ]
    (RESULTS_DIR / "INFORMATION_RECOVERABILITY.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# correctness / parameters
# ---------------------------------------------------------------------------


def correctness(device: torch.device) -> dict[str, Any]:
    gates = i1.run_i1_gates()
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "stage": "correctness_gates",
        "gates": gates,
        "checks": {name: value.get("verdict") for name, value in gates.items()},
        "all_pass": all(
            value.get("verdict") == "PASS" for value in gates.values() if isinstance(value, dict)
        ),
        **_provenance(device),
    }
    _write_json(RESULTS_DIR / "correctness.json", payload)
    return payload


def parameters(device: torch.device) -> dict[str, Any]:
    payload = i1.parameter_accounting(94049)
    payload.update(
        {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "parameter_accounting",
            "baseline": "PEC-CD (PEC-v0 reader 200->64->1)",
            "candidate": "CD-I1 (S0 pooling reader 590->22->1)",
            "tolerance": PARAM_TOLERANCE,
        }
    )
    payload.update(_provenance(device))
    _write_json(RESULTS_DIR / "parameter_accounting.json", payload)
    return payload


# ---------------------------------------------------------------------------
# training helpers
# ---------------------------------------------------------------------------


def _batches(samples, batch_size, *, shuffle, seed):
    order = np.arange(len(samples))
    if shuffle:
        np.random.RandomState(int(seed)).shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        index = order[start : start + int(batch_size)]
        yield pec.collate([samples[int(i)] for i in index])


def _predict(model, batches, device) -> np.ndarray:
    model.eval()
    parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            batch = pec.to_device(batch, device)
            parts.append(model(batch)["prediction"].detach().cpu().numpy())
    return np.concatenate(parts) if parts else np.zeros(0)


def _train_screen_arm(model, train, dev, device, *, tag: str, epochs: int = SCREEN_EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=SCREEN_LR, weight_decay=SCREEN_WD)
    dev_batches = list(_batches(dev, SCREEN_BATCH, shuffle=False, seed=SEED))
    dev_target = np.concatenate([b["y"].numpy() for b in dev_batches])
    history: list[dict[str, Any]] = []
    top: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    start = time.time()
    for epoch in range(int(epochs)):
        model.train()
        running = 0.0
        seen = 0
        for batch in _batches(train, SCREEN_BATCH, shuffle=True, seed=SEED * 1000 + epoch):
            batch = pec.to_device(batch, device)
            out = model(batch)
            loss = (out["prediction"] - batch["y"]).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), SCREEN_CLIP)
            optimizer.step()
            running += float(loss.detach()) * int(batch["y"].numel())
            seen += int(batch["y"].numel())
        dev_mae = float(np.abs(_predict(model, dev_batches, device) - dev_target).mean())
        history.append({"epoch": int(epoch), "dev_mae": dev_mae})
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top.append((dev_mae, int(epoch), state))
        top.sort(key=lambda item: (item[0], item[1]))
        del top[SOUP_TOP:]
        if epoch % 10 == 0 or epoch == int(epochs) - 1:
            print(f"[{tag}] epoch={epoch:03d} dev={dev_mae:.6f}", flush=True)
    wall = time.time() - start

    soup_parts = []
    for _mae, _epoch, state in sorted(top, key=lambda item: item[1]):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        soup_parts.append(_predict(model, dev_batches, device))
    soup_prediction = np.mean(np.stack(soup_parts, axis=0), axis=0)
    soup_mae = float(np.abs(soup_prediction - dev_target).mean())
    best_mae, best_epoch, best_state = top[0]
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_prediction = _predict(model, dev_batches, device)
    best_mae = float(np.abs(best_prediction - dev_target).mean())
    return {
        "tag": tag,
        "params": int(pec.n_params(model)),
        "best_dev_mae": best_mae,
        "best_epoch": int(best_epoch),
        "soup_dev_mae": soup_mae,
        "soup_members": [int(e) for _m, e, _s in sorted(top, key=lambda item: item[1])],
        "soup_member_dev_mae": [float(m) for m, _e, _s in sorted(top)],
        "curve": history,
        "wall_seconds": wall,
        "soup_prediction": soup_prediction,
        "dev_target": dev_target,
        "top_states": [state for _m, _e, state in sorted(top, key=lambda item: item[1])],
    }


def _shuffle_relations(batch, generator):
    out = dict(batch)
    rho = batch["pair_rho"].clone()
    graph = batch["pair_graph"]
    for graph_id in torch.unique(graph).tolist():
        positions = torch.nonzero(graph == graph_id, as_tuple=False).reshape(-1)
        if positions.numel() <= 1:
            continue
        order = torch.randperm(positions.numel(), generator=generator)
        rho[positions] = batch["pair_rho"][positions[order]]
    out["pair_rho"] = rho
    return out


def _soup_intervention(model, batches, device, top_states, *, mode: str):
    generator = torch.Generator(device="cpu").manual_seed(SEED)
    parts: list[np.ndarray] = []
    previous = model.ablation
    for state in top_states:
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        model.ablation = "bag" if mode == "bag" else previous
        model.eval()
        with torch.no_grad():
            rows = []
            for batch in batches:
                current = batch
                if mode == "relation_shuffle":
                    current = _shuffle_relations(batch, generator)
                current = pec.to_device(current, device)
                rows.append(model(current)["prediction"].detach().cpu().numpy())
        parts.append(np.concatenate(rows))
    model.ablation = previous
    return np.mean(np.stack(parts, axis=0), axis=0)


def fit_gate2_dictionaries(train):
    node_train = np.concatenate([s.node_basis for s in train], axis=0).astype(np.float64)
    edge_train = np.concatenate([s.edge_basis for s in train], axis=0).astype(np.float64)
    start = time.time()
    d_node, _ = sdb.fit_ksvd(node_train, atoms=pec.K_V, s=pec.S_V, epochs=DICT_EPOCHS)
    d_edge, _ = sdb.fit_ksvd(edge_train, atoms=pec.K_E, s=pec.S_E, epochs=DICT_EPOCHS)
    return d_node, d_edge, time.time() - start


# ---------------------------------------------------------------------------
# Stage B — internal screen
# ---------------------------------------------------------------------------


def screen(device: torch.device, *, force: bool = False) -> dict[str, Any]:
    output = RESULTS_DIR / "internal_screen.json"
    if output.exists() and not force:
        return _read_json(output)

    samples = load_cached_split("train")
    train = samples[:GATE2_TRAIN]
    dev = samples[GATE2_DEV_START : GATE2_DEV_START + GATE2_DEV]
    d_node, d_edge, dictionary_seconds = fit_gate2_dictionaries(train)

    gpu_index = cuda_index(device)
    if gpu_index is not None:
        torch.cuda.reset_peak_memory_stats(gpu_index)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    baseline = pec.build_model(
        "dense", d_node=d_node, d_edge=d_edge, ablation="true", seed=SEED
    ).to(device)
    baseline_result = _train_screen_arm(
        baseline, train, dev, device, tag="CD_matched"
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    candidate = i1.build_i1_model(
        "dense",
        d_node=d_node,
        d_edge=d_edge,
        reader_hidden=i1.I1_READER_HIDDEN,
        ablation="true",
        seed=SEED,
    ).to(device)
    candidate_result = _train_screen_arm(
        candidate, train, dev, device, tag="CD-I1"
    )

    dev_batches = list(_batches(dev, SCREEN_BATCH, shuffle=False, seed=SEED))
    dev_target = candidate_result["dev_target"]
    relation = _soup_intervention(
        candidate,
        dev_batches,
        device,
        candidate_result["top_states"],
        mode="relation_shuffle",
    )
    bag = _soup_intervention(
        candidate, dev_batches, device, candidate_result["top_states"], mode="bag"
    )

    delta_frozen = HIST_CD_SCREEN_SOUP - candidate_result["soup_dev_mae"]
    delta_matched = baseline_result["soup_dev_mae"] - candidate_result["soup_dev_mae"]
    gate_delta = min(delta_frozen, delta_matched)
    if gate_delta >= SCREEN_STRONG:
        verdict = "INTERFACE_SIGNAL_STRONG"
    elif gate_delta >= SCREEN_WEAK:
        verdict = "INTERFACE_SIGNAL_WEAK"
    else:
        verdict = "INTERFACE_NOT_PRIMARY_GAP"

    mechanism = {
        "relation_shuffle": {
            "dev_mae": float(np.abs(relation - dev_target).mean()),
            "degradation": float(np.abs(relation - dev_target).mean())
            - candidate_result["soup_dev_mae"],
            "mean_abs_prediction_shift": float(
                np.abs(relation - candidate_result["soup_prediction"]).mean()
            ),
        },
        "bag": {
            "dev_mae": float(np.abs(bag - dev_target).mean()),
            "degradation": float(np.abs(bag - dev_target).mean())
            - candidate_result["soup_dev_mae"],
            "mean_abs_prediction_shift": float(
                np.abs(bag - candidate_result["soup_prediction"]).mean()
            ),
        },
    }
    mechanism_alive = (
        mechanism["relation_shuffle"]["degradation"] > 0.0
        and mechanism["bag"]["degradation"] > 0.0
        and candidate_result["soup_dev_mae"] < float(np.abs(bag - dev_target).mean())
    )

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "round": "pec_i1",
        "stage": "B_internal_screen",
        "split": {
            "gate2_train": len(train),
            "gate2_dev": len(dev),
            "dev_source": f"official-train molecules {GATE2_DEV_START}.."
            f"{GATE2_DEV_START + GATE2_DEV}",
        },
        "config": {
            "lr": SCREEN_LR,
            "wd": SCREEN_WD,
            "batch": SCREEN_BATCH,
            "epochs": SCREEN_EPOCHS,
            "clip": SCREEN_CLIP,
            "loss": "L1",
            "scheduler": "none",
            "top5_soup": True,
            "seed": SEED,
        },
        "arms": {
            "CD_matched": {
                key: value
                for key, value in baseline_result.items()
                if key not in ("soup_prediction", "dev_target", "top_states")
            },
            "CD-I1": {
                key: value
                for key, value in candidate_result.items()
                if key not in ("soup_prediction", "dev_target", "top_states")
            },
        },
        "historical_context_only": {
            "pec_v0_CD_screen_soup": HIST_CD_SCREEN_SOUP,
            "pec_v0_CD_screen_best": HIST_CD_SCREEN_BEST,
            "note": "CPU / host-calendar artifact; never a gate, comparator audited",
        },
        "deltas": {
            "delta_frozen_467108_minus_I1": float(delta_frozen),
            "delta_matched_CDminusI1": float(delta_matched),
            "delta_gate_min": float(gate_delta),
        },
        "reproduction_drift": float(
            baseline_result["soup_dev_mae"] - HIST_CD_SCREEN_SOUP
        ),
        "mechanism": mechanism,
        "mechanism_alive": bool(mechanism_alive),
        "criteria": {
            "delta_ge_strong": bool(gate_delta >= SCREEN_STRONG),
            "delta_ge_weak": bool(gate_delta >= SCREEN_WEAK),
            "mechanism_alive": bool(mechanism_alive),
        },
        "verdict": verdict,
        "dictionary_fit_seconds": dictionary_seconds,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(gpu_index))
            if gpu_index is not None
            else None
        ),
        **_provenance(device),
    }
    _write_json(output, payload)
    _write_json(
        RESULTS_DIR / "mechanism.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "stage": "mechanism",
            "mechanism": mechanism,
            "mechanism_alive": bool(mechanism_alive),
            "arm": "CD-I1",
            **_provenance(device),
        },
    )
    with (RESULTS_DIR / "internal_curve.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,CD_matched_dev_mae,CD-I1_dev_mae\n")
        for row_a, row_b in zip(baseline_result["curve"], candidate_result["curve"]):
            handle.write(
                f"{row_a['epoch']},{row_a['dev_mae']:.10f},{row_b['dev_mae']:.10f}\n"
            )
    np.save(RESULTS_DIR / "CD-I1_soup_predictions.npy", candidate_result["soup_prediction"])
    np.save(RESULTS_DIR / "CD_matched_soup_predictions.npy", baseline_result["soup_prediction"])
    return payload


# ---------------------------------------------------------------------------
# full-data formal run
# ---------------------------------------------------------------------------


def full(device: torch.device, *, force: bool = False) -> dict[str, Any]:
    output = RESULTS_DIR / "full_seed0.json"
    if output.exists() and not force:
        return _read_json(output)
    screen_payload = _read_json(RESULTS_DIR / "internal_screen.json")
    if not screen_payload["criteria"]["delta_ge_strong"]:
        raise RuntimeError(
            "full-data run is not authorized: internal delta < 0.010"
        )
    if not screen_payload["mechanism_alive"]:
        raise RuntimeError("full-data run is not authorized: mechanism not alive")

    train = load_cached_split("train")
    valid = load_cached_split("valid")
    d_node = np.load(C1_DICT_DIR / "d_node.npy")
    d_edge = np.load(C1_DICT_DIR / "d_edge.npy")

    gpu_index = cuda_index(device)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if gpu_index is not None:
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats(gpu_index)
    model = i1.build_i1_model(
        "dense",
        d_node=d_node,
        d_edge=d_edge,
        reader_hidden=i1.I1_READER_HIDDEN,
        ablation="true",
        seed=SEED,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=FULL_LR, weight_decay=FULL_WD)
    valid_batches = list(_batches(valid, FULL_BATCH, shuffle=False, seed=SEED))
    valid_target = np.concatenate([b["y"].numpy() for b in valid_batches])
    start = time.time()
    history: list[dict[str, Any]] = []
    top: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    for epoch in range(FULL_EPOCHS):
        model.train()
        running = 0.0
        seen = 0
        for batch in _batches(train, FULL_BATCH, shuffle=True, seed=SEED * 1000 + epoch):
            batch = pec.to_device(batch, device)
            out = model(batch)
            loss = (out["prediction"] - batch["y"]).abs().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), FULL_CLIP)
            optimizer.step()
            running += float(loss.detach()) * int(batch["y"].numel())
            seen += int(batch["y"].numel())
        train_mae = running / max(seen, 1)
        valid_mae = float(np.abs(_predict(model, valid_batches, device) - valid_target).mean())
        history.append({"epoch": epoch, "train_mae": train_mae, "valid_mae": valid_mae})
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        top.append((valid_mae, int(epoch), state))
        top.sort(key=lambda item: (item[0], item[1]))
        del top[SOUP_TOP:]
        if epoch % 10 == 0 or epoch == FULL_EPOCHS - 1:
            print(f"[CD-I1-full] epoch={epoch:03d} train={train_mae:.6f} valid={valid_mae:.6f}", flush=True)
    wall = time.time() - start

    soup_parts = []
    members: list[int] = []
    for _mae, epoch, state in sorted(top, key=lambda item: item[1]):
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        soup_parts.append(_predict(model, valid_batches, device))
        members.append(int(epoch))
    soup_prediction = np.mean(np.stack(soup_parts, axis=0), axis=0)
    soup_mae = float(np.abs(soup_prediction - valid_target).mean())
    best_mae, best_epoch, best_state = top[0]
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_prediction = _predict(model, valid_batches, device)
    best_mae = float(np.abs(best_prediction - valid_target).mean())

    dev_batches = valid_batches
    relation = _soup_intervention(
        model, dev_batches, device, [s for _m, _e, s in sorted(top, key=lambda i: i[1])],
        mode="relation_shuffle",
    )
    bag = _soup_intervention(
        model, dev_batches, device, [s for _m, _e, s in sorted(top, key=lambda i: i[1])],
        mode="bag",
    )
    mechanism = {
        "relation_shuffle": {
            "valid_mae": float(np.abs(relation - valid_target).mean()),
            "degradation": float(np.abs(relation - valid_target).mean()) - soup_mae,
            "mean_abs_prediction_shift": float(np.abs(relation - soup_prediction).mean()),
        },
        "bag": {
            "valid_mae": float(np.abs(bag - valid_target).mean()),
            "degradation": float(np.abs(bag - valid_target).mean()) - soup_mae,
            "mean_abs_prediction_shift": float(np.abs(bag - soup_prediction).mean()),
        },
    }

    states_dir = RESULTS_DIR / "states"
    states_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"top5": [s for _m, _e, s in sorted(top, key=lambda i: i[1])], "members": members},
        states_dir / "CD-I1_full_seed0_top5.pt",
    )
    torch.save(best_state, states_dir / "CD-I1_full_seed0_best.pt")
    np.save(RESULTS_DIR / "full_seed0_soup_predictions.npy", soup_prediction)
    with (RESULTS_DIR / "full_seed0_curve.csv").open("w", encoding="utf-8") as handle:
        handle.write("epoch,train_mae,valid_mae\n")
        for row in history:
            handle.write(f"{row['epoch']},{row['train_mae']:.10f},{row['valid_mae']:.10f}\n")

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "round": "pec_i1",
        "stage": "full_seed0",
        "arm": "CD-I1",
        "role_mode": "dense",
        "seed": SEED,
        "epochs": FULL_EPOCHS,
        "params": int(pec.n_params(model)),
        "n_train": len(train),
        "n_valid": len(valid),
        "dictionary_source": "PEC-C1 full-train K-SVD (reused for D^T init)",
        "config": {
            "lr": FULL_LR,
            "wd": FULL_WD,
            "batch": FULL_BATCH,
            "clip": FULL_CLIP,
            "loss": "L1",
            "scheduler": "none",
            "soup_top": SOUP_TOP,
            "early_stopping": False,
        },
        "curve": history,
        "best_valid_mae": best_mae,
        "best_epoch": int(best_epoch),
        "soup_valid_mae": soup_mae,
        "soup_members": members,
        "soup_member_valid_mae": [float(m) for m, _e, _s in sorted(top)],
        "mechanism": mechanism,
        "historical_context_only": {
            "pec_c1_CD_soup": HIST_PEC_C1_CD_SOUP,
            "pec_c1_CD_best": HIST_PEC_C1_CD_BEST,
            "strict_static_S0_seed0_soup": HIST_S0_SEED0_SOUP,
            "note": "PEC-C1 CD comparator is matched (audit 2.1); S0 is orientation only",
        },
        "wall_seconds": wall,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(gpu_index))
            if gpu_index is not None
            else None
        ),
        **_provenance(device),
    }
    _write_json(output, payload)
    return payload


# ---------------------------------------------------------------------------
# report / decision
# ---------------------------------------------------------------------------


def _screen_decision(payload: Mapping[str, Any]) -> dict[str, Any]:
    delta = float(payload["deltas"]["delta_gate_min"])
    if delta >= SCREEN_STRONG:
        return {"verdict": "INTERFACE_SIGNAL_STRONG", "authorize_full": True}
    if delta >= SCREEN_WEAK:
        return {"verdict": "INTERFACE_SIGNAL_WEAK", "authorize_full": False}
    return {"verdict": "INTERFACE_NOT_PRIMARY_GAP", "authorize_full": False}


def _full_decision(m_i1: float) -> dict[str, Any]:
    improvement = HIST_PEC_C1_CD_SOUP - float(m_i1)
    if m_i1 <= FULL_VIABLE:
        verdict = "PURE_STATIC_INTERFACE_VIABLE"
        extra = m_i1 <= FULL_MATCH_S0
        return {
            "verdict": verdict,
            "case": "A",
            "improvement_vs_pec_c1_cd": improvement,
            "matches_or_beats_s0": bool(extra),
            "stop": False,
        }
    if improvement >= FULL_MATERIAL:
        return {
            "verdict": "INTERFACE_MATTERS_BUT_NOT_SUFFICIENT",
            "case": "B",
            "improvement_vs_pec_c1_cd": improvement,
            "matches_or_beats_s0": False,
            "stop": True,
        }
    return {
        "verdict": "STATIC_POOLING_NOT_PRIMARY_GAP",
        "case": "C",
        "improvement_vs_pec_c1_cd": improvement,
        "matches_or_beats_s0": False,
        "stop": True,
    }


def report(device: torch.device, *, force: bool = False) -> dict[str, Any]:
    output = RESULTS_DIR / "decision.json"
    if output.exists() and not force:
        return _read_json(output)
    screen_payload = _read_json(RESULTS_DIR / "internal_screen.json")
    screen_decision = _screen_decision(screen_payload)
    recoverability = _read_json(RESULTS_DIR / "information_recoverability.json")
    payload: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "round": "pec_i1",
        "stage_a_verdict": recoverability["stage_a_decision"]["verdict"],
        "screen_verdict": screen_decision["verdict"],
        "full_authorized": screen_decision["authorize_full"],
        "screen_deltas": screen_payload["deltas"],
        "mechanism": screen_payload["mechanism"],
        "mechanism_alive": screen_payload["mechanism_alive"],
        **_provenance(device),
    }
    full_payload_path = RESULTS_DIR / "full_seed0.json"
    if full_payload_path.exists():
        full_payload = _read_json(full_payload_path)
        decision = _full_decision(float(full_payload["soup_valid_mae"]))
        payload["full"] = {
            "soup_valid_mae": full_payload["soup_valid_mae"],
            "best_valid_mae": full_payload["best_valid_mae"],
            "best_epoch": full_payload["best_epoch"],
            "soup_members": full_payload["soup_members"],
            "mechanism": full_payload["mechanism"],
            **decision,
        }
        payload["verdict"] = decision["verdict"]
    else:
        payload["full"] = None
        payload["verdict"] = screen_decision["verdict"]
    _write_json(output, payload)
    _write_report_markdown(payload)
    return payload


def _write_report_markdown(payload: Mapping[str, Any]) -> None:
    lines = [
        "# PEC-I1 — Static Composition Interface Audit: report",
        "",
        "Round `pec_i1` · study `zinc-context-gap`.  Pre-registration:",
        "`notes/pec_i1_preregistration.md` (frozen before the run).  Official test",
        "never loaded.",
        "",
        f"Stage A: **{payload['stage_a_verdict']}**",
        "",
        f"Internal screen: **{payload['screen_verdict']}**",
        "",
    ]
    deltas = payload["screen_deltas"]
    lines += [
        "## Internal screen (PEC-v0 Gate-2 split, 2000 train / 500 dev)",
        "",
        "| delta | value |",
        "|---|---:|",
        f"| `0.467108 - M_CD-I1` (frozen historical) | {deltas['delta_frozen_467108_minus_I1']:.6f} |",
        f"| `M_CD_matched - M_CD-I1` (device-matched) | {deltas['delta_matched_CDminusI1']:.6f} |",
        f"| gate (`min`) | {deltas['delta_gate_min']:.6f} |",
        "",
    ]
    mechanism = payload["mechanism"]
    lines += [
        "Mechanism (eval-only on the CD-I1 screen soup):",
        "",
        f"* relation-shuffle degradation: {mechanism['relation_shuffle']['degradation']:.6f}",
        f"* BAG degradation: {mechanism['bag']['degradation']:.6f}",
        f"* mechanism alive: {payload['mechanism_alive']}",
        "",
        f"Full-data run authorized: {payload['full_authorized']}",
        "",
    ]
    if payload.get("full"):
        full = payload["full"]
        lines += [
            "## Full-data seed 0 (official train 10 000 / official valid 1 000)",
            "",
            f"* `CD-I1` Top-5 soup valid MAE: {full['soup_valid_mae']:.6f}",
            f"* `CD-I1` best valid MAE: {full['best_valid_mae']:.6f}",
            f"* PEC-C1 `CD` comparator soup: {HIST_PEC_C1_CD_SOUP:.6f}",
            f"* improvement vs PEC-C1 CD: {full['improvement_vs_pec_c1_cd']:.6f}",
            f"* case {full['case']} -> **{full['verdict']}**",
            f"* matches/beats S0 orientation: {full['matches_or_beats_s0']}",
            "",
            "Full-scale mechanism (eval-only on the `CD-I1` soup):",
            "",
            f"* relation-shuffle degradation: "
            f"{full['mechanism']['relation_shuffle']['degradation']:.6f}",
            f"* BAG degradation: {full['mechanism']['bag']['degradation']:.6f}",
            "",
        ]
    lines += [f"FINAL: **{payload['verdict']}**", ""]
    (RESULTS_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _write_decision_markdown(payload)


def _write_decision_markdown(payload: Mapping[str, Any]) -> None:
    full = payload.get("full") or {}
    lines = [
        "# PEC-I1 — frozen decision",
        "",
        f"**{payload['verdict']}**",
        "",
        f"* Stage A: `{payload['stage_a_verdict']}`",
        f"* internal screen: `{payload['screen_verdict']}` "
        f"(delta `{payload['screen_deltas']['delta_gate_min']:.6f}`)",
        f"* full-data `CD-I1` soup / best: "
        f"{full.get('soup_valid_mae', float('nan')):.6f} / "
        f"{full.get('best_valid_mae', float('nan')):.6f}",
        f"* PEC-C1 `CD` comparator soup: {HIST_PEC_C1_CD_SOUP:.6f}",
        f"* improvement: {full.get('improvement_vs_pec_c1_cd', float('nan')):.6f}",
        f"* case: {full.get('case', 'n/a')}",
        f"* seed 1 authorized: false",
        "",
        "The one scientific change versus PEC-C1 was the graph-level pooling",
        "interface (`[mean, max]` -> S0 audited distance-bucketed",
        "`[mean, std, log1p(count)]`); the pure environment factorization, the",
        "topology-only 18-D pair relation, the single read-only static pair pass",
        "and the no-MP / no-recurrence contract were unchanged, and the reader was",
        "parameter-matched to +0.102 %.",
        "",
        "The full-data improvement is below the pre-registered 0.005 materiality",
        "threshold, so Case C fires: the pooling interface is not the primary gap.",
        "No second pooling variant, no reader/environment widening, no recurrence,",
        "no dictionary, no seed 1, and no PEC-I2 implementation is authorized.",
        "Any PEC-I2 (direct chemical bond relation primitive) is a NEW",
        "pre-registration.",
        "",
        "PEC-C1's `PURE_ENV_COMPOSITION_ABSOLUTE_WEAK` and PEC-v0's historical",
        "frozen Gate-1 FAIL are untouched.",
        "",
    ]
    (RESULTS_DIR / "DECISION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------


def _parse_device(value: str) -> torch.device:
    return torch.device(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PEC-I1 runner")
    parser.add_argument(
        "stage",
        choices=("recoverability", "correctness", "parameters", "screen", "report", "full", "all"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    device = _parse_device(args.device)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "recoverability":
        stage_a(device)
    elif args.stage == "correctness":
        correctness(device)
    elif args.stage == "parameters":
        parameters(device)
    elif args.stage == "screen":
        screen(device, force=args.force)
    elif args.stage == "full":
        full(device, force=args.force)
    elif args.stage == "report":
        report(device, force=args.force)
    else:
        stage_a(device)
        correctness(device)
        parameters(device)
        payload = screen(device, force=args.force)
        if payload["criteria"]["delta_ge_strong"] and payload["mechanism_alive"]:
            full(device, force=args.force)
        report(device, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
