"""PEC-v0 runner — Pure Environment Composition on ZINC.

Round: ``pec_v0``.  Pre-registration: ``notes/pec_v0_preregistration.md``
(+ Amendment A1).  Prior-artifact audit: ``notes/pec_v0_prior_artifact_audit.md``.
Official ZINC **test is never loaded**.  Gates 0-2 use only official-train
internal splits, so official **valid** is also untouched here.

Stages
------
``cache gate0 gate1 gate2 report all``

* ``cache`` — build the occurrence-level pure-topology role cache (official
  train / valid; the valid split is cached but not read by Gates 0-2).
* ``gate0`` — CPU correctness (data-free + real-batch).
* ``gate1`` — label-free K-SVD dictionary + coarse-role recoverability.
* ``gate2`` — cheap internal task screen (C0 / CD / CK + BAG / SHUFFLE).
* ``report`` — aggregate the durable Markdown/JSON decision.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import pec_v0 as pec
from tracks.ksvd.experiments.luyin16 import pec_v0_gate0 as gate0
from tracks.ksvd.experiments.luyin16 import sdb_v0 as sdb
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    _data_to_graph,
    _load_zinc,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
TRACK_ROOT = REPO_ROOT / "tracks/ksvd"
RESULTS_DIR = TRACK_ROOT / "results/pec_v0"
CACHE_DIR = RESULTS_DIR / "cache"
ZINC_ROOT = REPO_ROOT / "data/ZINC"

PROTOCOL_VERSION = "pec_v0"
CACHE_SCHEMA = "pec_v0_occurrence_roles_v1"

FIT_MOLECULES = 8000
MONITOR_MOLECULES = 2000
GATE2_TRAIN = 2000
GATE2_DEV = 500
SEED = 0

KSVD_EPOCHS = 10
PROBE_MAX_ROWS = 50000
PROBE_SEED = 20260925

# Gate-1 frozen thresholds (pre-registration §6)
GATE1_E_REC_PASS = 0.25
GATE1_RANDOM_RATIO = 0.10
GATE1_USED_PASS = 12
GATE1_PROBE_MARGIN = 0.15

# Gate-2 frozen protocol (pre-registration §7)
GATE2_LR = 1.0e-3
GATE2_WD = 1.0e-5
GATE2_BATCH = 64
GATE2_EPOCHS = 60
GATE2_CLIP = 5.0
GATE2_DELTA = 0.003
GATE2_SHUFFLE_DEGRADE = 0.02
GATE2_DICT_SLACK = 0.002


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def run_gate0() -> dict[str, Any]:
    return gate0.run_gate0()


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


def _extract_split(split: str) -> list[pec.MoleculeSample]:
    dataset = _load_zinc(ZINC_ROOT, split)
    samples: list[pec.MoleculeSample] = []
    for data in dataset:
        graph, node_types, edge_types = _data_to_graph(data)
        samples.append(
            pec.build_sample(
                graph,
                node_types,
                edge_types,
                y=float(data.y.view(-1)[0]),
            )
        )
    return samples


def _cache_paths() -> tuple[Path, Path, Path]:
    return (
        CACHE_DIR / "train.pkl.gz",
        CACHE_DIR / "valid.pkl.gz",
        CACHE_DIR / "cache_meta.json",
    )


def _split_fingerprint(samples: Sequence[pec.MoleculeSample]) -> str:
    digest = hashlib.sha256()
    digest.update(str(len(samples)).encode())
    for sample in samples:
        digest.update(
            f"{sample.n_nodes}:{sample.n_edges}:{sample.y:.10f}|".encode()
        )
    return digest.hexdigest()


def build_datasets(force: bool = False) -> dict[str, Any]:
    train_path, valid_path, meta_path = _cache_paths()
    if not force and meta_path.exists() and train_path.exists() and valid_path.exists():
        meta = _read_json(meta_path)
        if meta.get("cache_schema") == CACHE_SCHEMA:
            return meta

    processed = ZINC_ROOT / "subset" / "processed"
    for split in ("train", "val"):
        if not (processed / f"{split}.pt").exists():
            raise RuntimeError(f"processed ZINC split missing: {processed / f'{split}.pt'}")
    if (processed / "test.pt").exists():
        # recorded only: the test split exists on disk but is never opened
        pass

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()
    train = _extract_split("train")
    valid = _extract_split("val")
    for path, samples in ((train_path, train), (valid_path, valid)):
        with gzip.open(path, "wb") as handle:
            pickle.dump(samples, handle, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {
        "cache_schema": CACHE_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "n_train": len(train),
        "n_valid": len(valid),
        "patch_radius": pec.PATCH_RADIUS,
        "node_role_dim": pec.NODE_ROLE_DIM,
        "edge_role_dim": pec.EDGE_ROLE_DIM,
        "official_test_loaded": False,
        "train_fingerprint": _split_fingerprint(train),
        "valid_fingerprint": _split_fingerprint(valid),
        "seconds": time.time() - start,
    }
    _write_json(meta_path, meta)
    return meta


def load_datasets() -> list[pec.MoleculeSample]:
    train_path, _valid_path, _meta_path = _cache_paths()
    if not train_path.exists():
        build_datasets()
    with gzip.open(train_path, "rb") as handle:
        return pickle.load(handle)


# ---------------------------------------------------------------------------
# dictionary / coding helpers
# ---------------------------------------------------------------------------


def _iht_codes(D: np.ndarray, X: np.ndarray, s: int) -> np.ndarray:
    Dt = torch.as_tensor(np.asarray(D), dtype=torch.float64)
    Dt = torch.nn.functional.normalize(Dt, dim=0, eps=1.0e-8)
    Xt = torch.as_tensor(np.asarray(X), dtype=torch.float64)
    with torch.no_grad():
        codes = pec.T.iht_codes(Dt, Xt, s=int(s), steps=pec.IHT_STEPS)
    return codes.numpy()


def _relative_error(X: np.ndarray, reconstruction: np.ndarray) -> float:
    num = float(np.sum((X - reconstruction) ** 2))
    den = float(np.sum(X * X))
    return num / max(den, 1.0e-30)


def _dictionary_report(
    name: str,
    D: np.ndarray,
    X_fit: np.ndarray,
    X_monitor: np.ndarray,
    s: int,
) -> dict[str, Any]:
    D_norm = pec.T.normalize_columns(np.asarray(D, dtype=np.float64))
    codes_fit = _iht_codes(D_norm, X_fit, s)
    codes_monitor = _iht_codes(D_norm, X_monitor, s)
    rec_fit = codes_fit @ D_norm.T
    rec_monitor = codes_monitor @ D_norm.T
    health = sdb.dictionary_health(codes_monitor, atoms=int(D_norm.shape[1]))
    l0 = np.count_nonzero(codes_monitor, axis=1)
    abs_codes = np.abs(codes_monitor)
    usage = abs_codes.sum(axis=0)
    usage = usage / max(float(usage.sum()), 1.0e-30)
    positive = usage[usage > 0.0]
    effective = float(np.exp(-(positive * np.log(positive)).sum())) if positive.size else 0.0
    argmax_used = int(np.unique(codes_monitor.argmax(axis=1)).size)
    return {
        "name": name,
        "e_rec_fit": _relative_error(X_fit, rec_fit),
        "e_rec_monitor": _relative_error(X_monitor, rec_monitor),
        "used_atoms": int(health["used_atoms"]),
        "argmax_used": argmax_used,
        "dead_atoms": int(health["dead_atoms"]),
        "effective_atom_count": effective,
        "mean_assignment_entropy": float(health["support_entropy_mean"]),
        "normalized_entropy": float(health["support_entropy_normalized"]),
        "top1_mass": float(health["top1_mass_mean"]),
        "top8_mass": float(health["top8_mass_mean"]),
        "row_coverage": float(health["row_coverage"]),
        "max_l0": int(l0.max()) if l0.size else 0,
        "exact_sparsity": bool(np.all(l0 == int(s))),
    }


def _coarse_role_probe(
    codes_fit: np.ndarray,
    labels_fit: np.ndarray,
    codes_monitor: np.ndarray,
    labels_monitor: np.ndarray,
    *,
    n_classes: int,
    seed: int = PROBE_SEED,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score

    rng = np.random.RandomState(seed)
    if codes_fit.shape[0] > PROBE_MAX_ROWS:
        index = rng.choice(codes_fit.shape[0], size=PROBE_MAX_ROWS, replace=False)
        codes_fit = codes_fit[index]
        labels_fit = labels_fit[index]
    if codes_monitor.shape[0] > PROBE_MAX_ROWS:
        index = rng.choice(codes_monitor.shape[0], size=PROBE_MAX_ROWS, replace=False)
        codes_monitor = codes_monitor[index]
        labels_monitor = labels_monitor[index]

    majority = np.bincount(labels_fit, minlength=int(n_classes)).argmax()
    baseline_pred = np.full_like(labels_monitor, int(majority))
    baseline = float(f1_score(labels_monitor, baseline_pred, average="macro"))
    model = LogisticRegression(max_iter=200, multi_class="auto")
    model.fit(codes_fit, labels_fit)
    prediction = model.predict(codes_monitor)
    macro = float(f1_score(labels_monitor, prediction, average="macro"))
    accuracy = float((prediction == labels_monitor).mean())
    return {
        "n_fit": int(codes_fit.shape[0]),
        "n_monitor": int(codes_monitor.shape[0]),
        "majority_class": int(majority),
        "majority_macro_f1": baseline,
        "probe_macro_f1": macro,
        "probe_accuracy": accuracy,
        "margin": macro - baseline,
    }


def gate1(force: bool = False) -> dict[str, Any]:
    output = RESULTS_DIR / "gate1_label_free.json"
    if output.exists() and not force:
        return _read_json(output)

    meta = build_datasets()
    samples = load_datasets()
    fit = samples[:FIT_MOLECULES]
    monitor = samples[FIT_MOLECULES : FIT_MOLECULES + MONITOR_MOLECULES]
    if not fit or not monitor:
        raise RuntimeError("internal split is empty")

    node_fit = np.concatenate([s.node_basis for s in fit], axis=0).astype(np.float64)
    node_monitor = np.concatenate([s.node_basis for s in monitor], axis=0).astype(np.float64)
    edge_fit = np.concatenate([s.edge_basis for s in fit], axis=0).astype(np.float64)
    edge_monitor = np.concatenate([s.edge_basis for s in monitor], axis=0).astype(np.float64)

    start = time.time()
    d_node, meta_node = sdb.fit_ksvd(
        node_fit, atoms=pec.K_V, s=pec.S_V, epochs=KSVD_EPOCHS
    )
    d_edge, meta_edge = sdb.fit_ksvd(
        edge_fit, atoms=pec.K_E, s=pec.S_E, epochs=KSVD_EPOCHS
    )
    fit_seconds = time.time() - start

    d_node_random = sdb.random_normalized_dictionary(
        pec.NODE_ROLE_DIM, pec.K_V, seed=sdb.DICT_SEED
    )
    d_edge_random = sdb.random_normalized_dictionary(
        pec.EDGE_ROLE_DIM, pec.K_E, seed=sdb.DICT_SEED + 1
    )
    pca_node = sdb.fit_pca_rank(node_fit, rank=pec.K_V)
    pca_edge = sdb.fit_pca_rank(edge_fit, rank=pec.K_E)

    node_reports = [
        _dictionary_report("ksvd", d_node, node_fit, node_monitor, pec.S_V),
        _dictionary_report("random", d_node_random, node_fit, node_monitor, pec.S_V),
    ]
    edge_reports = [
        _dictionary_report("ksvd", d_edge, edge_fit, edge_monitor, pec.S_E),
        _dictionary_report("random", d_edge_random, edge_fit, edge_monitor, pec.S_E),
    ]
    node_reports[0]["e_rec_pca16"] = _relative_error(
        node_monitor, pca_node.reconstruct(pca_node.dense_codes(node_monitor))
    )
    node_reports[0]["e_rec_random_ratio"] = node_reports[0]["e_rec_monitor"] / max(
        node_reports[1]["e_rec_monitor"], 1.0e-30
    )
    edge_reports[0]["e_rec_pca16"] = _relative_error(
        edge_monitor, pca_edge.reconstruct(pca_edge.dense_codes(edge_monitor))
    )
    edge_reports[0]["e_rec_random_ratio"] = edge_reports[0]["e_rec_monitor"] / max(
        edge_reports[1]["e_rec_monitor"], 1.0e-30
    )

    # coarse-role recoverability (diagnostic, train-fit probe)
    codes_node_monitor = _iht_codes(pec.T.normalize_columns(d_node), node_monitor, pec.S_V)
    codes_edge_monitor = _iht_codes(pec.T.normalize_columns(d_edge), edge_monitor, pec.S_E)
    codes_node_fit = _iht_codes(pec.T.normalize_columns(d_node), node_fit, pec.S_V)
    codes_edge_fit = _iht_codes(pec.T.normalize_columns(d_edge), edge_fit, pec.S_E)
    labels_node_fit = np.concatenate([s.occ_shell for s in fit])
    labels_node_monitor = np.concatenate([s.occ_shell for s in monitor])
    labels_edge_fit = np.concatenate([s.eocc_shellpair for s in fit])
    labels_edge_monitor = np.concatenate([s.eocc_shellpair for s in monitor])
    probe_node = _coarse_role_probe(
        codes_node_fit,
        labels_node_fit,
        codes_node_monitor,
        labels_node_monitor,
        n_classes=pec.SHELL_CLASSES,
    )
    probe_edge = _coarse_role_probe(
        codes_edge_fit,
        labels_edge_fit,
        codes_edge_monitor,
        labels_edge_monitor,
        n_classes=pec.SHELLPAIR_CLASSES,
    )

    criteria = {
        "node_e_rec": node_reports[0]["e_rec_monitor"] <= GATE1_E_REC_PASS,
        "edge_e_rec": edge_reports[0]["e_rec_monitor"] <= GATE1_E_REC_PASS,
        "node_random_ratio": node_reports[0]["e_rec_random_ratio"] <= GATE1_RANDOM_RATIO,
        "edge_random_ratio": edge_reports[0]["e_rec_random_ratio"] <= GATE1_RANDOM_RATIO,
        "node_used": node_reports[0]["used_atoms"] >= GATE1_USED_PASS,
        "edge_used": edge_reports[0]["used_atoms"] >= GATE1_USED_PASS,
        "node_exact_sparsity": node_reports[0]["exact_sparsity"],
        "edge_exact_sparsity": edge_reports[0]["exact_sparsity"],
        "node_probe": probe_node["margin"] >= GATE1_PROBE_MARGIN,
        "edge_probe": probe_edge["margin"] >= GATE1_PROBE_MARGIN,
    }
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "official_valid_read": False,
        "split": {
            "fit_molecules": len(fit),
            "monitor_molecules": len(monitor),
            "train_fingerprint": meta.get("train_fingerprint"),
        },
        "config": {
            "K_V": pec.K_V,
            "S_V": pec.S_V,
            "K_E": pec.K_E,
            "S_E": pec.S_E,
            "iht_steps": pec.IHT_STEPS,
            "ksvd_epochs": KSVD_EPOCHS,
        },
        "node": node_reports,
        "edge": edge_reports,
        "probe_node_shell": probe_node,
        "probe_edge_shellpair": probe_edge,
        "criteria": {key: bool(value) for key, value in criteria.items()},
        "verdict": "PASS" if all(criteria.values()) else "FAIL",
        "ksvd_meta": {"node": meta_node, "edge": meta_edge},
        "seconds": fit_seconds,
    }
    _write_json(output, payload)
    np.save(RESULTS_DIR / "d_node.npy", d_node)
    np.save(RESULTS_DIR / "d_edge.npy", d_edge)
    _write_gate1_markdown(payload)
    return payload


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _write_gate1_markdown(payload: Mapping[str, Any]) -> None:
    lines = [
        "# PEC-v0 — Gate 1 (label-free dictionary + coarse-role recoverability)",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        f"Split: fit {payload['split']['fit_molecules']} / monitor "
        f"{payload['split']['monitor_molecules']} official-train molecules. "
        "Official valid and test not read.",
        "",
        "| role | E_rec monitor (K-SVD) | E_rec fit | E_rec random | ratio | PCA16 | used atoms | max l0 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("node", "edge"):
        ksvd, random = payload[role][0], payload[role][1]
        lines.append(
            f"| {role} | {_fmt(ksvd['e_rec_monitor'])} | {_fmt(ksvd['e_rec_fit'])} | "
            f"{_fmt(random['e_rec_monitor'])} | {_fmt(ksvd['e_rec_random_ratio'])} | "
            f"{_fmt(ksvd.get('e_rec_pca16', float('nan')))} | {ksvd['used_atoms']} | "
            f"{ksvd['max_l0']} |"
        )
    lines += [
        "",
        "| probe | macro-F1 | majority macro-F1 | margin | accuracy |",
        "|---|---:|---:|---:|---:|",
        f"| alpha^V -> shell | {_fmt(payload['probe_node_shell']['probe_macro_f1'])} | "
        f"{_fmt(payload['probe_node_shell']['majority_macro_f1'])} | "
        f"{_fmt(payload['probe_node_shell']['margin'])} | "
        f"{_fmt(payload['probe_node_shell']['probe_accuracy'])} |",
        f"| alpha^E -> shellpair | {_fmt(payload['probe_edge_shellpair']['probe_macro_f1'])} | "
        f"{_fmt(payload['probe_edge_shellpair']['majority_macro_f1'])} | "
        f"{_fmt(payload['probe_edge_shellpair']['margin'])} | "
        f"{_fmt(payload['probe_edge_shellpair']['probe_accuracy'])} |",
        "",
        "Frozen criteria:",
        "",
    ]
    for key, value in payload["criteria"].items():
        lines.append(f"* `{key}`: {value}")
    lines.append("")
    lines.append(f"`official_test_loaded = {payload['official_test_loaded']}`")
    lines.append("")
    (RESULTS_DIR / "GATE1_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Gate 2 — cheap internal task screen
# ---------------------------------------------------------------------------


def _batches(
    samples: Sequence[pec.MoleculeSample],
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
):
    order = np.arange(len(samples))
    if shuffle:
        np.random.RandomState(int(seed)).shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        index = order[start : start + int(batch_size)]
        yield pec.collate([samples[int(i)] for i in index])


def _evaluate(model: pec.PECModel, batches, device: torch.device) -> tuple[float, np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            batch = pec.to_device(batch, device)
            out = model(batch)
            predictions.append(out["prediction"].detach().cpu().numpy())
            targets.append(batch["y"].detach().cpu().numpy())
    prediction = np.concatenate(predictions) if predictions else np.zeros(0)
    target = np.concatenate(targets) if targets else np.zeros(0)
    if prediction.size == 0:
        return float("nan"), prediction
    return float(np.abs(prediction - target).mean()), prediction


def _predict_with(model: pec.PECModel, batches, device: torch.device) -> np.ndarray:
    model.eval()
    out_parts: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batches:
            batch = pec.to_device(batch, device)
            out_parts.append(model(batch)["prediction"].detach().cpu().numpy())
    return np.concatenate(out_parts) if out_parts else np.zeros(0)


def _chem_shuffle_batch(
    batch: Mapping[str, torch.Tensor], generator: torch.Generator
) -> dict[str, torch.Tensor]:
    out = dict(batch)
    for key, graph_key in (("atom_idx", "node_graph"), ("bond_idx", "edge_graph")):
        values = batch[key].clone()
        graph = batch[graph_key]
        for graph_id in torch.unique(graph).tolist():
            positions = torch.nonzero(graph == graph_id, as_tuple=False).reshape(-1)
            if positions.numel() <= 1:
                continue
            order = torch.randperm(positions.numel(), generator=generator)
            values[positions] = batch[key][positions[order]]
        out[key] = values
    return out


def _train_arm(
    role_mode: str,
    *,
    train: Sequence[pec.MoleculeSample],
    dev: Sequence[pec.MoleculeSample],
    d_node: np.ndarray | None,
    d_edge: np.ndarray | None,
    ablation: str = "true",
    seed: int = SEED,
    device: torch.device | None = None,
) -> dict[str, Any]:
    device = device or torch.device("cpu")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = pec.build_model(
        role_mode,
        d_node=d_node,
        d_edge=d_edge,
        ablation=ablation,
        seed=seed,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=GATE2_LR, weight_decay=GATE2_WD)
    dev_batches = list(_batches(dev, GATE2_BATCH, shuffle=False, seed=seed))
    history: list[dict[str, Any]] = []
    top: list[tuple[float, dict[str, torch.Tensor]]] = []
    for epoch in range(GATE2_EPOCHS):
        model.train()
        for batch in _batches(train, GATE2_BATCH, shuffle=True, seed=seed * 1000 + epoch):
            batch = pec.to_device(batch, device)
            out = model(batch)
            loss = (out["prediction"] - batch["y"]).abs().mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GATE2_CLIP)
            optimizer.step()
        dev_mae, _ = _evaluate(model, dev_batches, device)
        history.append({"epoch": epoch, "dev_mae": dev_mae})
        state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        top.append((dev_mae, state))
        top = sorted(top, key=lambda item: item[0])[:5]

    # fixed equal-weight Top-5 epoch-checkpoint soup
    soup_parts: list[np.ndarray] = []
    for _mae, state in top:
        model.load_state_dict(state)
        soup_parts.append(_predict_with(model, dev_batches, device))
    soup_prediction = np.mean(np.stack(soup_parts, axis=0), axis=0)
    dev_target = np.concatenate([b["y"].numpy() for b in dev_batches])
    soup_mae = float(np.abs(soup_prediction - dev_target).mean())

    best = dict(top[0][1])
    model.load_state_dict(best)
    best_prediction = _predict_with(model, dev_batches, device)
    best_mae = float(np.abs(best_prediction - dev_target).mean())

    # mechanism: chemistry placement shuffle under the selected checkpoint
    generator = torch.Generator().manual_seed(seed)
    shuffled_parts: list[np.ndarray] = []
    neutral_parts: list[np.ndarray] = []
    neutral_mae: float | None = None
    with torch.no_grad():
        for batch in dev_batches:
            shuffled = pec.to_device(_chem_shuffle_batch(batch, generator), device)
            shuffled_parts.append(model(shuffled)["prediction"].cpu().numpy())
    shuffled_mae = float(
        np.abs(np.concatenate(shuffled_parts) - dev_target).mean()
    )

    if role_mode == "sparse":
        import copy

        rng = np.random.RandomState(seed + 777)
        neutral = copy.deepcopy(model)
        with torch.no_grad():
            neutral.d_node.copy_(
                torch.as_tensor(
                    pec.T.normalize_columns(rng.randn(pec.NODE_ROLE_DIM, pec.K_V)),
                    dtype=torch.float32,
                )
            )
            neutral.d_edge.copy_(
                torch.as_tensor(
                    pec.T.normalize_columns(rng.randn(pec.EDGE_ROLE_DIM, pec.K_E)),
                    dtype=torch.float32,
                )
            )
        neutral_parts = [_predict_with(neutral, dev_batches, device)]
        neutral_mae = float(np.abs(neutral_parts[0] - dev_target).mean())

    return {
        "role_mode": role_mode,
        "ablation": ablation,
        "params": pec.n_params(model),
        "best_dev_mae": best_mae,
        "soup_dev_mae": soup_mae,
        "best_epoch": int(min(history, key=lambda row: row["dev_mae"])["epoch"]),
        "top5_epochs": [
            int(min(history, key=lambda row: abs(row["dev_mae"] - mae))["epoch"])
            for mae, _state in top
        ],
        "curve": history,
        "chem_shuffle_dev_mae": shuffled_mae,
        "chem_shuffle_degradation": shuffled_mae - best_mae,
        "neutral_dictionary_dev_mae": (
            neutral_mae if role_mode == "sparse" else None
        ),
        "neutral_dictionary_prediction_shift": (
            float(np.abs(neutral_parts[0] - best_prediction).mean())
            if role_mode == "sparse"
            else None
        ),
    }


def gate2(force: bool = False) -> dict[str, Any]:
    output = RESULTS_DIR / "gate2_small_train.json"
    if output.exists() and not force:
        return _read_json(output)

    gate1_payload = gate1()
    gate1_criteria = gate1_payload.get("criteria", {})
    # Amendment A2 (notes/pec_v0_gate1_decision.md): the user-specified Gate-1
    # STOP condition is coarse rooted structural role recoverability. The frozen
    # numeric FAILs are recorded verbatim and are NOT rewritten.
    coarse_role_ok = bool(
        gate1_criteria.get("node_probe") and gate1_criteria.get("edge_probe")
    )
    if not coarse_role_ok:
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "git_commit": _git_commit(),
            "official_test_loaded": False,
            "verdict": "GATE1_FAIL",
            "gate1_frozen_verdict": gate1_payload.get("verdict"),
            "stop_reason": (
                "Gate 1 failed the coarse-role recoverability STOP condition; "
                "Gate 2 not run (no K/s rescue)."
            ),
        }
        _write_json(output, payload)
        return payload

    samples = load_datasets()
    train = samples[:GATE2_TRAIN]
    dev = samples[FIT_MOLECULES : FIT_MOLECULES + GATE2_DEV]

    # Gate-2 dictionaries are fit only on the Gate-2 train molecules.
    node_train = np.concatenate([s.node_basis for s in train], axis=0).astype(np.float64)
    edge_train = np.concatenate([s.edge_basis for s in train], axis=0).astype(np.float64)
    start = time.time()
    d_node, _ = sdb.fit_ksvd(node_train, atoms=pec.K_V, s=pec.S_V, epochs=KSVD_EPOCHS)
    d_edge, _ = sdb.fit_ksvd(edge_train, atoms=pec.K_E, s=pec.S_E, epochs=KSVD_EPOCHS)
    dict_seconds = time.time() - start

    device = torch.device("cpu")
    arms = {
        "C0_coarse": _train_arm(
            "coarse", train=train, dev=dev, d_node=None, d_edge=None
        ),
        "CD_dense": _train_arm(
            "dense", train=train, dev=dev, d_node=d_node, d_edge=d_edge
        ),
        "CK_sparse": _train_arm(
            "sparse", train=train, dev=dev, d_node=d_node, d_edge=d_edge
        ),
    }
    controls = {
        "CK_bag": _train_arm(
            "sparse", train=train, dev=dev, d_node=d_node, d_edge=d_edge, ablation="bag"
        ),
        "CK_shuffle": _train_arm(
            "sparse",
            train=train,
            dev=dev,
            d_node=d_node,
            d_edge=d_edge,
            ablation="shuffle",
        ),
    }

    ck = arms["CK_sparse"]
    cd = arms["CD_dense"]
    c0 = arms["C0_coarse"]
    bag = controls["CK_bag"]
    shuffle = controls["CK_shuffle"]

    criteria = {
        "chem_shuffle_material": ck["chem_shuffle_degradation"] >= GATE2_SHUFFLE_DEGRADE,
        "true_beats_bag": (bag["soup_dev_mae"] - ck["soup_dev_mae"]) >= GATE2_DELTA,
        "true_beats_shuffle": (
            shuffle["soup_dev_mae"] - ck["soup_dev_mae"]
        ) >= GATE2_DELTA,
        "dict_not_worse_than_dense": (ck["soup_dev_mae"] - cd["soup_dev_mae"]) <= GATE2_DICT_SLACK,
        "dictionary_alive": (
            ck.get("neutral_dictionary_prediction_shift") or 0.0
        ) > 0.0,
    }

    composition_supported = criteria["true_beats_bag"] and criteria["true_beats_shuffle"]
    environment_supported = criteria["chem_shuffle_material"]
    dictionary_supported = criteria["dict_not_worse_than_dense"]

    if not environment_supported or not composition_supported:
        verdict = "ENVIRONMENT_SUPPORTED_COMPOSITION_NULL"
        if not environment_supported and not composition_supported:
            verdict = "PURE_ENV_COMPOSITION_NOT_VIABLE"
    elif not dictionary_supported:
        verdict = "ENV_COMPOSITION_SUPPORTED_DENSE_NOT_DICT"
    else:
        verdict = "GATE2_PASS_BUY_SEED0"

    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "git_commit": _git_commit(),
        "official_test_loaded": False,
        "official_valid_read": False,
        "gate1_frozen_verdict": gate1_payload.get("verdict"),
        "gate1_frozen_criteria": gate1_criteria,
        "amendment_A2": (
            "Proceeded under notes/pec_v0_gate1_decision.md: the frozen Gate-1 "
            "numeric verdict is FAIL (node/edge random-ratio 0.1115/0.1024 > "
            "0.10, edge_used 11 < 12) and is reported verbatim; the "
            "user-specified coarse-role STOP condition passed (probe macro-F1 "
            "1.0/1.0). No rescue: K/s/epochs/thresholds/architecture unchanged."
        ),
        "split": {
            "gate2_train": len(train),
            "gate2_dev": len(dev),
            "dev_source": f"official-train molecules {FIT_MOLECULES}..{FIT_MOLECULES + GATE2_DEV}",
        },
        "config": {
            "lr": GATE2_LR,
            "wd": GATE2_WD,
            "batch": GATE2_BATCH,
            "epochs": GATE2_EPOCHS,
            "clip": GATE2_CLIP,
            "seed": SEED,
            "top5_soup": True,
        },
        "arms": arms,
        "controls": controls,
        "criteria": {key: bool(value) for key, value in criteria.items()},
        "verdict": verdict,
        "dictionary_fit_seconds": dict_seconds,
        "seconds": time.time() - start,
        "notes": {
            "absolute_mae_not_comparable_to_s0": (
                "Gate 2 trains on 2000 molecules; the strict-static S0 band used "
                "10000. Gate 2 is a relative screen only."
            ),
            "c0_vs_ck_params": {
                "c0": c0["params"],
                "dense": cd["params"],
                "sparse": ck["params"],
            },
        },
    }
    _write_json(output, payload)
    _write_gate2_markdown(payload)
    return payload


def _write_gate2_markdown(payload: Mapping[str, Any]) -> None:
    lines = [
        "# PEC-v0 — Gate 2 (cheap internal task screen)",
        "",
        f"Verdict: **{payload['verdict']}**",
        "",
        f"Train {payload['split']['gate2_train']} / dev {payload['split']['gate2_dev']} "
        "official-train molecules (official valid/test not read).",
        "",
        "| arm | params | best dev MAE | Top-5 soup dev MAE | chem-shuffle MAE | "
        "chem-shuffle degradation | neutral-dict MAE |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("C0_coarse", "CD_dense", "CK_sparse"):
        row = payload["arms"][name]
        neutral = row.get("neutral_dictionary_dev_mae")
        lines.append(
            f"| {name} | {row['params']} | {row['best_dev_mae']:.6f} | "
            f"{row['soup_dev_mae']:.6f} | {row['chem_shuffle_dev_mae']:.6f} | "
            f"{row['chem_shuffle_degradation']:+.6f} | "
            f"{'—' if neutral is None else f'{neutral:.6f}'} |"
        )
    for name in ("CK_bag", "CK_shuffle"):
        row = payload["controls"][name]
        lines.append(
            f"| {name} | {row['params']} | {row['best_dev_mae']:.6f} | "
            f"{row['soup_dev_mae']:.6f} | — | — | — |"
        )
    lines += ["", "Frozen criteria:", ""]
    for key, value in payload["criteria"].items():
        lines.append(f"* `{key}`: {value}")
    lines.append("")
    lines.append("`official_test_loaded = false`")
    lines.append("")
    (RESULTS_DIR / "GATE2_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PEC-v0 runner")
    parser.add_argument(
        "stage",
        choices=("cache", "gate0", "gate1", "gate2", "report", "all"),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage == "cache":
        meta = build_datasets(force=args.force)
        print(json.dumps(meta, indent=2, sort_keys=True, default=str))
    elif args.stage == "gate0":
        payload = run_gate0()
        _write_json(RESULTS_DIR / "gate0.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif args.stage == "gate1":
        payload = gate1(force=args.force)
        print(json.dumps({k: payload[k] for k in ("verdict", "criteria")}, indent=2))
    elif args.stage == "gate2":
        payload = gate2(force=args.force)
        print(json.dumps({k: payload.get(k) for k in ("verdict", "criteria")}, indent=2))
    elif args.stage == "all":
        build_datasets(force=args.force)
        gate0_payload = run_gate0()
        _write_json(RESULTS_DIR / "gate0.json", gate0_payload)
        gate1(force=args.force)
        gate2(force=args.force)
    else:  # report
        summary = {
            "gate0": RESULTS_DIR.joinpath("gate0.json").exists(),
            "gate1": _read_json(RESULTS_DIR / "gate1_label_free.json")["verdict"]
            if (RESULTS_DIR / "gate1_label_free.json").exists()
            else None,
            "gate2": _read_json(RESULTS_DIR / "gate2_small_train.json")["verdict"]
            if (RESULTS_DIR / "gate2_small_train.json").exists()
            else None,
        }
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
