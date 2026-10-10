"""cscl-prediction-feasibility-v2 driver: prep / train / compare (seed 0 only).

Question (protocol: ``configs/prediction_feasibility_v2.yaml``, frozen)
-----------------------------------------------------------------------
Can the historical non-GNN Full-Y computation, refit strictly on the current
fit_inner 7200 rows with raw-y-only supervision, beat the frozen O-rich seed-0
reference (dev MAE 0.33206) on the same internal split?

Discipline
----------
* seed 0 only (argparse enforces ``choices=[0]``).
* At most ONE formal training run; ``--smoke`` runs are short engineering
  checks and are never used as tuning evidence.
* ``target_decomposition.npz`` / ``g`` / ``ell`` / ``s`` / ``c`` are never
  loaded; supervision is raw y only.  The auxiliary loss is the model's own
  label-free structural reconstruction with the frozen ``H1_LAMBDA``.
* Official valid/test are never loaded anywhere in this driver.
* The legacy all-10k prep blob is used ONLY as an affine inversion basis to
  recover raw features from the all-10k-standardized caches, after a
  fit==all identity check (tolerance 1e-5 on nondegenerate columns).  Every
  fitted object consumed by the model (patch/ctx/topo/anchor standardizers,
  K-SVD dictionary, common subspace) is refit on fit_inner 7200 rows only.

Modes
-----
``prep``      Build the fit7200 prep blob (identity checks + K-SVD + subspace).
``train``     prep + train + frozen selection + export (the formal run).
``compare``   Paired per-molecule comparison of an exported Full-Y npz against
              the frozen O-rich predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_CODE_DIR))

import cscl_features as cf  # noqa: E402
from tracks.ksvd.experiments.luyin16 import (  # noqa: E402
    e2e_dictenv_clean_mechanism_v1 as cm,
    e2e_dictenv_common_subspace_dictionary_v1 as cssd,
    e2e_dictenv_p1 as p1,
    e2e_dictenv_scale_v1 as sc,
    sdb_v0 as sdb,
    zinc_e2e_dictenv_p1 as p1run,
    zinc_joint_dictionary_decision_v1 as zjd,
    zinc_static_dictionary_pair as sdp,
)

PROTOCOL_ID = "cscl-prediction-feasibility-v2"
CANDIDATE = "Full-Y-fit7200-seed0"

ENCODED_TRAIN = sdp.CACHE_DIR / "encoded_train.pt"
LEGACY_PREP = (
    REPO_ROOT
    / "tracks/ksvd/results/zinc_full_decomposition_valid_test_confirmation_v1/all_train_prep.npz"
)
LEGACY_PREP_META = LEGACY_PREP.with_name("prep_meta.json")

# frozen protocol constants
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_K = 5
TRAIN_SHUFFLE_OFFSET = 101
H1_LAMBDA = float(cm.H1_LAMBDA)
FULL_PARAMETERS = 408_651
KSVD_ATOMS, KSVD_S, KSVD_EPOCHS, KSVD_SEED = sdb.K_ATOMS, sdb.SPARSITY, 10, sdb.DICT_SEED
IDENTITY_TOL = 1.0e-5
V1_ORICH_EXPORT = (
    REPO_ROOT
    / "tracks/structure-contribution/results/cscl_v1_gpu/dev_preds_raw_units.npz"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _j(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _j(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_j(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_j(payload), indent=2), encoding="utf-8")


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=np.float32).tobytes()).hexdigest()


def sha256_rows(rows: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


# ---------------------------------------------------------------------------
# data loading (train rows only) + row alignment
# ---------------------------------------------------------------------------


def load_train_rows(log=print) -> list[Any]:
    """Historical encoded official-train cache + train env (10k rows).

    ``encoded_valid.pt`` / official valid / official test are never touched.
    """
    train = list(torch.load(ENCODED_TRAIN, map_location="cpu", weights_only=False))
    p1run.attach_env(train, "train")
    assert len(train) == 10000, len(train)
    log(f"[data] loaded encoded train rows={len(train)}")
    return train


def verify_row_alignment(train: Sequence[Any], log=print) -> dict[str, Any]:
    """cscl row i must be the same molecule as encoded-cache row i.

    Evidence: raw y identical on all 10000 rows and atom counts identical.
    """
    dataset = cf.load_official_train()
    assert len(dataset) == len(train) == 10000
    max_dy = 0.0
    n_atom_mismatch = 0
    for i in range(10000):
        dy = abs(float(dataset[i].y.reshape(-1)[0]) - float(train[i].y.reshape(-1)[0]))
        max_dy = max(max_dy, dy)
        if int(dataset[i].x.shape[0]) != int(train[i].dict_phi.shape[0]):
            n_atom_mismatch += 1
    ok = bool(max_dy == 0.0 and n_atom_mismatch == 0)
    log(f"[align] y max|dy|={max_dy} atom_mismatches={n_atom_mismatch} ok={ok}")
    if not ok:
        raise RuntimeError("row alignment between cscl and encoded cache failed")
    return {"max_abs_y_diff": max_dy, "atom_count_mismatches": n_atom_mismatch, "ok": ok}


# ---------------------------------------------------------------------------
# prep (fit7200)
# ---------------------------------------------------------------------------


def _identity_gap(raw: np.ndarray, all_std: zjd.Std, degenerate: np.ndarray | None) -> dict[str, float]:
    refit = zjd.Std.fit(raw, floor=zjd.CANON_STD_FLOOR)
    mean_gap = np.abs(refit.mean - all_std.mean)
    scale_gap = np.abs(refit.scale - all_std.scale)
    overall = float(mean_gap.max() + scale_gap.max())
    if degenerate is None:
        return {"overall": overall, "nondegenerate": overall}
    return {"overall": overall, "nondegenerate": float(mean_gap[~degenerate].max() + scale_gap[~degenerate].max())}


def build_prep_fit7200(
    train: Sequence[Any],
    idx: Mapping[str, np.ndarray],
    out_dir: Path,
    smoke: bool = False,
    log=print,
) -> tuple[Path, dict[str, Any]]:
    """Build the fit7200 prep blob with leakage guards (see module docstring)."""
    started = time.perf_counter()
    fit_rows = np.asarray(idx["fit_inner"], dtype=np.int64)
    if smoke:
        fit_rows = fit_rows[:512]
    meta_legacy = json.loads(LEGACY_PREP_META.read_text())
    if "fit_idx = all 10,000" not in meta_legacy.get("algorithm", ""):
        raise RuntimeError("legacy prep provenance unexpected; refusing inversion")
    blob10 = {k: z for k, z in np.load(LEGACY_PREP, allow_pickle=False).items()}
    patch_all = zjd.Std(blob10["patch_all_mean"], blob10["patch_all_scale"])
    ctx_all = zjd.Std(blob10["ctx_all_mean"], blob10["ctx_all_scale"])
    anchor_all = zjd.Std(blob10["anchor_all_mean"], blob10["anchor_all_scale"])
    topo_all = zjd.Std(blob10["topo_all_mean"], blob10["topo_all_scale"])

    patch_st, _ = zjd._stack(train, "patch_cont")
    ctx_st, _ = zjd._stack(train, "global_context")
    anchor_st, _ = zjd._stack(train, "anchor")
    topo_st, _ = zjd._stack(train, "topology_features")

    # inversion-basis identity check (fit==all must reproduce the stored stats)
    deg_anchor = np.where(anchor_all.scale <= 1e-5)[0]
    checks = {
        "patch": _identity_gap(patch_all.inverse(patch_st), patch_all, None),
        "ctx": _identity_gap(ctx_all.inverse(ctx_st), ctx_all, None),
        "topo": _identity_gap(topo_all.inverse(topo_st), topo_all, None),
        "anchor": _identity_gap(anchor_all.inverse(anchor_st), anchor_all, deg_anchor),
    }
    for name, gap in checks.items():
        log(f"[prep] identity {name}: overall={gap['overall']:.3e} nondeg={gap['nondegenerate']:.3e}")
        if gap["nondegenerate"] > IDENTITY_TOL:
            raise RuntimeError(f"inversion identity check failed for {name}: {gap}")

    n_mols = len(train)
    mol_of_row = np.concatenate([[i] * int(train[i].dict_phi.shape[0]) for i in range(n_mols)])
    fit_mask_nodes = np.isin(mol_of_row, fit_rows)  # per-atom-root blocks (patch, anchor)
    fit_mask_mols = np.isin(np.arange(n_mols), fit_rows)  # per-molecule blocks (ctx, topo)

    patch_fit = zjd.Std.fit(patch_all.inverse(patch_st)[fit_mask_nodes], floor=zjd.CANON_STD_FLOOR)
    ctx_fit = zjd.Std.fit(ctx_all.inverse(ctx_st)[fit_mask_mols], floor=zjd.CANON_STD_FLOOR)
    anchor_fit = zjd.Std.fit(anchor_all.inverse(anchor_st)[fit_mask_nodes], floor=zjd.CANON_STD_FLOOR)
    topo_fit = zjd.Std.fit(topo_all.inverse(topo_st)[fit_mask_mols], floor=zjd.CANON_STD_FLOOR)
    for name, st in (("patch", patch_fit), ("ctx", ctx_fit), ("anchor", anchor_fit), ("topo", topo_fit)):
        log(f"[prep] fit7200 {name}: scale range [{st.scale.min():.3e}, {st.scale.max():.3e}]")

    phi_fit = np.concatenate([train[i].dict_phi.numpy() for i in fit_rows], 0).astype(np.float64)
    log(f"[prep] fitting D_fit (K-SVD) on {phi_fit.shape[0]} fit_inner phi rows ...")
    t0 = time.perf_counter()
    D_fit, _ksvd_info = sdb.fit_ksvd(phi_fit, atoms=KSVD_ATOMS, s=KSVD_S, epochs=KSVD_EPOCHS, seed=KSVD_SEED)
    log(f"[prep] K-SVD {time.perf_counter() - t0:.1f}s")
    subspace_fit = cssd.build_common_subspace(phi_fit, q=1, kind="q1")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "prep_fit7200.npz"
    keep = dict(
        fit_idx=fit_rows,
        monitor_idx=np.asarray(idx["monitor"], dtype=np.int64),
        dev_idx=np.asarray(idx["dev"], dtype=np.int64),
        patch_fit_mean=patch_fit.mean,
        patch_fit_scale=patch_fit.scale,
        ctx_fit_mean=ctx_fit.mean,
        ctx_fit_scale=ctx_fit.scale,
        anchor_fit_mean=anchor_fit.mean,
        anchor_fit_scale=anchor_fit.scale,
        topo_fit_mean=topo_fit.mean,
        topo_fit_scale=topo_fit.scale,
        D_fit=D_fit.astype(np.float32),
        U_components=subspace_fit.components.astype(np.float32),
        U_rms=subspace_fit.rms.astype(np.float32),
    )
    np.savez_compressed(out_path, **keep)
    meta = {
        "protocol": PROTOCOL_ID,
        "candidate": CANDIDATE,
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "legacy_prep": {
            "path": str(LEGACY_PREP.relative_to(REPO_ROOT)),
            "sha256": file_sha256(LEGACY_PREP),
            "provenance_algorithm": meta_legacy.get("algorithm"),
            "use": "inversion basis only (patch/ctx/topo/anchor *_all_*)",
        },
        "encoded_train_sha256": file_sha256(ENCODED_TRAIN),
        "identity_checks": checks,
        "identity_tol": IDENTITY_TOL,
        "fit_rows": {
            "n": int(len(fit_rows)),
            "sha256_rows": sha256_rows(fit_rows),
            "monitor_sha256_rows": sha256_rows(np.asarray(idx["monitor"], dtype=np.int64)),
            "dev_sha256_rows": sha256_rows(np.asarray(idx["dev"], dtype=np.int64)),
        },
        "ksvd": {"atoms": KSVD_ATOMS, "sparsity": KSVD_S, "epochs": KSVD_EPOCHS, "seed": KSVD_SEED, "phi_rows": int(phi_fit.shape[0])},
        "subspace": {"q": 1, "kind": "q1"},
        "sha256": {
            "D_fit": sha256_array(D_fit.astype(np.float32)),
            "U_components": sha256_array(subspace_fit.components.astype(np.float32)),
            "patch_fit_mean": sha256_array(patch_fit.mean),
            "ctx_fit_mean": sha256_array(ctx_fit.mean),
            "anchor_fit_mean": sha256_array(anchor_fit.mean),
            "topo_fit_mean": sha256_array(topo_fit.mean),
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
        "seconds": float(time.perf_counter() - started),
    }
    write_json(out_dir / "prep_fit7200_meta.json", meta)
    log(f"[prep] done {meta['seconds']:.1f}s D_fit_sha={meta['sha256']['D_fit'][:16]}")
    return out_path, meta


def apply_prep_fit7200(data_list: Sequence[Any], prep: Mapping[str, Any], blob10: Mapping[str, np.ndarray]) -> None:
    """z = fit.transform(all.inverse(cached)) per block, in place (affine)."""
    patch_all = zjd.Std(blob10["patch_all_mean"], blob10["patch_all_scale"])
    ctx_all = zjd.Std(blob10["ctx_all_mean"], blob10["ctx_all_scale"])
    anchor_all = zjd.Std(blob10["anchor_all_mean"], blob10["anchor_all_scale"])
    topo_all = zjd.Std(blob10["topo_all_mean"], blob10["topo_all_scale"])
    patch_fit = zjd.Std(prep["patch_fit_mean"], prep["patch_fit_scale"])
    ctx_fit = zjd.Std(prep["ctx_fit_mean"], prep["ctx_fit_scale"])
    anchor_fit = zjd.Std(prep["anchor_fit_mean"], prep["anchor_fit_scale"])
    topo_fit = zjd.Std(prep["topo_fit_mean"], prep["topo_fit_scale"])
    for attr, all_st, fit_st in (
        ("patch_cont", patch_all, patch_fit),
        ("global_context", ctx_all, ctx_fit),
        ("anchor", anchor_all, anchor_fit),
        ("topology_features", topo_all, topo_fit),
    ):
        values, counts = zjd._stack(data_list, attr)
        zjd._unstack(fit_st.transform(all_st.inverse(values)), counts, attr, data_list)


# ---------------------------------------------------------------------------
# model + training
# ---------------------------------------------------------------------------


def build_model(prep: Mapping[str, Any], seed: int = 0) -> nn.Module:
    model = sc.build_scale_model(
        np.asarray(prep["D_fit"], np.float32),
        int(seed),
        zjd.fold_subspace(prep),
        sc.FULL,
        scale_seed=0,
    )
    audit = sc.scale_parameter_audit(model)
    if int(audit["actual_parameters"]) != FULL_PARAMETERS or not audit["parameter_exact"]:
        raise RuntimeError(f"parameter audit failed: {audit}")
    return model


def _batches(n: int, batch_size: int, generator: torch.Generator, shuffle: bool) -> list[list[int]]:
    if shuffle:
        order = torch.randperm(int(n), generator=generator).tolist()
    else:
        order = list(range(int(n)))
    return [order[s : s + batch_size] for s in range(0, int(n), batch_size)]


def _batch(data_list: Sequence[Any], indices: Sequence[int], targets: torch.Tensor, device: torch.device):
    batch = p1.env_collate([data_list[i] for i in indices])
    batch.y = targets[torch.as_tensor(list(indices), dtype=torch.long)].view(-1).clone()
    return batch.to(device)


@torch.no_grad()
def predict_raw(model: nn.Module, data_list: Sequence[Any], device: torch.device) -> np.ndarray:
    """Raw-unit predictions, molecules in the order of ``data_list``."""
    model.eval()
    preds: list[torch.Tensor] = []
    for indices in _batches(len(data_list), BATCH_SIZE, torch.Generator().manual_seed(0), False):
        batch = _batch(data_list, indices, torch.zeros(len(data_list)), device)
        preds.append(model(batch, mask=cm.C6_MASK).view(-1).detach().cpu())
    return torch.cat(preds).numpy().astype(np.float64)


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def train_candidate(
    train: Sequence[Any],
    idx: Mapping[str, np.ndarray],
    prep_path: Path,
    *,
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
    epochs: int = EPOCHS,
    smoke: bool = False,
    log=print,
) -> dict[str, Any]:
    """Formal seed-0 training with the frozen selection rule."""
    prep = {k: z for k, z in np.load(prep_path, allow_pickle=False).items()}
    blob10 = {k: z for k, z in np.load(LEGACY_PREP, allow_pickle=False).items()}
    apply_prep_fit7200(train, prep, blob10)
    fit_rows = [int(i) for i in idx["fit_inner"]]
    mon_rows = [int(i) for i in idx["monitor"]]
    dev_rows = [int(i) for i in idx["dev"]]
    if smoke:
        fit_rows, mon_rows, dev_rows = fit_rows[:512], mon_rows[:128], dev_rows[:128]
        epochs = 3
    fit_data = [train[i] for i in fit_rows]
    mon_data = [train[i] for i in mon_rows]
    dev_data = [train[i] for i in dev_rows]
    y_all = np.array([float(d.y.reshape(-1)[0]) for d in train], dtype=np.float64)
    y_fit_np = y_all[fit_rows]
    y_fit = torch.as_tensor(y_fit_np, dtype=torch.float32)
    y_mon = y_all[mon_rows]
    y_dev = y_all[dev_rows]

    seed_everything(seed)
    model = build_model(prep, seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    train_gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)

    curve: list[dict[str, Any]] = []
    ckpts: list[tuple[float, int, dict[str, torch.Tensor]]] = []
    started = time.perf_counter()
    stopped_reason = "completed"
    for epoch in range(1, epochs + 1):
        model.train()
        task_sum, rec_sum, n_mol, n_steps = 0.0, 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in _batches(len(fit_data), BATCH_SIZE, train_gen, True):
            batch = _batch(fit_data, indices, y_fit, device)
            prediction, aux = model(batch, mask=cm.C6_MASK, return_aux=True)
            task = torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))
            rec = model.reconstruction_loss(aux["phi"], aux["coord"])
            loss = task + H1_LAMBDA * rec
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            task_sum += float((prediction.view(-1) - batch.y.view(-1)).abs().sum())
            rec_sum += float(rec.detach())
            n_mol += int(batch.y.numel())
            n_steps += 1
        mon_pred = predict_raw(model, mon_data, device)
        mon_mae = mae(mon_pred, y_mon)
        curve.append(
            {
                "epoch": epoch,
                "train_task_mae": task_sum / max(n_mol, 1),
                "train_rec": rec_sum / max(n_steps, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
            }
        )
        ckpts.append((mon_mae, epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}))
        ckpts.sort(key=lambda t: (t[0], t[1]))
        if len(ckpts) > SOUP_K:
            ckpts.pop()
        if not np.isfinite(mon_mae):
            stopped_reason = "nonfinite_monitor"
            break
        if log and (epoch == 1 or epoch % 20 == 0 or epoch == epochs):
            log(
                f"[{CANDIDATE}] ep={epoch:03d} train={curve[-1]['train_task_mae']:.6f} "
                f"mon={mon_mae:.6f} rec={curve[-1]['train_rec']:.4g} {curve[-1]['seconds']:.1f}s"
            )
    wall = time.perf_counter() - started

    def evaluate(state: Mapping[str, torch.Tensor]) -> dict[str, float]:
        model.load_state_dict({k: v.to(device) for k, v in state.items()})
        return {
            "fit_mae": mae(predict_raw(model, fit_data, device), y_fit_np),
            "monitor_mae": mae(predict_raw(model, mon_data, device), y_mon),
            "dev_mae": mae(predict_raw(model, dev_data, device), y_dev),
        }

    soup_members = sorted(e for _, e, _ in ckpts)
    soup_state = {k: torch.stack([s[k].float() for _, _, s in ckpts]).mean(0) for k in ckpts[0][2]}
    best_state = dict(ckpts[0][2])
    last_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    res_soup = evaluate(soup_state)
    model.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
    dev_pred_soup = predict_raw(model, dev_data, device)
    fit_pred_soup = predict_raw(model, fit_data, device)
    b_cal = float(np.median(y_fit_np - fit_pred_soup))
    res_best = evaluate(best_state)
    res_last = evaluate(last_state)

    results: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "candidate": CANDIDATE,
        "seed": int(seed),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "epochs_run": len(curve),
        "stopped_reason": stopped_reason,
        "soup_members": soup_members,
        "selection_rule": "primary=top-5 monitor soup; secondary=best-monitor; tertiary=last; auxiliary=+median-fit calibration",
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP,
            "batch_size": BATCH_SIZE, "epochs": epochs, "h1_lambda": H1_LAMBDA,
            "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
            "optimizer": "Adam",
            "supervision": "raw y L1 + label-free structural reconstruction (disclosed auxiliary)",
        },
        "curve": curve,
        "wall_seconds": wall,
        "device": str(device),
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "primary_soup": {**res_soup, "calibrated_dev_mae": mae(dev_pred_soup + b_cal, y_dev), "b_fit": b_cal},
        "secondary_best_monitor": res_best,
        "tertiary_last": res_last,
        "fit_only_discipline": {
            "standardizers": "fit_inner rows only",
            "D_fit": "fit_inner phi rows only",
            "U_subspace": "fit_inner phi rows only",
            "calibration_b": "fit_inner y (post-hoc, auxiliary)",
            "monitor_or_dev_used_for_fitting": False,
            "g_c_decomp_loaded": False,
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if not smoke:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(soup_state, out_dir / "full_y_seed0_soup_state.pt")
        np.savez_compressed(
            out_dir / "full_y_seed0_dev_preds.npz",
            row=np.asarray(dev_rows, dtype=np.int64),
            pred=dev_pred_soup.astype(np.float32),
            pred_cal=(dev_pred_soup + b_cal).astype(np.float32),
            y=y_dev.astype(np.float32),
        )
        model.load_state_dict({k: v.to(device) for k, v in soup_state.items()})
        np.savez_compressed(
            out_dir / "full_y_seed0_monitor_preds.npz",
            row=np.asarray(mon_rows, dtype=np.int64),
            pred=predict_raw(model, mon_data, device).astype(np.float32),
            y=y_mon.astype(np.float32),
        )
    write_json(out_dir / f"train_{CANDIDATE}{'_smoke' if smoke else ''}.json", results)
    log(
        f"[{CANDIDATE}] soup dev {res_soup['dev_mae']:.5f} (cal {results['primary_soup']['calibrated_dev_mae']:.5f}) | "
        f"best {res_best['dev_mae']:.5f} | last {res_last['dev_mae']:.5f} | wall {wall:.0f}s"
    )
    return results


# ---------------------------------------------------------------------------
# paired comparison vs frozen O-rich
# ---------------------------------------------------------------------------


def run_compare(full_npz: Path, out_dir: Path, log=print) -> dict[str, Any]:
    z_full = np.load(full_npz)
    z_v1 = np.load(V1_ORICH_EXPORT)
    row = np.asarray(z_full["row"], dtype=np.int64)
    assert np.array_equal(row, np.sort(z_v1["row"])), "row sets must match"
    v1 = {int(r): i for i, r in enumerate(z_v1["row"])}
    order = np.array([v1[int(r)] for r in row])
    y = z_full["y"].astype(np.float64)
    assert np.allclose(y, z_v1["y"].astype(np.float64)[order]), "y mismatch between exports"
    e_full = np.abs(z_full["pred"].astype(np.float64) - y)
    e_or = np.abs(z_v1["pred_orich"].astype(np.float64)[order] - y)
    diff = e_full - e_or
    rng = np.random.default_rng(0)
    n = len(y)
    boots = np.empty(2000)
    for b in range(2000):
        smp = rng.integers(0, n, n)
        boots[b] = np.abs(e_full[smp] - e_or[smp]).mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    out = {
        "protocol": PROTOCOL_ID,
        "paired": True,
        "mae_full_y": float(e_full.mean()),
        "mae_orich": float(e_or.mean()),
        "mean_diff_full_minus_orich": float(diff.mean()),
        "bootstrap_ci95": [float(lo), float(hi)],
        "bootstrap_resamples": 2000,
        "bootstrap_seed": 0,
        "note": "molecule-level resampling on internal dev; NOT cross-seed training-variance evidence",
    }
    log(json.dumps(out, indent=1))
    write_json(out_dir / "paired_vs_orich.json", out)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=PROTOCOL_ID)
    parser.add_argument("mode", choices=("prep", "train", "compare"))
    parser.add_argument("--seed", type=int, default=0, choices=[0])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--full-npz", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    log = lambda *a, **k: print(*a, **k, flush=True)

    if args.mode == "compare":
        if args.full_npz is None:
            raise SystemExit("--full-npz required for compare")
        run_compare(args.full_npz, args.out, log=log)
        return 0

    idx = cf.build_split_indices([str(s) for s in cf.load_canonical_smiles()])
    train = load_train_rows(log=log)
    verify_row_alignment(train, log=log)
    if args.mode == "prep":
        build_prep_fit7200(train, idx, args.out, smoke=args.smoke, log=log)
        return 0
    prep_path, _ = build_prep_fit7200(train, idx, args.out, smoke=args.smoke, log=log)
    train_candidate(
        train, idx, prep_path,
        device=device, out_dir=args.out, seed=args.seed,
        epochs=args.epochs, smoke=args.smoke, log=log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
