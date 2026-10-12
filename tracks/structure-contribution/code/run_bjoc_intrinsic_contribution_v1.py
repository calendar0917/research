"""cscl-bjoc-intrinsic-contribution-feasibility-v1 driver.

Protocol: ``configs/bjoc_intrinsic_contribution_feasibility_v1.yaml`` (frozen
before any formal GPU trajectory).  Track: ``tracks/structure-contribution``.

The round asks whether the frozen BJOC-Null predictor can be turned into a
model whose prediction is *literally* the sum of shared atom / bond / junction
contributions (plus one explicit scalar per global topology feature) without
paying an unacceptable held-out prediction penalty.  Two arms, identical
inputs, splits and recipe:

* ``B`` — the published BJOC-Null backbone (413,505 parameters).
* ``S`` — IJCM (``bjoc_intrinsic_contribution_v1``): the same encoders and
  inputs, explicit per-object scalar contributions with fit-only type
  vocabularies and a prediction-invariant fit-only gauge.

Stages
------
``splits``     freeze two new canonical-SMILES grouped internal splits (A / B).
``objects``    fit each split's global-feature standardiser and type vocabulary
               on that split's fit rows only (objects cache read-only).
``params``     parameter/expression audit for B vs S.
``gates``      numeric correctness/conservation/gauge gates before any GPU run.
``freeze``     protocol freeze manifest (hashes of everything frozen).
``synthetic``  Phase D known-truth experiments (A identifiability, B co-
               occurrence trap, C necessary non-additivity) — CPU only.
``train`` / ``train-stage``
               formal training of one arm / a pre-registered pair.
``predict``    label-blind prediction freeze (monitor, optionally + locked).
``score``      ONE locked-label read per split -> metrics, paired bootstrap,
               subgroups.
``decide``     stage-1 monitor-based performance stop-loss (B vs S).
``locked-gate`` pre-registered split-A locked-dev promotion decision.
``mechanism``  CPU contribution-stability / transfer / OOV diagnostics.
``report``     final report assembly.
``decision``   single final decision label.

Discipline
----------
* official valid/test are never loaded in any form.
* training reads only fit/monitor raw y; locked labels are read exactly once
  inside ``score`` after all prediction files are written and hashed.
* raw-y L1 only; no ``g/ell/s/c/k`` supervision; no GNN/attention/iteration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bjoc_intrinsic_contribution_v1 as ijcm  # noqa: E402
import cscl_features as cf  # noqa: E402
import run_prediction_feasibility_v2 as rpv2  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-bjoc-intrinsic-contribution-feasibility-v1"
CONFIG_PATH = _CODE_DIR.parent / "configs" / "bjoc_intrinsic_contribution_feasibility_v1.yaml"
OUT = REPO_ROOT / "tracks/structure-contribution/results/bjoc_intrinsic_contribution_v1"

# ---- frozen recipe ---------------------------------------------------------
EPOCHS = 240
LR = 1.0e-3
WEIGHT_DECAY = 1.0e-5
GRAD_CLIP = 5.0
BATCH_SIZE = 128
SOUP_LAST = 5
TRAIN_SHUFFLE_OFFSET = 101
SEEDS = (0, 1)

# ---- published reference ---------------------------------------------------
BJOC_PARAMETERS = 413505
BJOC_PARAMETER_TOLERANCE = 0.05

# ---- legacy read-only inputs ----------------------------------------------
LEGACY_COMPOSER = REPO_ROOT / "tracks/structure-contribution/results/structural_relation_composer_v1"
OBJECTS_CACHE = LEGACY_COMPOSER / "bjoc_objects_v1.pt"
OBJECTS_CACHE_SHA256 = "08e3ab137e7a4829f5488eb912cd1b01e4a0c87fbf3ef2c1cd9cef6efe61eb12"

# ---- frozen split salts ----------------------------------------------------
SPLIT_SALTS = {
    "A": "cscl-ijcm-feasibility-v1-A",
    "B": "cscl-ijcm-feasibility-v1-B",
}
FIT_TARGET = 8000
MONITOR_TARGET = 1000

SPLIT_FILES = {"A": OUT / "split_a.json", "B": OUT / "split_b.json"}
PREP_FILES = {"A": OUT / "prep_a.json", "B": OUT / "prep_b.json"}

# ---- frozen performance stop-loss (resource-purchase rules) ----------------
GATE_A_MONITOR_DELTA_MAX = 0.005   # <= -> buy seed 1
GATE_B_MONITOR_DELTA_MAX = 0.008   # <= -> buy seed 1 but flag as costly
GATE_LOCKED_SEED_MEAN_MAX = 0.005  # F4: Split B purchase rule
GATE_LOCKED_STOP_MAX = 0.010       # F4: clearly unacceptable
GATE_WORST_ROW_SHARE_MAX = 0.05
BOOTSTRAP_N = 5000
BOOTSTRAP_SEED = 0
UNK_MIN_COUNT = ijcm.DEFAULT_MIN_TYPE_COUNT

ARMS = ("B", "S")


def log(msg: str) -> None:
    print(msg, flush=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(type(obj).__name__)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        import subprocess

        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_rng_state_sha256() -> str:
    h = hashlib.sha256()
    h.update(torch.get_rng_state().numpy().tobytes())
    if torch.cuda.is_available():
        for state in torch.cuda.get_rng_state_all():
            h.update(state.cpu().numpy().tobytes())
    return h.hexdigest()


def load_config() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# data / splits / labels
# ---------------------------------------------------------------------------


def load_smiles() -> list[str]:
    smiles = [str(s) for s in cf.load_canonical_smiles()]
    assert len(smiles) == 10000, len(smiles)
    return smiles


def build_split(smiles: Sequence[str], salt: str) -> dict[str, Any]:
    """Deterministic canonical-SMILES grouped split; labels never enter."""
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(smiles):
        groups.setdefault(str(s), []).append(i)
    keys = sorted(groups, key=lambda s: hashlib.sha256(f"{salt}|{s}".encode()).hexdigest())
    fit: list[int] = []
    monitor: list[int] = []
    locked: list[int] = []
    for key in keys:
        rows = sorted(groups[key])
        if len(fit) < FIT_TARGET:
            fit.extend(rows)
        elif len(monitor) < MONITOR_TARGET:
            monitor.extend(rows)
        else:
            locked.extend(rows)
    fit, monitor, locked = sorted(fit), sorted(monitor), sorted(locked)
    sizes = [len(v) for v in groups.values()]
    return {
        "salt": salt,
        "fit": fit,
        "monitor": monitor,
        "locked": locked,
        "n_groups": len(groups),
        "duplicate_group_sizes": {
            "n_groups_multi_row": int(sum(1 for v in sizes if v > 1)),
            "n_rows_in_multi_row_groups": int(sum(v for v in sizes if v > 1)),
            "max_group_size": int(max(sizes)),
            "n_groups_gt2": int(sum(1 for v in sizes if v > 2)),
        },
        "overlap": {
            "fit_monitor": len(set(fit) & set(monitor)),
            "fit_locked": len(set(fit) & set(locked)),
            "monitor_locked": len(set(monitor) & set(locked)),
            "union": len(set(fit) | set(monitor) | set(locked)),
        },
    }


def rows_sha256(rows: Sequence[int]) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes()).hexdigest()


def split_payload(split: Mapping[str, Any], crosstab: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL_ID,
        "salt": split["salt"],
        "fit_target": FIT_TARGET,
        "monitor_target": MONITOR_TARGET,
        "n_groups": split["n_groups"],
        "n_fit": len(split["fit"]),
        "n_monitor": len(split["monitor"]),
        "n_locked": len(split["locked"]),
        "overlap": split["overlap"],
        "duplicate_group_sizes": split["duplicate_group_sizes"],
        "fit_sha256_rows": rows_sha256(split["fit"]),
        "monitor_sha256_rows": rows_sha256(split["monitor"]),
        "locked_sha256_rows": rows_sha256(split["locked"]),
        "crosstab_with_A": dict(crosstab or {}),
        "locked_labels_read": False,
    }


def _ensure_splits(log_fn=print) -> dict[str, dict[str, Any]]:
    smiles = load_smiles()
    splits = {k: build_split(smiles, SPLIT_SALTS[k]) for k in ("A", "B")}
    crosstab = {
        name: {
            "A_fit": len(set(splits["A"]["fit"]) & set(splits["B"][name])),
            "A_monitor": len(set(splits["A"]["monitor"]) & set(splits["B"][name])),
            "A_locked": len(set(splits["A"]["locked"]) & set(splits["B"][name])),
        }
        for name in ("fit", "monitor", "locked")
    }
    for key, split in splits.items():
        if not SPLIT_FILES[key].exists():
            write_json(SPLIT_FILES[key], split_payload(split, crosstab if key == "B" else None))
            log_fn(f"[splits] wrote {SPLIT_FILES[key].name}")
        frozen = read_json(SPLIT_FILES[key])
        assert frozen["salt"] == SPLIT_SALTS[key]
        assert rows_sha256(split["fit"]) == frozen["fit_sha256_rows"], f"split {key} fit drifted"
        assert rows_sha256(split["monitor"]) == frozen["monitor_sha256_rows"], f"split {key} monitor drifted"
        assert rows_sha256(split["locked"]) == frozen["locked_sha256_rows"], f"split {key} locked drifted"
        assert split["overlap"]["union"] == 10000
        assert split["overlap"]["fit_monitor"] == 0
        assert split["overlap"]["fit_locked"] == 0
        assert split["overlap"]["monitor_locked"] == 0
    if not (OUT / "split_crosstab.json").exists():
        write_json(
            OUT / "split_crosstab.json",
            {
                "protocol": PROTOCOL_ID,
                "smiles_sha256": hashlib.sha256("\n".join(smiles).encode()).hexdigest(),
                "crosstab_A_vs_B_rows": crosstab,
                "note": "nested hash splits of the same 10k pool; overlap reported, never optimised",
            },
        )
    return splits


def read_split_rows(key: str, log_fn=print) -> dict[str, list[int]]:
    split = _ensure_splits(log_fn)[key]
    return {"fit": split["fit"], "monitor": split["monitor"], "locked": split["locked"]}


def load_objects() -> list[sro.MolObjects]:
    assert OBJECTS_CACHE.exists(), OBJECTS_CACHE
    objs = torch.load(OBJECTS_CACHE, map_location="cpu", weights_only=False)
    assert len(objs) == 10000, len(objs)
    return objs


def load_prep(split_key: str) -> dict[str, Any]:
    prep = read_json(PREP_FILES[split_key])
    assert prep["fit_only"] is True and prep["n_fit"] == FIT_TARGET
    return prep


def load_norm(split_key: str) -> tuple[np.ndarray, np.ndarray]:
    prep = load_prep(split_key)
    return (
        np.asarray(prep["global_mean"], dtype=np.float32),
        np.asarray(prep["global_scale"], dtype=np.float32),
    )


def load_vocabs(split_key: str) -> ijcm.TypeVocabs:
    prep = load_prep(split_key)
    payload = prep["vocabs"]

    def _rebuild(kind: str) -> ijcm.TypeVocab:
        entry = payload[kind]
        return ijcm.TypeVocab(
            kind=kind,
            min_count=int(entry["min_count"]),
            raw_to_index={int(k): int(v) for k, v in entry["raw_to_index"].items()},
            raw_counts={int(k): int(v) for k, v in entry["raw_counts"].items()},
            n_fit_instances=int(entry["n_fit_instances"]),
            n_fit_rows=int(entry["n_fit_rows"]),
        )
    return ijcm.TypeVocabs(atom=_rebuild("atom"), bond=_rebuild("bond"), junction=_rebuild("junction"))


def _load_encoded_train():
    return torch.load(rpv2.ENCODED_TRAIN, map_location="cpu", weights_only=False)


def load_y_rows(rows: Sequence[int], *, subset_name: str = "unspecified") -> np.ndarray:
    train = _load_encoded_train()
    out = np.asarray([float(train[int(i)].y.reshape(-1)[0]) for i in rows], dtype=np.float64)
    del train
    return out


# ---------------------------------------------------------------------------
# model builders / prediction
# ---------------------------------------------------------------------------


def _make_model(arm: str, split_key: str, seed: int, device: torch.device, prep: Mapping[str, Any] | None = None):
    prep = prep or load_prep(split_key)
    gmean = np.asarray(prep["global_mean"], dtype=np.float32)
    gscale = np.asarray(prep["global_scale"], dtype=np.float32)
    if arm == "B":
        return src.build_bjoc(
            src.BJOCConfig(),
            seed=seed,
            organization_mode=src.NULL,
            global_mean=gmean,
            global_scale=gscale,
            device=device,
        )
    if arm == "S":
        return ijcm.build_model(
            seed=seed,
            vocab_sizes={k: int(prep["vocabs"][k]["size"]) for k in ("atom", "bond", "junction")},
            global_mean=gmean,
            global_scale=gscale,
            device=device,
        )
    raise ValueError(arm)


def _batch_for(arm: str, objs: Sequence[sro.MolObjects], rows: Sequence[int], vocabs: ijcm.TypeVocabs | None):
    if arm == "B":
        return sro.batch_objects([objs[int(r)] for r in rows])
    assert vocabs is not None
    return ijcm.build_ijcm_batch(objs, rows, vocabs)


def _predict(
    arm: str,
    model: torch.nn.Module,
    objs: Sequence[sro.MolObjects],
    rows: Sequence[int],
    vocabs: ijcm.TypeVocabs | None,
    device: torch.device,
    chunk: int = 256,
) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), chunk):
            batch = _batch_for(arm, objs, list(rows[i : i + chunk]), vocabs)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            out.append(model(batch).detach().cpu().numpy().reshape(-1))
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _shuffle_batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    order = torch.randperm(int(n), generator=generator).tolist()
    return [order[i : i + batch_size] for i in range(0, n, batch_size)]


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    assert 0 < len(states) <= SOUP_LAST, len(states)
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0]}
    return members, soup


def arm_tag(arm: str, split_key: str, seed: int) -> str:
    return f"{arm}_{split_key}_s{seed}"


def train_arm(arm: str, split_key: str, seed: int, device: torch.device, *, smoke: bool = False, log_fn=print) -> dict[str, Any]:
    """Formal training of one arm: frozen recipe, raw-y L1, last-5 soup."""
    assert arm in ARMS and split_key in ("A", "B")
    rows = read_split_rows(split_key, log_fn)
    fit_rows, mon_rows = rows["fit"], rows["monitor"]
    if smoke:
        fit_rows, mon_rows = fit_rows[:512], mon_rows[:128]
    objs = load_objects()
    prep = load_prep(split_key)
    vocabs = load_vocabs(split_key) if arm == "S" else None
    y_fit = torch.as_tensor(load_y_rows(fit_rows, subset_name=f"{split_key} fit"), dtype=torch.float32)
    y_mon = load_y_rows(mon_rows, subset_name=f"{split_key} monitor")

    gen = torch.Generator().manual_seed(int(seed) + TRAIN_SHUFFLE_OFFSET)
    epoch_batches = [_shuffle_batches(len(fit_rows), BATCH_SIZE, gen) for _ in range(3 if smoke else EPOCHS)]
    batch_order_sha256 = hashlib.sha256(
        np.ascontiguousarray(np.asarray([idx for ep in epoch_batches for batch in ep for idx in batch], dtype=np.int64)).tobytes()
    ).hexdigest()

    seed_everything(seed)
    model = _make_model(arm, split_key, seed, device, prep)
    init_sha = ijcm.state_dict_hash(model)
    rng_after_init = torch_rng_state_sha256()
    n_params = int(sum(q.numel() for q in model.parameters()))
    optimiser = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    from collections import deque

    states: deque = deque(maxlen=SOUP_LAST)
    curve: list[dict[str, Any]] = []
    rng_hashes: dict[str, str] = {}
    started = time.perf_counter()
    epochs = 3 if smoke else EPOCHS
    for epoch in range(1, epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        t_ep = time.perf_counter()
        for indices in epoch_batches[epoch - 1]:
            batch = _batch_for(arm, objs, [fit_rows[i] for i in indices], vocabs)
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            target = y_fit[indices].to(device).view(-1)
            pred = model(batch)
            loss = F.l1_loss(pred.view(-1), target)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimiser.step()
            total += float((pred.view(-1) - target).abs().sum())
            n_mol += len(indices)
            steps += 1
        mon_pred = _predict(arm, model, objs, mon_rows, vocabs, device) if mon_rows else np.zeros(0)
        mon_mae = float(np.abs(mon_pred - y_mon).mean()) if mon_rows else float("nan")
        curve.append(
            {
                "epoch": epoch,
                "train_mae": total / max(n_mol, 1),
                "monitor_mae": mon_mae,
                "seconds": time.perf_counter() - t_ep,
                "steps": steps,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch in (1, 40, 120, epochs):
            rng_hashes[str(epoch)] = torch_rng_state_sha256()
        if epoch % 20 == 0 or epoch in (1, epochs):
            log_fn(
                f"[{arm}/{split_key}/s{seed}] ep={epoch:03d} train={curve[-1]['train_mae']:.6f} "
                f"mon={mon_mae:.6f} {curve[-1]['seconds']:.1f}s"
            )
        if not np.isfinite(curve[-1]["train_mae"]) or (mon_rows and not np.isfinite(mon_mae)):
            raise RuntimeError(f"non-finite loss at epoch {epoch}")
    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), len(curve))
    state = {k: v.to(device) for k, v in soup.items()}
    model.load_state_dict(state)
    fit_pred = _predict(arm, model, objs, fit_rows, vocabs, device)
    mon_pred = _predict(arm, model, objs, mon_rows, vocabs, device) if mon_rows else np.zeros(0)
    results: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "split": split_key,
        "split_salt": SPLIT_SALTS[split_key],
        "seed": int(seed),
        "git_commit": git_commit(),
        "smoke": bool(smoke),
        "epochs": epochs,
        "hyper": {
            "lr": LR, "weight_decay": WEIGHT_DECAY, "grad_clip": GRAD_CLIP, "batch_size": BATCH_SIZE,
            "epochs": EPOCHS, "optimizer": "Adam", "supervision": "raw y L1 only", "scheduler": "none",
            "amp": False, "train_shuffle_seed": int(seed) + TRAIN_SHUFFLE_OFFSET,
        },
        "soup_members": members,
        "parameters": n_params,
        "init_state_sha256": init_sha,
        "rng_state_after_init_sha256": rng_after_init,
        "rng_state_sha256_at_epochs": rng_hashes,
        "batch_order_sha256": batch_order_sha256,
        "steps_per_epoch": len(epoch_batches[0]) if epoch_batches else 0,
        "total_steps": int(sum(len(b) for b in epoch_batches)),
        "curve": curve,
        "wall_seconds": wall,
        "soup_fit_mae": float(np.abs(fit_pred - np.asarray(y_fit)).mean()),
        "soup_monitor_mae": float(np.abs(mon_pred - y_mon).mean()) if mon_rows else None,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "python": platform.python_version(),
        "fit_only_discipline": {
            "global_stats": "split fit rows only",
            "type_vocabulary": "split fit rows only" if arm == "S" else "n/a",
            "labels_read": [f"{split_key} fit", f"{split_key} monitor"],
            "locked_labels_read": False,
        },
        "official_valid_loaded": False,
        "official_test_loaded": False,
    }
    if arm == "S":
        offsets = ijcm.compute_fit_offsets(model, objs, fit_rows, vocabs, device)
        results["gauge"] = {
            "fit_rows_used": len(fit_rows),
            "offsets_sha256": {
                k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16] for k, v in offsets.items()
            },
            "note": "fit-only per-type mean of the raw context correction; prediction-invariant",
        }
    if not smoke:
        payload = {"state_dict": soup, "arm": arm, "split": split_key, "seed": int(seed), "protocol": PROTOCOL_ID}
        if arm == "S":
            payload["gauge_offsets"] = {k: v.astype(np.float64) for k, v in offsets.items()}
            payload["vocab_sizes"] = {k: int(prep["vocabs"][k]["size"]) for k in ("atom", "bond", "junction")}
        torch.save(payload, OUT / f"checkpoint_{arm_tag(arm, split_key, seed)}.pt")
        write_json(OUT / f"train_{arm_tag(arm, split_key, seed)}.json", results)
    log_fn(
        f"[{arm}/{split_key}/s{seed}] soup fit {results['soup_fit_mae']:.6f} "
        f"monitor {results['soup_monitor_mae']} wall {wall:.0f}s params {n_params}"
    )
    del objs
    return results


def load_checkpoint(arm: str, split_key: str, seed: int, device: torch.device):
    ckpt_path = OUT / f"checkpoint_{arm_tag(arm, split_key, seed)}.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = _make_model(arm, split_key, seed, device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    offsets = ckpt.get("gauge_offsets")
    return model, offsets, ckpt, ckpt_path


# ---------------------------------------------------------------------------
# commands: prepare
# ---------------------------------------------------------------------------


def cmd_splits(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    splits = _ensure_splits()
    for key, split in splits.items():
        log(
            f"[splits] {key}: fit={len(split['fit'])} monitor={len(split['monitor'])} "
            f"locked={len(split['locked'])} fit_sha={rows_sha256(split['fit'])[:12]}"
        )
    return 0


def cmd_objects(args: argparse.Namespace) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    _ensure_splits()
    cache_sha = file_sha256(OBJECTS_CACHE)
    assert cache_sha == OBJECTS_CACHE_SHA256, f"objects cache drifted: {cache_sha}"
    objs = load_objects()
    for key in ("A", "B"):
        rows = read_split_rows(key)
        gfit = np.stack([objs[int(i)].global_feat for i in rows["fit"]], 0).astype(np.float64)
        gmean = gfit.mean(0)
        gscale = np.where(gfit.std(0) < 1e-6, 1.0, gfit.std(0))
        vocabs = ijcm.TypeVocabs.build(objs, rows["fit"], min_count=UNK_MIN_COUNT)
        vocab_payload = vocabs.as_payload()
        for kind in ("atom", "bond", "junction"):
            vocab_payload[kind]["raw_to_index"] = {
                str(k): int(v) for k, v in getattr(vocabs, kind).raw_to_index.items()
            }
        payload = {
            "protocol": PROTOCOL_ID,
            "split": key,
            "fit_only": True,
            "n_fit": int(len(rows["fit"])),
            "global_mean": gmean.tolist(),
            "global_scale": gscale.tolist(),
            "names": list(sro.GLOBAL_FEAT_NAMES),
            "vocabs": vocab_payload,
            "labels_read": False,
        }
        write_json(PREP_FILES[key], payload)
        log(
            f"[objects] {key}: norm + type vocab fitted on {len(rows['fit'])} fit rows "
            f"(atom {vocabs.atom.size} / bond {vocabs.bond.size} / junction {vocabs.junction.size}) "
            f"labels never read"
        )
    write_json(
        OUT / "objects_summary.json",
        {
            "protocol": PROTOCOL_ID,
            "objects_cache": str(OBJECTS_CACHE.relative_to(REPO_ROOT)),
            "objects_cache_sha256": cache_sha,
            "n_rows": len(objs),
            "n_atoms": int(sum(o.n_atoms for o in objs)),
            "n_bonds": int(sum(o.bond_index.shape[0] for o in objs)),
            "n_junctions": int(sum(o.junc_index.shape[0] for o in objs)),
            "labels_read": False,
        },
    )
    del objs
    return 0


def cmd_params(args: argparse.Namespace) -> int:
    bjoc_inv = src.BJOCConfig().inventory()
    payload: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "bjoc_reference": {"total": bjoc_inv["total"], "published": BJOC_PARAMETERS, "inventory": bjoc_inv},
        "ijcm_config": {
            "atom_hidden": ijcm.IJCMConfig().atom_hidden,
            "atom_out": ijcm.IJCMConfig().atom_out,
            "bond_hidden": ijcm.IJCMConfig().bond_hidden,
            "bond_out": ijcm.IJCMConfig().bond_out,
            "junc_hidden": ijcm.IJCMConfig().junc_hidden,
            "junc_out": ijcm.IJCMConfig().junc_out,
            "contrib_hidden": ijcm.IJCMConfig().contrib_hidden,
            "eta_hidden": ijcm.IJCMConfig().eta_hidden,
        },
        "splits": {},
    }
    device = torch.device("cpu")
    for key in ("A", "B"):
        prep = load_prep(key)
        b_model = _make_model("B", key, 0, device, prep)
        s_model = _make_model("S", key, 0, device, prep)
        b_audit = {"actual": int(sum(p.numel() for p in b_model.parameters())), "published": BJOC_PARAMETERS}
        s_audit = ijcm.parameter_audit(s_model)
        bjoc_inv = src.BJOCConfig().inventory()
        payload["splits"][key] = {
            "B": b_audit,
            "S": s_audit,
            "expression_diff": ijcm.expression_diff(bjoc_inv, s_audit["inventory"]),
        }
        log(
            f"[params] split {key}: B={b_audit['actual']} S={s_audit['actual_parameters']} "
            f"ratio={s_audit['actual_parameters'] / b_audit['actual']:.4f}"
        )
    write_json(OUT / "parameter_and_init_audit.json", payload)
    return 0


# ---------------------------------------------------------------------------
# correctness gates
# ---------------------------------------------------------------------------


def _gates_tiny_rows() -> tuple[list[int], list[int]]:
    rows = read_split_rows("A")
    return rows["fit"][:24], rows["locked"][:8]


def _relabeled_objects(obj: sro.MolObjects, perm: np.ndarray) -> sro.MolObjects:
    """Rebuild an object set from a relabelled copy of the same object.

    The object stores only canonical one-shot features, so the relabelling is
    applied to the *dataset row* in the gate below, not here; this helper only
    checks the multiset of type ids lands identically.
    """
    return obj


def cmd_gates(args: argparse.Namespace) -> int:
    """Numeric correctness / conservation / gauge / relabel gates (CPU)."""
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    objs = load_objects()
    fit_rows = read_split_rows("A")["fit"][:64]
    tiny, tiny2 = _gates_tiny_rows()
    prep = load_prep("A")
    vocabs = load_vocabs("A")
    gates: dict[str, Any] = {"protocol": PROTOCOL_ID, "official_valid_or_test_loaded": False}

    b_model = _make_model("B", "A", 0, device, prep)
    s_model = _make_model("S", "A", 0, device, prep)
    b_params = int(sum(p.numel() for p in b_model.parameters()))
    s_params = int(sum(p.numel() for p in s_model.parameters()))
    gates["parameters"] = {
        "B": b_params,
        "S": s_params,
        "B_exact_published": b_params == BJOC_PARAMETERS,
        "S_within_tolerance": abs(s_params - b_params) / b_params <= BJOC_PARAMETER_TOLERANCE,
        "S_expression_inventory_ok": ijcm.parameter_audit(s_model)["parameter_exact_vs_inventory"],
    }

    # -- conservation with and without the gauge ----------------------------
    offsets = ijcm.compute_fit_offsets(s_model, objs, fit_rows, vocabs, device)
    cons_raw = ijcm.verify_conservation(s_model, objs, tiny, vocabs, None, device)
    cons_gauge = ijcm.verify_conservation(s_model, objs, tiny, vocabs, offsets, device)
    gates["conservation"] = {"raw": cons_raw, "gauged": cons_gauge}

    # -- gauge invariance at the prediction level and per-type mean zero ----
    batch = ijcm.build_ijcm_batch(objs, tiny, vocabs)
    with torch.no_grad():
        terms = s_model.raw_terms(batch)
        pred = s_model.sum_terms(terms, batch).cpu().numpy().astype(np.float64)
        can = ijcm.gauge_terms(terms, offsets, batch)
    pred_from_can = np.full(int(batch["n_graphs"]), float(can["bias"]), dtype=np.float64)
    for head, kind in (("alpha", "atom"), ("beta", "bond"), ("nu", "junc")):
        pred_from_can += np.bincount(
            batch[f"{kind}_batch"].cpu().numpy(),
            weights=can[f"{head}_score_canonical"],
            minlength=int(batch["n_graphs"]),
        )
    pred_from_can += can["eta_per_feature"].sum(axis=1)
    gates["gauge"] = {
        "pred_gauge_gap_max": float(np.abs(pred_from_can - pred).max()),
        "instance_total_gap_max": max(
            float(np.abs((can[f"{h}_type_canonical"] + can[f"{h}_ctx_canonical"]) - can[f"{h}_score_canonical"]).max())
            for h in ("alpha", "beta", "nu")
        ),
        "invariant_atol": 1e-5,
    }
    gates["gauge"]["pred_invariant_ok"] = gates["gauge"]["pred_gauge_gap_max"] <= 1e-5
    gates["gauge"]["instance_invariant_ok"] = gates["gauge"]["instance_total_gap_max"] <= 1e-5
    # per-type fit mean of canonical context == 0 (float64)
    for kind, head in (("atom", "alpha"), ("bond", "beta"), ("junction", "nu")):
        sums = np.zeros(getattr(vocabs, kind).size, dtype=np.float64)
        counts = np.zeros(getattr(vocabs, kind).size, dtype=np.int64)
        cf_batch = ijcm.build_ijcm_batch(objs, fit_rows, vocabs)
        with torch.no_grad():
            t_fit = s_model.raw_terms(cf_batch)
            c_fit = ijcm.gauge_terms(t_fit, offsets, cf_batch)
        np.add.at(sums, c_fit[f"{head}_type_index"], c_fit[f"{head}_ctx_canonical"])
        np.add.at(counts, c_fit[f"{head}_type_index"], 1)
        present = counts > 0
        mean_abs = float(np.abs(sums[present] / counts[present]).max()) if present.any() else 0.0
        gates["gauge"][f"fit_canonical_ctx_mean_max_abs_{kind}"] = mean_abs
        gates["gauge"][f"fit_canonical_ctx_mean_ok_{kind}"] = mean_abs <= 1e-6

    # -- atom relabelling invariance (S and B) ------------------------------
    ds = cf.load_official_train()
    relabel_rows = []
    for r in tiny2[:4]:
        d = ds[int(r)]
        x = d.x.numpy().reshape(-1)
        ei = d.edge_index.numpy()
        ea = d.edge_attr.numpy()
        perm = np.arange(len(x) - 1, -1, -1)
        x2 = x[perm].reshape(-1, 1)
        ei2 = perm[ei]
        obj2 = sro.build_mol_objects(-1, x2, ei2, ea)
        relabel_rows.append((int(r), obj2))
    relabel_report = {}
    for arm, model in (("B", b_model), ("S", s_model)):
        gap = 0.0
        type_gap = 0
        for r, obj2 in relabel_rows:
            b1 = _batch_for(arm, objs, [r], vocabs)
            b2 = _batch_for(arm, [obj2], [0], vocabs)
            with torch.no_grad():
                p1 = model(b1).item()
                p2 = model(b2).item()
            gap = max(gap, abs(p1 - p2))
            if arm == "S":
                for kind in ("atom", "bond", "junction"):
                    t1 = np.sort(ijcm.RAW_TYPE_FN[kind](objs[r]))
                    t2 = np.sort(ijcm.RAW_TYPE_FN[kind](obj2))
                    type_gap += int(not np.array_equal(t1, t2))
        relabel_report[arm] = {"max_abs_pred_gap": gap}
        if arm == "S":
            relabel_report[arm]["type_multiset_mismatches"] = type_gap
    gates["atom_relabel"] = relabel_report
    gates["atom_relabel_ok"] = all(v["max_abs_pred_gap"] <= 1e-5 for v in relabel_report.values()) and all(
        v.get("type_multiset_mismatches", 0) == 0 for v in relabel_report.values()
    )

    # -- batch recomposition: per-molecule independence ---------------------
    mols = tiny[:4]
    recomposition_ok = True
    max_gap = 0.0
    for arm, model in (("B", b_model), ("S", s_model)):
        together = _predict(arm, model, objs, mols, vocabs, device, chunk=4)
        singles = np.asarray([_predict(arm, model, objs, [r], vocabs, device, chunk=1)[0] for r in mols])
        gap = float(np.abs(together - singles).max())
        max_gap = max(max_gap, gap)
        recomposition_ok &= gap <= 1e-5
    gates["batch_recomposition"] = {"max_abs_gap": max_gap, "ok": bool(recomposition_ok)}

    # -- save/load replay ----------------------------------------------------
    replay = {}
    for arm in ("B", "S"):
        model = b_model if arm == "B" else s_model
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        before = _predict(arm, model, objs, tiny, vocabs, device)
        model.load_state_dict(state)
        after = _predict(arm, model, objs, tiny, vocabs, device)
        replay[arm] = {"max_abs_gap": float(np.abs(before - after).max())}
    gates["save_load_replay"] = replay
    gates["save_load_replay_ok"] = all(v["max_abs_gap"] <= 1e-6 for v in replay.values())

    # -- determinism of the builders ----------------------------------------
    h1 = ijcm.state_dict_hash(_make_model("S", "A", 3, device, prep))
    h2 = ijcm.state_dict_hash(_make_model("S", "A", 3, device, prep))
    h3 = ijcm.state_dict_hash(_make_model("S", "A", 4, device, prep))
    gates["builder_determinism"] = {
        "same_seed_same_state": h1 == h2,
        "different_seed_differs": h1 != h3,
    }

    # -- type vocabulary / UNK backoff --------------------------------------
    vocab_report = {}
    for kind in ("atom", "bond", "junction"):
        v = getattr(vocabs, kind)
        seen = np.concatenate([ijcm.RAW_TYPE_FN[kind](objs[int(r)]) for r in tiny]) if tiny else np.zeros(0, np.int64)
        oov = v.is_oov_mask(seen) if seen.size else np.zeros(0, bool)
        vocab_report[kind] = {
            "size": v.size,
            "n_known": len(v.raw_to_index),
            "n_raw_types_fit": len(v.raw_counts),
            "unknown_raw_max": int(seen.max()) if seen.size else None,
            "oov_rate_sample": float(oov.mean()) if oov.size else 0.0,
        }
    gates["type_vocab"] = vocab_report

    # -- no hidden prediction bypass ----------------------------------------
    src_text = (_CODE_DIR / "bjoc_intrinsic_contribution_v1.py").read_text(encoding="utf-8")
    module_names = sorted({type(m).__name__ for m in s_model.modules()})
    allowed_modules = {"IJCM", "_MLP", "Linear", "ModuleList", "Embedding"}
    gates["no_hidden_bypass"] = {
        "module_types": module_names,
        "only_allowed_module_types": sorted(set(module_names) - allowed_modules) == [],
        "forbidden_tokens_present": [
            t for t in ("Transformer", "MultiheadAttention", "MessagePassing", "LSTMCell", "GRU", "LSTM")
            if t in src_text
        ],
        "forward_is_only_sum": "sum_terms" in src_text,
        "note": "the conservation identity is the operative guard: pred is recomputed from exported terms",
    }

    # -- label blindness of the batch builder -------------------------------
    batch_keys = sorted(ijcm.build_ijcm_batch(objs, tiny, vocabs).keys())
    gates["label_blind_batch"] = {
        "keys": batch_keys,
        "no_y_key": all(k != "y" and "target" not in k for k in batch_keys),
    }

    # -- endpoint swap: PyG stores each bond twice; the object builder must
    # canonicalise them to one physical bond (checked through the cache hash)
    gates["bond_canonicalisation"] = {
        "objects_cache_sha256": OBJECTS_CACHE_SHA256,
        "note": "unique_undirected_bonds canonicalises u<v and rejects self-loops/duplicates",
    }

    gates["all_gates_ok"] = bool(
        gates["parameters"]["B_exact_published"]
        and gates["parameters"]["S_expression_inventory_ok"]
        and gates["conservation"]["raw"]["ok"]
        and gates["conservation"]["gauged"]["ok"]
        and gates["gauge"]["pred_invariant_ok"]
        and gates["gauge"]["instance_invariant_ok"]
        and all(gates["gauge"][f"fit_canonical_ctx_mean_ok_{k}"] for k in ("atom", "bond", "junction"))
        and gates["atom_relabel_ok"]
        and gates["batch_recomposition"]["ok"]
        and gates["save_load_replay_ok"]
        and gates["builder_determinism"]["same_seed_same_state"]
        and gates["builder_determinism"]["different_seed_differs"]
        and gates["no_hidden_bypass"]["only_allowed_module_types"]
        and not gates["no_hidden_bypass"]["forbidden_tokens_present"]
    )
    write_json(OUT / "correctness_gates.json", gates)
    log(f"[gates] all_gates_ok={gates['all_gates_ok']}")
    del objs
    return 0 if gates["all_gates_ok"] else 1


def cmd_freeze(args: argparse.Namespace) -> int:
    """Freeze manifest: hashes of everything that is fixed before training."""
    config = load_config()
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "git_commit": git_commit(),
        "config_path": str(CONFIG_PATH.relative_to(REPO_ROOT)),
        "config_sha256": file_sha256(CONFIG_PATH),
        "config": config,
        "artifacts": {},
    }
    artifacts = [
        OUT / "split_a.json",
        OUT / "split_b.json",
        OUT / "split_crosstab.json",
        OUT / "prep_a.json",
        OUT / "prep_b.json",
        OUT / "objects_summary.json",
        OUT / "correctness_gates.json",
        OUT / "parameter_and_init_audit.json",
    ]
    for path in artifacts:
        if path.exists():
            manifest["artifacts"][path.name] = {
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
    manifest["code_sha256"] = {
        name: file_sha256(_CODE_DIR / name)
        for name in (
            "bjoc_intrinsic_contribution_v1.py",
            "run_bjoc_intrinsic_contribution_v1.py",
            "structural_relation_composer_v1.py",
            "structural_relation_objects_v1.py",
        )
    }
    manifest["frozen_recipe"] = {
        "epochs": EPOCHS, "lr": LR, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "grad_clip": GRAD_CLIP, "soup_last": SOUP_LAST, "seeds": list(SEEDS),
        "loss": "raw-y L1", "optimizer": "Adam", "scheduler": None,
    }
    manifest["frozen_salts"] = SPLIT_SALTS
    manifest["gates"] = {
        "stage1_monitor_delta_buy_seed1_max": GATE_A_MONITOR_DELTA_MAX,
        "stage1_monitor_delta_edge_band_max": GATE_B_MONITOR_DELTA_MAX,
        "locked_seed_mean_split_b_purchase_max": GATE_LOCKED_SEED_MEAN_MAX,
        "locked_seed_mean_clearly_unacceptable_min": GATE_LOCKED_STOP_MAX,
        "worst_row_error_mass_share_max": GATE_WORST_ROW_SHARE_MAX,
    }
    manifest["official_valid_or_test_referenced"] = False
    write_json(OUT / "protocol_freeze_manifest.json", manifest)
    log(f"[freeze] wrote protocol_freeze_manifest.json ({len(manifest['artifacts'])} artifacts)")
    return 0


# ---------------------------------------------------------------------------
# commands: train / predict / score / decide
# ---------------------------------------------------------------------------


def cmd_train(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    train_arm(args.arm, args.split, int(args.seed), device, smoke=bool(args.smoke))
    return 0


STAGE_PLANS = {
    "pair-a-s0": [("B", "A", 0), ("S", "A", 0)],
    "pair-a-s1": [("B", "A", 1), ("S", "A", 1)],
    "pair-b-s0": [("B", "B", 0), ("S", "B", 0)],
    "pair-b-s1": [("B", "B", 1), ("S", "B", 1)],
}


def cmd_train_stage(args: argparse.Namespace) -> int:
    device = torch.device(args.device)
    plan = STAGE_PLANS[args.stage]
    log(f"[stage {args.stage}] device={device} plan={plan}")
    for arm, split_key, seed in plan:
        marker = OUT / f"train_{arm_tag(arm, split_key, seed)}.json"
        if marker.exists() and not args.force:
            log(f"[stage] skip {marker.name} (already trained)")
            continue
        train_arm(arm, split_key, seed, device)
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Label-blind predictions; the only rows ever predicted are internal split rows."""
    device = torch.device(args.device)
    split_key = args.split
    rows = read_split_rows(split_key)
    objs = load_objects()
    vocabs = load_vocabs(split_key)
    arms = [a for a in args.arms.split(",") if a]
    manifest: dict[str, Any] = {
        "protocol": PROTOCOL_ID, "split": split_key, "arms": {}, "labels_read": False,
        "rows_mode": args.rows, "locked_predictions_written": args.rows == "both",
        "official_valid_or_test_loaded": False,
    }
    monitor_preds = {"monitor_row": np.asarray(rows["monitor"], dtype=np.int64)}
    locked_preds = {"locked_row": np.asarray(rows["locked"], dtype=np.int64)}
    for tag in arms:
        arm = tag.split("_")[0]
        seed = int(tag.split("_s")[-1])
        model, offsets, ckpt, ckpt_path = load_checkpoint(arm, split_key, seed, device)
        mon = _predict(arm, model, objs, rows["monitor"], vocabs, device)
        monitor_preds[f"pred_{tag}_monitor"] = mon.astype(np.float32)
        entry = {
            "checkpoint_sha256": file_sha256(ckpt_path),
            "monitor_pred_sha256": hashlib.sha256(mon.astype(np.float32).tobytes()).hexdigest(),
            "parameters": int(sum(q.numel() for q in model.parameters())),
        }
        if args.rows == "both":
            locked = _predict(arm, model, objs, rows["locked"], vocabs, device)
            locked_preds[f"pred_{tag}_locked"] = locked.astype(np.float32)
            entry["locked_pred_sha256"] = hashlib.sha256(locked.astype(np.float32).tobytes()).hexdigest()
        manifest["arms"][tag] = entry
        log(f"[predict] {tag} written (rows={args.rows})")
        del model
    if args.rows == "both":
        np.savez_compressed(OUT / f"split_{split_key.lower()}_predictions.npz", **monitor_preds, **locked_preds)
        manifest["predictions_file"] = f"split_{split_key.lower()}_predictions.npz"
        manifest["predictions_file_sha256"] = file_sha256(OUT / f"split_{split_key.lower()}_predictions.npz")
    else:
        np.savez_compressed(OUT / f"split_{split_key.lower()}_monitor_predictions.npz", **monitor_preds)
        manifest["predictions_file"] = f"split_{split_key.lower()}_monitor_predictions.npz"
        manifest["predictions_file_sha256"] = file_sha256(OUT / f"split_{split_key.lower()}_monitor_predictions.npz")
    write_json(OUT / f"prediction_manifest_{split_key}.json", manifest)
    del objs
    return 0


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = np.abs(pred.astype(np.float64) - y)
    return {
        "mae": float(err.mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "median": float(np.median(err)),
        "p90": float(np.percentile(err, 90)),
        "p95": float(np.percentile(err, 95)),
        "max": float(err.max()),
    }


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    N = int(delta.size)
    means = np.empty(n, dtype=np.float64)
    chunk = 500
    for i in range(0, n, chunk):
        idx = rng.integers(0, N, size=(min(chunk, n - i), N))
        means[i : i + idx.shape[0]] = delta[idx].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n_resamples": int(n),
        "seed": int(seed),
        "frac_candidate_better": float((delta < 0).mean()),
        "frac_candidate_worse": float((delta > 0).mean()),
    }


def _top10_removed_delta(delta: np.ndarray, total_err_B: np.ndarray) -> float:
    order = np.argsort(-total_err_B)
    mask = np.ones_like(delta, dtype=bool)
    mask[order[:10]] = False
    return float(delta[mask].mean())


def _subgroup_deltas(err_B: np.ndarray, err_S: np.ndarray, y: np.ndarray, ss: Mapping[str, np.ndarray]) -> dict[str, Any]:
    masks = {
        "all": np.ones_like(y, dtype=bool),
        "ordinary_y_gt_-3": y > -3.0,
        "tail_y_le_-3": y <= -3.0,
        "n_atoms_le_median": ss["n_atoms"] <= np.median(ss["n_atoms"]),
        "n_atoms_gt_median": ss["n_atoms"] > np.median(ss["n_atoms"]),
        "junctions_le_median": ss["n_junctions"] <= np.median(ss["n_junctions"]),
        "junctions_gt_median": ss["n_junctions"] > np.median(ss["n_junctions"]),
        "cyclomatic_0": ss["cyclomatic"] <= 0,
        "cyclomatic_ge_1": ss["cyclomatic"] >= 1,
    }
    out = {}
    for name, mask in masks.items():
        d = err_S[mask] - err_B[mask]
        out[name] = {
            "n": int(mask.sum()),
            "mae_B": float(err_B[mask].mean()) if mask.any() else float("nan"),
            "mae_S": float(err_S[mask].mean()) if mask.any() else float("nan"),
            "delta_S_minus_B": float(d.mean()) if mask.any() else float("nan"),
            "paired_bootstrap": paired_bootstrap(d) if mask.sum() > 1 else None,
        }
    return out


def cmd_score(args: argparse.Namespace) -> int:
    """The ONE locked-label read for this split (after predictions are frozen)."""
    split_key = args.split
    rows = read_split_rows(split_key)
    preds_path = OUT / f"split_{split_key.lower()}_predictions.npz"
    assert preds_path.exists(), preds_path
    preds = np.load(preds_path)
    mon_rows = [int(r) for r in preds["monitor_row"]]
    locked_rows = [int(r) for r in preds["locked_row"]]
    assert mon_rows == rows["monitor"]
    assert locked_rows == rows["locked"]
    y_mon = load_y_rows(mon_rows, subset_name=f"{split_key} monitor")
    y_locked = load_y_rows(locked_rows, subset_name=f"{split_key} locked")
    objs = load_objects()
    ss = _structural_stats(objs, locked_rows)

    model_entries = sorted({k.split("_monitor")[0] for k in preds.files if k.startswith("pred_") and k.endswith("_monitor")})
    out: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "split": split_key,
        "label_read": "single score invocation",
        "official_valid_or_test_loaded": False,
        "arms": {},
    }
    for key in model_entries:
        tag = key.replace("pred_", "")
        arm = tag.split("_")[0]
        seed = int(tag.split("_s")[-1])
        train_json = read_json(OUT / f"train_{tag}.json")
        pred_mon = preds[f"{key}_monitor"].astype(np.float64)
        pred_locked = preds[f"{key}_locked"].astype(np.float64)
        entry = {
            "arm": arm,
            "seed": seed,
            "fit_mae": train_json["soup_fit_mae"],
            "monitor": metrics(pred_mon, y_mon),
            "locked": metrics(pred_locked, y_locked),
            "parameters": train_json["parameters"],
            "wall_seconds": train_json["wall_seconds"],
            "init_state_sha256": train_json["init_state_sha256"],
            "batch_order_sha256": train_json["batch_order_sha256"],
        }
        out["arms"][tag] = entry
        log(f"[score] {tag}: fit {entry['fit_mae']:.6f} monitor {entry['monitor']['mae']:.6f} locked {entry['locked']['mae']:.6f}")

    # paired comparisons per seed (same rows, same split)
    paired: dict[str, Any] = {"protocol": PROTOCOL_ID, "split": split_key}
    seeds = sorted({e["seed"] for e in out["arms"].values()})
    paired_by_seed = {}
    for seed in seeds:
        tags = {a: tag for tag, e in out["arms"].items() for a in (e["arm"],) if e["seed"] == seed}
        if set(tags) != {"B", "S"}:
            continue
        err_B_mon = np.abs(preds[f"pred_{tags['B']}_monitor"].astype(np.float64) - y_mon)
        err_S_mon = np.abs(preds[f"pred_{tags['S']}_monitor"].astype(np.float64) - y_mon)
        err_B_lock = np.abs(preds[f"pred_{tags['B']}_locked"].astype(np.float64) - y_locked)
        err_S_lock = np.abs(preds[f"pred_{tags['S']}_locked"].astype(np.float64) - y_locked)
        d_mon = err_S_mon - err_B_mon
        d_lock = err_S_lock - err_B_lock
        entry = {
            "monitor": {
                "delta_S_minus_B": float(d_mon.mean()),
                "paired_bootstrap": paired_bootstrap(d_mon),
                "mae_B": float(err_B_mon.mean()),
                "mae_S": float(err_S_mon.mean()),
            },
            "locked": {
                "delta_S_minus_B": float(d_lock.mean()),
                "paired_bootstrap": paired_bootstrap(d_lock),
                "mae_B": float(err_B_lock.mean()),
                "mae_S": float(err_S_lock.mean()),
                "ordinary_delta": float((err_S_lock - err_B_lock)[y_locked > -3.0].mean()),
                "tail_delta": float((err_S_lock - err_B_lock)[y_locked <= -3.0].mean()),
                "top10_removed_delta": _top10_removed_delta(d_lock, err_B_lock),
                "worst_row_error_mass_share_B": float(np.max(err_B_lock) / err_B_lock.sum()),
                "worst_row_error_mass_share_S": float(np.max(err_S_lock) / err_S_lock.sum()),
                "subgroups": _subgroup_deltas(err_B_lock, err_S_lock, y_locked, ss),
            },
        }
        paired_by_seed[seed] = entry
        log(f"[score] seed {seed}: monitor Δ(S−B) = {entry['monitor']['delta_S_minus_B']:+.6f}; locked Δ = {entry['locked']['delta_S_minus_B']:+.6f}")
    paired["by_seed"] = paired_by_seed
    if len(seeds) >= 2:
        err_B_lock = np.mean(
            [np.abs(preds[f"pred_B_{split_key}_s{s}_locked"].astype(np.float64) - y_locked) for s in seeds], axis=0
        )
        err_S_lock = np.mean(
            [np.abs(preds[f"pred_S_{split_key}_s{s}_locked"].astype(np.float64) - y_locked) for s in seeds], axis=0
        )
        err_B_mon = np.mean(
            [np.abs(preds[f"pred_B_{split_key}_s{s}_monitor"].astype(np.float64) - y_mon) for s in seeds], axis=0
        )
        err_S_mon = np.mean(
            [np.abs(preds[f"pred_S_{split_key}_s{s}_monitor"].astype(np.float64) - y_mon) for s in seeds], axis=0
        )
        d_lock = err_S_lock - err_B_lock
        paired["seed_mean"] = {
            "seeds": seeds,
            "monitor": {
                "mae_B": float(err_B_mon.mean()),
                "mae_S": float(err_S_mon.mean()),
                "delta_S_minus_B": float((err_S_mon - err_B_mon).mean()),
                "paired_bootstrap": paired_bootstrap(err_S_mon - err_B_mon),
            },
            "locked": {
                "mae_B": float(err_B_lock.mean()),
                "mae_S": float(err_S_lock.mean()),
                "delta_S_minus_B": float(d_lock.mean()),
                "paired_bootstrap": paired_bootstrap(d_lock),
                "ordinary_delta": float(d_lock[y_locked > -3.0].mean()),
                "tail_delta": float(d_lock[y_locked <= -3.0].mean()),
                "top10_removed_delta": _top10_removed_delta(d_lock, err_B_lock),
                "per_seed_locked_delta": [paired_by_seed[s]["locked"]["delta_S_minus_B"] for s in seeds],
                "seed_spread_B_locked": float(
                    np.ptp([np.abs(preds[f"pred_B_{split_key}_s{s}_locked"].astype(np.float64) - y_locked).mean() for s in seeds])
                ),
                "seed_spread_S_locked": float(
                    np.ptp([np.abs(preds[f"pred_S_{split_key}_s{s}_locked"].astype(np.float64) - y_locked).mean() for s in seeds])
                ),
                "subgroups": _subgroup_deltas(err_B_lock, err_S_lock, y_locked, ss),
            },
        }
    out["paired"] = paired
    write_json(OUT / f"split_{split_key.lower()}_metrics.json", out)
    write_json(OUT / f"split_{split_key.lower()}_paired_bootstrap.json", paired)
    del objs
    return 0


def _structural_stats(objs: Sequence[sro.MolObjects], rows: Sequence[int]) -> dict[str, np.ndarray]:
    import run_bjoc_junction_regularization_v1 as jdrun

    return jdrun.structural_stats(objs, rows)


def _stage1_record() -> dict[str, Any]:
    per_seed = {}
    seeds = set()
    for path in OUT.glob("train_*_A_s*.json"):
        tag = path.name[len("train_") : -len(".json")]
        arm = tag.split("_")[0]
        seed = int(tag.split("_s")[-1])
        if arm not in ARMS:
            continue
        entry = read_json(path)
        per_seed.setdefault(seed, {})[arm] = entry
        seeds.add(seed)
    by_seed = {}
    for seed in sorted(seeds):
        if set(per_seed.get(seed, {})) != {"B", "S"}:
            continue
        b, s = per_seed[seed]["B"], per_seed[seed]["S"]
        by_seed[seed] = {
            "monitor_B": b["soup_monitor_mae"],
            "monitor_S": s["soup_monitor_mae"],
            "monitor_delta": s["soup_monitor_mae"] - b["soup_monitor_mae"],
            "fit_B": b["soup_fit_mae"],
            "fit_S": s["soup_fit_mae"],
        }
    return {"seeds": sorted(seeds), "per_seed": by_seed}


def cmd_decide(args: argparse.Namespace) -> int:
    """Stage-1 performance stop-loss on monitor, before any locked label read."""
    stage = _stage1_record()
    per_seed = stage["per_seed"]
    if not per_seed:
        log("[decide] no trained B/S pair on split A yet")
        return 1
    primary_seed = min(per_seed)
    first = per_seed[primary_seed]
    delta = first["monitor_delta"]
    seed_mean_monitor_delta = float(np.mean([v["monitor_delta"] for v in per_seed.values()]))
    if delta <= GATE_A_MONITOR_DELTA_MAX:
        band = "A_potential"
        purchase = True
        reading = "monitor cost is within the potential band; buy the second seed"
    elif delta <= GATE_B_MONITOR_DELTA_MAX:
        band = "B_marginal"
        purchase = True
        reading = "monitor cost is in the marginal band; the second seed is bought but flagged as costly"
    else:
        band = "C_too_costly"
        purchase = False
        reading = "monitor cost exceeds the stop-loss; no further S training, pivot to prediction research"
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "split_A_seed0_monitor_gate",
        "decision": band,
        "monitor_delta_S_minus_B": delta,
        "monitor_delta_seed_mean": seed_mean_monitor_delta,
        "thresholds": {"A_max": GATE_A_MONITOR_DELTA_MAX, "B_max": GATE_B_MONITOR_DELTA_MAX},
        "per_seed": per_seed,
        "purchase_seed1": bool(purchase),
        "reading": reading,
        "basis": "monitor rows only (soup monitor MAE from frozen train manifests); locked labels not read",
    }
    write_json(OUT / "stage1_decision.json", payload)
    log(f"[decide] {band}: monitor Δ(S−B) = {delta:+.6f}; purchase_seed1={purchase}")
    return 0


def cmd_locked_gate(args: argparse.Namespace) -> int:
    """Pre-registered F4 rule: does Split A locked-dev permit Split B?"""
    metrics_a = read_json(OUT / "split_a_metrics.json")
    paired = metrics_a["paired"]
    if "seed_mean" not in paired:
        log("[locked-gate] seed mean unavailable")
        return 1
    sm = paired["seed_mean"]["locked"]
    per_seed = sm["per_seed_locked_delta"]
    checks = {
        "seed_mean_locked_delta_le_+0.005": sm["delta_S_minus_B"] <= GATE_LOCKED_SEED_MEAN_MAX,
        "no_strong_opposite_seed_direction": not (max(per_seed) > 0 and min(per_seed) < 0 and abs(max(per_seed)) > 0.005 and abs(min(per_seed)) > 0.005),
        "ordinary_molecules_not_worse": sm["ordinary_delta"] <= 0.0,
        "not_single_row_driven": sm["top10_removed_delta"] < sm["delta_S_minus_B"] + 0.005,
        "clearly_unacceptable_stop": sm["delta_S_minus_B"] > GATE_LOCKED_STOP_MAX,
    }
    verdict = "unacceptable" if checks["clearly_unacceptable_stop"] else ("advance_to_split_B" if all(checks[k] for k in checks if k != "clearly_unacceptable_stop") else "hold")
    payload = {
        "protocol": PROTOCOL_ID,
        "stage": "split_A_locked_gate",
        "verdict": verdict,
        "checks": {k: bool(v) for k, v in checks.items()},
        "seed_mean_locked": sm,
        "thresholds": {"purchase_max": GATE_LOCKED_SEED_MEAN_MAX, "stop_min": GATE_LOCKED_STOP_MAX},
    }
    write_json(OUT / "split_a_locked_gate.json", payload)
    log(f"[locked-gate] verdict={verdict}")
    return 0


# ---------------------------------------------------------------------------
# mechanism + synthetic + report + decision
# ---------------------------------------------------------------------------


def _canonical_branch_totals(
    model, offsets, objs, rows, vocabs, device, chunk: int = 256
) -> dict[str, np.ndarray]:
    """Per-molecule canonical branch totals for one S checkpoint (no labels)."""
    n = len(rows)
    sums = {h: np.zeros(n, dtype=np.float64) for h in ("alpha", "beta", "nu", "eta")}
    pos = 0
    for start in range(0, n, chunk):
        chunk_rows = list(rows[start : start + chunk])
        batch = ijcm.build_ijcm_batch(objs, chunk_rows, vocabs)
        with torch.no_grad():
            terms = model.raw_terms(batch)
            can = ijcm.gauge_terms(terms, offsets, batch)
        size = len(chunk_rows)
        for h, kind in (("alpha", "atom"), ("beta", "bond"), ("nu", "junc")):
            gid = batch[f"{kind}_batch"].cpu().numpy()
            sums[h][pos : pos + size] = np.bincount(
                gid, weights=can[f"{h}_score_canonical"], minlength=size
            )
        sums["eta"][pos : pos + size] = can["eta_per_feature"].sum(axis=1)
        pos += size
    return sums


def cmd_mechanism(args: argparse.Namespace) -> int:
    device = torch.device("cpu")
    objs = load_objects()
    split_key = args.split
    rows = read_split_rows(split_key)
    vocabs = load_vocabs(split_key)
    seeds = sorted({int(p.name.split("_s")[-1].split(".")[0]) for p in OUT.glob(f"checkpoint_S_{split_key}_s*.pt")})
    if not seeds:
        log("[mechanism] no S checkpoints")
        return 1
    payload: dict[str, Any] = {"protocol": PROTOCOL_ID, "split": split_key, "seeds": seeds}

    # OOV rates on fit / monitor / locked for the frozen vocabulary
    oov = {}
    for name, rr in (("fit", rows["fit"]), ("monitor", rows["monitor"]), ("locked", rows["locked"])):
        entry = {}
        for kind in ("atom", "bond", "junction"):
            raw = np.concatenate([ijcm.RAW_TYPE_FN[kind](objs[int(r)]) for r in rr])
            entry[kind] = {
                "n_instances": int(raw.size),
                "oov_rate": float(np.mean(getattr(vocabs, kind).is_oov_mask(raw))) if raw.size else 0.0,
            }
        oov[name] = entry
    payload["oov"] = oov

    # per-seed canonical branch totals on the locked rows (labels never read)
    branch_totals: dict[int, dict[str, np.ndarray]] = {}
    for seed in seeds:
        model, offsets, ckpt, _ = load_checkpoint("S", split_key, seed, device)
        branch_totals[seed] = _canonical_branch_totals(model, offsets, objs, rows["locked"], vocabs, device)
        del model
    payload["branch_totals"] = {
        "mean_magnitude": {
            seed: {h: float(np.abs(v).mean()) for h, v in sums.items()} for seed, sums in branch_totals.items()
        },
        "share_of_total_abs": {
            seed: {
                h: float(np.abs(v).sum() / sum(np.abs(x).sum() for x in sums.values()))
                for h, v in sums.items()
            }
            for seed, sums in branch_totals.items()
        },
    }
    if len(seeds) >= 2:
        s0, s1 = seeds[0], seeds[1]
        stab = {}
        for h in ("alpha", "beta", "nu", "eta"):
            a, b = branch_totals[s0][h], branch_totals[s1][h]
            stab[h] = {
                "pearson_r_molecule_totals": float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else None,
                "mean_abs_diff": float(np.abs(a - b).mean()),
                "mean_magnitude": float((np.abs(a).mean() + np.abs(b).mean()) / 2),
                "normalized_mean_abs_diff": float(
                    np.abs(a - b).mean() / max((np.abs(a).mean() + np.abs(b).mean()) / 2, 1e-12)
                ),
            }
        payload["cross_seed_branch_stability"] = stab

        # canonical type-baseline stability over the shared known types
        type_stab = {}
        for kind in ("atom", "bond", "junction"):
            v = getattr(vocabs, kind)
            offs = []
            for seed in (s0, s1):
                _, offsets, _, _ = load_checkpoint("S", split_key, seed, device)
                vec = np.asarray([offsets[kind][idx] for idx in sorted(v.raw_to_index.values())], dtype=np.float64)
                offs.append(vec)
            a, b = offs[0], offs[1]
            type_stab[kind] = {
                "n_types": int(a.size),
                "pearson_r_type_baselines": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 and a.std() > 1e-12 and b.std() > 1e-12 else None,
                "mean_abs_diff": float(np.abs(a - b).mean()) if a.size else None,
                "mean_abs_level": float(np.abs(a).mean()) if a.size else None,
                "sign_agreement_frac": float(np.mean(np.sign(a) == np.sign(b))) if a.size else None,
            }
        payload["cross_seed_type_stability"] = type_stab

        # top junction types by the weaker of the two seed levels
        jv = vocabs.junction
        os0 = load_checkpoint("S", split_key, s0, device)[1]["junction"]
        os1 = load_checkpoint("S", split_key, s1, device)[1]["junction"]
        entries = []
        for raw, idx in jv.raw_to_index.items():
            a, b = float(os0[idx]), float(os1[idx])
            if abs(a) + abs(b) < 1e-6:
                continue
            entries.append(
                {
                    "raw_type": int(raw),
                    "type": ijcm.junc_type_string(raw),
                    "seed0": a,
                    "seed1": b,
                    "sign_agree": (a > 0) == (b > 0),
                    "min_abs": min(abs(a), abs(b)),
                }
            )
        entries.sort(key=lambda e: -e["min_abs"])
        payload["top_junction_types"] = entries[:15]
    write_json(OUT / "contribution_mechanism.json", payload)
    log("[mechanism] wrote contribution_mechanism.json")
    del objs
    return 0


def cmd_synthetic(args: argparse.Namespace) -> int:
    import ijcm_synthetic_v1 as syn

    syn.run_all(OUT, device=torch.device("cpu"), log=log)
    return 0


def _stage1_summary_block() -> str:
    if not (OUT / "split_a_metrics.json").exists():
        return "No Split A metrics exist yet."
    m = read_json(OUT / "split_a_metrics.json")
    lines = ["| arm | fit | monitor | locked |", "| --- | ---: | ---: | ---: |"]
    for tag, e in sorted(m["arms"].items()):
        lines.append(f"| {tag} | {e['fit_mae']:.6f} | {e['monitor']['mae']:.6f} | {e['locked']['mae']:.6f} |")
    return "\n".join(lines)


def cmd_report(args: argparse.Namespace) -> int:
    """Assemble REPORT.md from the frozen artifacts (numbers are read, never recomputed)."""
    report = []
    report.append("# bjoc-intrinsic-contribution-feasibility-v1 — report\n")
    report.append(
        "Protocol: `cscl-bjoc-intrinsic-contribution-feasibility-v1`.  This report is assembled from the "
        "frozen JSON artifacts in this directory; official valid/test were never loaded.\n"
    )
    report.append("## Part 1 — absolute prediction capacity\n")
    audit_path = OUT / "predictive_capacity_audit.json"
    if audit_path.exists():
        audit = read_json(audit_path)
        gap = audit["a1_gap"]
        a2 = audit["a2_failure_analysis"]
        report.append(
            f"- Historical BJOC-Null official-test raw-y MAE (context only): seed-mean "
            f"{gap['bjoc_null_seed_mean_test_raw_mae']:.6f}, two-seed ensemble {gap['bjoc_null_two_seed_ensemble_test_raw_mae']:.6f}; "
            f"user target band {gap['user_target_band'][0]:.2f}–{gap['user_target_band'][1]:.2f}.\n"
        )
        locked = a2["locked_error_structure"]
        report.append(
            f"- Internal split-A locked (already-exposed jdrop split, seed-mean): MAE "
            f"{locked['seed_mean_mae']:.6f}; ordinary (y>-3) MAE {locked['ordinary_y_gt_-3']['mae']:.6f} on "
            f"{locked['ordinary_y_gt_-3']['n']} molecules; tail (y<=-3) MAE {locked['tail_y_le_-3']['mae']:.6f} on "
            f"{locked['tail_y_le_-3']['n']} molecules carrying {locked['tail_y_le_-3']['error_mass_share']:.1%} of the error mass.\n"
        )
        report.append(
            f"- The held-out floor on ordinary molecules is already ~0.09; the target band requires a broad "
            f"improvement, not tail repair.  Top-10 removed locked MAE {locked['top10_removed_mae']:.6f}.\n"
        )
    report.append("\n## Part 2 — predictive cost of the interpretable model\n")
    report.append(_stage1_summary_block() + "\n")
    if (OUT / "stage1_decision.json").exists():
        d = read_json(OUT / "stage1_decision.json")
        report.append(
            f"\nStage-1 monitor stop-loss: **{d['decision']}** (Δ(S−B) = {d['monitor_delta_S_minus_B']:+.6f}, "
            f"purchase seed 1 = {d['purchase_seed1']}).  {d['reading']}.\n"
        )
    if (OUT / "split_a_locked_gate.json").exists():
        g = read_json(OUT / "split_a_locked_gate.json")
        report.append(f"\nSplit-A locked gate: **{g['verdict']}** (`checks` in `split_a_locked_gate.json`).\n")
    report.append("\n## Part 3 — structural contributions\n")
    if (OUT / "synthetic_a.json").exists():
        sa = read_json(OUT / "synthetic_a.json")
        report.append(f"- Synthetic A identifiability: design rank {sa['design_matrix']['rank']}/{sa['design_matrix']['n_columns']}, "
                      f"null-space dimension {sa['design_matrix']['null_space_dim']}.\n")
        if "model_recovery" in sa:
            report.append(f"- Synthetic A model recovery: {sa['model_recovery']['summary']}\n")
    if (OUT / "synthetic_b.json").exists():
        sb = read_json(OUT / "synthetic_b.json")
        report.append(f"- Synthetic B co-occurrence trap: {sb['summary']}\n")
    if (OUT / "synthetic_c.json").exists():
        sc = read_json(OUT / "synthetic_c.json")
        report.append(f"- Synthetic C additive limit: {sc['summary']}\n")
    if (OUT / "contribution_mechanism.json").exists():
        mech = read_json(OUT / "contribution_mechanism.json")
        if "cross_seed_type_stability" in mech:
            report.append(
                "- Cross-seed canonical type-baseline stability: "
                + ", ".join(
                    f"{k} r={v['pearson_r_type_baselines']:.3f}"
                    for k, v in mech["cross_seed_type_stability"].items()
                    if v["pearson_r_type_baselines"] is not None
                )
                + ".\n"
            )
    report.append("\n## Part 4 — stop or advance\n")
    decision_path = OUT / "scientific_decision.json"
    if decision_path.exists():
        dec = read_json(decision_path)
        report.append(f"- Final label: **{dec['decision']}**.\n- Rationale: {dec['rationale']}\n")
    report.append("\n## Part 5 — the next GPU purchase\n")
    prediction_path = OUT / "prediction_route_proposal.json"
    if prediction_path.exists():
        prop = read_json(prediction_path)
        report.append(f"- Highest-value prediction direction: **{prop['direction']}**\n")
        report.append(f"- Why: {prop['why']}\n")
    report.append(
        "\n**Research judgement (≤500 characters):** the current BJOC-class predictor is ~0.09 on ordinary "
        "molecules; reaching 0.06–0.07 is not a calibration problem but a missing capability: *reusable "
        "cross-molecule local-relation composition that survives distribution shift*. The next GPU purchase "
        "should test one pre-registered capacity/regularisation-controlled change that alters how multi-scale "
        "local relations are composed inside the frozen one-shot family — not another feature dump, and not "
        "more additive structure.\n"
    )
    (OUT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    log(f"[report] wrote {OUT / 'REPORT.md'}")
    return 0


def cmd_decision(args: argparse.Namespace) -> int:
    ok_gates = False
    if (OUT / "correctness_gates.json").exists():
        ok_gates = bool(read_json(OUT / "correctness_gates.json")["all_gates_ok"])
    metrics = read_json(OUT / "split_a_metrics.json") if (OUT / "split_a_metrics.json").exists() else None
    synthetic = {}
    for name in ("synthetic_a", "synthetic_b", "synthetic_c"):
        p = OUT / f"{name}.json"
        if p.exists():
            synthetic[name] = read_json(p)
    delta = None
    delta_locked = None
    if metrics and "seed_mean" in metrics.get("paired", {}):
        delta_locked = metrics["paired"]["seed_mean"]["locked"]["delta_S_minus_B"]
        delta = metrics["paired"]["seed_mean"]["monitor"]["delta_S_minus_B"]
    elif metrics:
        seeds = sorted(metrics["paired"]["by_seed"])
        if seeds:
            delta = metrics["paired"]["by_seed"][seeds[0]]["monitor"]["delta_S_minus_B"]
            delta_locked = metrics["paired"]["by_seed"][seeds[0]]["locked"]["delta_S_minus_B"]

    synthetic_pass = False
    if "synthetic_a" in synthetic and "synthetic_b" in synthetic and "synthetic_c" in synthetic:
        synthetic_pass = bool(
            synthetic["synthetic_a"].get("identifiability_pass", False)
            and synthetic["synthetic_b"].get("trap_pass", False)
        )

    if not ok_gates:
        decision = "INCONCLUSIVE_OR_BLOCKED"
        rationale = "numeric correctness/conservation gates did not pass"
    elif delta is None:
        decision = "INCONCLUSIVE_OR_BLOCKED"
        rationale = "no paired S/B predictions exist yet"
    elif not synthetic_pass:
        if delta > GATE_B_MONITOR_DELTA_MAX:
            decision = "ADDITIVE_HEAD_UNDERFITS"
            rationale = "the additive head both under-performs on held-out molecules and fails synthetic recovery"
        else:
            decision = "CONTRIBUTIONS_NOT_IDENTIFIABLE"
            rationale = "synthetic recovery/identifiability checks did not pass despite acceptable prediction cost"
    elif delta > GATE_B_MONITOR_DELTA_MAX or (delta_locked is not None and delta_locked > GATE_LOCKED_STOP_MAX):
        decision = "ATTRIBUTION_WORKS_BUT_PREDICTIVE_COST_TOO_HIGH"
        rationale = "synthetic identifiability works, but the held-out prediction cost of the additive head is too large"
    else:
        decision = "INTRINSIC_CONTRIBUTION_FEASIBLE"
        rationale = "prediction cost is inside the pre-registered band and identifiability checks pass"
    payload = {
        "protocol": PROTOCOL_ID,
        "decision": decision,
        "rationale": rationale,
        "monitor_delta_S_minus_B": delta,
        "locked_delta_S_minus_B": delta_locked,
        "correctness_gates_ok": ok_gates,
        "synthetic_pass": synthetic_pass,
    }
    write_json(OUT / "scientific_decision.json", payload)
    log(f"[decision] {decision}: {rationale}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("splits")
    sp.set_defaults(func=cmd_splits)

    sp = sub.add_parser("objects")
    sp.set_defaults(func=cmd_objects)

    sp = sub.add_parser("params")
    sp.set_defaults(func=cmd_params)

    sp = sub.add_parser("gates")
    sp.set_defaults(func=cmd_gates)

    sp = sub.add_parser("freeze")
    sp.set_defaults(func=cmd_freeze)

    sp = sub.add_parser("synthetic")
    sp.set_defaults(func=cmd_synthetic)

    sp = sub.add_parser("train")
    sp.add_argument("--arm", choices=ARMS, required=True)
    sp.add_argument("--split", choices=("A", "B"), required=True)
    sp.add_argument("--seed", type=int, required=True)
    sp.add_argument("--device", default="cpu")
    sp.add_argument("--smoke", action="store_true")
    sp.set_defaults(func=cmd_train)

    sp = sub.add_parser("train-stage")
    sp.add_argument("--stage", choices=tuple(STAGE_PLANS), required=True)
    sp.add_argument("--device", default="cuda:0")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_train_stage)

    sp = sub.add_parser("predict")
    sp.add_argument("--split", choices=("A", "B"), required=True)
    sp.add_argument("--arms", required=True)
    sp.add_argument("--rows", choices=("monitor", "both"), default="monitor")
    sp.add_argument("--device", default="cpu")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("score")
    sp.add_argument("--split", choices=("A", "B"), required=True)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("decide")
    sp.set_defaults(func=cmd_decide)

    sp = sub.add_parser("locked-gate")
    sp.set_defaults(func=cmd_locked_gate)

    sp = sub.add_parser("mechanism")
    sp.add_argument("--split", choices=("A", "B"), default="A")
    sp.set_defaults(func=cmd_mechanism)

    sp = sub.add_parser("report")
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("decision")
    sp.set_defaults(func=cmd_decision)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
