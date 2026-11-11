"""Read-only error diagnostics on saved checkpoints (no training, no Test).

For each available checkpoint (historical W2 + the new W1/W3 best-valid
checkpoints) the full official Validation split is re-scored and stratified:

* exact-SMILES overlap between Validation and the official Train split,
  Validation MAE and bias for overlapping vs non-overlapping molecules,
* error by heavy-atom count bin,
* error by true-gap bin,
* error by presence of an out-of-vocabulary atom (UNK index 15).

Nothing here selects, trains or tunes anything; the checkpoints are exactly the
ones the training loop wrote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ew_common as C  # noqa: E402
import ew_data as DATA  # noqa: E402
import ew_model as M  # noqa: E402

import torch  # noqa: E402

UNK_ATOM_INDEX = 15
ATOM_BINS = ((0, 20), (21, 25), (26, 30), (31, 35), (36, 40), (41, 50), (51, 10_000))
GAP_BINS = ((0.0, 4.5), (4.5, 5.0), (5.0, 5.5), (5.5, 6.0), (6.0, 100.0))


def _stratum(pred: np.ndarray, y: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    err = pred[mask] - y[mask]
    return {
        "n": n,
        "mae_eV": float(np.mean(np.abs(err))),
        "bias_eV": float(np.mean(err)),
        "rmse_eV": float(np.sqrt(np.mean(err**2))),
    }


def _bootstrap_diff(
    pred: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, *, reps: int = 500, seed: int = 0
) -> dict[str, Any] | None:
    if a.sum() < 20 or b.sum() < 20:
        return None
    ea = np.abs(pred[a] - y[a])
    eb = np.abs(pred[b] - y[b])
    rng = np.random.default_rng(seed)
    diffs = np.empty(reps, dtype=np.float64)
    for i in range(reps):
        diffs[i] = float(ea[rng.integers(0, ea.size, ea.size)].mean() - eb[rng.integers(0, eb.size, eb.size)].mean())
    return {
        "mae_duplicate_minus_nonduplicate_eV": float(ea.mean() - eb.mean()),
        "ci95_low_eV": float(np.percentile(diffs, 2.5)),
        "ci95_high_eV": float(np.percentile(diffs, 97.5)),
        "reps": int(reps),
        "resampling": "independent percentile bootstrap over molecules within each group",
    }


def run_one(
    name: str,
    checkpoint: Path,
    task: Mapping[str, Any],
    device: torch.device,
    *,
    train_smiles: set[str] | None,
    subset_smiles: set[str] | None,
    batch_size: int,
) -> dict[str, Any]:
    if not Path(checkpoint).exists():
        return {"model": name, "available": False, "checkpoint": C.track_rel(checkpoint)}
    scale = {"W1": 1, "W2": 2, "W3": 3}[name]
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model, audit = M.build_model(task["spec"], task["norm"], scale=scale, seed=int(task["seed"]), device=device)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    report, pred_z = C.PQ_TRAIN.evaluate_split(model, task, device, which="valid", batch_size=int(batch_size))
    y = np.asarray(task["valid_y_eV"], dtype=np.float64)
    pred = np.asarray(C.PQ_DATA.from_z(pred_z, task["label_norm"]), dtype=np.float64)
    vcache = task["valid_cache"]
    n_atoms = np.diff(np.asarray(vcache["atom_ptr"], dtype=np.int64))
    smiles = DATA.smiles_of(vcache, task["valid_rows"])
    unk = np.zeros(len(y), dtype=bool)
    atom_type = np.asarray(vcache["atom_type"])
    atom_ptr = np.asarray(vcache["atom_ptr"], dtype=np.int64)
    unk_atoms = np.flatnonzero(atom_type == UNK_ATOM_INDEX)
    if unk_atoms.size:
        unk[np.unique(np.searchsorted(atom_ptr, unk_atoms, side="right") - 1)] = True

    out: dict[str, Any] = {
        "model": name,
        "available": True,
        "checkpoint": C.track_rel(checkpoint),
        "checkpoint_sha256": C.sha256_file(checkpoint),
        "checkpoint_bytes": int(Path(checkpoint).stat().st_size),
        "epoch": int(payload.get("epoch", -1)),
        "parameters": int(audit["total_parameters"]),
        "n_valid": int(y.size),
        "mae_eV": float(np.mean(np.abs(pred - y))),
        "report_mae_eV": float(report["mae_eV"]),
        "report_pred_sha256": report["pred_sha256"],
        "numpy_mae_eV": float(report["mae_eV_numpy"]),
        "crosscheck_delta_eV": float(report["crosscheck_delta_eV"]),
        "n_atoms_mean": float(n_atoms.mean()),
        "n_atoms_max": int(n_atoms.max()),
    }
    if train_smiles is not None:
        dup = np.asarray([s in train_smiles for s in smiles])
        non = ~dup
        out["official_train_smiles"] = {
            "n_train_smiles": len(train_smiles),
            "n_valid_duplicate": int(dup.sum()),
            "rate": float(dup.mean()),
            "duplicate": _stratum(pred, y, dup),
            "non_duplicate": _stratum(pred, y, non),
            "bootstrap": _bootstrap_diff(pred, y, dup, non),
        }
    if subset_smiles is not None:
        dup = np.asarray([s in subset_smiles for s in smiles])
        out["one_million_subset_smiles"] = {
            "n_subset_smiles": len(subset_smiles),
            "n_valid_duplicate": int(dup.sum()),
            "rate": float(dup.mean()),
            "duplicate": _stratum(pred, y, dup),
            "non_duplicate": _stratum(pred, y, ~dup),
        }
    within = {}
    seen: dict[str, int] = {}
    for s in smiles:
        seen[s] = seen.get(s, 0) + 1
    out["within_valid_duplicate_strings"] = int(sum(1 for s, c in seen.items() if c > 1))
    out["within_valid_duplicate_molecules"] = int(sum(c - 1 for s, c in seen.items() if c > 1))

    out["by_atom_count"] = []
    for lo, hi in ATOM_BINS:
        mask = (n_atoms >= lo) & (n_atoms <= hi)
        row = {"lo": int(lo), "hi": None if hi >= 10_000 else int(hi), **_stratum(pred, y, mask)}
        out["by_atom_count"].append(row)
    out["by_gap"] = []
    for lo, hi in GAP_BINS:
        mask = (y >= lo) & (y < hi)
        out["by_gap"].append({"lo": float(lo), "hi": float(hi), **_stratum(pred, y, mask)})
    out["by_unk"] = {
        "with_unk": _stratum(pred, y, unk),
        "without_unk": _stratum(pred, y, ~unk),
        "n_unk_molecules": int(unk.sum()),
        "unk_rate": float(unk.mean()),
    }
    out["residual_quantiles_eV"] = {
        f"q{int(q * 100)}": float(np.percentile(pred - y, q * 100)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
    }
    out["worst_molecules"] = [
        {"row": int(i), "smiles": smiles[i], "y_eV": float(y[i]), "pred_eV": float(pred[i]), "err_eV": float(pred[i] - y[i])}
        for i in np.argsort(-np.abs(pred - y))[:20]
    ]
    return out


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def to_markdown(results: Mapping[str, Mapping[str, Any]], path: Path) -> None:
    lines = ["# Read-only checkpoint diagnostics", ""]
    for name in ("W1", "W2", "W3"):
        r = results.get(name)
        if not r or not r.get("available"):
            lines.append(f"## {name}\n\nnot available\n")
            continue
        lines += [
            f"## {name}",
            "",
            f"checkpoint `{r['checkpoint']}` (SHA256 `{r['checkpoint_sha256'][:16]}…`), epoch {r['epoch']}, "
            f"{r['parameters']:,} parameters, full Validation MAE {_fmt(r['mae_eV'], 9)} eV "
            f"(official {_fmt(r['report_mae_eV'], 9)}, NumPy {_fmt(r['numpy_mae_eV'], 9)})",
            "",
        ]
        train = r.get("official_train_smiles")
        if train:
            d, n = train["duplicate"], train["non_duplicate"]
            lines += [
                f"**Official-Train exact-SMILES overlap:** {train['n_valid_duplicate']:,} / {r['n_valid']:,} "
                f"({100 * train['rate']:.3f}%). MAE duplicate {_fmt(d.get('mae_eV'), 6)} eV (n={d.get('n', 0):,}), "
                f"non-duplicate {_fmt(n.get('mae_eV'), 6)} eV (n={n.get('n', 0):,}).",
                "",
            ]
            if train.get("bootstrap"):
                b = train["bootstrap"]
                lines += [
                    f"Bootstrap MAE difference (duplicate − non-duplicate) {b['mae_duplicate_minus_nonduplicate_eV']:+.6f} eV, "
                    f"95% CI [{b['ci95_low_eV']:+.6f}, {b['ci95_high_eV']:+.6f}].",
                    "",
                ]
        sub = r.get("one_million_subset_smiles")
        if sub:
            lines += [
                f"1M-subset overlap {sub['n_valid_duplicate']:,} ({100 * sub['rate']:.3f}%); "
                f"MAE dup {_fmt(sub['duplicate'].get('mae_eV'), 6)} vs non-dup {_fmt(sub['non_duplicate'].get('mae_eV'), 6)} eV.",
                "",
            ]
        lines += [
            "| atom-count bin | n | MAE (eV) | bias (eV) |",
            "|---|---:|---:|---:|",
        ]
        for row in r["by_atom_count"]:
            hi = row["hi"] if row["hi"] is not None else "+"
            lines.append(f"| {row['lo']}–{hi} | {row['n']:,} | {_fmt(row.get('mae_eV'), 6)} | {_fmt(row.get('bias_eV'), 6)} |")
        lines += ["", "| true-gap bin (eV) | n | MAE (eV) | bias (eV) |", "|---|---:|---:|---:|"]
        for row in r["by_gap"]:
            lines.append(
                f"| [{row['lo']:.1f}, {row['hi']:.1f}) | {row['n']:,} | {_fmt(row.get('mae_eV'), 6)} | {_fmt(row.get('bias_eV'), 6)} |"
            )
        u = r["by_unk"]
        lines += [
            "",
            f"UNK-containing molecules: {u['n_unk_molecules']:,} ({100 * u['unk_rate']:.3f}%); "
            f"MAE with UNK {_fmt(u['with_unk'].get('mae_eV'), 6)} eV, without {_fmt(u['without_unk'].get('mae_eV'), 6)} eV.",
            "",
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", action="append", default=[], metavar="MODEL=PATH")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--out", default=str(C.ANALYSIS / "diagnostics.json"))
    parser.add_argument("--skip-official-train", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    # the inherited frozen task (its run name is the historical one)
    task = C.PQ_DATA.load_task("PCQM_W2_1M", verify_hashes=False)
    wanted: dict[str, Path] = {
        "W2": C.OLD_ROUND / "runs" / C.HISTORICAL["run"] / "checkpoint_best_valid.pt",
        "W1": C.RUNS / "PCQM_W1_1M" / "checkpoint_best_valid.pt",
        "W3": C.RUNS / "PCQM_W3_1M" / "checkpoint_best_valid.pt",
    }
    for item in args.checkpoint:
        name, _, path = str(item).partition("=")
        if not path:
            raise SystemExit(f"--checkpoint expects MODEL=PATH, got {item!r}")
        wanted[name.upper()] = Path(path)
    train_smiles = None
    subset_smiles = None
    if not args.skip_official_train:
        try:
            train_smiles = set(DATA.official_train_smiles())
            C.log(f"[diagnostics] official Train SMILES loaded: {len(train_smiles):,} unique strings")
        except Exception as exc:  # pragma: no cover - dataset availability
            C.log(f"[diagnostics] official Train SMILES unavailable ({type(exc).__name__}: {exc}); continuing without it")
    subset_smiles = set(DATA.smiles_of(task["train_cache"], range(int(task["train_cache"]["n_molecules"]))))
    C.log(f"[diagnostics] 1M-subset SMILES: {len(subset_smiles):,} unique strings")
    results = {
        name: run_one(
            name,
            path,
            task,
            device,
            train_smiles=train_smiles,
            subset_smiles=subset_smiles,
            batch_size=int(args.batch_size),
        )
        for name, path in wanted.items()
    }
    payload = {
        "schema": "lcrc-pcqm4mv2-efficiency-width-diagnostics-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "device": str(device),
        "official_train_smiles_used": bool(train_smiles),
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    C.write_json(out, payload)
    to_markdown(results, out.parent / "diagnostics.md")
    C.log(f"[diagnostics] wrote {C.track_rel(out)} and diagnostics.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
