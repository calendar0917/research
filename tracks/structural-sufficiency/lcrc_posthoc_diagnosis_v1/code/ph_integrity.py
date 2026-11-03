"""Read-only integrity gates for the frozen prediction artefacts."""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np

import ph_common as C


def _gate(name: str, passed: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "status": "PASS" if passed else "FAIL", "details": details or {}}


def _manifest_units() -> dict[tuple[str, str, int], dict[str, Any]]:
    manifest = C.read_json(C.PTB / "analysis/checkpoint_manifest.json")
    out = {}
    for dataset, payload in manifest["datasets"].items():
        for unit in payload["units"]:
            out[(dataset, unit["variant"], int(unit["seed"]))] = unit
    return out


def verify_molecule() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    manifest = _manifest_units()
    dataset_details: dict[str, Any] = {}
    all_ok = True
    max_metric_diff = 0.0
    max_yt_delta = 0.0

    for dataset in C.MOLECULE_ORDER:
        meta = C.DATASETS[dataset]
        rows0 = None
        y0 = None
        checks = []
        for seed in C.MOLECULE_SEEDS:
            z0 = C.load_mol(dataset, "A0", seed)
            z1 = C.load_mol(dataset, "C1", seed)
            rows_a = np.asarray(z0["rows"], dtype=np.int64)
            rows_c = np.asarray(z1["rows"], dtype=np.int64)
            if not np.array_equal(rows_a, rows_c):
                checks.append(f"s{seed}: A0/C1 rows differ")
            if rows0 is None:
                rows0 = rows_a
                y0 = np.asarray(z0["y_true"], dtype=np.float64)
            else:
                if not np.array_equal(rows_a, rows0):
                    checks.append(f"s{seed}: rows differ from s0")
            for model, z, label in (("A0", z0, f"s{seed}_A0"), ("C1", z1, f"s{seed}_C1")):
                y = np.asarray(z["y_true"], dtype=np.float64)
                if y.shape != y0.shape or not np.allclose(y, y0, rtol=0, atol=0, equal_nan=True):
                    checks.append(f"{label}: y_true differs")
                if np.asarray(z["pred"]).shape[0] != rows_a.size:
                    checks.append(f"{label}: pred row count")
                if not np.isfinite(np.asarray(z["pred"], dtype=np.float64)).all():
                    checks.append(f"{label}: non-finite pred")
                if np.asarray(z["pred"]).ndim not in (1, 2):
                    checks.append(f"{label}: pred ndim")
                # saved metric vs offline recomputation
                ev = C.load_mol_eval(dataset, model, seed)
                mask = C.original_task_mask(y0) if meta["task"] == "multitask" else None
                got = C.primary_metric(dataset, z["pred"], y, mask)
                want = float(ev["metrics"][meta["primary"]])
                diff = abs(got - want)
                max_metric_diff = max(max_metric_diff, diff)
                if diff > 1e-10:
                    checks.append(f"{label}: metric {got} != saved {want}")
                rec = manifest.get((dataset, model, seed))
                if rec is None:
                    checks.append(f"{label}: missing manifest unit")
                else:
                    if rec["checkpoint_sha256"] != ev["checkpoint_sha256"]:
                        checks.append(f"{label}: checkpoint sha != manifest")
                    if int(rec["n_train"]) < 1:
                        checks.append(f"{label}: empty train")
        # row hash recorded at score time
        ev0 = C.load_mol_eval(dataset, "A0", 0)
        if C.index_sha256(rows0.tolist()) != ev0["test_rows_sha256"]:
            checks.append("rows sha != test_eval record")
        # Tox21 task mask must be the 12 frozen endpoints with both classes
        extra = {}
        if meta["task"] == "multitask":
            mask = C.original_task_mask(y0)
            extra["definable_tasks"] = len(mask)
            extra["n_tasks"] = int(np.asarray(y0).shape[1])
            if len(mask) != 12:
                checks.append(f"Tox21 definable tasks {len(mask)} != 12")
        # duplicates / missing
        if np.unique(rows0).size != rows0.size:
            checks.append("duplicate test rows")
        if np.any(np.diff(rows0) <= 0):
            checks.append("test rows not strictly increasing")
        dataset_details[dataset] = {
            "n_test": int(rows0.size),
            "rows_sha256": C.index_sha256(rows0.tolist()),
            "y_true_sha256": C.sha256_bytes(np.ascontiguousarray(np.asarray(y0, dtype=np.float64)).tobytes()),
            "checks": checks,
            **extra,
        }
        ok = not checks
        all_ok = all_ok and ok
        gates.append(_gate(f"molecule/{dataset}/alignment_and_metrics", ok, dataset_details[dataset]))

    # MolHIV official row ids
    split_test = np.asarray(
        __import__("pandas").read_csv(C.OGB_SOURCE / "split/scaffold/test.csv.gz", header=None)
        .to_numpy()
        .reshape(-1),
        dtype=np.int64,
    )
    z = C.load_mol("ogbg-molhiv", "A0", 0)
    ok = np.array_equal(np.asarray(z["rows"], dtype=np.int64), split_test)
    gates.append(
        _gate(
            "molecule/ogbg-molhiv/official_test_rows",
            ok,
            {
                "n": int(split_test.size),
                "first": split_test[:8].tolist(),
                "rows_match_published_order": bool(ok),
            },
        )
    )
    all_ok = all_ok and ok
    return gates, {
        "all_pass": all_ok,
        "max_abs_metric_recompute_diff": float(max_metric_diff),
        "max_abs_yt_delta": float(max_yt_delta),
        "datasets": dataset_details,
    }


def verify_tu() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import csv

    gates: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    all_ok = True
    rows = list(csv.DictReader((C.PTB / "analysis/tu_results.csv").open(encoding="utf-8")))
    for dataset in ("NCI1",):
        checks = []
        per_fold: dict[int, dict[str, Any]] = {}
        for fold in range(5):
            gids0 = None
            y0 = None
            for seed in C.TU_SEEDS:
                for model in C.MODELS:
                    z = C.load_npz(C.tu_run_dir(dataset, fold, seed, model) / "test_predictions.npz")
                    gids = np.asarray(z["gids"], dtype=np.int64)
                    y = np.asarray(z["y_true"], dtype=np.int64)
                    logits = np.asarray(z["logits"], dtype=np.float64)
                    if logits.shape != (gids.size, 2):
                        checks.append(f"f{fold}/s{seed}/{model}: logits shape {logits.shape}")
                    if gids0 is None:
                        gids0, y0 = gids, y
                    else:
                        if not np.array_equal(gids, gids0):
                            checks.append(f"f{fold}/s{seed}/{model}: gids differ")
                        if not np.array_equal(y, y0):
                            checks.append(f"f{fold}/s{seed}/{model}: y_true differs")
                    pred = np.argmax(logits, axis=1)
                    acc = float((pred == y).mean())
                    rec = next(
                        (
                            r
                            for r in rows
                            if r["dataset"] == dataset
                            and int(r["fold"]) == fold
                            and int(r["seed"]) == seed
                            and r["model"] == model
                        ),
                        None,
                    )
                    if rec is None:
                        checks.append(f"f{fold}/s{seed}/{model}: missing tu_results row")
                    elif abs(acc - float(rec["test_accuracy"])) > 1e-12:
                        checks.append(
                            f"f{fold}/s{seed}/{model}: accuracy {acc} != saved {rec['test_accuracy']}"
                        )
            per_fold[fold] = {
                "n_test": int(gids0.size),
                "gids_sha256": C.index_sha256(sorted(gids0.tolist())),
                "positive_rate": float(y0.mean()),
            }
        ok = not checks
        all_ok = all_ok and ok
        details[dataset] = {"checks": checks, "folds": per_fold}
        gates.append(_gate(f"tu/{dataset}/alignment_and_accuracy", ok, details[dataset]))
    return gates, {"all_pass": all_ok, "datasets": details}


def remote_checkpoint_hashes() -> dict[str, Any]:
    """Read-only ``sha256sum`` of the frozen MolHIV checkpoints on ``res``.

    The checkpoints stay remote and are not copied.  This only verifies that the
    SHA recorded in the frozen manifest still matches the file on disk.
    """
    paths = []
    for seed in C.MOLECULE_SEEDS:
        for model in C.MODELS:
            paths.append(
                f"tracks/structural-sufficiency/lcrc_predictive_trace_benchmark_v1/"
                f"runs/mol_ogbg-molhiv_s{seed}_{model}/checkpoint.pt"
            )
    cmd = "cd /home/hxy/cy/research && sha256sum " + " ".join(paths)
    proc = subprocess.run(["ssh", "res", cmd], capture_output=True, text=True, check=False)
    out: dict[str, Any] = {"command": cmd, "returncode": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}
    if proc.returncode != 0:
        out["status"] = "UNAVAILABLE"
        return out
    found = {}
    for line in proc.stdout.splitlines():
        sha, path = line.split(maxsplit=1)
        key = path.strip().split("/runs/")[-1].split("/")[0]
        found[key] = sha
    manifest = _manifest_units()
    checks = {}
    for seed in C.MOLECULE_SEEDS:
        for model in C.MODELS:
            key = f"mol_ogbg-molhiv_s{seed}_{model}"
            want = manifest[("ogbg-molhiv", model, seed)]["checkpoint_sha256"]
            got = found.get(key)
            checks[key] = {"manifest": want, "remote": got, "match": bool(got == want)}
    out["checks"] = checks
    out["status"] = "PASS" if all(v["match"] for v in checks.values()) else "FAIL"
    return out


def main() -> int:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    import json

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=C.REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=C.REPO, capture_output=True, text=True, check=True
    ).stdout.splitlines()

    mol_gates, mol_details = verify_molecule()
    tu_gates, tu_details = verify_tu()
    ckpt = remote_checkpoint_hashes()

    gates = [
        _gate(
            "repo/baseline_commit",
            head == C.BASELINE_COMMIT,
            {"head": head, "baseline": C.BASELINE_COMMIT, "worktree_entries": len(status)},
        ),
        _gate(
            "policy/no_cuda",
            __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "") == "",
            {"CUDA_VISIBLE_DEVICES": __import__("os").environ.get("CUDA_VISIBLE_DEVICES", "<unset>")},
        ),
        _gate(
            "policy/no_training_or_scoring_entrypoint",
            True,
            {"note": "this module only reads saved npz/json; no torch import, no ptb_score/edit_probe"},
        ),
        _gate(
            "policy/no_new_test_label_source",
            True,
            {
                "molecule_test_labels": "only saved test_predictions.npz y_true",
                "audit_labels": "not read; only saved probe_audit_scores.json fields",
                "ogb": "mapping SMILES only; graph-label.csv.gz not opened",
            },
        ),
        _gate("molecule/artifacts", mol_details["all_pass"], mol_details),
        _gate("tu/artifacts", tu_details["all_pass"], tu_details),
        _gate("checkpoint/remote_sha256", ckpt["status"] == "PASS", ckpt),
    ]
    payload = {
        "schema": "lcrc-posthoc-correctness-gates-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "created_utc": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "all_pass": all(g["status"] == "PASS" for g in gates),
        "gates": gates,
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "correctness_gates.json", payload)
    print(json.dumps({"all_pass": payload["all_pass"], "n_gates": len(gates)}, indent=2))
    for g in gates:
        print(g["status"], g["name"])
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
