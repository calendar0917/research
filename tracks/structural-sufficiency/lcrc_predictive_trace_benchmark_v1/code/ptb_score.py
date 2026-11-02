"""One-shot Test scoring, only after ``checkpoint_manifest.json`` froze the dataset.

For every frozen ``(dataset, model, seed)`` unit this script
1. re-verifies the checkpoint SHA-256 against the frozen manifest,
2. loads the Test structure cache *and* its labels (the only place in the round
   where molecule Test targets are decoded),
3. scores the single frozen checkpoint once and writes ``test_eval.json``,
4. appends the access to ``analysis/test_access_ledger.json``.

All molecule Test results are ``EXPOSED_TEST_EXPLORATORY``: earlier rounds already
read these Test labels, so this is not a fresh blind test.  ``ogbg-molhiv`` also
reports the official OGB evaluator ROC-AUC.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

import ptb_common as C
import ptb_data as D
import ptb_model as M

import mb2_metrics as MET


def _ledger_append(event: dict[str, Any]) -> None:
    path = C.ANALYSIS / "test_access_ledger.json"
    payload = C.read_json(path) if path.exists() else {"schema": "lcrc-ptb-test-access-v1", "events": []}
    payload["events"].append(event)
    C.write_json(path, payload)


def score_unit(dataset: str, model: str, seed: int, manifest: dict[str, Any]) -> dict[str, Any]:
    entry = next(
        (u for u in manifest["datasets"][dataset]["units"] if u["variant"] == model and u["seed"] == seed),
        None,
    )
    if entry is None:
        raise RuntimeError(f"{dataset}/{model}/s{seed} is not in the frozen manifest")
    run_dir = C.mol_run_dir(dataset, model, seed)
    out_path = run_dir / "test_eval.json"
    if out_path.exists():
        raise RuntimeError(f"{C.track_rel(out_path)} exists; Test is scored exactly once")
    ckpt_path = run_dir / "checkpoint.pt"
    sha = C.sha256_file(ckpt_path)
    if sha != entry["checkpoint_sha256"]:
        raise RuntimeError(f"{dataset}/{model}/s{seed}: checkpoint changed after freezing")

    view = D.molecule_view(dataset)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_obj = M.load_checkpoint_model(
        view.spec, view.train_norm, model, ckpt["state_dict"], "cuda", n_tasks=int(ckpt["n_tasks"])
    )
    test_cache, y_test, test_rows = D.molecule_test(dataset)
    pred = D_module_predict(model_obj, test_cache, torch.device("cuda"))
    metrics = MET.task_metrics(view.task, pred, y_test, view.target_norm, view.task_names)
    official = None
    if dataset == "ogbg-molhiv":
        official = MET.ogb_evaluator_roc_auc(np.asarray(y_test).reshape(-1), np.asarray(pred).reshape(-1))

    eval_record = {
        "schema": "lcrc-ptb-test-eval-v1",
        "round_id": C.ROUND_ID,
        "dataset": dataset,
        "variant": model,
        "seed": int(seed),
        "task": view.task,
        "n_test": int(len(test_rows)),
        "test_rows_sha256": C.index_sha256(test_rows),
        "checkpoint_sha256": sha,
        "manifest_sha256": manifest["datasets"][dataset]["manifest_sha256"],
        "metrics": metrics,
        "official_ogb_roc_auc": official,
        "result_label": C.RESULT_LABEL_MOL,
        "test_exposure": "EXPOSED_TEST_EXPLORATORY",
        "scored_utc": C.utcnow(),
        "selection_used_test": False,
    }
    C.write_json(out_path, eval_record)
    np.savez(
        run_dir / "test_predictions.npz",
        rows=np.asarray(test_rows, dtype=np.int64),
        pred=pred,
        y_true=np.asarray(y_test),
    )
    sel = C.read_json(run_dir / "selection.json")
    sel["test_evaluated"] = True
    sel["test_evaluated_utc"] = eval_record["scored_utc"]
    sel["test_eval_path"] = C.track_rel(out_path)
    C.write_json(run_dir / "selection.json", sel)
    _ledger_append(
        {
            "utc": eval_record["scored_utc"],
            "dataset": dataset,
            "variant": model,
            "seed": int(seed),
            "n_test": int(len(test_rows)),
            "checkpoint_sha256": sha,
            "manifest_sha256": manifest["datasets"][dataset]["manifest_sha256"],
            "frozen_before_access": True,
            "result_label": C.RESULT_LABEL_MOL,
        }
    )
    return eval_record


def D_module_predict(model, cache, device) -> np.ndarray:
    import lcrc_data as PREP_DATA

    model.eval()
    n = int(cache["n_molecules"])
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n, 256):
            rows = list(range(start, min(start + 256, n)))
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in PREP_DATA.collate(cache, rows).items()}
            pred = model(batch)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            out.append(pred.detach().float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--all-frozen", action="store_true")
    args = ap.parse_args(argv)
    manifest = C.read_json(C.ANALYSIS / "checkpoint_manifest.json")
    datasets = [args.dataset] if args.dataset else list(manifest.get("frozen_datasets", []))
    if args.all_frozen and not datasets:
        C.log("no frozen dataset yet")
        return 0
    for dataset in datasets:
        if not manifest["datasets"][dataset]["frozen"]:
            C.log(f"{dataset}: not frozen, skipping Test scoring")
            continue
        for model in C.MODELS:
            for seed in C.MOLECULE_SEEDS:
                if (C.mol_run_dir(dataset, model, seed) / "test_eval.json").exists():
                    # already scored exactly once (a dataset frozen in an earlier pass)
                    C.log(f"{dataset}/{model}/s{seed}: already scored, not re-reading Test")
                    continue
                rec = score_unit(dataset, model, seed, manifest)
                C.log(
                    f"{dataset}/{model}/s{seed}: {MET.primary_name(rec['task'])} "
                    f"= {rec['metrics'][MET.primary_name(rec['task'])]:.6f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
