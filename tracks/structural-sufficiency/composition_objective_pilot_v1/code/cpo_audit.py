"""CPU audit of the existing cut-wise objective and pair-delta provenance.

Reads the existing best checkpoints read-only.  Never reads official Test labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cpo_common as C

RUNS = {
    "A_TRC_FACTOR_CUTRESP": {
        "family": "parent", "arm": "FACTOR",
        "run_dir": C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0",
        "role": "parent reuse",
    },
    "B_CSOI_ADD": {
        "family": "csoi", "arm": "CSOI_ADD",
        "run_dir": C.CSOI_ROUND / "results" / "CSOI_ADD_LIPO_R0",
        "role": "CSOI pilot",
    },
    "C_CSOI_GLOBAL": {
        "family": "csoi", "arm": "CSOI_GLOBAL",
        "run_dir": C.CSOI_ROUND / "results" / "CSOI_GLOBAL_LIPO_R0",
        "role": "CSOI pilot",
    },
    "D_CSOI_REL_EXTRA": {
        "family": "csoi", "arm": "CSOI_REL",
        "run_dir": C.CSOI_ROUND / "results" / "CSOI_REL_LIPO_R0",
        "role": "extra existing checkpoint, not used for decisions",
    },
}
SCOPE_NAMES = {
    "train_in_sample_pairs": "Train×Train (in-sample)",
    "train_valid_cross_pairs": "Train×Valid (one side out-of-sample)",
    "valid_only_pairs": "Valid×Valid (both out-of-sample, n≈15)",
}


def _scopes(pack: Mapping[str, Any]) -> dict[str, np.ndarray]:
    split = np.asarray(pack["molecule_split"], np.uint8)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    sa, sb = split[a], split[b]
    allowed = (sa != 2) & (sb != 2)
    return {
        "train_in_sample_pairs": allowed & (sa == 0) & (sb == 0),
        "train_valid_cross_pairs": allowed & (((sa == 0) & (sb == 1)) | ((sa == 1) & (sb == 0))),
        "valid_only_pairs": allowed & (sa == 1) & (sb == 1),
        "_excluded_test_touching": ~allowed,
    }


def _by_bins(values: np.ndarray, edges: Sequence[float], labels: Sequence[str]) -> np.ndarray:
    out = np.full(values.size, len(labels) - 1, np.int64)
    for i in range(len(labels)):
        lo = -np.inf if i == 0 else edges[i - 1]
        hi = np.inf if i == len(labels) - 1 else edges[i]
        out[(values >= lo) & (values < hi)] = i
    return out


def _stratum_table(per: Mapping[str, np.ndarray], bins: np.ndarray,
                   labels: Sequence[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i, name in enumerate(labels):
        m = bins == i
        if not m.any():
            continue
        out[name] = {
            "n_molecules": int(m.sum()),
            "var_cut_mean": float(np.mean(per["Var_cut"][m])),
            "var_cut_median": float(np.median(per["Var_cut"][m])),
            "var_cut_p90": float(np.percentile(per["Var_cut"][m], 90)),
            "l_graph_mean": float(np.mean(per["L_graph"][m])),
            "l_cut_mean": float(np.mean(per["L_cut"][m])),
            "var_share": float(np.sum(per["Var_cut"][m]) / max(1e-12, np.sum(per["L_cut"][m]))),
            "cut_count_mean": float(np.mean(per["n_cuts"][m])),
        }
    return out


def _parse_stratum(parses: Mapping[str, np.ndarray], key: str,
                   edges: Sequence[float], labels: Sequence[str]) -> dict[str, Any]:
    vals = np.asarray(parses[key], np.float64)
    bins = _by_bins(vals, edges, labels)
    sq = np.asarray(parses["cut_sq"], np.float64)
    out: dict[str, Any] = {}
    for i, name in enumerate(labels):
        m = bins == i
        if m.any():
            out[name] = {"n_parses": int(m.sum()),
                         "cut_mse_std": float(np.mean(sq[m]))}
    return out


def audit_run(name: str, entry: Mapping[str, Any], cfg: Mapping[str, Any],
              pack: Mapping[str, Any], log=print) -> dict[str, Any]:
    model, ckpt = C.load_checkpoint(entry["family"], cfg, pack, entry["arm"],
                                    entry["run_dir"], device="cpu")
    scaler = ckpt["scaler"]
    adapter = C.CutAdapter(model, entry["family"], pack, device="cpu")
    adapter.prepare()
    rows = {"train": np.asarray(pack["train_idx"], np.int64),
            "valid": np.asarray(pack["valid_idx"], np.int64)}
    y_std = (np.asarray(pack["labels"], np.float64) - scaler["mean"]) / scaler["std"]
    out: dict[str, Any] = {
        "run_id": entry["run_dir"].name, "family": entry["family"],
        "arm": entry["arm"], "role": entry["role"],
        "best_epoch": int(ckpt["epoch"]),
        "params": model.parameter_ledger()["total"],
        "scaler": scaler,
        "splits": {},
    }
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for split, idx in rows.items():
        log(f"  audit {name} {split} ({idx.size} molecules)")
        res = C.evaluate_cut_rows(adapter, pack, idx, y_std, scaler,
                                  batch=(64 if entry["family"] == "parent" else 32),
                                  collect_parses=True)
        arrays[split] = res
        per = res["per_molecule"]
        summary = {k: v for k, v in res.items() if k not in ("per_molecule", "parses")}
        # global min/max of the actual cut predictions, from the parse records
        parses = res["parses"]
        summary["cut_pred_global_min_std"] = float(np.min(parses["f_cut"])) if parses["f_cut"].size else float("nan")
        summary["cut_pred_global_max_std"] = float(np.max(parses["f_cut"])) if parses["f_cut"].size else float("nan")
        summary["cut_pred_global_min"] = summary["cut_pred_global_min_std"] * float(scaler["std"]) + float(scaler["mean"])
        summary["cut_pred_global_max"] = summary["cut_pred_global_max_std"] * float(scaler["std"]) + float(scaler["mean"])
        summary["cut_var_over_sigma2"] = summary["cut_var_std"] / float(scaler["std"]) ** 2
        summary["l_graph_over_sigma2"] = summary["graph_mse_std"] / float(scaler["std"]) ** 2
        summary["l_cut_over_sigma2"] = summary["cut_mse_std"] / float(scaler["std"]) ** 2
        summary["cut_count_histogram"] = {
            str(int(k)): int(v) for k, v in zip(*np.unique(per["n_cuts"], return_counts=True))
        }
        summary["strata_by_mean_S_size"] = _stratum_table(
            per, _by_bins(per["mean_S_size"], [1.5, 2.5, 3.5, 4.5, 8.5],
                          ["S=1", "S=2", "S=3", "S=4", "S=5-8", "S>=9"]),
            ["S=1", "S=2", "S=3", "S=4", "S=5-8", "S>=9"])
        summary["strata_by_mean_C_size"] = _stratum_table(
            per, _by_bins(per["mean_C_size"], [0.5, 2.5, 4.5, 8.5, 16.5, 32.5],
                          ["C=0 (fallback)", "C=1-2", "C=3-4", "C=5-8", "C=9-16", "C=17-32", "C>=33"]),
            ["C=0 (fallback)", "C=1-2", "C=3-4", "C=5-8", "C=9-16", "C=17-32", "C>=33"])
        summary["parse_cut_mse_by_S_size"] = _parse_stratum(
            parses, "S_size", [1.5, 2.5, 3.5, 4.5, 8.5],
            ["S=1", "S=2", "S=3", "S=4", "S=5-8", "S>=9"])
        summary["parse_cut_mse_by_C_size"] = _parse_stratum(
            parses, "C_size", [0.5, 2.5, 4.5, 8.5, 16.5, 32.5],
            ["C=0 (fallback)", "C=1-2", "C=3-4", "C=5-8", "C=9-16", "C=17-32", "C>=33"])
        out["splits"][split] = summary
    tr, va = out["splits"]["train"], out["splits"]["valid"]
    out["train_valid_comparison"] = {
        "var_cut_mean_train": tr["cut_var_std"], "var_cut_mean_valid": va["cut_var_std"],
        "var_cut_ratio_valid_over_train": va["cut_var_std"] / max(1e-12, tr["cut_var_std"]),
        "graph_mse_train": tr["graph_mse_std"], "graph_mse_valid": va["graph_mse_std"],
        "graph_mse_ratio_valid_over_train": va["graph_mse_std"] / max(1e-12, tr["graph_mse_std"]),
        "var_share_train": tr["var_share_of_lcut"], "var_share_valid": va["var_share_of_lcut"],
        "cancel_share_train": tr["cancel_share"], "cancel_share_valid": va["cancel_share"],
        "cut_span_train": tr["cut_span_std"], "cut_span_valid": va["cut_span_std"],
    }
    return out, arrays


# ---------------------------------------------------------------------------
# pair-delta provenance
# ---------------------------------------------------------------------------


def pair_delta_provenance(name: str, entry: Mapping[str, Any], cfg: Mapping[str, Any],
                          pack: Mapping[str, Any], preds: Mapping[str, np.ndarray],
                          adapter: C.CutAdapter, sigma: float) -> dict[str, Any]:
    labels = np.asarray(pack["labels"], np.float64)
    a = np.asarray(pack["pair_mol_a"], np.int64)
    b = np.asarray(pack["pair_mol_b"], np.int64)
    scopes = _scopes(pack)
    pred_mol = C.predictions_by_molecule(pack, preds)
    n_pairs = a.size
    direct_std = np.concatenate([
        adapter.direct_pair_delta(np.arange(lo, min(lo + 512, n_pairs), dtype=np.int64))
        for lo in range(0, n_pairs, 512)
    ]) if n_pairs else np.zeros(0)
    direct = direct_std * float(sigma)
    whole = pred_mol[b] - pred_mol[a]
    true = labels[b] - labels[a]
    out: dict[str, Any] = {
        "definitions": {
            "whole_molecule_delta": "ŷ(G_b) - ŷ(G_a), ŷ = molecule-level best-checkpoint prediction (f_bar scaled to label units)",
            "direct_cut_delta": "f(C,S_b) - f(C,S_a) at the shared C and cut bond, best-checkpoint cut response, scaled to label units",
        },
        "pair_directory": {"n_pairs_total": int(n_pairs),
                           "scopes": {k: int(v.sum()) for k, v in scopes.items() if not k.startswith("_")},
                           "excluded_test_touching": int(scopes["_excluded_test_touching"].sum())},
        "scopes": {},
    }
    for scope, mask in scopes.items():
        if scope.startswith("_"):
            continue
        if not mask.any():
            out["scopes"][scope] = {"n": 0}
            continue
        rec = {
            "n": int(mask.sum()),
            "whole_molecule_delta": C.pair_delta_stats(true[mask], whole[mask]),
            "direct_cut_delta": C.pair_delta_stats(true[mask], direct[mask]),
            "difference_between_definitions": {
                "mean_abs_delta_pred_difference": float(np.mean(np.abs(direct[mask] - whole[mask]))),
                "corr_direct_vs_whole_delta_pred": C.pearson(direct[mask], whole[mask]),
                "mean_direct_delta_pred": float(np.mean(direct[mask])),
                "mean_whole_delta_pred": float(np.mean(whole[mask])),
            },
        }
        if scope == "train_valid_cross_pairs":
            split = np.asarray(pack["molecule_split"], np.uint8)
            valid_mol = np.where(split[a] == 1, a, b)
            rec["cluster_bootstrap_by_valid_molecule"] = {
                "whole_molecule_delta": C.cluster_bootstrap_pair_stats(
                    true[mask], whole[mask], valid_mol[mask], n_boot=2000, seed=0),
                "direct_cut_delta": C.cluster_bootstrap_pair_stats(
                    true[mask], direct[mask], valid_mol[mask], n_boot=2000, seed=0),
            }
        out["scopes"][scope] = rec
    return out


# ---------------------------------------------------------------------------
# source audit
# ---------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return C.sha256_file(path)


def _snippet(path: Path, first: str, last: str) -> dict[str, Any]:
    lines = path.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if first in line]
    ends = [i for i, line in enumerate(lines) if last in line and i >= (starts[0] if starts else 0)]
    if not starts or not ends:
        return {"path": str(path), "sha256": _sha(path), "found": False}
    i0, i1 = starts[0], ends[0]
    return {"path": str(path), "sha256": _sha(path), "found": True,
            "line_start": i0 + 1, "line_end": i1 + 1,
            "source": "\n".join(lines[i0:i1 + 1])}


def source_audit() -> dict[str, Any]:
    csoi_train = C.CSOI_CODE / "csoi_train.py"
    csoi_model = C.CSOI_CODE / "csoi_model.py"
    csoi_analyze = C.CSOI_CODE / "csoi_analyze.py"
    trc_train = C.PARENT_CODE / "trc_train.py"
    trc_model = C.PARENT_CODE / "trc_model.py"
    return {
        "csoi_cutwise_main_loss": _snippet(csoi_train,
                                           "sq = (f_cut - y[owner]) ** 2",
                                           "l_main = per_mol[:n_main].mean()"),
        "parent_cutwise_main_loss": _snippet(trc_train,
                                             "sq = (f_cut - y[owner]) ** 2",
                                             "l_main = per_mol[:n_main].mean()"),
        "csoi_f_bar": _snippet(csoi_model,
                               "f_bar = f.new_zeros(rows.numel())",
                               "f_bar = f_bar / counts.to(f.dtype).clamp(min=1.0)"),
        "parent_f_bar": _snippet(trc_model,
                                 "f_bar = torch.zeros(rows.numel()",
                                 "f_bar = f_bar / counts.float().clamp(min=1.0)"),
        "csoi_pair_cut_delta": _snippet(csoi_model,
                                        "def pair_cut_delta",
                                        "return f[:n] - f[n:]"),
        "parent_pair_cut_delta": _snippet(trc_model,
                                          "def pair_cut_delta",
                                          "return self.cut_scores(h_c, h_b, e) - self.cut_scores(h_c, h_a, e)"),
        "old_pair_metrics_uses_whole_molecule_prediction": _snippet(
            csoi_analyze, "dp = pred[b[mask]] - pred[a[mask]]",
            "rec[scope] = _delta_stats(dl, dp)"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = C.load_pack(cfg, args.dataset)
    log = print
    log(f"cache {pack['cache_sha256']} content={pack.get('content_sha256')} path={pack['cache_path_used']}")
    audit: dict[str, Any] = {
        "protocol": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack["cache_sha256"],
        "cache_content_sha256": pack.get("content_sha256"),
        "cache_path_used": pack["cache_path_used"],
        "test_policy": "no official Test label read; only Train/Valid molecules used",
        "source_audit": source_audit(),
        "arms": {},
        "pair_delta_provenance": {},
    }
    arrays_by_arm: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    adapters: dict[str, C.CutAdapter] = {}
    for name, entry in RUNS.items():
        run, arrays = audit_run(name, entry, cfg, pack, log=log)
        audit["arms"][name] = run
        arrays_by_arm[name] = arrays
        adapters[name] = C.CutAdapter(C.load_checkpoint(entry["family"], cfg, pack, entry["arm"],
                                                        entry["run_dir"], device="cpu")[0],
                                      entry["family"], pack, device="cpu")
        adapters[name].prepare()

    # paired whole-molecule predictions for the provenance check
    old_json_path = C.CSOI_ROUND / "results" / "pair_metrics.json"
    old_json = C.read_json(old_json_path) if old_json_path.is_file() else {}
    for name, entry in RUNS.items():
        try:
            preds = C.load_predictions(entry["run_dir"], kind="best")
        except FileNotFoundError:
            log(f"  [skip pair provenance] {name}: no predictions")
            continue
        scaler = audit["arms"][name]["scaler"]
        prov = pair_delta_provenance(name, entry, cfg, pack, preds,
                                     adapters[name], scaler["std"])
        # compare our whole-molecule numbers with the old pair_metrics.json
        if old_json:
            old = old_json.get("by_arm", {}).get(name, {})
            checks = {}
            for scope in ("train_in_sample_pairs", "train_valid_cross_pairs", "valid_only_pairs"):
                o = old.get(scope, {})
                n = prov["scopes"].get(scope, {})
                if o and n.get("n"):
                    checks[scope] = {
                        "old_delta_mae": o.get("delta_mae"),
                        "recomputed_whole_delta_mae": n["whole_molecule_delta"]["delta_mae"],
                        "abs_diff_delta_mae": abs(float(o["delta_mae"]) - float(n["whole_molecule_delta"]["delta_mae"])),
                        "old_delta_rmse": o.get("delta_rmse"),
                        "recomputed_whole_delta_rmse": n["whole_molecule_delta"]["delta_rmse"],
                        "abs_diff_delta_rmse": abs(float(o["delta_rmse"]) - float(n["whole_molecule_delta"]["delta_rmse"])),
                    }
            prov["matches_old_pair_metrics_json"] = checks
        audit["pair_delta_provenance"][name] = prov
        log(f"  pair provenance {name}")

    # short factual interpretation, no causal claim
    audit["summary"] = {}
    for name, run in audit["arms"].items():
        tr = run["splits"]["train"]
        va = run["splits"]["valid"]
        audit["summary"][name] = {
            "train_var_share_of_lcut": tr["var_share_of_lcut"],
            "valid_var_share_of_lcut": va["var_share_of_lcut"],
            "train_var_cut_std": tr["cut_var_std"],
            "valid_var_cut_std": va["cut_var_std"],
            "train_l_graph_std": tr["graph_mse_std"],
            "valid_l_graph_std": va["graph_mse_std"],
            "identity_max_abs_train": tr["identity_max_abs"],
            "identity_max_abs_valid": va["identity_max_abs"],
        }
    out_path = Path(args.out) if args.out else C.analysis_dir() / "loss_decomposition.json"
    C.write_json(out_path, audit)
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())