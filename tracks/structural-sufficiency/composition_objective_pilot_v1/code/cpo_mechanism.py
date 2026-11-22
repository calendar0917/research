"""Post-training mechanism diagnostics at the best checkpoints.

For every old/new arm that has a checkpoint, records the frozen-weight
interaction contribution, attention concentration (CSOI interaction arms),
gradient-norm snapshots from the training curves, and cut-dispersion/cancellation
statistics.  CPU only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cpo_common as C
import cpo_train as T
import csoi_diag as CDIAG

ARMS = {
    "A_TRC_FACTOR_CUTRESP": ("parent", "FACTOR", C.PARENT_ROUND / "results" / "LIPO_FACTOR_CUTRESP_R0"),
    "B_CSOI_ADD": ("csoi", "CSOI_ADD", C.CSOI_ROUND / "results" / "CSOI_ADD_LIPO_R0"),
    "C_CSOI_GLOBAL": ("csoi", "CSOI_GLOBAL", C.CSOI_ROUND / "results" / "CSOI_GLOBAL_LIPO_R0"),
    "F_GRAPH": ("parent", "FACTOR", C.run_dir("F_GRAPH_LIPO_R0")),
    "B_GRAPH": ("csoi", "CSOI_ADD", C.run_dir("B_GRAPH_LIPO_R0")),
    "C_GRAPH": ("csoi", "CSOI_GLOBAL", C.run_dir("C_GRAPH_LIPO_R0")),
}


def _curve_rows(run_dir: Path) -> list[dict[str, Any]]:
    path = Path(run_dir) / "curves.csv"
    return C.read_csv(path) if path.is_file() else []


def _grad_fields(row: Mapping[str, Any]) -> dict[str, float]:
    return {k: float(v) for k, v in row.items()
            if (k.startswith("grad_") or k == "grad_norm_total")
            and isinstance(v, (int, float)) and np.isfinite(float(v))}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="lipo")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    cfg = C.load_config()
    pack = C.load_pack(cfg, args.dataset)
    valid = np.asarray(pack["valid_idx"], np.int64)
    audit_path = C.analysis_dir() / "loss_decomposition.json"
    audit = C.read_json(audit_path) if audit_path.is_file() else {"arms": {}}
    out: dict[str, Any] = {
        "protocol": cfg["protocol_id"],
        "generated_utc": C.now_utc(),
        "cache_sha256": pack["cache_sha256"],
        "test_policy": "Valid molecules only; Test labels never read",
        "definition": {
            "interaction_contribution": "f_full(S,C) - f_additive(S,C) on 16 Valid molecules, frozen best weights",
            "attention": "per-query attention entropy / effective tokens on 16 Valid molecules",
            "gradient": "first-epoch and best-epoch snapshots from curves.csv (the first training step of the epoch)",
        },
        "arms": {},
    }
    for name, (family, arm, run_dir) in ARMS.items():
        if not (Path(run_dir) / "state_best.pt").is_file():
            print(f"[skip] {name}: no checkpoint")
            continue
        model, ckpt = C.load_checkpoint(family, cfg, pack, arm, run_dir, device=args.device)
        diag = T.interaction_diagnostics(model, family, pack, valid, args.device)
        attention = None
        if family == "csoi" and getattr(model, "interaction", False):
            att = CDIAG._attention_stats(model, pack, valid, args.device, max_mol=16)
            attention = {
                "effective_tokens_p50_all_heads": att["effective_tokens"]["all_heads"].get("p50"),
                "effective_tokens_p90_all_heads": att["effective_tokens"]["all_heads"].get("p90"),
                "entropy_nats_p50_all_heads": att["attention_entropy_nats"]["all_heads"].get("p50"),
                "effective_token_ratio": att.get("effective_token_ratio"),
                "frac_query_tokens_effectively_one_hot": att.get("frac_query_tokens_effectively_one_hot"),
                "keys_per_query_p50": att.get("keys_per_query_tokens", {}).get("p50"),
                "n_query_tokens": att.get("n_query_tokens"),
                "n_entries": att.get("n_entries"),
            }
        rows = _curve_rows(Path(run_dir))
        best_epoch = int(ckpt["epoch"])
        first = rows[0] if rows else {}
        best = next((r for r in rows if int(float(r.get("epoch", -1))) == best_epoch), rows[-1] if rows else {})
        rec: dict[str, Any] = {
            "family": family, "model_arm": arm, "run_dir": str(run_dir),
            "best_epoch": best_epoch,
            "params": model.parameter_ledger(),
            "diagnostics": diag,
            "attention": attention,
            "gradients_first_epoch": _grad_fields(first),
            "gradients_best_epoch": _grad_fields(best),
            "curve_best": {
                k: best.get(k) for k in (
                    "train_rmse", "valid_rmse", "train_cut_var_std", "valid_cut_var_std",
                    "train_cut_mse_std", "valid_cut_mse_std",
                    "train_cut_rmse_std", "valid_cut_rmse_std",
                    "train_graph_mse_std", "valid_graph_mse_std",
                    "train_cancel_share", "valid_cancel_share",
                    "train_mixed_sign_fraction", "valid_mixed_sign_fraction",
                    "train_var_share_of_lcut", "valid_var_share_of_lcut",
                    "aux_loss") if k in best
            },
        }
        a = audit.get("arms", {}).get(name)
        if a is not None:
            rec["audit_best"] = {
                split: {k: a["splits"][split].get(k) for k in (
                    "cut_var_std", "graph_mse_std", "cut_mse_std", "cancel_share",
                    "mixed_sign_fraction", "var_share_of_lcut", "cut_span_std")}
                for split in ("train", "valid")
            }
        out["arms"][name] = rec
        print(name, "best", best_epoch, "interaction",
              diag.get("interaction_contribution"), "attention", attention)
    out_path = C.analysis_dir() / "mechanism_diagnostics.json"
    C.write_json(out_path, out)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())