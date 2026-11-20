"""Write the artefact/identity manifest for this round (no computation of results)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import fu_common as FC  # noqa: E402
import rd_common as RC  # noqa: E402

PARENT_CODE = (
    "rd_common.py", "rd_model.py", "rd_train.py", "rd_dictionary.py",
    "rd_dict_run.py", "rd_fragment.py",
)
ROUND_CODE = (
    "fu_common.py", "fu_model.py", "fu_train.py", "fu_joint_run.py",
    "fu_checks.py", "fu_dict_k16.py", "fu_edits.py", "fu_analyze.py",
)


def main() -> int:
    follow = FC.load_config()
    recipe = FC.load_parent_config()
    joint = FC.run_dir("lipo", "JOINT")
    status = RC.read_json(joint / "status.json")
    summary = RC.read_json(joint / "summary.json")
    stress = FC.run_dir("lipo", "RESP") / "dictionary_stress_k16"
    stress_summary = RC.read_json(stress / "stress_summary.json")
    edits = FC.ROUND_DIR / "analysis" / "edit_support.json"

    manifest = {
        "round_id": follow["round_id"],
        "parent_round": follow["parent_round"],
        "parent_commit": follow["parent_commit"],
        "protocol_hash": FC.protocol_hash(),
        "parent_protocol_hash": RC.protocol_hash(recipe),
        "config": {
            "path": str(FC.CONFIG_PATH.relative_to(FC.REPO_ROOT)),
            "sha256": RC.sha256_file(FC.CONFIG_PATH),
            "parent_config_path": str(FC.S0_CONFIG_PATH.relative_to(FC.REPO_ROOT)),
            "parent_config_sha256": RC.sha256_file(FC.S0_CONFIG_PATH),
        },
        "executed_commit": status.get("commit"),
        "code_sha256": {
            **{name: RC.sha256_file(CODE_DIR / name) for name in ROUND_CODE},
            **{f"parent/{name}": RC.sha256_file(FC.S0_CODE / name) for name in PARENT_CODE},
        },
        "data_identity": {
            "fragment_cache": {
                "path": "tracks/structural-sufficiency/response_dictionary_small_s0_v1/data/ogbg_mollipo_rd_frag_v1.npz",
                "sha256": RC.sha256_file(FC.S0_DIR / "data" / "ogbg_mollipo_rd_frag_v1.npz"),
            },
            "official_cache_sha256": summary["data_cache_sha256"],
            "canonical_init_state_sha256": RC.sha256_file(FC.S0_DIR / "data" / "canonical_init_state.pt"),
            "init_state_hash": summary["init_state_hash"],
            "train_order_sha256": status.get("train_order_sha256"),
            "parent_resp_checkpoint_sha256": stress_summary["checkpoint_sha256"],
            "edit_pairs_sha256": RC.sha256_file(FC.REPO_ROOT / follow["edit_catalogue"]["path"]),
        },
        "regime": {
            "device": summary["environment"].get("device") if isinstance(summary.get("environment"), dict) else None,
            "environment": summary.get("environment"),
            "precision": summary.get("precision"),
            "seconds": summary.get("seconds"),
            "gpu_hours": summary.get("seconds", 0.0) / 3600.0,
            "eval_batch_size": summary.get("eval_batch_size"),
            "n_parameters": summary.get("n_parameters"),
            "epochs": summary.get("epochs"),
        },
        "artifacts": {
            "curve.csv": RC.sha256_file(joint / "curve.csv"),
            "summary.json": RC.sha256_file(joint / "summary.json"),
            "preflight.json": RC.sha256_file(joint / "preflight.json"),
            "predictions_best.npz": RC.sha256_file(joint / "predictions_best.npz"),
            "predictions_final.npz": RC.sha256_file(joint / "predictions_final.npz"),
            "dictionary_stress_summary": RC.sha256_file(stress / "stress_summary.json"),
            "dictionary_replay": RC.sha256_file(stress / "replay.json"),
            "edit_support": RC.sha256_file(edits),
            "round_summary": RC.sha256_file(FC.ROUND_DIR / "summary.json"),
        },
        "parent_units_reused_read_only": {
            arm: {
                "summary_commit": RC.read_json(FC.S0_RESULTS / f"lipo_{arm}" / "summary.json").get("commit"),
                "best_epoch": RC.read_json(FC.S0_RESULTS / f"lipo_{arm}" / "summary.json")["best"]["epoch"],
                "init_state_hash": RC.read_json(FC.S0_RESULTS / f"lipo_{arm}" / "summary.json")["init_state_hash"],
            }
            for arm in ("RESP", "ADD")
        },
    }
    path = FC.ROUND_DIR / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
