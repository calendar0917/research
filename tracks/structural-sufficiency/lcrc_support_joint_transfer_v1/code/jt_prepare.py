"""Freeze the config, build the two Train-only support artifacts, audit splits.

Run this **before** any formal training.  It writes:

* ``configs/experiment.json`` and ``analysis/protocol_freeze.json``;
* ``analysis/support/lipo_support.npz|.json``;
* ``analysis/support/pcba_support.npz|.json``;
* ``analysis/support_manifest.json`` with byte hashes and Omega identities;
* ``analysis/split_audit.json`` with official-source row-id equality.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import jt_common as C  # noqa: E402
import jt_support as S  # noqa: E402

S._prepare()


def read_split_csv(path: Path) -> np.ndarray:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        tokens = handle.read().split()
    return np.asarray([int(t) for t in tokens], dtype=np.int64)


def _raw_index_hash(rows: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(np.asarray(rows, dtype=np.int64)).tobytes())


def _support_fingerprint(support: Any) -> dict[str, Any]:
    return {
        **S.omega_identity(support),
        "omega_count_sum": int(np.asarray(support.omega_count).sum()),
        "omega_docfreq_sum": int(np.asarray(support.omega_docfreq).sum()),
        "pair_mode_counts": {
            "omega": int((np.asarray(support.pair_mode) == 0).sum()),
            "marginal_ls": int((np.asarray(support.pair_mode) == 1).sum()),
            "type_equal_mean": int((np.asarray(support.pair_mode) == 2).sum()),
        },
    }


def prepare_lipo() -> dict[str, Any]:
    import cr_common as CC
    import cr_data as D

    bundle = D.build_bundle("ogbg-mollipo")
    audit_path = C.READOUT_ROUND / "analysis/dataset_audit.json"
    historical = C.read_json(audit_path)["ogbg-mollipo"]

    rows = {}
    for split in C.SPLITS:
        official = read_split_csv(C.REPO / f"data/ogb/ogbg_mollipo/split/scaffold/{split}.csv.gz")
        payload_ids = np.asarray(bundle.global_row_ids[split], dtype=np.int64)
        cache_ids = np.asarray([int(x) for x in bundle.caches[split]["molecule_ids"]], dtype=np.int64)
        rows[split] = {
            "n": int(payload_ids.size),
            "official_csv_equals_payload": bool(np.array_equal(official, payload_ids)),
            "official_csv_equals_cache_order": bool(np.array_equal(official, cache_ids)),
            "payload_equals_cache_order": bool(np.array_equal(payload_ids, cache_ids)),
            "row_id_sha256": C.sha256_array(payload_ids),
            "historical_row_id_sha256": historical["row_id_sha256"][split],
            "matches_historical_hash": C.sha256_array(payload_ids) == historical["row_id_sha256"][split],
            "official_csv_sha256": C.sha256_array(official),
        }
    scaler = bundle.scaler
    hist_scaler = historical["target_scaler"]
    split_audit = {
        "dataset": "ogbg-mollipo",
        "source": "official OGB scaffold split CSV",
        "rows": rows,
        "sizes": bundle.split_sizes,
        "target_scaler": scaler,
        "historical_target_scaler": hist_scaler,
        "target_scaler_matches": all(
            abs(float(scaler[k]) - float(hist_scaler[k])) <= 1e-12 for k in ("mean", "std", "n")
        ),
        "cache_hashes": D.verify_frozen_files("ogbg-mollipo"),
        "all_row_ids_equal_official": all(v["official_csv_equals_payload"] for v in rows.values()),
        "all_cache_orders_equal_official": all(v["official_csv_equals_cache_order"] for v in rows.values()),
        "all_rows_match_historical": all(v["matches_historical_hash"] for v in rows.values()),
    }

    support = S.build_lipo_support(bundle)
    # Train-only proof: recomputing from Train alone reproduces Omega exactly.
    train_only = S._modules()["sj_support"].build("lipo", bundle.caches["train"], spec=bundle.spec)
    if not np.array_equal(np.asarray(support.omega_code), np.asarray(train_only.omega_code)):
        raise RuntimeError("Lipo Omega is not the Train-only pair set")
    if not np.array_equal(np.asarray(support.P), np.asarray(train_only.P)):
        raise RuntimeError("Lipo projector differs from the Train-only projector")
    path, meta_path = S.save_support(support, C.ANALYSIS / "support/lipo_support.npz")
    return {
        "split_audit": split_audit,
        "support": support,
        "support_path": path,
        "meta_path": meta_path,
        "fingerprint": _support_fingerprint(support),
        "train_only_recompute_equal": True,
    }


def prepare_pcba() -> dict[str, Any]:
    spec = S.pcba_spec()
    split_payload = C.read_json(C.PCBA_SPLIT_PATH)
    rows = {}
    for split in C.SPLITS:
        official = read_split_csv(C.REPO / f"data/ogb/ogbg_molpcba/split/scaffold/{split}.csv.gz")
        stored = np.asarray(split_payload["rows"][split], dtype=np.int64)
        rows[split] = {
            "n": int(stored.size),
            "official_csv_equals_stored": bool(np.array_equal(official, stored)),
            "official_csv_sha256": C.sha256_array(official),
            "stored_sha256": C.sha256_array(stored),
            "stored_raw_index_sha256": _raw_index_hash(stored),
            "stored_index_sha256": split_payload.get("index_sha256", {}).get(split),
        }
    gids = np.load(C.PC_CACHE / "gids.npy", mmap_mode="r")
    split_audit = {
        "dataset": "ogbg-molpcba",
        "source": "official OGB scaffold split CSV compared elementwise with the frozen split JSON",
        "rows": rows,
        "sizes": {k: int(v["n"]) for k, v in rows.items()},
        "cache_gids_are_official_order": bool(np.array_equal(np.asarray(gids), np.arange(gids.size, dtype=gids.dtype))),
        "cache_dir": C.track_rel(C.PC_CACHE),
        "label_artifact_sha256": split_payload.get("partition_sha256"),
        "all_row_ids_equal_official": all(v["official_csv_equals_stored"] for v in rows.values()),
        "stored_index_hashes_match": all(
            v["stored_index_sha256"] == v["stored_raw_index_sha256"] for v in rows.values()
        ),
    }
    caches = S.pcba_split_caches(split_payload)
    support = S.build_pcba_support(spec, caches)
    train_only = S.build_pcba_support(spec, {"train": caches["train"], "valid": None, "test": None})
    if not np.array_equal(np.asarray(support.omega_code), np.asarray(train_only.omega_code)):
        raise RuntimeError("PCBA Omega is not the Train-only pair set")
    if not np.array_equal(np.asarray(support.P), np.asarray(train_only.P)):
        raise RuntimeError("PCBA projector differs from the Train-only projector")
    path, meta_path = S.save_support(support, C.ANALYSIS / "support/pcba_support.npz")
    return {
        "split_audit": split_audit,
        "support": support,
        "support_path": path,
        "meta_path": meta_path,
        "fingerprint": _support_fingerprint(support),
        "train_only_recompute_equal": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="lipo,pcba")
    args = parser.parse_args()
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.CONFIGS.mkdir(parents=True, exist_ok=True)

    payload = C.config_payload()
    C.write_json(C.CONFIGS / "experiment.json", payload)
    freeze = {
        "schema": "lcrc-support-joint-transfer-freeze-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "frozen_utc": C.utcnow(),
        "config_payload": payload,
        "config_sha256": C.sha256_file(C.CONFIGS / "experiment.json"),
        "code_hashes": {
            name: C.sha256_file(CODE_DIR / name)
            for name in sorted(p.name for p in CODE_DIR.glob("jt_*.py"))
        },
        "reused_assets": {
            "lipo_a": C.track_rel(C.LIPO_A_DIR),
            "w2": C.track_rel(C.W2_DIR),
            "fixed1": C.track_rel(C.FIXED1_DIR),
            "lipo_code": C.track_rel(C.READOUT_CODE),
            "support_v0_code": C.track_rel(C.SJ_CODE),
            "pcba_bs_code": C.track_rel(C.BS_CODE),
            "pcba_dscr_code": C.track_rel(C.DSCR_CODE),
        },
        "environment": C.environment(),
    }
    freeze["freeze_hash"] = C.sha256_text(
        C.json.dumps({k: v for k, v in freeze.items() if k != "freeze_hash"}, sort_keys=True, default=str)
    ) if hasattr(C, "json") else C.sha256_text(str(freeze))
    C.write_json(C.ANALYSIS / "protocol_freeze.json", freeze)

    manifest: dict[str, Any] = {"schema": "lcrc-support-joint-transfer-support-manifest-v1", "round_id": C.ROUND_ID}
    split_audits: dict[str, Any] = {}
    for task in tasks:
        if task == "lipo":
            out = prepare_lipo()
        elif task == "pcba":
            out = prepare_pcba()
        else:
            raise KeyError(task)
        split_audits[task] = out["split_audit"]
        manifest[task] = {
            "support_path": C.track_rel(out["support_path"]),
            "support_sha256": C.sha256_file(out["support_path"]),
            "meta_path": C.track_rel(out["meta_path"]),
            "meta_sha256": C.sha256_file(out["meta_path"]),
            "fingerprint": out["fingerprint"],
            "train_only_recompute_equal": bool(out["train_only_recompute_equal"]),
            "summary": out["support"].summary(),
        }
        C.write_json(C.ANALYSIS / "support_manifest.json", manifest)
        C.write_json(C.ANALYSIS / "split_audit.json", split_audits)
    print(C.json.dumps(manifest, indent=2, default=str))


if __name__ == "__main__":
    main()
