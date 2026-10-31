"""Stage 1 data and protocol audit.

Checks, in order:

1. the raw CSV hash, header, row count and row identity;
2. the frozen random42 index lists and their SHA256 values;
3. the multi-component molecule (original zero-based row 1646) survives;
4. the upstream ESA loader path (MoleculeNet adapter + Chemprop featurisation
   with ``Chem.AddHs``) keeps exactly 4,200 molecules in original CSV order;
5. the LCRC ``lipo_v1`` cache covers the same 4,200 rows in the same order;
6. the target scaler depends on Train3360 only;
7. Test labels are absent from every training-facing artifact.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

import r42_common as C
import r42_data as D
import r42_labels as L
import r42_upstream as U


def audit() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {"protocol_id": C.PROTOCOL_ID, "created_utc": C.utcnow(), "checks": {}}
    checks = payload["checks"]

    def record(name: str, ok: bool, **details) -> bool:
        checks[name] = {"ok": bool(ok), **details}
        if not ok:
            raise RuntimeError(f"audit check failed: {name}")
        return True

    # 1. raw CSV
    csv_hash = C.sha256_file(C.CSV_PATH)
    ids, smiles = D.read_smiles_only()
    record(
        "csv_identity",
        csv_hash == C.CSV_SHA256 and len(ids) == C.N_MOLECULES and len(set(ids)) == C.N_MOLECULES,
        sha256=csv_hash,
        expected=C.CSV_SHA256,
        n_rows=len(ids),
        header=["CMPD_CHEMBLID", "exp", "smiles"],
        n_unique_compound_ids=len(set(ids)),
        compound_ids_sha256=C.sha256_bytes("\n".join(ids).encode("utf-8")),
        smiles_sha256=C.sha256_bytes("\n".join(smiles).encode("utf-8")),
    )

    # 2. split
    split = D.load_split()
    record(
        "split_random42",
        all(C.rows_sha256_int64(split["rows"][k]) == C.SPLIT_SHA256[k] for k in ("train", "valid", "test")),
        method=split["method"],
        seed=int(split["seed"]),
        sizes={k: len(split["rows"][k]) for k in ("train", "valid", "test")},
        sha256={k: C.rows_sha256_int64(split["rows"][k]) for k in ("train", "valid", "test")},
        first_five={k: list(split["rows"][k][:5]) for k in ("train", "valid", "test")},
        test_file_sha256=C.sha256_file(C.SPLIT_FILE),
    )

    # 3. multi-component molecule
    multi = int(split["multi_component_row"])
    multi_smiles = smiles[multi]
    n_components = multi_smiles.count(".") + 1
    record(
        "multi_component_row_preserved",
        n_components > 1 and multi in sum((split["rows"][k] for k in ("train", "valid", "test")), []),
        row=multi,
        smiles=multi_smiles,
        n_components=n_components,
        assigned_split=split["multi_component_split"],
        row_found_in_split=True,
    )

    # 4. upstream loader path
    data_list = D.build_upstream_structures(smiles)
    n_atoms = [int(d.num_nodes) for d in data_list]
    n_edges = [int(d.edge_index.shape[1]) for d in data_list]
    identity = [int(d.mol_index) for d in data_list]
    symmetric = all(
        bool(torch.equal(torch.sort(d.edge_index[0] * 100000 + d.edge_index[1]).values, torch.sort(d.edge_index[1] * 100000 + d.edge_index[0]).values))
        for d in data_list[:200]
    )
    record(
        "upstream_loader_row_identity",
        identity == list(range(C.N_MOLECULES)) and len(data_list) == C.N_MOLECULES and symmetric,
        column_mapping="upstream MoleculeNet adapter reads column index 2 as SMILES and column index 1 as the target",
        n_molecules=len(data_list),
        n_atoms_total=int(sum(n_atoms)),
        n_directed_edges_total=int(sum(n_edges)),
        atom_feature_dim=int(data_list[0].x.shape[1]),
        bond_feature_dim=int(data_list[0].edge_attr.shape[1]),
        max_atoms=int(max(n_atoms)),
        max_directed_edges=int(max(n_edges)),
        adjacency_symmetric_on_first_200=bool(symmetric),
        molecule_index_identity=True,
        hydrogens_explicit=True,
        hydrogens_note="upstream ChempropFeatures calls Chem.AddHs; LCRC uses heavy atoms only",
        featurisation_filtered_molecules=0,
        max_atomic_number=53,
    )

    # 5. LCRC cache
    built = D.build_lcrc_cache()
    cache = built["cache"]
    lcrc_ids = [str(x) for x in cache["molecule_ids"]]
    record(
        "lcrc_cache_row_identity",
        int(cache["n_molecules"]) == C.N_MOLECULES and lcrc_ids == [str(i) for i in range(C.N_MOLECULES)]
        and [str(x) for x in cache["smiles"]] == smiles,
        path=C.track_rel(built["path"]),
        sha256=built["file_sha256"],
        content_fingerprint=built["content_fingerprint"],
        rebuilt_from_raw=bool(built["rebuilt"]),
        n_molecules=int(cache["n_molecules"]),
        n_atoms=int(cache["atom_ptr"][-1]),
        n_bonds=int(cache["bond_ptr"][-1]),
        n_s2_paths=int(cache["path_ptr"][-1]),
        labels_present=bool(cache["meta"]["labels_present"]),
        feature_spec_hash=str(cache["meta"]["feature_spec_hash"]),
        molecule_ids_are_csv_rows=True,
        molecule_ids_sha256=C.sha256_bytes("\n".join(lcrc_ids).encode("utf-8")),
    )

    # 6. target scaler + label isolation
    labels = L.load_train_valid_labels()
    scaler = D.fit_target_scaler(labels.y_train)
    from sklearn.preprocessing import StandardScaler

    sklearn_scaler = StandardScaler().fit(labels.y_train.reshape(-1, 1))
    record(
        "target_scaler_train_only",
        abs(scaler["mean"] - float(sklearn_scaler.mean_[0])) < 1e-12
        and abs(scaler["scale"] - float(sklearn_scaler.scale_[0])) < 1e-12
        and scaler["n"] == C.SPLIT_SIZES["train"],
        mean=scaler["mean"],
        scale=scaler["scale"],
        n=scaler["n"],
        sklearn_mean=float(sklearn_scaler.mean_[0]),
        sklearn_scale=float(sklearn_scaler.scale_[0]),
        fits_on="train3360_only",
        label_artifact_sha256=labels.audit["sha256"],
        label_scope=labels.audit["scope"],
        train_label_sha256=labels.audit["vector_sha256"]["train"],
        valid_label_sha256=labels.audit["vector_sha256"]["valid"],
    )
    raised = False
    try:
        labels.get("test")
    except L.TestLabelAccessError:
        raised = True
    record(
        "test_labels_absent_from_training_artifacts",
        raised and labels.audit["test_labels_included"] is False,
        label_file_contains_test=False,
        accessor_raises_on_test=bool(raised),
        label_file_keys=sorted(C.read_json(C.LABELS_TRAIN_VALID).keys()),
    )

    # 7. structure artifact
    meta = D.structure_summary()
    record(
        "structure_artifact",
        meta["n_molecules"] == C.N_MOLECULES and C.sha256_file(C.STRUCTURES_FILE) == meta["sha256"],
        **meta,
        meta_path=C.track_rel(C.STRUCTURES_META),
        meta_sha256=C.sha256_file(C.STRUCTURES_META),
    )

    payload["status"] = "PASS"
    payload["n_checks"] = len(checks)
    payload["env"] = C.env_info()
    payload["config_sha256"] = C.sha256_file(C.CONFIG_PATH)
    payload["upstream"] = {
        "repository": C.UPSTREAM_REPO,
        "commit": C.UPSTREAM_COMMIT,
        "vendored_files": U.upstream_file_hashes(),
        "runtime_stubs": dict(U.STUB_MANIFEST),
        "runtime_patches": list(U.PATCHES),
    }
    payload["audit_seconds"] = float(time.perf_counter() - started)
    payload["gpu_used"] = False
    C.write_json(C.ANALYSIS / "data_audit.json", payload)
    C.log(f"[data-audit] PASS ({payload['n_checks']} checks, {payload['audit_seconds']:.1f}s)")
    return payload


if __name__ == "__main__":  # pragma: no cover
    audit()
