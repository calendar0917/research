"""Verify: did the fulltrain round's official-valid read attach *train* pair
structures to *valid* roots (local_mol_id 0..999 -> root_base[0..999] of the
train payload)?

Test: evaluate the frozen COMP soup on official-valid twice --
(a) exactly the historical path (train payload), reproducing the frozen
    valid_frozen_predictions.npz,
(b) a payload whose incidence structure (root_base / pair_ptr / pair_t /
    pair_a / pair_wJ / pair_wI / root_atom) is built from the *valid* graphs'
    own adjacency (env_valid phi/atom for features; train-frozen phi scaler
    and kappa unchanged).

Both payloads differ ONLY in the incidence structure.  If (a) and (b) differ
materially, the historical valid numbers carry a local-channel structure bug.

Official-test is never touched.  This is the round's authorized single
official-valid read for Stage-4 preparation (no scores are selected on).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_vs_mlp_seed0_v1 as mlpmod,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_component_supervision_fulltrain_confirmation_seed0_v1 as zft,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_dictionary_component_supervision_seed0_v1 as zcs,
)
from tracks.ksvd.experiments.luyin16 import zinc_joint_dictionary_decision_v1 as zjd

FULL_DIR = Path("tracks/ksvd/results/zinc_component_supervision_fulltrain_confirmation_seed0_v1")
VAL_PROCESSED = zjd.REPO_ROOT / "data/ZINC/subset/processed/val.pt"


def load_raw_valid_graphs() -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    loaded = torch.load(VAL_PROCESSED, map_location="cpu", weights_only=False)
    data, slices, _cls = loaded
    x = data["x"].reshape(-1).numpy().astype(np.int64, copy=False)
    edge_index = data["edge_index"].numpy().astype(np.int64, copy=False)
    edge_attr = data["edge_attr"].reshape(-1).numpy().astype(np.int64, copy=False)
    node_slices = slices["x"].numpy().astype(np.int64)
    edge_slices = slices["edge_index"].numpy().astype(np.int64)
    graphs: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(len(node_slices) - 1):
        lo, hi = int(node_slices[index]), int(node_slices[index + 1])
        elo, ehi = int(edge_slices[index]), int(edge_slices[index + 1])
        graphs.append((x[lo:hi].copy(), edge_index[:, elo:ehi].copy(), edge_attr[elo:ehi].copy()))
    if len(graphs) != 1000:
        raise RuntimeError(f"raw valid graph count {len(graphs)} != 1000")
    return graphs


def build_valid_structure() -> dict[str, np.ndarray]:
    """Incidence structure of the 1000 valid graphs (label-free)."""
    env = torch.load(
        "tracks/ksvd/results/e2e_dictenv_p1/cache/env_valid.pt",
        map_location="cpu", weights_only=False,
    )
    phi, atom, node_sizes = (
        env["phi"].numpy(), env["atom"].numpy().astype(np.int64),
        env["node_sizes"].numpy().astype(np.int64),
    )
    graphs = load_raw_valid_graphs()
    if int(node_sizes.sum()) != int(phi.shape[0]):
        raise RuntimeError("env_valid node-size mismatch")
    for index, (atom_types, _ei, _ea) in enumerate(graphs):
        if int(atom_types.shape[0]) != int(node_sizes[index]):
            raise RuntimeError(f"valid graph {index} node count mismatch")
        if not np.array_equal(atom[int(node_sizes[:index].sum()):int(node_sizes[: index + 1].sum())], atom_types):
            raise RuntimeError(f"valid graph {index} root atom order mismatch")

    root_base = np.concatenate([[0], np.cumsum(node_sizes)]).astype(np.int64)
    total_roots = int(node_sizes.sum())
    pair_t: list[np.ndarray] = []
    pair_a: list[np.ndarray] = []
    pair_j: list[np.ndarray] = []
    pair_i: list[np.ndarray] = []
    counts: list[int] = []
    root_d = np.zeros(total_roots, dtype=np.int64)
    for index in range(1000):
        atom_types, edge_index, edge_attr = graphs[index]
        count, _stats = prev.molecule_incidence(atom_types, edge_index, edge_attr)
        d, n_t, n_a, w_joint, w_ind = prev.root_weights(count)
        support = prev._support_union(count, n_t, n_a)
        base = int(root_base[index])
        root_d[base : base + len(d)] = d.astype(np.int64)
        for local in range(len(d)):
            rows = np.nonzero(support[local].reshape(-1))[0]
            if rows.size == 0:
                counts.append(0)
                continue
            pair_t.append((rows // prev.ATOM_CATEGORIES).astype(np.int64))
            pair_a.append((rows % prev.ATOM_CATEGORIES).astype(np.int64))
            pair_j.append(w_joint[local].reshape(-1)[rows].astype(np.float32))
            pair_i.append(w_ind[local].reshape(-1)[rows].astype(np.float32))
            counts.append(int(rows.size))
    return {
        "root_base": root_base,
        "pair_ptr": np.concatenate([[0], np.cumsum(np.asarray(counts, dtype=np.int64))]).astype(np.int64),
        "pair_t": np.concatenate(pair_t).astype(np.int64),
        "pair_a": np.concatenate(pair_a).astype(np.int64),
        "pair_wJ": np.concatenate(pair_j).astype(np.float32),
        "pair_wI": np.concatenate(pair_i).astype(np.float32),
        "root_atom": atom.astype(np.int64),
        "node_sizes": node_sizes,
    }


def main() -> int:
    torch.set_num_threads(8)
    # frozen full-train payload + soup + calibration
    with np.load(FULL_DIR / "full_train_payload.npz", allow_pickle=False) as z:
        train_arrays = {key: z[key] for key in z.files}
    kappa_M = float(train_arrays["kappa"].reshape(-1)[0])
    cal = __import__("json").loads((FULL_DIR / "calibration.json").read_text())

    valid, _meta = zft.load_valid_data(FULL_DIR)
    frozen = np.load(FULL_DIR / "valid_frozen_predictions.npz", allow_pickle=False)

    print("[build] valid incidence structure ...", flush=True)
    vstruct = build_valid_structure()

    # (b) payload with the valid incidence structure, everything else frozen
    valid_arrays = dict(train_arrays)
    for key in ("root_base", "pair_ptr", "pair_t", "pair_a", "pair_wJ", "pair_wI", "root_atom"):
        valid_arrays[key] = vstruct[key].astype(train_arrays[key].dtype)

    results: dict[str, float] = {}
    h_map: dict[str, np.ndarray] = {}
    for tag, arrays in (("train_payload", train_arrays), ("valid_payload", valid_arrays)):
        payload = prev.TuplePayload(valid_arrays if tag == "valid_payload" else train_arrays)
        model = zft.build_component_model("COMP", payload, kappa_M)
        state = torch.load(FULL_DIR / "COMP_raw_soup_state.pt", map_location="cpu", weights_only=True)
        model.load_state_dict({k: v.to("cpu") for k, v in state.items()}, strict=True)
        model.eval()
        h, _comp = zcs.evaluate_state_components(model, valid, torch.device("cpu"))
        h_map[tag] = h
        if tag == "train_payload":
            repro = float(np.max(np.abs(h - frozen["COMP_h_raw"].astype(np.float64))))
            print(f"[reproduce] historical path vs frozen COMP_h_raw max_abs = {repro:.3e}")
            results["reproduce_max_abs"] = repro
        else:
            b_y = float(cal["per_arm"]["COMP"]["b_y"])
            q = frozen["q_raw"].astype(np.float64)
            y = frozen["y"].astype(np.float64)
            mae_a = float(np.mean(np.abs(h_map["train_payload"] + q + b_y - y)))
            mae_b = float(np.mean(np.abs(h + q + b_y - y)))
            print(f"[mae] y_cal train-payload path = {mae_a:.6f}")
            print(f"[mae] y_cal valid-structure path = {mae_b:.6f}")
            results["mae_y_train_payload"] = mae_a
            results["mae_y_valid_payload"] = mae_b
            d = h_map["train_payload"] - h
            results["h_max_abs_diff"] = float(np.max(np.abs(d)))
            results["h_mean_abs_diff"] = float(np.mean(np.abs(d)))
            print(f"[diff] h_raw max_abs = {results['h_max_abs_diff']:.6f} mean_abs = {results['h_mean_abs_diff']:.6f}")

    np.savez_compressed(
        "valid_structure_check.npz",
        h_train_payload=h_map["train_payload"].astype(np.float32),
        h_valid_payload=h_map["valid_payload"].astype(np.float32),
    )
    __import__("json").dump(
        {**results, "train_pairs": int(train_arrays["pair_ptr"][-1]), "valid_pairs": int(vstruct["pair_ptr"][-1])},
        open("valid_structure_check.json", "w"),
        indent=2,
    )
    print("[done]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
