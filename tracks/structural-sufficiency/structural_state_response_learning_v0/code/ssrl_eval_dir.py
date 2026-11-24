"""Build the compact official-Valid evaluation directory for SSRL v0.

The heavy reconstruction is inherited read-only from the frozen parent rounds:

* ``ssa_data.load_w3`` / ``build_observations`` -- the exact W3 checkpoint
  inference and the SFEC v1 Valid observation table, including the frozen
  metric reproduction check;
* ``ssa_data.select_rectangles`` -- the fixed-hash SSTA rectangle directory;
* ``ssa_sample`` / ``ssa_state`` / ``ssa_action`` -- the frozen same-edit screen
  and its 946 / 38 rectangle masks.

The output is a single small NPZ (plus JSON metadata) that lets the SSRL
evaluation code compute the replacement-delta and interaction metrics from
whole-molecule predictions without re-reading the SFEC Valid cell tables.
No Test label is read.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ssrl_common as C


def main() -> int:
    import ssa_common as SC
    import ssa_data as SD
    import ssa_sample as SMP
    import ssa_state as SS
    import ssa_action as ACT

    t0 = time.perf_counter()
    SC.set_torch_threads()
    print("loading frozen W3 and the full official Valid split", flush=True)
    w3 = SD.load_w3()
    print(f"  W3 Valid MAE {w3['mae_eV']:.12f} eV (checkpoint {w3['checkpoint_valid_mae_eV']:.12f})",
          flush=True)
    if abs(w3["mae_eV"] - w3["checkpoint_valid_mae_eV"]) > 1e-6:
        raise RuntimeError("frozen W3 inference does not reproduce the checkpoint MAE")

    print("reconstructing SFEC v1 Valid observations", flush=True)
    obs = SD.build_observations(w3)
    print(f"  observations {len(obs['key'])}; evaluable {int(obs['main'].sum())}; "
          f"contexts {obs['ctx_v'].n_ctx}", flush=True)

    print("selecting the frozen SSTA rectangle directory", flush=True)
    sel = SD.select_rectangles(obs)
    print(f"  rectangles {len(sel['rect']['pair_key'])}; pair types {len(sel['pair_types'])}",
          flush=True)

    print("selecting the fixed-hash state sample and the same-edit screen", flush=True)
    sample = SMP.select_state_sample(obs, sel)
    machine = SS.StateMachine(w3["model"], w3["spec"], w3["norm"], w3["label_norm"])
    states = SMP.compute_states(w3, obs, sample, machine)
    states["mol_ids"] = np.asarray(sample["mol_ids"], np.int64)
    cal = SS.calibrate_shift(machine, states["scores"])
    U = SMP.build_U(machine, states)
    gids = np.asarray(w3["gids"], np.int64)
    pred_by_cache = np.asarray(w3["pred_eV"], np.float64)
    frozen_pred_eV_mol = pred_by_cache[np.asarray(states["cache_rows"], np.int64)]
    action = ACT.analyse(obs, sel, sample, states, U, machine,
                         frozen_pred_eV_mol, {})
    sup = action["support"]
    print(f"  same-edit {sup['n_rectangles_same_edit']} / {sup['n_pair_types_same_edit']} pair types; "
          f"strict {sup['n_rectangles_same_edit_and_same_delta_g']} / "
          f"{sup['n_pair_types_same_edit_and_same_delta_g']}", flush=True)

    # The frozen SSTA support is the identity anchor of this directory.
    ref = C.read_json(C.SSTA / "analysis" / "action_support.json")
    ref_sup = ref["support"]
    for key in ("n_rectangles_same_edit", "n_pair_types_same_edit",
                "n_rectangles_same_edit_and_same_delta_g",
                "n_pair_types_same_edit_and_same_delta_g"):
        if int(sup[key]) != int(ref_sup[key]):
            raise RuntimeError(f"same-edit support mismatch for {key}: {sup[key]} != {ref_sup[key]}")
    if int(sup["n_rectangles_same_edit"]) != 946:
        raise RuntimeError(f"expected 946 same-edit rectangles, got {sup['n_rectangles_same_edit']}")
    if int(sup["n_rectangles_same_edit_and_same_delta_g"]) != 38:
        raise RuntimeError("expected 38 strict same-edit rectangles")

    sample_idx = np.asarray(sample["rect_index"], np.int64)
    same_mask = np.asarray(action["masks"]["same_edit"], bool)
    strict_mask = np.asarray(action["masks"]["same_edit_and_same_delta_g"], bool)
    if sample_idx.size != same_mask.size or sample_idx.size != strict_mask.size:
        raise RuntimeError("sample index and action masks have different lengths")

    rect = {k: np.asarray(v) for k, v in sel["rect"].items()}
    obs_ctx = obs["ctx_v"]
    pairs = obs["pairs_v"]
    cell_mol = np.asarray(obs["cell_mol"], np.int64)
    cell_row_gid = np.asarray(obs["cell_row"], np.int64)
    cell_cache_row = np.asarray(obs["cell_cache_row"], np.int64)
    cell_clean = np.asarray(obs["cell_clean"], bool)
    ctx_C = np.asarray(obs_ctx.C, np.int64)
    ctx_seen = np.asarray(obs["ctx_seen"], bool)
    ctx_q = np.asarray(obs["ctx_q_of_ctx"], np.int64)
    rank_i = np.asarray(obs["rank_i"], np.int64)
    rank_j = np.asarray(obs["rank_j"], np.int64)
    pair_sign = np.where(rank_i < rank_j, 1.0, -1.0).astype(np.float64)

    arrays = {
        # rectangle directory (full frozen SSTA set)
        **{f"rect_{k}": v for k, v in rect.items()},
        "rect_pair_types": np.asarray(sel["pair_types"], np.int64),
        "rect_n_possible_per_pair": np.asarray(sel["n_possible_per_pair"], np.int64),
        "rect_n_contexts_per_pair": np.asarray(sel["n_contexts_per_pair"], np.int64),
        # same-edit / strict membership over the fixed state sample
        "sample_rect_index": sample_idx,
        "sample_same_edit_mask": same_mask,
        "sample_strict_mask": strict_mask,
        "same_edit_rect_index": sample_idx[same_mask],
        "strict_rect_index": sample_idx[strict_mask],
        # replacement observations (full frozen SFEC v1 table)
        "pair_key": np.asarray(obs["key"], np.int64),
        "pair_ctx": np.asarray(pairs["ctx"], np.int64),
        "pair_pos_i": np.asarray(pairs["pos_i"], np.int64),
        "pair_pos_j": np.asarray(pairs["pos_j"], np.int64),
        "pair_delta": np.asarray(pairs["delta"], np.float64),
        "pair_sign": pair_sign,
        "pair_rank_i": rank_i,
        "pair_rank_j": rank_j,
        "pair_main": np.asarray(obs["main"], bool),
        "pair_zero": np.asarray(obs["preds"]["ZERO"], np.float64),
        "pair_pair": np.asarray(obs["preds"]["PAIR"], np.float64),
        "pair_pair_q": np.asarray(obs["preds"]["PAIR_Q"], np.float64),
        "pair_local": np.asarray(obs["preds"]["LOCAL"], np.float64),
        "pair_lcrc_frozen": np.asarray(obs["delta_pred"], np.float64),
        # evaluable subsets (frozen definitions)
        **{f"subset_{k}": np.asarray(v, bool) for k, v in obs["subsets"].items()},
        # cell-level identity and whole-molecule prediction mapping
        "cell_mol": cell_mol,
        "cell_row_gid": cell_row_gid,
        "cell_cache_row": cell_cache_row,
        "cell_clean": cell_clean,
        "cell_pred_eV_frozen": np.asarray(obs["cell_pred_eV"], np.float64),
        "ctx_C": ctx_C,
        "ctx_seen": ctx_seen,
        "ctx_q": ctx_q,
        # full official Valid labels for independent MAE cross-checks
        "valid_gids": gids,
        "valid_y_eV": np.asarray(w3["y_eV"], np.float64),
        "valid_pred_eV_frozen": np.asarray(w3["pred_eV"], np.float64),
        "valid_pred_z_frozen": np.asarray(w3["pred_z"], np.float64),
    }
    out = C.DATA_DIR / "valid_eval_directory.npz"
    np.savez_compressed(out, **arrays)
    meta = {
        "schema": "ssrl-v0-valid-eval-directory",
        "round": "structural-state-response-learning-v0",
        "created_utc": C.now_utc(),
        "git_commit": C.git_commit(),
        "source": "frozen SSTA v0 / SED v0 / SFEC v1 official Valid directory",
        "path": C.track_rel(out),
        "sha256": C.sha256_file(out),
        "w3": {
            "checkpoint": C.track_rel(C.LCW_CHECKPOINT),
            "checkpoint_sha256": w3["checkpoint_sha256"],
            "checkpoint_epoch": int(w3["checkpoint_epoch"]),
            "checkpoint_valid_mae_eV": float(w3["checkpoint_valid_mae_eV"]),
            "reproduced_valid_mae_eV": float(w3["mae_eV"]),
            "parameters": int(w3["parameters"]),
            "model": str(w3["model_name"]),
        },
        "observations": {
            "n": int(len(obs["key"])),
            "n_evaluable": int(obs["main"].sum()),
            "n_contexts": int(obs_ctx.n_ctx),
            "subsets": {k: int(np.asarray(v).sum()) for k, v in obs["subsets"].items()},
        },
        "rectangles": {
            "n": int(len(rect["pair_key"])),
            "n_pair_types": int(len(sel["pair_types"])),
            "n_possible": int(np.asarray(sel["n_possible_per_pair"]).sum()),
            "n_same_edit": int(sup["n_rectangles_same_edit"]),
            "n_same_edit_pair_types": int(sup["n_pair_types_same_edit"]),
            "n_strict": int(sup["n_rectangles_same_edit_and_same_delta_g"]),
            "n_strict_pair_types": int(sup["n_pair_types_same_edit_and_same_delta_g"]),
            "ssta_support_reproduced": True,
        },
        "state_sample": {
            "n_rectangles": int(sample_idx.size),
            "n_molecules": int(len(sample["mol_ids"])),
            "n_covered_pair_types": int(sample["n_covered_pair_types"]),
            "attention_shift_c_k": cal["c_k"],
            "phi_dim": int(machine.phi_dim),
        },
        "test_labels_read": False,
        "test_policy": "official Test labels are hidden and never indexed",
    }
    C.write_json(C.DATA_DIR / "valid_eval_directory.json", meta)

    # Also save the complete same-edit mechanism rows as a small CSV for the
    # report; these are subsets of the frozen SSTA directory, not re-selected.
    rows = []
    for tag, idx in (("same_edit", np.flatnonzero(same_mask)),
                     ("strict", np.flatnonzero(strict_mask))):
        for pos in idx.tolist():
            i = int(sample_idx[pos])
            rows.append({
                "subset": tag,
                "rect_index": i,
                "pair_key": int(rect["pair_key"][i]),
                "ctx1": int(rect["ctx1"][i]),
                "ctx2": int(rect["ctx2"][i]),
                "delta_true_1": float(rect["delta_true_1"][i]),
                "delta_true_2": float(rect["delta_true_2"][i]),
                "D_true_eV": float(rect["delta_true_2"][i] - rect["delta_true_1"][i]),
                "D_frozen_W3_eV": float(rect["delta_lcrc_2"][i] - rect["delta_lcrc_1"][i]),
            })
    C.write_csv(C.ANALYSIS_DIR / "same_edit_rectangles.csv", rows,
                columns=["subset", "rect_index", "pair_key", "ctx1", "ctx2",
                         "delta_true_1", "delta_true_2", "D_true_eV", "D_frozen_W3_eV"])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB) in "
          f"{time.perf_counter() - t0:.1f}s; sha256 {meta['sha256']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
