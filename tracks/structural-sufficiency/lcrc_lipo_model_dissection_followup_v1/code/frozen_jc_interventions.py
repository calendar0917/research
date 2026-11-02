"""Independent frozen-weight J/C interventions on the trained FULL and C1 models.

For each of the six repeat-122 checkpoints (A0/FULL and C1/COMPACT, seeds 0/1/2)
the model is loaded once in eval mode.  Four aggregation conditions are
evaluated on the complete 420-molecule Valid split:

    NATIVE : J_full + C_full
    J_ONLY : J_marginal_bt2 + C_full
    C_ONLY : J_full + C_marginal_bt2
    BOTH   : J_marginal_bt2 + C_marginal_bt2

No weights are updated, no checkpoint is selected again and no training is
started.  The original round code is imported read-only; the aggregation switch
is implemented in a subclass that reuses the frozen ``LocalContent`` helpers.
"""

from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import followup_common as F

M = F.M
C = F.C
D = F.D

CONDITIONS = (
    ("NATIVE", "full", "full"),
    ("J_ONLY", "marginal_bt2", "full"),
    ("C_ONLY", "full", "marginal_bt2"),
    ("BOTH", "marginal_bt2", "marginal_bt2"),
)
COVERAGE_ATOL = 1e-6
COVERAGE_RTOL = 1e-5
PARITY_ATOL = 1e-4
PARITY_METRIC_ATOL = 1e-5
BOTH_ATOL = 1e-6
BOTH_RTOL = 1e-5
DECOMP_ATOL = 1e-5
DECOMP_RTOL = 1e-5


class JCLocalContent(M.LocalContent):
    """Frozen local encoder with independent J and C aggregation switches."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jc_joint_agg = "full"
        self.jc_content_agg = "full"

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
        *,
        s2_agg_override: str | None = None,
        zero_s2: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # The override is deliberately ignored: this adapter controls J and C
        # separately through ``jc_*`` attributes.
        n_atoms = a.shape[0]
        width = a.shape[1]
        joint_agg = self.jc_joint_agg
        content_agg = self.jc_content_agg
        if joint_agg not in ("full", "marginal", "marginal_bt2"):
            raise ValueError(joint_agg)
        if content_agg not in ("full", "marginal", "marginal_bt2"):
            raise ValueError(content_agg)

        idx = batch["bond_index"]
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            degree = a.new_zeros(n_atoms)
            degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        else:
            degree = a.new_zeros(n_atoms)
            b_sum = a.new_zeros((n_atoms, width))
        b_mean = b_sum / degree.clamp(min=1.0).unsqueeze(1)

        if self.s1_joint:
            if batch["nbr_code"].numel():
                nbr = a.new_zeros((n_atoms, self.rank)).index_add_(
                    0, batch["nbr_atom"], self.nbr_emb(batch["nbr_code"])
                )
            else:
                nbr = a.new_zeros((n_atoms, self.rank))
        else:
            nbr = None

        layout = self._layout()
        if not layout or zero_s2:
            path_agg = a.new_zeros((n_atoms, sum(self._part_dim(k) for k in layout)))
        else:
            nonmarg = [k for k in layout if not (k == "joint" and joint_agg != "full")]
            if nonmarg:
                cat = torch.cat([self._nonmarginal_part(k, batch) for k in nonmarg], dim=1)
                if batch["path_cat"].numel():
                    agg_nm = a.new_zeros((n_atoms, cat.shape[1])).index_add_(0, batch["path_atom"], cat)
                else:
                    agg_nm = a.new_zeros((n_atoms, cat.shape[1]))
            else:
                agg_nm = a.new_zeros((n_atoms, 0))
            chunks: dict[str, torch.Tensor] = {}
            pos = 0
            for k in nonmarg:
                dim = self._part_dim(k)
                chunks[k] = agg_nm[:, pos : pos + dim]
                pos += dim
            if self.s2_joint and joint_agg != "full":
                chunks["joint"] = self._joint_agg(batch, a, joint_agg)
            path_agg = torch.cat([chunks[k] for k in layout], dim=1)

        if not self.s2_content:
            content_agg_val = a.new_zeros((n_atoms, 0))
        elif zero_s2:
            content_agg_val = a.new_zeros((n_atoms, width))
        else:
            content_agg_val = self._content_agg(batch, a, content_agg)

        parts: list[torch.Tensor] = [a, b_sum]
        if self.b_mean:
            parts.append(b_mean)
        if self.s1_joint:
            parts.append(nbr)
        if layout:
            parts.append(path_agg)
        if self.s2_content:
            parts.append(content_agg_val)
        if self.scalar_cols:
            parts.append(scal[:, list(self.scalar_cols)])
        x = torch.cat(parts, dim=1)
        if x.shape[1] != len(self.local_cols):
            raise RuntimeError(f"local width {x.shape[1]} != frozen column count {len(self.local_cols)}")
        h = self.norm(self.mlp(x))
        return h, {"local_input": x, "path_agg": path_agg, "content_agg": content_agg_val}


def attach_jc_local(model: M.DissectLCRC) -> JCLocalContent:
    old = model.local
    new = JCLocalContent(model.spec, model.variant, model.cfg)
    new.load_state_dict(old.state_dict(), strict=True)
    model.local = new
    return new


def set_condition(model: M.DissectLCRC, joint_agg: str, content_agg: str) -> None:
    model.local.jc_joint_agg = joint_agg
    model.local.jc_content_agg = content_agg


def load_checkpoint(variant: str, seed: int, view: Any) -> tuple[M.DissectLCRC, dict[str, Any], str]:
    path = F.run_dir(F.MAIN_REPEAT, variant, seed) / "checkpoint.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = M.load_checkpoint_model(view.spec, view.train_norm, variant, ckpt["state_dict"], torch.device("cpu"))
    model.eval()
    return model, ckpt, F.rel(path)


def extract_path_parts(local: M.LocalContent, aux: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    layout = local._layout()
    out: dict[str, torch.Tensor] = {}
    pos = 0
    path_agg = aux["path_agg"]
    for key in layout:
        dim = local._part_dim(key)
        out[key] = path_agg[:, pos : pos + dim].detach().numpy().astype(np.float64)
        pos += dim
    return out


def _summary(values: np.ndarray, prefix: str = "") -> dict[str, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return {f"{prefix}{k}": None for k in ("mean", "sd", "median", "p90", "p95", "max", "min", "mean_abs")}
    return {
        f"{prefix}mean": float(arr.mean()),
        f"{prefix}sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}median": float(np.median(arr)),
        f"{prefix}p90": float(np.percentile(arr, 90)),
        f"{prefix}p95": float(np.percentile(arr, 95)),
        f"{prefix}max": float(arr.max()),
        f"{prefix}min": float(arr.min()),
        f"{prefix}mean_abs": float(np.abs(arr).mean()),
    }


def gate_stats(model: M.DissectLCRC, batch: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    local = model.local
    path_cat = batch["path_cat"].detach().cpu().numpy().astype(np.int64)
    path_end63 = batch["path_end63"].detach().cpu().numpy().astype(np.int64)
    path_atom = batch["path_atom"].detach().cpu().numpy().astype(np.int64)
    atom_batch = batch["atom_batch"].detach().cpu().numpy().astype(np.int64)
    atom_off = batch["atom_off"].detach().cpu().numpy().astype(np.int64)
    n_atoms = int(batch["atom_feat"].shape[0])
    bt2 = path_cat % int(C.N_BOND_TYPES)
    bt2_end = path_end63 % int(C.N_BOND_TYPES)
    if not np.array_equal(bt2, bt2_end):
        raise SystemExit("bt2 from path_cat and path_end63 disagree")
    with torch.no_grad():
        t = local.path_gate(batch["path_cat"]).detach().cpu().numpy().reshape(-1)

    key = path_atom * int(C.N_BOND_TYPES) + bt2
    n_possible = n_atoms * int(C.N_BOND_TYPES)
    uniq, counts = np.unique(key, return_counts=True)
    # Group SD over path occurrences in each non-empty (root, bt2) group.
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    t_sorted = t[order]
    starts = np.searchsorted(key_sorted, uniq, side="left")
    ends = np.searchsorted(key_sorted, uniq, side="right")
    group_sd = np.asarray(
        [np.std(t_sorted[a:b], ddof=0) if b - a > 1 else 0.0 for a, b in zip(starts, ends)],
        dtype=np.float64,
    )
    multi = counts >= 2
    weighted_mean = float(np.sum(t) / t.size) if t.size else None
    weighted_sd = float(np.sqrt(np.mean((t - weighted_mean) ** 2))) if t.size else None
    # Decompose group SD weighted by group path count for the mechanism view.
    return {
        "n_atoms": n_atoms,
        "n_paths": int(path_cat.size),
        "n_possible_root_bt2_groups": int(n_possible),
        "n_nonempty_groups": int(uniq.size),
        "n_empty_groups": int(n_possible - uniq.size),
        "n_singleton_groups": int(np.sum(counts == 1)),
        "n_multi_groups": int(np.sum(counts >= 2)),
        "path_coverage_singleton": float(np.sum(counts[counts == 1]) / path_cat.size) if path_cat.size else None,
        "path_coverage_multi": float(np.sum(counts[counts >= 2]) / path_cat.size) if path_cat.size else None,
        "singleton_group_fraction_of_nonempty": float(np.sum(counts == 1) / uniq.size) if uniq.size else None,
        "multi_group_fraction_of_nonempty": float(np.sum(counts >= 2) / uniq.size) if uniq.size else None,
        "path_weighted_gate_mean": weighted_mean,
        "path_weighted_gate_sd": weighted_sd,
        "path_gate_min": float(t.min()) if t.size else None,
        "path_gate_max": float(t.max()) if t.size else None,
        "group_path_count_min": int(counts.min()) if counts.size else None,
        "group_path_count_median": float(np.median(counts)) if counts.size else None,
        "group_path_count_max": int(counts.max()) if counts.size else None,
        "group_gate_sd_all_nonempty": _summary(group_sd),
        "group_gate_sd_multi_only": _summary(group_sd[multi]),
        "fraction_nonempty_groups_with_gate_sd_gt_1e-12": float(np.mean(group_sd > 1e-12)) if group_sd.size else None,
        "fraction_multi_groups_with_gate_sd_gt_1e-12": float(np.mean(group_sd[multi] > 1e-12)) if multi.any() else None,
        "gate_vocabulary_note": "Gate statistics are over actual Valid path occurrences/groups, not unused vocabulary rows.",
    }


def vector_diff_summary(
    full: np.ndarray,
    marginal: np.ndarray,
    roots: np.ndarray,
    molecule_index: np.ndarray,
    atom_index: np.ndarray,
    row_id: np.ndarray,
    smiles: Sequence[str],
    label: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    diff = marginal - full
    d_abs = np.linalg.norm(diff, axis=1)
    d_rel = d_abs / (np.linalg.norm(full, axis=1) + np.linalg.norm(marginal, axis=1) + 1e-8)
    max_abs_full = np.max(np.abs(full), axis=1)
    max_abs_marg = np.max(np.abs(marginal), axis=1)
    threshold = COVERAGE_ATOL + COVERAGE_RTOL * np.maximum(max_abs_full, max_abs_marg)
    covered = np.max(np.abs(diff), axis=1) > threshold
    max_i = int(np.argmax(d_abs))
    mol_cover = np.unique(molecule_index[covered]) if covered.any() else np.asarray([], dtype=np.int64)
    summary = {
        "vector": label,
        "n_roots": int(full.shape[0]),
        "dim": int(full.shape[1]),
        "d_abs_quantiles": _summary(d_abs),
        "d_rel_quantiles": _summary(d_rel),
        "coverage_threshold_rule": "maxabs(delta) > 1e-6 + 1e-5 * max(maxabs(full), maxabs(marginal))",
        "n_covered_roots": int(covered.sum()),
        "root_coverage": float(covered.mean()) if covered.size else None,
        "n_covered_molecules": int(mol_cover.size),
        "molecule_coverage": float(mol_cover.size / len(np.unique(molecule_index))) if len(np.unique(molecule_index)) else None,
        "max_d_abs_root_global_index": int(roots[max_i]),
        "max_d_abs_molecule_index": int(molecule_index[max_i]),
        "max_d_abs_row_id": int(row_id[max_i]),
        "max_d_abs_atom_index_in_molecule": int(atom_index[max_i]),
        "max_d_abs_smiles": str(smiles[molecule_index[max_i]]),
        "max_d_abs": float(d_abs[max_i]),
        "max_d_rel": float(d_rel[max_i]),
        "mean_full_vector_norm": float(np.linalg.norm(full, axis=1).mean()),
        "mean_marginal_vector_norm": float(np.linalg.norm(marginal, axis=1).mean()),
        "continuous_changes_retained": True,
    }
    arrays = {
        "root_global": roots,
        "molecule_index": molecule_index,
        "atom_index_in_molecule": atom_index,
        "row_id": row_id,
        "d_abs": d_abs,
        "d_rel": d_rel,
        "covered": covered,
        "full_norm": np.linalg.norm(full, axis=1),
        "marginal_norm": np.linalg.norm(marginal, axis=1),
        "max_abs_full": max_abs_full,
        "max_abs_marginal": max_abs_marg,
    }
    return summary, arrays


def implementation_checks() -> dict[str, Any]:
    """Small algebra tests for empty/singleton groups, bias count and cross-molecule keys."""
    torch.manual_seed(0)
    rank = 3
    W = torch.randn(rank, rank)
    b = torch.randn(rank)
    # Empty groups: no occurrence, no bias.
    key_empty = torch.empty(0, dtype=torch.long)
    out_empty = M.marginal_joint_agg(
        torch.empty(0, rank), torch.empty(0, rank), key_empty, 3, W, b
    )
    c_empty = M.marginal_content_agg(
        torch.empty(0, 1), torch.empty(0, 4), key_empty, 3
    )
    empty_pass = bool(torch.count_nonzero(out_empty) == 0 and torch.count_nonzero(c_empty) == 0)
    # Singleton groups: marginal equals the corresponding FULL occurrence.
    u = torch.randn(1, rank)
    w = torch.randn(1, rank)
    key1 = torch.zeros(1, dtype=torch.long)
    joint_marg1 = M.marginal_joint_agg(u, w, key1, 1, W, b)
    joint_full1 = torch.nn.functional.linear(u * w, W, b)
    t = torch.tensor([[0.37]])
    z = torch.randn(1, 4)
    content_marg1 = M.marginal_content_agg(t, z, key1, 1)
    content_full1 = t * z
    singleton_pass = bool(torch.allclose(joint_marg1, joint_full1, atol=1e-6, rtol=1e-5) and torch.allclose(content_marg1, content_full1, atol=1e-6, rtol=1e-5))
    # Constant gate: C_full == C_marginal for any grouping.
    z2 = torch.randn(5, 4)
    key2 = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    const_t = torch.full((5, 1), 0.42)
    c_marg = M.marginal_content_agg(const_t, z2, key2, 3)
    c_full = torch.zeros(3, 4).index_add_(0, key2, const_t * z2)
    constant_pass = bool(torch.allclose(c_marg, c_full, atol=1e-6, rtol=1e-5))
    # Two molecules must not share a group even if their local root index is 0.
    u2 = torch.randn(2, rank)
    w2 = torch.randn(2, rank)
    # global roots 0 (molecule A) and 3 (molecule B), same bt2=0
    key3 = torch.tensor([0 * 5 + 0, 3 * 5 + 0], dtype=torch.long)
    out3 = M.marginal_joint_agg(u2, w2, key3, 20, W, b)
    expected = torch.zeros(20, rank)
    expected[0] = torch.nn.functional.linear(u2[0] * w2[0], W, b)
    expected[15] = torch.nn.functional.linear(u2[1] * w2[1], W, b)
    cross_pass = bool(torch.allclose(out3, expected, atol=1e-6, rtol=1e-5))
    # Bias must be counted n_g times, once per group.
    u3 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    w3 = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    key3b = torch.tensor([0, 0], dtype=torch.long)
    out_bias = M.marginal_joint_agg(u3, w3, key3b, 1, W, b)
    expected_bias = torch.nn.functional.linear(torch.tensor([[2.0, 0.0, 0.0]]), W, 2.0 * b)
    bias_pass = bool(torch.allclose(out_bias, expected_bias, atol=1e-6, rtol=1e-5))
    return {
        "empty_groups_zero": empty_pass,
        "singleton_equals_full": singleton_pass,
        "constant_gate_content_full_equals_marginal": constant_pass,
        "two_molecules_do_not_cross_group": cross_pass,
        "joint_bias_counted_n_g_times": bias_pass,
        "all_passed": all([empty_pass, singleton_pass, constant_pass, cross_pass, bias_pass]),
    }


def process_checkpoint(
    variant: str,
    seed: int,
    view: Any,
    batch: Mapping[str, torch.Tensor],
    rows: np.ndarray,
    y: np.ndarray,
    smiles: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model, ckpt, ckpt_rel = load_checkpoint(variant, seed, view)
    state_before = F.sha256_tensors(model.state_dict())
    batch_hash_before = F.sha256_tensors({k: v for k, v in batch.items() if torch.is_tensor(v)})
    saved = F.load_valid_predictions(F.MAIN_REPEAT, variant, seed)
    aligned = F.align_predictions(saved, {"rows": rows, "y": y})
    if not aligned["ok"]:
        raise SystemExit(f"saved prediction alignment failed for {variant} s{seed}: {aligned}")
    sigma = float(ckpt["target_norm"]["scale"])
    mu = float(ckpt["target_norm"]["mean"])
    with torch.no_grad():
        y_orig, aux_orig = model(batch, return_aux=True)
        y_orig_both = model(batch, s2_agg_override="marginal_bt2")
    pred_orig = y_orig.detach().numpy().astype(np.float64) * sigma + mu
    pred_orig_both = y_orig_both.detach().numpy().astype(np.float64) * sigma + mu
    local = attach_jc_local(model)
    state_after_attach = F.sha256_tensors(model.state_dict())
    if state_after_attach != state_before:
        raise SystemExit(f"state hash changed after attaching adapter for {variant} s{seed}")

    native_saved_diff = float(np.max(np.abs(pred_orig - aligned["raw"])))
    native_saved_rmse = F.metrics(pred_orig, y)["rmse"] - F.metrics(aligned["raw"], y)["rmse"]
    native_saved_mae = F.metrics(pred_orig, y)["mae"] - F.metrics(aligned["raw"], y)["mae"]
    native_parity = bool(
        native_saved_diff <= PARITY_ATOL
        and abs(native_saved_rmse) <= PARITY_METRIC_ATOL
        and abs(native_saved_mae) <= PARITY_METRIC_ATOL
        and np.array_equal(np.asarray(rows), np.asarray(aligned["rows"]))
    )

    # Evaluate the four conditions once each on this checkpoint.
    outputs: dict[str, dict[str, Any]] = {}
    native_h: np.ndarray | None = None
    native_parts: dict[str, np.ndarray] | None = None
    native_content: np.ndarray | None = None
    for name, joint_agg, content_agg in CONDITIONS:
        set_condition(model, joint_agg, content_agg)
        with torch.no_grad():
            raw_pred, aux = model(batch, return_aux=True)
        pred_std = raw_pred.detach().numpy().astype(np.float64)
        pred_raw = pred_std * sigma + mu
        contrib_raw = aux["contrib"].detach().numpy().astype(np.float64) * sigma
        rho_raw = aux["rho"].detach().numpy().astype(np.float64) * sigma
        h = aux["h"].detach().numpy().astype(np.float64)
        atom_sum_std = aux["contrib_atom"].detach().numpy().astype(np.float64)
        L_std_from_atoms = np.zeros(rows.size, dtype=np.float64)
        atom_batch_np = batch["atom_batch"].detach().numpy().astype(np.int64)
        np.add.at(L_std_from_atoms, atom_batch_np, atom_sum_std)
        recon_residual = pred_raw - (mu + contrib_raw + rho_raw)
        atom_recon_residual = contrib_raw - L_std_from_atoms * sigma
        if name == "NATIVE":
            native_h = h
            native_parts = extract_path_parts(local, aux)
            native_content = aux["content_agg"].detach().numpy().astype(np.float64)
        outputs[name] = {
            "pred_raw": pred_raw,
            "pred_std": pred_std,
            "contrib_raw": contrib_raw,
            "rho_raw": rho_raw,
            "h": h,
            "atom_sum_std": atom_sum_std,
            "L_atom_recon_residual": float(np.max(np.abs(atom_recon_residual))),
            "raw_recon_residual": float(np.max(np.abs(recon_residual))),
            "aux": aux,
        }

    assert native_h is not None and native_parts is not None and native_content is not None
    native = outputs["NATIVE"]
    native_pred = native["pred_raw"]
    native_metrics = F.metrics(native_pred, y)

    # BOTH parity against the frozen override path, using the same state/batch/dtype.
    both_pred_std = outputs["BOTH"]["pred_std"]
    both_max_abs = float(np.max(np.abs(both_pred_std - y_orig_both.detach().numpy().astype(np.float64))))
    both_allclose = bool(
        torch.allclose(
            torch.as_tensor(outputs["BOTH"]["pred_raw"]),
            torch.as_tensor(pred_orig_both),
            atol=BOTH_ATOL,
            rtol=BOTH_RTOL,
        )
    )
    both_parity = bool(both_allclose and both_max_abs <= BOTH_ATOL + BOTH_RTOL * max(np.max(np.abs(pred_orig_both)), 1.0))

    # T/E/K and the untouched vector must remain identical where required.
    parts_j_only = extract_path_parts(local, outputs["J_ONLY"]["aux"])
    parts_c_only = extract_path_parts(local, outputs["C_ONLY"]["aux"])
    invariance: dict[str, Any] = {}
    for key in ("type", "end", "closure"):
        invariance[f"J_ONLY_{key}_max_abs_diff"] = float(np.max(np.abs(parts_j_only[key] - native_parts[key])))
        invariance[f"C_ONLY_{key}_max_abs_diff"] = float(np.max(np.abs(parts_c_only[key] - native_parts[key])))
    invariance["J_ONLY_joint_changed_max_abs"] = float(np.max(np.abs(parts_j_only["joint"] - native_parts["joint"])))
    invariance["C_ONLY_joint_max_abs_diff"] = float(np.max(np.abs(parts_c_only["joint"] - native_parts["joint"])))
    invariance["J_ONLY_content_max_abs_diff"] = float(
        np.max(np.abs(outputs["J_ONLY"]["aux"]["content_agg"].detach().numpy() - native_content))
    )
    invariance["C_ONLY_content_changed_max_abs"] = float(
        np.max(np.abs(outputs["C_ONLY"]["aux"]["content_agg"].detach().numpy() - native_content))
    )
    invariance["C_ONLY_content_differs"] = bool(invariance["C_ONLY_content_changed_max_abs"] > 0)
    invariance["J_ONLY_content_unchanged"] = bool(invariance["J_ONLY_content_max_abs_diff"] <= 1e-7)
    invariance["C_ONLY_joint_unchanged"] = bool(invariance["C_ONLY_joint_max_abs_diff"] <= 1e-7)
    invariance["J_ONLY_tek_unchanged"] = bool(
        all(invariance[f"J_ONLY_{k}_max_abs_diff"] <= 1e-7 for k in ("type", "end", "closure"))
    )
    invariance["C_ONLY_tek_unchanged"] = bool(
        all(invariance[f"C_ONLY_{k}_max_abs_diff"] <= 1e-7 for k in ("type", "end", "closure"))
    )
    invariance["state_hash_unchanged"] = bool(state_after_attach == state_before)
    invariance["batch_hash_unchanged"] = bool(
        F.sha256_tensors({k: v for k, v in batch.items() if torch.is_tensor(v)}) == batch_hash_before
    )

    # Per-molecule response and decomposition.
    molecule_records: list[dict[str, Any]] = []
    condition_summary: dict[str, Any] = {}
    for name, joint_agg, content_agg in CONDITIONS:
        item = outputs[name]
        pred = item["pred_raw"]
        delta_pred = pred - native_pred
        delta_L = item["contrib_raw"] - native["contrib_raw"]
        delta_rho = item["rho_raw"] - native["rho_raw"]
        decomp_res = delta_pred - (delta_L + delta_rho)
        cond_metrics = F.metrics(pred, y)
        err_native = native_pred - y
        err_cond = pred - y
        abs_native = np.abs(err_native)
        abs_cond = np.abs(err_cond)
        summary = {
            "condition": name,
            "joint_agg": joint_agg,
            "content_agg": content_agg,
            "rmse": cond_metrics["rmse"],
            "mae": cond_metrics["mae"],
            "delta_rmse_vs_native": cond_metrics["rmse"] - native_metrics["rmse"],
            "delta_mae_vs_native": cond_metrics["mae"] - native_metrics["mae"],
            "delta_pred": _summary(delta_pred),
            "delta_pred_mean": float(np.mean(delta_pred)),
            "delta_pred_mean_abs": float(np.mean(np.abs(delta_pred))),
            "delta_pred_median_abs": float(np.median(np.abs(delta_pred))),
            "delta_pred_p90_abs": float(np.percentile(np.abs(delta_pred), 90)),
            "delta_pred_p95_abs": float(np.percentile(np.abs(delta_pred), 95)),
            "delta_pred_max_abs": float(np.max(np.abs(delta_pred))),
            "fraction_molecules_abs_delta_pred_gt_1e-4": float(np.mean(np.abs(delta_pred) > 1e-4)),
            "delta_L_mean_abs": float(np.mean(np.abs(delta_L))),
            "delta_rho_mean_abs": float(np.mean(np.abs(delta_rho))),
            "max_abs_decomposition_residual": float(np.max(np.abs(decomp_res))),
            "decomposition_residual_tolerance": DECOMP_ATOL,
            "decomposition_passed": bool(np.all(np.abs(decomp_res) <= DECOMP_ATOL + DECOMP_RTOL * np.maximum(np.abs(delta_pred), 1.0))),
            "delta_abs_error_mean": float(np.mean(abs_cond - abs_native)),
            "delta_se_mean": float(np.mean(err_cond * err_cond - err_native * err_native)),
            "n_ae_improved": int(np.sum(abs_cond < abs_native)),
            "n_ae_worsened": int(np.sum(abs_cond > abs_native)),
            "n_ae_equal": int(np.sum(abs_cond == abs_native)),
        }
        # Per-root ||delta h||_2 distribution.
        dh = item["h"] - native_h
        dh_norm = np.linalg.norm(dh, axis=1)
        summary["delta_h_per_root"] = _summary(dh_norm)
        summary["delta_h_max_root_global_index"] = int(np.argmax(dh_norm))
        summary["delta_h_max"] = float(dh_norm.max())
        condition_summary[name] = summary
        for pos in range(rows.size):
            molecule_records.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "condition": name,
                    "joint_agg": joint_agg,
                    "content_agg": content_agg,
                    "row_id": int(rows[pos]),
                    "smiles": smiles[pos],
                    "y": float(y[pos]),
                    "pred_native": float(native_pred[pos]),
                    "pred_condition": float(pred[pos]),
                    "delta_pred_raw": float(delta_pred[pos]),
                    "delta_L_raw": float(delta_L[pos]),
                    "delta_rho_raw": float(delta_rho[pos]),
                    "decomposition_residual": float(decomp_res[pos]),
                    "e_native": float(err_native[pos]),
                    "e_condition": float(err_cond[pos]),
                    "delta_abs_error": float(abs_cond[pos] - abs_native[pos]),
                    "delta_se": float(err_cond[pos] ** 2 - err_native[pos] ** 2),
                }
            )

    # J and C vector differences, per root.
    atom_batch_np = batch["atom_batch"].detach().numpy().astype(np.int64)
    atom_off_np = batch["atom_off"].detach().numpy().astype(np.int64)
    atom_index_in_mol = np.arange(atom_batch_np.size, dtype=np.int64) - atom_off_np[atom_batch_np]
    mol_row = rows[atom_batch_np]
    roots = np.arange(atom_batch_np.size, dtype=np.int64)
    j_full = native_parts["joint"]
    c_full = native_content
    j_marg = parts_j_only["joint"]
    c_marg = outputs["C_ONLY"]["aux"]["content_agg"].detach().numpy().astype(np.float64)
    j_summary, j_arrays = vector_diff_summary(j_full, j_marg, roots, atom_batch_np, atom_index_in_mol, mol_row, smiles, "J")
    c_summary, c_arrays = vector_diff_summary(c_full, c_marg, roots, atom_batch_np, atom_index_in_mol, mol_row, smiles, "C")

    # BOTH nonlinear residual.
    dp_both = outputs["BOTH"]["pred_raw"] - native_pred
    dp_j = outputs["J_ONLY"]["pred_raw"] - native_pred
    dp_c = outputs["C_ONLY"]["pred_raw"] - native_pred
    residual = dp_both - dp_j - dp_c
    nonlinear = {
        "definition": "r_i = delta_pred_BOTH_i - delta_pred_J_ONLY_i - delta_pred_C_ONLY_i",
        "summary": _summary(residual),
        "mean_signed": float(np.mean(residual)),
        "mean_abs": float(np.mean(np.abs(residual))),
        "median_abs": float(np.median(np.abs(residual))),
        "p90_abs": float(np.percentile(np.abs(residual), 90)),
        "p95_abs": float(np.percentile(np.abs(residual), 95)),
        "max_abs": float(np.max(np.abs(residual))),
        "note": "Non-additivity under this intervention; J/C separate effects are not an exact additive value split.",
    }

    gate = gate_stats(model, batch)

    summary_item: dict[str, Any] = {
        "variant": variant,
        "seed": seed,
        "best_epoch": int(ckpt["best_epoch"]),
        "checkpoint": ckpt_rel,
        "checkpoint_sha256": F.sha256_file(F.run_dir(F.MAIN_REPEAT, variant, seed) / "checkpoint.pt"),
        "target_norm": {"mean": mu, "scale": sigma},
        "native_metrics": native_metrics,
        "native_parity": {
            "saved_predictions_max_abs_diff": native_saved_diff,
            "saved_rmse_diff": native_saved_rmse,
            "saved_mae_diff": native_saved_mae,
            "prediction_tolerance": PARITY_ATOL,
            "metric_tolerance": PARITY_METRIC_ATOL,
            "passed": bool(native_parity),
        },
        "both_parity": {
            "max_abs_diff_standardised": both_max_abs,
            "allclose_atol": BOTH_ATOL,
            "allclose_rtol": BOTH_RTOL,
            "passed": bool(both_parity),
        },
        "invariance": invariance,
        "conditions": condition_summary,
        "gate": gate,
        "J_vector_diff": j_summary,
        "C_vector_diff": c_summary,
        "nonlinear_interaction": nonlinear,
        "state_hash_before_after": [state_before, state_after_attach],
        "native_decomposition": {
            "pred_raw_max_reconstruction_residual": native["raw_recon_residual"],
            "L_from_atom_sum_max_reconstruction_residual": native["L_atom_recon_residual"],
            "identity": "pred_raw = target_mean + sigma*(L+rho); L=sum_v f(h_v)",
        },
    }
    arrays = {
        "root_global": roots,
        "molecule_index": atom_batch_np,
        "atom_index_in_molecule": atom_index_in_mol,
        "row_id": mol_row,
        "d_abs_J": j_arrays["d_abs"],
        "d_rel_J": j_arrays["d_rel"],
        "covered_J": j_arrays["covered"],
        "d_abs_C": c_arrays["d_abs"],
        "d_rel_C": c_arrays["d_rel"],
        "covered_C": c_arrays["covered"],
        "dh_NATIVE": np.linalg.norm(outputs["NATIVE"]["h"] - native_h, axis=1),
        "dh_J_ONLY": np.linalg.norm(outputs["J_ONLY"]["h"] - native_h, axis=1),
        "dh_C_ONLY": np.linalg.norm(outputs["C_ONLY"]["h"] - native_h, axis=1),
        "dh_BOTH": np.linalg.norm(outputs["BOTH"]["h"] - native_h, axis=1),
    }
    return summary_item, molecule_records, arrays


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-unit-checks", action="store_true")
    args = ap.parse_args(argv)
    F.TRAINED_JC.mkdir(parents=True, exist_ok=True)

    checks = implementation_checks()
    if not checks["all_passed"]:
        raise SystemExit(f"implementation checks failed: {checks}")
    if not args.skip_unit_checks:
        F.write_json(F.TRAINED_JC / "implementation_checks.json", checks)

    ref = F.valid_reference()
    view = ref["view"]
    rows = np.asarray(ref["rows"], dtype=np.int64)
    y = np.asarray(ref["y"], dtype=np.float64)
    smiles = list(ref["smiles"])
    batch = D.to_device(D.PREP_DATA.collate(view.valid_cache, list(range(rows.size))), torch.device("cpu"))

    summaries: list[dict[str, Any]] = []
    molecule_records_all: list[dict[str, Any]] = []
    gate_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    arrays_dir = F.TRAINED_JC / "per_checkpoint"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    for variant in F.INTERVENTION_VARIANTS:
        for seed in F.SEEDS:
            F.C.log(f"[trained-jc] {variant} seed {seed}")
            try:
                summary, mol_rows, arrays = process_checkpoint(variant, seed, view, batch, rows, y, smiles)
            except Exception as exc:  # keep failure visible; do not hide it
                failures.append({"variant": variant, "seed": seed, "error": repr(exc)})
                F.write_json(F.TRAINED_JC / "failures.json", failures)
                continue
            if not summary["native_parity"]["passed"] or not summary["both_parity"]["passed"]:
                failures.append(
                    {
                        "variant": variant,
                        "seed": seed,
                        "error": "native or BOTH parity failed; intervention interpretation stopped for this checkpoint",
                        "native_parity": summary["native_parity"],
                        "both_parity": summary["both_parity"],
                    }
                )
                F.write_json(F.TRAINED_JC / "failures.json", failures)
                continue
            summaries.append(summary)
            molecule_records_all.extend(mol_rows)
            gate_rows.append({"variant": variant, "seed": seed, **summary["gate"]})
            np.savez_compressed(arrays_dir / f"root_arrays_{variant}_s{seed}.npz", **arrays)

    if failures:
        F.write_json(F.TRAINED_JC / "failures.json", failures)
    else:
        F.write_json(F.TRAINED_JC / "failures.json", [])

    # Cross-seed descriptive summary for each condition.
    by_condition: dict[str, Any] = {}
    for name, _, _ in CONDITIONS:
        items = [s for s in summaries if name in s["conditions"]]
        if not items:
            continue
        vals = [s["conditions"][name] for s in items]
        by_condition[name] = {
            "n_checkpoints": len(vals),
            "delta_rmse_per_checkpoint": [v["delta_rmse_vs_native"] for v in vals],
            "delta_rmse_mean": float(np.mean([v["delta_rmse_vs_native"] for v in vals])),
            "delta_mae_per_checkpoint": [v["delta_mae_vs_native"] for v in vals],
            "delta_mae_mean": float(np.mean([v["delta_mae_vs_native"] for v in vals])),
            "delta_pred_mean_abs_per_checkpoint": [v["delta_pred_mean_abs"] for v in vals],
            "delta_pred_max_abs_per_checkpoint": [v["delta_pred_max_abs"] for v in vals],
            "fraction_molecules_abs_delta_pred_gt_1e-4_per_checkpoint": [
                v["fraction_molecules_abs_delta_pred_gt_1e-4"] for v in vals
            ],
            "delta_rmse_signs": {
                s["variant"] + "_s" + str(s["seed"]): s["conditions"][name]["delta_rmse_vs_native"]
                for s in items
            },
            "note": "Per-seed values are shown first; the mean is descriptive over the six checkpoints, not a new independent sample.",
        }

    payload = {
        "schema": "lcrc-lipo-followup-trained-jc-v1",
        "created_utc": F.utcnow(),
        "git_commit": F.git_commit(),
        "repeat": F.MAIN_REPEAT,
        "valid_molecules": int(rows.size),
        "checkpoints": [f"{s['variant']}_s{s['seed']}" for s in summaries],
        "conditions": [x[0] for x in CONDITIONS],
        "n_condition_evaluations": int(len(summaries) * len(CONDITIONS)),
        "n_checkpoints_loaded": len(summaries),
        "new_training_runs": 0,
        "new_test_inference_or_rescoring": 0,
        "device": "cpu",
        "eval_mode": True,
        "gradients_disabled": True,
        "implementation_checks": checks,
        "summaries": summaries,
        "cross_seed_by_condition": by_condition,
        "gate_rows": gate_rows,
        "failures": failures,
        "interpretation_limits": [
            "These are frozen-weight dependence diagnostics, not retrained generalisation gains.",
            "Internal vector changes near zero mean the tested intervention was weak, not that pairing is useless.",
            "Large internal changes with small output changes mean low output sensitivity on these Valid samples, not that the module is removable everywhere.",
            "Even a better post-intervention Valid error does not replace a retrained comparison and does not justify deployment replacement.",
            "Initial gate=1 degradation and artificial gate probes are historical background, not this round's trained-gate result.",
        ],
    }
    F.write_json(F.TRAINED_JC / "summary.json", payload)
    F.write_json(F.TRAINED_JC / "gate_summary.json", gate_rows)
    F.write_csv(F.TRAINED_JC / "per_molecule_conditions.csv", molecule_records_all)
    print(f"completed {len(summaries)} checkpoints, {len(summaries) * len(CONDITIONS)} condition evaluations")
    for row in gate_rows:
        print(
            row["variant"], "s" + str(row["seed"]),
            "paths", row["n_paths"], "gate mean/sd", round(row["path_weighted_gate_mean"], 6), round(row["path_weighted_gate_sd"], 6),
            "multi groups", row["n_multi_groups"], "empty", row["n_empty_groups"],
        )
    for name, item in by_condition.items():
        print(name, "dR", round(item["delta_rmse_mean"], 6), "dA", round(item["delta_mae_mean"], 6), item["delta_rmse_per_checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
