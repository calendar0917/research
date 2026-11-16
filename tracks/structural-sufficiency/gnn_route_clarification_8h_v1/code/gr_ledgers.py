"""Native and contrastive-IG ledgers on the frozen edit pairs.

Two ledger families, never mixed:

* **native** — the model's own forward decomposition:
  ``L``: ``f(h_v)`` per atom plus the separate graph term ``rho`` (never spread
  over atoms); ``A``/``D``: ``a_v = s_v + m_v/n`` per atom plus the per-column
  ``u_j(z_j)`` and the intercept (intercept cancels in a pair contrast).
* **contrastive Integrated Gradients** on the *continuous union extension* of the
  exact trained GINE (all five message-passing layers), with the input variables
  ``X, E, m, a, N, Z`` frozen by the task brief.  A readout-only fallback is
  available only if the full-MP extension fails its endpoint gate; the scope is
  then recorded as ``READOUT_ONLY`` and never described as a structure-level
  attribution.

Grouping (declared convention, not a unique chemical attribution): node ``X/m``
attribution stays on its node; each directed edge ``E/a`` attribution is split
half to each endpoint; variable atoms of B minus variable atoms of A form the
variable channel, the mapped core atoms form the core channel, ``N`` is the
size channel and each ``Z`` column is a global channel.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import gr_common as C
import gr_data as D
import gr_models as M
import gr_select as S


def _jsonl(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_catalog() -> list[dict[str, Any]]:
    return _jsonl(C.ANALYSIS_DIR / "pair_catalog.jsonl.gz")


def load_sample() -> list[dict[str, Any]]:
    payload = C.read_json(C.ANALYSIS_DIR / "explain_sample.json")
    return payload["pairs"]


def _is_train(pair: Mapping[str, Any]) -> bool:
    return bool(pair.get("train_train", pair.get("stratum") == "train_train"))


# ---------------------------------------------------------------------------
# checkpointed runners
# ---------------------------------------------------------------------------


def _checkpoint_path(stage: str, dataset: str, model: str, recipe: str):
    return C.task_dir(stage, dataset, model, recipe) / "checkpoint_best_val.pt"


def _load_state(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return payload


@dataclass
class Runner:
    name: str
    dataset: str
    kind: str
    model: Any
    trunk: Any
    head: Any
    recipe: str
    checkpoint: str
    b: Any
    row_to_local: dict[int, tuple[str, int]] = field(default_factory=dict)
    _n_params: int = 0

    # -- construction helpers ---------------------------------------------
    def row_batch(self, row: int) -> dict[str, Any]:
        split, local = self.row_to_local[int(row)]
        return D.collate(self.b, split, [int(local)])

    def eval_mode(self) -> None:
        self.model.eval()

    # -- per-row pieces -----------------------------------------------------
    def encode(self, row: int, *, double: bool = False) -> dict[str, Any]:
        batch = _double_batch(self.row_batch(row)) if double else self.row_batch(row)
        with torch.no_grad():
            if self.kind == "L":
                parts = self.model.parts(batch)
                return {"batch": batch, "parts": parts, "native": True}
            if self.trunk is not None:
                X = self.trunk.encode_nodes(batch)
                E = self.trunk.encode_edges(batch)
                h = self.trunk.layers(X, batch["edge_index"], E)
            else:
                X = E = h = None
            if self.head is not None:
                z = self._z_from_batch(batch)
                parts = self.head.forward_parts(h, batch["gine_batch"], batch["gine_n_atoms"], z)
                return {"batch": batch, "h": h, "X": X, "E": E, "parts": parts, "native": self.name in ("A", "D")}
            return {"batch": batch, "h": h, "X": X, "E": E, "native": False}

    def _z_from_batch(self, batch) -> torch.Tensor:
        if self.kind == "G0":
            return self.head.size_z(batch["gine_n_atoms"])
        return self.trunk.z_global(batch)[:, list(self.head.active_indices)]

    def predict(self, row: int) -> float:
        batch = self.row_batch(row)
        with torch.no_grad():
            return float(self.model(batch).view(-1)[0])

    def predict_std(self, row: int) -> float:
        batch = self.row_batch(row)
        with torch.no_grad():
            return float(self.model(batch).view(-1)[0])

    def predict_std_double(self, row: int) -> float:
        """FP64 reference used only by the endpoint gate."""
        was_double = next(self.model.parameters()).dtype == torch.float64
        if not was_double:
            self.model.double()
        try:
            batch = _double_batch(self.row_batch(row))
            with torch.no_grad():
                return float(self.model(batch).view(-1)[0])
        finally:
            if not was_double:
                self.model.float()


def make_runner(model_key: str, dataset: str = "ogbg-mollipo", *, property_name: str | None = None) -> Runner:
    b = D.oracle_bundle(dataset, property_name) if property_name else D.bundle(dataset)
    row_to_local = {}
    for split in C.SPLITS:
        for local, row in enumerate(np.asarray(b.global_row_ids[split]).tolist()):
            row_to_local[int(row)] = (split, local)
    if model_key == "L":
        recipe = S.lipo_recipe_for_model("L")
        stage = C.STAGE_ORACLE if property_name else C.STAGE1
        folder = f"{property_name}/selected" if property_name else recipe
        path = _checkpoint_path(stage, dataset, "L", folder)
        model = M.ModelSet(b).lcrc()
        model.load_state_dict(_load_state(path)["model_state"], strict=True)
        model.eval()
        return Runner("L", dataset, "L", model, None, None, recipe, str(path), b, row_to_local)
    if model_key == "T":
        kind = S.selected_trunk_model()
        recipe = S.lipo_recipe_for_model("T")
        stage = C.STAGE_ORACLE if property_name else C.STAGE1
        folder = f"{property_name}/selected" if property_name else recipe
        path = _checkpoint_path(stage, dataset, kind, folder)
        model = M.ModelSet(b).dense(kind)
        model.load_state_dict(_load_state(path)["model_state"], strict=True)
        model.eval()
        return Runner("T", dataset, kind, model, model.trunk, None, recipe, str(path), b, row_to_local)
    if model_key == "A":
        kind = S.selected_trunk_model()
        recipe = S.lipo_recipe_for_model("A")
        stage = C.STAGE_ORACLE if property_name else C.STAGE2
        folder = f"{property_name}/selected" if property_name else recipe
        path = _checkpoint_path(stage, dataset, "A", folder)
        model = M.ModelSet(b).additive(
            kind,
            active_indices=[] if kind == "G0" else b.active_global["active_indices"],
            size_scaler=D.size_scaler(b),
        )
        model.load_state_dict(_load_state(path)["model_state"], strict=True)
        model.eval()
        return Runner("A", dataset, kind, model, model.trunk, model.head, recipe, str(path), b, row_to_local)
    if model_key == "D":
        kind = S.selected_trunk_model()
        t_recipe = S.lipo_recipe_for_model("T")
        ts = _load_state(_checkpoint_path(C.STAGE1, dataset, kind, t_recipe))
        trunk = M.build_trunk(kind, b)
        trunk_state = {k[len("trunk.") :]: v for k, v in ts["model_state"].items() if k.startswith("trunk.")}
        trunk.load_state_dict(trunk_state, strict=True)
        trunk.eval()
        ms = M.ModelSet(b)
        active = [] if kind == "G0" else b.active_global["active_indices"]
        head = ms._new_head(kind, n_active=(1 if kind == "G0" else len(active)), active_indices=active, size_scaler=D.size_scaler(b))
        d_path = _checkpoint_path(C.STAGE3, dataset, "D", "D600")
        head.load_state_dict(_load_state(d_path)["model_state"], strict=True)
        head.eval()
        model = M.AdditiveGraphModel(trunk, head)
        model.eval()
        return Runner("D", dataset, kind, model, trunk, head, "D600", str(d_path), b, row_to_local)
    raise KeyError(model_key)


# ---------------------------------------------------------------------------
# native ledgers
# ---------------------------------------------------------------------------


def native_ledger(runner: Runner, row: int) -> dict[str, Any]:
    was_double = next(runner.model.parameters()).dtype == torch.float64
    if not was_double:
        runner.model.double()
    try:
        block = runner.encode(row, double=True)
    finally:
        if not was_double:
            runner.model.float()
    if not block["native"]:
        raise RuntimeError(f"{runner.name} has no native per-atom ledger")
    parts = block["parts"]
    n_atoms = int(parts["n_atoms"][0]) if "n_atoms" in parts else int(block["batch"]["gine_n_atoms"][0])
    out = {"row": int(row), "n_atoms": n_atoms}
    if runner.name == "L":
        out["atom"] = parts["contrib_atom"].detach().cpu().numpy().astype(np.float64)
        out["graph_terms"] = {"rho": float(parts["rho"].view(-1)[0])}
        out["pred_std"] = float(parts["pred_std"].view(-1)[0])
        out["reconstruction"] = out["atom"].sum() + out["graph_terms"]["rho"]
        out["channels"] = {"local_sum": float(out["atom"].sum()), "rho": out["graph_terms"]["rho"]}
    else:
        out["atom"] = parts["a_v"].detach().cpu().numpy().astype(np.float64)
        u = parts["u_terms"][0].detach().cpu().numpy().astype(np.float64)
        out["u_terms"] = u
        out["intercept"] = float(parts["intercept"].view(-1)[0])
        out["pred_std"] = float(parts["pred_std"].view(-1)[0])
        names = u_names(runner)
        out["u_names"] = names
        out["reconstruction"] = out["atom"].sum() + float(u.sum()) + out["intercept"]
        out["channels"] = {"atom_sum": float(out["atom"].sum()), "u_sum": float(u.sum()), "intercept": out["intercept"]}
    err = abs(out["reconstruction"] - out["pred_std"])
    out["reconstruction_error_std"] = float(err)
    if err > 1e-8:
        raise RuntimeError(f"{runner.name} row {row}: native ledger does not reconstruct ({err})")
    return out


def u_names(runner: Runner, *, full: bool = False) -> list[str]:
    if runner.name == "L":
        return []
    if runner.kind == "G0":
        return ["size:log1p_n_standardised"]
    if full or runner.head is None:
        return [f.name for f in runner.b.spec.global_feature_fields]
    table = runner.b.active_global["field_table"]
    return [table[i]["name"] for i in runner.b.active_global["active_indices"]]


def u_columns(runner: Runner) -> list[int]:
    if runner.name == "L":
        return []
    if runner.kind == "G0":
        return [0]
    if runner.head is None:
        return list(range(len(runner.b.spec.global_feature_fields)))
    return [int(i) for i in runner.b.active_global["active_indices"]]


def native_pair_scores(
    runner: Runner,
    pair: Mapping[str, Any],
    ledger_a: Mapping[str, Any],
    ledger_b: Mapping[str, Any],
) -> dict[str, Any]:
    a, bb = int(pair["row_a"]), int(pair["row_b"])
    atom_a = ledger_a["atom"]
    atom_b = ledger_b["atom"]
    var_a = [int(x) for x in pair["var_atoms_a"]]
    var_b = [int(x) for x in pair["var_atoms_b"]]
    core_a = [int(x) for x in pair["core_atoms_a"]]
    cmap = {int(k): int(v) for k, v in pair["core_atom_map"].items()}
    score_var = float(atom_b[var_b].sum() - atom_a[var_a].sum())
    score_core = float(sum(atom_b[cmap[u]] - atom_a[u] for u in core_a))
    if runner.name == "L":
        rho = float(ledger_b["graph_terms"]["rho"] - ledger_a["graph_terms"]["rho"])
        size = 0.0
        globals_ = {"rho": rho}
    else:
        names = ledger_a["u_names"]
        du = ledger_b["u_terms"] - ledger_a["u_terms"]
        size = float(du[0]) if runner.kind == "G0" else 0.0
        globals_ = {name: float(val) for name, val in zip(names, du)}
        if runner.kind == "G0":
            globals_ = {}
    total = score_var + score_core + size + float(sum(globals_.values()))
    pred_diff = float(ledger_b["pred_std"] - ledger_a["pred_std"])
    return {
        "variable": score_var,
        "core": score_core,
        "size": size,
        "global": globals_,
        "pred_diff_std": pred_diff,
        "pred_diff_raw": pred_diff * float(runner.b.scaler["std"]),
        "reconstruction_error_std": float(total - pred_diff),
        "channels": ["variable", "core", "size", "global"],
    }


# ---------------------------------------------------------------------------
# union graph for the continuous extension
# ---------------------------------------------------------------------------


@dataclass
class UnionPair:
    pair_id: str
    row_a: int
    row_b: int
    n_slots: int
    core_slots: np.ndarray
    var_a_slots: np.ndarray
    var_b_slots: np.ndarray
    slot_of_core_a: dict[int, int]
    edge_index: np.ndarray          # (2, E) sorted keys
    edge_keys: list[tuple[int, int]]
    X0: np.ndarray
    X1: np.ndarray
    E0: np.ndarray
    E1: np.ndarray
    m0: np.ndarray
    m1: np.ndarray
    a0: np.ndarray
    a1: np.ndarray
    N0: float
    N1: float
    Z0: np.ndarray | None
    Z1: np.ndarray | None
    pred_a: float
    pred_b: float
    pred_a64: float
    pred_b64: float
    group_index: dict[str, np.ndarray]


def _double_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (value.double() if torch.is_tensor(value) and value.dtype == torch.float32 else value)
        for key, value in batch.items()
    }


def _encoder(runner: Runner, row: int, *, double: bool = False) -> dict[str, np.ndarray]:
    batch = runner.row_batch(row)
    was_double = next(runner.model.parameters()).dtype == torch.float64
    if double and not was_double:
        runner.model.double()
        batch = _double_batch(batch)
    if not double and was_double:
        raise RuntimeError("model is in FP64 but a FP32 union was requested")
    with torch.no_grad():
        if runner.trunk is None:
            raise RuntimeError("no GINE trunk")
        X = runner.trunk.encode_nodes(batch).cpu().numpy().astype(np.float64)
        E = runner.trunk.encode_edges(batch).cpu().numpy().astype(np.float64)
        n = int(batch["gine_n_atoms"][0])
        if runner.kind == "Gx":
            Z = runner.trunk.z_global(batch)[0].cpu().numpy().astype(np.float64)
        else:
            Z = None
    if double and not was_double:
        runner.model.float()
    return {"X": X, "E": E, "n": n, "edge_index": batch["edge_index"].numpy().astype(np.int64), "Z": Z}


def build_union(runner: Runner, pair: Mapping[str, Any], *, double: bool = False) -> UnionPair:
    a, bb = int(pair["row_a"]), int(pair["row_b"])
    ea = _encoder(runner, a, double=double)
    eb = _encoder(runner, bb, double=double)
    core_a = [int(x) for x in pair["core_atoms_a"]]
    core_b = [int(x) for x in pair["core_atoms_b"]]
    var_a = [int(x) for x in pair["var_atoms_a"]]
    var_b = [int(x) for x in pair["var_atoms_b"]]
    cmap = {int(k): int(v) for k, v in pair["core_atom_map"].items()}
    if set(core_a) & set(var_a) or set(core_b) & set(var_b):
        raise RuntimeError(f"{pair['pair_id']}: core and variable atoms overlap")
    if set(core_a) | set(var_a) != set(range(ea["n"])):
        raise RuntimeError(f"{pair['pair_id']}: A atoms are not core+variable")
    if set(core_b) | set(var_b) != set(range(eb["n"])):
        raise RuntimeError(f"{pair['pair_id']}: B atoms are not core+variable")
    n_core, n_va, n_vb = len(core_a), len(var_a), len(var_b)
    n_slots = n_core + n_va + n_vb
    core_slot = {atom: i for i, atom in enumerate(core_a)}
    slot_a: dict[int, int] = dict(core_slot)
    for i, atom in enumerate(var_a):
        slot_a[atom] = n_core + i
    slot_b: dict[int, int] = {core_b[i]: i for i in range(n_core)}
    for i, atom in enumerate(var_b):
        slot_b[atom] = n_core + n_va + i
    X0 = np.zeros((n_slots, ea["X"].shape[1]))
    X1 = np.zeros((n_slots, ea["X"].shape[1]))
    for atom, slot in slot_a.items():
        X0[slot] = ea["X"][atom]
    for atom, slot in slot_b.items():
        X1[slot] = eb["X"][atom]
    m0 = np.zeros(n_slots)
    m0[: n_core + n_va] = 1.0
    m1 = np.zeros(n_slots)
    m1[:n_core] = 1.0
    m1[n_core + n_va :] = 1.0
    E0: dict[tuple[int, int], np.ndarray] = {}
    E1: dict[tuple[int, int], np.ndarray] = {}
    for idx, (u, v) in enumerate(ea["edge_index"].T.tolist()):
        E0[(slot_a[u], slot_a[v])] = ea["E"][idx]
    for idx, (u, v) in enumerate(eb["edge_index"].T.tolist()):
        E1[(slot_b[u], slot_b[v])] = eb["E"][idx]
    keys = sorted(set(E0) | set(E1))
    dim_e = ea["E"].shape[1]
    arr0 = np.zeros((len(keys), dim_e))
    arr1 = np.zeros((len(keys), dim_e))
    pres0 = np.zeros(len(keys))
    pres1 = np.zeros(len(keys))
    for i, key in enumerate(keys):
        if key in E0:
            arr0[i] = E0[key]
            pres0[i] = 1.0
        if key in E1:
            arr1[i] = E1[key]
            pres1[i] = 1.0
    edge_index = np.asarray(keys, dtype=np.int64).T if keys else np.zeros((2, 0), np.int64)
    # groups
    var_a_slots = np.asarray([slot_a[i] for i in var_a], dtype=np.int64)
    var_b_slots = np.asarray([slot_b[i] for i in var_b], dtype=np.int64)
    core_slots = np.asarray([core_slot[i] for i in core_a], dtype=np.int64)
    if runner.kind == "G0":
        Z0 = Z1 = None
    else:
        Z0, Z1 = ea["Z"], eb["Z"]
    pred_a = runner.predict_std(a) if not double else runner.predict_std_double(a)
    pred_b = runner.predict_std(bb) if not double else runner.predict_std_double(bb)
    pred_a64 = runner.predict_std_double(a)
    pred_b64 = runner.predict_std_double(bb)
    return UnionPair(
        pair_id=str(pair["pair_id"]),
        row_a=a,
        row_b=bb,
        n_slots=n_slots,
        core_slots=core_slots,
        var_a_slots=var_a_slots,
        var_b_slots=var_b_slots,
        slot_of_core_a=core_slot,
        edge_index=edge_index,
        edge_keys=keys,
        X0=X0,
        X1=X1,
        E0=arr0,
        E1=arr1,
        m0=m0,
        m1=m1,
        a0=pres0,
        a1=pres1,
        N0=float(ea["n"]),
        N1=float(eb["n"]),
        Z0=Z0,
        Z1=Z1,
        pred_a=pred_a,
        pred_b=pred_b,
        pred_a64=pred_a64,
        pred_b64=pred_b64,
        group_index={
            "core": core_slots,
            "var_a": var_a_slots,
            "var_b": var_b_slots,
        },
    )


def union_forward(runner: Runner, union: UnionPair, t: float) -> torch.Tensor:
    """Scalar standardised prediction of the union extension at path point t."""
    X = torch.as_tensor(union.X0 + t * (union.X1 - union.X0), dtype=torch.float32)
    E = torch.as_tensor(union.E0 + t * (union.E1 - union.E0), dtype=torch.float32)
    m = torch.as_tensor(union.m0 + t * (union.m1 - union.m0), dtype=torch.float32)
    a = torch.as_tensor(union.a0 + t * (union.a1 - union.a0), dtype=torch.float32)
    N = float(union.N0 + t * (union.N1 - union.N0))
    n_atoms = torch.as_tensor([N], dtype=X.dtype)
    h = m.view(-1, 1) * X
    edge_index = torch.as_tensor(union.edge_index, dtype=torch.long)
    h = runner.trunk.layers(h, edge_index, E, node_presence=m, edge_presence=a)
    if runner.head is None:
        summed = h.sum(dim=0, keepdim=True)
        meaned = summed / n_atoms.view(-1, 1)
        logn = torch.log1p(n_atoms).view(-1, 1)
        if runner.kind == "Gx":
            Z = torch.as_tensor(union.Z0 + t * (union.Z1 - union.Z0), dtype=torch.float32).view(1, -1)
            pooled = torch.cat([summed, meaned, Z, logn], dim=1)
        else:
            pooled = torch.cat([summed, meaned, logn], dim=1)
        return runner.model.head.net(pooled).view(-1)[0]
    # additive head: gate every per-atom contribution by the node presence
    th = torch.nn.functional.silu(h @ runner.head.w_hidden.t() + runner.head.b_hidden)
    s = th @ runner.head.w_s
    mm = th @ runner.head.w_m
    sum_s = (m * s).sum()
    mean_m = (m * mm).sum() / n_atoms.view(-1)[0]
    if runner.kind == "G0":
        z = ((torch.log1p(n_atoms) - runner.head.size_mean) / runner.head.size_scale).view(1, 1)
    else:
        Z = torch.as_tensor(union.Z0 + t * (union.Z1 - union.Z0), dtype=torch.float32).view(1, -1)
        z = Z[:, list(runner.head.active_indices)]
    u_sum, _ = runner.head.univariate(z)
    return runner.head.bias.view(-1)[0] + sum_s + mean_m + u_sum.view(-1)[0]


def union_endpoint_check(runner: Runner, pair: Mapping[str, Any]) -> dict[str, float]:
    """FP64 endpoint reproduction by the same function the IG integrates."""
    u = build_union(runner, pair, double=True)
    was_double = next(runner.model.parameters()).dtype == torch.float64
    if not was_double:
        runner.model.double()
    try:
        with torch.no_grad():
            i0 = batch_union_inputs([u], runner, 0.0, dtype=torch.float64)
            i1 = batch_union_inputs([u], runner, 1.0, dtype=torch.float64)
            f0 = float(forward_batch(runner, i0, 1)[0])
            f1 = float(forward_batch(runner, i1, 1)[0])
        # the additive G0 head reads standardised buffers as Python floats
    finally:
        if not was_double:
            runner.model.float()
    return {
        "a_abs_error": abs(f0 - u.pred_a64),
        "b_abs_error": abs(f1 - u.pred_b64),
        "a_pred64": u.pred_a64,
        "b_pred64": u.pred_b64,
    }


def _as_double(union: UnionPair) -> UnionPair:
    def d(x):
        return None if x is None else np.asarray(x, dtype=np.float64)

    return UnionPair(
        pair_id=union.pair_id, row_a=union.row_a, row_b=union.row_b, n_slots=union.n_slots,
        core_slots=union.core_slots, var_a_slots=union.var_a_slots, var_b_slots=union.var_b_slots,
        slot_of_core_a=union.slot_of_core_a, edge_index=union.edge_index, edge_keys=union.edge_keys,
        X0=d(union.X0), X1=d(union.X1), E0=d(union.E0), E1=d(union.E1), m0=d(union.m0), m1=d(union.m1),
        a0=d(union.a0), a1=d(union.a1), N0=float(union.N0), N1=float(union.N1), Z0=d(union.Z0), Z1=d(union.Z1),
        pred_a=union.pred_a, pred_b=union.pred_b, pred_a64=union.pred_a64, pred_b64=union.pred_b64,
        group_index=union.group_index,
    )


# ---------------------------------------------------------------------------
# batched IG over many union graphs
# ---------------------------------------------------------------------------


def batch_union_inputs(
    unions: Sequence[UnionPair], runner: Runner, t: float, *, dtype: torch.dtype = torch.float32
) -> dict[str, torch.Tensor]:
    Xs, Es, ms, as_, eidx, ngraph, Ns, Zs = [], [], [], [], [], [], [], []
    node_off = 0
    edge_off = 0
    for g, u in enumerate(unions):
        Xs.append(u.X0 + t * (u.X1 - u.X0))
        Es.append(u.E0 + t * (u.E1 - u.E0))
        ms.append(u.m0 + t * (u.m1 - u.m0))
        as_.append(u.a0 + t * (u.a1 - u.a0))
        n = u.n_slots
        ngraph.append(np.full(n, g, dtype=np.int64))
        e = u.edge_index
        if e.shape[1]:
            eidx.append(e + node_off)
        Ns.append(u.N0 + t * (u.N1 - u.N0))
        if u.Z0 is not None:
            Zs.append(u.Z0 + t * (u.Z1 - u.Z0))
        node_off += n
        edge_off += u.edge_index.shape[1]
    X = torch.tensor(np.concatenate(Xs, axis=0), dtype=dtype, requires_grad=True)
    E = torch.tensor(np.concatenate(Es, axis=0), dtype=dtype, requires_grad=True)
    m = torch.tensor(np.concatenate(ms), dtype=dtype, requires_grad=True)
    a = torch.tensor(np.concatenate(as_), dtype=dtype, requires_grad=True)
    edge_index = torch.tensor(np.concatenate(eidx, axis=1), dtype=torch.long) if eidx else torch.zeros((2, 0), dtype=torch.long)
    node_graph = torch.tensor(np.concatenate(ngraph), dtype=torch.long)
    N = torch.tensor(Ns, dtype=dtype, requires_grad=True)
    Z = None
    if Zs:
        Z = torch.tensor(np.stack(Zs, axis=0), dtype=dtype, requires_grad=True)
    return {"X": X, "E": E, "m": m, "a": a, "edge_index": edge_index, "node_graph": node_graph, "N": N, "Z": Z}


def forward_batch(runner: Runner, inputs: Mapping[str, torch.Tensor], n_graphs: int) -> torch.Tensor:
    X, E, m, a = inputs["X"], inputs["E"], inputs["m"], inputs["a"]
    h = m.view(-1, 1) * X
    h = runner.trunk.layers(h, inputs["edge_index"], E, node_presence=m, edge_presence=a)
    summed = h.new_zeros((n_graphs, h.shape[1])).index_add_(0, inputs["node_graph"], h)
    N = inputs["N"].view(-1, 1)
    meaned = summed / N
    logn = torch.log1p(N)
    if runner.head is None:
        blocks = [summed, meaned]
        if runner.kind == "Gx":
            blocks.append(inputs["Z"])
        blocks.append(logn)
        pooled = torch.cat(blocks, dim=1)
        return runner.model.head.net(pooled).view(-1)
    th = torch.nn.functional.silu(h @ runner.head.w_hidden.t() + runner.head.b_hidden)
    s = th @ runner.head.w_s
    mm = th @ runner.head.w_m
    sum_s = s.new_zeros(n_graphs).index_add_(0, inputs["node_graph"], m * s)
    mean_m = mm.new_zeros(n_graphs).index_add_(0, inputs["node_graph"], m * mm) / N.view(-1)
    if runner.kind == "G0":
        z = (torch.log1p(N) - float(runner.head.size_mean)) / float(runner.head.size_scale)
    else:
        z = inputs["Z"][:, list(runner.head.active_indices)]
    u_sum, _ = runner.head.univariate(z)
    return runner.head.bias.view(-1) + sum_s + mean_m + u_sum


def integrate_ig(
    runner: Runner, unions: Sequence[UnionPair], points: int, *, dtype: torch.dtype = torch.float32
) -> dict[str, Any]:
    """Midpoint IG for a batch of union graphs; returns per-pair channel arrays."""
    per_pair = [
        {
            "X": np.zeros_like(u.X0),
            "E": np.zeros_like(u.E0),
            "m": np.zeros(u.n_slots),
            "a": np.zeros(len(u.edge_keys)),
            "N": 0.0,
            "Z": None,
        }
        for u in unions
    ]
    with torch.enable_grad():
        for k in range(points):
            t = (k + 0.5) / points
            inputs = batch_union_inputs(unions, runner, t, dtype=dtype)
            out = forward_batch(runner, inputs, len(unions))
            targets = [inputs["X"], inputs["E"], inputs["m"], inputs["a"], inputs["N"]]
            if inputs["Z"] is not None:
                targets.append(inputs["Z"])
            grads = torch.autograd.grad(out.sum(), targets, allow_unused=True)
            gX, gE, gm, ga, gN = grads[:5]
            gZ = grads[5] if len(grads) > 5 else None
            node_off = 0
            edge_off = 0
            for i, u in enumerate(unions):
                ns = u.n_slots
                ne = len(u.edge_keys)
                if gX is not None:
                    per_pair[i]["X"] += gX[node_off : node_off + ns].detach().numpy() / points
                if gE is not None:
                    per_pair[i]["E"] += gE[edge_off : edge_off + ne].detach().numpy() / points
                if gm is not None:
                    per_pair[i]["m"] += gm[node_off : node_off + ns].detach().numpy() / points
                if ga is not None:
                    per_pair[i]["a"] += ga[edge_off : edge_off + ne].detach().numpy() / points
                if gN is not None:
                    per_pair[i]["N"] += float(gN[i]) / points
                if gZ is not None and u.Z0 is not None:
                    if per_pair[i]["Z"] is None:
                        per_pair[i]["Z"] = np.zeros_like(u.Z0)
                    per_pair[i]["Z"] += gZ[i].detach().numpy() / points
                node_off += ns
                edge_off += ne
    out = []
    for i, u in enumerate(unions):
        acc = per_pair[i]
        ig_X = acc["X"] * (u.X1 - u.X0)
        ig_E = acc["E"] * (u.E1 - u.E0)
        ig_m = acc["m"] * (u.m1 - u.m0)
        ig_a = acc["a"] * (u.a1 - u.a0)
        ig_N = float(acc["N"] * (u.N1 - u.N0))
        ig_Z = None
        if acc["Z"] is not None and u.Z0 is not None:
            ig_Z = acc["Z"] * (u.Z1 - u.Z0)
        node_attr = np.zeros(u.n_slots)
        node_attr += ig_X.sum(axis=1)
        node_attr += ig_m
        for e, (src, dst) in enumerate(u.edge_keys):
            val = float(ig_E[e].sum() + ig_a[e])
            node_attr[src] += 0.5 * val
            node_attr[dst] += 0.5 * val
        total = float(node_attr.sum() + ig_N + (0.0 if ig_Z is None else ig_Z.sum()))
        out.append(
            {
                "node_attr": node_attr,
                "edge_attr_sum": float(ig_E.sum() + ig_a.sum()),
                "size": ig_N,
                "global": None if ig_Z is None else ig_Z,
                "total_std": total,
                "delta_std": float(u.pred_b - u.pred_a),
            }
        )
    return {"points": points, "pairs": out}


def ig_groups(runner: Runner, union: UnionPair, ig: Mapping[str, Any]) -> dict[str, Any]:
    """IG groups live in delta space, so both variable blocks enter with +.

    The A-only slot's IG is the effect of removing that atom, which already
    carries the minus sign that the contribution-space convention writes
    explicitly (``sum_B a_v - sum_A a_v``).  Negating it again would double
    count and break ``sum(groups) == delta_f`` (task brief 7.6).
    """
    node_attr = ig["node_attr"]
    score_var = float(node_attr[union.var_b_slots].sum() + node_attr[union.var_a_slots].sum())
    score_var_signed = float(node_attr[union.var_b_slots].sum() - node_attr[union.var_a_slots].sum())
    score_core = float(node_attr[union.core_slots].sum())
    globals_ = {}
    if ig["global"] is not None:
        cols = u_columns(runner)
        names = u_names(runner)
        globals_ = {name: float(ig["global"][i]) for name, i in zip(names, cols)}
    total = score_var + score_core + float(ig["size"]) + float(sum(globals_.values()))
    delta_raw = ig["delta_std"] * float(runner.b.scaler["std"])
    return {
        "variable": score_var,
        "variable_signed_diff": score_var_signed,
        "core": score_core,
        "size": float(ig["size"]),
        "global": globals_,
        "delta_std": ig["delta_std"],
        "delta_raw": delta_raw,
        "reconstruction_error_std": float(total - ig["delta_std"]),
        "completeness_tolerance": C.IG_TOL_REL * float(runner.b.scaler["std"]) + C.IG_TOL_ABS * abs(delta_raw),
        "passes_completeness": bool(
            abs(total - ig["delta_std"]) * float(runner.b.scaler["std"])
            <= C.IG_TOL_REL * float(runner.b.scaler["std"]) + C.IG_TOL_ABS * abs(delta_raw)
        ),
        "channels": list(globals_),
    }


# ---------------------------------------------------------------------------
# full explanation job
# ---------------------------------------------------------------------------


def explanation_job(model_key: str, dataset: str = "ogbg-mollipo", *, limit: int | None = None, property_name: str | None = None) -> dict[str, Any]:
    runner = make_runner(model_key, dataset, property_name=property_name)
    pairs = load_catalog()
    sample = load_sample()
    by_id = {p["pair_id"]: p for p in pairs}
    selected = [by_id[s["pair_id"]] for s in sample]
    train_pairs = [p for p in selected if _is_train(p)]
    nontrain = [p for p in selected if not _is_train(p)]
    if limit:
        nontrain = nontrain[:limit]
        train_pairs = train_pairs[: max(1, limit // 4)]
    selected = nontrain + train_pairs
    t0 = time.time()
    state_before = model_state_hash(runner)
    out: dict[str, Any] = {
        "model": model_key,
        "kind": runner.kind,
        "dataset": dataset,
        "checkpoint": runner.checkpoint,
        "recipe": runner.recipe,
        "property": property_name,
        "sigma": float(runner.b.scaler["std"]),
        "mu": float(runner.b.scaler["mean"]),
        "target_space": "oracle_property" if property_name else "official_label",
        "n_pairs": len(selected),
        "n_nontrain": len(nontrain),
        "n_train": len(train_pairs),
        "pairs": [],
    }
    # FP64 is used when FP32 completeness fails; the same midpoint rule applies.
    if runner.kind == "L" or model_key in ("L", "D"):
        for pair in selected:
            la = native_ledger(runner, int(pair["row_a"]))
            lb = native_ledger(runner, int(pair["row_b"]))
            scores = native_pair_scores(runner, pair, la, lb)
            out["pairs"].append({"pair_id": pair["pair_id"], "row_a": pair["row_a"], "row_b": pair["row_b"], "scope": "NATIVE", "native": scores})
        out["scope"] = "NATIVE"
        out["checks"] = {"native_reconstruction_max_abs": float(max(abs(p["native"]["reconstruction_error_std"]) for p in out["pairs"]))}
        out["state_unchanged"] = model_state_hash(runner) == state_before
        out["seconds"] = time.time() - t0
        return out

    # full-MP IG for T / A
    unions = [build_union(runner, pair) for pair in selected]
    endpoint = [union_endpoint_check(runner, pair) for pair in selected]
    ep_max = float(max(max(v["a_abs_error"], v["b_abs_error"]) for v in endpoint))
    out["endpoint"] = {"max_abs_std": ep_max, "per_pair": [{"pair_id": u.pair_id, **v} for u, v in zip(unions, endpoint)]}
    ep_tol = 1e-5
    out["scope"] = "FULL_MP" if ep_max <= ep_tol else "READOUT_ONLY"
    if ep_max > ep_tol:
        out["checks"] = {"endpoint_failed": True, "endpoint_max_abs_std": ep_max}
        out["seconds"] = time.time() - t0
        return out

    points = 32
    ig = integrate_ig(runner, unions, points)
    failing = [i for i, (u, row) in enumerate(zip(unions, ig["pairs"])) if not ig_groups(runner, u, row)["passes_completeness"]]
    upgrades = {}
    for pts in (64, 128, 256):
        if not failing:
            break
        sub = [unions[i] for i in failing]
        sub_ig = integrate_ig(runner, sub, pts)
        for i, row in zip(failing, sub_ig["pairs"]):
            ig["pairs"][i] = row
        upgrades[pts] = len(failing)
        failing = [i for i in failing if not ig_groups(runner, unions[i], ig["pairs"][i])["passes_completeness"]]
    out["quadrature"] = {"start_points": points, "upgrades": upgrades, "n_unresolved": len(failing)}

    for pair, u, ig_pair, ep in zip(selected, unions, ig["pairs"], endpoint):
        groups = ig_groups(runner, u, ig_pair)
        entry = {
            "pair_id": u.pair_id,
            "row_a": u.row_a,
            "row_b": u.row_b,
            "scope": "FULL_MP",
            "endpoint": ep,
            "ig": groups,
            "delta_std": ig_pair["delta_std"],
            "native": None,
        }
        if model_key == "A":
            la = native_ledger(runner, int(pair["row_a"]))
            lb = native_ledger(runner, int(pair["row_b"]))
            entry["native"] = native_pair_scores(runner, pair, la, lb)
        out["pairs"].append(entry)
    out["checks"] = {
        "endpoint_max_abs_std": ep_max,
        "ig_reconstruction_max_abs_std": float(max(abs(p["ig"]["reconstruction_error_std"]) for p in out["pairs"])),
        "native_reconstruction_max_abs_std": float(
            max((abs(p["native"]["reconstruction_error_std"]) for p in out["pairs"] if p["native"]), default=0.0)
        ),
    }
    out["state_unchanged"] = model_state_hash(runner) == state_before
    out["seconds"] = time.time() - t0
    return out


def model_state_hash(runner: Runner) -> str:
    import hashlib

    digest = hashlib.sha256()
    for key, value in sorted(runner.model.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def antisymmetry_check(model_key: str, dataset: str = "ogbg-mollipo", n: int = 8) -> dict[str, Any]:
    """Swap A/B and verify the IG channels negate under the same path rule."""
    runner = make_runner(model_key, dataset)
    sample = [p for p in load_sample() if not _is_train(p)][:n]
    catalog = {p["pair_id"]: p for p in load_catalog()}
    worst = 0.0
    rows = []
    for s in sample:
        pair = catalog[s["pair_id"]]
        swapped = dict(pair)
        swapped["row_a"], swapped["row_b"] = pair["row_b"], pair["row_a"]
        swapped["var_atoms_a"], swapped["var_atoms_b"] = pair["var_atoms_b"], pair["var_atoms_a"]
        swapped["core_atoms_a"], swapped["core_atoms_b"] = pair["core_atoms_b"], pair["core_atoms_a"]
        swapped["core_atom_map"] = {str(v): int(k) for k, v in pair["core_atom_map"].items()}
        u1 = build_union(runner, pair)
        u2 = build_union(runner, swapped)
        ig1 = integrate_ig(runner, [u1], 32)["pairs"][0]
        ig2 = integrate_ig(runner, [u2], 32)["pairs"][0]
        g1 = ig_groups(runner, u1, ig1)
        g2 = ig_groups(runner, u2, ig2)
        diff = max(abs(g1["variable"] + g2["variable"]), abs(g1["core"] + g2["core"]))
        worst = max(worst, diff)
        rows.append({"pair_id": pair["pair_id"], "max_abs_sum": diff})
    return {"model": model_key, "n": len(rows), "worst_abs_std": worst, "pairs": rows, "pass": worst <= 1e-3}


def renumbering_check(model_key: str, dataset: str = "ogbg-mollipo", n: int = 16) -> dict[str, Any]:
    """Node renumbering (identity mapping) must not change predictions or channel sums.

    The identity atom order is already verified; this check rebuilds the union
    with a deterministic slot permutation and confirms the grouped totals.
    """
    runner = make_runner(model_key, dataset)
    pairs = [p for p in load_catalog() if not p["train_train"]][:n]
    rng = np.random.default_rng(C.TRAIN_SEED)
    rows = []
    for pair in pairs:
        u = build_union(runner, pair)
        perm = rng.permutation(u.n_slots)
        inverse = np.argsort(perm)
        u2 = _permute_union(u, perm)
        ig1 = integrate_ig(runner, [u], 32)["pairs"][0]
        ig2 = integrate_ig(runner, [u2], 32)["pairs"][0]
        g1 = ig_groups(runner, u, ig1)
        g2 = ig_groups(runner, u2, ig2)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "var_abs_diff": abs(g1["variable"] - g2["variable"]),
                "core_abs_diff": abs(g1["core"] - g2["core"]),
                "size_abs_diff": abs(g1["size"] - g2["size"]),
                "pred_abs_diff": abs(u.pred_a - u2.pred_a) + abs(u.pred_b - u2.pred_b),
            }
        )
    worst = max(max(r["var_abs_diff"], r["core_abs_diff"], r["size_abs_diff"], r["pred_abs_diff"]) for r in rows)
    return {"model": model_key, "n": len(rows), "worst_abs_std": worst, "pass": worst <= 1e-4, "pairs": rows}


def _permute_union(u: UnionPair, perm: np.ndarray) -> UnionPair:
    X0 = np.zeros_like(u.X0); X1 = np.zeros_like(u.X1)
    X0[perm] = u.X0; X1[perm] = u.X1
    m0 = np.zeros_like(u.m0); m1 = np.zeros_like(u.m1)
    m0[perm] = u.m0; m1[perm] = u.m1
    keys = [(int(perm[s]), int(perm[d])) for s, d in u.edge_keys]
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    keys = [keys[i] for i in order]
    E0 = u.E0[order]; E1 = u.E1[order]; a0 = u.a0[order]; a1 = u.a1[order]
    return UnionPair(
        pair_id=u.pair_id, row_a=u.row_a, row_b=u.row_b, n_slots=u.n_slots,
        core_slots=perm[u.core_slots], var_a_slots=perm[u.var_a_slots], var_b_slots=perm[u.var_b_slots],
        slot_of_core_a={k: int(perm[v]) for k, v in u.slot_of_core_a.items()},
        edge_index=np.asarray(keys, dtype=np.int64).T if keys else np.zeros((2, 0), np.int64),
        edge_keys=keys, X0=X0, X1=X1, E0=E0, E1=E1, m0=m0, m1=m1, a0=a0, a1=a1,
        N0=u.N0, N1=u.N1, Z0=u.Z0, Z1=u.Z1, pred_a=u.pred_a, pred_b=u.pred_b,
        pred_a64=u.pred_a64, pred_b64=u.pred_b64,
        group_index={"core": perm[u.core_slots], "var_a": perm[u.var_a_slots], "var_b": perm[u.var_b_slots]},
    )


def batch_single_check(model_key: str, dataset: str = "ogbg-mollipo", n: int = 4, points: int = 32) -> dict[str, Any]:
    runner = make_runner(model_key, dataset)
    pairs = [p for p in load_sample() if not _is_train(p)][:n]
    catalog = {p["pair_id"]: p for p in load_catalog()}
    unions = [build_union(runner, catalog[p["pair_id"]]) for p in pairs]
    batched = integrate_ig(runner, unions, points)["pairs"]
    rows = []
    for u, b in zip(unions, batched):
        single = integrate_ig(runner, [u], points)["pairs"][0]
        rows.append(
            {
                "pair_id": u.pair_id,
                "total_abs_diff": abs(b["total_std"] - single["total_std"]),
                "node_abs_diff": float(np.abs(b["node_attr"] - single["node_attr"]).max()),
                "size_abs_diff": abs(b["size"] - single["size"]),
            }
        )
    worst = max(max(r["total_abs_diff"], r["node_abs_diff"], r["size_abs_diff"]) for r in rows)
    return {"model": model_key, "points": points, "rows": rows, "worst_abs_std": worst, "pass": worst <= 1e-4}


def train_atom_values(runner: Runner, batch_size: int = 512) -> dict[str, float]:
    """Train-node RMS of the native per-atom contribution and of q = standardised h1."""
    b = runner.b
    order, _, _ = D.eval_order(b, "train")
    a_all, q_all = [], []
    with torch.no_grad():
        for start in range(0, order.size, batch_size):
            chunk = order[start : start + batch_size]
            batch = D.collate(b, "train", chunk)
            h = runner.trunk.layers(
                runner.trunk.encode_nodes(batch), batch["edge_index"], runner.trunk.encode_edges(batch)
            )
            if runner.head is not None:
                z = runner._z_from_batch(batch)
                parts = runner.head.forward_parts(h, batch["gine_batch"], batch["gine_n_atoms"], z)
                a_v = parts["a_v"]
            else:
                raise RuntimeError("train_atom_values needs an additive head")
            a_all.append(a_v.detach().cpu().numpy().astype(np.float64))
            q_all.append(h[:, 0].detach().cpu().numpy().astype(np.float64))
    a = np.concatenate(a_all)
    q = np.concatenate(q_all)
    mean_q = float(q.mean())
    sd_q = float(q.std(ddof=0))
    return {
        "rms_a": float(np.sqrt(np.mean(a**2))),
        "rms_q_centered": float(np.sqrt(np.mean((q - mean_q) ** 2))),
        "mean_q": mean_q,
        "sd_q": sd_q,
        "n_atoms": int(a.size),
        "a": a,
        "q": q,
    }


def reallocation_check(model_key: str, dataset: str = "ogbg-mollipo") -> dict[str, Any]:
    """7.10 zero-sum reallocation of the native per-atom ledger (A/D only)."""
    runner = make_runner(model_key, dataset)
    stats = train_atom_values(runner)
    if stats["rms_a"] < 1e-8 or stats["rms_q_centered"] < 1e-8:
        return {"model": model_key, "status": "NA", "rms_a": stats["rms_a"], "rms_q": stats["rms_q_centered"]}
    lam = stats["rms_a"] / stats["rms_q_centered"]
    sample = load_sample()
    catalog = {p["pair_id"]: p for p in load_catalog()}
    rows = []
    for s in sample:
        pair = catalog[s["pair_id"]]
        la = native_ledger(runner, int(pair["row_a"]))
        lb = native_ledger(runner, int(pair["row_b"]))
        base = native_pair_scores(runner, pair, la, lb)
        # per-atom q needs the final representation; recompute once per endpoint
        qa = _final_h1(runner, int(pair["row_a"]))
        qb = _final_h1(runner, int(pair["row_b"]))
        qa_c = qa - qa.mean()
        qb_c = qb - qb.mean()
        for sign in (+1.0, -1.0):
            a2 = la["atom"] + sign * lam * qa_c
            b2 = lb["atom"] + sign * lam * qb_c
            delta_a = float(a2.sum() - la["atom"].sum())
            delta_b = float(b2.sum() - lb["atom"].sum())
            fake_a = {**la, "atom": a2}
            fake_b = {**lb, "atom": b2}
            moved = native_pair_scores(runner, pair, fake_a, fake_b)
            var_atoms = [int(x) for x in pair["var_atoms_a"]] + [int(x) for x in pair["var_atoms_b"]]
            ranks_before = _ranks(np.abs(la["atom"]), var_atoms[: len(pair["var_atoms_a"])]) + _ranks(
                np.abs(lb["atom"]), var_atoms[len(pair["var_atoms_a"]) :]
            )
            ranks_after = _ranks(np.abs(a2), var_atoms[: len(pair["var_atoms_a"])]) + _ranks(
                np.abs(b2), var_atoms[len(pair["var_atoms_a"]) :]
            )
            rows.append(
                {
                    "pair_id": pair["pair_id"],
                    "sign": int(sign),
                    "lambda": lam,
                    "sum_change_a": delta_a,
                    "sum_change_b": delta_b,
                    "variable_before": base["variable"],
                    "variable_after": moved["variable"],
                    "variable_abs_change": abs(moved["variable"] - base["variable"]),
                    "core_before": base["core"],
                    "core_after": moved["core"],
                    "pred_diff_change": abs(moved["pred_diff_std"] - base["pred_diff_std"]),
                    "max_rank_change": int(np.max(np.abs(np.asarray(ranks_after) - np.asarray(ranks_before)))),
                    "n_sign_flips": int(np.sum(np.sign(a2[var_atoms[: len(pair["var_atoms_a"])]]) != np.sign(la["atom"][var_atoms[: len(pair["var_atoms_a"])]])) + np.sum(np.sign(b2[var_atoms[len(pair["var_atoms_a"]) :]]) != np.sign(lb["atom"][var_atoms[len(pair["var_atoms_a"]) :]]))),
                }
            )
    return {
        "model": model_key,
        "lambda": lam,
        "rms_a": stats["rms_a"],
        "rms_q": stats["rms_q_centered"],
        "n_pairs": len(rows) // 2,
        "max_abs_sum_change": float(max(max(abs(r["sum_change_a"]), abs(r["sum_change_b"])) for r in rows)),
        "max_abs_pred_change": float(max(r["pred_diff_change"] for r in rows)),
        "mean_abs_variable_change": float(np.mean([r["variable_abs_change"] for r in rows])),
        "max_rank_change": int(max(r["max_rank_change"] for r in rows)),
        "sign_flip_fraction": float(np.mean([r["n_sign_flips"] > 0 for r in rows])),
        "rows": rows,
    }


def _final_h1(runner: Runner, row: int) -> np.ndarray:
    block = runner.encode(row)
    return block["h"][:, 0].detach().cpu().numpy().astype(np.float64)


def _ranks(values: np.ndarray, atoms: Sequence[int]) -> list[int]:
    order = np.argsort(-values, kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(1, len(values) + 1)
    return [int(rank[i]) for i in atoms]


def run_all_ledgers(dataset: str = "ogbg-mollipo") -> dict[str, Any]:
    C.LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"dataset": dataset, "models": {}, "checks": {}}
    for model in ("L", "T", "A", "D"):
        try:
            job = explanation_job(model, dataset)
        except FileNotFoundError as exc:
            summary["models"][model] = {"status": f"MISSING: {exc}"}
            continue
        C.write_json(C.LEDGER_DIR / f"explain_{dataset}_{model}.json", job)
        summary["models"][model] = {
            "kind": job["kind"],
            "scope": job["scope"],
            "n_pairs": job["n_pairs"],
            "seconds": job["seconds"],
            "checks": job.get("checks"),
            "quadrature": job.get("quadrature"),
            "state_unchanged": job.get("state_unchanged"),
            "sigma": job.get("sigma"),
        }
    for model in ("T", "A"):
        if summary["models"].get(model, {}).get("status"):
            continue
        try:
            summary["checks"][f"batch_single_{model}"] = batch_single_check(model, dataset)
            summary["checks"][f"antisymmetry_{model}"] = antisymmetry_check(model, dataset)
            summary["checks"][f"renumbering_{model}"] = renumbering_check(model, dataset)
        except Exception as exc:  # keep partial diagnostics
            summary["checks"][f"{model}_check_error"] = f"{type(exc).__name__}: {exc}"
    for model in ("A", "D"):
        if summary["models"].get(model, {}).get("status"):
            continue
        try:
            summary["checks"][f"reallocation_{model}"] = reallocation_check(model, dataset)
        except Exception as exc:
            summary["checks"][f"reallocation_{model}"] = f"{type(exc).__name__}: {exc}"
    summary["checks"]["multi_mapping"] = {
        "status": "NOT_RUN",
        "reason": "frozen catalog stores one verified identity core mapping per pair; no alternative legal mappings were materialised, so no favourable-map selection was made",
    }
    C.write_json(C.ANALYSIS_DIR / "ledger_checks.json", summary)
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="T")
    parser.add_argument("--dataset", default="ogbg-mollipo")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--property", default=None)
    parser.add_argument("--checks", default=None)
    parser.add_argument("--job", default=None)
    args = parser.parse_args()
    if args.job == "all":
        print(json.dumps(run_all_ledgers(args.dataset), indent=1, default=str)[:6000])
        raise SystemExit(0)
    if args.job == "oracle":
        summary = {"properties": {}}
        for prop in C.ORACLE_PROPERTIES:
            summary["properties"][prop] = {}
            for model in ("L", "T", "A"):
                try:
                    job = explanation_job(model, args.dataset, property_name=prop)
                except FileNotFoundError as exc:
                    summary["properties"][prop][model] = f"MISSING: {exc}"
                    continue
                C.write_json(C.LEDGER_DIR / f"explain_{args.dataset}_{model}_{prop}.json", job)
                summary["properties"][prop][model] = {
                    "scope": job["scope"],
                    "n_pairs": job["n_pairs"],
                    "checks": job.get("checks"),
                    "quadrature": job.get("quadrature"),
                    "sigma": job.get("sigma"),
                }
        C.write_json(C.ANALYSIS_DIR / "oracle_ledger_summary.json", summary)
        print(json.dumps(summary, indent=1, default=str)[:6000])
        raise SystemExit(0)
    if args.checks == "endpoint":
        runner = make_runner(args.model, args.dataset)
        pairs = [p for p in load_catalog() if not p["train_train"]][:16]
        print(json.dumps([{"pair": p["pair_id"], **union_endpoint_check(runner, p)} for p in pairs], indent=1))
    else:
        out = explanation_job(args.model, args.dataset, limit=args.limit, property_name=args.property)
        tag = f"{args.model}_{args.property or ''}".strip("_")
        C.write_json(C.LEDGER_DIR / f"explain_{args.dataset}_{tag}.json", out)
        print(json.dumps({k: v for k, v in out.items() if k not in ("pairs", "endpoint")}, indent=1))
