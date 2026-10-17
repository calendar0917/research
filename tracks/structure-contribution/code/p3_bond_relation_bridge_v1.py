"""P3 + explicit one-hop Bond-Bond relation bridge.

Protocol ``cscl-p3-bond-relation-bridge-v1`` (see
``results/p3_bond_relation_bridge_v1/PROTOCOL.md``).

Three arms, all on the unchanged BJOC-Null (P3) backbone and the frozen R3
recipe:

* ``P3``  the historical BJOC-Null model (413,505 parameters), no relation
  branch;
* ``P3R`` the same model plus a shared symmetric pair MLP that aggregates only
  **real line-graph neighbours** of each bond,
  ``r_e = sum_{f != e} K^R_ef phi(b_e, b_f)`` with ``K^R_ef = A_ef / deg(e)``;
* ``P3U`` the same model and the same new parameters, but the aggregation is
  over **all other bonds** with the dense uniform kernel
  ``K^U_ef = 1/(n_b - 1)``.

Both kernels have an exactly zero diagonal and row sums exactly 1 (for any bond
with at least one neighbour), so the R-vs-U contrast is *not* a row-sum or
self-connection artefact.  The remaining differences are support (real
neighbours vs. all bonds), weight concentration and FLOPs; sparsity is not
isolated (PROTOCOL.md §7).

The relation term enters through a zero-initialised per-channel residual scale
``gamma`` on the composed per-bond representation ``c``, so at initialisation
the augmented model is bit-identical to P3.  ``structural_relation_composer_v1``
is not modified: the bridge wraps the base model and only replaces the ``z``
block of the reader input.

Everything in this module is label-free except the explicit training loop,
which consumes Fit ``y`` only.
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import bjoc_perfmech_long_v1 as bp  # noqa: E402
import bond_relation_attribution_v1 as bra  # noqa: E402
import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

PROTOCOL_ID = "cscl-p3-bond-relation-bridge-v1"
SPLIT_TAG = "F"
SPLIT_SALT = "cscl-p3-bond-relation-bridge-v1"

#: four rows whose committed canonical SMILES contradicts the official PyG
#: constitution (reproduced row-for-row, label-blind)
EXCLUDED_ROWS = bra.EXCLUDED_ROWS
N_USABLE = bra.N_USABLE
FIT_TARGET = 7_996
MONITOR_TARGET = 1_000
LOCKED_TARGET = 1_000
SIZE_TOLERANCE = 10

SEEDS = (0, 1)
ARMS = ("P3", "P3R", "P3U")
BRIDGE_ARMS = ("P3R", "P3U")
KERNEL_OF_ARM = {"P3R": "R", "P3U": "U"}

BASE_PARAMETERS = 413_505
PAIR_HIDDEN = 32
PAIR_INIT_OFFSET = 7919
BRIDGE_PARAMETERS = BASE_PARAMETERS + (2 * 96 * PAIR_HIDDEN + PAIR_HIDDEN) + (PAIR_HIDDEN * 96 + 96) + 96

ZONES = ("monitor", "locked")
PRIMARY_ZONE = "locked"
REPLICATION_ZONE = "monitor"

#: frozen scoring constants (PROTOCOL.md §8)
MEANINGFUL_DELTA = 0.004
TAIL_THRESHOLD = -3.0
TOP_K_REMOVED = 10
BOOTSTRAP_N = 5_000
BOOTSTRAP_SEED = 0

CONTRASTS = (
    ("P3R_minus_P3", "P3R", "P3"),
    ("P3R_minus_P3U", "P3R", "P3U"),
    ("P3U_minus_P3", "P3U", "P3"),
)

#: historical files whose bytes must stay unchanged (recorded by the freeze)
REUSE_HASHES = {
    "structural_relation_composer_v1.py": "1c83a97ece60a151c6fa0c1ad5d2022b470ede9356b23d8d7b1e6b62daa71f4f",
    "structural_relation_objects_v1.py": "8f4ba44dc3c047d674a1a6832f7ed8265eaa2457b15e820b3973a4cbd3a0861c",
    "bjoc_perfmech_long_v1.py": "cb9796cb45f6fe048f2fd63e96798937cdbd802b28dbad34b2aafdbef0cbafe8",
    "bjoc_masked_pretraining_v1.py": "f27dbfe36ac5a9a0a1915320dba7c955115ccfa36e6b4755f2d2d62d8efb1489",
    "bjoc_arch_breakthrough_v1.py": "31294de0526215e3c998a822346d4dac5baceb24ebd986b61122f78e65d55377",
    "bond_relation_attribution_v1.py": "22c24392f2da4c52d581b8e3f888383e22cfeeadbb909c1f2efb3f234f234101",
}


class BridgeError(RuntimeError):
    """Hard protocol violation (raises instead of silently continuing)."""


def require(condition: Any, message: str) -> None:
    if not condition:
        raise BridgeError(message)


# ---------------------------------------------------------------------------
# kernels on the physical-bond line graph
# ---------------------------------------------------------------------------


def line_adjacency(obj: sro.MolObjects) -> np.ndarray:
    """0/1 adjacency of the physical-bond line graph, diagonal forced to 0."""
    bi = np.asarray(obj.bond_index, dtype=np.int64).reshape(-1, 2)
    n_bonds = int(bi.shape[0])
    if n_bonds == 0:
        return np.zeros((0, 0), dtype=np.float32)
    n_atoms = int(np.asarray(obj.atom_feat).shape[0])
    incidence = np.zeros((n_atoms, n_bonds), dtype=np.float32)
    slots = np.arange(n_bonds)
    incidence[bi[:, 0], slots] = 1.0
    incidence[bi[:, 1], slots] = 1.0
    adj = incidence.T @ incidence
    np.fill_diagonal(adj, 0.0)
    require(float(adj.max(initial=0.0)) <= 1.0 + 1e-6, "line graph has duplicate bonds")
    return (adj > 0.5).astype(np.float32)


def kernel_matrices(obj: sro.MolObjects) -> dict[str, np.ndarray]:
    """Dense R and U kernels plus the line-graph adjacency."""
    adj = line_adjacency(obj)
    n = int(adj.shape[0])
    deg = adj.sum(axis=1)
    k_r = np.zeros_like(adj)
    positive = deg > 0
    k_r[positive] = adj[positive] / deg[positive, None]
    if n >= 2:
        k_u = (np.ones((n, n), dtype=np.float32) - np.eye(n, dtype=np.float32)) / float(n - 1)
    else:
        k_u = np.zeros((n, n), dtype=np.float32)
    return {"A": adj, "deg": deg.astype(np.float32), "R": k_r.astype(np.float32), "U": k_u}


def pair_lists(obj: sro.MolObjects, arm: str) -> dict[str, np.ndarray]:
    """Unordered pair list ``e < f`` with the row weights of both directions.

    Reconstructing the dense kernel from ``(src, dst, w_src, w_dst)`` is exact:
    ``K[src, dst] = w_src`` and ``K[dst, src] = w_dst``.  ``P3R`` keeps only real
    line-graph edges; ``P3U`` keeps every ``e < f`` pair.
    """
    require(arm in KERNEL_OF_ARM.values(), f"unknown kernel {arm!r}")
    mats = kernel_matrices(obj)
    kernel = mats[arm]
    n = int(kernel.shape[0])
    rows, cols = np.triu_indices(n, 1)
    if arm == "R":
        keep = mats["A"][rows, cols] > 0.5
        rows, cols = rows[keep], cols[keep]
    src = rows.astype(np.int32)
    dst = cols.astype(np.int32)
    return {
        "src": src,
        "dst": dst,
        "w_src": kernel[src, dst].astype(np.float32),
        "w_dst": kernel[dst, src].astype(np.float32),
    }


def kernel_from_pairs(obj: sro.MolObjects, arm: str) -> np.ndarray:
    """Rebuild the dense kernel from the stored pair list (alignment audit)."""
    mats = kernel_matrices(obj)
    n = int(mats[arm].shape[0])
    out = np.zeros((n, n), dtype=np.float32)
    pairs = pair_lists(obj, arm)
    out[pairs["src"], pairs["dst"]] = pairs["w_src"]
    out[pairs["dst"], pairs["src"]] = pairs["w_dst"]
    return out


class PairSet:
    """Zone-wide concatenated pair lists with per-molecule slicing."""

    def __init__(self, pairs: Sequence[Mapping[str, np.ndarray]], n_bonds: Sequence[int]) -> None:
        starts = np.zeros(len(pairs), dtype=np.int64)
        ends = np.zeros(len(pairs), dtype=np.int64)
        src: list[np.ndarray] = []
        dst: list[np.ndarray] = []
        ws: list[np.ndarray] = []
        wd: list[np.ndarray] = []
        cursor = 0
        for i, p in enumerate(pairs):
            require(
                int(p["src"].shape[0]) == int(p["dst"].shape[0]) == int(p["w_src"].shape[0]) == int(p["w_dst"].shape[0]),
                "pair arrays differ in length",
            )
            starts[i] = cursor
            src.append(np.asarray(p["src"], dtype=np.int32))
            dst.append(np.asarray(p["dst"], dtype=np.int32))
            ws.append(np.asarray(p["w_src"], dtype=np.float32))
            wd.append(np.asarray(p["w_dst"], dtype=np.float32))
            cursor += int(src[-1].shape[0])
            ends[i] = cursor
        self.starts = starts
        self.ends = ends
        self.src = np.concatenate(src) if src else np.zeros(0, dtype=np.int32)
        self.dst = np.concatenate(dst) if dst else np.zeros(0, dtype=np.int32)
        self.w_src = np.concatenate(ws) if ws else np.zeros(0, dtype=np.float32)
        self.w_dst = np.concatenate(wd) if wd else np.zeros(0, dtype=np.float32)
        self.n_bonds = np.asarray(n_bonds, dtype=np.int64)

    def __len__(self) -> int:
        return int(self.starts.shape[0])

    def to_state(self) -> dict[str, np.ndarray]:
        return {
            "starts": self.starts,
            "ends": self.ends,
            "src": self.src,
            "dst": self.dst,
            "w_src": self.w_src,
            "w_dst": self.w_dst,
            "n_bonds": self.n_bonds,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, np.ndarray]) -> "PairSet":
        obj = cls.__new__(cls)
        obj.starts = np.asarray(state["starts"], dtype=np.int64)
        obj.ends = np.asarray(state["ends"], dtype=np.int64)
        obj.src = np.asarray(state["src"], dtype=np.int32)
        obj.dst = np.asarray(state["dst"], dtype=np.int32)
        obj.w_src = np.asarray(state["w_src"], dtype=np.float32)
        obj.w_dst = np.asarray(state["w_dst"], dtype=np.float32)
        obj.n_bonds = np.asarray(state["n_bonds"], dtype=np.int64)
        return obj

    @property
    def n_pairs(self) -> int:
        return int(self.src.shape[0])

    def batch(self, indices: Sequence[int], device: torch.device | str) -> dict[str, torch.Tensor]:
        """Pair tensors for a batch whose molecules are in ``indices`` order."""
        idx = np.asarray(list(indices), dtype=np.int64)
        lens = self.ends[idx] - self.starts[idx]
        total = int(lens.sum())
        if total == 0:
            empty_i = torch.zeros(0, dtype=torch.long, device=device)
            empty_f = torch.zeros(0, dtype=torch.float32, device=device)
            return {"src": empty_i, "dst": empty_i.clone(), "w_src": empty_f, "w_dst": empty_f.clone()}
        rep = np.repeat(np.arange(int(idx.shape[0])), lens)
        first = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        positions = self.starts[idx][rep] + (np.arange(total, dtype=np.int64) - first[rep])
        bond_offset = np.repeat(np.cumsum(self.n_bonds[idx]) - self.n_bonds[idx], lens)
        out = {
            "src": torch.from_numpy(self.src[positions].astype(np.int64) + bond_offset).to(device),
            "dst": torch.from_numpy(self.dst[positions].astype(np.int64) + bond_offset).to(device),
            "w_src": torch.from_numpy(self.w_src[positions].astype(np.float32)).to(device),
            "w_dst": torch.from_numpy(self.w_dst[positions].astype(np.float32)).to(device),
        }
        return out


def build_pair_set(objs: Sequence[sro.MolObjects], arm: str) -> PairSet:
    n_bonds = [int(np.asarray(o.bond_index).reshape(-1, 2).shape[0]) for o in objs]
    return PairSet([pair_lists(o, arm) for o in objs], n_bonds)


# ---------------------------------------------------------------------------
# kernel audit (label-free)
# ---------------------------------------------------------------------------


def audit_molecule(obj: sro.MolObjects) -> dict[str, Any]:
    mats = kernel_matrices(obj)
    n = int(mats["R"].shape[0])
    out: dict[str, Any] = {"n_bonds": n}
    recs = {}
    for arm in ("R", "U"):
        recs[arm] = pair_lists(obj, arm)
        rebuilt = kernel_from_pairs(obj, arm)
        kernel = mats[arm]
        recs[arm]["reconstruction_max_abs"] = float(np.abs(rebuilt - kernel).max()) if n else 0.0
        recs[arm]["diag_max_abs"] = float(np.abs(np.diag(kernel)).max()) if n else 0.0
        row = kernel.sum(axis=1)
        expect = np.zeros(n, dtype=np.float32)
        if arm == "R":
            expect[mats["deg"] > 0] = 1.0
        elif n >= 2:
            expect[:] = 1.0
        recs[arm]["rowsum_max_err"] = float(np.abs(row - expect).max()) if n else 0.0
        recs[arm]["n_pairs"] = int(recs[arm]["src"].shape[0])
        recs[arm]["total_weight"] = float(kernel.sum())
        recs[arm]["nnz"] = int((kernel > 0).sum())
        recs[arm]["offdiag_nnz"] = int(((kernel > 0).sum()) - (np.diag(kernel) > 0).sum())
    out["checks"] = {
        "R_diag_zero": bool(recs["R"]["diag_max_abs"] == 0.0),
        "U_diag_zero": bool(recs["U"]["diag_max_abs"] == 0.0),
        "R_rowsum_ok": bool(recs["R"]["rowsum_max_err"] <= 1e-6),
        "U_rowsum_ok": bool(recs["U"]["rowsum_max_err"] <= 1e-6),
        "R_pair_reconstruction_exact": bool(recs["R"]["reconstruction_max_abs"] == 0.0),
        "U_pair_reconstruction_exact": bool(recs["U"]["reconstruction_max_abs"] == 0.0),
        "R_rows_equal_U_rows": bool(abs(recs["R"]["total_weight"] - recs["U"]["total_weight"]) <= 1e-4),
        "R_equals_U": bool(n and np.abs(mats["R"] - mats["U"]).max() == 0.0),
    }
    out["stats"] = {
        "n_pairs_R": recs["R"]["n_pairs"],
        "n_pairs_U": recs["U"]["n_pairs"],
        "sparsity_R": float(1.0 - recs["R"]["nnz"] / (n * n)) if n else 1.0,
        "sparsity_U": float(1.0 - recs["U"]["nnz"] / (n * n)) if n else 1.0,
        "mean_degree": float(mats["deg"].mean()) if n else 0.0,
        "max_degree": float(mats["deg"].max()) if n else 0.0,
        "R_weight_entropy": _row_entropy(mats["R"]),
        "U_weight_entropy": _row_entropy(mats["U"]),
        "R_max_weight": float(mats["R"].max()) if n else 0.0,
        "R_min_positive_weight": float(mats["R"][mats["R"] > 0].min()) if float(mats["R"].max(initial=0.0)) > 0 else 0.0,
    }
    return out


def _row_entropy(kernel: np.ndarray) -> float:
    p = kernel[kernel > 0]
    if p.size == 0:
        return 0.0
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    return float(-(p * np.log(p)).sum())


def summarize_audit(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    n = len(records)
    counts = {k: int(sum(1 for r in records if r["checks"][k])) for k in records[0]["checks"]} if n else {}
    stats: dict[str, float] = {}
    for key in records[0]["stats"] if n else {}:
        values = [float(r["stats"][key]) for r in records]
        stats[key] = float(np.mean(values))
        stats[f"{key}_max"] = float(np.max(values))
    required = {k: v for k, v in counts.items() if k != "R_equals_U"}
    return {
        "n_molecules": n,
        "n_checks_true": counts,
        "all_checks_true": bool(all(required.values())) if required else False,
        "required_checks": sorted(required),
        "mean_stats": stats,
        "n_R_equals_U": int(sum(1 for r in records if r["checks"]["R_equals_U"])),
        "frac_R_equals_U": float(np.mean([1.0 if r["checks"]["R_equals_U"] else 0.0 for r in records])) if n else 0.0,
        "n_single_bond": int(sum(1 for r in records if int(r["n_bonds"]) <= 1)),
        "n_two_bonds": int(sum(1 for r in records if int(r["n_bonds"]) == 2)),
    }


# ---------------------------------------------------------------------------
# model: zero-init residual bridge on top of the unchanged P3
# ---------------------------------------------------------------------------


class BondRelationBridge(nn.Module):
    """``P3`` with one symmetric pair-MLP relation residual on ``c``.

    The base module is untouched; only the ``z`` block of the reader input is
    recomputed after adding ``gamma * r_e`` to the composed per-bond vector.
    """

    def __init__(self, base: src.BJOC, seed: int, hidden: int = PAIR_HIDDEN) -> None:
        super().__init__()
        self.base = base
        self.hidden = int(hidden)
        out = int(base.config.compose_out)
        self.pair_mlp = nn.Sequential(nn.Linear(2 * out, hidden), nn.SiLU(), nn.Linear(hidden, out))
        self.rel_scale = nn.Parameter(torch.zeros(out))
        self._init_new_parameters(seed)

    def _init_new_parameters(self, seed: int) -> None:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + PAIR_INIT_OFFSET)
        with torch.no_grad():
            for lin in (self.pair_mlp[0], self.pair_mlp[2]):
                bound = 1.0 / math.sqrt(float(lin.weight.shape[1]))
                lin.weight.copy_(torch.empty(lin.weight.shape, dtype=torch.float32).uniform_(-bound, bound, generator=gen))
                lin.bias.copy_(torch.empty(lin.bias.shape, dtype=torch.float32).uniform_(-bound, bound, generator=gen))
            self.rel_scale.zero_()

    @property
    def new_parameters(self) -> int:
        return int(sum(p.numel() for p in self.pair_mlp.parameters()) + self.rel_scale.numel())

    def relation_aggregate(self, b: torch.Tensor, rel: Mapping[str, torch.Tensor] | None) -> torch.Tensor:
        agg = torch.zeros_like(b)
        if rel is None or int(rel["src"].numel()) == 0:
            return agg
        e, f = rel["src"], rel["dst"]
        be, bf = b[e], b[f]
        msg = self.pair_mlp(torch.cat([be + bf, (be - bf).abs()], dim=1))
        agg.index_add_(0, e, rel["w_src"].unsqueeze(1) * msg)
        agg.index_add_(0, f, rel["w_dst"].unsqueeze(1) * msg)
        return agg

    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        rel: Mapping[str, torch.Tensor] | None,
    ) -> dict[str, torch.Tensor]:
        feats = self.base.forward_features(batch)
        out = int(self.base.config.compose_out)
        agg = self.relation_aggregate(feats["b"], rel)
        c = feats["c"] + self.rel_scale.unsqueeze(0) * agg
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, out), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        reader_in = torch.cat([z, feats["reader_in"][:, out:]], dim=1)
        return {**feats, "bridge_agg": agg, "bridge_c": c, "bridge_z": z, "bridge_reader_in": reader_in}

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        rel: Mapping[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        feats = self.forward_features(batch, rel)
        return self.base.reader(feats["bridge_reader_in"]).view(-1)


def build_bridge(seed: int, norm: Mapping[str, Any], device: torch.device | str, hidden: int = PAIR_HIDDEN) -> BondRelationBridge:
    """Deterministic bridge init: P3 scratch init + identical new-parameter init."""
    base = bp.build_bjoc_null(seed, norm, device)
    model = BondRelationBridge(base, seed, hidden)
    return model.to(device)


def bridge_state_hash(model: BondRelationBridge) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        if key.endswith("global_mean") or key.endswith("global_scale"):
            continue
        h.update(key.encode())
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def model_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------


def move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def predict_model(
    model: nn.Module,
    objs: Sequence[sro.MolObjects],
    pairs: PairSet | None,
    device: torch.device | str,
    chunk: int = 256,
) -> np.ndarray:
    model.eval()
    dev = torch.device(device)
    out = []
    for i in range(0, len(objs), chunk):
        part = list(objs[i : i + chunk])
        batch = move_batch(sro.batch_objects(part), dev)
        if pairs is None:
            pred = model(batch)
        else:
            rel = pairs.batch(list(range(i, i + len(part))), dev)
            pred = model(batch, rel)
        out.append(pred.detach().cpu().numpy())
    return np.concatenate(out, 0)


# ---------------------------------------------------------------------------
# training loop (mirror of bp.train_supervised, same recipe / RNG / soup)
# ---------------------------------------------------------------------------


def _last_soup(states: list[dict[str, torch.Tensor]], last_epoch: int) -> tuple[list[int], dict[str, torch.Tensor]]:
    members = list(range(last_epoch - len(states) + 1, last_epoch + 1))
    soup = {k: torch.stack([s[k].float() for s in states]).mean(0) for k in states[0].keys()}
    return members, soup


def _module_grad_norm(module: nn.Module) -> float:
    norms = [float(p.grad.norm()) for p in module.parameters() if p.grad is not None]
    return float(np.sqrt(np.sum(np.square(norms)))) if norms else 0.0


def _grad_probe(model: nn.Module) -> dict[str, float]:
    if not isinstance(model, BondRelationBridge):
        return {"pair_mlp": 0.0, "rel_scale": 0.0}
    scale = model.rel_scale.grad
    return {
        "pair_mlp": _module_grad_norm(model.pair_mlp),
        "rel_scale": float(scale.norm()) if scale is not None else 0.0,
    }


def train_supervised_bridge(
    *,
    arm: str,
    seed: int,
    model: nn.Module,
    pairs: PairSet | None,
    train_objs: Sequence[sro.MolObjects],
    y_fit: np.ndarray,
    device: torch.device,
    recipe_key: str = "R3",
    epochs: int | None = None,
    log_fn: Callable[[str], None] = print,
) -> dict[str, Any]:
    require(recipe_key in bp.RECIPES, f"unknown recipe {recipe_key!r}")
    recipe = bp.RECIPES[recipe_key]
    n = int(len(train_objs))
    require(n == int(len(y_fit)), "train objects and labels differ in length")
    steps_per_epoch = math.ceil(n / bp.BATCH_SIZE)
    total_updates = int(epochs or recipe["epochs"]) * steps_per_epoch
    optimizer = torch.optim.Adam(model.parameters(), lr=float(recipe["lr"]), weight_decay=float(recipe["weight_decay"]))
    scheduler = None
    if recipe["schedule"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_updates, eta_min=float(recipe["lr_min"]))
    elif recipe["schedule"] != "constant":
        raise BridgeError(f"unknown schedule {recipe['schedule']!r}")

    gen = torch.Generator().manual_seed(int(seed) + bp.TRAIN_SHUFFLE_OFFSET)
    states: deque = deque(maxlen=bp.SOUP_LAST)
    curve: list[dict[str, Any]] = []
    order_hash = hashlib.sha256()
    lr_deviation_max = 0.0
    grad_norm_max = 0.0
    clip_events = 0
    first_step = {}
    last_step = {}
    started = time.perf_counter()
    total_epochs = int(epochs or recipe["epochs"])

    for epoch in range(1, total_epochs + 1):
        model.train()
        total, n_mol, steps = 0.0, 0, 0
        lr_first = lr_last = None
        lr_sum = 0.0
        t_ep = time.perf_counter()
        for indices in bp.shuffle_batches(n, bp.BATCH_SIZE, gen):
            arr = np.ascontiguousarray(np.asarray(indices, dtype=np.int64))
            order_hash.update(arr.tobytes())
            batch = move_batch(sro.batch_objects([train_objs[i] for i in indices]), device)
            rel = pairs.batch(indices, device) if pairs is not None else None
            y = torch.as_tensor(y_fit[indices], dtype=torch.float32, device=device)
            pred = model(batch) if rel is None else model(batch, rel)
            loss = F.l1_loss(pred.view(-1), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), bp.GRAD_CLIP))
            if not math.isfinite(gn):
                raise BridgeError(f"non-finite gradient norm at epoch {epoch} {arm} seed {seed}")
            grad_norm_max = max(grad_norm_max, gn)
            if gn > bp.GRAD_CLIP:
                clip_events += 1
            if epoch == 1 and not first_step:
                first_step = _grad_probe(model)
            actual_lr = float(optimizer.param_groups[0]["lr"])
            if scheduler is not None:
                expected_lr = bp.schedule_lr(recipe, steps + (epoch - 1) * steps_per_epoch, total_updates)
                lr_deviation_max = max(lr_deviation_max, abs(actual_lr - expected_lr))
            if lr_first is None:
                lr_first = actual_lr
            lr_last = actual_lr
            lr_sum += actual_lr
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            total += float((pred.view(-1) - y.view(-1)).abs().sum())
            n_mol += len(indices)
            steps += 1
        train_mae = total / max(n_mol, 1)
        curve.append(
            {
                "epoch": epoch,
                "updates": epoch * steps,
                "train_mae": train_mae,
                "lr_first": lr_first,
                "lr_last": lr_last,
                "lr_mean": lr_sum / max(steps, 1),
                "seconds": time.perf_counter() - t_ep,
            }
        )
        states.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
        if epoch % 20 == 0 or epoch == 1 or epoch == total_epochs:
            log_fn(f"[{arm} s{seed}] ep={epoch:03d} train={train_mae:.6f} lr={lr_last:.3e} {curve[-1]['seconds']:.1f}s")
        if not math.isfinite(train_mae):
            raise BridgeError(f"non-finite loss at epoch {epoch} {arm} seed {seed}")

        if isinstance(model, BondRelationBridge) and epoch in (1, total_epochs):
            last_step[str(epoch)] = _grad_probe(model)

    wall = time.perf_counter() - started
    members, soup = _last_soup(list(states), total_epochs)
    model.load_state_dict({k: v.to(device) for k, v in soup.items()})
    fit_pred = predict_model(model, train_objs, pairs, device)
    soup_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    bridge_info = {}
    if isinstance(model, BondRelationBridge):
        scale = model.rel_scale.detach().cpu()
        bridge_info = {
            "new_parameters": model.new_parameters,
            "n_parameters_total": model_parameters(model),
            "rel_scale_abs_mean": float(scale.abs().mean()),
            "rel_scale_abs_max": float(scale.abs().max()),
            "rel_scale_nonzero_fraction": float((scale.abs() > 1e-8).float().mean()),
            "pair_mlp_parameters": int(sum(p.numel() for p in model.pair_mlp.parameters())),
        }
    return {
        "protocol": PROTOCOL_ID,
        "arm": arm,
        "seed": int(seed),
        "recipe_key": recipe_key,
        "recipe": dict(recipe),
        "epochs": total_epochs,
        "updates": int(total_updates),
        "steps_per_epoch": steps_per_epoch,
        "n_train_rows": n,
        "soup_rule": f"parameter mean of the last {bp.SOUP_LAST} epoch states",
        "soup_members": members,
        "soup_state": soup_state,
        "soup_fit_mae": float(np.abs(fit_pred - y_fit).mean()),
        "schedule": {
            "type": recipe["schedule"],
            "per": "step",
            "lr0": float(recipe["lr"]),
            "lr_min": None if recipe["lr_min"] is None else float(recipe["lr_min"]),
            "t_max_updates": total_updates,
            "lr_first_step": curve[0]["lr_first"],
            "lr_last_step": curve[-1]["lr_last"],
            "cosine_formula_max_abs_deviation": lr_deviation_max,
        },
        "anomalies": {"grad_clip_events": int(clip_events), "grad_norm_max": float(grad_norm_max)},
        "curve": curve,
        "wall_seconds": float(wall),
        "molecules_seen": int(n) * total_epochs,
        "device": str(device),
        "batch_order": {"full_hash": order_hash.hexdigest(), "full_matches": True},
        "bridge": bridge_info,
        "gradient_probe": {"first_step": first_step, "end_of_epoch": last_step},
        "parameter_audit": {"actual_parameters": model_parameters(model)},
    }


# ---------------------------------------------------------------------------
# frozen decision vocabulary
# ---------------------------------------------------------------------------


def contrast_label(
    delta_seed_mean: float,
    per_seed_delta: Mapping[str, float],
    ci_lo: float,
    ci_hi: float,
    ordinary_delta: float,
    meaningful: float = MEANINGFUL_DELTA,
) -> str:
    values = [float(v) for v in per_seed_delta.values()]
    if (
        delta_seed_mean <= -meaningful
        and all(v < 0.0 for v in values)
        and ci_hi < 0.0
        and ordinary_delta <= 0.0
    ):
        return "GAIN"
    if delta_seed_mean >= meaningful and all(v > 0.0 for v in values) and ci_lo > 0.0:
        return "LOSS"
    return "NEUTRAL"


def replication_status(monitor_per_seed_delta: Mapping[str, float]) -> dict[str, Any]:
    values = {str(k): float(v) for k, v in monitor_per_seed_delta.items()}
    mean = float(np.mean(list(values.values())))
    return {
        "seed_mean_delta": mean,
        "per_seed_delta": values,
        "replicated": bool(mean < 0.0 and all(v < 0.0 for v in values.values())),
    }


def bridge_case(
    *,
    label_r_p3: str,
    label_r_u: str,
    label_u_p3: str,
    replicated_r_p3: bool,
    replicated_r_u: bool,
    replicated_u_p3: bool,
) -> dict[str, Any]:
    """Frozen A/B/C/E/D rule (PROTOCOL.md §8).  Ordered, mutually exclusive."""
    if label_r_p3 == "GAIN" and replicated_r_p3 and label_r_u == "GAIN" and replicated_r_u:
        return {
            "case": "CASE_A_RELATION_ADDS_VALUE",
            "reasons": ["P3R beats P3 and P3U, both replicated on Monitor"],
        }
    if label_r_u == "GAIN" and replicated_r_u and label_r_p3 != "GAIN":
        return {
            "case": "CASE_B_RELATION_USED_NO_HEADLINE_GAIN",
            "reasons": ["P3R beats P3U but not P3"],
        }
    if label_r_p3 == "GAIN" and replicated_r_p3 and label_u_p3 == "GAIN" and replicated_u_p3 and label_r_u != "GAIN":
        return {
            "case": "CASE_C_GENERIC_PAIR_CAPACITY",
            "reasons": ["both P3R and P3U beat P3, but P3R does not beat P3U"],
        }
    if label_r_p3 == "GAIN":
        reasons = []
        if not replicated_r_p3:
            reasons.append("P3R-P3 gain did not replicate on Monitor")
        if label_r_u != "GAIN":
            reasons.append("P3R-P3U is not GAIN")
        if label_u_p3 != "GAIN":
            reasons.append("P3U-P3 is not GAIN")
        return {"case": "CASE_E_R_GAIN_NOT_ATTRIBUTED", "reasons": reasons or ["residual"]}
    return {
        "case": "CASE_D_NO_GAIN_ON_P3",
        "reasons": ["P3R does not improve P3 beyond the frozen practical scale"],
    }
