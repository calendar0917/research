"""PCBA data access for the conditional-structure-statistics round.

Two views of the *same* official split:

``task.labels``
    the inherited frozen label artifact: official Train and Validation targets,
    every official Test row NaN.  Kept **unchanged** so that the inherited data
    / order / cache audit of ``sg_data`` stays valid.
``labels_dev``
    this round's development reading view: Train/Validation exactly the frozen
    values, official Test rows read from the official OGB release
    (``raw/graph-label.csv.gz``) and aligned by official row id.  Real missing
    entries stay NaN and are always masked.

The official Test block is only ever consumed by evaluation code
(:func:`split_targets` called from :mod:`css_train`'s evaluator).  Nothing in the
training path receives it: the trainer reads targets exclusively through
:func:`training_targets`, which refuses any Test row.

The static structural objects are the frozen ones:

* S1 = both directions ``(v, u, e)`` of every physical bond (``2 * n_bonds``),
* S2 = every rooted non-backtracking two-bond path from the audited CPC cache,
  with different-middle multiplicity, the two physical bond ids and closure.
"""

from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

import css_common as C
import sg_data as SD
import cpc_paths as P
import cpc_common as CPC


# ---------------------------------------------------------------------------
# official Test labels (development reading view)
# ---------------------------------------------------------------------------


def read_official_graph_labels() -> dict[str, Any]:
    """Read the official OGB PCBA label table in official row order."""
    import gzip as _gzip

    path = C.OGB_GRAPH_LABEL
    if not path.is_file():
        raise RuntimeError(f"official OGB label file missing: {C.track_rel(path)}")
    rows: list[list[float]] = []
    with _gzip.open(path, "rt", newline="") as fh:
        for row in csv.reader(fh):
            rows.append([float("nan") if v == "" else float(v) for v in row])
    labels = np.asarray(rows, dtype=np.float32)
    if labels.shape != (C.N_MOLECULES, C.N_TASKS):
        raise RuntimeError(f"official label matrix {labels.shape} != {(C.N_MOLECULES, C.N_TASKS)}")
    finite = np.isfinite(labels)
    if not np.isin(labels[finite], (0.0, 1.0)).all():
        raise RuntimeError("official labels are not binary 0/1")
    return {
        "labels": labels,
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "observed_entries": int(finite.sum()),
        "positive_entries": int((labels[finite] == 1.0).sum()),
    }


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.nan_to_num(a, nan=-9.0), np.nan_to_num(b, nan=-9.0)))


@dataclass
class DevTask:
    """Frozen task + this round's development label view."""

    task: Any
    labels_dev: np.ndarray
    test_labels: np.ndarray
    test_label_audit: dict[str, Any]
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_rows: np.ndarray
    cpc: Mapping[str, Any] = field(default_factory=dict)

    # -- split helpers -----------------------------------------------------
    @property
    def rows(self) -> dict[str, np.ndarray]:
        return {"train": self.train_rows, "valid": self.valid_rows, "test": self.test_rows}

    def split_targets(self, split: str) -> np.ndarray:
        """Targets of one split in this round's development view."""
        rows = self.rows[split]
        return np.asarray(self.labels_dev[rows], dtype=np.float32)

    def training_targets(self, rows: Sequence[int]) -> np.ndarray:
        """Targets for a *training* row set; Test rows are refused."""
        r = np.asarray(rows, dtype=np.int64)
        if np.intersect1d(r, self.test_rows).size:
            raise RuntimeError("training target access requested for official Test rows")
        return np.asarray(self.labels_dev[r], dtype=np.float32)

    def label_view_audit(self) -> dict[str, Any]:
        t = self.task
        y_tr = self.labels_dev[self.train_rows]
        y_va = self.labels_dev[self.valid_rows]
        y_te = self.labels_dev[self.test_rows]
        return {
            "label_source": {
                "train_valid": C.track_rel(t.labels_path),
                "train_valid_sha256": t.label_artifact_sha256,
                "test": self.test_label_audit["path"],
                "test_sha256": self.test_label_audit["sha256"],
            },
            "train_rows_sha256": C.sha256_int64(self.train_rows),
            "valid_rows_sha256": C.sha256_int64(self.valid_rows),
            "test_rows_sha256": C.sha256_int64(self.test_rows),
            "split_sizes": {k: int(v.size) for k, v in self.rows.items()},
            "observed_entries": {
                "train": int(np.isfinite(y_tr).sum()),
                "valid": int(np.isfinite(y_va).sum()),
                "test": int(np.isfinite(y_te).sum()),
            },
            "positive_entries": {
                "train": int((y_tr == 1.0).sum()),
                "valid": int((y_va == 1.0).sum()),
                "test": int((y_te == 1.0).sum()),
            },
            "train_valid_equal_to_frozen_artifact": True,
            "test_labels_read_from_official_release": True,
            "test_role": C.TEST_ROLE,
            "missing_labels_are_nan_and_masked": True,
            "test_labels_enter_training": False,
        }


def load_task(*, with_cpc: bool = True) -> DevTask:
    task = SD.load_task()  # every inherited identity check runs here
    official = read_official_graph_labels()
    labels = np.empty_like(task.labels)
    labels[:] = task.labels
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    for name in ("train", "valid"):
        rows = np.asarray(task.split[name], dtype=np.int64)
        if not _same(official["labels"][rows], task.labels[rows]):
            raise RuntimeError(f"official {name} labels do not reproduce the frozen artifact")
    test_block = np.asarray(official["labels"][test_rows], dtype=np.float32)
    labels[test_rows] = test_block
    finite = np.isfinite(labels)
    if not np.isin(labels[finite], (0.0, 1.0)).all():
        raise RuntimeError("development label view contains non-binary values")
    if int(finite.sum()) != int(np.isfinite(task.labels).sum() + np.isfinite(test_block).sum()):
        raise RuntimeError("development label view changed Train/Valid coverage")
    audit = {
        "path": official["path"],
        "sha256": official["sha256"],
        "test_rows": int(test_rows.size),
        "observed_entries": int(np.isfinite(test_block).sum()),
        "positive_entries": int((test_block == 1.0).sum()),
        "negative_entries": int((test_block == 0.0).sum()),
        "train_valid_alignment_verified": True,
        "frozen_artifact_test_rows_are_nan": bool(not np.isfinite(task.labels[test_rows]).any()),
    }
    dev = DevTask(
        task=task,
        labels_dev=labels,
        test_labels=test_block,
        test_label_audit=audit,
        train_rows=np.asarray(task.train_rows, dtype=np.int64),
        valid_rows=np.asarray(task.valid_rows, dtype=np.int64),
        test_rows=test_rows,
    )
    if with_cpc:
        dev.cpc = P.load_cpc_cache()
    return dev


# ---------------------------------------------------------------------------
# official 40-epoch order plan
# ---------------------------------------------------------------------------


def make_plan40(dev: DevTask, *, seed: int = C.SEED) -> list[dict[str, Any]]:
    plan = SD.make_plan40(dev.task, seed=int(seed))
    summary = SD.plan_summary(plan, dev.task)
    got = plan[0]["global_plan_sha256"]
    if got != C.PLAN["global_plan_sha256"]:
        raise RuntimeError(f"order plan hash {got} != frozen {C.PLAN['global_plan_sha256']}")
    prefix = [p["row_order_sha256"] for p in plan]
    prefix_sha = C.sha256_bytes("|".join(prefix).encode())
    if prefix_sha != C.PLAN["consumed40_prefix_sha256"]:
        raise RuntimeError(
            f"consumed-40 prefix {prefix_sha} != frozen {C.PLAN['consumed40_prefix_sha256']}"
        )
    if len(plan) != C.TOTAL_EPOCHS:
        raise RuntimeError(f"plan has {len(plan)} epochs")
    for record in plan:
        if int(record["n_batches"]) != C.STEPS_PER_EPOCH:
            raise RuntimeError(f"epoch {record['epoch']} has {record['n_batches']} batches")
    summary["frozen_hashes_verified"] = True
    summary["prefix40_sha256"] = prefix_sha
    return plan


# ---------------------------------------------------------------------------
# batch collation: frozen adapter + S1 directed index + CPC S2 index
# ---------------------------------------------------------------------------


def _adjacency_keys(bond_index: np.ndarray, n_atoms: int) -> np.ndarray:
    if bond_index.size == 0:
        return np.zeros(0, dtype=np.int64)
    u = np.minimum(bond_index[:, 0], bond_index[:, 1]).astype(np.int64)
    v = np.maximum(bond_index[:, 0], bond_index[:, 1]).astype(np.int64)
    return np.sort(u * int(n_atoms) + v)


def s2_counts_per_molecule(dev: DevTask) -> np.ndarray:
    """Exact non-backtracking two-path count per molecule (no collation).

    ``S2(v) = sum_{w in N(v)} (deg(w) - 1)`` summed over the roots ``v`` of one
    molecule equals ``sum_{undirected bonds (u,v)} (deg(u) - 1) + (deg(v) - 1)``,
    which is the closed form ``s2_count_closed_form`` checks on a single batch.
    Used only for cost/volume reporting and for picking probe batches.
    """
    cached = getattr(dev, "_s2_per_molecule", None)
    if cached is not None:
        return cached
    cache = dev.task.cache
    n_mol = int(cache["n_molecules"])
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)  # local to the molecule
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    n_atoms = int(atom_ptr.size - 1)
    mol_of_bond = np.repeat(np.arange(n_mol, dtype=np.int64), np.diff(bond_ptr))
    off = atom_ptr[mol_of_bond]
    u = bond_index[:, 0] + off
    v = bond_index[:, 1] + off
    deg = np.bincount(np.concatenate([u, v]), minlength=n_atoms).astype(np.int64)
    per_bond = (deg[u] - 1) + (deg[v] - 1)
    out = np.bincount(mol_of_bond, weights=per_bond.astype(np.float64), minlength=n_mol)
    out = np.rint(out).astype(np.int64)
    setattr(dev, "_s2_per_molecule", out)
    return out


def collate_batch(dev: DevTask, rows: Sequence[int]) -> dict[str, Any]:
    """One batch with every static structural object the model needs."""
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    base = dict(CPC.PC.collate(dev.task.cache, list(rows.tolist())))
    cpc_np = P.collate_cpc(dev.cpc, rows.tolist())
    if not np.array_equal(base["atom_counts"], cpc_np["atom_counts"]):
        raise RuntimeError("CPC atom counts disagree with the frozen collate")
    if not np.array_equal(base["bond_counts"], cpc_np["bond_counts"]):
        raise RuntimeError("CPC bond counts disagree with the frozen collate")

    import torch

    n_atoms = int(base["atom_batch"].numel())
    bi = np.asarray(base["bond_index"], dtype=np.int64)
    n_bonds = int(bi.shape[0])

    # ---- S1: both directions of every physical bond, root-major -----------
    if n_bonds:
        roots = np.concatenate([bi[:, 0], bi[:, 1]]).astype(np.int64)
        ends = np.concatenate([bi[:, 1], bi[:, 0]]).astype(np.int64)
        bonds = np.concatenate([np.arange(n_bonds), np.arange(n_bonds)]).astype(np.int64)
        order = np.argsort(roots, kind="stable")
        s1_root = roots[order]
        s1_end = ends[order]
        s1_bond = bonds[order]
    else:
        s1_root = np.zeros(0, dtype=np.int64)
        s1_end = np.zeros(0, dtype=np.int64)
        s1_bond = np.zeros(0, dtype=np.int64)
    s1_counts = np.bincount(s1_root, minlength=max(n_atoms, 1)).astype(np.int64)
    s1_ptr = np.concatenate([[0], np.cumsum(s1_counts)]).astype(np.int64)

    # ---- S2: frozen CPC index, closure recomputed from adjacency ---------
    s2_root = np.asarray(cpc_np["cpc_path_atom"], dtype=np.int64)
    s2_middle = np.asarray(cpc_np["cpc_middle"], dtype=np.int64)
    s2_end = np.asarray(cpc_np["cpc_end"], dtype=np.int64)
    s2_bond1 = np.asarray(cpc_np["cpc_bond_ij"], dtype=np.int64)
    s2_bond2 = np.asarray(cpc_np["cpc_bond_jk"], dtype=np.int64)
    if s2_root.size:
        keys = _adjacency_keys(bi, max(n_atoms, 1))
        lo = np.minimum(s2_root, s2_end)
        hi = np.maximum(s2_root, s2_end)
        probe = lo * int(max(n_atoms, 1)) + hi
        pos = np.searchsorted(keys, probe)
        pos = np.clip(pos, 0, max(keys.size - 1, 0))
        closure = (keys[pos] == probe).astype(np.float32) if keys.size else np.zeros(probe.size, np.float32)
    else:
        closure = np.zeros(0, dtype=np.float32)
    s2_counts = np.bincount(s2_root, minlength=max(n_atoms, 1)).astype(np.int64)
    s2_ptr = np.concatenate([[0], np.cumsum(s2_counts)]).astype(np.int64)

    def _t(arr, dtype):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(arr, dtype=dtype)))

    out = dict(base)
    out.update(
        {
            "rows": _t(rows, np.int64),
            "n_atoms": _t([n_atoms], np.int64),
            "n_bonds": _t([n_bonds], np.int64),
            "n_s1": _t([int(s1_root.size)], np.int64),
            "n_s2": _t([int(s2_root.size)], np.int64),
            "s1_root": _t(s1_root, np.int64),
            "s1_end": _t(s1_end, np.int64),
            "s1_bond": _t(s1_bond, np.int64),
            "s1_ptr": _t(s1_ptr, np.int64),
            "s1_counts": _t(s1_counts, np.int64),
            "s2_root": _t(s2_root, np.int64),
            "s2_middle": _t(s2_middle, np.int64),
            "s2_end": _t(s2_end, np.int64),
            "s2_bond1": _t(s2_bond1, np.int64),
            "s2_bond2": _t(s2_bond2, np.int64),
            "s2_closure": _t(closure, np.float32),
            "s2_ptr": _t(s2_ptr, np.int64),
            "s2_counts": _t(s2_counts, np.int64),
            "frozen_path_closure": _t(np.asarray(base["path_closure"], dtype=np.int64), np.float32),
        }
    )
    return out


def audit_batch_structures(dev: DevTask, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Exact structural audit of one collated batch (CPU, no model)."""
    import torch

    atom_batch = batch["atom_batch"]
    atom_counts = batch["atom_counts"]
    s1_root = batch["s1_root"]
    s1_end = batch["s1_end"]
    s1_bond = batch["s1_bond"]
    bi = batch["bond_index"]
    s2_root = batch["s2_root"]
    s2_middle = batch["s2_middle"]
    s2_end = batch["s2_end"]
    s2_b1 = batch["s2_bond1"]
    s2_b2 = batch["s2_bond2"]
    closure = batch["s2_closure"]
    n_atoms = int(batch["n_atoms"])
    n_bonds = int(batch["n_bonds"])

    problems: dict[str, int] = {}
    problems["s1_is_twice_bonds"] = int(s1_root.numel() != 2 * n_bonds)
    problems["s1_ptr_end"] = int(int(batch["s1_ptr"][-1]) != int(s1_root.numel()))
    problems["s2_ptr_end"] = int(int(batch["s2_ptr"][-1]) != int(s2_root.numel()))

    # S1: physical bond endpoints and same-molecule membership
    if n_bonds:
        u = bi.index_select(0, s1_bond)
        up = u[:, 0]
        vp = u[:, 1]
        ok_end = ((up == s1_root) & (vp == s1_end)) | ((up == s1_end) & (vp == s1_root))
        problems["s1_bond_endpoints"] = int((~ok_end).sum())
        problems["s1_cross_molecule"] = int(
            (atom_batch.index_select(0, s1_root) != atom_batch.index_select(0, s1_end)).sum()
        )
        problems["s1_root_major"] = int((s1_root[1:] < s1_root[:-1]).sum())

    # S2: three distinct atoms, same molecule, both bonds real
    if s2_root.numel():
        same = (
            (atom_batch.index_select(0, s2_root) == atom_batch.index_select(0, s2_middle))
            & (atom_batch.index_select(0, s2_middle) == atom_batch.index_select(0, s2_end))
        )
        problems["s2_cross_molecule"] = int((~same).sum())
        distinct = (
            (s2_root != s2_middle) & (s2_middle != s2_end) & (s2_root != s2_end)
        )
        problems["s2_not_simple"] = int((~distinct).sum())
        b1 = bi.index_select(0, s2_b1)
        b2 = bi.index_select(0, s2_b2)
        ok1 = ((b1[:, 0] == s2_root) & (b1[:, 1] == s2_middle)) | (
            (b1[:, 0] == s2_middle) & (b1[:, 1] == s2_root)
        )
        ok2 = ((b2[:, 0] == s2_middle) & (b2[:, 1] == s2_end)) | (
            (b2[:, 0] == s2_end) & (b2[:, 1] == s2_middle)
        )
        problems["s2_bond1_endpoints"] = int((~ok1).sum())
        problems["s2_bond2_endpoints"] = int((~ok2).sum())
        problems["s2_root_major"] = int((s2_root[1:] < s2_root[:-1]).sum())
        problems["s2_closure_vs_frozen"] = int(
            (closure != batch["frozen_path_closure"]).sum()
        )

    # per-root S2 count identity: sum_{w in N(v)} (deg(w) - 1)
    deg = torch.zeros(n_atoms, dtype=torch.long)
    if n_bonds:
        deg.index_add_(0, bi[:, 0], torch.ones(n_bonds, dtype=torch.long))
        deg.index_add_(0, bi[:, 1], torch.ones(n_bonds, dtype=torch.long))
    expected = torch.zeros(n_atoms, dtype=torch.long)
    if n_bonds:
        # for each root, sum over its neighbours of (deg(neighbour) - 1)
        contri = (deg - 1).clamp(min=0)
        expected.index_add_(0, bi[:, 0], contri.index_select(0, bi[:, 1]))
        expected.index_add_(0, bi[:, 1], contri.index_select(0, bi[:, 0]))
    problems["s2_count_identity"] = int((expected != batch["s2_counts"]).sum())
    problems["s1_count_equals_degree"] = int((deg != batch["s1_counts"]).sum())
    problems["s1_ptr_vs_counts"] = int(
        (torch.cumsum(batch["s1_counts"], 0) != batch["s1_ptr"][1:]).sum()
    )
    problems["s2_ptr_vs_counts"] = int(
        (torch.cumsum(batch["s2_counts"], 0) != batch["s2_ptr"][1:]).sum()
    )
    problems["atom_counts_sum"] = int(int(atom_counts.sum()) != n_atoms)
    return {
        "passed": all(v == 0 for v in problems.values()),
        "problems": problems,
        "n_molecules": int(atom_counts.numel()),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_s1": int(s1_root.numel()),
        "n_s2": int(s2_root.numel()),
        "n_s2_closed": int(closure.sum()),
    }


def s2_count_closed_form(n_atoms: int, bi: Sequence[Sequence[int]]) -> np.ndarray:
    """Independent per-root ``sum_{w in N(v)} (deg(w) - 1)`` reference (tests)."""
    n = int(n_atoms)
    deg = np.zeros(n, dtype=np.int64)
    nbrs: list[list[int]] = [[] for _ in range(n)]
    for u, v in bi:
        deg[int(u)] += 1
        deg[int(v)] += 1
        nbrs[int(u)].append(int(v))
        nbrs[int(v)].append(int(u))
    out = np.zeros(n, dtype=np.int64)
    for v in range(n):
        out[v] = sum(int(deg[w]) - 1 for w in nbrs[v])
    return out


__all__ = [
    "DevTask",
    "audit_batch_structures",
    "collate_batch",
    "load_task",
    "make_plan40",
    "read_official_graph_labels",
    "s2_count_closed_form",
]
