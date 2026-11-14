"""D-MPNN performance reference (chemprop v2.2.1 computation).

What this is
------------
The message-passing, aggregation and FFN modules are **byte copies of the pinned
``chemprop`` 2.2.1 source** (``vendor/cpv221``), executed with a minimal
dependency shim (no lightning ``Trainer``, no chemprop CLI).  The featurizer is a
direct transcription of ``MultiHotAtomFeaturizer.v2`` + ``MultiHotBondFeaturizer``
+ ``SimpleMoleculeMolGraphFeaturizer.__call__`` from the same tag, with the
feature widths asserted to be (72, 14).

It is a genuine, independent performance *coordinate*, not a controlled causal
experiment: the input features, parameterisation and recipe all differ from C1.

Model (fixed): native v2 atom/bond features -> ``BondMessagePassing``
hidden 300, depth 3, bias False, undirected False, ReLU, message dropout 0 ->
mean aggregation -> ``MLP`` 300 -> 300 -> 1 (ReLU, dropout 0).  No extra
descriptors, no pretraining, no ensemble.

Recipe (fixed): batch 64, Adam, 100 epochs, 2-epoch linear warmup from
``init_lr=1e-4`` to the peak, then the pinned Noam-like exponential decay to
``min_lr=1e-4``; weight_decay 0, clip 5.  Regression targets are Train
standardised; classification uses ``BCEWithLogits``.

Featurisation is CPU work and is materialised locally (the compute host has no
RDKit); the training/eval host only consumes the prepared arrays.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lcpr_common as C

VENDOR = C.CODE_DIR / "vendor" / "cpv221"
import sys as _sys

if str(VENDOR) not in _sys.path:
    _sys.path.insert(0, str(VENDOR))

from chemprop.data import BatchMolGraph  # noqa: E402
from chemprop.nn.agg import MeanAggregation  # noqa: E402
from chemprop.nn.ffn import MLP as CPMLP  # noqa: E402
from chemprop.nn.message_passing.base import BondMessagePassing  # noqa: E402

ATOM_FDIM, BOND_FDIM = 72, 14
FEATURE_VERSION = "chemprop-v2.2.1-native-72-14"


# ---------------------------------------------------------------------------
# featurisation (transcribed from the pinned source)
# ---------------------------------------------------------------------------


def featurizer_shapes() -> dict[str, int]:
    if (ATOM_FDIM, BOND_FDIM) != (72, 14):
        raise AssertionError("chemprop v2.2.1 SimpleMoleculeMolGraphFeaturizer().shape != (72, 14)")
    return {"atom_fdim": ATOM_FDIM, "bond_fdim": BOND_FDIM}


def _atom_featurizer():
    from rdkit.Chem.rdchem import HybridizationType

    atomic_nums = {j: i for i, j in enumerate(list(range(1, 37)) + [53])}
    degrees = {i: i for i in range(6)}
    formal_charges = {j: i for i, j in enumerate([-1, -2, 1, 2, 0])}
    chiral_tags = {i: i for i in range(4)}
    num_hs = {i: i for i in range(5)}
    hybridizations = {
        ht: i for i, ht in enumerate([
            HybridizationType.S, HybridizationType.SP, HybridizationType.SP2,
            HybridizationType.SP2D, HybridizationType.SP3, HybridizationType.SP3D,
            HybridizationType.SP3D2,
        ])
    }
    subs = [atomic_nums, degrees, formal_charges, chiral_tags, num_hs, hybridizations]
    size = sum([1 + len(d) for d in subs]) + 2

    def feat(a) -> np.ndarray:
        x = np.zeros(size)
        if a is None:
            return x
        i = 0
        for value, choices in zip(
            [a.GetAtomicNum(), a.GetTotalDegree(), a.GetFormalCharge(),
             int(a.GetChiralTag()), int(a.GetTotalNumHs()), a.GetHybridization()],
            subs,
        ):
            x[i + choices.get(value, len(choices))] = 1
            i += len(choices) + 1
        x[i] = int(a.GetIsAromatic())
        x[i + 1] = 0.01 * a.GetMass()
        return x

    return feat, size


def _bond_featurizer():
    from rdkit.Chem.rdchem import BondType

    bond_types = [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC]
    stereos = list(range(6))
    #: pinned ``MultiHotBondFeaturizer.__len__``: 1 + len(bond_types) + 2 + (len(stereo) + 1)
    size = 1 + len(bond_types) + 2 + (len(stereos) + 1)

    def feat(b) -> np.ndarray:
        x = np.zeros(size, int)
        if b is None:
            x[0] = 1
            return x
        i = 1
        bt = b.GetBondType()
        bit = bond_types.index(bt) if bt in bond_types else len(bond_types)
        if bit != len(bond_types):
            x[i + bit] = 1
        i += len(bond_types)
        x[i] = int(b.GetIsConjugated())
        x[i + 1] = int(b.IsInRing())
        i += 2
        s = int(b.GetStereo())
        x[i + (stereos.index(s) if s in stereos else len(stereos))] = 1
        return x

    return feat, size


def molgraph(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"RDKit cannot parse SMILES {smiles!r}")
    Chem.SanitizeMol(mol)
    af, adim = _atom_featurizer()
    bf, bdim = _bond_featurizer()
    if (adim, bdim) != (ATOM_FDIM, BOND_FDIM):
        raise AssertionError(f"featurizer widths {(adim, bdim)} != {(ATOM_FDIM, BOND_FDIM)}")
    n_atoms, n_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    V = np.zeros((1, adim), np.single) if n_atoms == 0 else np.array([af(a) for a in mol.GetAtoms()], np.single)
    E = np.empty((2 * n_bonds, bdim))
    ei0: list[int] = []
    ei1: list[int] = []
    i = 0
    for bond in mol.GetBonds():
        x_e = bf(bond)
        E[i: i + 2] = x_e
        u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ei0.extend([u, v])
        ei1.extend([v, u])
        i += 2
    rev = np.arange(len(E)).reshape(-1, 2)[:, ::-1].ravel()
    return V, E, np.asarray([ei0, ei1], int), rev


def featurize_path(dataset: str) -> Path:
    return C.CACHE / f"dmpnn_{dataset}.npz"


def build_features(dataset: str, *, force: bool = False) -> dict[str, Any]:
    import lcpr_data as D

    path = featurize_path(dataset)
    if path.exists() and not force:
        z = np.load(path, allow_pickle=False)
        return {"dataset": dataset, "path": C.track_rel(path), "reused": True,
                "n_molecules": int(z["n_molecules"])}
    src = D.load_source(dataset, labels="none")
    smiles = [str(x) for x in src["smiles"]]
    V_parts, E_parts, edge_parts, rev_parts = [], [], [], []
    atom_off = 0
    edge_off = 0
    mol_atom_ptr = [0]
    mol_edge_ptr = [0]
    for smi in smiles:
        V, E, ei, rev = molgraph(smi)
        V_parts.append(V)
        E_parts.append(E)
        edge_parts.append(ei + atom_off if ei.size else ei)
        rev_parts.append(rev + edge_off)
        atom_off += V.shape[0]
        edge_off += E.shape[0]
        mol_atom_ptr.append(atom_off)
        mol_edge_ptr.append(edge_off)
    V = np.concatenate(V_parts, axis=0)
    E = np.concatenate(E_parts, axis=0) if E_parts else np.zeros((0, BOND_FDIM))
    edge_index = np.concatenate(edge_parts, axis=1) if edge_parts else np.zeros((2, 0), int)
    rev_edge_index = np.concatenate(rev_parts) if rev_parts else np.zeros(0, int)
    np.savez_compressed(
        path, V=V.astype(np.float32), E=E.astype(np.float32),
        edge_index=edge_index.astype(np.int32), rev_edge_index=rev_edge_index.astype(np.int32),
        mol_atom_ptr=np.asarray(mol_atom_ptr, np.int64),
        mol_edge_ptr=np.asarray(mol_edge_ptr, np.int64),
        n_molecules=np.int64(len(smiles)),
        n_atoms=np.int64(V.shape[0]), n_edges=np.int64(E.shape[0]),
        atom_fdim=np.int64(ATOM_FDIM), bond_fdim=np.int64(BOND_FDIM),
    )
    return {"dataset": dataset, "path": C.track_rel(path), "reused": False,
            "n_molecules": len(smiles), "n_atoms": int(V.shape[0]), "n_edges": int(E.shape[0]),
            "sha256": C.sha256_file(path), "features": FEATURE_VERSION}


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class DMPNNReference(nn.Module):
    def __init__(self, d_h: int = 300):
        super().__init__()
        self.message_passing = BondMessagePassing(
            d_v=ATOM_FDIM, d_e=BOND_FDIM, d_h=d_h, bias=False, depth=3, dropout=0.0,
            activation="relu", undirected=False,
        )
        self.agg = MeanAggregation()
        self.predictor = CPMLP.build(d_h, 1, hidden_dim=d_h, n_layers=1, dropout=0.0, activation="relu")

    def forward(self, bmg: BatchMolGraph) -> torch.Tensor:
        H = self.message_passing(bmg)
        g = self.agg(H, bmg.batch)
        return self.predictor(g).reshape(-1)


def load_feat(path: str | Path) -> dict[str, np.ndarray]:
    """Materialise the feature archive once.

    ``np.load`` on an ``.npz`` returns lazy members that re-decompress on *every*
    ``__getitem__``; batching then costs seconds per call.  All arrays used here
    are small (Lipophilicity: 1.6 MB), so they are loaded eagerly.
    """
    with np.load(path, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _batch(feat: Mapping[str, np.ndarray], rows: Sequence[int], device) -> BatchMolGraph:
    """Batch a set of *global* row ids; handles arbitrary (non-contiguous) rows."""
    ap = np.asarray(feat["mol_atom_ptr"])
    ep = np.asarray(feat["mol_edge_ptr"])
    V_parts, E_parts, EI_parts, R_parts, b_parts = [], [], [], [], []
    a_off = 0
    e_off = 0
    for i, r in enumerate(int(x) for x in rows):
        a0, a1 = int(ap[r]), int(ap[r + 1])
        e0, e1 = int(ep[r]), int(ep[r + 1])
        V_parts.append(np.asarray(feat["V"][a0:a1], np.float32))
        E_parts.append(np.asarray(feat["E"][e0:e1], np.float32))
        EI_parts.append(np.asarray(feat["edge_index"][:, e0:e1], np.int64) - a0 + a_off)
        R_parts.append(np.asarray(feat["rev_edge_index"][e0:e1], np.int64) - e0 + e_off)
        b_parts.append(np.full(a1 - a0, i, np.int64))
        a_off += a1 - a0
        e_off += e1 - e0
    V = torch.from_numpy(np.concatenate(V_parts) if V_parts else np.zeros((0, ATOM_FDIM), np.float32))
    E = torch.from_numpy(np.concatenate(E_parts) if E_parts else np.zeros((0, BOND_FDIM), np.float32))
    edge = torch.from_numpy(np.concatenate(EI_parts, axis=1) if EI_parts else np.zeros((2, 0), np.int64))
    rev = torch.from_numpy(np.concatenate(R_parts) if R_parts else np.zeros(0, np.int64))
    vbatch = torch.from_numpy(np.concatenate(b_parts) if b_parts else np.zeros(0, np.int64))
    return BatchMolGraph(V, E, edge, rev, vbatch).to(device)


def _explicit_forward(mp: BondMessagePassing, bmg: BatchMolGraph) -> torch.Tensor:
    """Independent per-edge implementation of the pinned formulas (cross-check)."""
    V, E, edge_index, rev = bmg.V, bmg.E, bmg.edge_index, bmg.rev_edge_index
    src, dst = edge_index[0], edge_index[1]
    H0 = mp.W_i(torch.cat([V[src], E], dim=1))
    H = mp.tau(H0)
    n_atoms = V.shape[0]
    for _ in range(1, mp.depth):
        M = torch.zeros(n_atoms, H.shape[1], dtype=H.dtype, device=H.device)
        for j in range(H.shape[0]):
            M[dst[j]] = M[dst[j]] + H[j]
        M = M[src] - H[rev]
        H = mp.tau(H0 + mp.W_h(M))
    Mv = torch.zeros(n_atoms, H.shape[1], dtype=H.dtype, device=H.device)
    for j in range(H.shape[0]):
        Mv[dst[j]] = Mv[dst[j]] + H[j]
    return mp.tau(mp.W_o(torch.cat([V, Mv], dim=1)))


def parity_check(seed: int = 7717) -> dict[str, Any]:
    """Same weights, two independent implementations: outputs and gradients."""
    torch.manual_seed(seed)
    mp = BondMessagePassing(d_v=ATOM_FDIM, d_e=BOND_FDIM, d_h=24, bias=False, depth=3,
                            dropout=0.0, activation="relu", undirected=False)
    smiles = ["CCO", "c1ccccc1O", "CC(=O)Nc1ccc(O)cc1"]
    Vs, Es, eis, revs, batch = [], [], [], [], []
    off_a = 0
    off_e = 0
    for i, smi in enumerate(smiles):
        V, E, ei, rev = molgraph(smi)
        Vs.append(V)
        Es.append(E)
        eis.append(ei + off_a)
        revs.append(rev + off_e)
        batch.append(np.full(V.shape[0], i))
        off_a += V.shape[0]
        off_e += E.shape[0]
    bmg = BatchMolGraph(
        torch.from_numpy(np.concatenate(Vs).astype(np.float32)),
        torch.from_numpy(np.concatenate(Es).astype(np.float32)),
        torch.from_numpy(np.concatenate(eis, axis=1).astype(np.int64)),
        torch.from_numpy(np.concatenate(revs).astype(np.int64)),
        torch.from_numpy(np.concatenate(batch).astype(np.int64)),
    )
    H_ref = mp(bmg)
    H_exp = _explicit_forward(mp, bmg)
    out_diff = float((H_ref - H_exp).abs().max())
    g1 = torch.autograd.grad(H_ref.sum(), [p for p in mp.parameters()], retain_graph=True, allow_unused=True)
    mp.zero_grad(set_to_none=True)
    g2 = torch.autograd.grad(H_exp.sum(), [p for p in mp.parameters()], allow_unused=True)
    grad_diff = max(
        float((a - b).abs().max()) if a is not None and b is not None else 0.0 for a, b in zip(g1, g2)
    )
    return {
        "n_graphs": len(smiles),
        "n_atoms": int(bmg.V.shape[0]),
        "n_directed_edges": int(bmg.E.shape[0]),
        "max_abs_output_diff": out_diff,
        "max_abs_grad_diff": grad_diff,
        "tolerance": 1e-6,
        "passed": bool(out_diff < 1e-6 and grad_diff < 1e-6),
        "note": (
            "Cross-check between the vendored chemprop v2.2.1 BondMessagePassing forward and an "
            "independent explicit per-edge implementation of the same pinned formulas at identical weights."
        ),
    }


# ---------------------------------------------------------------------------
# training
# ---------------------------------------------------------------------------


def _noam_lambda(step: int, warmup_steps: int, cooldown_steps: int, init_lr: float,
                 max_lr: float, final_lr: float) -> float:
    if step < warmup_steps:
        factor = (max_lr - init_lr) / max(warmup_steps, 1)
        return step * factor / init_lr + 1.0
    if step < warmup_steps + cooldown_steps:
        cooldown_factor = (final_lr / max_lr) ** (1.0 / max(cooldown_steps, 1))
        return max_lr * (cooldown_factor ** (step - warmup_steps)) / init_lr
    return final_lr / init_lr


def train_dmpnn(*, candidate: str, dataset: str, device, out_dir: Path, smoke: bool = False) -> dict[str, Any]:
    import lcpr_data as D
    import lcpr_train as T

    cand = C.CANDIDATES[candidate]
    recipe = dict(C.DMPNN_RECIPES[cand["recipe"]])
    meta = D.dataset_meta(dataset)
    task = meta["task"]
    epochs = 2 if smoke else int(recipe["epochs"])
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not smoke and C.git_dirty():
        raise SystemExit("formal run requested on a dirty tracked worktree; commit first")

    feat_info = build_features(dataset)
    feat = load_feat(featurize_path(dataset))
    tgt = D.targets(dataset, include_test=True)
    tnorm = tgt["target_norm"]
    train_rows = np.asarray(tgt["rows"]["train"], np.int64)
    n_train = len(train_rows)
    bs = int(recipe["batch_size"])
    steps_per_epoch = math.ceil(n_train / bs)
    total_steps = steps_per_epoch * epochs

    torch.manual_seed(int(C.TRAIN_SEED))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(C.TRAIN_SEED))
    model = DMPNNReference(d_h=int(C.DMPNN_FAMILY["hidden"])).to(device)
    n_params = int(sum(p.numel() for p in model.parameters()))
    torch.manual_seed(int(C.TRAIN_SEED))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(C.TRAIN_SEED))

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=float(recipe.get("optimizer_base_lr", recipe["peak_lr"])),
                                 weight_decay=float(recipe["weight_decay"]))
    warmup_steps = int(round(recipe["warmup_epochs"] * steps_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: _noam_lambda(
            s, warmup_steps, total_steps - warmup_steps, float(recipe["init_lr"]),
            float(recipe["peak_lr"]), float(recipe["min_lr"]),
        ),
    )
    #: LambdaLR multiplies the optimizer's *base* lr by ``lambda(step)``.  With
    #: ``lambda`` returning ``desired_lr / init_lr`` the base lr must be
    #: ``init_lr`` for the schedule to realise ``desired_lr`` (see ERRATA E4).
    _base = float(optimizer.param_groups[0]["initial_lr"])
    _lam = sched.lr_lambdas[0]
    schedule_endpoints = {
        "optimizer_base_lr": _base,
        "lambda_0": float(_lam(0)),
        "lambda_warmup_end": float(_lam(warmup_steps)),
        "lambda_last": float(_lam(total_steps - 1)),
        "lr_step0": _base * float(_lam(0)),
        "lr_warmup_end": _base * float(_lam(warmup_steps)),
        "lr_last": _base * float(_lam(total_steps - 1)),
        "warmup_steps": int(warmup_steps),
        "total_steps": int(total_steps),
        "scheduler_step_after_optimizer_step": True,
    }

    def _eval_all():
        preds = {}
        was_training = model.training
        snap = T.rng_snapshot(device)
        model.eval()
        with torch.no_grad():
            for split in C.SPLITS:
                rows = [int(x) for x in tgt["rows"][split]]
                chunks = []
                for i in range(0, len(rows), 1024):
                    bmg = _batch(feat, rows[i: i + 1024], device)
                    chunks.append(model(bmg).detach().float().cpu().numpy().reshape(-1))
                preds[split] = np.concatenate(chunks).astype(np.float64)
        T.rng_restore(snap, device)
        model.train(was_training)
        return preds

    curve: list[dict[str, Any]] = []
    best = None
    best_epoch = None
    best_state = None
    started = time.perf_counter()
    shuffle_gen = torch.Generator().manual_seed(int(C.SHUFFLE_SEED))
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.perf_counter()
        order = torch.randperm(n_train, generator=shuffle_gen).numpy()
        sq = 0.0
        seen = 0
        for step in range(steps_per_epoch):
            sel = order[step * bs: (step + 1) * bs]
            bmg = _batch(feat, train_rows[sel], device)
            pred = model(bmg)
            if task == "regression":
                z = torch.as_tensor(
                    (tgt["y"]["train"][sel] - float(tnorm["mean"])) / float(tnorm["scale"]),
                    dtype=torch.float32, device=device,
                )
                loss = F.mse_loss(pred, z)
            else:
                y = torch.as_tensor(tgt["y"]["train"][sel], dtype=torch.float32, device=device)
                loss = F.binary_cross_entropy_with_logits(pred, y)
                z = y
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(recipe["gradient_clip"]))
            optimizer.step()
            sched.step()
            e = pred.detach() - z
            sq += float((e * e).sum())
            seen += int(e.numel())
        train_seconds = time.perf_counter() - t0
        preds = _eval_all()
        metrics = {s: T.split_metrics(task, preds[s], tgt["y"][s], tnorm) for s in C.SPLITS}
        value = T.primary_value(task, metrics["valid"])
        improved = T.is_better(task, value, best)
        if improved:
            best, best_epoch = value, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        row = {"epoch": epoch, "lr_end": float(optimizer.param_groups[0]["lr"]),
               "train_mse_online": sq / max(seen, 1), "is_best": bool(improved),
               "train_seconds": float(train_seconds)}
        for s in C.SPLITS:
            for k, v in metrics[s].items():
                if isinstance(v, (int, float, np.floating)):
                    row[f"{s}_{k}"] = float(v)
        curve.append(row)
    wall = time.perf_counter() - started
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    best_preds = _eval_all()
    best_metrics = {s: T.split_metrics(task, best_preds[s], tgt["y"][s], tnorm) for s in C.SPLITS}

    T._write_curve(out_dir / "curve.csv", curve)
    torch.save({"state_dict": best_state, "best_epoch": int(best_epoch),
                "best_metric_value": float(best), "candidate": candidate, "dataset": dataset,
                "parameters": n_params, "features": FEATURE_VERSION}, out_dir / "best.pt")
    np.savez(out_dir / "predictions_best.npz", **{
        f"{s}_{k}": v for s in C.SPLITS
        for k, v in (("rows", np.asarray(tgt["rows"][s], np.int64)), ("y", tgt["y"][s]),
                     ("pred_std", best_preds[s]))
    })
    payload = {
        "schema": "lcrc-performance-route-dmpnn-v1",
        "round_id": C.ROUND_ID, "protocol_id": C.PROTOCOL_ID, "status": "completed",
        "smoke": bool(smoke), "candidate": candidate, "family": "DMPNN",
        "dataset": dataset, "task": task, "split_method": meta["split_method"],
        "implementation": {
            "source": "chemprop v2.2.1 (pinned), vendored message-passing/agg/FFN source",
            "featurizer": FEATURE_VERSION,
            "labelled_as": "D-MPNN formula reproduction on the pinned publication module",
            "features_info": feat_info,
        },
        "model": {**C.DMPNN_FAMILY, "parameters": n_params},
        "recipe": {**recipe, "epochs_run": len(curve), "steps_per_epoch": steps_per_epoch,
                   "optimizer_steps": len(curve) * steps_per_epoch,
                   "optimizer_base_lr": _base,
                   "schedule_endpoints": schedule_endpoints},
        "data": {"n_train": int(n_train), "n_valid": int(meta["sizes"][1]), "n_test": int(meta["sizes"][2]),
                 "rows_sha256": {s: C.row_sha256(tgt["rows"][s]) for s in C.SPLITS},
                 "target_norm": dict(tnorm), "test_used_for_selection": False},
        "metrics": {"best": best_metrics},
        "timing": {"train_wall_seconds": wall,
                   "process_wall_seconds": float(time.perf_counter() - T.PROCESS_T0)},
        "gpu": {"device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
                "gpu_hours_charged": float((time.perf_counter() - T.PROCESS_T0) / 3600.0)
                if device.type == "cuda" else 0.0},
        "environment": {**C.env_info(), "device": str(device)},
        "provenance": {"git_commit": C.git_commit(), "git_dirty_tracked": C.git_dirty(),
                       "code_hashes": C.code_hashes()},
    }
    C.write_json(out_dir / "train.json", payload)
    C.log(f"[{candidate}@{dataset}] best_epoch={best_epoch}/{len(curve)} params={n_params} "
          f"valid={T.primary_value(task, best_metrics['valid']):.6f} "
          f"test={T.primary_value(task, best_metrics['test']):.6f} wall={wall:.1f}s")
    return payload


__all__ = ["build_features", "featurize_path", "featurizer_shapes", "molgraph", "parity_check",
           "DMPNNReference", "train_dmpnn", "FEATURE_VERSION"]
