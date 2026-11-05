"""Lightweight, post-hoc attention diagnostics for Arm A only (no training)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import nmp_common as C  # noqa: E402
import nmp_checkpoint as CK  # noqa: E402
import nmp_data as D  # noqa: E402
import nmp_model as M  # noqa: E402
import pc_model as BM  # noqa: E402


def _checkpoint(seed: int) -> Path:
    phase = "phaseC_seed0_A" if seed == 0 else "phaseD_seed1_A"
    return C.RUNS / phase / "checkpoint_epoch40.pt"


def _load_model(task: Any, seed: int, device: torch.device) -> M.NMPModel:
    ref, info, base = M.build_reference(task.spec, task.norm, C.N_TASKS, seed)
    model, _ = M.build_arm(
        "A", task.spec, task.norm, C.N_TASKS, seed=int(seed), device=device,
        base_init_state=base, zero_residual=True,
    )
    payload = CK.load_checkpoint(_checkpoint(seed))
    if payload.get("arm") != "A" or int(payload.get("seed", -1)) != int(seed):
        raise RuntimeError(f"checkpoint identity mismatch: {payload.get('arm')} seed {payload.get('seed')}")
    if int(payload.get("epoch", -1)) != 40:
        raise RuntimeError(f"attention diagnostic requires the fixed epoch-40 checkpoint, got {payload.get('epoch')}")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model


def _subset_rows(task: Any, split: str, n: int) -> np.ndarray:
    rows = np.asarray(task.valid_rows if split == "valid" else task.train_rows, dtype=np.int64)
    rows = np.sort(rows)
    idx = np.linspace(0, len(rows) - 1, int(n), dtype=np.int64)
    return rows[idx]


def _run_model(task, model, rows, device):
    batch = BM.to_device(BM.collate(task.cache, rows.tolist()), device)
    with torch.no_grad():
        a, b, scal = model._encode(batch)
        h = model.drop(model.local(batch, a, b, scal))
        _, alpha, pooled = model.readout(h, batch["atom_batch"], int(batch["n_graphs"]), return_attention=True)
    return batch, alpha.detach().cpu().numpy(), pooled.detach().cpu().numpy()


def _entropy(alpha: np.ndarray, atom_batch: np.ndarray, n_graphs: int) -> list[list[float]]:
    out = []
    for g in range(n_graphs):
        rows = alpha[atom_batch == g]
        e = []
        for k in range(rows.shape[1]):
            p = rows[:, k]
            e.append(float(-(p * np.log(np.clip(p, 1e-12, None))).sum() / np.log(max(2, rows.shape[0]))))
        out.append(e)
    return out


def diagnostics(task, *, split: str, n_molecules: int, device: torch.device) -> dict[str, Any]:
    rows = _subset_rows(task, split, n_molecules)
    models = {s: _load_model(task, s, device) for s in (0, 1)}
    outputs = {}
    for seed, model in models.items():
        batch, alpha, pooled = _run_model(task, model, rows, device)
        outputs[seed] = {
            "batch": batch,
            "alpha": alpha,
            "pooled": pooled,
            "atom_batch": batch["atom_batch"].detach().cpu().numpy(),
        }
    b0 = outputs[0]
    atom_batch = b0["atom_batch"]
    n_graphs = int(b0["batch"]["n_graphs"])
    alpha0, alpha1 = b0["alpha"], outputs[1]["alpha"]
    ent0 = _entropy(alpha0, atom_batch, n_graphs)
    ent1 = _entropy(alpha1, atom_batch, n_graphs)

    # Seed consistency at atom level and top-1 agreement per molecule/head.
    pearson = []
    top1_agree = []
    for k in range(alpha0.shape[1]):
        x, y = alpha0[:, k], alpha1[:, k]
        if x.std() > 1e-12 and y.std() > 1e-12:
            pearson.append(float(np.corrcoef(x, y)[0, 1]))
        else:  # pragma: no cover - constant weights
            pearson.append(0.0)
        agrees = 0
        for g in range(n_graphs):
            idx = np.where(atom_batch == g)[0]
            agrees += int(idx[np.argmax(alpha0[idx, k])] == idx[np.argmax(alpha1[idx, k])])
        top1_agree.append(float(agrees / max(1, n_graphs)))

    # Permutation check on the first few molecules.
    perm_diffs = []
    for g in range(min(8, n_graphs)):
        molecule_rows = rows[g : g + 1]
        batch, a, _ = _run_model(task, models[0], molecule_rows, device)
        n = a.shape[0]
        perm = np.random.default_rng(1234 + g).permutation(n)
        with torch.no_grad():
            aa, bb, ss = models[0]._encode(batch)
            h = models[0].drop(models[0].local(batch, aa, bb, ss))[torch.as_tensor(perm, device=batch["atom_feat"].device)]
            # recompute attention on permuted h with the same single-graph batch
            _, ap, _ = models[0].readout(h, batch["atom_batch"], int(batch["n_graphs"]), return_attention=True)
        perm_diffs.append(float(np.abs(ap.detach().cpu().numpy() - a[perm]).max()))

    # Molecule-level records with top atoms and structural categories.
    molecules = []
    atom_type_names = list(task.spec.atom_type_names)
    for g in range(n_graphs):
        idx = np.where(atom_batch == g)[0]
        al = alpha0[idx]
        top = []
        categories = []
        for k in range(al.shape[1]):
            order = np.argsort(-al[:, k])[:2]
            top_atoms = [int(i) for i in order]
            cats = []
            for local in top_atoms:
                row = int(idx[local])
                at = int(b0["batch"]["atom_type"][row].detach().cpu().item())
                sc = b0["batch"]["atom_scalar"][row].detach().cpu().numpy()
                cats.append({
                    "element": atom_type_names[at] if at < len(atom_type_names) else "UNK",
                    "degree": int(round(float(np.expm1(sc[3])))),
                    "in_ring": int(round(float(sc[4]))),
                    "atom_local_index": int(local),
                    "weight": float(al[local, k]),
                })
            top.append({
                "head": k,
                "top1_weight": float(al[order[0], k]),
                "top1_entropy_normalized": float(ent0[g][k]),
                "top_atoms": cats,
            })
            categories.extend(c["element"] for c in cats)
        molecules.append({
            "molecule_index": int(g),
            "row": int(rows[g]),
            "n_atoms": int(idx.size),
            "attention_weights": np.round(al, 6).tolist(),
            "entropy_normalized": [float(x) for x in ent0[g]],
            "entropy_normalized_seed1": [float(x) for x in ent1[g]],
            "top_atoms": top,
            "top_element_counts": {e: categories.count(e) for e in sorted(set(categories))},
        })

    npz_path = C.ANALYSIS / "attention_pooled.npz"
    np.savez_compressed(
        npz_path,
        rows=rows,
        atom_batch=atom_batch,
        weights_seed0=alpha0,
        weights_seed1=alpha1,
        pooled_seed0=b0["pooled"],
        pooled_seed1=outputs[1]["pooled"],
        atom_type=b0["batch"]["atom_type"].detach().cpu().numpy(),
        atom_scalar=b0["batch"]["atom_scalar"].detach().cpu().numpy(),
    )
    mean_entropy = np.asarray(ent0, dtype=np.float64).mean(axis=0)
    return {
        "schema": "lcrc-pcba-no-mp-attention-diagnostic-v1",
        "round_id": C.ROUND_ID,
        "arm": "A",
        "split": split,
        "n_molecules": int(n_graphs),
        "rows": [int(r) for r in rows],
        "checkpoint_seed0": C.track_rel(_checkpoint(0)),
        "checkpoint_seed1": C.track_rel(_checkpoint(1)),
        "selection": "deterministic evenly spaced sorted rows; fixed epoch 40; no checkpoint selection",
        "heads": 4,
        "mean_entropy_normalized_seed0": [float(x) for x in mean_entropy],
        "seed_consistency": {
            "pearson_atom_weights_per_head": pearson,
            "top1_agreement_per_head": top1_agree,
        },
        "permutation_max_abs_weight_diff": float(max(perm_diffs)) if perm_diffs else 0.0,
        "permutation_tolerance": 1e-5,
        "permutation_passed": bool(all(d <= 1e-5 for d in perm_diffs)),
        "molecules": molecules,
        "pooled_vectors_file": C.track_rel(npz_path),
        "pooled_vector_norms_seed0": [
            float(x) for x in np.linalg.norm(b0["pooled"].reshape(int(n_graphs), 4, -1), axis=2).mean(axis=0)
        ],
        "caveat": (
            "attention weights are readout diagnostics, not chemical causal attributions; "
            "the four heads are not task-specific attributions for the 128 assays"
        ),
        "test_evaluated": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="valid", choices=("train", "valid"))
    parser.add_argument("--n-molecules", type=int, default=C.ATTENTION_DIAGNOSTIC_MOLECULES)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(C.ANALYSIS / "attention_diagnostic.json"))
    args = parser.parse_args(argv)
    task = D.load_task()
    payload = diagnostics(
        task, split=args.split, n_molecules=int(args.n_molecules), device=torch.device(args.device)
    )
    C.write_json(Path(args.out), payload)
    C.log(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
