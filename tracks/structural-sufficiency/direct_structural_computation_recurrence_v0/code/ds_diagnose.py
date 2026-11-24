"""DSCR v0 mechanism diagnostics (Phase G) on frozen checkpoints -- no training.

Three families of measurements, all on the *frozen* best checkpoints:

1. **Arithmetic reproduction.**  ``own`` recomputes each arm's macro AP with an
   independent implementation of its layer-2 arithmetic; it must match the value
   recorded by the training run, otherwise nothing below is trustworthy.
2. **Readout substitution.**  ``BAG2 + real2_readout`` and
   ``REAL2 + bag2_readout`` swap only the layer-2 aggregation while keeping every
   weight frozen.  This asks what the *pairing structure* buys at fixed capacity,
   without retraining.  ``BAG2 + real2_readout_mean`` adds the degree
   normalisation back, so "pairing" and "normalisation" can be separated.
3. **Context strength** (``gamma=0`` / ``gamma=1``) on the frozen weights: a
   degeneracy probe, *not* a hyper-parameter search, and never used for selection.
4. **Traceable molecules** (at most three): the branch contributions that make the
   REAL2/BAG2 difference on one task, traced to concrete atoms and neighbours.

The Test column is reported for every variant but never enters a decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import ds_common as C  # noqa: E402
import ds_model as M  # noqa: E402
import pc_metrics as MET  # noqa: E402
import pc_model as PC  # noqa: E402

#: 15-class frozen ``lipo_v1`` atom vocabulary (element, aromatic).
ATOM_CLASSES = (
    ("B", False), ("Br", False), ("C", False), ("C", True), ("Cl", False),
    ("F", False), ("I", False), ("N", False), ("N", True), ("O", False),
    ("O", True), ("P", False), ("S", False), ("S", True), ("Si", False),
)
#: (arm, variant, gamma) for every frozen-weight probe.
VARIANTS = (
    ("REAL2", "own", None),
    ("REAL2", "gamma0", 0.0),
    ("REAL2", "gamma1", 1.0),
    ("REAL2", "bag2_readout", None),
    ("BAG2", "own", None),
    ("BAG2", "gamma0", 0.0),
    ("BAG2", "gamma1", 1.0),
    ("BAG2", "real2_readout_sum", None),
    ("BAG2", "real2_readout_mean", None),
)


def symbol(atom_type: int) -> str:
    if 0 <= int(atom_type) < len(ATOM_CLASSES):
        el, arom = ATOM_CLASSES[int(atom_type)]
        return f"{el}{'a' if arom else ''}"
    return f"?{int(atom_type)}"


def _one_hop(z, src, dst, edge_msg, c, g) -> torch.Tensor:
    if src.numel() == 0:
        return torch.zeros_like(z)
    return torch.zeros_like(z).index_add_(
        0, dst, F.silu(z.index_select(0, src) + edge_msg + g * c.index_select(0, src))
    )


def _pairs_mean(model: M.DSModel, z, src, dst, edge_msg, c, g) -> torch.Tensor:
    n_atoms = int(z.shape[0])
    pairs = M.bag2_pair_index(src, dst, n_atoms)
    n_pairs = int(pairs["n_pairs"])
    acc = torch.zeros_like(z)
    if n_pairs == 0:
        return acc
    outer, inner = pairs["outer_edge"], pairs["inner_source"]
    chunk = max(1, int(model.bag2_chunk))
    for start in range(0, n_pairs, chunk):
        stop = min(n_pairs, start + chunk)
        term = (
            z.index_select(0, src.index_select(0, outer[start:stop]))
            + edge_msg.index_select(0, outer[start:stop])
            + g * c.index_select(0, inner[start:stop])
        )
        acc.index_add_(0, pairs["pair_dst"][start:stop], F.silu(term))
    return acc / pairs["counts"].clamp(min=1.0).unsqueeze(1).to(z.dtype)


def arm_logits(model: M.DSModel, batch: Mapping[str, Any], variant: str, gamma: float | None) -> torch.Tensor:
    """Layer-2 arithmetic chosen by ``variant`` on top of this checkpoint's own h0/h1."""
    g = float(C.GAMMA if gamma is None else gamma)
    _a, _b, _s, h0 = model.encode_h0(batch)
    src, dst, edge_msg = M.directed_bonds(model, batch)
    h1, _m1 = model.stage1(h0, src, dst, edge_msg)
    blk2 = model.blocks[1]
    z_target = blk2.norm(h1)
    z = blk2.norm(h0)
    deg = M.incident_degree(int(z.shape[0]), dst, z)
    c, _deg2 = M.branch_context(z, src, dst, edge_msg)
    # the readout actually executed: ``own``/gamma probes keep the checkpoint's own readout
    readout = {"own": model.arm, "gamma0": model.arm, "gamma1": model.arm,
               "bag2_readout": "BAG2", "real2_readout_sum": "REAL2",
               "real2_readout_mean": "REAL2_mean"}.get(variant)
    if readout is None:  # pragma: no cover - defensive
        raise KeyError(variant)
    if readout == "BAG2":
        m2 = _pairs_mean(model, z, src, dst, edge_msg, c, g)
    else:
        m2 = _one_hop(z, src, dst, edge_msg, c, g)
        if readout == "REAL2_mean":
            m2 = m2 / deg.clamp(min=1.0).unsqueeze(1).to(z.dtype)
    h2 = h1 + M._apply_ffn(blk2, z_target, m2)
    return model.post_local(h2, batch)


class _Variant(torch.nn.Module):
    """Adapter so ``PC.predict_logits`` can drive an alternative layer-2 arithmetic."""

    def __init__(self, model: M.DSModel, variant: str, gamma: float | None):
        super().__init__()
        self.model = model
        self.variant = variant
        self.gamma = gamma

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        return arm_logits(self.model, batch, self.variant, self.gamma)


def load_frozen(arm: str, seed: int, device: torch.device, task: Any) -> tuple[M.DSModel, dict[str, Any], Path]:
    d = C.run_dir(arm, seed)
    ckpt = d / "checkpoint_best.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{arm} seed{seed} has no best checkpoint at {ckpt}")
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    model, _ = M.build_ds_model(task.spec, task.norm, C.N_TASKS, arm=arm, seed=int(seed), device="cpu")
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, C.read_json(d / "run.json"), ckpt


def score(model: M.DSModel, task: Any, device: torch.device, split: str, y: np.ndarray | None,
          variant: str, gamma: float | None) -> tuple[np.ndarray, dict[str, Any]]:
    rows = getattr(task, f"{split}_rows")
    logits = PC.predict_logits(_Variant(model, variant, gamma), task.cache, rows, device,
                               batch_size=C.EVAL_BATCH[split])
    yy = np.asarray(task.labels[rows]) if y is None else y
    rep = MET.validation_report(yy, PC.sigmoid(logits))
    return logits, {
        "ap": float(rep["ap"]),
        "n_tasks_scored": int(rep["n_tasks_scored"]),
        "per_task_ap": rep["per_task_ap"],
        "positive": rep["positive"],
    }


# ---------------------------------------------------------------------------
# traceable molecules
# ---------------------------------------------------------------------------


def _atom_start(task: Any, gid: int) -> int:
    return int(task.cache["atom_ptr"][gid])


def _terms(z, src, dst, edge_msg, c, g, *, mode: str) -> dict[int, list[dict[str, Any]]]:
    """Per target atom: every layer-2 term with the atom indices behind it."""
    out: dict[int, list[dict[str, Any]]] = {}
    if mode == "one_hop":
        for e in range(int(src.numel())):
            j, i = int(src[e]), int(dst[e])
            term = F.silu(z[j] + edge_msg[e] + g * c[j])
            out.setdefault(i, []).append({"term": term, "source_atom": j})
        return out
    pairs = M.bag2_pair_index(src, dst, int(z.shape[0]))
    counts = pairs["counts"]
    outer, inner = pairs["outer_edge"], pairs["inner_source"]
    for k in range(int(pairs["n_pairs"])):
        e = int(outer[k])
        j, i = int(src[e]), int(dst[e])
        l = int(inner[k])
        term = F.silu(z[j] + edge_msg[e] + g * c[l]) / float(counts[i].item() or 1)
        out.setdefault(i, []).append({"term": term, "source_atom": j, "inner_atom": l})
    return out


def _sum_terms(terms: Sequence[Mapping[str, Any]]) -> torch.Tensor | None:
    if not terms:
        return None
    return torch.stack([t["term"] for t in terms]).sum(0)


def _trace_one(model: M.DSModel, task: Any, gid: int, batch: Mapping[str, Any], mode: str) -> dict[str, Any]:
    g = float(C.GAMMA)
    with torch.inference_mode():
        _a, _b, _s, h0 = model.encode_h0(batch)
        src, dst, edge_msg = M.directed_bonds(model, batch)
        h1, _m1 = model.stage1(h0, src, dst, edge_msg)
        z = model.blocks[1].norm(h0)
        c, deg = M.branch_context(z, src, dst, edge_msg)
        terms = _terms(z, src, dst, edge_msg, c, g, mode=mode)
        n_atoms = int(z.shape[0])
        msgs = {i: _sum_terms(terms.get(i, [])) for i in range(n_atoms)}
        ranked = sorted((float(m.norm()) for m in msgs.values() if m is not None), reverse=True)
        tops = []
        a0 = _atom_start(task, gid)
        atom_type = torch.as_tensor(task.cache["atom_type"])
        for i in [i for i in range(n_atoms) if msgs.get(i) is not None]:
            tops.append((float(msgs[i].norm()), i))
        tops.sort(reverse=True)
        out = {
            "n_atoms": n_atoms,
            "max_degree": int(deg.max().item()) if deg.numel() else 0,
            "mean_degree": float(deg.mean().item()) if deg.numel() else 0.0,
            "message_norm_max": ranked[0] if ranked else 0.0,
            "message_norm_mean": float(np.mean(ranked)) if ranked else 0.0,
            "top_atoms": [],
        }
        for nrm, i in tops[:3]:
            branches = sorted(
                ({"norm": float(t["term"].norm()), **{k: v for k, v in t.items() if k != "term"}} for t in terms[i]),
                key=lambda x: -x["norm"],
            )[:3]
            for br in branches:
                for key in ("source_atom", "inner_atom"):
                    if key in br:
                        br[f"{key}_symbol"] = symbol(int(atom_type[a0 + br[key]]))
                        br[f"{key}_degree"] = int(deg[br[key]].item())
            out["top_atoms"].append(
                {
                    "atom": int(i),
                    "atom_symbol": symbol(int(atom_type[a0 + i])),
                    "degree": int(deg[i].item()) if deg.numel() else 0,
                    "message_norm": nrm,
                    "top_branches": branches,
                }
            )
    return out


def trace_molecules(
    real2: M.DSModel,
    bag2: M.DSModel,
    task: Any,
    *,
    logits2: np.ndarray,
    logitsb: np.ndarray,
    task_idx: int,
    n_trace: int = 3,
) -> list[dict[str, Any]]:
    rows = np.asarray(task.valid_rows)
    d = logits2[:, task_idx] - logitsb[:, task_idx]
    order = np.argsort(-d)
    picks: list[tuple[str, int]] = [("real2_wins", int(order[0])), ("bag2_wins", int(order[-1]))]
    deg_hi = []
    for pos in range(len(rows)):
        gid = int(rows[pos])
        b0 = int(task.cache["bond_ptr"][gid])
        b1 = int(task.cache["bond_ptr"][gid + 1])
        bonds = np.asarray(task.cache["bond_index"][b0:b1])
        deg = int(np.bincount(bonds.reshape(-1)).max()) if bonds.size else 0
        deg_hi.append((deg, pos))
    near = [pos for deg, pos in deg_hi if deg >= 4 and abs(d[pos]) < 0.05]
    if near:
        picks.append(("near_tie_high_degree", int(max(near, key=lambda p: d[p]))))
    out: list[dict[str, Any]] = []
    for label, pos in picks[:n_trace]:
        gid = int(rows[pos])
        dev = next(real2.parameters()).device
        batch = {k: (v.to(dev) if torch.is_tensor(v) else v)
                 for k, v in PC.collate(task.cache, [gid]).items()}
        rec = {
            "kind": label,
            "valid_row": pos,
            "global_row": gid,
            "smiles": _smiles(gid),
            "task": int(task_idx),
            "logit_real2": float(logits2[pos, task_idx]),
            "logit_bag2": float(logitsb[pos, task_idx]),
            "delta_logit_real2_minus_bag2": float(d[pos]),
            "REAL2": _trace_one(real2, task, gid, batch, "one_hop"),
            "BAG2": _trace_one(bag2, task, gid, batch, "pairs"),
        }
        out.append(rec)
    return out


def _smiles_path() -> Path:
    try:
        import lr_common as LC  # inherited: the frozen pcba_v1 cache location

        return Path(LC.CACHE_PATH) / "smiles.jsonl"
    except Exception:  # pragma: no cover - fallback keeps the trace optional
        return C.SS / "lcrc_pcba_scale_pilot_v1/cache/pcba_v1/smiles.jsonl"


def _smiles(gid: int) -> str | None:
    path = _smiles_path()
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i == gid:
                try:
                    return str(json.loads(line))
                except json.JSONDecodeError:
                    return line.strip().strip('"')
    return None


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(C.ANALYSIS / "mechanism_diagnostics.json"))
    parser.add_argument("--no-trace", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    task = C.SD.load_task()
    test_info = C.load_test_labels(task, cache_dir=C.RUNS / "_shared")
    y_test = np.asarray(test_info["labels"])

    models: dict[str, M.DSModel] = {}
    meta: dict[str, Any] = {}
    variants: dict[str, Any] = {}
    logits_store: dict[tuple[str, str, str], np.ndarray] = {}

    def run_variant(arm: str, variant: str, gamma: float | None) -> dict[str, Any]:
        entry: dict[str, Any] = {"variant": variant, "gamma": gamma, "splits": {}}
        for split in ("valid", "test"):
            lg, rep = score(models[arm], task, device, split,
                            y_test if split == "test" else None, variant, gamma)
            entry["splits"][split] = rep
            logits_store[(arm, variant, split)] = lg
        return entry

    for arm in ("REAL2", "BAG2"):
        model, run, ckpt = load_frozen(arm, 0, device, task)
        models[arm] = model
        meta[arm] = {
            "run_dir": str(C.run_dir(arm, 0)),
            "checkpoint": str(ckpt),
            "checkpoint_sha256": C.sha256_file(ckpt),
            "best_epoch": int(run["best_epoch"]),
            "recorded_best_valid_ap": float(run["best_valid_ap"]),
            "recorded_test_ap": float(run["best_checkpoint_eval"]["test"]["ap"]),
        }
        own = run_variant(arm, "own", None)
        variants[f"{arm}_own"] = own
        for varm, variant, gamma in VARIANTS:
            if varm != arm or variant == "own":
                continue
            entry = run_variant(arm, variant, gamma)
            entry["valid_delta_vs_own"] = entry["splits"]["valid"]["ap"] - own["splits"]["valid"]["ap"]
            entry["test_delta_vs_own"] = entry["splits"]["test"]["ap"] - own["splits"]["test"]["ap"]
            variants[f"{arm}_{variant}"] = entry

    reproduction = {}
    for arm in ("REAL2", "BAG2"):
        got_v = float(variants[f"{arm}_own"]["splits"]["valid"]["ap"])
        got_t = float(variants[f"{arm}_own"]["splits"]["test"]["ap"])
        rec_v = float(meta[arm]["recorded_best_valid_ap"])
        rec_t = float(meta[arm]["recorded_test_ap"])
        reproduction[arm] = {
            "valid_recorded": rec_v, "valid_recomputed": got_v, "valid_abs_diff": abs(got_v - rec_v),
            "test_recorded": rec_t, "test_recomputed": got_t, "test_abs_diff": abs(got_t - rec_t),
            # re-scoring identical weights differs by <=~1.4e-5 on this GPU
            # (index_add_ atomic summation order); the gate is a corruption guard
            "ok": bool(abs(got_v - rec_v) < 1e-4 and abs(got_t - rec_t) < 1e-4),
            "gate": 1e-4,
        }

    traces: list[dict[str, Any]] = []
    trace_task = None
    if not args.no_trace:
        pa = variants["REAL2_own"]["splits"]["valid"]["per_task_ap"]
        pb = variants["BAG2_own"]["splits"]["valid"]["per_task_ap"]
        common = [t for t in range(C.N_TASKS) if pa[t] is not None and pb[t] is not None]
        if common:
            trace_task = int(max(common, key=lambda t: abs(pa[t] - pb[t])))
            traces = trace_molecules(
                models["REAL2"], models["BAG2"], task,
                logits2=logits_store[("REAL2", "own", "valid")],
                logitsb=logits_store[("BAG2", "own", "valid")],
                task_idx=trace_task,
            )

    substitution = {
        "bag2_weights_real2_readout": {
            "valid_ap": variants["BAG2_real2_readout_sum"]["splits"]["valid"]["ap"],
            "delta_vs_bag2_own": variants["BAG2_real2_readout_sum"]["valid_delta_vs_own"],
            "delta_vs_real2_own": variants["BAG2_real2_readout_sum"]["splits"]["valid"]["ap"]
            - variants["REAL2_own"]["splits"]["valid"]["ap"],
        },
        "bag2_weights_real2_readout_mean": {
            "valid_ap": variants["BAG2_real2_readout_mean"]["splits"]["valid"]["ap"],
            "delta_vs_bag2_own": variants["BAG2_real2_readout_mean"]["valid_delta_vs_own"],
        },
        "real2_weights_bag2_readout": {
            "valid_ap": variants["REAL2_bag2_readout"]["splits"]["valid"]["ap"],
            "delta_vs_real2_own": variants["REAL2_bag2_readout"]["valid_delta_vs_own"],
            "delta_vs_bag2_own": variants["REAL2_bag2_readout"]["splits"]["valid"]["ap"]
            - variants["BAG2_own"]["splits"]["valid"]["ap"],
        },
    }

    payload = {
        "schema": "dscr-mechanism-diagnostics-v1",
        "round_id": C.ROUND_ID,
        "arms": meta,
        "reproduction": reproduction,
        "variants": variants,
        "substitution": substitution,
        "trace": {"task": trace_task, "molecules": traces},
        "interpretation_limits": [
            "readout substitution keeps every weight frozen: it compares computations at fixed capacity, "
            "not architectures retrained for the swapped readout",
            "the REAL2 readout is a plain sum while the BAG2 readout is degree-normalised; "
            "real2_readout_mean separates pairing from normalisation",
            "gamma is a degeneracy probe on frozen weights, not a hyper-parameter search, and never enters selection",
            "one checkpoint pair, one seed: mechanism evidence, not a population estimate",
        ],
        "test_role": C.TEST_ROLE,
        "created_utc": C.utcnow(),
    }
    C.write_json(Path(args.out), payload)
    C.log(f"[diagnose] reproduction {json.dumps(reproduction)}")
    C.log(f"[diagnose] substitution {json.dumps({k: round(v['valid_ap'], 6) for k, v in substitution.items()})}")
    C.log(f"[diagnose] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
