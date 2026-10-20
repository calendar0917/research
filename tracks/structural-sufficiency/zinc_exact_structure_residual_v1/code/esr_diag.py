"""Post-hoc diagnostics for the phase-1 negative result.

These checks were not part of the frozen training protocol; they only read the
frozen artifacts and answer "is the measured degradation an implementation
artifact or a property of the treatment?":

1. end-to-end type alignment: recompute corrected r2/r1 keys for a sample of
   valid molecules from the raw objects and compare with the cached assignment;
2. residual-scale sweep: valid MAE as the trained ``beta`` is multiplied by
   0 / 0.1 / 0.25 / 0.5 / 1 / 1.5 (scale 0 must equal the zero-residual read);
3. shared-stack drift: relative L2 distance of every non-residual module
   against the same-seed B soup, with B seed0 vs B seed1 as a noise reference.

Writes ``analysis/diagnostics.json``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, str(REPO), str(REPO / "tracks/structure-contribution/code")):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import esr_data as D  # noqa: E402
import esr_models as M  # noqa: E402
import esr_runner as R  # noqa: E402

SHARED = ("atom_encoder", "bond_encoder", "junction_encoder", "group_composer", "reader")


def alignment_check(n_molecules: int = 25) -> dict:
    objs = R.load_objects_valid()
    assigned = R.load_assigned("valid")
    vocab = R.load_vocab()
    index = vocab["index"]
    rng = np.random.default_rng(0)
    rows = sorted(int(i) for i in rng.choice(len(objs), size=n_molecules, replace=False))
    mismatches = 0
    checked_atoms = 0
    for i in rows:
        rec = D.all_molecule_keys([objs[i]])[0]
        D.attach_root_types([rec], [objs[i]])
        r2 = np.asarray([index.get(k, -1) for k in rec["r2"]], dtype=np.int32)
        r1 = np.asarray([index.get(k, -1) for k in rec["r1"]], dtype=np.int32)
        star = np.where(r2 >= 0, r2, r1)
        cached = np.asarray(assigned[i]["t_star"])
        mismatches += int(np.sum(star != cached))
        mismatches += int(np.sum(r2 != np.asarray(assigned[i]["r2_id"])))
        mismatches += int(np.sum(r1 != np.asarray(assigned[i]["r1_id"])))
        checked_atoms += int(star.size)
    return {"molecules": rows, "atoms": checked_atoms, "mismatches": mismatches, "ok": mismatches == 0}


def scale_sweep(keys: tuple[str, ...] = ("E_eta1", "H_eta1"), scales=(0.0, 0.1, 0.25, 0.5, 1.0, 1.5)) -> dict:
    y, _ = R.valid_labels()
    objs = R.load_objects_valid()
    assigned = R.load_assigned("valid")
    device = torch.device("cpu")
    out: dict = {}
    for key in keys:
        model, _, _ = R.model_for_run(key, 0, device)
        residual = model.atom_residual
        original = residual.beta.detach().clone()
        out[key] = {}
        with torch.no_grad():
            for s in scales:
                residual.beta.copy_(original).mul_(float(s))
                pred = M.predict(model, objs, assigned, device)
                out[key][str(s)] = float(np.abs(pred.astype(np.float64) - y).mean())
            residual.beta.copy_(original)
    return out


def train_dependence(keys: tuple[str, ...] = ("E_eta1", "H_eta1"), n_molecules: int = 2000) -> dict:
    """Train-set zero-correction probe: does the branch carry the training signal too?"""
    objs = R.load_objects_train()[:n_molecules]
    assigned = R.load_assigned("train")[:n_molecules]
    y = R.f10k._load_y_rows(list(range(n_molecules)))
    device = torch.device("cpu")
    out: dict = {}
    for key in keys:
        model, _, _ = R.model_for_run(key, 0, device)
        residual = model.atom_residual
        original = residual.beta.detach().clone()
        preds = {}
        with torch.no_grad():
            for s in (0.0, 1.0):
                residual.beta.copy_(original).mul_(float(s))
                preds[str(s)] = M.predict(model, objs, assigned, device, batch_size=512).astype(np.float64)
            residual.beta.copy_(original)
        out[key] = {
            "train_mae_scale0": float(np.abs(preds["0.0"] - y).mean()),
            "train_mae_scale1": float(np.abs(preds["1.0"] - y).mean()),
            "n_molecules": n_molecules,
        }
    return out


def reader_blocks() -> dict:
    """First-reader-layer weight mass per input block and residual share of a_sum."""
    device = torch.device("cpu")
    y, _ = R.valid_labels()
    objs = R.load_objects_valid()
    assigned = R.load_assigned("valid")
    out: dict = {}
    for key, seed in (("B", 0), ("E_eta1", 0), ("H_eta1", 0)):
        model, arm, _ = R.model_for_run(key, seed, device)
        w = model.reader.lins[0].weight.detach().to(torch.float64)
        blocks = {"z": (0, 96), "a_sum": (96, 192), "a_sq": (192, 288), "g": (288, 321)}
        total = float((w ** 2).sum())
        entry = {name: float((w[:, a:b] ** 2).sum() / total) for name, (a, b) in blocks.items()}
        # per-molecule residual share of the summed atom embedding
        shares = []
        with torch.no_grad():
            for i in range(0, len(objs), 256):
                rows = list(range(i, min(i + 256, len(objs))))
                batch = M.batch_with_types(objs, rows, assigned, device)
                h = model.encode_atoms({**batch, "atom_type": torch.full_like(batch["atom_type"], -1)})
                idx = batch["atom_batch"]
                corr = (
                    model.atom_residual(batch["atom_type"])
                    if arm != "B"
                    else torch.zeros_like(h)
                )
                for m in range(len(rows)):
                    hh = h[idx == m]
                    aa = corr[idx == m]
                    shares.append(float(aa.sum(0).norm() / hh.sum(0).norm().clamp(min=1e-9)))
        entry["residual_share_a_sum_mean"] = float(np.mean(shares)) if arm != "B" else 0.0
        entry["residual_share_a_sum_p90"] = float(np.percentile(shares, 90)) if arm != "B" else 0.0
        out[key] = entry
    return out


def stack_drift() -> dict:
    device = torch.device("cpu")
    states = {}
    for key, seed in (("B", 0), ("B", 1), ("E_eta1", 0), ("H_eta1", 0)):
        model, _, _ = R.model_for_run(key, seed, device)
        states[f"{key}_s{seed}"] = model.state_dict()
    out: dict = {}
    for name, st in states.items():
        if name.startswith("B_"):
            continue
        seed = int(name.split("_s")[1])
        ref = states[f"B_s{seed}"]
        per_module = {}
        for module in SHARED:
            num = den = 0.0
            for k in ref:
                if not k.startswith(f"{module}."):
                    continue
                a = st[k].detach().to(torch.float64)
                b = ref[k].detach().to(torch.float64)
                num += float(((a - b) ** 2).sum())
                den += float((b ** 2).sum())
            per_module[module] = float(np.sqrt(num / max(den, 1e-12)))
        out[name] = per_module
    # noise reference: B seed0 vs B seed1 (same recipe, different init/order/SSL)
    noise = {}
    for module in SHARED:
        num = den = 0.0
        for k in states["B_s0"]:
            if not k.startswith(f"{module}."):
                continue
            a = states["B_s1"][k].detach().to(torch.float64)
            b = states["B_s0"][k].detach().to(torch.float64)
            num += float(((a - b) ** 2).sum())
            den += float((b ** 2).sum())
        noise[module] = float(np.sqrt(num / max(den, 1e-12)))
    out["B_s1_vs_B_s0_noise"] = noise
    return out


def main() -> int:
    R.ANALYSIS.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": R.PROTOCOL_ID,
        "note": "post-hoc diagnostics, not part of the frozen protocol; read-only over frozen artifacts",
        "alignment": alignment_check(),
        "scale_sweep": scale_sweep(),
        "train_dependence": train_dependence(),
        "reader_weight_blocks": reader_blocks(),
        "stack_drift_relative_l2": stack_drift(),
    }
    R.write_json(R.ANALYSIS / "diagnostics.json", payload)
    print(f"alignment ok={payload['alignment']['ok']} atoms={payload['alignment']['atoms']}")
    for k, v in payload["scale_sweep"].items():
        print(k, {s: round(m, 5) for s, m in v.items()})
    for k, v in payload["stack_drift_relative_l2"].items():
        print("drift", k, {m: round(x, 3) for m, x in v.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
