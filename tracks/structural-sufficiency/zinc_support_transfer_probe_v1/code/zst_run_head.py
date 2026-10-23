"""Driver: train G / S on C_fit, select on C_dev, evaluate once on Q_A / Q_B / Q_bg.

CPU only. Reads the frozen base predictions from ``predictions/base_F_s{seed}.npz``
and the frozen split manifest; writes head weights to ``runs/heads/`` and all
records to ``analysis/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import zst_common as Z  # noqa: E402
import zst_head as H  # noqa: E402
import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402

RUNS_HEADS = Z.RUNS / "heads"


def zero_init_replay_check() -> dict:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    f = object.__new__(H.Features)
    n = 8
    f.ctx = rng.normal(size=(n, Z.CTX_DIM))
    f.aux = rng.normal(size=(n, 2))
    f.atoms = rng.normal(size=(n, 4, Z.ATOM_IN_DIM))
    f.mask = np.zeros((n, 4))
    f.mask[:, :2] = 1.0
    f.gate = np.ones(n)
    H.attach_tensors(f)
    out = {}
    for kind in ("G", "S"):
        model = H.HeadS() if kind == "S" else H.HeadG()
        out[kind] = float(np.max(np.abs(H.predict_head(model, f))))
    return out


def sign_check() -> dict:
    """A constant positive correction must reduce L1 when the residual is positive."""
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    f = object.__new__(H.Features)
    f.ctx = rng.normal(size=(4, Z.CTX_DIM))
    f.aux = rng.normal(size=(4, 2))
    f.atoms = rng.normal(size=(4, 3, Z.ATOM_IN_DIM))
    f.mask = np.ones((4, 3))
    f.gate = np.ones(4)
    H.attach_tensors(f)
    model = H.HeadG()
    with torch.no_grad():
        model.readout[-1].bias.fill_(0.1)
    h = H.predict_head(model, f)
    base = np.zeros(4)
    dpos = float(np.abs(base + h - 1.0).mean() - np.abs(base - 1.0).mean())
    dneg = float(np.abs(base + h + 1.0).mean() - np.abs(base + 1.0).mean())
    return {"h": float(h[0]), "delta_ae_positive_residual": dpos,
            "delta_ae_negative_residual": dneg,
            "ok": bool(h[0] > 0.099 and dpos < 0 and dneg > 0)}


def s_path_check() -> dict:
    """The shared structural encoder must receive a finite non-zero gradient and
    the atom block must change the output after optimisation starts."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    f = object.__new__(H.Features)
    n = 16
    f.ctx = rng.normal(size=(n, Z.CTX_DIM))
    f.aux = rng.normal(size=(n, 2))
    f.atoms = rng.normal(size=(n, 5, Z.ATOM_IN_DIM))
    f.mask = np.ones((n, 5))
    f.gate = np.ones(n)
    H.attach_tensors(f)
    model = H.HeadS()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    y = torch.linspace(0.5, -0.5, n)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        h = H.forward_head(model, f)
        F_loss = torch.nn.functional.l1_loss(h, y)
        F_loss.backward()
        opt.step()
    g = model.psi[0].weight.grad
    finite = all(torch.isfinite(p.grad).all().item() for p in model.parameters()
                 if p.grad is not None)
    h_full = H.predict_head(model, f)
    f2 = object.__new__(H.Features)
    f2.__dict__.update(f.__dict__)
    f2.atoms_t = torch.zeros_like(f.atoms_t)
    h_zero = H.predict_head(model, f2)
    return {"psi_grad_abs_sum_after_5_steps": float(g.abs().sum()),
            "all_grads_finite": bool(finite),
            "output_change_when_atom_block_zeroed": float(np.max(np.abs(h_full - h_zero))),
            "ok": bool(float(g.abs().sum()) > 0 and finite
                       and float(np.max(np.abs(h_full - h_zero))) > 1e-6)}


def compact(rec: dict) -> dict:
    """Drop the per-molecule arrays (they are delivered as a CSV)."""
    out = json.loads(json.dumps(rec))
    for kind in ("G", "S"):
        for e in out.get("arms", {}).get(kind, []):
            for tag in ("q", "q_endpoint"):
                for qname, m in e.get(tag, {}).items():
                    m.pop("per_molecule", None)
    for qname, b in out.get("base_metrics", {}).items():
        b.pop("per_molecule", None)
    return out


def stride_stats(f: H.Features, base, y) -> dict:
    g = f.gate > 0
    return {"n": int(len(y)), "n_gated": int(g.sum()),
            "base_mae": float(np.abs(base - y).mean()),
            "base_mae_gated": float(np.abs(base[g] - y[g]).mean()) if g.any() else None,
            "unseen_atom_fraction_mean": float((f.mask.sum(1) / np.maximum(
                np.exp(f.aux[:, 1]) - 1.0, 1.0)).mean())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--manifest", default=str(Z.ANALYSIS / "split_manifest.json"))
    ap.add_argument("--steps", type=int, default=H.STEPS)
    ap.add_argument("--shuffle-control", action="store_true")
    args = ap.parse_args()

    man = json.load(open(args.manifest, encoding="utf-8"))
    roles = {k: np.asarray(v, dtype=np.int64) for k, v in man["roles"].items()}
    tkeys = {k: np.asarray(v, dtype=np.int64) for k, v in man["target_keys"].items()}
    tkeys_q = {"Q_A": tkeys["A"], "Q_B": tkeys["B"]}
    cache = D.load_cache("train")
    keys = Z.load_keys()
    mol1, mol2 = Z.molecule_key_sets(keys)
    atom_k1, atom_k2, parent = Z.atom_tables(keys, int(keys["n_key2"]))
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)

    pc = H.param_counts()
    checks = {
        "head_params": pc,
        "base_params": 413505,
        "total_params": {k: 413505 + v for k, v in pc.items()},
        "param_cap_ok": all(413505 + v <= 500_000 for v in pc.values()),
        "head_param_match_fraction": abs(pc["G"] - pc["S"]) / pc["S"],
        "zero_init_replay_max_abs_correction": zero_init_replay_check(),
        "sign": sign_check(),
        "s_path": s_path_check(),
        "support_gates_pass": bool(man["gates"]["all_pass"]),
    }
    sets = {k: set(roles[k].tolist()) for k in roles}
    checks["query_rows_outside_fit_and_dev"] = bool(
        not (sets["Q_A"] & (sets["F"] | sets["C_fit"] | sets["C_dev"]))
        and not (sets["Q_B"] & (sets["F"] | sets["C_fit"] | sets["C_dev"]))
        and not (sets["Q_bg"] & (sets["F"] | sets["C_fit"] | sets["C_dev"])))
    checks["cdev_outside_fit"] = bool(not (sets["C_dev"] & (sets["F"] | sets["C_fit"])))
    checks["all_pass"] = bool(checks["param_cap_ok"] and checks["support_gates_pass"]
                              and checks["query_rows_outside_fit_and_dev"]
                              and checks["cdev_outside_fit"]
                              and checks["sign"]["ok"] and checks["s_path"]["ok"]
                              and all(v == 0.0 for v in checks["zero_init_replay_max_abs_correction"].values()))
    Z.write_json(Z.ANALYSIS / "cpu_checks.json", checks)
    print("[checks]", json.dumps(checks, indent=1)[:1500], flush=True)
    if not checks["all_pass"]:
        raise SystemExit("CPU checks failed")

    RUNS_HEADS.mkdir(parents=True, exist_ok=True)
    all_records = {}
    for seed in [int(s) for s in args.seeds.split(",")]:
        data = H.run_base_seed(seed, man, cache, atom_k1, atom_k2, atom_ptr)
        rec, trained = H.build_and_train(seed, data, tkeys_q, mol2, steps=args.steps)
        rec.update({
            "seed": seed,
            "base_ckpt_sha256": data["base_ckpt_sha256"],
            "support_hash": data["support_hash"],
            "fit_rows_sha256": __import__("hashlib").sha256(np.ascontiguousarray(data["fit_rows"]).tobytes()).hexdigest(),
            "n_fit": int(len(data["fit_rows"])),
            "c_fit": stride_stats(data["f_cfit"], data["b_cfit"], data["y_cfit"]),
            "c_dev": stride_stats(data["f_cdev"], data["b_cdev"], data["y_cdev"]),
            "q_roles": {k: stride_stats(data["f_q"][k], data["b_q"][k], data["y_q"][k])
                        for k in data["f_q"]},
        })
        r = data["y_cfit"] - data["b_cfit"]
        g = data["f_cfit"].gate > 0
        rec["c_fit_residual_gated"] = {
            "mean": float(r[g].mean()), "std": float(r[g].std()),
            "mean_abs": float(np.abs(r[g]).mean()),
            "frac_positive": float((r[g] > 0).mean())}
        for kind in ("G", "S"):
            wd = rec["wd_selection"][kind]["selected_wd"]
            for entry in rec["arms"][kind]:
                tag = f"{kind}_s{seed}_wd{wd}_i{entry['init']}"
                p = RUNS_HEADS / f"{tag}.pt"
                torch.save({"kind": kind, "seed": seed, "wd": float(wd),
                            "init": int(entry["init"]), "steps_used": entry["steps_used"],
                            "state_dict": trained[f"{kind}_wd{wd}_i{entry['init']}"]["state_dict"],
                            "standardiser": {k: v.tolist() for k, v in data["standardiser"].items()},
                            "base_ckpt_sha256": data["base_ckpt_sha256"],
                            "support_hash": data["support_hash"]}, p)
                entry["checkpoint"] = str(p)
                entry["checkpoint_sha256"] = C.sha256_file(p)
                pe = RUNS_HEADS / f"{tag}_endpoint.pt"
                torch.save({"kind": kind, "seed": seed, "wd": float(wd),
                            "init": int(entry["init"]), "steps_used": int(entry["curve"][-1]["step"]),
                            "state_dict": trained[f"{kind}_wd{wd}_i{entry['init']}"]["endpoint_state_dict"],
                            "standardiser": {k: v.tolist() for k, v in data["standardiser"].items()},
                            "base_ckpt_sha256": data["base_ckpt_sha256"],
                            "support_hash": data["support_hash"]}, pe)
                entry["endpoint_checkpoint"] = str(pe)
                entry["endpoint_checkpoint_sha256"] = C.sha256_file(pe)
        # C_fit / C_dev rows: the head is *fitted* on C_fit and *selected* on C_dev,
        # so these roles have no honest post-selection head output; only the base
        # prediction is delivered (plus the support statistics).
        with open(Z.ANALYSIS / f"base_rows_s{seed}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["molecule_id", "stage", "role", "target_family", "seed", "y", "base_pred",
                        "G_pred_selected", "S_pred_selected", "G_pred_endpoint", "S_pred_endpoint",
                        "n_unseen", "unseen_frac", "min_n1", "mean_n1", "min_n2", "mean_n2",
                        "gated", "base_ckpt_sha256", "support_hash", "n_head_inits"])
            for role, f, b, y in (("C_fit", data["f_cfit"], data["b_cfit"], data["y_cfit"]),
                                  ("C_dev", data["f_cdev"], data["b_cdev"], data["y_cdev"])):
                ss = f.support_stats
                for i, mid in enumerate(f.rows):
                    w.writerow([int(mid), "new_holdout", role, "CD", seed, f"{y[i]:.6f}", f"{b[i]:.6f}",
                                "", "", "", "", int(ss["n_unseen"][i]), f"{ss['unseen_frac'][i]:.6f}",
                                f"{ss['min_n1'][i]:.4f}", f"{ss['mean_n1'][i]:.4f}",
                                f"{ss['min_n2'][i]:.4f}", f"{ss['mean_n2'][i]:.4f}",
                                int(f.gate[i] > 0), rec["base_ckpt_sha256"], rec["support_hash"], 3])

        all_records[str(seed)] = rec
        Z.write_json(Z.RUNS / "records" / f"head_records_full_s{seed}.json", rec)
        Z.write_json(Z.ANALYSIS / f"head_records_s{seed}.json", compact(rec))

    (Z.RUNS / "records").mkdir(parents=True, exist_ok=True)
    Z.write_json(Z.RUNS / "records" / "head_records_full.json", all_records)
    Z.write_json(Z.ANALYSIS / "head_records.json", {k: compact(v) for k, v in all_records.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
