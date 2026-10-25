"""Frozen-weight mechanism diagnostics (forward only, no training).

Answers "which explicit statistic reaches which part of the prediction" for the
*selected* checkpoints.  All measurements are dependency probes on frozen
weights; they do not estimate the effect of retraining without the mechanism,
and the report says so.  Official test is never loaded.

Activations are captured with hooks on the production model, so the numbers
cannot drift from the trained computation.  The only input/interface change is a
temporary conductance override (F) or a rotated root binding (Q), both of which
keep every per-object count and every molecule-level record multiset intact.

Run: uv run python .../sps_mechdiag.py --runs runs/QF_s0 runs/FC_s0
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_AD = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_AD)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import sps_data as SD  # noqa: E402
import sps_models as M  # noqa: E402

SPS = _HERE.parent
ROWS = np.arange(1000, dtype=np.int64)


def load(cand: str, run: Path, ckpt: str, device="cpu"):
    st = torch.load(run / ckpt, map_location="cpu", weights_only=False)
    model, _ = M.build_model(cand, SD.load_fit(), int(st["seed"]), torch.device(device),
                             ssl_path=None)
    model.load_state_dict(st["state_dict"])
    model.eval()
    return model, st


@contextlib.contextmanager
def capture_out(module, store: list):
    h = module.register_forward_hook(lambda _m, _i, o: store.append(o.detach()))
    try:
        yield
    finally:
        h.remove()


@contextlib.contextmanager
def capture_in(module, store: list):
    h = module.register_forward_pre_hook(lambda _m, i: store.append(i[0].detach()))
    try:
        yield
    finally:
        h.remove()


def batches(cand: str, rows=ROWS, split="valid"):
    ad = D.load_cache(split)
    coll = SD.collate_factory(cand)
    for s in range(0, len(rows), 256):
        yield ad, coll(ad, rows[s:s + 256], cand, valid=True)


def roll_binding(batch: dict) -> dict:
    """Rotate the root<->record binding by one root inside every molecule.

    The molecule's multiset of (content, distance) records, its per-root record
    counts and its structure are all unchanged; only which root owns which
    record (and which counts) moves.
    """
    n_loc = batch["atom_counts"].long()
    m_loc = batch["bond_counts"].long()
    B = int(batch["n_graphs"])
    a_base = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(n_loc ** 2, 0)[:-1]])
    b_base = torch.cat([torch.zeros(1, dtype=torch.long),
                        torch.cumsum(n_loc * m_loc, 0)[:-1]])
    out = dict(batch)
    for key in ("atom_rec_atom", "atom_rec_dist", "atom_rec_group", "atom_rec_root"):
        src = batch[key]
        dst = src.clone()
        for i in range(B):
            n, s = int(n_loc[i]), int(a_base[i])
            dst[s:s + n * n] = torch.roll(src[s:s + n * n], n)
        out[key] = dst
    for key in ("bond_rec_group", "bond_rec_root", "bond_rec_bond"):
        src = batch[key]
        dst = src.clone()
        for i in range(B):
            n, m, s = int(n_loc[i]), int(m_loc[i]), int(b_base[i])
            if m:
                dst[s:s + n * m] = torch.roll(src[s:s + n * m], m)
        out[key] = dst
    for key in ("bond_rec_iu", "bond_rec_iv"):
        src = batch[key]
        dst = src.clone()
        for i in range(B):
            n, m, s = int(n_loc[i]), int(m_loc[i]), int(b_base[i])
            if m:
                nb, ab = n * n, int(a_base[i])
                dst[s:s + n * m] = (src[s:s + n * m] - ab + n) % nb + ab
        out[key] = dst
    # counts are recomputed from the rolled records so the probe input is
    # self-consistent (a count is only ever a function of its own records)
    A = int(batch["root_cnt"].shape[0])
    seg = torch.cat([out["atom_rec_root"] * 6 + out["atom_rec_group"],
                     out["bond_rec_root"] * 6 + out["bond_rec_group"] + 3])
    cnt = torch.zeros(A * 6, dtype=batch["root_cnt"].dtype)
    cnt.index_add_(0, seg, torch.ones_like(seg, dtype=cnt.dtype))
    out["root_cnt"] = cnt.reshape(A, 6)
    # invariance self-check: the molecule-level record multiset is untouched
    assert torch.equal(out["root_cnt"].sum(0), batch["root_cnt"].sum(0))
    return out


@torch.no_grad()
def q_diag(model, rows=ROWS) -> dict:
    out: dict[str, float] = {}
    view = SD.load_q_view("valid")
    ad = D.load_cache("valid")
    n = np.diff(np.asarray(ad["atom_ptr"])).astype(np.int64)
    m = np.diff(np.asarray(ad["bond_ptr"])).astype(np.int64)
    sel = np.concatenate([np.arange(int(np.asarray(ad["atom_ptr"])[r]),
                                    int(np.asarray(ad["atom_ptr"])[r + 1])) for r in rows])
    cnt = np.asarray(view["root_grp_count"])[sel]
    out["root_group_count_mean"] = cnt.mean(0).tolist()
    out["root_group_nonzero_fraction"] = (cnt > 0).mean(0).tolist()
    out["records_per_root_mean"] = float((n ** 2 + n * m).mean())

    surv, pred0, pred1 = [], [], []
    for ad_b, batch in batches("QF", rows):
        hs: list[torch.Tensor] = []
        with capture_out(model.root_encoder, hs):
            p0 = model(batch)
        h = hs[0]
        B = int(batch["n_graphs"])
        s = h.new_zeros((B, h.shape[1])).index_add_(0, batch["atom_batch"], h)
        l1 = h.new_zeros((B, h.shape[1])).index_add_(0, batch["atom_batch"], h.abs())
        surv.append((s.norm(dim=1) / l1.norm(dim=1).clamp(min=1e-9)).cpu().numpy())
        p1 = model(roll_binding(batch))
        pred0.append(p0.cpu().numpy())
        pred1.append(p1.cpu().numpy())
    p0 = np.concatenate(pred0)
    p1 = np.concatenate(pred1)
    y = np.asarray(ad["labels"])[rows]
    sv = np.concatenate(surv)
    out["valid_mae"] = float(np.abs(p0 - y).mean())
    out["root_pool_survival_median"] = float(np.median(sv))
    out["root_pool_survival_p10_p90"] = [float(np.percentile(sv, 10)), float(np.percentile(sv, 90))]
    out["root_rebind_mae_change"] = float(np.abs(p1 - y).mean() - np.abs(p0 - y).mean())
    out["root_rebind_mean_abs_pred_shift"] = float(np.abs(p1 - p0).mean())
    out["root_rebind_pred_corr"] = float(np.corrcoef(p0, p1)[0, 1])
    return out


@contextlib.contextmanager
def override_conductance(model, z_star: torch.Tensor):
    """Replace the conductance logit (w = 0.1 + 1.9 sigmoid(z)) with a constant."""
    orig = model.conductance

    class _Const(torch.nn.Module):
        def forward(self, _b):
            return z_star.unsqueeze(-1)

    model.conductance = _Const()
    try:
        yield
    finally:
        model.conductance = orig


def _logit_w(w: torch.Tensor) -> torch.Tensor:
    """Inverse of w = 0.1 + 1.9*sigmoid(z), with a small safety clamp."""
    w = w.clamp(0.1 + 1e-6, 2.0 - 1e-6)
    return torch.log((w - 0.1) / (2.0 - w))


@torch.no_grad()
def f_diag(model, rows=ROWS) -> dict:
    out: dict[str, float] = {}
    ad = D.load_cache("valid")
    w_all, r_cv, p_cv, pred0, wmean_pred, fixed_pred = [], [], [], [], [], []
    for ad_b, batch in batches("FC", rows):
        ws, ain, bin_ = [], [], []
        with capture_out(model.conductance, ws), capture_in(model.atom_lift, ain), \
                capture_in(model.bond_lift, bin_):
            p0 = model(batch)
        w = 0.1 + 1.9 * torch.sigmoid(ws[0].squeeze(-1))          # [E] positive weights
        w_all.append(w.cpu().numpy())
        ra = ain[0][:, M.WIDTH:]                                  # [A,4] root statistics
        pe = bin_[0][:, M.WIDTH:]                                 # [E,4] edge statistics
        n_loc = batch["atom_counts"].long()
        b_loc = batch["bond_counts"].long()
        off = 0
        for i in range(int(batch["n_graphs"])):
            nn = int(n_loc[i])
            if nn > 1:
                rr = ra[off:off + nn].cpu().numpy()
                r_cv.append((rr.std(0) / np.maximum(np.abs(rr.mean(0)), 1e-12)).tolist())
            off += nn
        off = 0
        for i in range(int(batch["n_graphs"])):
            mm = int(b_loc[i])
            if mm > 1:
                pp = pe[off:off + mm].cpu().numpy()
                p_cv.append((pp.std(0) / np.maximum(np.abs(pp.mean(0)), 1e-12)).tolist())
            off += mm
        B = int(batch["n_graphs"])
        ones = torch.ones_like(w)
        w_mol = torch.zeros(B).index_add_(0, batch["edge_batch"], w) / torch.zeros(B).index_add_(
            0, batch["edge_batch"], ones).clamp(min=1)
        with override_conductance(model, _logit_w(w_mol[batch["edge_batch"]])):
            pm = model(batch)
        with override_conductance(model, torch.full_like(w, math.log(0.9))):
            pf = model(batch)
        pred0.append(p0.cpu().numpy())
        wmean_pred.append(pm.cpu().numpy())
        fixed_pred.append(pf.cpu().numpy())
    w = np.concatenate(w_all)
    y = np.asarray(ad["labels"])[rows]
    p0 = np.concatenate(pred0)
    pm = np.concatenate(wmean_pred)
    pf = np.concatenate(fixed_pred)
    bp = np.asarray(ad["bond_ptr"])
    out["valid_mae"] = float(np.abs(p0 - y).mean())
    out["w_mean"] = float(w.mean())
    out["w_sd"] = float(w.std())
    out["w_min"] = float(w.min())
    out["w_max"] = float(w.max())
    out["w_range_fraction_of_interval"] = float((w.max() - w.min()) / 1.9)
    out["within_molecule_w_sd_median"] = float(np.median(
        [np.std(w[bp[r]:bp[r + 1]]) for r in rows if bp[r + 1] - bp[r] > 1]))
    out["root_stat_within_molecule_cv_median_by_tau"] = np.median(np.array(r_cv), 0).tolist()
    out["edge_stat_within_molecule_cv_median_by_tau"] = np.median(np.array(p_cv), 0).tolist()
    out["w_to_molecule_mean_mae_change"] = float(np.abs(pm - y).mean() - np.abs(p0 - y).mean())
    out["w_to_one_mae_change"] = float(np.abs(pf - y).mean() - np.abs(p0 - y).mean())
    out["w_to_one_mean_abs_pred_shift"] = float(np.abs(pf - p0).mean())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=[str(SPS / "runs" / "QF_s0"),
                                                  str(SPS / "runs" / "FC_s0")])
    ap.add_argument("--checkpoint", default="best.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(SPS / "analysis" / "mechanism.json"))
    args = ap.parse_args(argv)
    torch.set_num_threads(4)
    res: dict = {}
    for r in args.runs:
        run = Path(r)
        cand = run.name.split("_")[0]
        model, st = load(cand, run, args.checkpoint, args.device)
        entry = {"run": run.name, "checkpoint": args.checkpoint, "epoch": int(st["epoch"]),
                 "stored_valid_mae": float(st["valid_mae"])}
        if cand == "QF":
            entry.update(q_diag(model))
        elif cand == "FC":
            entry.update(f_diag(model))
        res[cand] = entry
        print(json.dumps(res[cand], indent=2, default=float), flush=True)
    res["scope"] = ("frozen selected checkpoints, official valid split, forward only; "
                    "dependency probes, not retraining counterfactuals; "
                    "official_test_loaded=false")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, default=float) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
