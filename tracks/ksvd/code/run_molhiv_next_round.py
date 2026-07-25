"""
Next-round design trials on molhiv (medium n): B/C/D/A.

B — dict fine grid + multi-seed residual (vs size)
C — patch_feat topo vs chem
D — ring_boost sampling
A — node-level s_v gate into GINE (smoke)

Protocol: molhiv-next-round-v0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv  # noqa: E402
from code.graph_level import GraphLevelConfig, encode_graph, learn_shared_D_graph_level  # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def fit_auc(Xtr, ytr, Xva, yva, Xte, yte, seed: int = 0) -> dict[str, float]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=4000, random_state=seed, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)

    def _auc(X, y):
        if len(np.unique(y)) < 2:
            return float("nan")
        proba = clf.predict_proba(X)
        classes = list(clf.named_steps["logisticregression"].classes_)
        if 1.0 in classes:
            col = classes.index(1.0)
        elif 1 in classes:
            col = classes.index(1)
        else:
            col = int(np.argmax(classes))
        return float(roc_auc_score(y, proba[:, col]))

    return {"train_auc": _auc(Xtr, ytr), "valid_auc": _auc(Xva, yva), "test_auc": _auc(Xte, yte)}


def size_feat(graphs) -> np.ndarray:
    return np.array([[g.n, g.num_edges()] for g in graphs], dtype=np.float64)


def encode_all(graphs, D, cfg, mode, node_feats=None, edge_feats=None):
    rows = []
    for i, g in enumerate(graphs):
        nf = node_feats[i] if node_feats is not None else None
        ef = edge_feats[i] if edge_feats is not None else None
        s, _ = encode_graph(
            g, D, cfg, mode=mode, seed=cfg.seed + i * 17, node_feat=nf, edge_feat=ef
        )
        rows.append(s)
    d = max(r.shape[0] for r in rows)
    X = np.zeros((len(rows), d), dtype=np.float64)
    for j, r in enumerate(rows):
        X[j, : r.shape[0]] = r
    return X


def eval_struct(
    graphs,
    y,
    tr,
    va,
    te,
    cfg: GraphLevelConfig,
    mode: str = "coverage",
    node_feats=None,
    edge_feats=None,
    size: np.ndarray | None = None,
) -> dict[str, Any]:
    D, dinfo = learn_shared_D_graph_level(
        graphs, tr, cfg, mode=mode, node_feats=node_feats, edge_feats=edge_feats
    )
    X = encode_all(graphs, D, cfg, mode, node_feats, edge_feats)
    only = fit_auc(X[tr], y[tr], X[va], y[va], X[te], y[te], cfg.seed)
    out: dict[str, Any] = {
        "s_only": only,
        "recon": dinfo.get("recon_rel"),
        "cover": dinfo.get("mean_traj_cover"),
        "dim": int(X.shape[1]),
        "n_atoms": cfg.n_atoms,
        "T": cfg.T,
        "patch_feat": cfg.patch_feat,
        "ring_boost": cfg.ring_boost,
        "pool": cfg.pool,
    }
    if size is not None:
        Xs = np.hstack([X, size])
        with_s = fit_auc(Xs[tr], y[tr], Xs[va], y[va], Xs[te], y[te], cfg.seed)
        out["s+size"] = with_s
        out["delta_val_vs_size"] = with_s["valid_auc"] - fit_auc(
            size[tr], y[tr], size[va], y[va], size[te], y[te], cfg.seed
        )["valid_auc"]
        # store size baseline once outside; recompute cheap
    return out


def run_BCD(max_graphs: int, seeds: list[int], out: dict) -> None:
    log("=== B/C/D structure residual ===")
    # load once with features for chem
    bundle = load_molhiv(max_graphs=max_graphs, seed=seeds[0], with_features=True)
    graphs, y = bundle.graphs, bundle.y
    tr, va, te = bundle.split["train"], bundle.split["valid"], bundle.split["test"]
    nf, ef = bundle.node_feats, bundle.edge_feats
    sz = size_feat(graphs)
    size_base = fit_auc(sz[tr], y[tr], sz[va], y[va], sz[te], y[te], seeds[0])
    out["size_baseline"] = size_base
    out["meta"] = bundle.meta
    log(f"size val={size_base['valid_auc']:.4f} test={size_base['test_auc']:.4f}")

    base = GraphLevelConfig(
        p=0.5,
        q=2.0,
        walk_length=8,
        max_nodes=8,
        max_walks=12,
        edge_decay=0.7,
        n_atoms=8,
        T=2,
        T_min=1,
        ksvd_iter=6,
        readout_mode="pool",
        pool="max",
        seed=seeds[0],
        max_train_patches=6000,
        patch_feat="topo",
        ring_boost=0.0,
    )

    # B: dict grid (single seed first, then multi-seed on best)
    dict_grid = [
        ("A6T2", dict(n_atoms=6, T=2, T_min=1)),
        ("A8T2", dict(n_atoms=8, T=2, T_min=1)),
        ("A8T3", dict(n_atoms=8, T=3, T_min=2)),
        ("A12T2", dict(n_atoms=12, T=2, T_min=1)),
        ("A12T3", dict(n_atoms=12, T=3, T_min=2)),
        ("A16T3", dict(n_atoms=16, T=3, T_min=2)),
    ]
    out["B_dict"] = {}
    for name, kw in dict_grid:
        cfg = replace(base, **kw, seed=seeds[0])
        log(f"[B] {name}")
        r = eval_struct(graphs, y, tr, va, te, cfg, node_feats=None, edge_feats=None, size=sz)
        # fix delta vs stored size_base
        if "s+size" in r:
            r["delta_val_vs_size"] = r["s+size"]["valid_auc"] - size_base["valid_auc"]
        out["B_dict"][name] = r
        log(
            f"  s_only val={r['s_only']['valid_auc']:.4f} "
            f"s+size val={r['s+size']['valid_auc']:.4f} dlt={r['delta_val_vs_size']:+.4f}"
        )

    best_b = max(out["B_dict"].items(), key=lambda kv: kv[1]["s+size"]["valid_auc"])
    log(f"[B] best by s+size valid: {best_b[0]}")
    best_kw = dict(dict_grid[[n for n, _ in dict_grid].index(best_b[0])][1])

    # multi-seed on best + A8T2 default
    out["B_multiseed"] = {}
    for tag, kw in [("best", best_kw), ("A8T2", dict(n_atoms=8, T=2, T_min=1))]:
        rows = []
        for sd in seeds:
            b2 = load_molhiv(max_graphs=max_graphs, seed=sd, with_features=False)
            g2, y2 = b2.graphs, b2.y
            tr2, va2, te2 = b2.split["train"], b2.split["valid"], b2.split["test"]
            sz2 = size_feat(g2)
            sb = fit_auc(sz2[tr2], y2[tr2], sz2[va2], y2[va2], sz2[te2], y2[te2], sd)
            cfg = replace(base, **kw, seed=sd)
            r = eval_struct(g2, y2, tr2, va2, te2, cfg, size=sz2)
            r["delta_val_vs_size"] = r["s+size"]["valid_auc"] - sb["valid_auc"]
            r["size_valid"] = sb["valid_auc"]
            r["subsample_seed"] = sd
            rows.append(r)
            log(
                f"  [B multi {tag} seed={sd}] dlt={r['delta_val_vs_size']:+.4f} "
                f"s+size={r['s+size']['valid_auc']:.4f}"
            )
        out["B_multiseed"][tag] = {
            "runs": rows,
            "mean_delta_val": float(np.mean([r["delta_val_vs_size"] for r in rows])),
            "std_delta_val": float(np.std([r["delta_val_vs_size"] for r in rows])),
            "mean_s+size_val": float(np.mean([r["s+size"]["valid_auc"] for r in rows])),
        }

    # C: chem vs topo (fixed A8T2)
    out["C_chem"] = {}
    for feat in ("topo", "chem"):
        cfg = replace(base, n_atoms=8, T=2, T_min=1, patch_feat=feat, seed=seeds[0])
        log(f"[C] patch_feat={feat}")
        r = eval_struct(
            graphs,
            y,
            tr,
            va,
            te,
            cfg,
            node_feats=nf if feat == "chem" else None,
            edge_feats=ef if feat == "chem" else None,
            size=sz,
        )
        r["delta_val_vs_size"] = r["s+size"]["valid_auc"] - size_base["valid_auc"]
        out["C_chem"][feat] = r
        log(
            f"  s+size val={r['s+size']['valid_auc']:.4f} dlt={r['delta_val_vs_size']:+.4f} "
            f"dim={r['dim']}"
        )

    # D: ring boost
    out["D_ring"] = {}
    for boost in (0.0, 1.0, 2.0, 4.0):
        cfg = replace(
            base, n_atoms=8, T=2, T_min=1, ring_boost=boost, patch_feat="topo", seed=seeds[0]
        )
        log(f"[D] ring_boost={boost}")
        r = eval_struct(graphs, y, tr, va, te, cfg, size=sz)
        r["delta_val_vs_size"] = r["s+size"]["valid_auc"] - size_base["valid_auc"]
        out["D_ring"][str(boost)] = r
        log(
            f"  s+size val={r['s+size']['valid_auc']:.4f} dlt={r['delta_val_vs_size']:+.4f} "
            f"cover={r.get('cover')}"
        )


def run_A_node_gate(max_graphs: int, seed: int, epochs: int, out: dict) -> None:
    """Node-level s_v gated into GINE vs gine_only / graph concat."""
    log("=== A node-level gate dual ===")
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F

        if not getattr(torch.load, "_ksvd_patched", False):
            _orig = torch.load

            def _load(*a, **k):
                k.setdefault("weights_only", False)
                return _orig(*a, **k)

            _load._ksvd_patched = True  # type: ignore[attr-defined]
            torch.load = _load  # type: ignore[assignment]

        from ogb.graphproppred import Evaluator, PygGraphPropPredDataset
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv, global_mean_pool
    except ImportError as e:
        out["A_node_gate"] = {"error": str(e)}
        log(f"A skip: {e}")
        return

    from code.node_struct import NodeStructConfig, encode_graph_nodes, learn_shared_D

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    root = Path(__file__).resolve().parents[3] / "data" / "ogb"

    bundle = load_molhiv(root=root, max_graphs=max_graphs, seed=seed, with_features=False)
    graphs = bundle.graphs
    tr = np.asarray(bundle.split["train"])
    va = np.asarray(bundle.split["valid"])
    te = np.asarray(bundle.split["test"])

    ncfg = NodeStructConfig(
        mode="B0",
        max_nodes=8,
        n_atoms=8,
        T=2,
        T_min=1,
        ksvd_iter=6,
        seed=seed,
        feat="abs_coef",
        pool="max",
    )
    log("[A] learn node-level D (B0 patches)...")
    D, dinfo = learn_shared_D(graphs, tr, ncfg)
    log(f"  recon={dinfo.get('recon_rel'):.4f} patches={dinfo.get('n_patches_used')}")

    # per-graph node features list + graph-level max pool for concat baseline
    node_S: list[np.ndarray] = []
    graph_S = []
    for i, g in enumerate(graphs):
        Sv, _ = encode_graph_nodes(g, D, ncfg)
        node_S.append(Sv.astype(np.float32))
        graph_S.append(np.max(np.abs(Sv), axis=0))
        if (i + 1) % 500 == 0:
            log(f"  node encode {i+1}/{len(graphs)}")
    graph_S_arr = np.stack(graph_S, axis=0).astype(np.float32)
    struct_dim = int(graph_S_arr.shape[1])

    # PyG subset aligned with load_molhiv
    pyg = PygGraphPropPredDataset(name="ogbg-molhiv", root=str(root))
    sp = pyg.get_idx_split()
    tr0 = np.asarray(sp["train"])
    va0 = np.asarray(sp["valid"])
    te0 = np.asarray(sp["test"])
    n_full = len(pyg)
    frac = max_graphs / n_full
    rng = np.random.default_rng(seed)
    n_tr = max(1, int(round(len(tr0) * frac)))
    n_va = max(1, int(round(len(va0) * frac)))
    n_te = max(1, int(round(len(te0) * frac)))
    total = n_tr + n_va + n_te
    while total > max_graphs and n_tr > 1:
        n_tr -= 1
        total -= 1
    while total < max_graphs and n_tr < len(tr0):
        n_tr += 1
        total += 1
    tr_o = rng.choice(tr0, min(n_tr, len(tr0)), False)
    va_o = rng.choice(va0, min(n_va, len(va0)), False)
    te_o = rng.choice(te0, min(n_te, len(te0)), False)
    keep = np.unique(np.concatenate([tr_o, va_o, te_o]))
    keep.sort()
    assert len(keep) == len(graphs)

    data_list = []
    for new_i, old_i in enumerate(keep):
        d = pyg[int(old_i)].clone()
        d.idx = torch.tensor([new_i], dtype=torch.long)
        d.y = d.y.view(-1).float()
        # attach node struct as extra tensor (n_nodes, struct_dim)
        d.s_node = torch.tensor(node_S[new_i], dtype=torch.float32)
        data_list.append(d)

    train_set = [data_list[i] for i in tr.tolist()]
    valid_set = [data_list[i] for i in va.tolist()]
    test_set = [data_list[i] for i in te.tolist()]
    tl = DataLoader(train_set, batch_size=64, shuffle=True)
    vl = DataLoader(valid_set, batch_size=64, shuffle=False)
    el = DataLoader(test_set, batch_size=64, shuffle=False)
    evaluator = Evaluator(name="ogbg-molhiv")
    GS = torch.tensor(graph_S_arr, dtype=torch.float32, device=device)

    class GINEStack(nn.Module):
        def __init__(self, hidden=64, layers=3, node_struct_dim=0, fuse="none"):
            super().__init__()
            self.fuse = fuse
            self.atom_encoder = AtomEncoder(hidden)
            self.bond_encoder = BondEncoder(hidden)
            self.node_struct_dim = node_struct_dim
            if fuse == "node_gate" and node_struct_dim > 0:
                self.s_proj = nn.Linear(node_struct_dim, hidden)
                self.gate = nn.Linear(hidden * 2, hidden)
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            for _ in range(layers):
                mlp = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden)
                )
                self.convs.append(GINEConv(mlp, train_eps=True))
                self.bns.append(nn.BatchNorm1d(hidden))
            self.hidden = hidden

        def forward(self, data):
            x = self.atom_encoder(data.x)
            if self.fuse == "node_gate" and self.node_struct_dim > 0:
                # data.s_node is batched via PyG — need to handle Batch
                s = data.s_node
                if s.dim() == 2 and s.size(0) == x.size(0):
                    z = F.relu(self.s_proj(s))
                    g = torch.sigmoid(self.gate(torch.cat([x, z], dim=-1)))
                    x = g * x + (1 - g) * z
            edge_attr = self.bond_encoder(data.edge_attr)
            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, data.edge_index, edge_attr)))
            return global_mean_pool(x, data.batch)

    class Head(nn.Module):
        def __init__(self, fuse, struct_dim, hidden=64, layers=3):
            super().__init__()
            self.fuse = fuse
            node_dim = struct_dim if fuse == "node_gate" else 0
            self.backbone = GINEStack(hidden, layers, node_dim, fuse if fuse == "node_gate" else "none")
            if fuse == "graph_concat":
                self.sp = nn.Linear(struct_dim, hidden)
                self.head = nn.Linear(hidden * 2, 1)
            else:
                self.head = nn.Linear(hidden, 1)

        def forward(self, data, g_s):
            h = self.backbone(data)
            if self.fuse == "graph_concat":
                return self.head(torch.cat([h, F.relu(self.sp(g_s))], dim=-1))
            return self.head(h)

    def collate_ok():
        # ensure Batch merges s_node — PyG default cats node attrs if same
        pass

    def run_fusion(fuse: str):
        model = Head(fuse, struct_dim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        def epoch(loader, train: bool):
            model.train(train)
            ys, ps = [], []
            for batch in loader:
                batch = batch.to(device)
                # s_node: after batching should be (sum_n, d)
                g_s = GS[batch.idx.view(-1)]
                out = model(batch, g_s).view(-1)
                y = batch.y.view(-1).float()
                m = (y == 0) | (y == 1)
                if int(m.sum()) == 0:
                    continue
                loss = F.binary_cross_entropy_with_logits(out[m], y[m])
                if train:
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                ys.append(y[m].detach().cpu())
                ps.append(out[m].detach().cpu())
            y_all = torch.cat(ys).numpy().reshape(-1, 1)
            p_all = torch.cat(ps).sigmoid().numpy().reshape(-1, 1)
            return float(evaluator.eval({"y_true": y_all, "y_pred": p_all})["rocauc"])

        best_v, best_t, best_e = -1.0, -1.0, -1
        for ep in range(1, epochs + 1):
            tr_a = epoch(tl, True)
            va_a = epoch(vl, False)
            te_a = epoch(el, False)
            if va_a > best_v:
                best_v, best_t, best_e = va_a, te_a, ep
            if ep % 5 == 0 or ep == 1:
                log(
                    f"  {fuse} ep{ep} tr={tr_a:.3f} va={va_a:.3f} te={te_a:.3f} best_te={best_t:.3f}"
                )
        return {
            "fusion": fuse,
            "best_val": best_v,
            "test_at_best_val": best_t,
            "best_ep": best_e,
        }

    results = {}
    for fuse in ("gine_only", "graph_concat", "node_gate"):
        log(f"[A] fusion={fuse}")
        results[fuse] = run_fusion(fuse)
    out["A_node_gate"] = {
        "n": len(graphs),
        "struct_dim": struct_dim,
        "node_cfg": "B0 A8T2",
        "epochs": epochs,
        "results": results,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-graphs", type=int, default=5000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip-A", action="store_true")
    ap.add_argument("--only-A", action="store_true")
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    t0 = time.time()
    out: dict[str, Any] = {
        "protocol_id": "molhiv-next-round-v0",
        "max_graphs": args.max_graphs,
        "seeds": seeds,
    }

    if not args.only_A:
        run_BCD(args.max_graphs, seeds, out)
    if not args.skip_A:
        run_A_node_gate(min(args.max_graphs, 3000), seeds[0], args.epochs, out)

    out["elapsed_sec"] = round(time.time() - t0, 2)
    out_dir = _TRACK / "results" / "molhiv"
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / f"next_round_n{args.max_graphs}.json"
    # numpy-safe dump
    def _conv(o):
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    with open(jpath, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=_conv)

    # markdown
    lines = [
        "# molhiv next-round (B/C/D/A)",
        "",
        f"n≈{args.max_graphs} · seeds={seeds}",
        "",
    ]
    if "size_baseline" in out:
        lines += [
            f"Size baseline valid **{out['size_baseline']['valid_auc']:.4f}**",
            "",
            "## B dict grid (s+size valid, seed0)",
            "",
            "| name | s_only val | s+size val | Δ vs size |",
            "|------|------------|------------|-----------|",
        ]
        for k, v in out.get("B_dict", {}).items():
            lines.append(
                f"| {k} | {v['s_only']['valid_auc']:.4f} | {v['s+size']['valid_auc']:.4f} | "
                f"{v['delta_val_vs_size']:+.4f} |"
            )
        lines += ["", "## B multi-seed mean Δ(s+size − size) valid", ""]
        for k, v in out.get("B_multiseed", {}).items():
            lines.append(
                f"- **{k}**: mean Δ={v['mean_delta_val']:+.4f} ± {v['std_delta_val']:.4f}"
            )
        lines += ["", "## C chem vs topo", ""]
        for k, v in out.get("C_chem", {}).items():
            lines.append(
                f"- **{k}**: s+size val={v['s+size']['valid_auc']:.4f} Δ={v['delta_val_vs_size']:+.4f}"
            )
        lines += ["", "## D ring_boost", ""]
        for k, v in out.get("D_ring", {}).items():
            lines.append(
                f"- **boost={k}**: s+size val={v['s+size']['valid_auc']:.4f} Δ={v['delta_val_vs_size']:+.4f}"
            )
    if "A_node_gate" in out and "results" in out["A_node_gate"]:
        lines += ["", "## A node gate dual (smoke)", ""]
        for k, v in out["A_node_gate"]["results"].items():
            lines.append(
                f"- **{k}**: val={v['best_val']:.4f} test@best={v['test_at_best_val']:.4f}"
            )
    lines += ["", f"Elapsed: {out['elapsed_sec']}s", ""]
    mpath = out_dir / f"NEXT_ROUND_n{args.max_graphs}_SUMMARY.md"
    mpath.write_text("\n".join(lines), encoding="utf-8")
    log(f"Wrote {jpath}")
    log(f"Wrote {mpath}")


if __name__ == "__main__":
    main()
