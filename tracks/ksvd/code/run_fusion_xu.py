"""
Fusion ablations under Xu GIN paper epoch rule (global e*).

Variants:
  gin_only     — attributes only
  concat       — x' = [x; s_v]
  residual     — after each GINConv: h = h + W s_v  (W bias-free)
  gate         — h = h + σ(g)·W s_v, g = Linear([h; s_v])

Node structure: shared D on train fold, per-node B0 |coef| (12-d), z-scored.

Protocol: single global epoch = argmax_e mean_fold Acc(e); report mean±std at e*.

Clear per-fold / per-epoch logging.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.node_struct import (  # noqa: E402
    NodeStructConfig,
    encode_dataset_nodes,
    learn_shared_D,
    zscore_node_features,
)
from code.pipeline import graph_from_edge_index  # noqa: E402

PROTOCOL = {
    "protocol_id": "xu-gin-fusion-v0",
    "epoch_select": "global e* = argmax mean_fold Acc(e)",
    "fold_seed": 42,
    "n_folds": 10,
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _mlp(in_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.BatchNorm1d(hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    )


class GINFusion(nn.Module):
    """
    mode:
      only     — no structure
      concat   — structure already in x; plain GIN
      residual — x is attr only; s in data.s; residual after each conv
      gate     — same, gated residual
    """

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        num_classes: int,
        num_layers: int = 4,
        dropout: float = 0.5,
        mode: str = "only",
        struct_dim: int = 0,
    ):
        super().__init__()
        self.mode = mode
        self.dropout = dropout
        self.struct_dim = struct_dim
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers):
            d_in = in_dim if i == 0 else hidden
            self.convs.append(GINConv(_mlp(d_in, hidden), train_eps=False))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.res_projs = nn.ModuleList()
        self.gates = nn.ModuleList()
        if mode in ("residual", "gate") and struct_dim > 0:
            for _ in range(num_layers):
                # bias-free: s=0 ⇒ pure GIN path
                self.res_projs.append(nn.Linear(struct_dim, hidden, bias=False))
                if mode == "gate":
                    self.gates.append(nn.Linear(hidden + struct_dim, 1))

        n_pred = num_layers + 1
        self.predictors = nn.ModuleList(
            [nn.Linear(in_dim, num_classes)] + [nn.Linear(hidden, num_classes) for _ in range(num_layers)]
        )
        assert len(self.predictors) == n_pred

    def forward(self, x, edge_index, batch, s=None):
        reps = [x]
        h = x
        for li, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h = F.relu(bn(conv(h, edge_index)))
            if self.mode in ("residual", "gate") and s is not None and len(self.res_projs) > li:
                delta = self.res_projs[li](s)
                if self.mode == "gate":
                    g = torch.sigmoid(self.gates[li](torch.cat([h, s], dim=-1)))
                    h = h + g * delta
                else:
                    h = h + delta
            reps.append(h)
        score = 0
        for pred, rep in zip(self.predictors, reps):
            logits = pred(global_add_pool(rep, batch))
            logits = F.dropout(logits, p=self.dropout, training=self.training)
            score = score + logits
        return score


@torch.no_grad()
def accuracy(model, loader, device, use_s: bool) -> float:
    model.eval()
    correct = total = 0
    for data in loader:
        data = data.to(device)
        s = data.s if use_s and hasattr(data, "s") else None
        pred = model(data.x, data.edge_index, data.batch, s).argmax(-1)
        correct += int((pred == data.y.view(-1)).sum())
        total += data.num_graphs
    return correct / max(total, 1)


def load_raw(name: str, root: Path):
    from torch_geometric.datasets import TUDataset

    ds = TUDataset(root=str(root), name=name)
    graphs, raw, ys = [], [], []
    for data in ds:
        n = int(data.num_nodes)
        graphs.append(graph_from_edge_index(n, data.edge_index.cpu().numpy()))
        x = data.x.float() if data.x is not None else torch.ones((n, 1))
        yi = int(data.y.view(-1)[0].item())
        ys.append(yi)
        raw.append(Data(x=x, edge_index=data.edge_index, y=torch.tensor([yi]), num_nodes=n))
    y = np.array(ys, dtype=np.int64)
    uniq = sorted(set(y.tolist()))
    remap = {u: i for i, u in enumerate(uniq)}
    y = np.array([remap[int(v)] for v in y], dtype=np.int64)
    for i, d in enumerate(raw):
        d.y = torch.tensor([int(y[i])])
    return graphs, raw, y, len(uniq)


def attach_struct_fields(raw, structs, mode: str):
    """
    only: ignore structs
    concat: x = [x; s]
    residual/gate: keep x attr; store s on data.s
    """
    out = []
    if mode == "only" or structs is None:
        for d in raw:
            out.append(Data(x=d.x, edge_index=d.edge_index, y=d.y, num_nodes=d.num_nodes))
        return out
    for d, s in zip(raw, structs):
        st = torch.tensor(s, dtype=torch.float32)
        if mode == "concat":
            x = torch.cat([d.x, st], dim=-1)
            out.append(Data(x=x, edge_index=d.edge_index, y=d.y, num_nodes=d.num_nodes))
        elif mode in ("residual", "gate"):
            out.append(
                Data(x=d.x, edge_index=d.edge_index, y=d.y, num_nodes=d.num_nodes, s=st)
            )
        else:
            out.append(Data(x=d.x, edge_index=d.edge_index, y=d.y, num_nodes=d.num_nodes))
    return out


def run_variant(
    variant: str,
    graphs,
    raw,
    y,
    num_classes: int,
    epochs: int,
    hidden: int,
    batch_size: int,
    dropout: float,
    lr: float,
    device: torch.device,
    log_every: int,
    patch_mode: str = "B0",
    p: float = 1.0,
    q: float = 1.0,
    walk_length: int = 6,
    max_nodes: int = 8,
    num_walks: int = 1,
    pool: str = "mean",
) -> dict:
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=PROTOCOL["fold_seed"])
    folds = list(skf.split(np.zeros(len(y)), y))
    mode = {
        "gin_only": "only",
        "concat": "concat",
        "residual": "residual",
        "gate": "gate",
    }[variant]

    log(f"\n{'='*60}")
    log(f"VARIANT: {variant}  (fusion mode={mode})")
    log(
        f"PATCH: mode={patch_mode}  p={p} q={q} L={walk_length} m={max_nodes}  "
        f"r={num_walks} pool={pool}  (node2vec α: return=1/p, in-out=1/q)"
    )
    log(f"{'='*60}")

    fold_data = []
    for fold, (tr, te) in enumerate(folds):
        log(f"\n[prep] fold {fold}/9  train={len(tr)} test={len(te)}")
        if mode == "only":
            data_all = attach_struct_fields(raw, None, "only")
            struct_dim = 0
            in_dim = int(raw[0].x.size(-1))
            recon = None
            log(f"  gin_only in_dim={in_dim}")
        else:
            ncfg = NodeStructConfig(
                mode=patch_mode,
                max_nodes=max_nodes,
                walk_length=walk_length,
                num_walks=num_walks,
                pool=pool,
                p=p,
                q=q,
                no_backtrack=True,
                n_atoms=12,
                T=3,
                T_min=2,
                ksvd_iter=5,
                seed=fold,
                feat="abs_coef",
            )
            t_d = time.time()
            D, meta = learn_shared_D(graphs, tr, ncfg)
            log(
                f"  shared D: patches={int(meta.get('n_train_patches', 0))} "
                f"(raw={int(meta.get('n_patches_raw', 0))})  "
                f"recon_train={meta.get('recon_rel', 0):.4f}  "
                f"col_recon={meta.get('recon_col_mean', 0):.4f}±{meta.get('recon_col_std', 0):.4f}  "
                f"({time.time()-t_d:.1f}s)"
            )
            structs, ed = encode_dataset_nodes(graphs, D, ncfg)
            structs = zscore_node_features(structs, tr)
            data_all = attach_struct_fields(raw, structs, mode)
            struct_dim = int(structs[0].shape[1])
            in_dim = int(data_all[0].x.size(-1))
            recon = meta.get("recon_rel")
            log(
                f"  node struct dim={struct_dim}  gin in_dim={in_dim}  "
                f"r={ncfg.num_walks} pool={ncfg.pool}  "
                f"encode |S|={ed.get('encode_mean_|S|', 0):.2f}  "
                f"encode patch_recon={ed.get('encode_patch_recon_mean', 0):.4f}"
                f"±{ed.get('encode_patch_recon_std', 0):.4f}"
            )
        fold_data.append((tr, te, data_all, in_dim, struct_dim, recon))

    acc_mat = np.zeros((10, epochs), dtype=np.float64)
    use_s = mode in ("residual", "gate")
    t0 = time.time()

    for fold, (tr, te, data_all, in_dim, struct_dim, recon) in enumerate(fold_data):
        log(f"\n--- train fold {fold}/9 ---")
        torch.manual_seed(1000 + fold)
        model = GINFusion(
            in_dim=in_dim,
            hidden=hidden,
            num_classes=num_classes,
            num_layers=4,
            dropout=dropout,
            mode=mode if mode != "concat" else "concat",
            struct_dim=struct_dim,
        ).to(device)
        # concat uses plain GIN path (mode only residual/gate use s)
        if mode == "concat":
            model.mode = "only"
            use_s_fold = False
        else:
            use_s_fold = use_s

        opt = torch.optim.Adam(model.parameters(), lr=lr)
        tr_loader = DataLoader([data_all[i] for i in tr], batch_size=batch_size, shuffle=True)
        te_loader = DataLoader([data_all[i] for i in te], batch_size=64, shuffle=False)

        for ep in range(epochs):
            if ep > 0 and ep % 50 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                log(f"  fold {fold} ep {ep+1}: lr decay → {opt.param_groups[0]['lr']:.5f}")
            model.train()
            loss_sum, n_batch = 0.0, 0
            for data in tr_loader:
                data = data.to(device)
                s = data.s if use_s_fold and hasattr(data, "s") else None
                opt.zero_grad()
                out = model(data.x, data.edge_index, data.batch, s)
                loss = F.cross_entropy(out, data.y.view(-1))
                loss.backward()
                opt.step()
                loss_sum += float(loss.item())
                n_batch += 1
            acc = accuracy(model, te_loader, device, use_s_fold)
            acc_mat[fold, ep] = acc
            if (ep + 1) % log_every == 0 or ep == 0 or ep == epochs - 1:
                log(
                    f"  fold {fold} | ep {ep+1:3d}/{epochs} | "
                    f"loss {loss_sum/max(n_batch,1):.4f} | held-out Acc {acc*100:.1f}%"
                )

        fmax = float(acc_mat[fold].max())
        fmax_ep = int(acc_mat[fold].argmax()) + 1
        log(
            f"  fold {fold} done | last {acc_mat[fold,-1]*100:.1f}% | "
            f"fold-max {fmax*100:.1f}% @ep{fmax_ep}"
        )

    mean_curve = acc_mat.mean(axis=0)
    e_star = int(np.argmax(mean_curve))
    fold_at = acc_mat[:, e_star]
    log(f"\n[{variant}] global e*={e_star+1}  mean Acc={fold_at.mean()*100:.2f}% ± {fold_at.std()*100:.2f}%")
    for f in range(10):
        log(f"  fold {f} @e*: {fold_at[f]*100:.1f}%")

    return {
        "acc_mean": float(fold_at.mean()),
        "acc_std": float(fold_at.std()),
        "acc_percent_mean": float(fold_at.mean() * 100),
        "acc_percent_std": float(fold_at.std() * 100),
        "epoch_star": e_star + 1,
        "fold_scores_at_star": fold_at.tolist(),
        "per_fold_max_mean": float(acc_mat.max(axis=1).mean()),
        "seconds": time.time() - t0,
        "in_dim": int(fold_data[0][3]),
        "struct_dim": int(fold_data[0][4]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MUTAG")
    ap.add_argument("--epochs", type=int, default=350)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--log_every", type=int, default=25, help="print every N epochs within fold")
    ap.add_argument("--patch", default="B0", choices=["B0", "RW"], help="node patch: 1-hop or random walk")
    ap.add_argument("--p", type=float, default=1.0, help="node2vec return param (high=less backtrack)")
    ap.add_argument("--q", type=float, default=1.0, help="node2vec in-out (q>1 local/BFS-like, q<1 DFS-like)")
    ap.add_argument("--walk_length", type=int, default=6)
    ap.add_argument("--max_nodes", type=int, default=8, help="cap |S| for patch")
    ap.add_argument("--num_walks", type=int, default=1, help="r walks per node (RW mode)")
    ap.add_argument("--pool", default="mean", choices=["mean", "max", "mean_max"])
    ap.add_argument(
        "--variants",
        default="auto",
        help="comma list: gin_only,concat,residual,gate or auto",
    )
    args = ap.parse_args()
    device = torch.device(args.device)

    if args.variants == "auto":
        # RW run: skip gin_only (reuse prior baseline)
        variants = ["concat", "gate"] if args.patch == "RW" else [
            "gin_only",
            "concat",
            "residual",
            "gate",
        ]
    else:
        variants = [v.strip() for v in args.variants.split(",") if v.strip()]

    log(f"device={device} dataset={args.dataset} epochs={args.epochs} hidden={args.hidden}")
    log(f"protocol={PROTOCOL['protocol_id']}  {PROTOCOL['epoch_select']}")
    log(
        f"patch={args.patch} p={args.p} q={args.q} L={args.walk_length} m={args.max_nodes}  "
        f"r={args.num_walks} pool={args.pool}  variants={variants}"
    )
    log(
        "recon note: train recon is for THIS patch family only; "
        "do not rank B0 vs RW by recon. Downstream Acc is primary."
    )

    root = _TRACK.parents[1] / "data" / "TUD"
    graphs, raw, y, num_classes = load_raw(args.dataset, root)
    log(f"loaded {len(y)} graphs, {num_classes} classes, attr_dim={raw[0].x.size(-1)}")

    results = {
        "protocol": PROTOCOL,
        "dataset": args.dataset,
        "hps": vars(args),
        "patch": {
            "mode": args.patch,
            "p": args.p,
            "q": args.q,
            "walk_length": args.walk_length,
            "max_nodes": args.max_nodes,
            "num_walks": args.num_walks,
            "pool": args.pool,
            "how_selected": (
                "B0: S={v}∪N(v). RW: r node2vec walks from v; each → induced G[S]; "
                "x_i=OMP(D,y_i); s_v=pool_i(x_i) (mean/max/mean_max)."
            ),
            "p_q_selection": (
                "Not grid-searched on MUTAG. Default multi-walk run uses p=0.5,q=2 (local bias)."
            ),
            "recon_note": "recon comparable only within same patch family; not B0 vs RW ranking.",
        },
        "variants": {},
        "reference_gin_only_xu": {
            "acc_percent_mean": 89.4,
            "acc_percent_std": 5.8,
            "note": "from prior xu-gin-fusion B0 run / gin_xu_protocol",
        },
    }

    for variant in variants:
        results["variants"][variant] = run_variant(
            variant,
            graphs,
            raw,
            y,
            num_classes,
            args.epochs,
            args.hidden,
            args.batch_size,
            args.dropout,
            args.lr,
            device,
            args.log_every,
            patch_mode=args.patch,
            p=args.p,
            q=args.q,
            walk_length=args.walk_length,
            max_nodes=args.max_nodes,
            num_walks=args.num_walks,
            pool=args.pool,
        )

    tag = f"fusion_xu_{args.patch.lower()}"
    path = _TRACK / "results" / f"{tag}.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    sm = _TRACK / "results" / f"{tag.upper()}_SUMMARY.md"
    sm.write_text(_render(results), encoding="utf-8")
    log(f"\nwrote {path}")
    log(f"wrote {sm}")
    log("\n===== FINAL =====")
    if "gin_only" not in results["variants"]:
        log(
            f"  {'gin_only':10s}  {results['reference_gin_only_xu']['acc_percent_mean']:5.1f} ± "
            f"{results['reference_gin_only_xu']['acc_percent_std']:.1f}%  (reference, not re-run)"
        )
    for k, v in results["variants"].items():
        log(
            f"  {k:10s}  {v['acc_percent_mean']:5.1f} ± {v['acc_percent_std']:.1f}%  "
            f"@e*{v['epoch_star']}"
        )
    return 0


def _render(r: dict) -> str:
    patch = r.get("patch", {})
    lines = [
        "# Fusion ablations (Xu global epoch)",
        "",
        f"Dataset: **{r['dataset']}** · `{r['protocol']['protocol_id']}`",
        "",
        f"Patch: `{patch}`",
        "",
        "| variant | Acc % @ e* | e* | per-fold-max mean % |",
        "|---------|------------|----|---------------------|",
    ]
    ref = r.get("reference_gin_only_xu")
    if ref and "gin_only" not in r["variants"]:
        lines.append(
            f"| gin_only (ref) | {ref['acc_percent_mean']:.1f} ± {ref['acc_percent_std']:.1f} | — | — |"
        )
    for k, v in r["variants"].items():
        lines.append(
            f"| {k} | {v['acc_percent_mean']:.1f} ± {v['acc_percent_std']:.1f} | "
            f"{v['epoch_star']} | {v['per_fold_max_mean']*100:.1f} |"
        )
    lines += [
        "",
        "## Fusion definitions",
        "",
        "- **gin_only**: attributes only",
        "- **concat**: `x' = [x; s_v]`",
        "- **residual**: after each GINConv, `h = h + W s_v` (W bias-free)",
        "- **gate**: `h = h + σ(g) W s_v`, `g = Linear([h; s_v])`",
        "",
        "## How subgraph is chosen",
        "",
        patch.get("how_selected", ""),
        "",
        "## p, q selection",
        "",
        patch.get("p_q_selection", ""),
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
