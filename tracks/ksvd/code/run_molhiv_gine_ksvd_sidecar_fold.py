"""Fair GINE baseline with frozen KSVD patch sidecars on one internal fold."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig, learn_shared_D_graph_level
from .run_molhiv_typed_relation_folds import _encode_one
from .run_molhiv_channelwise_relation_folds import _channelwise


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


class Model(nn.Module):
    def __init__(self, hidden: int, layers: int, aux_dim: int, dropout: float):
        super().__init__()
        self.atom_encoder = AtomEncoder(hidden)
        self.bond_encoder = BondEncoder(hidden)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, train_eps=True))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.jk_gates = nn.Parameter(torch.zeros(max(0, layers - 1)))
        self.dropout = dropout
        self.head = nn.Linear(hidden + aux_dim, 1)

    def forward(self, d):
        x = self.atom_encoder(d.x)
        edge = self.bond_encoder(d.edge_attr)
        states = []
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(bn(conv(x, d.edge_index, edge)))
            x = F.dropout(x, p=self.dropout, training=self.training)
            states.append(global_mean_pool(x, d.batch))
        h = states[-1]
        for gate, earlier in zip(self.jk_gates, states[:-1]):
            h = h + torch.tanh(gate) * earlier
        if hasattr(d, "aux"):
            h = torch.cat([h, d.aux], dim=1)
        return self.head(h).view(-1)


def auc(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for b in loader:
            b=b.to(device); logits=model(b); ys.append(b.y.view(-1).cpu().numpy()); ps.append(torch.sigmoid(logits).cpu().numpy())
    y=np.concatenate(ys); p=np.concatenate(ps)
    return float(roc_auc_score(y,p))


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument("--fold-cache",default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--fold",type=int,default=0)
    ap.add_argument("--epochs",type=int,default=20)
    ap.add_argument("--batch-size",type=int,default=128)
    ap.add_argument("--hidden",type=int,default=64)
    ap.add_argument("--layers",type=int,default=3)
    ap.add_argument("--dropout",type=float,default=0.0)
    ap.add_argument("--lr",type=float,default=1e-3)
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--output",required=True)
    args=ap.parse_args(); seed_all(args.seed); device=torch.device(args.device)
    t0=time.time()
    bundle=load_molhiv(with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None: raise RuntimeError("features missing")
    z=np.load(args.fold_cache); tr=np.asarray(z[f"fold_{args.fold}_train_indices"],dtype=np.int64); va=np.asarray(z[f"fold_{args.fold}_valid_indices"],dtype=np.int64)
    cfg=GraphLevelConfig(n_atoms=8,T=2,T_min=1,ksvd_iter=4,seed=0,max_train_patches=4000,max_patches_per_graph=8,patch_feat="wl_chem_ring",normalize_patches=True,readout_mode="pool",pool="max")
    D,_=learn_shared_D_graph_level(bundle.graphs,tr,cfg,node_feats=bundle.node_feats,edge_feats=bundle.edge_feats)
    all_idx=np.concatenate([tr,va]); bags=[]; rels=[]; shufs=[]
    for gi in all_idx:
        i=int(gi); bag,rel,shuf=_encode_one(bundle.graphs[i],D,cfg,bundle.node_feats[i],bundle.edge_feats[i],shuffle_seed=7919+i)
        bags.append(bag); rels.append(_channelwise(rel,8,"row")); shufs.append(_channelwise(shuf,8,"row"))
    bags=np.stack(bags); rels=np.stack(rels); shufs=np.stack(shufs); ntr=len(tr)
    scaler_b=StandardScaler().fit(bags[:ntr]); scaler_r=StandardScaler().fit(rels[:ntr]); scaler_s=StandardScaler().fit(shufs[:ntr])
    bags=scaler_b.transform(bags); rels=scaler_r.transform(rels); shufs=scaler_s.transform(shufs)
    sidecars={"none":None,"bag":bags,"relation":rels,"bag_relation":np.concatenate([bags,rels],1),"bag_shuffled_relation":np.concatenate([bags,shufs],1)}
    data={}
    for gi in all_idx:
        i=int(gi); g=bundle.graphs[i]; src=[]; dst=[]; attrs=[]
        for u,v in sorted(g.edges()):
            a=np.asarray(bundle.edge_feats[i][(u,v)],dtype=np.int64); src += [u,v]; dst += [v,u]; attrs += [a,a]
        ei=torch.tensor([src,dst],dtype=torch.long) if src else torch.empty((2,0),dtype=torch.long)
        ea=torch.tensor(np.stack(attrs),dtype=torch.long) if attrs else torch.empty((0,3),dtype=torch.long)
        data[i]=Data(x=torch.tensor(bundle.node_feats[i],dtype=torch.long),edge_index=ei,edge_attr=ea,y=torch.tensor([float(bundle.y[i])],dtype=torch.float32),graph_index=torch.tensor([i]))
    results={"protocol_id":"molhiv-gine-ksvd-sidecar-internal-fold-v1","config":vars(args),"fold":args.fold,"variants":{}}
    for name,side in sidecars.items():
        seed_all(args.seed); aux_dim=0 if side is None else side.shape[1]
        items=[]
        for j,gi in enumerate(all_idx):
            d=data[int(gi)]
            if side is not None: d.aux=torch.tensor(side[j:j+1],dtype=torch.float32)
            items.append(d)
        tr_items=items[:ntr]; va_items=items[ntr:]
        train_loader=DataLoader(tr_items,batch_size=args.batch_size,shuffle=True)
        valid_loader=DataLoader(va_items,batch_size=args.batch_size,shuffle=False)
        model=Model(args.hidden,args.layers,aux_dim,args.dropout).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        for _ in range(args.epochs):
            model.train()
            for b in train_loader:
                b=b.to(device); out=model(b); y=b.y.view(-1); loss=F.binary_cross_entropy_with_logits(out,y); opt.zero_grad(); loss.backward(); opt.step()
        results["variants"][name]={"train_auc":auc(model,DataLoader(tr_items,batch_size=args.batch_size),device),"valid_auc":auc(model,valid_loader,device),"aux_dim":aux_dim}
        print(name,json.dumps(results["variants"][name]),flush=True)
    results["elapsed_sec"]=time.time()-t0; out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(results,indent=2)); print(f"wrote {out}")


if __name__=="__main__": main()
