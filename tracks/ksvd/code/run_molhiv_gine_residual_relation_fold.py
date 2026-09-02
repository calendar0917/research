"""GINE plus a zero-initialized, controlled KSVD relation residual."""

from __future__ import annotations

import argparse, json, random, time
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


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class ResidualGINE(nn.Module):
    def __init__(self, hidden, layers, aux_dim, dropout):
        super().__init__(); self.atom=AtomEncoder(hidden); self.bond=BondEncoder(hidden)
        self.convs=nn.ModuleList(); self.bns=nn.ModuleList()
        for _ in range(layers):
            mlp=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,hidden)); self.convs.append(GINEConv(mlp,train_eps=True)); self.bns.append(nn.BatchNorm1d(hidden))
        self.gates=nn.Parameter(torch.zeros(max(0,layers-1))); self.dropout=dropout
        self.main=nn.Linear(hidden,1); self.side=nn.Linear(aux_dim,1); self.side_gate=nn.Parameter(torch.zeros(1))
    def forward(self,d):
        x=self.atom(d.x); e=self.bond(d.edge_attr); states=[]
        for conv,bn in zip(self.convs,self.bns):
            x=F.relu(bn(conv(x,d.edge_index,e))); x=F.dropout(x,p=self.dropout,training=self.training); states.append(global_mean_pool(x,d.batch))
        h=states[-1]
        for gate,earlier in zip(self.gates,states[:-1]): h=h+torch.tanh(gate)*earlier
        return (self.main(h).view(-1) + torch.tanh(self.side_gate)*self.side(d.aux).view(-1))


def eval_auc(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for b in loader:
            b=b.to(device); ys.append(b.y.view(-1).cpu().numpy()); ps.append(torch.sigmoid(model(b)).cpu().numpy())
    return float(roc_auc_score(np.concatenate(ys),np.concatenate(ps)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold-cache',default='tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz'); ap.add_argument('--fold',type=int,default=0); ap.add_argument('--epochs',type=int,default=20); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--hidden',type=int,default=64); ap.add_argument('--layers',type=int,default=3); ap.add_argument('--dropout',type=float,default=0.0); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu'); ap.add_argument('--output',required=True); args=ap.parse_args()
    seed_all(args.seed); device=torch.device(args.device); t0=time.time()
    b=load_molhiv(with_features=True); assert b.node_feats is not None and b.edge_feats is not None
    z=np.load(args.fold_cache); tr=np.asarray(z[f'fold_{args.fold}_train_indices'],dtype=np.int64); va=np.asarray(z[f'fold_{args.fold}_valid_indices'],dtype=np.int64); all_idx=np.concatenate([tr,va]); ntr=len(tr)
    cfg=GraphLevelConfig(n_atoms=8,T=2,T_min=1,ksvd_iter=4,seed=0,max_train_patches=4000,max_patches_per_graph=8,patch_feat='wl_chem_ring',normalize_patches=True,readout_mode='pool',pool='max')
    D,_=learn_shared_D_graph_level(b.graphs,tr,cfg,node_feats=b.node_feats,edge_feats=b.edge_feats)
    bags=[]; rels=[]; shufs=[]
    for gi in all_idx:
        i=int(gi); bag,rel,shuf=_encode_one(b.graphs[i],D,cfg,b.node_feats[i],b.edge_feats[i],shuffle_seed=7919+i); bags.append(bag); rels.append(_channelwise(rel,8,'row')); shufs.append(_channelwise(shuf,8,'row'))
    bags=np.stack(bags); rels=np.stack(rels); shufs=np.stack(shufs); sb=StandardScaler().fit(bags[:ntr]); sr=StandardScaler().fit(rels[:ntr]); ss=StandardScaler().fit(shufs[:ntr]); bags=sb.transform(bags); rels=sr.transform(rels); shufs=ss.transform(shufs)
    sidecars={'relation':rels,'shuffled_relation':shufs}
    data=[]
    for j,gi in enumerate(all_idx):
        i=int(gi); g=b.graphs[i]; src=[]; dst=[]; attrs=[]
        for u,v in sorted(g.edges()):
            a=np.asarray(b.edge_feats[i][(u,v)],dtype=np.int64); src += [u,v]; dst += [v,u]; attrs += [a,a]
        ei=torch.tensor([src,dst],dtype=torch.long) if src else torch.empty((2,0),dtype=torch.long); ea=torch.tensor(np.stack(attrs),dtype=torch.long) if attrs else torch.empty((0,3),dtype=torch.long)
        data.append(Data(x=torch.tensor(b.node_feats[i],dtype=torch.long),edge_index=ei,edge_attr=ea,y=torch.tensor([float(b.y[i])],dtype=torch.float32)))
    out={'protocol_id':'molhiv-gine-ksvd-residual-relation-fold-v1','config':vars(args),'fold':args.fold,'variants':{}}
    for name,side in sidecars.items():
        seed_all(args.seed); items=[]
        for j,d in enumerate(data): d2=d.clone(); d2.aux=torch.tensor(side[j:j+1],dtype=torch.float32); items.append(d2)
        tr_loader=DataLoader(items[:ntr],batch_size=args.batch_size,shuffle=True); va_loader=DataLoader(items[ntr:],batch_size=args.batch_size,shuffle=False); model=ResidualGINE(args.hidden,args.layers,side.shape[1],args.dropout).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        for _ in range(args.epochs):
            model.train()
            for batch in tr_loader:
                batch=batch.to(device); loss=F.binary_cross_entropy_with_logits(model(batch),batch.y.view(-1)); opt.zero_grad(); loss.backward(); opt.step()
        out['variants'][name]={'train_auc':eval_auc(model,DataLoader(items[:ntr],batch_size=args.batch_size),device),'valid_auc':eval_auc(model,va_loader,device),'final_side_gate':float(torch.tanh(model.side_gate).detach().cpu().item())}; print(name,json.dumps(out['variants'][name]),flush=True)
    out['elapsed_sec']=time.time()-t0; p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)); print(f'wrote {p}')
if __name__=='__main__': main()
