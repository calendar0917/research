"""One-layer message passing over chemistry-aware KSVD patch graphs."""

from __future__ import annotations

import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig, bundle_to_Y, learn_shared_D_graph_level, sparse_code_patch_matrix, sample_patches_graph_level


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def patch_relations(g, patches, node_feat, edge_feat):
    p=len(patches); A=np.zeros((3,p,p),dtype=np.float64)
    for i in range(p):
        for j in range(i+1,p):
            si,sj=patches[i],patches[j]; overlap=sorted(si&sj); cross=[]
            for u in si:
                for v in g.neighbors(u):
                    if v in sj and u<v: cross.append((u,v))
            if not overlap and not cross: continue
            geom=len(overlap)/max(1,min(len(si),len(sj))) + len(cross)/max(1,len(si)+len(sj))
            ov=len(overlap)/max(1,min(len(si),len(sj)))
            cb=len(cross)/max(1,len(si)+len(sj))
            A[0,i,j]=A[0,j,i]=geom; A[1,i,j]=A[1,j,i]=ov; A[2,i,j]=A[2,j,i]=cb
    for r in range(3):
        d=A[r].sum(1,keepdims=True); A[r]=A[r]/np.maximum(d,1e-12)
    return A


def encode_graph(g,D,cfg,node_feat,edge_feat,shuffle_seed=None):
    bundle,_=sample_patches_graph_level(g,cfg,seed=cfg.seed)
    Y,_=bundle_to_Y(g,bundle,cfg.max_nodes,cfg.order_mode,patch_feat=cfg.patch_feat,node_feat=node_feat,edge_feat=edge_feat)
    _,X=sparse_code_patch_matrix(Y,D,cfg); X=np.abs(X.T)
    patches=bundle.node_sets[:X.shape[0]]; A=patch_relations(g,patches,node_feat,edge_feat)
    Ash=A.copy()
    if shuffle_seed is not None and X.shape[0]>1:
        p=np.random.default_rng(shuffle_seed).permutation(X.shape[0]); Ash=Ash[:,p][:,:,p]
    return X,A,Ash


class PatchDataset(Dataset):
    def __init__(self, rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]


def collate(batch):
    B=len(batch); P=max(x[0].shape[0] for x in batch); K=batch[0][0].shape[1]; R=3
    xs=torch.zeros(B,P,K); As=torch.zeros(B,R,P,P); masks=torch.zeros(B,P); ys=torch.zeros(B)
    for i,(x,a,y) in enumerate(batch):
        p=x.shape[0]; xs[i,:p]=torch.tensor(x,dtype=torch.float32); As[i,:,:p,:p]=torch.tensor(a,dtype=torch.float32); masks[i,:p]=1.; ys[i]=float(y)
    return xs,As,masks,ys


class PatchRelModel(nn.Module):
    def __init__(self,k,h,relation=True):
        super().__init__(); self.relation=relation; self.self_lin=nn.Linear(k,h); self.rel_lins=nn.ModuleList([nn.Linear(k,h) for _ in range(3)]); self.head=nn.Linear(h,1)
    def forward(self,x,A,mask):
        h=F.relu(self.self_lin(x))
        if self.relation:
            for r in range(3): h=F.relu(h + torch.bmm(A[:,r],self.rel_lins[r](x)))
        pooled=(h*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True).clamp_min(1.)
        return self.head(pooled).view(-1)


def evaluate(model,loader,device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,a,m,y in loader:
            o=model(x.to(device),a.to(device),m.to(device)); ys.append(y.numpy()); ps.append(torch.sigmoid(o).cpu().numpy())
    return float(roc_auc_score(np.concatenate(ys),np.concatenate(ps)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold-cache',default='tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz'); ap.add_argument('--fold',type=int,default=0); ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--hidden',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu'); ap.add_argument('--output',required=True); args=ap.parse_args()
    seed_all(args.seed); device=torch.device(args.device); t0=time.time(); b=load_molhiv(with_features=True); assert b.node_feats is not None and b.edge_feats is not None
    z=np.load(args.fold_cache); tr=np.asarray(z[f'fold_{args.fold}_train_indices'],dtype=np.int64); va=np.asarray(z[f'fold_{args.fold}_valid_indices'],dtype=np.int64); cfg=GraphLevelConfig(n_atoms=8,T=2,T_min=1,ksvd_iter=4,seed=0,max_train_patches=4000,max_patches_per_graph=8,patch_feat='wl_chem_ring',normalize_patches=True,readout_mode='pool',pool='max')
    D,_=learn_shared_D_graph_level(b.graphs,tr,cfg,node_feats=b.node_feats,edge_feats=b.edge_feats)
    rows={k:[] for k in ('correct','shuffled')}
    for gi in np.concatenate([tr,va]):
        i=int(gi); x,a,ash=encode_graph(b.graphs[i],D,cfg,b.node_feats[i],b.edge_feats[i],shuffle_seed=7919+i); rows['correct'].append((x,a,float(b.y[i]))); rows['shuffled'].append((x,ash,float(b.y[i])))
    results={'protocol_id':'molhiv-patch-relation-message-fold-v1','config':vars(args),'fold':args.fold,'variants':{}}
    for relname in ('none','correct','shuffled'):
        seed_all(args.seed)
        chosen=rows['correct'] if relname!='shuffled' else rows['shuffled']; train_rows=chosen[:len(tr)]; valid_rows=chosen[len(tr):]
        # The no-relation control uses the same patch nodes but zeroes all edges.
        if relname=='none': train_rows=[(x,np.zeros_like(a),y) for x,a,y in train_rows]; valid_rows=[(x,np.zeros_like(a),y) for x,a,y in valid_rows]
        tl=DataLoader(PatchDataset(train_rows),batch_size=args.batch_size,shuffle=True,collate_fn=collate); vl=DataLoader(PatchDataset(valid_rows),batch_size=args.batch_size,shuffle=False,collate_fn=collate); model=PatchRelModel(8,args.hidden,relation=True).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        for _ in range(args.epochs):
            model.train()
            for x,a,m,y in tl:
                o=model(x.to(device),a.to(device),m.to(device)); loss=F.binary_cross_entropy_with_logits(o,y.to(device)); opt.zero_grad(); loss.backward(); opt.step()
        results['variants'][relname]={'train_auc':evaluate(model,DataLoader(PatchDataset(train_rows),batch_size=args.batch_size,collate_fn=collate),device),'valid_auc':evaluate(model,vl,device)}; print(relname,json.dumps(results['variants'][relname]),flush=True)
    results['elapsed_sec']=time.time()-t0; p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(results,indent=2)); print(f'wrote {p}')
if __name__=='__main__': main()
