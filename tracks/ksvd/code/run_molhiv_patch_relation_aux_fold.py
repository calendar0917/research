"""Joint patch classification and relation-identification auxiliary task."""

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
from .graph_level import GraphLevelConfig, bundle_to_Y, learn_shared_D_graph_level, sparse_code_patch_matrix, sample_patches_graph_level, _chem_hist_for_patch


def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s)


def relations(g,patches):
    p=len(patches); A=np.zeros((3,p,p),dtype=np.float64)
    for i in range(p):
        for j in range(i+1,p):
            si,sj=patches[i],patches[j]; ov=len(si&sj); cross=sum(1 for u in si for v in g.neighbors(u) if v in sj and u<v)
            if not ov and not cross: continue
            A[0,i,j]=A[0,j,i]=ov/max(1,min(len(si),len(sj)))+cross/max(1,len(si)+len(sj)); A[1,i,j]=A[1,j,i]=ov/max(1,min(len(si),len(sj))); A[2,i,j]=A[2,j,i]=cross/max(1,len(si)+len(sj))
    for r in range(3): A[r]/=np.maximum(A[r].sum(1,keepdims=True),1e-12)
    return A


def encode(g,D,cfg,nf,ef,shuffle_seed):
    bundle,_=sample_patches_graph_level(g,cfg,seed=cfg.seed); Y,_=bundle_to_Y(g,bundle,cfg.max_nodes,cfg.order_mode,patch_feat=cfg.patch_feat,node_feat=nf,edge_feat=ef); _,X=sparse_code_patch_matrix(Y,D,cfg); X=np.abs(X.T); patches=bundle.node_sets[:X.shape[0]]; chem=np.stack([_chem_hist_for_patch(S,nf,ef,g) for S in patches]); A=relations(g,patches); p=np.random.default_rng(shuffle_seed).permutation(X.shape[0]); Ash=A[:,p][:,:,p] if X.shape[0]>1 else A.copy(); return np.concatenate([X,chem],1),A,Ash


class DS(Dataset):
    def __init__(self,rows): self.rows=rows
    def __len__(self): return len(self.rows)
    def __getitem__(self,i): return self.rows[i]


def collate(batch):
    B=len(batch); P=max(q[0].shape[0] for q in batch); K=batch[0][0].shape[1]; x=torch.zeros(B,P,K); a=torch.zeros(B,3,P,P); m=torch.zeros(B,P); y=torch.zeros(B)
    for i,(xx,aa,yy) in enumerate(batch): p=xx.shape[0]; x[i,:p]=torch.tensor(xx,dtype=torch.float32); a[i,:,:p,:p]=torch.tensor(aa,dtype=torch.float32); m[i,:p]=1.; y[i]=yy
    return x,a,m,y


class Model(nn.Module):
    def __init__(self,k,h):
        super().__init__(); self.enc=nn.Linear(k,h); self.msg=nn.ModuleList([nn.Linear(h,h) for _ in range(3)]); self.rel=nn.ModuleList([nn.Linear(h,h,bias=False) for _ in range(3)]); self.gate=nn.ModuleList([nn.Linear(2*h,1) for _ in range(3)]); self.cls=nn.Linear(h,1)
    def forward(self,x,a,m,use_relation=True):
        h=F.relu(self.enc(x)); base=h
        if use_relation:
            for r in range(3):
                msg=torch.bmm(a[:,r],h); g=torch.sigmoid(self.gate[r](torch.cat([h,msg],-1))); h=F.relu(h+g*self.msg[r](msg))
        pooled=(h*m.unsqueeze(-1)).sum(1)/m.sum(1,keepdim=True).clamp_min(1.); return self.cls(pooled).view(-1),base
    def relation_loss(self,h,a,m):
        losses=[]
        for r in range(3):
            n=h.shape[1]; idx=torch.triu_indices(n,n,offset=1,device=h.device); valid=(m[:,idx[0]]*m[:,idx[1]])>0
            hi=h[:,idx[0]]; hj=h[:,idx[1]]; scores=(self.rel[r](hi)*hj).sum(-1); labels=(a[:,r,idx[0],idx[1]]>0).float(); valid=valid & ((labels.sum(1)>0)&((1-labels).sum(1)>0)).unsqueeze(1)
            if not valid.any(): continue
            # Per-graph balanced weighting keeps the dense non-edge class from dominating.
            l=[]
            for b in range(h.shape[0]):
                mask=valid[b]
                if not mask.any(): continue
                yy=labels[b,mask]; ss=scores[b,mask]; pos=yy.sum(); neg=yy.numel()-pos
                if pos<=0 or neg<=0: continue
                l.append(F.binary_cross_entropy_with_logits(ss,yy,pos_weight=(neg/pos).clamp(1,20)))
            if l: losses.append(torch.stack(l).mean())
        return torch.stack(losses).mean() if losses else h.sum()*0


def auc(model,loader,device,use_relation):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,a,m,y in loader: o,_=model(x.to(device),a.to(device),m.to(device),use_relation); ys.append(y.numpy()); ps.append(torch.sigmoid(o).cpu().numpy())
    return float(roc_auc_score(np.concatenate(ys),np.concatenate(ps)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--fold-cache',default='tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz'); ap.add_argument('--fold',type=int,default=0); ap.add_argument('--epochs',type=int,default=25); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--hidden',type=int,default=64); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--aux-weight',type=float,default=0.5); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--device',default='cpu'); ap.add_argument('--output',required=True); args=ap.parse_args()
    seed_all(args.seed); device=torch.device(args.device); t0=time.time(); b=load_molhiv(with_features=True); assert b.node_feats is not None and b.edge_feats is not None
    z=np.load(args.fold_cache); tr=np.asarray(z[f'fold_{args.fold}_train_indices'],dtype=np.int64); va=np.asarray(z[f'fold_{args.fold}_valid_indices'],dtype=np.int64); all_idx=np.concatenate([tr,va]); ntr=len(tr); cfg=GraphLevelConfig(n_atoms=8,T=2,T_min=1,ksvd_iter=4,seed=0,max_train_patches=4000,max_patches_per_graph=8,patch_feat='wl_chem_ring',normalize_patches=True,readout_mode='pool',pool='max'); D,_=learn_shared_D_graph_level(b.graphs,tr,cfg,node_feats=b.node_feats,edge_feats=b.edge_feats)
    raw=[]
    for gi in all_idx:
        i=int(gi); raw.append(encode(b.graphs[i],D,cfg,b.node_feats[i],b.edge_feats[i],7919+i))
    scaler=StandardScaler().fit(np.vstack([r[0] for r in raw[:ntr]])); rows={'correct':[],'shuffled':[]}
    for j,(x,a,ash) in enumerate(raw): x=scaler.transform(x); y=float(b.y[int(all_idx[j])]); rows['correct'].append((x,a,y)); rows['shuffled'].append((x,ash,y))
    out={'protocol_id':'molhiv-patch-relation-aux-fold-v1','config':vars(args),'fold':args.fold,'variants':{}}
    for name in ('none','correct','shuffled'):
        seed_all(args.seed); chosen=rows['correct'] if name!='shuffled' else rows['shuffled']; trr=chosen[:ntr]; var=chosen[ntr:]; tl=DataLoader(DS(trr),batch_size=args.batch_size,shuffle=True,collate_fn=collate); vl=DataLoader(DS(var),batch_size=args.batch_size,shuffle=False,collate_fn=collate); model=Model(trr[0][0].shape[1],args.hidden).to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr)
        for _ in range(args.epochs):
            model.train()
            for x,a,m,y in tl:
                o,h=model(x.to(device),a.to(device),m.to(device),name!='none'); loss=F.binary_cross_entropy_with_logits(o,y.to(device));
                if name!='none': loss=loss+args.aux_weight*model.relation_loss(h,a.to(device),m.to(device))
                opt.zero_grad(); loss.backward(); opt.step()
        out['variants'][name]={'train_auc':auc(model,DataLoader(DS(trr),batch_size=args.batch_size,collate_fn=collate),device,name!='none'),'valid_auc':auc(model,vl,device,name!='none')}; print(name,json.dumps(out['variants'][name]),flush=True)
    out['elapsed_sec']=time.time()-t0; p=Path(args.output); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(out,indent=2)); print(f'wrote {p}')
if __name__=='__main__': main()
