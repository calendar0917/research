"""Frozen full-data evaluation of a stable real-prototype MIL member.

The raw-patch PCA metric is fit on official train only.  A prototype-seed-fixed,
graph-balanced bank supplies exact observed train patches; downstream training
uses a fixed model seed and 30 epochs.  Official valid/test are evaluated once
after training so separately launched members can form the predeclared
three-prototype-bank probability ensemble.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))
from code.data_molhiv import load_molhiv


def sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


def seed_all(seed: int, torch: Any) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def rows_for(indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i)+1]), dtype=np.int64)
        for i in indices
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--latent-cache', required=True)
    ap.add_argument('--token-cache', required=True)
    ap.add_argument('--max-graphs', type=int, default=8000)
    ap.add_argument('--data-seed', type=int, default=0)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--prototype-seed', type=int, required=True)
    ap.add_argument('--candidate-bank-size', type=int, default=256)
    ap.add_argument('--n-prototypes', type=int, default=32)
    ap.add_argument('--sparsity', type=int, default=3)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--hidden', type=int, default=64)
    ap.add_argument('--dropout', type=float, default=0.15)
    ap.add_argument('--temperature', type=float, default=0.25)
    ap.add_argument('--task-lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--grad-clip', type=float, default=5.0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--num-workers', type=int, default=0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if min(args.candidate_bank_size,args.n_prototypes,args.sparsity,args.epochs,args.batch_size) <= 0:
        raise ValueError('positive size required')

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.utils.features import get_atom_feature_dims
        from torch.utils.data import DataLoader, Dataset
        from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
        from torch_geometric.utils import softmax as graph_softmax
    except ImportError as exc:
        raise RuntimeError(f'missing dependencies: {exc}') from exc

    t0=time.time()
    bundle=load_molhiv(max_graphs=None if args.max_graphs<=0 else args.max_graphs,
                       seed=args.data_seed,with_features=True)
    if bundle.node_feats is None: raise AssertionError('atom features missing')
    labels=np.asarray(bundle.y,dtype=np.float32)
    original=np.asarray(bundle.meta['original_indices'],dtype=np.int64)
    tr=np.asarray(bundle.split['train'],dtype=np.int64)
    va=np.asarray(bundle.split['valid'],dtype=np.int64)
    te=np.asarray(bundle.split['test'],dtype=np.int64)
    if any(np.intersect1d(a,b).size for a,b in ((tr,va),(tr,te),(va,te))):
        raise AssertionError('official splits overlap')

    with np.load(args.latent_cache,allow_pickle=False) as f:
        offsets=np.asarray(f['offsets'],dtype=np.int64)
        latents_np=np.asarray(f['latents'],dtype=np.float32)
        latent_meta={k:np.asarray(f[k],dtype=np.int64) for k in ('original_indices','train_indices','valid_indices','test_indices')}
    with np.load(args.token_cache,allow_pickle=False) as f:
        tok={k:np.asarray(f[k]) for k in f.files}
    checks=((latent_meta['original_indices'],original,'latent original'),
            (latent_meta['train_indices'],tr,'latent train'),
            (latent_meta['valid_indices'],va,'latent valid'),
            (latent_meta['test_indices'],te,'latent test'),
            (np.asarray(tok['offsets'],dtype=np.int64),offsets,'token offsets'),
            (np.asarray(tok['original_indices'],dtype=np.int64),original,'token original'),
            (np.asarray(tok['train_indices'],dtype=np.int64),tr,'token train'))
    for a,b,name in checks:
        if not np.array_equal(a,b): raise ValueError(f'misaligned {name}')
    if latents_np.shape[0] != int(offsets[-1]): raise ValueError('latent rows mismatch')
    all_rows=rows_for(np.concatenate([tr,va,te]),offsets)
    norms=np.linalg.norm(latents_np[all_rows],axis=1)
    if float(np.mean(norms>0.99)) < .999: raise ValueError('missing/nonunit terminal latents')

    token_meta_path=Path(args.token_cache).with_suffix('.json')
    token_meta=json.loads(token_meta_path.read_text())
    if int(token_meta.get('dictionary_fit_n_graphs',-1)) != len(tr):
        raise ValueError('dictionary was not fit on all and only official train graphs')
    fit_hash=sha256(tr)
    if token_meta.get('dictionary_fit_indices_sha256') != fit_hash:
        raise ValueError('dictionary fit index hash is not official train')

    latent_meta_path=Path(args.latent_cache).with_suffix('.json')
    latent_audit=json.loads(latent_meta_path.read_text())
    if not bool(latent_audit.get('official_valid_encoded',False)) or not bool(latent_audit.get('official_test_encoded',False)):
        raise ValueError('terminal latent cache does not contain official valid/test')
    latent_token_name=Path(latent_audit.get('config',{}).get('token_cache','')).name
    if latent_token_name != Path(args.token_cache).name:
        raise ValueError('latent cache was not derived from requested token cache')

    if args.candidate_bank_size > len(tr):
        raise ValueError('candidate bank exceeds official-train graph count')
    rng=np.random.default_rng(args.prototype_seed+910_003)
    candidate_source_positions=rng.choice(
        len(tr),size=args.candidate_bank_size,replace=False
    ).astype(np.int64)
    candidate_source_graphs=tr[candidate_source_positions]
    candidate_rows=np.asarray([
        rng.integers(int(offsets[int(i)]),int(offsets[int(i)+1]))
        for i in candidate_source_graphs
    ],dtype=np.int64)
    bank=normalize_rows(latents_np[candidate_rows])
    random_selection=rng.choice(
        args.candidate_bank_size,size=args.n_prototypes,replace=False
    ).astype(np.int64)
    prototypes={'stable_random_node_mil':bank[random_selection]}
    for name,p in prototypes.items():
        if p.shape != (args.n_prototypes,latents_np.shape[1]):
            raise ValueError(f'{name} prototype shape {p.shape}')

    atom_dims=[int(x) for x in get_atom_feature_dims()]
    latent_tensor=torch.from_numpy(latents_np)
    class IndexDataset(Dataset):
        def __init__(self,idx): self.idx=np.asarray(idx,dtype=np.int64)
        def __len__(self): return len(self.idx)
        def __getitem__(self,i): return int(self.idx[i])
    def collate(indices):
        rows=[]; centers=[]; gids=[]
        for bi,raw in enumerate(indices):
            i=int(raw); lo,hi=int(offsets[i]),int(offsets[i+1]); n=hi-lo
            rows.append(np.arange(lo,hi,dtype=np.int64))
            centers.append(np.asarray(bundle.node_feats[i],dtype=np.int64))
            gids.append(np.full(n,bi,dtype=np.int64))
        return {'rows':torch.from_numpy(np.concatenate(rows)),
                'center':torch.from_numpy(np.concatenate(centers)),
                'graph_batch':torch.from_numpy(np.concatenate(gids)),
                'labels':torch.from_numpy(labels[np.asarray(indices,dtype=np.int64)]),
                'indices':torch.tensor(indices,dtype=torch.long)}
    def loader(idx,shuffle,offset):
        gen=torch.Generator().manual_seed(args.seed+offset)
        return DataLoader(IndexDataset(idx),batch_size=args.batch_size,shuffle=shuffle,
                          generator=gen if shuffle else None,num_workers=args.num_workers,
                          collate_fn=collate)

    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention=nn.Sequential(nn.Linear(args.hidden,args.hidden//2),nn.Tanh(),nn.Linear(args.hidden//2,1))
            self.head=nn.Sequential(nn.Linear(3*args.hidden,args.hidden),nn.LayerNorm(args.hidden),nn.SiLU(),nn.Dropout(args.dropout),nn.Linear(args.hidden,1))
        def forward(self,h,batch):
            a=graph_softmax(self.attention(h).view(-1),batch)
            return self.head(torch.cat([global_add_pool(h*a[:,None],batch),global_mean_pool(h,batch),global_max_pool(h,batch)],1)).view(-1)
    class Model(nn.Module):
        def __init__(self,p):
            super().__init__(); p=torch.tensor(p,dtype=torch.float32); self.register_buffer('prototypes',p/p.norm(dim=1,keepdim=True).clamp_min(1e-12))
            self.center_embeddings=nn.ModuleList([nn.Embedding(d,args.hidden) for d in atom_dims])
            self.identity_embeddings=nn.Parameter(torch.empty(args.n_prototypes,args.hidden))
            self.affinity_embeddings=nn.Parameter(torch.empty(args.n_prototypes,args.hidden))
            self.node_mlp=nn.Sequential(nn.Linear(3*args.hidden+2,args.hidden),nn.LayerNorm(args.hidden),nn.SiLU(),nn.Dropout(args.dropout),nn.Linear(args.hidden,args.hidden),nn.SiLU())
            self.readout=Readout(); self.reset_parameters()
        def reset_parameters(self):
            for e in self.center_embeddings: nn.init.xavier_uniform_(e.weight)
            nn.init.xavier_uniform_(self.identity_embeddings); nn.init.xavier_uniform_(self.affinity_embeddings)
            for m in self.modules():
                if isinstance(m,nn.Linear): nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        def forward(self,z,center,batch):
            cosine=z@self.prototypes.T; positive=torch.relu(cosine)
            if args.sparsity < args.n_prototypes:
                values,indices=positive.topk(args.sparsity,dim=1); codes=torch.zeros_like(positive).scatter_(1,indices,values)
            else: codes=positive
            mass=codes.sum(1,keepdim=True); normalized=codes/mass.clamp_min(1e-6)
            sharp=torch.softmax(codes/args.temperature,dim=1)*(codes>0).float(); sharp=sharp/sharp.sum(1,keepdim=True).clamp_min(1e-6)
            center_h=0.0
            for field,e in enumerate(self.center_embeddings): center_h=center_h+e(center[:,field])
            h=self.node_mlp(torch.cat([center_h,normalized@self.identity_embeddings,sharp@self.affinity_embeddings,torch.log1p(mass),cosine.max(1,keepdim=True).values],1))
            return self.readout(h,batch),h

    device=torch.device(args.device)
    @torch.no_grad()
    def evaluate(model,idx,offset):
        model.eval(); logits=[]; ys=[]; order=[]
        for b in loader(idx,False,offset):
            z=latent_tensor[b['rows']].to(device); center=b['center'].to(device); gb=b['graph_batch'].to(device)
            out,_=model(z,center,gb); logits.append(out.cpu().numpy()); ys.append(b['labels'].numpy()); order.append(b['indices'].numpy())
        logit=np.concatenate(logits).astype(np.float64); y=np.concatenate(ys).astype(np.float64); indices=np.concatenate(order).astype(np.int64)
        prob=1/(1+np.exp(-np.clip(logit,-60,60)))
        return {'auc':float(roc_auc_score(y,prob)),'n_graphs':len(y),'n_positive':int(y.sum()),
                'indices':indices.tolist(),'y':y.astype(int).tolist(),'logits':logit.tolist(),'probabilities':prob.tolist(),'probability_sha256':sha256(prob)}

    results={}
    for ci,name in enumerate(('stable_random_node_mil',)):
        start=time.time(); seed_all(args.seed,torch); model=Model(prototypes[name]).to(device)
        opt=torch.optim.AdamW(model.parameters(),lr=args.task_lr,weight_decay=args.weight_decay)
        history=[]
        for epoch in range(1,args.epochs+1):
            model.train(); total=0.; seen=0
            for b in loader(tr,True,1000+ci*10000):
                z=latent_tensor[b['rows']].to(device); center=b['center'].to(device); gb=b['graph_batch'].to(device); y=b['labels'].to(device).float()
                out,_=model(z,center,gb); loss=F.binary_cross_entropy_with_logits(out,y)
                opt.zero_grad(set_to_none=True); loss.backward()
                if args.grad_clip>0: torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
                opt.step(); total+=float(loss.detach())*len(y); seen+=len(y)
            history.append({'epoch':epoch,'train_loss':total/max(seen,1)})
            if epoch==1 or epoch%5==0 or epoch==args.epochs: print(f'{name} epoch={epoch:02d} loss={history[-1]["train_loss"]:.5f}',flush=True)
        train_eval=evaluate(model,tr,2000+ci*10000)
        valid_eval=evaluate(model,va,3000+ci*10000)
        test_eval=evaluate(model,te,4000+ci*10000)
        results[name]={'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                       'prototype_sha256':sha256(prototypes[name]),'train':train_eval,'official_valid':valid_eval,'official_test':test_eval,
                       'history':history,'elapsed_sec':time.time()-start}
        print(f'{name} valid={valid_eval["auc"]:.6f} test={test_eval["auc"]:.6f}',flush=True)

    report={'protocol_id':'molhiv-stable-real-prototype-full-official-v1','date':'2026-07-28','seed':args.seed,
            'selection_policy':{'architecture_and_hyperparameters_frozen_before_official_test':True,'train_data':'official train only','epoch_policy':f'fixed {args.epochs} epochs','official_valid_evaluations_per_model':1,'official_test_evaluations_per_model':1,'post_test_tuning_permitted':False,'ensemble_policy':'predeclared arithmetic mean of prototype seeds 20260728/20260729/20260730 probabilities with model seed 0'},
            'config':vars(args),'n_train':len(tr),'n_valid':len(va),'n_test':len(te),'n_train_positive':int(labels[tr].sum()),'n_valid_positive':int(labels[va].sum()),'n_test_positive':int(labels[te].sum()),
            'official_train_sha256':fit_hash,'latent_cache_sha256':sha256(latents_np),'dictionary_sha256':sha256(np.asarray(tok['dictionary_ksvd'])),'candidate_source_positions':candidate_source_positions.tolist(),'candidate_source_graphs':candidate_source_graphs.tolist(),'candidate_rows':candidate_rows.tolist(),'random_selection':random_selection.tolist(),'random_source_graphs':candidate_source_graphs[random_selection].tolist(),'random_source_node_rows':candidate_rows[random_selection].tolist(),'results':results,'elapsed_sec':time.time()-t0,'output':args.output}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps({'output':str(out),'seed':args.seed,'auc':{n:{'valid':results[n]['official_valid']['auc'],'test':results[n]['official_test']['auc']} for n in results},'elapsed_sec':report['elapsed_sec']},indent=2),flush=True)

if __name__=='__main__': main()
