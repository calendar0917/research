"""Terminal fixed-epoch 3-layer original-node GINE on official MolHIV splits."""
from __future__ import annotations
import argparse,hashlib,json,random,sys,time
from pathlib import Path
from typing import Any
import numpy as np
from sklearn.metrics import roc_auc_score
_TRACK=Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path: sys.path.insert(0,str(_TRACK))
from code.data_molhiv import load_molhiv

def sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def seed_all(seed,torch):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    try: torch.use_deterministic_algorithms(True)
    except Exception: pass

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-graphs',type=int,default=8000); ap.add_argument('--data-seed',type=int,default=0); ap.add_argument('--seed',type=int,default=0); ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--batch-size',type=int,default=128); ap.add_argument('--hidden',type=int,default=64); ap.add_argument('--layers',type=int,default=3); ap.add_argument('--dropout',type=float,default=0.0); ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--device',default='cpu'); ap.add_argument('--num-workers',type=int,default=0); ap.add_argument('--output',required=True); args=ap.parse_args()
    if min(args.epochs,args.batch_size,args.hidden,args.layers)<=0: raise ValueError('invalid config')
    try:
        import torch,torch.nn as nn,torch.nn.functional as F
        from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from torch_geometric.nn import GINEConv,global_mean_pool
    except ImportError as e: raise RuntimeError(f'missing dependencies: {e}') from e
    t0=time.time(); seed_all(args.seed,torch); device=torch.device(args.device)
    b=load_molhiv(max_graphs=None if args.max_graphs<=0 else args.max_graphs,seed=args.data_seed,with_features=True)
    if b.node_feats is None or b.edge_feats is None: raise AssertionError('features missing')
    labels=np.asarray(b.y,dtype=np.float32); tr=np.asarray(b.split['train'],dtype=np.int64); va=np.asarray(b.split['valid'],dtype=np.int64); te=np.asarray(b.split['test'],dtype=np.int64)
    if any(np.intersect1d(a,c).size for a,c in ((tr,va),(tr,te),(va,te))): raise AssertionError('split overlap')
    all_indices=np.concatenate([tr,va,te]); data={}
    for count,raw in enumerate(all_indices,1):
        i=int(raw); g=b.graphs[i]; src=[]; dst=[]; attrs=[]
        for u,v in sorted(g.edges()):
            a=np.asarray(b.edge_feats[i][(u,v)],dtype=np.int64); src += [u,v]; dst += [v,u]; attrs += [a,a]
        if src: edge_index=torch.tensor([src,dst],dtype=torch.long); edge_attr=torch.tensor(np.stack(attrs),dtype=torch.long)
        else: edge_index=torch.empty((2,0),dtype=torch.long); edge_attr=torch.empty((0,3),dtype=torch.long)
        data[i]=Data(x=torch.tensor(b.node_feats[i],dtype=torch.long),edge_index=edge_index,edge_attr=edge_attr,y=torch.tensor([labels[i]],dtype=torch.float32),graph_index=torch.tensor([i],dtype=torch.long))
        if count%1000==0 or count==len(all_indices): print(f'encoded graphs {count}/{len(all_indices)}',flush=True)
    def loader(idx,shuffle,offset):
        gen=torch.Generator().manual_seed(args.seed+9173+offset) if shuffle else None
        return DataLoader([data[int(i)] for i in idx],batch_size=args.batch_size,shuffle=shuffle,generator=gen,num_workers=args.num_workers)
    class Model(nn.Module):
        def __init__(self):
            super().__init__(); self.atom_encoder=AtomEncoder(args.hidden); self.bond_encoder=BondEncoder(args.hidden); self.convs=nn.ModuleList(); self.bns=nn.ModuleList()
            for _ in range(args.layers):
                mlp=nn.Sequential(nn.Linear(args.hidden,args.hidden),nn.ReLU(),nn.Linear(args.hidden,args.hidden)); self.convs.append(GINEConv(mlp,train_eps=True)); self.bns.append(nn.BatchNorm1d(args.hidden))
            self.jk_gates=nn.Parameter(torch.zeros(max(0,args.layers-1))); self.head=nn.Linear(args.hidden,1)
        def forward(self,d):
            x=self.atom_encoder(d.x); edge=self.bond_encoder(d.edge_attr); states=[]
            for conv,bn in zip(self.convs,self.bns):
                x=F.relu(bn(conv(x,d.edge_index,edge))); x=F.dropout(x,p=args.dropout,training=self.training); states.append(global_mean_pool(x,d.batch))
            h=states[-1]
            for gate,earlier in zip(self.jk_gates,states[:-1]): h=h+torch.tanh(gate)*earlier
            return self.head(h).view(-1)
    model=Model().to(device); opt=torch.optim.Adam(model.parameters(),lr=args.lr); history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); total=0.; seen=0
        for batch in loader(tr,True,0):
            batch=batch.to(device); out=model(batch); y=batch.y.view(-1).float(); loss=F.binary_cross_entropy_with_logits(out,y)
            opt.zero_grad(); loss.backward(); opt.step(); total+=float(loss)*len(y); seen+=len(y)
        history.append({'epoch':epoch,'train_loss':total/max(seen,1)})
        if epoch==1 or epoch%5==0 or epoch==args.epochs: print(f'fixed_gine epoch={epoch:02d} loss={history[-1]["train_loss"]:.5f}',flush=True)
    @torch.no_grad()
    def evaluate(idx,offset):
        model.eval(); ys=[]; logits=[]
        for batch in loader(idx,False,offset):
            batch=batch.to(device); logits.append(model(batch).cpu().numpy()); ys.append(batch.y.view(-1).cpu().numpy())
        y=np.concatenate(ys).astype(np.float64); logit=np.concatenate(logits).astype(np.float64); indices=np.asarray(idx,dtype=np.int64); prob=1/(1+np.exp(-np.clip(logit,-60,60)))
        return {'auc':float(roc_auc_score(y,prob)),'n_graphs':len(y),'n_positive':int(y.sum()),'indices':indices.tolist(),'y':y.astype(int).tolist(),'logits':logit.tolist(),'probabilities':prob.tolist(),'probability_sha256':sha(prob)}
    train_eval=evaluate(tr,1); valid_eval=evaluate(va,2); test_eval=evaluate(te,3)
    report={'protocol_id':'molhiv-fixed-epoch-original-node-gine-official-terminal-v1','date':'2026-07-28','seed':args.seed,'architecture':{'supervised_original_node_gine_layers':args.layers,'hidden':args.hidden,'jk_readout':'zero-initialized gated sum of mean-pooled layer states','ksvd_features':False},'selection_policy':{'architecture_and_hyperparameters_frozen_before_official_test':True,'train_data':'official train only','epoch_policy':f'fixed {args.epochs} epochs','official_valid_evaluations':1,'official_test_evaluations':1,'post_test_tuning_permitted':False,'ensemble_policy':'predeclared arithmetic mean of seed 0/1/2 probabilities'},'config':vars(args),'n_train':len(tr),'n_valid':len(va),'n_test':len(te),'n_train_positive':int(labels[tr].sum()),'n_valid_positive':int(labels[va].sum()),'n_test_positive':int(labels[te].sum()),'official_train_sha256':sha(tr),'encoded_graphs':len(data),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'final_jk_gates':torch.tanh(model.jk_gates).detach().cpu().tolist(),'train':train_eval,'official_valid':valid_eval,'official_test':test_eval,'history':history,'elapsed_sec':time.time()-t0,'output':args.output}
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2)); print(json.dumps({'output':str(out),'seed':args.seed,'valid_auc':valid_eval['auc'],'test_auc':test_eval['auc'],'elapsed_sec':report['elapsed_sec']},indent=2),flush=True)
if __name__=='__main__': main()
