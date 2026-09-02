"""Validation-only conditional KSVD dictionaries for MolHIV.

Labels are used only to construct two training patch pools: positive and
negative.  At inference the same unlabeled patch matrix is encoded against
both KSVD dictionaries.  No test graph is loaded or evaluated.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig, bundle_to_Y, encode_patch_matrix, sample_patches_graph_level
from .ksvd import ksvd
from .run_molhiv_ksvd_feasibility import _random_patch_dictionary
from .run_molhiv_next_round import size_feat


def fit(Xtr, ytr, Xva, yva, seed=0):
    clf=make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, random_state=seed, class_weight='balanced'))
    clf.fit(Xtr,ytr)
    def auc(X,y):
        p=clf.predict_proba(X); classes=list(clf.named_steps['logisticregression'].classes_)
        return float(roc_auc_score(y,p[:,classes.index(1)]))
    return {'train_auc':auc(Xtr,ytr),'valid_auc':auc(Xva,yva)}


def select(Y, labels, want, n, seed):
    rng=np.random.default_rng(seed)
    pos=np.flatnonzero(labels==1); neg=np.flatnonzero(labels==0)
    n=min(n, len(labels)); np_=n//2; nn=n-np_
    if want=='positive': idx=rng.choice(pos,np_,replace=np_>len(pos))
    else: idx=rng.choice(neg,nn,replace=nn>len(neg))
    return Y[:,idx]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-graphs',type=int,default=0); ap.add_argument('--data-seed',type=int,default=0)
    ap.add_argument('--dict-seeds',default='0,1,2'); ap.add_argument('--n-atoms',type=int,default=8)
    ap.add_argument('--sparsity',type=int,default=2); ap.add_argument('--ksvd-iter',type=int,default=4)
    ap.add_argument('--max-patches-per-class',type=int,default=2000); ap.add_argument('--max-patches-per-graph',type=int,default=8)
    ap.add_argument('--output',required=True); ap.add_argument('--dictionary-output',default=None)
    args=ap.parse_args(); seeds=[int(s) for s in args.dict_seeds.split(',') if s.strip()]
    max_graphs=None if args.max_graphs<=0 else args.max_graphs
    b=load_molhiv(max_graphs=max_graphs,seed=args.data_seed,with_features=True)
    cfg=GraphLevelConfig(n_atoms=args.n_atoms,T=args.sparsity,T_min=1,ksvd_iter=args.ksvd_iter,seed=0,max_train_patches=4000,max_patches_per_graph=args.max_patches_per_graph,patch_feat='wl_chem_ring',normalize_patches=True,readout_mode='pool',pool='max')
    cols=[]; labels=[]; train_counts=[]
    t=time.time()
    for gi in b.split['train']:
        gi=int(gi); g=b.graphs[gi]
        pb,_=sample_patches_graph_level(g,cfg,seed=cfg.seed+gi)
        sets=pb.node_sets
        if len(sets)>args.max_patches_per_graph:
            rng=np.random.default_rng(cfg.seed+gi*1009); sel=rng.choice(len(sets),size=args.max_patches_per_graph,replace=False); sets=[sets[int(i)] for i in sorted(sel)]
        Y,_=bundle_to_Y(g,type('PB',(),{'node_sets':sets})(),cfg.max_nodes,cfg.order_mode,patch_feat=cfg.patch_feat,node_feat=b.node_feats[gi],edge_feat=b.edge_feats[gi])
        Y=Y/np.maximum(np.linalg.norm(Y,axis=0,keepdims=True),1e-12)
        cols.extend([Y[:,j] for j in range(Y.shape[1]) if np.linalg.norm(Y[:,j])>1e-12]); labels.extend([int(b.y[gi]>0.5)]*Y.shape[1]); train_counts.append(Y.shape[1])
    Ypool=np.stack(cols,axis=1); labels=np.asarray(labels,dtype=np.int64)
    pos_pool=Ypool[:,labels==1]; neg_pool=Ypool[:,labels==0]
    dictionaries={}; info={}; results={}; encoded={}
    tr=np.asarray(b.split['train'],dtype=np.int64); va=np.asarray(b.split['valid'],dtype=np.int64); encode_idx=np.concatenate([tr,va])
    for seed in seeds:
        for kind, pool in [('pos',pos_pool),('neg',neg_pool)]:
            n=min(args.max_patches_per_class,pool.shape[1]); rng=np.random.default_rng(1729+seed*101+ (0 if kind=='pos' else 1)); idx=rng.choice(pool.shape[1],n,replace=False); Y=pool[:,idx]
            D,_,inf=ksvd(Y,n_atoms=args.n_atoms,T=args.sparsity,T_min=1,n_iter=args.ksvd_iter,seed=seed+ (1000 if kind=='neg' else 0)); name=f'{kind}_ksvd_seed{seed}'; dictionaries[name]=D; info[name]=inf
            # matched random-patch control is useful for the paired check
            rn=f'{kind}_random_seed{seed}'; dictionaries[rn]=_random_patch_dictionary(Y,args.n_atoms,seed+ (1000 if kind=='neg' else 0))
    # Vectorization is the expensive part.  Build each graph's Y once and
    # immediately encode it with every dictionary instead of rebuilding Y for
    # every seed/family.
    rows={name:[] for name in dictionaries}; metas={name:[] for name in dictionaries}
    for gi0 in encode_idx:
        gi=int(gi0); g=b.graphs[gi]; pb,_=sample_patches_graph_level(g,cfg,seed=cfg.seed+gi*13)
        Y,_=bundle_to_Y(g,pb,cfg.max_nodes,cfg.order_mode,patch_feat=cfg.patch_feat,node_feat=b.node_feats[gi],edge_feat=b.edge_feats[gi])
        for name,D in dictionaries.items():
            s,meta=encode_patch_matrix(Y,D,cfg,return_patch_errors=True); rows[name].append(s); metas[name].append(meta)
    encoded={name:np.stack(values) for name,values in rows.items()}
    size=size_feat(b.graphs); size_tr,size_va=size[tr],size[va]; ytr,yva=b.y[tr],b.y[va]; base=fit(size_tr,ytr,size_va,yva)
    pair_features={}
    for seed in seeds:
        for prefix in ['ksvd','random']:
            kp=f'pos_{prefix}_seed{seed}'; kn=f'neg_{prefix}_seed{seed}'
            zp,mp=encoded[kp],metas[kp]; zn,mn=encoded[kn],metas[kn]; ntr=len(tr)
            # Separate channels and a simple signed residual. All are unlabeled at encoding time.
            feats={'concat':np.hstack([zp,zn]), 'diff':zp-zn, 'absdiff':np.abs(zp-zn), 'concat_diff':np.hstack([zp,zn,zp-zn])}
            # scalar reconstruction residual: positive dictionary should fit positive-like graphs better
            feats['recon_diff']=np.column_stack([[a['recon_rel']-bb['recon_rel'] for a,bb in zip(mp,mn)]])
            margin_rows=[]
            for a,bb in zip(mp,mn):
                margin=np.asarray(a['patch_recon_errors'])-np.asarray(bb['patch_recon_errors'])
                q=np.quantile(margin,[0.0,0.1,0.25,0.5,0.75,0.9,1.0])
                k=min(3,len(margin))
                margin_rows.append(np.concatenate([q,[margin.mean(),float(np.mean(margin<0)),float(np.sort(margin)[:k].mean())]]))
            feats['patch_margin']=np.stack(margin_rows)
            pair_features[(prefix,seed)]=feats
            for mode,X in feats.items():
                r=fit(X[:ntr],ytr,X[ntr:],yva)
                rs=fit(np.hstack([X[:ntr],size_tr]),ytr,np.hstack([X[ntr:],size_va]),yva)
                results[f'{prefix}_seed{seed}_{mode}']={'only':r,'plus_size':rs,'delta_valid_vs_size':rs['valid_auc']-base['valid_auc']}
                print(f'{prefix}_seed{seed}_{mode}',json.dumps(results[f'{prefix}_seed{seed}_{mode}']),flush=True)

    # Permutation-align atoms to seed 0 before averaging their graph-level
    # activations.  KSVD atoms are unordered; an unaligned average is invalid.
    stability={}
    if len(seeds)>1:
        ref_seed=seeds[0]
        for prefix in ['ksvd','random']:
            aligned={'pos':[],'neg':[]}; recon=[]; align_info=[]
            for seed in seeds:
                seed_info={'seed':seed}
                for kind in ['pos','neg']:
                    ref=dictionaries[f'{kind}_{prefix}_seed{ref_seed}']
                    cur=dictionaries[f'{kind}_{prefix}_seed{seed}']
                    if seed==ref_seed:
                        order=np.arange(args.n_atoms); scores=np.ones(args.n_atoms)
                    else:
                        sim=np.abs(ref.T@cur); row,col=linear_sum_assignment(-sim)
                        order=col[np.argsort(row)]; scores=sim[row,col][np.argsort(row)]
                    aligned[kind].append(encoded[f'{kind}_{prefix}_seed{seed}'][:,order])
                    seed_info[kind]={'mean_abs_cosine':float(scores.mean()),'min_abs_cosine':float(scores.min()),'order':order.tolist()}
                recon.append(pair_features[(prefix,seed)]['recon_diff'])
                align_info.append(seed_info)
            zp=np.mean(aligned['pos'],axis=0); zn=np.mean(aligned['neg'],axis=0)
            ensemble={'concat':np.hstack([zp,zn]),'diff':zp-zn,'absdiff':np.abs(zp-zn),'concat_diff':np.hstack([zp,zn,zp-zn]),'recon_diff':np.mean(recon,axis=0),'patch_margin':np.mean([pair_features[(prefix,seed)]['patch_margin'] for seed in seeds],axis=0)}
            stability[prefix]=align_info
            for mode,X in ensemble.items():
                r=fit(X[:ntr],ytr,X[ntr:],yva); rs=fit(np.hstack([X[:ntr],size_tr]),ytr,np.hstack([X[ntr:],size_va]),yva)
                key=f'{prefix}_ensemble_{mode}'
                results[key]={'only':r,'plus_size':rs,'delta_valid_vs_size':rs['valid_auc']-base['valid_auc']}
                print(key,json.dumps(results[key]),flush=True)
    out={'config':vars(args),'test_policy':'test split was not encoded or evaluated by this script','pool_stats':{'n_pool':int(Ypool.shape[1]),'n_positive_pool':int(pos_pool.shape[1]),'positive_pool_fraction':float(labels.mean()),'elapsed_sec':time.time()-t},'size':base,'results':results,'training_info':info,'stability':stability}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2))
    if args.dictionary_output:
        np.savez_compressed(args.dictionary_output,**dictionaries)
    print('wrote',args.output,flush=True)

if __name__=='__main__': main()
