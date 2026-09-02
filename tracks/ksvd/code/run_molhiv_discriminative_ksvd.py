"""Validation-only label-augmented K-SVD for MolHIV.

A balanced train-patch matrix Y is augmented with two train-label rows before
K-SVD: [Y; alpha * one_hot(label)].  The learned chemistry block is normalized
and used with the unchanged OMP + max-pool graph encoder.  Labels never enter
valid/test encoding, and this script does not evaluate test.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np

from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig
from .ksvd import ksvd
from .run_molhiv_ksvd_feasibility import _random_patch_dictionary
from .run_molhiv_label_aware_ksvd import _collect_patch_pool, _encode_indices, _fit_train_valid, _select_training_patches
from .run_molhiv_next_round import size_feat


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--max-graphs',type=int,default=0); ap.add_argument('--data-seed',type=int,default=0)
    ap.add_argument('--dict-seeds',default='0'); ap.add_argument('--label-weights',default='0.1,0.3,1.0')
    ap.add_argument('--n-atoms',type=int,default=8); ap.add_argument('--sparsity',type=int,default=2)
    ap.add_argument('--ksvd-iter',type=int,default=4); ap.add_argument('--max-train-patches',type=int,default=4000)
    ap.add_argument('--selection-seed',type=int,default=1729); ap.add_argument('--output',required=True)
    ap.add_argument('--dictionary-output',default=None); args=ap.parse_args()
    seeds=[int(x) for x in args.dict_seeds.split(',') if x.strip()]; weights=[float(x) for x in args.label_weights.split(',') if x.strip()]
    b=load_molhiv(max_graphs=None if args.max_graphs<=0 else args.max_graphs,seed=args.data_seed,with_features=True)
    cfg=GraphLevelConfig(n_atoms=args.n_atoms,T=args.sparsity,T_min=1,ksvd_iter=args.ksvd_iter,seed=0,max_train_patches=args.max_train_patches,max_patches_per_graph=8,patch_feat='wl_chem_ring',normalize_patches=True,readout_mode='pool',pool='max')
    started=time.time(); Ypool,records,pool_stats=_collect_patch_pool(b,cfg)
    Y,selected,selection_stats=_select_training_patches(Ypool,records,'0.50',args.max_train_patches,args.selection_seed)
    labels=np.asarray([records[int(i)].label for i in selected],dtype=np.int64)
    H=np.zeros((2,Y.shape[1]),dtype=np.float64); H[labels,np.arange(Y.shape[1])]=1.0
    dictionaries={}; training_info={}
    for alpha in weights:
        for seed in seeds:
            Yaug=np.vstack([Y,alpha*H])
            Daug,_,info=ksvd(Yaug,n_atoms=args.n_atoms,T=args.sparsity,T_min=1,n_iter=args.ksvd_iter,seed=seed)
            D=Daug[:Y.shape[0]].copy(); D/=np.maximum(np.linalg.norm(D,axis=0,keepdims=True),1e-12)
            name=f'label_aug_a{alpha:g}_seed{seed}'; dictionaries[name]=D; training_info[name]={**info,'label_weight':alpha}
            dictionaries[f'random_patch_seed{seed}']=_random_patch_dictionary(Y,args.n_atoms,seed)
    tr=np.asarray(b.split['train'],dtype=np.int64); va=np.asarray(b.split['valid'],dtype=np.int64); idx=np.concatenate([tr,va])
    encoded=_encode_indices(b,idx,dictionaries,cfg); ntr=len(tr); ytr=b.y[tr]; yva=b.y[va]
    size=size_feat(b.graphs); str_,sva=size[tr],size[va]; base=_fit_train_valid(str_,ytr,sva,yva)
    results={}
    for name,X in encoded.items():
        only=_fit_train_valid(X[:ntr],ytr,X[ntr:],yva)
        plus=_fit_train_valid(np.hstack([X[:ntr],str_]),ytr,np.hstack([X[ntr:],sva]),yva)
        results[name]={'only':only,'plus_size':plus,'delta_valid_vs_size':plus['valid_auc']-base['valid_auc']}
        print(name,json.dumps(results[name]),flush=True)
    out={'config':vars(args),'test_policy':'test split was not encoded or evaluated by this script','pool_stats':pool_stats,'selection_stats':selection_stats,'size':base,'results':results,'training_info':training_info,'elapsed_sec':time.time()-started}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2))
    if args.dictionary_output: np.savez_compressed(args.dictionary_output,**dictionaries)
    print('wrote',args.output,flush=True)

if __name__=='__main__': main()
