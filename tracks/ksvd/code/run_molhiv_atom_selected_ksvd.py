"""Train-only discriminative atom selection from an overcomplete KSVD vocabulary.

Learn a modest candidate dictionary on balanced training patches, rank atoms by
the signed positive-vs-negative difference in sparse-code usage, retain equal
numbers of positive- and negative-associated atoms, then use unchanged OMP and
max pooling on train/official-valid graphs. Test is never evaluated.
"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np
from .data_molhiv import load_molhiv
from .graph_level import GraphLevelConfig
from .ksvd import ksvd,_omp
from .run_molhiv_ksvd_feasibility import _random_patch_dictionary
from .run_molhiv_label_aware_ksvd import _collect_patch_pool,_select_training_patches,_encode_indices,_fit_train_valid
from .run_molhiv_next_round import size_feat


def sparse_codes(D,Y,T):
    X=np.zeros((D.shape[1],Y.shape[1]))
    for i in range(Y.shape[1]): X[:,i]=_omp(D,Y[:,i],T)
    return X


def select_atoms(D,X,labels,n_keep):
    A=np.abs(X); pos=A[:,labels==1]; neg=A[:,labels==0]
    mu_pos=pos.mean(axis=1); mu_neg=neg.mean(axis=1)
    var=pos.var(axis=1)/max(pos.shape[1],1)+neg.var(axis=1)/max(neg.shape[1],1)
    score=(mu_pos-mu_neg)/np.sqrt(var+1e-8)
    n_pos=n_keep//2; n_neg=n_keep-n_pos
    hi=[int(i) for i in np.argsort(-score)[:n_pos]]; lo=[int(i) for i in np.argsort(score) if int(i) not in hi][:n_neg]
    chosen=np.asarray(hi+lo,dtype=np.int64)
    return D[:,chosen],{'chosen':chosen.tolist(),'scores':score[chosen].tolist(),'all_scores':score.tolist(),'positive_atoms':hi,'negative_atoms':lo}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--max-graphs',type=int,default=0); ap.add_argument('--data-seed',type=int,default=0)
    ap.add_argument('--dict-seeds',default='0'); ap.add_argument('--candidate-atoms',type=int,default=16); ap.add_argument('--keep-atoms',type=int,default=8)
    ap.add_argument('--train-sparsity',type=int,default=3); ap.add_argument('--encode-sparsity',type=int,default=2); ap.add_argument('--ksvd-iter',type=int,default=4)
    ap.add_argument('--max-train-patches',type=int,default=4000); ap.add_argument('--selection-seed',type=int,default=1729)
    ap.add_argument('--output',required=True); ap.add_argument('--dictionary-output',default=None); args=ap.parse_args()
    seeds=[int(x) for x in args.dict_seeds.split(',') if x.strip()]
    b=load_molhiv(max_graphs=None if args.max_graphs<=0 else args.max_graphs,seed=args.data_seed,with_features=True)
    pool_cfg=GraphLevelConfig(n_atoms=args.candidate_atoms,T=args.train_sparsity,T_min=1,ksvd_iter=args.ksvd_iter,seed=0,max_train_patches=args.max_train_patches,max_patches_per_graph=8,patch_feat='wl_chem_ring',normalize_patches=True,readout_mode='pool',pool='max')
    started=time.time(); Ypool,records,pool_stats=_collect_patch_pool(b,pool_cfg); Y,selected,sel_stats=_select_training_patches(Ypool,records,'0.50',args.max_train_patches,args.selection_seed)
    labels=np.asarray([records[int(i)].label for i in selected],dtype=np.int64); dictionaries={}; selection={}; training_info={}
    for seed in seeds:
        D,X,info=ksvd(Y,n_atoms=args.candidate_atoms,T=args.train_sparsity,T_min=1,n_iter=args.ksvd_iter,seed=seed)
        Ds,si=select_atoms(D,X,labels,args.keep_atoms); dictionaries[f'ksvd_selected_seed{seed}']=Ds; selection[f'ksvd_seed{seed}']=si; training_info[f'ksvd_seed{seed}']=info
        Dr=_random_patch_dictionary(Y,args.candidate_atoms,seed); Xr=sparse_codes(Dr,Y,args.train_sparsity); Drs,sir=select_atoms(Dr,Xr,labels,args.keep_atoms)
        dictionaries[f'random_selected_seed{seed}']=Drs; selection[f'random_seed{seed}']=sir
    cfg=GraphLevelConfig(n_atoms=args.keep_atoms,T=args.encode_sparsity,T_min=1,ksvd_iter=args.ksvd_iter,seed=0,max_train_patches=args.max_train_patches,max_patches_per_graph=8,patch_feat='wl_chem_ring',normalize_patches=True,readout_mode='pool',pool='max')
    tr=np.asarray(b.split['train'],dtype=np.int64); va=np.asarray(b.split['valid'],dtype=np.int64); idx=np.concatenate([tr,va]); enc=_encode_indices(b,idx,dictionaries,cfg); ntr=len(tr); ytr,yva=b.y[tr],b.y[va]
    size=size_feat(b.graphs); st,sv=size[tr],size[va]; base=_fit_train_valid(st,ytr,sv,yva); results={}
    for name,Xg in enc.items():
        only=_fit_train_valid(Xg[:ntr],ytr,Xg[ntr:],yva); plus=_fit_train_valid(np.hstack([Xg[:ntr],st]),ytr,np.hstack([Xg[ntr:],sv]),yva)
        results[name]={'only':only,'plus_size':plus,'delta_valid_vs_size':plus['valid_auc']-base['valid_auc']}; print(name,json.dumps(results[name]),flush=True)
    out={'config':vars(args),'test_policy':'test split was not encoded or evaluated by this script','pool_stats':pool_stats,'selection_stats':sel_stats,'size':base,'results':results,'atom_selection':selection,'training_info':training_info,'elapsed_sec':time.time()-started}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(out,indent=2))
    if args.dictionary_output: np.savez_compressed(args.dictionary_output,**dictionaries)
    print('wrote',args.output,flush=True)
if __name__=='__main__': main()
