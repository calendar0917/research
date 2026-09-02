"""Typed Beam8 BASE/ANCHOR classification on TU Mutagenicity."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud, node_feature_readout
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_real_structure_ksvd import graph_basic_features
from .run_beam8_nci1_chain_classification import (
    BeamGraph, DEFAULT_ROOT, ROOT, _adjacency, _atomic_json, _atomic_text,
    _encode, _fit_dictionary, _graph_token_feature, _normalize_patch_tokens,
    _position_and_residual_features, _shuffle_permutation, prepare_graph,
)
from .run_beam8_nci1_compact_relation_followup import _compact_feature
from .run_beam8_mutagenicity_fair95_completion import _base_cover
from .run_beam8_mutagenicity_typed_anchor import typed_anchor_cover
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_CLASSIFICATION_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_mutagenicity_typed_classification_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_MUTAGENICITY_TYPED_CLASSIFICATION_20260813.md"
VARIANTS = (
    "FEATURE_STATS",
    "BASE_RAW_BAG", "BASE_RAW_TRUE", "BASE_RAW_SHUFFLED",
    "BASE_INIT_TRUE", "BASE_FINAL_TRUE",
    "ANCHOR_RAW_BAG", "ANCHOR_RAW_TRUE", "ANCHOR_RAW_SHUFFLED",
    "ANCHOR_INIT_BAG", "ANCHOR_INIT_TRUE",
    "ANCHOR_FINAL_BAG", "ANCHOR_FINAL_TRUE", "ANCHOR_FINAL_SHUFFLED",
)


def _typed_vectors(slot_nodes: Sequence[Sequence[int]], typed: np.ndarray, edge_dim: int, patch_size: int = 8) -> np.ndarray:
    rows=[]
    for nodes in slot_nodes:
        row=[]
        for left in range(patch_size):
            for right in range(left+1,patch_size):
                value=int(typed[nodes[left],nodes[right]]) if left<len(nodes) and right<len(nodes) else 0
                row.extend(float(value==edge_type) for edge_type in range(1,edge_dim+1))
        rows.append(row)
    return np.asarray(rows,dtype=np.float64)


def _item_from_cover(index:int, label:int, graph:Any, node_features:np.ndarray, typed:np.ndarray, cover:Any, edge_dim:int)->BeamGraph:
    adjacency=_adjacency(graph)
    slot_nodes=tuple(tuple(int(node) for node in patch.node_ids) for patch in cover.patches)
    node_sets=tuple(frozenset(nodes) for nodes in slot_nodes)
    segments=tuple(int(v) for v in cover.segment_ids)
    position,residual,coverage=_position_and_residual_features(adjacency,node_sets,segments)
    x=np.asarray(node_features,dtype=np.float64)
    hist=np.stack([x[np.asarray(nodes,dtype=np.int64)].mean(axis=0) for nodes in slot_nodes])
    return BeamGraph(index=index,label=label,vectors=_typed_vectors(slot_nodes,typed,edge_dim),node_histograms=hist,
                     node_sets=node_sets,slot_nodes=slot_nodes,centers=tuple(int(p.center) for p in cover.patches),
                     segment_ids=segments,position_features=position,residual_features=residual,
                     graph_features=node_feature_readout(x),stats=graph_basic_features(graph),
                     sampling={"components":len(set(segments)),"component_sizes":[],"patches":len(slot_nodes),"partial_components":0,**coverage})


def _fit_score(train:np.ndarray,train_y:np.ndarray,test:np.ndarray,test_y:np.ndarray)->dict[str,float]:
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=1.0,random_state=0));model.fit(train,train_y);pred=model.predict(test)
    return {"balanced_accuracy":float(balanced_accuracy_score(test_y,pred)),"accuracy":float(accuracy_score(test_y,pred))}


def _features(base:Sequence[BeamGraph],anchor:Sequence[BeamGraph],train_indices:np.ndarray,dictionary:dict[str,Any],shuffle_seed:int)->dict[str,np.ndarray]:
    def families(items):
        raw=[np.concatenate([item.vectors,item.node_histograms],axis=1) for item in items]
        init=[np.concatenate([_encode(item,dictionary['initial'],dictionary['mean'],3),item.node_histograms],axis=1) for item in items]
        final=[np.concatenate([_encode(item,dictionary['final'],dictionary['mean'],3),item.node_histograms],axis=1) for item in items]
        return raw,init,final
    br,bi,bf=families(base); ar,ai,af=families(anchor)
    # Shared family-wise normalization is fitted on ANCHOR train patches, then applied to BASE using identical moments.
    def shared(anchor_tokens,base_tokens):
        train=np.concatenate([anchor_tokens[int(i)] for i in train_indices],axis=0);mean=train.mean(0,keepdims=True);scale=np.where(train.std(0,keepdims=True)>1e-8,train.std(0,keepdims=True),1.0)
        return [(v-mean)/scale for v in base_tokens],[(v-mean)/scale for v in anchor_tokens]
    br,ar=shared(ar,br);bi,ai=shared(ai,bi);bf,af=shared(af,bf)
    rows={name:[] for name in VARIANTS}
    for idx,(b,a) in enumerate(zip(base,anchor)):
        rows['FEATURE_STATS'].append(np.concatenate([b.graph_features,b.stats]))
        permutation_b=_shuffle_permutation(len(br[idx]),seed=shuffle_seed+b.index*1009)
        permutation_a=_shuffle_permutation(len(ar[idx]),seed=shuffle_seed+a.index*1009)
        rows['BASE_RAW_BAG'].append(_graph_token_feature(b,br[idx],chain=False));rows['BASE_RAW_TRUE'].append(_compact_feature(b,br[idx]));rows['BASE_RAW_SHUFFLED'].append(_compact_feature(b,br[idx],permutation_b))
        rows['BASE_INIT_TRUE'].append(_compact_feature(b,bi[idx]));rows['BASE_FINAL_TRUE'].append(_compact_feature(b,bf[idx]))
        rows['ANCHOR_RAW_BAG'].append(_graph_token_feature(a,ar[idx],chain=False));rows['ANCHOR_RAW_TRUE'].append(_compact_feature(a,ar[idx]));rows['ANCHOR_RAW_SHUFFLED'].append(_compact_feature(a,ar[idx],permutation_a))
        rows['ANCHOR_INIT_BAG'].append(_graph_token_feature(a,ai[idx],chain=False));rows['ANCHOR_INIT_TRUE'].append(_compact_feature(a,ai[idx]))
        rows['ANCHOR_FINAL_BAG'].append(_graph_token_feature(a,af[idx],chain=False));rows['ANCHOR_FINAL_TRUE'].append(_compact_feature(a,af[idx]));rows['ANCHOR_FINAL_SHUFFLED'].append(_compact_feature(a,af[idx],permutation_a))
    return {k:np.stack(v) for k,v in rows.items()}


def _paired(folds,left,right):
    values=np.asarray([f['scores'][left]['balanced_accuracy']-f['scores'][right]['balanced_accuracy'] for f in folds])
    return {"mean":float(values.mean()),"wins":int(np.sum(values>1e-12)),"ties":int(np.sum(np.abs(values)<=1e-12)),"losses":int(np.sum(values< -1e-12)),"values":values.tolist()}


def _summary(folds):
    variants={v:{"balanced_accuracy_mean":float(np.mean([f['scores'][v]['balanced_accuracy'] for f in folds])),"balanced_accuracy_std":float(np.std([f['scores'][v]['balanced_accuracy'] for f in folds])),"accuracy_mean":float(np.mean([f['scores'][v]['accuracy'] for f in folds]))} for v in VARIANTS}
    paired={"base_relation":_paired(folds,'BASE_RAW_TRUE','BASE_RAW_SHUFFLED'),"anchor_increment":_paired(folds,'ANCHOR_INIT_TRUE','BASE_INIT_TRUE'),"anchor_relation":_paired(folds,'ANCHOR_FINAL_TRUE','ANCHOR_FINAL_SHUFFLED'),"anchor_over_bag":_paired(folds,'ANCHOR_FINAL_TRUE','ANCHOR_FINAL_BAG'),"ksvd_update":_paired(folds,'ANCHOR_FINAL_TRUE','ANCHOR_INIT_TRUE'),"attribute_complement":_paired(folds,'ANCHOR_INIT_TRUE','FEATURE_STATS')}
    checks={"base_relation":paired['base_relation']['mean']>=.01 and paired['base_relation']['wins']>=2,"anchor_increment":paired['anchor_increment']['mean']>=.005 and paired['anchor_increment']['wins']>=2,"anchor_relation":paired['anchor_relation']['mean']>=.01 and paired['anchor_relation']['wins']>=2,"anchor_over_bag":paired['anchor_over_bag']['mean']>=0 and paired['anchor_over_bag']['losses']<=1,"ksvd_update":paired['ksvd_update']['mean']>=.01 and paired['ksvd_update']['wins']>=2,"attribute_complement":paired['attribute_complement']['mean']>=.01 and paired['attribute_complement']['wins']>=2}
    if checks['anchor_increment'] and checks['anchor_relation']: decision='TYPED_ANCHOR_AND_BEAM8_RELATION_PASS'
    elif checks['anchor_increment']: decision='TYPED_SEMANTIC_COMPLETENESS_ONLY'
    elif checks['base_relation'] or checks['anchor_relation']: decision='BEAM8_RELATION_DETECTABLE_NO_ANCHOR_INCREMENT'
    else: decision='TYPED_BEAM8_CLASSIFICATION_BELOW_GATE'
    return {"variants":variants,"paired":paired,"checks":checks,"decision":decision}


def run(args):
    started=time.time();graphs,labels,features,metadata=load_tud('Mutagenicity',args.dataset_root);raw,raw_labels,_classes,edge_dim=_load_edge_pyg('Mutagenicity',args.dataset_root)
    base=[];anchor=[]
    for index,(graph,label,x,data) in enumerate(zip(graphs,labels,features,raw)):
        prepared=prepare_graph(index,graph,int(label),x,patch_size=8,overlap=2,retained_beam=8,edge_capacity_multiplier=1.5,seed=20260813);adj=_adjacency(graph);typed=_typed_adjacency(data,edge_dim);bc=_base_cover(prepared,adj);ac,_=typed_anchor_cover(prepared,adj,typed,edge_dim)
        base.append(_item_from_cover(index,int(label),graph,x,typed,bc,edge_dim));anchor.append(_item_from_cover(index,int(label),graph,x,typed,ac,edge_dim))
        if (index+1)%500==0 or index+1==len(graphs):print(f"prepare {index+1}/{len(graphs)}",flush=True)
    folds=[];splitter=StratifiedKFold(n_splits=3,shuffle=True,random_state=0)
    for fold,(train,test) in enumerate(splitter.split(np.zeros(len(labels)),labels)):
        dictionary=_fit_dictionary(anchor,train,n_atoms=24,sparsity=3,iterations=5,max_train_patches=3000,seed=fold);mat=_features(base,anchor,train,dictionary,731421+fold)
        scores={v:_fit_score(x[train],labels[train],x[test],labels[test]) for v,x in mat.items()};folds.append({"fold_index":fold,"scores":scores,"recon_rel":dictionary['training'].get('recon_rel')})
        print(f"fold {fold}: BASE={scores['BASE_RAW_TRUE']['balanced_accuracy']:.4f} ANCHOR_INIT={scores['ANCHOR_INIT_TRUE']['balanced_accuracy']:.4f} ANCHOR_FINAL={scores['ANCHOR_FINAL_TRUE']['balanced_accuracy']:.4f}",flush=True)
    return {"protocol":PROTOCOL,"dataset":metadata|{"edge_dim":edge_dim},"folds":folds,"summary":_summary(folds),"seconds":time.time()-started}


def render(p):
    s=p['summary'];lines=["# Beam8/Mutagenicity typed-anchor classification","",f"> 协议：`{p['protocol']}`  ",f"> 判定：`{s['decision']}`","","| variant | balanced accuracy | accuracy |","|---|---:|---:|"]
    for v in VARIANTS:r=s['variants'][v];lines.append(f"| {v} | {r['balanced_accuracy_mean']:.4f} ± {r['balanced_accuracy_std']:.4f} | {r['accuracy_mean']:.4f} |")
    lines.extend(["","## Paired attribution","","| comparison | mean | W/T/L |","|---|---:|---:|"])
    for k,r in s['paired'].items():lines.append(f"| {k} | {r['mean']:+.4f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines.extend(["","## Checks",""])
    for k,v in s['checks'].items():lines.append(f"- {k}：`{v}`；")
    lines.extend(["","## Boundary","","- BASE/ANCHOR 共用 outer-train ANCHOR dictionary 与 normalization。","- anchor 是独立 segment；其收益不能解释为连续 Beam8 chain 收益。"]);return "\n".join(lines)+"\n"


def main():
    q=argparse.ArgumentParser();q.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT);q.add_argument('--json',type=Path,default=DEFAULT_JSON);q.add_argument('--report',type=Path,default=DEFAULT_REPORT);a=q.parse_args();p=run(a);_atomic_json(a.json,p);_atomic_text(a.report,render(p));print(a.report);return 0
if __name__=='__main__':raise SystemExit(main())

