"""Invariant typed anchors for attributed Mutagenicity Beam8 items."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .attributed_beam8 import exact_typed_canonical_order, typed_slot_vector
from .data_tud import load_tud, node_feature_readout
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_beam8_nci1_chain_classification import (
    BeamGraph, DEFAULT_ROOT, ROOT, _adjacency, _atomic_json, _atomic_text,
    _position_and_residual_features,
)
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_real_structure_ksvd import graph_basic_features


PROTOCOL="tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_ANCHOR_PROTOCOL_20260813.md"
DEFAULT_JSON=ROOT/"tracks/ksvd/results/luyin14/attributed_beam8_mutagenicity_anchor_20260813.json"
DEFAULT_REPORT=ROOT/"tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_MUTAGENICITY_ANCHOR_20260813.md"


def _covered_types(item:BeamGraph,typed:np.ndarray,edge_dim:int)->tuple[np.ndarray,np.ndarray]:
    upper=typed[np.triu_indices(len(typed),k=1)]
    total=np.asarray([np.sum(upper==value) for value in range(1,edge_dim+1)],dtype=np.int64)
    pairs=set()
    for nodes in item.slot_nodes:
        for left_index,left in enumerate(nodes):
            for right in nodes[left_index+1:]:
                if typed[left,right]:pairs.add(tuple(sorted((int(left),int(right)))))
    covered=np.zeros(edge_dim,dtype=np.int64)
    for left,right in pairs:covered[int(typed[left,right])-1]+=1
    return total,covered


def attributed_anchor_item(
    base:BeamGraph,
    graph:Any,
    node_features:np.ndarray,
    typed:np.ndarray,
    *,
    edge_dim:int,
    patch_size:int=8,
)->tuple[BeamGraph,int]:
    adjacency=_adjacency(graph);features=np.asarray(node_features,dtype=np.float64);node_types=np.argmax(features,axis=1).astype(np.int64)
    real_node_sets=list(base.node_sets);relation_node_sets=list(base.node_sets);slot_nodes=list(base.slot_nodes);centers=list(base.centers);segments=list(base.segment_ids);vectors=list(base.vectors);histograms=list(base.node_histograms);anchor_positions=[]
    total,covered=_covered_types(base,typed,edge_dim);covered_binary=set()
    for nodes in slot_nodes:
        for left_index,left in enumerate(nodes):
            for right in nodes[left_index+1:]:
                if adjacency[left,right]:covered_binary.add(tuple(sorted((int(left),int(right)))))
    anchors=0
    for edge_type in range(1,edge_dim+1):
        if total[edge_type-1]==0 or covered[edge_type-1]>0:continue
        left,right=np.nonzero(np.triu(typed==edge_type,k=1));candidates=[]
        for u,v in zip(left,right):
            nodes=(int(u),int(v));patch_typed=typed[np.ix_(nodes,nodes)];patch_types=node_types[np.asarray(nodes,dtype=np.int64)]
            for root in nodes:
                result=exact_typed_canonical_order(patch_typed,nodes,patch_types,root=root);ordered=tuple(int(node) for node in result.node_ids);vector=typed_slot_vector(ordered,typed,node_types,patch_size=patch_size,node_dim=features.shape[1],edge_dim=edge_dim);histogram=features[np.asarray(ordered)].mean(axis=0)
                candidates.append((tuple(vector.tolist())+tuple(histogram.tolist()),ordered,int(root),vector,histogram))
        _key,ordered,root,vector,histogram=min(candidates,key=lambda row:row[0])
        real_node_sets.append(frozenset(ordered));relation_node_sets.append(frozenset({-1-anchors}));slot_nodes.append(ordered);centers.append(root);segments.append(max(segments,default=-1)+1)
        vectors.append(vector);histograms.append(histogram)
        anchor_positions.append(np.asarray([0.0,0.0,0.0,0.0,1.0,1.0,0.0,2.0/max(len(adjacency),1)],dtype=np.float64))
        covered_binary.add(tuple(sorted(ordered)))
        anchors+=1
    _position,_residual,coverage=_position_and_residual_features(adjacency,real_node_sets,segments)
    position=np.vstack([base.position_features,*anchor_positions]) if anchor_positions else base.position_features.copy()
    item=BeamGraph(index=base.index,label=base.label,vectors=np.stack(vectors),node_histograms=np.stack(histograms),node_sets=tuple(relation_node_sets),slot_nodes=tuple(slot_nodes),centers=tuple(centers),segment_ids=tuple(segments),position_features=position,residual_features=base.residual_features.copy(),graph_features=node_feature_readout(features),stats=graph_basic_features(graph),sampling={**base.sampling,"patches":len(vectors),**coverage})
    return item,anchors


def run(args:argparse.Namespace)->dict[str,Any]:
    started=time.time();graphs,_labels,features,metadata=load_tud('Mutagenicity',args.dataset_root);raw,_raw_labels,_classes,edge_dim=_load_edge_pyg('Mutagenicity',args.dataset_root)
    total=np.zeros(edge_dim,dtype=np.int64);base_cov=np.zeros(edge_dim,dtype=np.int64);anchor_cov=np.zeros(edge_dim,dtype=np.int64);graph_total=np.zeros(edge_dim,dtype=np.int64);graph_base=np.zeros(edge_dim,dtype=np.int64);graph_anchor=np.zeros(edge_dim,dtype=np.int64);anchors=[];unchanged=[]
    for index,(graph,x,data) in enumerate(zip(graphs,features,raw)):
        typed=_typed_adjacency(data,edge_dim);base=prepare_attributed_beam_graph(index,graph,0,x,typed,edge_dim=edge_dim);anchor,count=attributed_anchor_item(base,graph,x,typed,edge_dim=edge_dim)
        t,b=_covered_types(base,typed,edge_dim);_t,a=_covered_types(anchor,typed,edge_dim);total+=t;base_cov+=b;anchor_cov+=a;graph_total+=(t>0);graph_base+=(b>0);graph_anchor+=(a>0);anchors.append(count)
        unchanged.append(np.array_equal(base.vectors,anchor.vectors[:len(base.vectors)]) and base.slot_nodes==anchor.slot_nodes[:len(base.slot_nodes)] and base.segment_ids==anchor.segment_ids[:len(base.segment_ids)])
        if (index+1)%500==0 or index+1==len(graphs):print(f"{index+1}/{len(graphs)}",flush=True)
    anchors=np.asarray(anchors,dtype=np.float64);summary={"mean_anchors":float(anchors.mean()),"p95_anchors":float(np.quantile(anchors,.95)),"max_anchors":int(anchors.max(initial=0)),"base_unchanged":bool(all(unchanged)),"type_recall_base":(base_cov/np.maximum(total,1)).tolist(),"type_recall_anchor":(anchor_cov/np.maximum(total,1)).tolist(),"type_graph_coverage_base":(graph_base/np.maximum(graph_total,1)).tolist(),"type_graph_coverage_anchor":(graph_anchor/np.maximum(graph_total,1)).tolist()}
    checks={"all_type_graph_coverage":bool(np.all(graph_anchor==graph_total)),"rare_recall":summary['type_recall_anchor'][-1]>=.8,"mean_anchors":summary['mean_anchors']<=.25,"p95_anchors":summary['p95_anchors']<=1,"base_unchanged":summary['base_unchanged']}
    return {"protocol":PROTOCOL,"dataset":metadata|{"edge_dim":edge_dim},"summary":summary,"checks":checks,"decision":"ATTRIBUTED_ANCHOR_READY_FOR_CLASSIFICATION" if all(checks.values()) else "ATTRIBUTED_ANCHOR_BELOW_GATE","seconds":time.time()-started}


def render(p:dict[str,Any])->str:
    s=p['summary'];lines=["# Attributed Beam8/Mutagenicity typed-anchor audit","",f"> 协议：`{p['protocol']}`  ",f"> 判定：`{p['decision']}`","",f"- anchors mean/p95/max：`{s['mean_anchors']:.4f}` / `{s['p95_anchors']:.1f}` / `{s['max_anchors']}`；",f"- BASE unchanged：`{s['base_unchanged']}`；",f"- type recall BASE：`{s['type_recall_base']}`；",f"- type recall ANCHOR：`{s['type_recall_anchor']}`；",f"- type graph coverage BASE：`{s['type_graph_coverage_base']}`；",f"- type graph coverage ANCHOR：`{s['type_graph_coverage_anchor']}`；","","## Checks",""]
    for k,v in p['checks'].items():lines.append(f"- {k}：`{v}`；")
    return "\n".join(lines)+"\n"


def main()->int:
    q=argparse.ArgumentParser();q.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT);q.add_argument('--json',type=Path,default=DEFAULT_JSON);q.add_argument('--report',type=Path,default=DEFAULT_REPORT);a=q.parse_args();p=run(a);_atomic_json(a.json,p);_atomic_text(a.report,render(p));print(a.report);return 0
if __name__=='__main__':raise SystemExit(main())
