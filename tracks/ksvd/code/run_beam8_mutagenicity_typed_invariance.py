"""Relabel-invariance audit for binary Beam8 covers carrying typed payloads."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_tud import load_tud
from .pipeline import graph_from_edge_index
from .run_beam8_mutagenicity_typed_classification import _typed_vectors
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text, _graph_token_feature,
    prepare_graph,
)
from .run_beam8_nci1_compact_relation_followup import _compact_feature


PROTOCOL="tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_INVARIANCE_PROTOCOL_20260813.md"
DEFAULT_JSON=ROOT/"tracks/ksvd/results/luyin14/beam8_mutagenicity_typed_invariance_20260813.json"
DEFAULT_REPORT=ROOT/"tracks/ksvd/results/luyin14/BEAM8_MUTAGENICITY_TYPED_INVARIANCE_20260813.md"


def _graph_from_adjacency(adjacency:np.ndarray)->Any:
    left,right=np.nonzero(adjacency)
    return graph_from_edge_index(len(adjacency),np.asarray([left,right],dtype=np.int64))


def _tokens(item:Any,typed:np.ndarray,edge_dim:int)->np.ndarray:
    return np.concatenate([_typed_vectors(item.slot_nodes,typed,edge_dim),item.node_histograms],axis=1)


def _sorted_rows(values:np.ndarray)->list[tuple[float,...]]:
    return sorted(tuple(float(x) for x in row) for row in values)


def run(args:argparse.Namespace)->dict[str,Any]:
    started=time.time();graphs,_labels,features,metadata=load_tud('Mutagenicity',args.dataset_root);raw,_raw_labels,_classes,edge_dim=_load_edge_pyg('Mutagenicity',args.dataset_root)
    trials=[]
    for index,(graph,x,data) in enumerate(zip(graphs[:args.graphs],features[:args.graphs],raw[:args.graphs])):
        adjacency=np.zeros((graph.n,graph.n),dtype=np.int8)
        for u,v in graph.edges():adjacency[u,v]=adjacency[v,u]=1
        typed=_typed_adjacency(data,edge_dim);base=prepare_graph(index,graph,0,x,patch_size=8,overlap=2,retained_beam=8,edge_capacity_multiplier=1.5,seed=20260813);base_tokens=_tokens(base,typed,edge_dim);base_readout=_compact_feature(base,base_tokens)
        for trial in range(args.permutations):
            rng=np.random.default_rng(910000+index*1009+trial);permutation=rng.permutation(graph.n) # new -> old
            rel_adj=adjacency[np.ix_(permutation,permutation)];rel_typed=typed[np.ix_(permutation,permutation)];rel_x=np.asarray(x)[permutation]
            rel=prepare_graph(index,_graph_from_adjacency(rel_adj),0,rel_x,patch_size=8,overlap=2,retained_beam=8,edge_capacity_multiplier=1.5,seed=20260813);rel_tokens=_tokens(rel,rel_typed,edge_dim);mapped=tuple(frozenset(int(permutation[node]) for node in nodes) for nodes in rel.node_sets)
            chain_match=mapped==base.node_sets
            row_match=base_tokens.shape==rel_tokens.shape and np.array_equal(base_tokens,rel_tokens)
            multiset_match=base_tokens.shape==rel_tokens.shape and _sorted_rows(base_tokens)==_sorted_rows(rel_tokens)
            readout_match=base_readout.shape==_compact_feature(rel,rel_tokens).shape and np.allclose(base_readout,_compact_feature(rel,rel_tokens),atol=1e-12,rtol=0)
            trials.append({"graph_index":index,"permutation":trial,"chain_match":chain_match,"token_row_match":row_match,"token_multiset_match":multiset_match,"readout_match":readout_match})
        if (index+1)%100==0 or index+1==min(args.graphs,len(graphs)):print(f"{index+1}/{min(args.graphs,len(graphs))}",flush=True)
    rates={key:float(np.mean([row[key] for row in trials])) for key in ('chain_match','token_row_match','token_multiset_match','readout_match')}
    passed=all(value==1.0 for value in rates.values())
    failures={key:[row for row in trials if not row[key]][:20] for key in rates}
    return {"protocol":PROTOCOL,"dataset":metadata|{"edge_dim":edge_dim},"config":{"graphs":args.graphs,"permutations":args.permutations},"rates":rates,"failure_examples":failures,"decision":"TYPED_BEAM8_INVARIANCE_PASS" if passed else "TYPED_BEAM8_INVARIANCE_FAILED_REIMPLEMENT_REQUIRED","seconds":time.time()-started}


def render(p:dict[str,Any])->str:
    lines=["# Beam8/Mutagenicity typed relabel-invariance audit","",f"> 协议：`{p['protocol']}`  ",f"> 判定：`{p['decision']}`","",f"- graphs/permutations：`{p['config']['graphs']}` / `{p['config']['permutations']}`；","","| invariant | match rate |","|---|---:|"]
    for k,v in p['rates'].items():lines.append(f"| {k} | {v:.6f} |")
    lines.extend(["","## Boundary","","- permutation 使用 new-index → old-index 映射；node features 与 typed adjacency 同步重排。","- gate 要求所有 trial 四项严格为 1.0。"]);return "\n".join(lines)+"\n"


def main()->int:
    q=argparse.ArgumentParser();q.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT);q.add_argument('--graphs',type=int,default=512);q.add_argument('--permutations',type=int,default=3);q.add_argument('--json',type=Path,default=DEFAULT_JSON);q.add_argument('--report',type=Path,default=DEFAULT_REPORT);a=q.parse_args();p=run(a);_atomic_json(a.json,p);_atomic_text(a.report,render(p));print(a.report);return 0
if __name__=='__main__':raise SystemExit(main())

