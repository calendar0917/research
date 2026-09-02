"""Relabel-invariance audit for attributed typed Beam8."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_tud import load_tud
from .pipeline import graph_from_edge_index
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_beam8_nci1_chain_classification import DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text
from .run_beam8_nci1_compact_relation_followup import _compact_feature


PROTOCOL="tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_INVARIANCE_PROTOCOL_20260813.md"
DEFAULT_JSON=ROOT/"tracks/ksvd/results/luyin14/attributed_beam8_mutagenicity_invariance_20260813.json"
DEFAULT_REPORT=ROOT/"tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_MUTAGENICITY_INVARIANCE_20260813.md"


def _graph(adjacency:np.ndarray)->Any:
    left,right=np.nonzero(adjacency)
    return graph_from_edge_index(len(adjacency),np.asarray([left,right],dtype=np.int64))


def _tokens(item:Any)->np.ndarray:
    return np.concatenate([item.vectors,item.node_histograms],axis=1)


def _sorted_rows(values:np.ndarray)->list[tuple[float,...]]:
    return sorted(tuple(float(x) for x in row) for row in values)


def run(args:argparse.Namespace)->dict[str,Any]:
    started=time.time();graphs,_labels,features,metadata=load_tud('Mutagenicity',args.dataset_root);raw,_raw_labels,_classes,edge_dim=_load_edge_pyg('Mutagenicity',args.dataset_root);trials=[]
    for index,(graph,x,data) in enumerate(zip(graphs[:args.graphs],features[:args.graphs],raw[:args.graphs])):
        adjacency=np.zeros((graph.n,graph.n),dtype=np.int8)
        for u,v in graph.edges():adjacency[u,v]=adjacency[v,u]=1
        typed=_typed_adjacency(data,edge_dim);base=prepare_attributed_beam_graph(index,graph,0,x,typed);base_tokens=_tokens(base);base_readout=_compact_feature(base,base_tokens)
        for trial in range(args.permutations):
            permutation=np.random.default_rng(920000+index*1009+trial).permutation(graph.n)
            rel_adj=adjacency[np.ix_(permutation,permutation)];rel_typed=typed[np.ix_(permutation,permutation)];rel_x=np.asarray(x)[permutation]
            rel=prepare_attributed_beam_graph(index,_graph(rel_adj),0,rel_x,rel_typed);rel_tokens=_tokens(rel);mapped=tuple(frozenset(int(permutation[node]) for node in nodes) for nodes in rel.node_sets)
            compact=_compact_feature(rel,rel_tokens)
            trials.append({"graph_index":index,"permutation":trial,"chain_exact_match":mapped==base.node_sets,
                           "token_row_match":base_tokens.shape==rel_tokens.shape and np.array_equal(base_tokens,rel_tokens),
                           "token_multiset_match":base_tokens.shape==rel_tokens.shape and _sorted_rows(base_tokens)==_sorted_rows(rel_tokens),
                           "readout_match":base_readout.shape==compact.shape and np.allclose(base_readout,compact,atol=1e-12,rtol=0)})
        if (index+1)%100==0 or index+1==min(args.graphs,len(graphs)):print(f"{index+1}/{min(args.graphs,len(graphs))}",flush=True)
    rates={key:float(np.mean([row[key] for row in trials])) for key in ('chain_exact_match','token_row_match','token_multiset_match','readout_match')}
    gate=all(rates[key]==1.0 for key in ('token_row_match','token_multiset_match','readout_match'))
    failures={key:[row for row in trials if not row[key]][:20] for key in rates}
    return {"protocol":PROTOCOL,"dataset":metadata|{"edge_dim":edge_dim},"config":{"graphs":args.graphs,"permutations":args.permutations},"rates":rates,"failure_examples":failures,"decision":"ATTRIBUTED_BEAM8_INVARIANCE_PASS" if gate else "ATTRIBUTED_BEAM8_INVARIANCE_FAILED_STOP_TYPED_ROUTE","seconds":time.time()-started}


def render(p:dict[str,Any])->str:
    lines=["# Attributed Beam8/Mutagenicity relabel-invariance audit","",f"> 协议：`{p['protocol']}`  ",f"> 判定：`{p['decision']}`","","| invariant | match rate | gate |","|---|---:|---:|"]
    for key,value in p['rates'].items():lines.append(f"| {key} | {value:.6f} | {'diagnostic' if key=='chain_exact_match' else 'required'} |")
    lines.extend(["","## Boundary","","- exact chain node IDs may differ only inside attributed automorphisms; token and readout must remain exact.","- classification is permitted only when all three representation gates equal 1.0."]);return "\n".join(lines)+"\n"


def main()->int:
    q=argparse.ArgumentParser();q.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT);q.add_argument('--graphs',type=int,default=512);q.add_argument('--permutations',type=int,default=3);q.add_argument('--json',type=Path,default=DEFAULT_JSON);q.add_argument('--report',type=Path,default=DEFAULT_REPORT);a=q.parse_args();p=run(a);_atomic_json(a.json,p);_atomic_text(a.report,render(p));print(a.report);return 0
if __name__=='__main__':raise SystemExit(main())

