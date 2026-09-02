"""Label-free geometry audit for invariant attributed Beam8."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_beam8_mutagenicity_typed_feasibility import (
    _decision, _summary, _typed_adjacency,
)
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_beam8_nci1_chain_classification import DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text


PROTOCOL="tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_GEOMETRY_PROTOCOL_20260813.md"
DEFAULT_JSON=ROOT/"tracks/ksvd/results/luyin14/attributed_beam8_mutagenicity_geometry_20260813.json"
DEFAULT_REPORT=ROOT/"tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_MUTAGENICITY_GEOMETRY_20260813.md"
GEOMETRIES={"s10_o3":(10,3),"s8_o2":(8,2)}


def run(args:argparse.Namespace)->dict[str,Any]:
    started=time.time();graphs,labels,features,metadata=load_tud('Mutagenicity',args.dataset_root);raw,raw_labels,_classes,edge_dim=_load_edge_pyg('Mutagenicity',args.dataset_root);typed=[_typed_adjacency(d,edge_dim) for d in raw];results={}
    for name,(size,overlap) in GEOMETRIES.items():
        items=[]
        for index,(graph,x,t) in enumerate(zip(graphs,features,typed)):
            items.append(prepare_attributed_beam_graph(index,graph,0,x,t,patch_size=size,overlap=overlap))
            if (index+1)%500==0 or index+1==len(graphs):print(f"{name} {index+1}/{len(graphs)}",flush=True)
        results[name]=_summary(items,typed,edge_dim,seed=731421)
    return {"protocol":PROTOCOL,"dataset":metadata|{"edge_dim":edge_dim},"geometries":results,"decision":_decision(results['s10_o3'],results['s8_o2']),"seconds":time.time()-started}


def render(p:dict[str,Any])->str:
    lines=["# Attributed Beam8/Mutagenicity geometry audit","",f"> 协议：`{p['protocol']}`  ",f"> 判定：`{p['decision']['decision']}`","","| metric | s10/o3 | s8/o2 |","|---|---:|---:|"]
    metrics=("mean_patches","single_patch_fraction","graphs_with_chain_fraction","mean_directed_chain_edges","partial_component_fraction","mean_edge_coverage","edge_coverage_p10","mean_node_coverage","rare_type_graph_coverage")
    for m in metrics:lines.append(f"| {m} | {p['geometries']['s10_o3'][m]:.4f} | {p['geometries']['s8_o2'][m]:.4f} |")
    lines.extend(["","## Per-bond recall","","| geometry | type | recall | graph recall |","|---|---:|---:|---:|"])
    for g in GEOMETRIES:
        for r in p['geometries'][g]['per_edge_type']:lines.append(f"| {g} | {r['edge_type']} | {r['aggregate_recall']:.4f} | {r['mean_graph_recall']:.4f} |")
    lines.extend(["","## Binding",""])
    for g in GEOMETRIES:lines.append(f"- {g} node margin `{p['geometries'][g]['node_chain_binding']['true_minus_shuffled']:+.4f}`；new-bond margin `{p['geometries'][g]['new_bond_chain_binding']['true_minus_shuffled']:+.4f}`。")
    lines.extend(["","## Checks",""])
    for k,v in p['decision']['checks'].items():lines.append(f"- {k}：`{v}`；")
    return "\n".join(lines)+"\n"


def main()->int:
    q=argparse.ArgumentParser();q.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT);q.add_argument('--json',type=Path,default=DEFAULT_JSON);q.add_argument('--report',type=Path,default=DEFAULT_REPORT);a=q.parse_args();p=run(a);_atomic_json(a.json,p);_atomic_text(a.report,render(p));print(a.report);return 0
if __name__=='__main__':raise SystemExit(main())

