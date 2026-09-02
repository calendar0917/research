"""Minimal typed-anchor coverage audit for Mutagenicity Beam8 BASE."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np

from .canonical_slots import exact_canonical_order
from .data_tud import load_tud
from .overlap_cover import _make_cover, _make_patch
from .run_beam8_nci1_chain_classification import DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text, _adjacency, prepare_graph
from .run_beam8_mutagenicity_fair95_completion import _base_cover, _covered_by_type
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_luyin14_edge_aware_joint import _load_edge_pyg
from .run_luyin14_route import _completion_nodes, _covered_edges


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_MUTAGENICITY_TYPED_ANCHOR_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_mutagenicity_typed_anchor_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_MUTAGENICITY_TYPED_ANCHOR_20260813.md"


def typed_anchor_cover(item: Any, adjacency: np.ndarray, typed: np.ndarray, edge_dim: int, *, patch_size: int = 8) -> tuple[Any, int]:
    base = _base_cover(item, adjacency)
    patches = list(base.patches)
    segments = list(base.segment_ids)
    targets = list(base.target_edges)
    bridges = list(base.bridge_lengths)
    binary_covered = _covered_edges(adjacency, patches)
    total, covered = _covered_by_type(base, typed, edge_dim)
    anchors = 0
    for edge_type in range(1, edge_dim + 1):
        if total[edge_type - 1] == 0 or covered[edge_type - 1] > 0:
            continue
        left, right = np.nonzero(np.triu(typed == edge_type, k=1))
        candidates = sorted((int(u), int(v)) for u, v in zip(left, right))
        target = max(
            candidates,
            key=lambda pair: (
                int(np.sum(adjacency[pair[0]])) + int(np.sum(adjacency[pair[1]])),
                pair,
            ),
        )
        nodes = _completion_nodes(
            adjacency, target, patch_size=patch_size, covered_edges=binary_covered
        )
        local = adjacency[np.ix_(nodes, nodes)]
        others = tuple(node for node in nodes if node != target[0])
        canonical = exact_canonical_order(
            local,
            nodes,
            color_cells=((target[0],), others) if others else ((target[0],),),
        )
        patches.append(_make_patch(adjacency, canonical.node_ids, target[0]))
        segments.append(max(segments, default=-1) + 1)
        targets.append(target)
        bridges.append(0)
        binary_covered = _covered_edges(adjacency, patches)
        anchors += 1
    cover = _make_cover("beam8_base_typed_anchor", patches, segments, targets, bridges)
    return cover, anchors


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, _classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if not np.array_equal(labels, raw_labels): raise RuntimeError("loaders disagree")
    total = np.zeros(edge_dim, dtype=np.int64); base_covered = np.zeros(edge_dim, dtype=np.int64); anchor_covered = np.zeros(edge_dim, dtype=np.int64)
    graph_total = np.zeros(edge_dim, dtype=np.int64); graph_base = np.zeros(edge_dim, dtype=np.int64); graph_anchor = np.zeros(edge_dim, dtype=np.int64)
    anchors=[]; unchanged=[]
    for index,(graph,node_features,data) in enumerate(zip(graphs,features,raw)):
        item=prepare_graph(index,graph,0,node_features,patch_size=8,overlap=2,retained_beam=8,edge_capacity_multiplier=1.5,seed=20260813)
        adjacency=_adjacency(graph); typed=_typed_adjacency(data,edge_dim); base=_base_cover(item,adjacency); anchor,count=typed_anchor_cover(item,adjacency,typed,edge_dim)
        t,b=_covered_by_type(base,typed,edge_dim); _t,a=_covered_by_type(anchor,typed,edge_dim)
        total+=t; base_covered+=b; anchor_covered+=a; anchors.append(count)
        graph_total+=(t>0); graph_base+=(b>0); graph_anchor+=(a>0)
        unchanged.append(all(tuple(p.node_ids)==tuple(q.node_ids) for p,q in zip(base.patches,anchor.patches[:len(base.patches)])))
        if (index+1)%500==0 or index+1==len(graphs): print(f"{index+1}/{len(graphs)}",flush=True)
    anchors=np.asarray(anchors,dtype=np.float64)
    summary={"mean_anchors":float(anchors.mean()),"p95_anchors":float(np.quantile(anchors,.95)),"max_anchors":int(anchors.max(initial=0)),"base_unchanged":bool(all(unchanged)),
             "type_aggregate_recall_base":(base_covered/np.maximum(total,1)).tolist(),"type_aggregate_recall_anchor":(anchor_covered/np.maximum(total,1)).tolist(),
             "type_graph_coverage_base":(graph_base/np.maximum(graph_total,1)).tolist(),"type_graph_coverage_anchor":(graph_anchor/np.maximum(graph_total,1)).tolist(),
             "type_total_edges":total.tolist(),"type_graphs":graph_total.tolist()}
    checks={"all_type_graph_coverage":bool(np.all(graph_anchor==graph_total)),"rare_aggregate_recall":bool(summary['type_aggregate_recall_anchor'][-1]>=.8),"mean_anchors":bool(summary['mean_anchors']<=.25),"p95_anchors":bool(summary['p95_anchors']<=1),"base_unchanged":summary['base_unchanged']}
    return {"protocol":PROTOCOL,"dataset":metadata|{"edge_dim":edge_dim},"summary":summary,"checks":checks,"decision":"TYPED_ANCHOR_READY_FOR_CLASSIFICATION" if all(checks.values()) else "TYPED_ANCHOR_BELOW_GATE","seconds":time.time()-started}


def render(p:dict[str,Any])->str:
    s=p['summary']; lines=["# Beam8/Mutagenicity typed-anchor audit","",f"> 协议：`{p['protocol']}`  ",f"> 判定：`{p['decision']}`","",f"- mean/p95/max anchors：`{s['mean_anchors']:.4f}` / `{s['p95_anchors']:.1f}` / `{s['max_anchors']}`；",f"- BASE unchanged：`{s['base_unchanged']}`；",f"- aggregate recall BASE：`{s['type_aggregate_recall_base']}`；",f"- aggregate recall ANCHOR：`{s['type_aggregate_recall_anchor']}`；",f"- graph coverage BASE：`{s['type_graph_coverage_base']}`；",f"- graph coverage ANCHOR：`{s['type_graph_coverage_anchor']}`；","","## Checks",""]
    for k,v in p['checks'].items(): lines.append(f"- {k}：`{v}`；")
    lines.extend(["","## Boundary","","- anchor 使用 bond type 输入，但不使用 graph label；原 Beam8 BASE chain 完全不变。","- anchor 是显式新 segment，不能解释为连续 chain relation 收益。"]); return "\n".join(lines)+"\n"


def main()->int:
    q=argparse.ArgumentParser();q.add_argument('--dataset-root',type=Path,default=DEFAULT_ROOT);q.add_argument('--json',type=Path,default=DEFAULT_JSON);q.add_argument('--report',type=Path,default=DEFAULT_REPORT);a=q.parse_args();p=run(a);_atomic_json(a.json,p);_atomic_text(a.report,render(p));print(a.report);return 0
if __name__=='__main__':raise SystemExit(main())
