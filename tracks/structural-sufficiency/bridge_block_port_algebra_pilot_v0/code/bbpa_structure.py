"""Exact bridge / block / port decomposition and tree-DP schedule.

The decomposition is a pure function of the undirected typed graph.  Every
atom belongs to exactly one block, every non-bridge bond to exactly one block,
every bridge bond to exactly one block-tree interface, and the block tree is a
forest whose edges are the real bridge bonds.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------


class DSU:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(int(n), dtype=np.int64)
        self.rank = np.zeros(int(n), dtype=np.int8)

    def find(self, x: int) -> int:
        p = self.parent
        root = int(x)
        while p[root] != root:
            root = int(p[root])
        while p[int(x)] != root:
            nxt = int(p[int(x)])
            p[int(x)] = root
            x = nxt
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def find_bridges(n_atoms: int, bu: np.ndarray, bv: np.ndarray) -> np.ndarray:
    """Tarjan bridge finding; returns a boolean mask over edge ids.

    Iterative DFS keeps the function independent of Python's recursion limit.
    Parallel edges are handled by edge id, not by endpoint pair.
    """
    n = int(n_atoms)
    bu = np.asarray(bu, np.int64)
    bv = np.asarray(bv, np.int64)
    m = int(bu.size)
    if bv.size != m:
        raise ValueError("bond endpoint length mismatch")
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(bu.tolist(), bv.tolist())):
        if u < 0 or u >= n or v < 0 or v >= n:
            raise ValueError("bond endpoint outside atom range")
        if u == v:
            continue  # self-loop cannot be a bridge
        adj[u].append((v, e))
        adj[v].append((u, e))
    disc = np.full(n, -1, np.int64)
    low = np.zeros(n, np.int64)
    timer = 0
    is_bridge = np.zeros(m, bool)
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer
        timer += 1
        stack: list[tuple[int, int, int]] = [(root, -1, 0)]
        while stack:
            v, parent_edge, pos = stack[-1]
            if pos < len(adj[v]):
                stack[-1] = (v, parent_edge, pos + 1)
                u, e = adj[v][pos]
                if e == parent_edge:
                    continue
                if disc[u] == -1:
                    disc[u] = low[u] = timer
                    timer += 1
                    stack.append((u, e, 0))
                else:
                    low[v] = min(low[v], disc[u])
            else:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[v])
                    if low[v] > disc[p]:
                        is_bridge[parent_edge] = True
    return is_bridge


def bfs_dist(n_atoms: int, bu: np.ndarray, bv: np.ndarray, start: int) -> np.ndarray:
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v in zip(np.asarray(bu, np.int64).tolist(), np.asarray(bv, np.int64).tolist()):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    dist = np.full(int(n_atoms), -1, np.int64)
    dist[int(start)] = 0
    q = deque([int(start)])
    while q:
        v = q.popleft()
        for u in adj[v]:
            if dist[u] == -1:
                dist[u] = dist[v] + 1
                q.append(u)
    return dist


# ---------------------------------------------------------------------------
# decomposition objects
# ---------------------------------------------------------------------------


@dataclass
class Port:
    port_id: int
    block: int          # block containing the port atom
    atom: int           # molecule-local atom id
    bridge: int         # bridge id
    other_block: int


@dataclass
class Block:
    block_id: int
    atoms: np.ndarray          # sorted molecule-local atom ids
    bonds: np.ndarray          # sorted internal (non-bridge) bond ids
    ports: list[int] = field(default_factory=list)


@dataclass
class Bridge:
    bridge_id: int
    bond: int                  # raw molecule-local bond id
    atom_u: int
    atom_v: int
    block_u: int
    block_v: int
    port_u: int
    port_v: int


@dataclass
class Decomp:
    n_atoms: int
    n_bonds: int
    bu: np.ndarray
    bv: np.ndarray
    block_of_atom: np.ndarray
    blocks: list[Block]
    bridges: list[Bridge]
    ports: list[Port]
    block_adj: list[list[tuple[int, int]]]   # (neighbour block, bridge id)
    components: list[list[int]]              # block ids per connected component
    edge_step: np.ndarray                    # directed edge -> DP step
    edge_source: np.ndarray                  # directed edge -> source block
    edge_target: np.ndarray                  # directed edge -> target block
    edge_port: np.ndarray                    # directed edge -> source-side port
    edge_bridge: np.ndarray                  # directed edge -> bridge id
    edge_deps: list[list[dict[str, int]]]    # dependency entries per directed edge
    block_port_dist: dict[int, np.ndarray]   # block -> (n_ports, n_ports) distances
    max_step: int


def _blocks_from_mask(n_atoms: int, bu: np.ndarray, bv: np.ndarray,
                      is_bridge: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    dsu = DSU(n_atoms)
    for e, (u, v) in enumerate(zip(bu.tolist(), bv.tolist())):
        if not bool(is_bridge[e]) and u != v:
            dsu.union(u, v)
    groups: dict[int, list[int]] = {}
    for a in range(n_atoms):
        groups.setdefault(dsu.find(a), []).append(a)
    roots = sorted(groups.keys())
    block_of_atom = np.full(n_atoms, -1, np.int64)
    atoms_by_block: list[list[int]] = []
    for bid, root in enumerate(roots):
        atoms = sorted(groups[root])
        atoms_by_block.append(atoms)
        block_of_atom[atoms] = bid
    return block_of_atom, atoms_by_block


def decompose(n_atoms: int, bu: np.ndarray, bv: np.ndarray) -> Decomp:
    n = int(n_atoms)
    bu = np.asarray(bu, np.int64)
    bv = np.asarray(bv, np.int64)
    m = int(bu.size)
    if not (bu.size == bv.size == m):
        raise ValueError("bond arrays disagree")
    is_bridge = find_bridges(n, bu, bv)
    block_of_atom, atoms_by_block = _blocks_from_mask(n, bu, bv, is_bridge)

    internal_bonds: list[list[int]] = [[] for _ in atoms_by_block]
    for e in range(m):
        if bool(is_bridge[e]):
            continue
        u, v = int(bu[e]), int(bv[e])
        if u != v and block_of_atom[u] != block_of_atom[v]:
            raise AssertionError("non-bridge bond crosses blocks")
        internal_bonds[int(block_of_atom[u])].append(e)
    blocks = [Block(block_id=i, atoms=np.asarray(atoms, np.int64),
                    bonds=np.asarray(sorted(internal_bonds[i]), np.int64))
              for i, atoms in enumerate(atoms_by_block)]

    bridge_edges = [e for e in range(m) if bool(is_bridge[e])]
    bridges: list[Bridge] = []
    ports: list[Port] = []
    block_adj: list[list[tuple[int, int]]] = [[] for _ in blocks]
    for bidx, e in enumerate(bridge_edges):
        u, v = int(bu[e]), int(bv[e])
        bu_block, bv_block = int(block_of_atom[u]), int(block_of_atom[v])
        if bu_block == bv_block:
            raise AssertionError("bridge endpoints in the same block")
        port_u = len(ports)
        ports.append(Port(port_id=port_u, block=bu_block, atom=u, bridge=bidx,
                          other_block=bv_block))
        port_v = len(ports)
        ports.append(Port(port_id=port_v, block=bv_block, atom=v, bridge=bidx,
                          other_block=bu_block))
        blocks[bu_block].ports.append(port_u)
        blocks[bv_block].ports.append(port_v)
        block_adj[bu_block].append((bv_block, bidx))
        block_adj[bv_block].append((bu_block, bidx))
        bridges.append(Bridge(bridge_id=bidx, bond=e, atom_u=u, atom_v=v,
                              block_u=bu_block, block_v=bv_block,
                              port_u=port_u, port_v=port_v))

    for b in blocks:
        b.ports.sort()
    for adj in block_adj:
        adj.sort()

    # components and tree validation
    components: list[list[int]] = []
    seen = np.zeros(len(blocks), bool)
    for start in range(len(blocks)):
        if seen[start]:
            continue
        comp: list[int] = []
        q = deque([start])
        seen[start] = True
        while q:
            b = q.popleft()
            comp.append(b)
            for nb, _ in block_adj[b]:
                if not seen[nb]:
                    seen[nb] = True
                    q.append(nb)
        components.append(sorted(comp))
    if len(bridges) != len(blocks) - len(components):
        raise AssertionError("block graph is not a forest")
    if sum(int(b.bonds.size) for b in blocks) + len(bridges) != m:
        raise AssertionError("internal bonds + bridges != all bonds")

    # rooted schedule (deterministic per component; predictions are root-free)
    parent = np.full(len(blocks), -1, np.int64)
    depth = np.zeros(len(blocks), np.int64)
    order_all: list[int] = []
    for comp in components:
        root = int(comp[0])
        order: list[int] = []
        q = deque([root])
        parent[root] = -1
        while q:
            b = q.popleft()
            order.append(b)
            for nb, _ in block_adj[b]:
                if nb == parent[b]:
                    continue
                if parent[nb] != -1:
                    continue
                parent[nb] = b
                depth[nb] = depth[b] + 1
                q.append(nb)
        order_all.extend(order)
    height = np.zeros(len(blocks), np.int64)
    for b in sorted(order_all, key=lambda x: -int(depth[x])):
        children = [nb for nb, _ in block_adj[b] if parent[nb] == b]
        height[b] = 0 if not children else 1 + max(int(height[c]) for c in children)
    hmax = int(height.max()) if len(blocks) else 0

    n_dir = 2 * len(bridges)
    edge_step = np.zeros(n_dir, np.int64)
    edge_source = np.zeros(n_dir, np.int64)
    edge_target = np.zeros(n_dir, np.int64)
    edge_port = np.zeros(n_dir, np.int64)
    edge_bridge = np.zeros(n_dir, np.int64)
    dir_index: dict[tuple[int, int], int] = {}
    for br in bridges:
        e0, e1 = 2 * br.bridge_id, 2 * br.bridge_id + 1
        dir_index[(br.block_u, br.block_v)] = e0
        dir_index[(br.block_v, br.block_u)] = e1
        edge_source[e0], edge_target[e0], edge_port[e0] = br.block_u, br.block_v, br.port_u
        edge_source[e1], edge_target[e1], edge_port[e1] = br.block_v, br.block_u, br.port_v
        edge_bridge[e0] = edge_bridge[e1] = br.bridge_id
        u, v = br.block_u, br.block_v
        if parent[u] == v:
            edge_step[e0] = int(height[u])                    # child -> parent
            edge_step[e1] = hmax + 1 + int(depth[v])          # parent -> child
        elif parent[v] == u:
            edge_step[e1] = int(height[v])
            edge_step[e0] = hmax + 1 + int(depth[u])
        else:
            raise AssertionError("bridge is not a parent-child edge")

    # port distances inside each block (BFS on internal graph only)
    block_port_dist: dict[int, np.ndarray] = {}
    for b in blocks:
        k = len(b.ports)
        dmat = np.zeros((k, k), np.int64)
        if k:
            atom_to_local = {int(a): i for i, a in enumerate(b.atoms.tolist())}
            bu_local = np.asarray([atom_to_local[int(x)] for x in bu[b.bonds].tolist()],
                                  np.int64)
            bv_local = np.asarray([atom_to_local[int(x)] for x in bv[b.bonds].tolist()],
                                  np.int64)
            for i, p in enumerate(b.ports):
                dist_all = bfs_dist(len(b.atoms), bu_local, bv_local,
                                    atom_to_local[int(ports[p].atom)])
                for j, q in enumerate(b.ports):
                    dmat[i, j] = dist_all[atom_to_local[int(ports[q].atom)]]
        block_port_dist[b.block_id] = dmat

    # dependency entries for every directed edge
    edge_deps: list[list[dict[str, int]]] = []
    for e in range(n_dir):
        i, j = int(edge_source[e]), int(edge_target[e])
        p_ij = int(edge_port[e])
        deps: list[dict[str, int]] = []
        if len(block_adj[i]) > 1:
            ports_i = blocks[i].ports
            local_index = {p: idx for idx, p in enumerate(ports_i)}
            p_ij_local = local_index[p_ij]
            for k, bridge_ki in block_adj[i]:
                if k == j:
                    continue
                dep_edge = dir_index[(k, i)]
                p_ik = int(edge_port[dir_index[(i, k)]])  # port of i facing k
                p_ki = int(edge_port[dep_edge])          # port of k facing i
                dist = int(block_port_dist[i][p_ij_local, local_index[p_ik]])
                deps.append({
                    "dep_edge": dep_edge,
                    "dep_port": p_ki,
                    "bridge": int(bridge_ki),
                    "port": p_ik,
                    "dist_bucket": int(min(dist, 5)),
                })
        edge_deps.append(deps)

    max_step = int(edge_step.max()) if n_dir else -1
    return Decomp(n_atoms=n, n_bonds=m, bu=bu, bv=bv,
                  block_of_atom=block_of_atom, blocks=blocks, bridges=bridges,
                  ports=ports, block_adj=block_adj, components=components,
                  edge_step=edge_step, edge_source=edge_source, edge_target=edge_target,
                  edge_port=edge_port, edge_bridge=edge_bridge, edge_deps=edge_deps,
                  block_port_dist=block_port_dist, max_step=max_step)


def reschedule(d: Decomp, root_mode: str = "first") -> Decomp:
    """Recompute the DP step schedule with a different per-component root.

    Only ``edge_step`` / ``max_step`` change; the states and the readout are a
    function of the graph, not of the scheduling root.  Used by the root-choice
    invariance check.
    """
    if root_mode not in ("first", "last"):
        raise ValueError("root_mode must be 'first' or 'last'")
    n_blocks = len(d.blocks)
    parent = np.full(n_blocks, -1, np.int64)
    depth = np.zeros(n_blocks, np.int64)
    order_all: list[int] = []
    for comp in d.components:
        root = int(comp[0] if root_mode == "first" else comp[-1])
        order: list[int] = []
        q = deque([root])
        parent[root] = -1
        seen = {root}
        while q:
            b = q.popleft()
            order.append(b)
            for nb, _ in d.block_adj[b]:
                if nb in seen:
                    continue
                seen.add(nb)
                parent[nb] = b
                depth[nb] = depth[b] + 1
                q.append(nb)
        order_all.extend(order)
    height = np.zeros(n_blocks, np.int64)
    for b in sorted(order_all, key=lambda x: -int(depth[x])):
        children = [nb for nb, _ in d.block_adj[b] if parent[nb] == b]
        height[b] = 0 if not children else 1 + max(int(height[c]) for c in children)
    hmax = int(height.max()) if n_blocks else 0
    edge_step = np.zeros_like(d.edge_step)
    for br in d.bridges:
        e0, e1 = 2 * br.bridge_id, 2 * br.bridge_id + 1
        u, v = br.block_u, br.block_v
        if parent[u] == v:
            edge_step[e0] = int(height[u])
            edge_step[e1] = hmax + 1 + int(depth[v])
        elif parent[v] == u:
            edge_step[e1] = int(height[v])
            edge_step[e0] = hmax + 1 + int(depth[u])
        else:
            raise AssertionError("bridge is not a parent-child edge")
    d.edge_step = edge_step.astype(np.int64)
    d.max_step = int(edge_step.max()) if edge_step.size else -1
    return d


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------


def validate(d: Decomp, atom_features: np.ndarray | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {}
    seen = np.zeros(d.n_atoms, np.int64)
    for b in d.blocks:
        seen[b.atoms] += 1
    if not np.all(seen == 1):
        raise AssertionError("an atom is not in exactly one block")
    bond_seen = np.zeros(d.n_bonds, np.int64)
    for b in d.blocks:
        bond_seen[b.bonds] += 1
    for br in d.bridges:
        bond_seen[br.bond] += 1
    if not np.all(bond_seen == 1):
        raise AssertionError("a bond is not in exactly one block/interface")
    info["n_blocks"] = len(d.blocks)
    info["n_bridges"] = len(d.bridges)
    info["n_components"] = len(d.components)
    info["ports"] = len(d.ports)

    # every bridge separates two complementary, non-overlapping supports that
    # cover its connected component
    for br in d.bridges:
        # exclude the bridge itself by graph surgery
        keep = np.ones(d.n_bonds, bool)
        keep[int(br.bond)] = False
        dist_u = bfs_dist(d.n_atoms, d.bu[keep], d.bv[keep], br.atom_u)
        dist_v = bfs_dist(d.n_atoms, d.bu[keep], d.bv[keep], br.atom_v)
        side_u = set(np.where(dist_u >= 0)[0].tolist())
        side_v = set(np.where(dist_v >= 0)[0].tolist())
        if side_u & side_v:
            raise AssertionError("bridge sides overlap")
        if len(side_u) + len(side_v) != d.n_atoms:
            # multi-component molecule: sides cover their component, not the whole mol
            pass
        comp_blocks = None
        for comp in d.components:
            if br.block_u in comp:
                comp_blocks = comp
                break
        if comp_blocks is None:
            raise AssertionError("bridge component missing")
        comp_atoms = set()
        for b in comp_blocks:
            comp_atoms.update(d.blocks[b].atoms.tolist())
        if side_u | side_v != comp_atoms:
            raise AssertionError("bridge sides do not cover the component")
        if side_u == side_v:
            raise AssertionError("empty bridge cut")

    # directed-edge schedule must be a valid topological order
    for e, deps in enumerate(d.edge_deps):
        for dep in deps:
            if d.edge_step[dep["dep_edge"]] >= d.edge_step[e]:
                raise AssertionError("dependency does not precede its state")
    info["max_step"] = int(d.max_step)
    return info


# ---------------------------------------------------------------------------
# structural statistics
# ---------------------------------------------------------------------------


def tree_diameter(d: Decomp, comp: Sequence[int]) -> int:
    if len(comp) <= 1:
        return 0
    members = set(int(x) for x in comp)

    def far(start: int) -> tuple[int, int]:
        dist = {start: 0}
        q = deque([start])
        while q:
            b = q.popleft()
            for nb, _ in d.block_adj[b]:
                if nb in members and nb not in dist:
                    dist[nb] = dist[b] + 1
                    q.append(nb)
        node = max(dist, key=lambda x: dist[x])
        return node, dist[node]

    a, _ = far(int(comp[0]))
    _, diam = far(a)
    return int(diam)


def structure_stats(decomps: Sequence[Decomp], n_molecules: int,
                    smiles: Sequence[str] | None = None) -> dict[str, Any]:
    n = int(n_molecules)
    n_bridges = np.asarray([len(d.bridges) for d in decomps], np.int64)
    n_blocks = np.asarray([len(d.blocks) for d in decomps], np.int64)
    n_atoms = np.asarray([d.n_atoms for d in decomps], np.int64)
    block_sizes = np.asarray([len(b.atoms) for d in decomps for b in d.blocks], np.int64)
    internal_bonds = np.asarray([len(b.bonds) for d in decomps for b in d.blocks], np.int64)
    bridge_side_sizes = []
    for d in decomps:
        for br in d.bridges:
            keep = np.ones(d.n_bonds, bool)
            keep[int(br.bond)] = False
            side_u = bfs_dist(d.n_atoms, d.bu[keep], d.bv[keep], br.atom_u)
            side_v = bfs_dist(d.n_atoms, d.bu[keep], d.bv[keep], br.atom_v)
            bridge_side_sizes.append(int((side_u >= 0).sum()))
            bridge_side_sizes.append(int((side_v >= 0).sum()))
    diameters = []
    for d in decomps:
        ds = [tree_diameter(d, comp) for comp in d.components]
        diameters.append(int(max(ds)) if ds else 0)
    diam_arr = np.asarray(diameters, np.int64)

    def dist_stats(a: np.ndarray) -> dict[str, Any]:
        if a.size == 0:
            return {"n": 0}
        return {
            "n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.quantile(a, 0.9)), "max": int(a.max()),
            "min": int(a.min()),
        }

    return {
        "n_molecules": n,
        "n_bridges_total": int(n_bridges.sum()),
        "n_molecules_with_bridge": int((n_bridges > 0).sum()),
        "n_molecules_without_bridge": int((n_bridges == 0).sum()),
        "bridge_coverage_fraction": float((n_bridges > 0).mean()) if n else 0.0,
        "no_bridge_fraction": float((n_bridges == 0).mean()) if n else 0.0,
        "bridges_per_molecule": dist_stats(n_bridges),
        "blocks_per_molecule": dist_stats(n_blocks),
        "atoms_per_molecule": dist_stats(n_atoms),
        "block_size": dist_stats(block_sizes),
        "internal_bonds_per_block": dist_stats(internal_bonds),
        "bridge_side_size": dist_stats(np.asarray(bridge_side_sizes, np.int64)),
        "block_tree_diameter": dist_stats(diam_arr),
        "block_tree_diameter_hist": {str(int(k)): int(v) for k, v in
                                     zip(*np.unique(diam_arr, return_counts=True))},
        "molecules_with_diameter_ge_3": int((diam_arr >= 3).sum()),
        "fraction_with_diameter_ge_3": float((diam_arr >= 3).mean()) if n else 0.0,
        "molecules_with_diameter_ge_2": int((diam_arr >= 2).sum()),
        "fraction_with_diameter_ge_2": float((diam_arr >= 2).mean()) if n else 0.0,
        "max_blocks_per_molecule": int(n_blocks.max()) if n else 0,
        "max_bridges_per_molecule": int(n_bridges.max()) if n else 0,
    }


def build_all(raw: Mapping[str, Any], rows: Sequence[int]) -> list[Decomp]:
    from bbpa_data import molecule_view

    out = []
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    bu_all = np.asarray(raw["bond_u"], np.int64)
    bv_all = np.asarray(raw["bond_v"], np.int64)
    for row in np.asarray(rows, np.int64).tolist():
        view = molecule_view(raw, row)
        n = int(view["n_atoms"])
        b0 = int(bond_ptr[row])
        b1 = int(bond_ptr[row + 1])
        out.append(decompose(n, bu_all[b0:b1], bv_all[b0:b1]))
    return out