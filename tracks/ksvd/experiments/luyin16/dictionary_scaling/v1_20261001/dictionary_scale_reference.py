"""Width-family reference, NumPy only; no ZINC fit or PyTorch equivalence claim.

Source audit: calendar0917/research@00c203624682a9ca46d54a225721e25015d8b28c.
The reference starts AFTER the unchanged 446-wide semantic/slot interface.
The graph auxiliary 40-vector starts AFTER the unchanged global/topology MLPs.
Small uses d=48, K=96, p=16. Full multiplies task-path widths by three.
The existing structural dictionary, stem and graph auxiliaries stay fixed.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from latent_dictionary_bridge_reference import bridge, frame_initialization


def affine_count(i, o):
    return i * o + o


def mlp_count(i, h, o, norm=False):
    return affine_count(i, h) + affine_count(h, o) + (2 * h if norm else 0)


@dataclass(frozen=True)
class WidthSpec:
    multiplier: int

    def __post_init__(self):
        if self.multiplier not in (1, 2, 3):
            raise ValueError("audit widths are 1, 2, 3; only 1/3 are named presets")

    @property
    def d(self):
        return 48 * self.multiplier

    @property
    def k(self):
        return 2 * self.d

    @property
    def p(self):
        return 16 * self.multiplier

    @property
    def reader_input(self):
        return 2 * self.d + 1 + 5 * (2 * self.p + 1) + 32 + 8

    def inventory(self):
        m, d, p = self.multiplier, self.d, self.p
        return {
            "structural_dictionary": 65 * 32,
            "node_binding": (33 + 28) * 96,
            "edge_binding": (3 * 33 + 4) * 48,
            "node_slot_encoder": mlp_count(96, 64, 48),
            "edge_slot_encoder": mlp_count(48, 48, 32),
            "fusion": mlp_count(446, 114 * m, d),
            "task_dictionary_D_V": 2 * d * self.k,
            "pair_projection": d * p,
            "relation_encoder": mlp_count(15, 32 * m, p, norm=True),
            "distance_gate": 5 * p,
            "pair_encoder": mlp_count(4 * p, 64 * m, p, norm=True),
            "global_encoder": mlp_count(62, 32, 32, norm=True),
            "topology_encoder": mlp_count(25, 16, 8),
            "reader": (affine_count(self.reader_input, 13 * m)
                       + affine_count(13 * m, 13 * m)
                       + affine_count(13 * m, 1)),
        }

    def parameters(self):
        return sum(self.inventory().values())


def relu(x):
    return np.maximum(x, 0)


def mlp_init(rng, i, h, o, *, norm=False, activation="relu", final_relu=True):
    def linear(a, b):
        s = 1 / np.sqrt(a)
        return rng.uniform(-s, s, (a, b)), rng.uniform(-s, s, b)
    w1, b1 = linear(i, h)
    w2, b2 = linear(h, o)
    return dict(w1=w1, b1=b1, w2=w2, b2=b2,
                gamma=np.ones(h) if norm else None,
                beta=np.zeros(h) if norm else None,
                activation=activation, final_relu=final_relu)


def mlp(x, state):
    z = x @ state["w1"] + state["b1"]
    if state["gamma"] is not None:
        z = (z - z.mean(axis=1, keepdims=True)) / np.sqrt(z.var(axis=1, keepdims=True) + 1e-5)
        z = z * state["gamma"] + state["beta"]
    z = z / (1 + np.exp(-z)) if state["activation"] == "silu" else relu(z)
    z = z @ state["w2"] + state["b2"]
    return relu(z) if state["final_relu"] else z


def state_init(spec, seed=41):
    rng = np.random.default_rng(seed)
    m, d, p = spec.multiplier, spec.d, spec.p
    D, V = frame_initialization(d, seed=0)
    h = 13 * m
    s = dict(
        fusion=mlp_init(rng, 446, 114 * m, d, activation="silu", final_relu=False),
        D=D, V=V,
        projection=rng.uniform(-1 / np.sqrt(d), 1 / np.sqrt(d), (d, p)),
        relation=mlp_init(rng, 15, 32 * m, p, norm=True),
        gate=rng.normal(size=(5, p)),
        pair=mlp_init(rng, 4 * p, 64 * m, p, norm=True),
        reader=mlp_init(rng, spec.reader_input, h, h, final_relu=True),
        final_w=rng.uniform(-1 / np.sqrt(h), 1 / np.sqrt(h), (h, 1)),
        final_b=rng.uniform(-1 / np.sqrt(h), 1 / np.sqrt(h), 1),
    )
    return s


def pool(value, batch, n_graphs, zero_count=True):
    """Actual parent statistic: sum, sum-of-squares, log1p(count); C6 zeros count."""
    total = np.zeros((n_graphs, value.shape[1]), dtype=value.dtype)
    squared = np.zeros_like(total)
    np.add.at(total, batch, value)
    np.add.at(squared, batch, value * value)
    count = np.log1p(np.bincount(batch, minlength=n_graphs)).astype(value.dtype)[:, None]
    if zero_count:
        count = count * 0
    return np.concatenate([total, squared, count], axis=1)


def forward(state, fused_input, batch, pairs, buckets, relations, auxiliary,
            *, zero_count=True, return_aux=False):
    n_graphs = len(auxiliary)
    h = mlp(fused_input, state["fusion"])
    E, code_aux = bridge(h, state["D"], state["V"])
    u = E @ state["projection"]
    left, right = u[pairs[0]], u[pairs[1]]
    relation = mlp(relations, state["relation"])
    gate = 1 + np.tanh(state["gate"][buckets])
    z = np.concatenate([left + right, np.abs(left - right), left * right * gate, relation], axis=1)
    pair_value = mlp(z, state["pair"])
    unary = pool(E, batch, n_graphs, zero_count)
    pair_batch = batch[pairs[0]]
    pooled_pair = np.concatenate([
        pool(pair_value[buckets == b], pair_batch[buckets == b], n_graphs, zero_count)
        for b in range(5)
    ], axis=1)
    graph = np.concatenate([unary, pooled_pair, auxiliary], axis=1)
    prediction = (mlp(graph, state["reader"]) @ state["final_w"] + state["final_b"]).ravel()
    if return_aux:
        return prediction, dict(h=h, E=E, alpha=code_aux["alpha"], u=u, pair=pair_value, graph=graph)
    return prediction


def repeated_map(size, multiplier):
    return np.tile(np.arange(size), multiplier)


def expand_linear(w, b, input_map, output_map):
    """Eval-mode Net2Wider embedding: divide replicated incoming weights."""
    multiplicity = np.bincount(input_map, minlength=w.shape[0])
    wide = w[np.ix_(input_map, output_map)] / multiplicity[input_map, None]
    return wide, None if b is None else b[output_map].copy()


def expand_mlp(s, input_map, hidden_map, output_map):
    w1, b1 = expand_linear(s["w1"], s["b1"], input_map, hidden_map)
    w2, b2 = expand_linear(s["w2"], s["b2"], hidden_map, output_map)
    return dict(w1=w1, b1=b1, w2=w2, b2=b2,
                gamma=None if s["gamma"] is None else s["gamma"][hidden_map].copy(),
                beta=None if s["beta"] is None else s["beta"][hidden_map].copy(),
                activation=s["activation"], final_relu=s["final_relu"])


def reader_input_map(multiplier):
    """Map widened sum/squared/bucket blocks back to Small; counts stay scalar."""
    result, offset = [], 0
    for width in [48, 48, 1] + [16, 16, 1] * 5 + [32, 8]:
        copies = multiplier if width in (48, 16) else 1
        result.extend(np.tile(np.arange(offset, offset + width), copies))
        offset += width
    return np.asarray(result, dtype=int)


def embed_small(small, multiplier):
    """Contains-Small witness only. NEVER use this symmetric init for formal fitting."""
    if multiplier not in (2, 3):
        raise ValueError("widening witness expects 2 or 3")
    m = multiplier
    dm, pm = repeated_map(48, m), repeated_map(16, m)
    D, V = np.zeros((48 * m, 96 * m)), np.zeros((96 * m, 48 * m))
    for block in range(m):
        D[48 * block:48 * (block + 1), 96 * block:96 * (block + 1)] = small["D"]
        V[96 * block:96 * (block + 1), 48 * block:48 * (block + 1)] = small["V"]
    proj, _ = expand_linear(small["projection"], None, dm, pm)
    pair_input_map = np.concatenate([pm + 16 * i for i in range(4)])
    rh = repeated_map(13, m)
    fw, fb = expand_linear(small["final_w"], small["final_b"], rh, np.arange(1))
    return dict(
        fusion=expand_mlp(small["fusion"], np.arange(446), repeated_map(114, m), dm),
        D=D, V=V, projection=proj,
        relation=expand_mlp(small["relation"], np.arange(15), repeated_map(32, m), pm),
        gate=small["gate"][:, pm].copy(),
        pair=expand_mlp(small["pair"], pair_input_map, repeated_map(64, m), pm),
        reader=expand_mlp(small["reader"], reader_input_map(m), rh, rh),
        final_w=fw, final_b=fb,
    )


def parameter_arrays(state):
    arrays = []
    for v in state.values():
        if isinstance(v, np.ndarray):
            arrays.append(v)
        elif isinstance(v, dict):
            arrays.extend(x for x in v.values() if isinstance(x, np.ndarray))
    return arrays
