"""Diagnose 8k scaffold-fold real-prototype bank stability without new split evaluation.

Uses only saved official-train outer-fold caches and OOF predictions.  It measures
prototype matching, node/graph coverage drift, prototype-usage drift, and prediction
complementarity across the three frozen label-free real-patch banks.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, spearmanr
from sklearn.metrics import log_loss, roc_auc_score

BANK_SEEDS = (20260728, 20260729, 20260730)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))


def balanced_logloss(y: np.ndarray, prob: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.int64); prob = np.asarray(prob, dtype=np.float64)
    n_pos = max(int(y.sum()), 1); n_neg = max(int((1 - y).sum()), 1)
    w = np.where(y == 1, 0.5 / n_pos, 0.5 / n_neg)
    return float(log_loss(y, prob, sample_weight=w, labels=[0, 1]))


def rows_for_graphs(indices: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(offsets[int(i)], offsets[int(i) + 1], dtype=np.int64) for i in indices])


def assignment_metrics(
    latents: np.ndarray,
    offsets: np.ndarray,
    graph_indices: np.ndarray,
    prototypes: np.ndarray,
    graph_to_scaffold: dict[int, str],
) -> dict:
    rows = rows_for_graphs(graph_indices, offsets)
    z = np.asarray(latents[rows], dtype=np.float32)
    cosine = z @ prototypes.T
    positive = np.maximum(cosine, 0.0)
    top1 = np.argmax(positive, axis=1)
    best = np.max(positive, axis=1)
    top3 = np.partition(positive, -min(3, positive.shape[1]), axis=1)[:, -min(3, positive.shape[1]):]
    mass = top3.sum(axis=1)
    usage = np.bincount(top1, minlength=prototypes.shape[0]).astype(np.float64)
    usage /= max(usage.sum(), 1.0)

    graph_mean = []
    graph_q10 = []
    graph_uncovered25 = []
    graph_uncovered50 = []
    graph_entropy = []
    graph_mass = []
    scaffold_values: dict[str, list[float]] = {}
    cursor = 0
    for raw in graph_indices:
        i = int(raw); n = int(offsets[i + 1] - offsets[i]); sl = slice(cursor, cursor + n); cursor += n
        b = best[sl]; ids = top1[sl]
        graph_mean.append(float(b.mean()))
        graph_q10.append(float(np.quantile(b, 0.10)))
        graph_uncovered25.append(float(np.mean(b < 0.25)))
        graph_uncovered50.append(float(np.mean(b < 0.50)))
        graph_mass.append(float(mass[sl].mean()))
        counts = np.bincount(ids, minlength=prototypes.shape[0]).astype(np.float64)
        counts /= max(counts.sum(), 1.0)
        nz = counts[counts > 0]
        graph_entropy.append(float(-np.sum(nz * np.log(nz)) / np.log(prototypes.shape[0])))
        scaffold_values.setdefault(graph_to_scaffold[i], []).append(float(b.mean()))
    scaffold_means = np.asarray([np.mean(v) for v in scaffold_values.values()], dtype=np.float64)
    graph_mean_arr = np.asarray(graph_mean, dtype=np.float64)
    return {
        'n_graphs': int(len(graph_indices)),
        'n_nodes': int(len(rows)),
        'node_mean_best_positive_cosine': float(best.mean()),
        'node_q10_best_positive_cosine': float(np.quantile(best, 0.10)),
        'node_uncovered_fraction_lt025': float(np.mean(best < 0.25)),
        'node_uncovered_fraction_lt050': float(np.mean(best < 0.50)),
        'mean_top3_positive_mass': float(mass.mean()),
        'active_prototypes': int(np.sum(usage > 0)),
        'usage_entropy_normalized': float(-np.sum(usage[usage > 0] * np.log(usage[usage > 0])) / np.log(len(usage))),
        'usage_distribution': usage.tolist(),
        'graph_mean_best_cosine_mean': float(graph_mean_arr.mean()),
        'graph_mean_best_cosine_std': float(graph_mean_arr.std(ddof=1)),
        'graph_q10_best_cosine_mean': float(np.mean(graph_q10)),
        'graph_uncovered25_mean': float(np.mean(graph_uncovered25)),
        'graph_uncovered50_mean': float(np.mean(graph_uncovered50)),
        'graph_assignment_entropy_mean': float(np.mean(graph_entropy)),
        'graph_top3_mass_mean': float(np.mean(graph_mass)),
        'n_scaffolds': int(len(scaffold_means)),
        'scaffold_unweighted_coverage_mean': float(scaffold_means.mean()),
        'scaffold_unweighted_coverage_std': float(scaffold_means.std(ddof=1)) if len(scaffold_means) > 1 else 0.0,
        'scaffold_unweighted_coverage_q10': float(np.quantile(scaffold_means, 0.10)),
        '_graph_mean': graph_mean_arr,
        '_best': best.astype(np.float64),
    }


def clean_metrics(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith('_')}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--result-dir', default='results/molhiv')
    ap.add_argument('--output', default='results/molhiv/realprototype_bank_stability_diagnostic_scaffold3.json')
    ap.add_argument('--markdown', default='results/molhiv/REALPROTOTYPE_BANK_STABILITY_DIAGNOSTIC_20260728.md')
    args = ap.parse_args()
    root = Path(args.result_dir)
    fold_cache = root / 'molhiv_n8000_scaffold_folds3_seed20260726.npz'
    with np.load(fold_cache, allow_pickle=False) as f:
        official_train = np.asarray(f['official_train_indices'], dtype=np.int64)
        groups = np.asarray(f['train_scaffold_groups']).astype(str)
        group_map = {int(i): str(g) for i, g in zip(official_train, groups)}

    out = {
        'protocol_id': 'molhiv-realprototype-bank-stability-diagnostic-scaffold3-v1',
        'date': '2026-07-28',
        'data_policy': {
            'dataset': '8k stratified subset',
            'selection_data': 'official-train only',
            'outer_validation': 'three Bemis-Murcko scaffold folds',
            'official_valid_evaluations': 0,
            'official_test_evaluations': 0,
            'banks': list(BANK_SEEDS),
        },
        'folds': [],
    }

    for fold in range(3):
        latent_path = root / f'rawpatch_pca64_latents_n8000_scaffoldfit_fold{fold}.npz'
        with np.load(latent_path, allow_pickle=False) as f:
            offsets = np.asarray(f['offsets'], dtype=np.int64)
            latents = np.asarray(f['latents'], dtype=np.float32)
        docs = []
        for seed in BANK_SEEDS:
            p = root / f'stable_fixedrandom_bank{seed}_fold{fold}_modelseed0.json'
            docs.append(json.loads(p.read_text()))
        fit_idx = np.asarray(docs[0]['results']['fixed_random']['fit']['graph_indices'], dtype=np.int64)
        held_idx = np.asarray(docs[0]['results']['fixed_random']['heldout']['graph_indices'], dtype=np.int64)
        labels = np.asarray(docs[0]['results']['fixed_random']['heldout']['labels'], dtype=np.int64)
        if set(fit_idx) & set(held_idx): raise ValueError('outer split overlap')
        if any(int(i) not in group_map for i in np.concatenate([fit_idx, held_idx])): raise ValueError('missing scaffold group')

        bank_rows = []
        bank_metrics = []
        scores = []
        for seed, d in zip(BANK_SEEDS, docs):
            r = d['results']['fixed_random']
            if not np.array_equal(held_idx, np.asarray(r['heldout']['graph_indices'], dtype=np.int64)): raise ValueError('heldout index mismatch')
            if not np.array_equal(labels, np.asarray(r['heldout']['labels'], dtype=np.int64)): raise ValueError('heldout label mismatch')
            source_rows = np.asarray(r['selection']['source_node_rows'], dtype=np.int64)
            prototypes = np.asarray(latents[source_rows], dtype=np.float32)
            prototypes /= np.maximum(np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-12)
            bank_rows.append(prototypes)
            fit_m = assignment_metrics(latents, offsets, fit_idx, prototypes, group_map)
            held_m = assignment_metrics(latents, offsets, held_idx, prototypes, group_map)
            drift = {
                'heldout_minus_fit_node_mean_best_cosine': held_m['node_mean_best_positive_cosine'] - fit_m['node_mean_best_positive_cosine'],
                'heldout_minus_fit_graph_coverage': held_m['graph_mean_best_cosine_mean'] - fit_m['graph_mean_best_cosine_mean'],
                'heldout_minus_fit_scaffold_q10_coverage': held_m['scaffold_unweighted_coverage_q10'] - fit_m['scaffold_unweighted_coverage_q10'],
                'fit_heldout_usage_js_divergence': js_divergence(np.asarray(fit_m['usage_distribution']), np.asarray(held_m['usage_distribution'])),
                'node_best_cosine_ks_statistic': float(ks_2samp(fit_m['_best'], held_m['_best']).statistic),
            }
            logits = np.asarray(r['heldout']['scores'], dtype=np.float64); prob = sigmoid(logits); scores.append(logits)
            bank_metrics.append({
                'prototype_seed': seed,
                'auc': float(roc_auc_score(labels, logits)),
                'balanced_logloss': balanced_logloss(labels, prob),
                'prototype_source_graphs': r['selection']['source_graph_indices'],
                'prototype_source_scaffold_count': int(len({group_map[int(i)] for i in r['selection']['source_graph_indices']})),
                'fit': clean_metrics(fit_m),
                'heldout': clean_metrics(held_m),
                'drift': drift,
                '_fit_graph_coverage': fit_m['_graph_mean'],
                '_held_graph_coverage': held_m['_graph_mean'],
            })

        matching = []
        pred_pairs = []
        for a, b in combinations(range(3), 2):
            cos = bank_rows[a] @ bank_rows[b].T
            rr, cc = linear_sum_assignment(-cos)
            matched = cos[rr, cc]
            la, lb = scores[a], scores[b]
            pa, pb = sigmoid(la), sigmoid(lb)
            matching.append({
                'bank_a': BANK_SEEDS[a], 'bank_b': BANK_SEEDS[b],
                'mean_optimal_matched_cosine': float(matched.mean()),
                'q10_optimal_matched_cosine': float(np.quantile(matched, 0.10)),
                'min_optimal_matched_cosine': float(matched.min()),
                'fraction_matched_cosine_ge080': float(np.mean(matched >= 0.80)),
                'fraction_matched_cosine_ge090': float(np.mean(matched >= 0.90)),
                'source_graph_overlap': int(len(set(bank_metrics[a]['prototype_source_graphs']) & set(bank_metrics[b]['prototype_source_graphs']))),
            })
            pred_pairs.append({
                'bank_a': BANK_SEEDS[a], 'bank_b': BANK_SEEDS[b],
                'logit_pearson': float(np.corrcoef(la, lb)[0, 1]),
                'logit_spearman': float(spearmanr(la, lb).statistic),
                'mean_absolute_probability_difference': float(np.mean(np.abs(pa - pb))),
                'pair_probability_ensemble_auc': float(roc_auc_score(labels, 0.5 * (pa + pb))),
            })

        probs = np.stack([sigmoid(x) for x in scores], axis=0)
        ensemble_prob = probs.mean(axis=0)
        aucs = np.asarray([x['auc'] for x in bank_metrics])
        graph_disagreement = probs.std(axis=0)
        fold_out = {
            'fold': fold,
            'n_fit': int(len(fit_idx)), 'n_heldout': int(len(held_idx)), 'n_heldout_positive': int(labels.sum()),
            'banks': [{k: v for k, v in x.items() if not k.startswith('_')} for x in bank_metrics],
            'bank_matching': matching,
            'prediction_pairing': pred_pairs,
            'ensemble': {
                'auc': float(roc_auc_score(labels, ensemble_prob)),
                'balanced_logloss': balanced_logloss(labels, ensemble_prob),
                'member_auc_mean': float(aucs.mean()),
                'ensemble_minus_member_auc_mean': float(roc_auc_score(labels, ensemble_prob) - aucs.mean()),
                'mean_graph_probability_std': float(graph_disagreement.mean()),
                'positive_graph_probability_std': float(graph_disagreement[labels == 1].mean()),
                'negative_graph_probability_std': float(graph_disagreement[labels == 0].mean()),
            },
        }
        out['folds'].append(fold_out)

    def collect(path):
        vals = []
        for fold in out['folds']:
            cur = fold
            for key in path: cur = cur[key]
            vals.append(float(cur))
        return vals

    out['aggregate'] = {
        'prototype_ensemble_auc_by_fold': collect(['ensemble', 'auc']),
        'prototype_ensemble_auc_mean': float(np.mean(collect(['ensemble', 'auc']))),
        'ensemble_minus_member_auc_mean_by_fold': collect(['ensemble', 'ensemble_minus_member_auc_mean']),
        'ensemble_minus_member_auc_mean': float(np.mean(collect(['ensemble', 'ensemble_minus_member_auc_mean']))),
        'mean_pairwise_matched_cosine': float(np.mean([m['mean_optimal_matched_cosine'] for f in out['folds'] for m in f['bank_matching']])),
        'mean_pairwise_prediction_pearson': float(np.mean([m['logit_pearson'] for f in out['folds'] for m in f['prediction_pairing']])),
        'mean_heldout_minus_fit_coverage': float(np.mean([b['drift']['heldout_minus_fit_graph_coverage'] for f in out['folds'] for b in f['banks']])),
        'mean_fit_heldout_usage_js': float(np.mean([b['drift']['fit_heldout_usage_js_divergence'] for f in out['folds'] for b in f['banks']])),
        'mean_heldout_scaffold_q10_coverage': float(np.mean([b['heldout']['scaffold_unweighted_coverage_q10'] for f in out['folds'] for b in f['banks']])),
    }

    Path(args.output).write_text(json.dumps(out, indent=2))
    a = out['aggregate']
    lines = [
        '# Real-prototype bank stability diagnostic（2026-07-28）', '',
        '仅使用 8k subset 的 official-train 三个 scaffold outer folds；official valid/test evaluation 均为 0。', '',
        '## Aggregate', '',
        f"- prototype ensemble fold AUC: `{a['prototype_ensemble_auc_by_fold']}`；mean `{a['prototype_ensemble_auc_mean']:.6f}`。",
        f"- ensemble 相对单 bank AUC 均值的 fold gains: `{a['ensemble_minus_member_auc_mean_by_fold']}`；mean `{a['ensemble_minus_member_auc_mean']:+.6f}`。",
        f"- bank 间 optimal-matching prototype cosine mean: `{a['mean_pairwise_matched_cosine']:.6f}`。",
        f"- bank prediction pairwise Pearson mean: `{a['mean_pairwise_prediction_pearson']:.6f}`。",
        f"- heldout − fit graph coverage mean: `{a['mean_heldout_minus_fit_coverage']:+.6f}`。",
        f"- fit/heldout prototype-usage JS mean: `{a['mean_fit_heldout_usage_js']:.6f}`。",
        f"- heldout scaffold-unweighted coverage q10 mean: `{a['mean_heldout_scaffold_q10_coverage']:.6f}`。", '',
        '## Fold summary', '',
        '| Fold | Member AUCs | Ensemble AUC | Ens−member mean | Mean graph prob std |',
        '|---:|---|---:|---:|---:|',
    ]
    for f in out['folds']:
        member = '/'.join(f"{b['auc']:.4f}" for b in f['banks'])
        e = f['ensemble']
        lines.append(f"| {f['fold']} | {member} | {e['auc']:.4f} | {e['ensemble_minus_member_auc_mean']:+.4f} | {e['mean_graph_probability_std']:.5f} |")
    lines += ['', '## Interpretation rule', '',
              '- 若 bank matching cosine 高但 predictions 分歧大，主要问题在 downstream optimization。',
              '- 若 bank matching cosine 低且 ensemble gain 大，prototype identity/coverage 是主要方差源。',
              '- 若 heldout coverage 明显下降或 usage JS 很高，应优先做 scaffold-stable vocabulary。',
              '- 若 coverage 稳定但 prediction 分歧仍大，应优先研究 occurrence composition/readout。', '']
    Path(args.markdown).write_text('\n'.join(lines))
    print(json.dumps({'output': args.output, 'markdown': args.markdown, 'aggregate': out['aggregate']}, indent=2))


if __name__ == '__main__':
    main()
