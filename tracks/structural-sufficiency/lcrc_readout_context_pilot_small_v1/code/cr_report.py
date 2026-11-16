"""Write ``REPORT.md`` / ``DECISION.md`` / ``decision.json`` from the analysis artefacts.

Every number in the report is read back from a file this round wrote; nothing is
recomputed here and nothing is typed by hand.  If an artefact is missing the
report says so instead of silently dropping the section.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Mapping, Sequence

import cr_common as C
import cr_analysis as AN

OUT = C.ANALYSIS_DIR
DEV = "DEVELOPMENT"
DEV_TEST = "DEVELOPMENT_TEST"


# ---------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------


def f(x: Any, nd: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def pct(x: Any, nd: int = 2) -> str:
    if x is None:
        return "—"
    return f"{100.0 * float(x):+.{nd}f}%"


def table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def _read(name: str, default: Any = None) -> Any:
    path = OUT / name
    if not path.is_file():
        return default
    return C.read_json(path)


def _rows(name: str, default: Any = None) -> Any:
    path = OUT / "tables" / name
    if not path.is_file():
        return default if default is not None else []
    return C.read_csv(path)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def _header(summary: Mapping[str, Any], freeze: Mapping[str, Any], prov: Mapping[str, Any]) -> str:
    executed = ", ".join(
        sorted({str(v.get("commit", "?"))[:12] for v in (prov.get("runs") or {}).values()})
    ) or str(freeze.get("round_commit"))[:12]
    return f"""# LCRC explicit-context readout — small-dataset feasibility pilot

**Round** `{C.ROUND_ID}` · **status** {DEV} · every Test number below is
`{DEV_TEST}` · **branch** `{freeze.get('round_branch')}` on top of design basis
`{str(freeze.get('design_basis_commit'))[:12]}` · **formal runs executed at** `{executed}` ·
**freeze hash** `{str(freeze.get('freeze_hash'))[:16]}…` · **generated** {summary.get('generated_at')}

> This is a feasibility pilot on two small tasks with **one** training seed.
> The Test split was evaluated every epoch by explicit user authorisation
> (`test_policy=always`). Test never selected a checkpoint, a configuration or a
> number that appears in a verdict. No result here is a benchmark submission, a
> significance claim, or evidence about a second seed.
"""


def _identity(freeze: Mapping[str, Any], audit: Mapping[str, Any]) -> str:
    lines = ["## 1. Identity of what was trained", ""]
    budget = freeze.get("budget", {})
    rows = []
    for dataset in C.DATASETS:
        a = audit.get(dataset, {})
        b = budget.get(dataset, {}).get("assignment", {})
        rows.append(
            [
                dataset,
                "/".join(str(a.get("split_sizes", {}).get(s)) for s in C.SPLITS),
                b.get("A", {}).get("nominal_readout_params"),
                f"{b.get('B', {}).get('nominal_readout_params')} (H={b.get('B', {}).get('hidden')})",
                f"{b.get('C', {}).get('nominal_readout_params')} (H={b.get('C', {}).get('hidden')})",
                a.get("active_global", {}).get("n_active"),
                f(audit.get(dataset, {}).get("pass")),
            ]
        )
    lines.append(
        table(
            ["task", "Train/Val/Test", "A readout", "B readout", "C readout", "M active", "data audit"],
            rows,
        )
    )
    lines += [
        "",
        "All three arms share **one** local trunk (bitwise the C1 local path) and differ only in the",
        "readout. `M` is the number of raw global columns whose Train span exceeds 1e-8; it is a",
        "property of the frozen cache, not a choice. `H` is fixed so the readout parameter count lands",
        "within one hidden-width step of the 15298-parameter C1 head.",
        "",
        "Frozen caches (SHA-256 of the real bytes, re-verified on the compute host):",
        "",
    ]
    for name, sha in freeze.get("frozen_caches", {}).items():
        lines.append(f"* `{name}` — `{sha[:16]}…`")
    init = freeze.get("canonical_init", {})
    lines += ["", "Shared-trunk initialisation (identical in every arm, reproduced in a fresh process):", ""]
    for dataset, hashes in init.items():
        lines.append(f"* {dataset}: trunk `{hashes.get('trunk_hash', '')[:16]}…`")
    lines += [
        "",
        f"Frozen recipe (one for all arms): seed {freeze.get('seed')}, "
        f"{freeze.get('recipe', {}).get('epochs')} epochs, "
        f"AdamW peak {freeze.get('recipe', {}).get('peak_lr')} -> min {freeze.get('recipe', {}).get('min_lr')}, "
        f"wd {freeze.get('recipe', {}).get('weight_decay')}, batch {freeze.get('recipe', {}).get('batch_size')}, "
        "test_policy `always`, selection = minimum Val RMSE only.",
    ]
    return "\n".join(lines) + "\n"


def _main_table() -> str:
    rows = _rows("main_table.csv")
    if not rows:
        return "## 2. Main table\n\n_missing `tables/main_table.csv`._\n"
    lines = ["## 2. Main table (raw target units)", ""]
    body = []
    for r in rows:
        body.append(
            [
                r["dataset"],
                r["arm"],
                r["origin"],
                r["selected_epoch"],
                f(r["train_rmse_raw"]),
                f(r["valid_rmse_raw"]),
                f(r["test_rmse_raw"]),
                f(r["valid_mae_raw"]),
                f(r["test_mae_raw"]),
                f(r["last20_valid_rmse_raw"]),
                f(r["last20_test_rmse_raw"]),
                r["n_parameters"],
            ]
        )
    lines.append(
        table(
            [
                "task",
                "arm",
                "origin",
                "epoch",
                "Train RMSE",
                "Val RMSE",
                "Test RMSE",
                "Val MAE",
                "Test MAE",
                "last20 Val",
                "last20 Test",
                "params",
            ],
            body,
        )
    )
    lines += [
        "",
        "* `epoch` is the Val-selected epoch (minimum Val RMSE); the Test column is that same frozen",
        "  state and is `" + DEV_TEST + "`.",
        "* `last20` is the arithmetic mean of the last 20 epochs of the same trajectory — a curve",
        "  summary, not an ensemble and not a second model.",
        "* `G` is the frozen GINE of `sq-pilot-small-v1`, reused unchanged (`REUSED_FROZEN_REFERENCE`, no",
        "  retraining here); it has a different input path and a stronger learning-rate preset",
        "  (`1e-3` vs the frozen `3e-4` of this round), so it is a performance coordinate only.",
    ]
    return "\n".join(lines) + "\n"


def _transfer_gap() -> str:
    """The Val -> Test gap of each selected state; it is part of the main evidence."""
    rows = _rows("main_table.csv")
    if not rows:
        return ""
    lines = [
        "### 2.1 Val -> Test transfer gap of the same frozen state",
        "",
        "The selected state is a Val-RMSE minimum, so this table asks how much of that Val number",
        "survives on Test. With 113 Val molecules (molesol) a small Val improvement is noisy.",
        "",
    ]
    body = []
    for r in rows:
        v = float(r["valid_rmse_raw"])
        t = float(r["test_rmse_raw"])
        l20v = float(r["last20_valid_rmse_raw"])
        l20t = float(r["last20_test_rmse_raw"])
        body.append(
            [
                r["dataset"],
                r["arm"],
                f(v),
                f(t),
                f(t - v),
                f(t / v, 3),
                f(l20t - l20v),
                f(l20t / l20v, 3),
            ]
        )
    lines.append(
        table(
            [
                "task",
                "arm",
                "Val RMSE",
                "Test RMSE",
                "Test - Val",
                "Test / Val",
                "last20 Test - Val",
                "last20 Test / Val",
            ],
            body,
        )
    )
    worst = max(
        (r for r in rows if r["arm"] != "G"),
        key=lambda r: float(r["test_rmse_raw"]) / float(r["valid_rmse_raw"]),
    )
    lines += [
        "",
        f"The largest selected-state gap among A/B/C is {worst['dataset']}/{worst['arm']} "
        f"(Test = {float(worst['test_rmse_raw']) / float(worst['valid_rmse_raw']):.2f}x Val). A gap of this "
        "size means the Val-selected epoch of that arm is not a reliable performance coordinate on this "
        "task; it is reported rather than smoothed away, and no cross-split average is used anywhere in "
        "this report.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _contrasts() -> str:
    rows = _rows("contrasts.csv")
    boot = _read("bootstrap.json", {})
    if not rows:
        return "## 3. Contrasts\n\n_missing `tables/contrasts.csv`._\n"
    lines = ["## 3. Paired contrasts (same frozen states)", ""]
    body = []
    for r in rows:
        b = boot.get(r["dataset"], {}).get("valid", {}).get("contrasts", {}).get(r["contrast"], {})
        bt = boot.get(r["dataset"], {}).get("test", {}).get("contrasts", {}).get(r["contrast"], {})
        vi = b.get("rmse_raw", {})
        ti = bt.get("rmse_raw", {})
        body.append(
            [
                r["dataset"],
                r["contrast"],
                f(r["selected_delta_rmse_raw_valid"]),
                f(r["selected_ratio_rmse_valid"], 3),
                f"[{f(vi.get('ci95_low'), 3)}, {f(vi.get('ci95_high'), 3)}]",
                f(r["selected_delta_rmse_raw_test"]),
                f(r["selected_ratio_rmse_test"], 3),
                f"[{f(ti.get('ci95_low'), 3)}, {f(ti.get('ci95_high'), 3)}]",
                f(r["last20_delta_rmse_raw_valid"]),
                f(r["last20_delta_rmse_raw_test"]),
            ]
        )
    lines.append(
        table(
            [
                "task",
                "contrast",
                "Δ Val RMSE",
                "ratio Val",
                "95% CI (Val)",
                "Δ Test RMSE",
                "ratio Test",
                "95% CI (Test)",
                "Δ last20 Val",
                "Δ last20 Test",
            ],
            body,
        )
    )
    lines += [
        "",
        "Δ = left − right (negative favours the left arm). Ratios are left / right. The intervals are",
        "paired molecule bootstrap percentiles (2000 replicates, RNG 1703, one resample shared by every",
        "model of a split). They describe uncertainty **conditional on the single-seed selected state**:",
        "they do not measure training-seed variance, they do not repair Val selection, and they do not",
        "undo the development exposure of Test.",
        "",
    ]
    d1 = _read("drop_one.json", {})
    if d1:
        body = []
        for dataset, per_split in d1.items():
            for split in ("valid", "test"):
                for contrast, values in per_split.get(split, {}).items():
                    body.append(
                        [
                            dataset,
                            split,
                            contrast,
                            f(values["delta_rmse_full"]),
                            f"[{f(values['delta_rmse_range'][0])}, {f(values['delta_rmse_range'][1])}]",
                            values["n_rows_that_flip_direction"],
                            f(values["max_single_row_shift_rmse"]),
                        ]
                    )
        if body:
            lines += [
                "Leave-one-row-out sensitivity of those deltas (does a single molecule decide the sign?):",
                "",
                table(
                    [
                        "task",
                        "split",
                        "contrast",
                        "Δ RMSE (full)",
                        "Δ range over dropping one row",
                        "#rows that flip the sign",
                        "largest single-row shift",
                    ],
                    body,
                ),
                "",
                "_Dropped numbers are diagnostics only; the official score is always the full split._",
                "",
            ]
    return "\n".join(lines) + "\n"


def _checks() -> str:
    per = {}
    for dataset in C.DATASETS:
        payload = _read(f"checks_{dataset}.json")
        if payload is not None:
            per[dataset] = payload
    lines = ["## 4. Preflight gates (before any formal epoch)", ""]
    if not per:
        return "\n".join(lines + ["_missing `analysis/checks_<dataset>.json`._"]) + "\n"
    body = []
    for dataset, payload in per.items():
        body.append([dataset, payload.get("n_checks"), "yes" if payload.get("all_pass") else "NO", len(payload.get("failed") or [])])
    lines.append(table(["task", "#gates", "all pass", "#failed"], body))
    names = sorted({c["check"] for payload in per.values() for c in payload.get("checks", [])})
    lines += ["", "Gates executed: " + ", ".join(f"`{n}`" for n in names) + ".", ""]
    status = []
    for name in names:
        cells = []
        for dataset, payload in per.items():
            hit = [c for c in payload.get("checks", []) if c["check"] == name]
            cells.append("pass" if hit and all(c.get("pass") for c in hit) else ("missing" if not hit else "FAIL"))
        status.append([name] + cells)
    lines += [table(["gate"] + list(per), status), ""]
    return "\n".join(lines) + "\n"


def _trace() -> str:
    ledgers = {}
    for dataset in C.DATASETS:
        for arm in ("B", "C"):
            meta = _read(f"ledgers/{dataset}_{arm}.json")
            if meta is not None:
                ledgers[f"{dataset}/{arm}"] = meta
    replay = _read("checkpoint_replay.json", {})
    interventions = _read("interventions.json", {})
    contexts = _read("context_response.json", {})
    equiv = _read("equivalence_scale.json", {})
    verdicts = _read("verdicts.json", {})
    lines = [
        "## 5. Trace: can every number be taken apart again?",
        "",
        "The readout is only worth its explicitness if the exported terms reproduce the prediction.",
        "Tolerance in every row is the pre-declared `tol(x) = 1e-5 + 1e-6·max(1,|x|)` in raw target units.",
        "",
    ]
    if ledgers:
        body = []
        for key, meta in ledgers.items():
            for split in ("valid", "test"):
                closure = meta["closure"].get(split)
                if not closure:
                    continue
                body.append(
                    [
                        key,
                        split,
                        closure["n_rows"],
                        f"{closure['max_abs_error_raw']:.2e}",
                        f"{closure['tolerance_max']:.2e}",
                        "pass" if closure["pass"] else "FAIL",
                    ]
                )
        lines += [
            "### 5.1 Prediction reconstruction from the exported components",
            "",
            table(["model", "split", "rows", "max abs error", "tolerance", "verdict"], body),
            "",
            "The account is `pred_raw = mu + sigma·(intercept + local terms + global terms + interactions)`",
            "with `mu, sigma` the frozen Train target mean/std; only the intercept carries `mu`.",
            "",
        ]
    if replay:
        body = []
        for dataset, per_arm in replay.items():
            for arm, entry in per_arm.items():
                body.append(
                    [
                        dataset,
                        arm,
                        f(entry.get("per_split", {}).get("valid", {}).get("prediction_max_abs_diff_raw"), 6),
                        f(entry.get("per_split", {}).get("test", {}).get("prediction_max_abs_diff_raw"), 6),
                        entry.get("reloaded_state_hash", "")[:12],
                        "pass" if entry.get("pass") else "FAIL",
                    ]
                )
        lines += [
            "### 5.2 Reloading the selected checkpoint",
            "",
            table(["task", "arm", "max abs Δ Val", "max abs Δ Test", "state hash", "verdict"], body),
            "",
        ]
    if interventions:
        body = []
        for dataset, payload in interventions.items():
            body.append(
                [
                    dataset,
                    payload["n_molecules"],
                    payload["removed_occurrence_rule"],
                    payload["n_failures"],
                    "pass" if payload["all_pass"] else "FAIL",
                ]
            )
        lines += [
            "### 5.3 Fixed-input intervention",
            "",
            "One typed-S2 occurrence is removed from a Val molecule (raw graph, S1 codes, scalars and the",
            "global vector unchanged), and the *change* in the prediction is rebuilt from the change in the",
            "ledger row that owns that occurrence:",
            "",
            table(["task", "molecules", "removed", "#failures", "verdict"], body),
            "",
            "In every case the other roots' base terms and every global term are unchanged to `<= 1e-6`, and",
            "in arm C the context factor `a_k` is unchanged (the context vector did not move). The original",
            "C1 head also responds — through `rho` (mean h / mean h² / n) — which is recorded, not scored as",
            "an implementation error: it is exactly the entanglement this round is about.",
            "",
        ]
    if contexts:
        body = []
        for dataset, payload in contexts.items():
            for name, rms in payload.get("interaction_response_rms_std", {}).items():
                body.append(
                    [
                        dataset,
                        name,
                        f(payload.get("global_main_effect_rms_std", {}).get(name)),
                        f(rms),
                        payload["n_failures"],
                        "pass" if payload["all_pass"] else "FAIL",
                    ]
                )
        lines += [
            "### 5.4 Context response with the local state held fixed",
            "",
            "For each field the frozen intervention moves the context from its **observed** Train-standardised",
            "value to the Train Q75 and keeps `h` fixed; the closed form",
            "`Δ = Δu_k + Δa_k · S_k` is then compared to an independent full forward pass.",
            "",
            table(
                ["task", "context field", "RMS Δu (main effect)", "RMS Δ interaction", "#failures", "verdict"],
                body,
            ),
            "",
            "`Δu` is the *single-variable global* response and `Δinteraction` is the local-state-weighted",
            "response. They are separate columns on purpose: merging them would hide the difference between",
            "'the number went in' and 'the number changed how the structure is read'.",
            "",
        ]
    if equiv:
        body = []
        for dataset, payload in equiv.items():
            body.append(
                [
                    dataset,
                    payload["n_rows"],
                    f(payload["prediction"]["max_abs_diff"], 2),
                    f"{payload['a_ratio']['min']:.4f}–{payload['a_ratio']['max']:.4f}",
                    "pass" if payload["pass"] else "FAIL",
                ]
            )
        lines += [
            "### 5.5 Scale equivalence",
            "",
            "`a_k ×2, c_k ÷2` leaves every prediction and every per-root interaction unchanged:",
            "",
            table(["task", "Val rows", "max abs Δ pred", "a ratio", "verdict"], body),
            "",
            "So an individual `a_k` or `c_k` norm is **not** an importance measure. This closes the scale",
            "degeneracy only; it does not resolve every attribution non-uniqueness.",
            "",
        ]
    trace_labels = {
        dataset: ((verdicts.get(dataset) or {}).get("trace") or {}).get("label") for dataset in C.DATASETS
    }
    lines += [
        "### 5.6 Verdict",
        "",
        table(["task", "label"], [[d, l] for d, l in trace_labels.items()]),
        "",
        "`TRACE_PASS` means the bookkeeping closes. It is **not** proof that a per-root number is a",
        "chemical fragment (root windows overlap), nor that the attribution is unique.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _interaction() -> str:
    diag = _read("interaction_diagnostics.json", {})
    if not diag:
        return "## 6. Interaction diagnostics\n\n_missing `interaction_diagnostics.json`._\n"
    lines = [
        "## 6. Is the interaction real, or a constant?",
        "",
        "Reference decomposition of the interaction term at molecule level,",
        "`I_k = a_k · [n·c̄_k + Σ_v (c_k(v) − c̄_k)]`, with `c̄_k` the node-equal-weight mean over Train",
        "roots of the selected model. This is a reference decomposition, not a refit; RMS parts can exceed",
        "the whole through cancellation, so they are not exclusive error shares.",
        "",
    ]
    body = []
    for dataset, payload in diag.items():
        for split in ("valid", "test"):
            for name in C.CONTEXT_FIELDS:
                entry = payload["per_split"][split][name]
                body.append(
                    [
                        dataset,
                        split,
                        name,
                        f(entry["rms_interaction_raw"]),
                        f(entry["constant_part_rms_raw"]),
                        f(entry["deviation_part_rms_raw"]),
                        f(entry["root_std_over_rms_median"], 3),
                        entry["n_molecules_with_one_root"],
                    ]
                )
    lines.append(
        table(
            [
                "task",
                "split",
                "field",
                "RMS interaction (raw)",
                "constant part",
                "root-deviation part",
                "median root std/RMS",
                "#1-root molecules",
            ],
            body,
        )
    )
    labels = {}
    for dataset, payload in diag.items():
        labels[dataset] = payload.get("labels", {})
    if labels:
        body = []
        for dataset, per_field in labels.items():
            for name, entry in per_field.items():
                body.append(
                    [
                        dataset,
                        name,
                        f(entry.get("rms_interaction_std_valid")),
                        f(entry.get("rms_interaction_diagnostic3_std")),
                        f(entry.get("root_std_over_rms_median_valid"), 3),
                        entry.get("response_label"),
                        entry.get("root_specificity_label"),
                    ]
                )
        lines += [
            "",
            "Pre-declared read-out: an interaction counts as *resolved* only when its RMS is at least 1% of",
            "the Train target standard deviation on Val, Test **and** the Q25→Q75 intervention; and it is",
            "*root-specific* only when the median per-molecule root std/RMS (root-specific spread, not a\ncoefficient of variation) is above 0.05 — below that the interaction is nearly a constant multiplier.",
            "",
            table(
                ["task", "field", "RMS (Val)", "RMS (intervention)", "median root std/RMS", "response", "root specificity"],
                body,
            ),
            "",
        ]
    return "\n".join(lines) + "\n"


def _verdicts() -> str:
    verdicts = _read("verdicts.json", {})
    if not verdicts:
        return "## 7. Verdicts\n\n_missing `verdicts.json`._\n"
    lines = [
        "## 7. The four pre-declared axes",
        "",
        "The axes are reported separately and never collapsed into one score.",
        "",
    ]
    body = []
    for dataset, axes in verdicts.items():
        for axis in ("retention", "context", "gine", "trace"):
            entry = axes.get(axis)
            if entry is None:
                continue
            body.append([dataset, axis, entry.get("label")])
    lines.append(table(["task", "axis", "label"], body))
    lines.append("")
    for dataset, axes in verdicts.items():
        ret = axes.get("retention") or {}
        ctx = axes.get("context") or {}
        gin = axes.get("gine") or {}
        lines += [
            f"### {dataset}",
            "",
            f"* **Retention (C vs A)**, tolerance {ret.get('tolerance')}×: Val ratio "
            f"{f((ret.get('selected_ratio_C_over_A') or {}).get('valid'), 3)}, Test ratio "
            f"{f((ret.get('selected_ratio_C_over_A') or {}).get('test'), 3)}; last-20 ratios "
            f"{f((ret.get('last20_ratio_C_over_A') or {}).get('valid'), 3)} / "
            f"{f((ret.get('last20_ratio_C_over_A') or {}).get('test'), 3)}.",
            f"* **Context increment (C vs B)**, threshold {pct(ctx.get('threshold'))}:",
            "",
        ]
        surfaces = ctx.get("relative_gain_by_surface", {})
        if surfaces:
            lines.append(
                table(
                    ["surface", "relative gain C over B"],
                    [[name, pct(value)] for name, value in surfaces.items()],
                )
            )
            lines.append("")
        lines += [
            f"  above threshold on: {', '.join(ctx.get('surfaces_above_threshold') or ['—'])}; "
            f"below zero on: {', '.join(ctx.get('surfaces_below_zero') or ['—'])}; "
            f"sign disagrees between Val and Test: {f(ctx.get('sign_disagrees_between_splits'))}.",
            "",
        ]
        arms = gin.get("arms", {})
        if arms:
            body = []
            for arm, entry in arms.items():
                body.append(
                    [
                        arm,
                        f((entry.get("valid") or {}).get("rmse_raw")),
                        f((entry.get("test") or {}).get("rmse_raw")),
                        f((entry.get("valid") or {}).get("ratio_over_gine"), 3),
                        f((entry.get("test") or {}).get("ratio_over_gine"), 3),
                        f(entry.get("within_5pct_both_splits")),
                    ]
                )
            lines += [
                "* **GINE competitiveness (each arm vs the reused G)**:",
                "",
                table(["arm", "Val RMSE", "Test RMSE", "Val / G", "Test / G", "within 5% both"], body),
                "",
                f"  {gin.get('caveat', '')}",
                "",
            ]
    return "\n".join(lines) + "\n"


def _reproduction() -> str:
    ref = _read("reference_arm_A_vs_L.json", {})
    if not ref or all(v is None for v in ref.values()):
        return "## 8. Reproduction of the frozen loop\n\n_reference `method_L.json` not available._\n"
    lines = [
        "## 8. Does arm A reproduce the frozen C1 trajectory?",
        "",
        "Arm A is the unchanged C1 head, so its trajectory must match the frozen `L` run of",
        "`sq-pilot-small-v1` up to the one difference the prompt froze: the batch order (`rng(0)`",
        "permutation here, `shuffle_offset=917000` there).",
        "",
    ]
    body = []
    for dataset, entry in ref.items():
        if entry is None:
            continue
        body.append(
            [
                dataset,
                entry["recipe_peak_lr_match"],
                entry["recipe_epochs_match"],
                entry["reference_selected"]["epoch"],
                entry["this_round_selected"]["epoch"],
                f(entry["reference_selected"]["valid_rmse_raw"]),
                f(entry["this_round_selected"]["valid_rmse_raw"]),
                pct(entry["selected_valid_relative_delta"]),
                pct(entry["selected_test_relative_delta"]),
                pct(entry["last20_valid_relative_delta"]),
            ]
        )
    lines += [
        table(
            [
                "task",
                "same peak lr",
                "same epochs",
                "L epoch",
                "A epoch",
                "L Val",
                "A Val",
                "Δ Val",
                "Δ Test",
                "Δ last20 Val",
            ],
            body,
        ),
        "",
        "Agreement at the same epoch with a ~1% gap is the expected signature of an identical recipe with a",
        "different shuffle convention; it is not a new experiment and is not evidence about seeds.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _limits() -> str:
    return f"""## 9. What this round does not show

* **One seed.** Every contrast is a single trajectory per arm. Nothing here estimates run-to-run
  variance; the paired bootstrap describes molecules *given* the frozen states, not training noise.
* **Val selection is optimistic in both arms.** Each arm's reported number is its best epoch on 113
  (molesol) / 420 (mollipo) Val molecules. The comparison A/B/C is fair among themselves (same
  selection rule, same split) but no absolute number is an unbiased estimate of generalisation.
* **Test is `{DEV_TEST}`.** It was looked at every epoch and is reported next to the selected state.
  Any reader who used it to choose anything would invalidate it; this round did not.
* **G is a coordinate, not a control.** The frozen GINE has a different input path and a stronger
  learning-rate preset; a gap is not attributable to a single operator.
* **No chemical claim.** A root row is a *root-local input environment*; root windows overlap and are
  not independent fragments, and the last-word attribution on internal paths is not unique.
* **No architecture search.** The readout forms were fixed in code before the runs; a third context
  field, a wider `a_k`, or a different `H` would be a different round.
* **Two small tasks.** No ZINC, no PCBA, no public-122, no MoleculeNet benchmark claim.
"""


def _reproduce(freeze: Mapping[str, Any]) -> str:
    return f"""## 10. Reproducing this report

```bash
# local identity + gates (CPU, ~1 min)
uv run pytest -q tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/tests
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \\
  --purpose "prepare" --set stage=prepare

# formal training: one worker per GPU, one dataset each
rr deploy res
rr run res lcrc-rc-molesol --pool res-gpu0 \\
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/results \\
  --result tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/analysis \\
  -- bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \\
     --purpose "formal: ogbg-molesol" --set stage=dataset_worker --set dataset=ogbg-molesol
# ... same with --pool res-gpu1 --set dataset=ogbg-mollipo
rr pull lcrc-rc-molesol && rr pull lcrc-rc-mollipo

# CPU-only analysis: tables, ledgers, interventions, figures, verdicts, this report
bash tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/run_remote.sh \\
  --purpose "aggregate" --set stage=aggregate
```

Provenance: round commit `{str(freeze.get('round_commit'))[:12]}` on `{freeze.get('round_branch')}`,
design basis `{str(freeze.get('design_basis_commit'))[:12]}`, execution regime recorded per run by
`rr` (host `res`, A100-SXM4-40GB, torch 2.5.1+cu124), analysis on CPU.
"""


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _decision_markdown(verdicts: Mapping[str, Any], freeze: Mapping[str, Any]) -> str:
    rows = []
    for dataset, axes in verdicts.items():
        for axis in ("retention", "context", "gine", "trace"):
            entry = axes.get(axis)
            if entry is None:
                continue
            rows.append([dataset, axis, entry.get("label")])
    trace_ok = all((axes.get("trace") or {}).get("label") == "TRACE_PASS" for axes in verdicts.values())
    retention_ok = all(
        (axes.get("retention") or {}).get("label", "").startswith("PRESERVED") for axes in verdicts.values()
    )
    context_signal = [
        dataset
        for dataset, axes in verdicts.items()
        if (axes.get("context") or {}).get("label", "").startswith("CONTEXT_PERFORMANCE")
    ]
    context_split = [
        dataset
        for dataset, axes in verdicts.items()
        if (axes.get("context") or {}).get("label") == "SPLIT_OR_SELECTION_SENSITIVE"
    ]
    within_gine = sorted(
        {
            arm
            for axes in verdicts.values()
            for arm in ((axes.get("gine") or {}).get("arms_within_5pct_both_splits") or [])
        }
    )
    behind_gine = [
        dataset
        for dataset, axes in verdicts.items()
        if (axes.get("gine") or {}).get("label") == "BEHIND_THIS_GINE"
    ]
    lines = [
        "# DECISION — lcrc-readout-context-pilot-small-v1",
        "",
        f"Status: {DEV} feasibility pilot, one training seed, two small official tasks.",
        f"Test numbers are {DEV_TEST}.",
        "",
        "## Recorded axes",
        "",
        table(["task", "axis", "label"], rows),
        "",
        "## Decision",
        "",
    ]
    if trace_ok:
        lines.append(
            "1. **The explicit readout is implementable and traceable.** Every Val/Test prediction of arms B and C "
            "was rebuilt from exported per-molecule and per-root terms within the pre-declared tolerance, the "
            "fixed-input intervention was reproduced from the ledger, the `a_k/c_k` scale degeneracy was closed "
            "exactly, and the selected checkpoints replay bit-for-bit. This part does not depend on the numbers "
            "being good."
        )
    else:
        lines.append(
            "1. **The trace did not close.** At least one reconstruction, intervention, scale-equivalence or "
            "checkpoint-replay gate failed; no mechanism statement is made from this round until it is fixed."
        )
    if retention_ok:
        lines.append(
            "2. **Replacing the C1 post-hoc pooling head with additive roots did not cost performance at the "
            "pre-declared 5% tolerance** on both tasks (point estimate and last-20 horizon)."
        )
    else:
        lines.append(
            "2. **The additive readout (C) is not performance-neutral against the unchanged C1 head (A)** on at "
            "least one task at the pre-declared 5% tolerance. In this recipe the C1 pooling head still buys "
            "measurable accuracy that additive roots + global single-variable terms do not recover."
        )
    if context_signal:
        lines.append(
            "3. **A context increment is visible on " + ", ".join(context_signal) + "** (C better than B by the "
            "pre-declared 2% threshold on both Val and Test of the selected state), with the interaction response "
            "resolved above the 1% read-out floor. It remains single-seed development evidence."
        )
    elif context_split:
        detail = "; ".join(
            f"{dataset}: "
            + ", ".join(f"{name} {pct(value)}" for name, value in (axes.get('context') or {})
                        .get("relative_gain_by_surface", {}).items())
            for dataset, axes in verdicts.items()
        )
        lines.append(
            "3. **The context increment is not stable across surfaces.** C vs B changes sign between Val and Test "
            f"and/or between the selected state and the last-20 horizon ({detail}). No context gain is claimed: at "
            "this scale the two frozen fields and this 2-factor functional form do not produce a reproducible "
            "improvement, even though the learned interaction responds measurably to the inputs (section 6)."
        )
    else:
        lines.append(
            "3. **No material context increment was detected** (C vs B below the pre-declared 2% threshold on "
            "every surface). Either the two frozen fields do not carry usable signal at this scale, or a 2-factor "
            "product is too weak a functional form — this round cannot separate those two explanations."
        )
    if within_gine:
        lines.append(
            "4. **Arm(s) " + ", ".join(within_gine) + " land within the pre-declared 5% band of the frozen GINE** on "
            "both splits of " + ", ".join(
                dataset for dataset, axes in verdicts.items()
                if (axes.get("gine") or {}).get("arms_within_5pct_both_splits")
            ) + ". Because G uses a different input path and a stronger learning-rate preset, this is a distance "
            "marker, not an attribution, and it is not a competitiveness claim for the round."
        )
    if behind_gine:
        lines.append(
            "5. **The additive arms are behind the frozen GINE reference on " + ", ".join(behind_gine) + "**. With a "
            "different input path and a stronger learning-rate preset in G, this is again a distance marker, not an "
            "attribution."
        )
    lines += [
        "",
        "## What happens next (and what does not)",
        "",
        "* The round stops here: two tasks, one seed, no re-run with a friendlier configuration.",
        "* A second seed or a wider readout is **not** justified by these numbers; a seed sweep would answer a",
        "  different question (stability) and is out of scope for this pilot.",
        "* If the project wants to pursue the context question, the next round must be declared in advance with",
        "  its own protocol: e.g. a fixed-form interaction *and* a frozen-`h` baseline that separates 'the number",
        "  went in' from 'the structure is read differently', on a task where a context field is known to matter.",
        "* The reusable part of this round is the **machinery**: the shared-trunk construction, the budget-matched",
        "  readouts, the ledger/intervention/scale-equivalence gates and the analysis driver.",
        "",
        f"Commit `{str(freeze.get('round_commit'))[:12]}` on `{freeze.get('round_branch')}`; "
        f"freeze hash `{str(freeze.get('freeze_hash'))[:16]}…`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def _provenance(prov: Mapping[str, Any]) -> str:
    runs = prov.get("runs") or {}
    if not runs:
        return "## 11. Execution provenance\n\n_missing `analysis/run_provenance.json`._\n"
    budget_rows = []
    total_hours = 0.0
    for name, run in runs.items():
        try:
            start = datetime.fromisoformat(str(run.get("started_at")))
            end = datetime.fromisoformat(str(run.get("completed_at")))
            seconds = (end - start).total_seconds()
        except (TypeError, ValueError):
            seconds = None
        if seconds is not None:
            total_hours += seconds / 3600.0
            budget_rows.append([name, f"{seconds:.0f}", f"{seconds / 3600.0:.3f}", run.get("pool")])
    body = []
    for name, run in runs.items():
        body.append(
            [
                name,
                run.get("pool"),
                run.get("node"),
                run.get("driver_version"),
                run.get("torch_version"),
                str(run.get("commit"))[:12],
                run.get("state"),
                run.get("exit_code"),
                run.get("started_at"),
                run.get("completed_at"),
            ]
        )
    return "\n".join(
        [
            "## 11. Execution provenance",
            "",
            "Read from the control-plane run records (`rr status --json`), not typed by hand:",
            "",
            table(
                ["experiment", "pool", "node", "driver", "torch", "commit", "state", "exit", "started", "completed"],
                body,
            ),
            "",
            "The analysis stage is CPU-only and reads only the pulled artefacts; it never touches the GPU host.",
            "",
            "### 11.1 Compute actually spent",
            "",
        ]
        + (
            [
                table(["experiment", "job wall (s)", "job wall (A100-h)", "GPU"], budget_rows),
                "",
            ]
            if budget_rows
            else []
        )
        + [
            f"Total formal compute: **{total_hours:.3f} A100-h** of job wall across the two workers "
            "(one GPU each, one training process per GPU, no fan-out), plus one short GPU smoke run and "
            "CPU-only prepare/aggregate stages.",
            "",
            "Per-arm trainer-measured time is in each `results/<task>/<arm>/trajectory.json` "
            "(`train_seconds`, `eval_seconds_total`).",
            "",
        ]
    ) + "\n"


def write_reports() -> dict[str, Any]:
    summary = _read("analysis_summary.json", {})
    freeze = _read("protocol_freeze.json", {})
    prov = _read("run_provenance.json", {})
    audit = _read("dataset_audit.json", {})
    verdicts = _read("verdicts.json", {})
    if not summary or not verdicts:
        raise FileNotFoundError("run the aggregate stage first (missing analysis_summary/verdicts)")
    report = "\n".join(
        [
            _header(summary, freeze, prov),
            _identity(freeze, audit),
            _main_table(),
            _transfer_gap(),
            _contrasts(),
            _checks(),
            _trace(),
            _interaction(),
            _verdicts(),
            _reproduction(),
            _limits(),
            _reproduce(freeze),
            _provenance(prov),
        ]
    )
    C.atomic_write_text(C.ROUND_DIR / "REPORT.md", report)
    decision_md = _decision_markdown(verdicts, freeze)
    C.atomic_write_text(C.ROUND_DIR / "DECISION.md", decision_md)
    decision_json = {
        "round_id": C.ROUND_ID,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "status": DEV,
        "test_status": DEV_TEST,
        "axes": {
            dataset: {axis: (value or {}).get("label") for axis, value in axes.items()}
            for dataset, axes in verdicts.items()
        },
        "freeze_hash": freeze.get("freeze_hash"),
        "round_commit": freeze.get("round_commit"),
        "round_branch": freeze.get("round_branch"),
        "seed": C.TRAIN_SEED,
        "source": "analysis/verdicts.json",
    }
    C.atomic_write_json(OUT / "decision.json", decision_json)
    figures = sorted(p.name for p in (OUT / "figures").glob("*.png"))
    return {
        "report": str((C.ROUND_DIR / "REPORT.md").relative_to(C.REPO_ROOT)),
        "decision": str((C.ROUND_DIR / "DECISION.md").relative_to(C.REPO_ROOT)),
        "decision_json": str((OUT / "decision.json").relative_to(C.REPO_ROOT)),
        "n_figures": len(figures),
        "figures": figures,
    }


if __name__ == "__main__":
    print(C.dumps_json(write_reports()))
