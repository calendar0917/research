"""Write a compact, restart-safe report for the luyin16 ZINC suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _load(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def summarize(suite_dir: Path, result_path: Path) -> dict[str, Any]:
    suite_dir.mkdir(parents=True, exist_ok=True)
    result_files = sorted(suite_dir.glob("zinc_*.json"))
    payloads = {path.stem: _load(path) for path in result_files}
    payloads = {name: payload for name, payload in payloads.items() if payload is not None}
    radius_payloads = {
        name: payload
        for name, payload in payloads.items()
        if name.startswith("zinc_radius") and "preflight" not in name and "smoke" not in name
    }
    lines = [
        "# luyin16 ZINC 长程与归因 suite",
        "",
        "本报告由 suite 的最后一个任务自动生成。所有 radius 任务使用同一套冻结 patch 对象，",
        "字典只在对应 official train patches 上学习；test 仅对晋级视图做 train+valid refit。",
        "",
        "## 数据源",
        "",
    ]
    preflight = payloads.get("zinc_preflight")
    if preflight:
        source = preflight.get("source", {})
        lines.extend(
            [
                f"- 状态：`{preflight.get('status', 'unknown')}`；split sizes：`{preflight.get('sizes', {})}`",
                f"- policy：`{source.get('policy', 'unknown')}`",
                f"- dataset URL：`{source.get('dataset_url', 'unknown')}`",
                f"- split URL：`{source.get('split_url', 'unknown')}`",
                f"- raw 文件数：`{len(source.get('raw_files', []))}`（详见 preflight JSON）",
            ]
        )
    else:
        lines.append("- 尚未得到 preflight JSON。")
    lines.extend(["", "## Radius 诊断", "", "| run | status | train truncation | train pair coverage | best validation views |", "|---|---|---:|---:|---|"])
    radius_rows = []
    for name, payload in sorted(radius_payloads.items()):
        sampling = payload.get("sampling", {}).get("split_summaries", {}).get("train", {})
        views = payload.get("views", {})
        ranked = sorted(
            ((view, row.get("valid_mae_mean")) for view, row in views.items() if row.get("valid_mae_mean") is not None),
            key=lambda item: item[1],
        )
        best = ", ".join(f"`{view}` ({_fmt(mae)})" for view, mae in ranked[:5]) or "—"
        radius_rows.append(
            {
                "run": name,
                "status": payload.get("status"),
                "train_truncation": sampling.get("mean_truncation_rate"),
                "train_pair_coverage": sampling.get("mean_pair_coverage"),
                "best_validation": ranked[:10],
                "optuna": payload.get("optuna", {}),
            }
        )
        lines.append(
            f"| `{name}` | `{payload.get('status', 'unknown')}` | "
            f"{_fmt(sampling.get('mean_truncation_rate'))} | "
            f"{_fmt(sampling.get('mean_pair_coverage'))} | {best} |"
        )
    if not radius_rows:
        lines.append("| — | no result yet | — | — | — |")
    lines.extend(["", "## 固定预算视图对照", "", "| run | global_all | local typed raw | typed KSVD init | typed KSVD final | global + typed final |", "|---|---:|---:|---:|---:|---:|"])
    key_views = (
        "global_all",
        "local_typed_raw",
        "local_typed_ksvd_init",
        "local_typed_ksvd_final",
        "global_all_plus_typed_ksvd_final",
    )
    for name, payload in sorted(radius_payloads.items()):
        values = []
        for view in key_views:
            row = payload.get("views", {}).get(view, {})
            values.append(_fmt(row.get("valid_mae_mean")))
        lines.append(f"| `{name}` | " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## 解释规则",
            "",
            "- radius-1/2/3 固定 `max_nodes=12`，因此可直接比较感受野；wide-development 不与 full 数字混表。",
            "- radius-3-wide-full 与 radius-3-full 使用同一 official split 和训练预算，仅改变 `max_nodes: 12→20`，用于归因截断。",
            "- `local_*_raw` 检查统计对象；`*_ksvd_init` 与 `*_ksvd_final` 在相同初始化下检查 K-SVD 更新归因。",
            "- `global_structure` 含全局距离/直径等长程统计；`global_attributes` 与局部 attributes 用于属性—结构解耦。",
            "- screen-only 视图只做一次固定参数 validation 快筛；晋级视图才做多 seed 和 test。",
        ]
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary = {"suite_dir": str(suite_dir), "preflight": preflight, "radii": radius_rows}
    result_path.with_suffix(".json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    summarize(_resolve(args.suite_dir), _resolve(args.result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
