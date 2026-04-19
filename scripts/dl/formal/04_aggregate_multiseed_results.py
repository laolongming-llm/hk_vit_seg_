"""Formal 多 seed 结果聚合脚本。

功能与作用：
1. 读取每个 seed 的 overall_metrics.csv，汇总为统一结果表。
2. 计算各 split 在多 seed 上的均值与标准差，输出工程汇总文件。
3. 生成简要 markdown 结果说明，便于阶段性汇报。
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from train_lib import get_run_paths, load_config, write_csv

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    """解析聚合脚本参数。"""
    parser = argparse.ArgumentParser(description="Aggregate formal multi-seed evaluation results.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/baseline_v1/train_vit_seg_baseline_v1_seed42.yaml",
            "configs/baseline_v1/train_vit_seg_baseline_v1_seed3407.yaml",
            "configs/baseline_v1/train_vit_seg_baseline_v1_seed2026.yaml",
        ],
        help="Config paths used in formal runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory for aggregated reports. Empty uses outputs/evaluation_reports/<date>_baseline_results.",
    )
    return parser.parse_args()


def load_seed_overall_metrics(config_path: str) -> pd.DataFrame:
    """读取单个 seed 的 overall 指标文件并附加 run 元信息。"""
    cfg = load_config(config_path)
    resolved_config = Path(cfg["_meta"]["resolved_config_path"])
    run_paths = get_run_paths(cfg, resolved_config)
    metric_path = run_paths["metrics_dir"] / "overall_metrics.csv"
    if not metric_path.exists():
        raise FileNotFoundError(f"Overall metrics not found: {metric_path}")
    df = pd.read_csv(metric_path)
    df["run_subdir"] = run_paths["run_dir"].name
    df["seed"] = int(cfg.get("experiment", {}).get("seed", -1))
    df["config_path"] = str(resolved_config)
    return df


def build_split_summary(seed_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """对每个 split 计算关键指标的均值与标准差。"""
    summary_rows: List[Dict[str, Any]] = []
    for split_name, group in seed_df.groupby("split"):
        row = {
            "split": split_name,
            "num_runs": int(len(group)),
            "miou_mean": float(group["miou"].mean()),
            "miou_std": float(group["miou"].std(ddof=0)),
            "mf1_mean": float(group["mf1"].mean()),
            "mf1_std": float(group["mf1"].std(ddof=0)),
            "overall_accuracy_mean": float(group["overall_accuracy"].mean()),
            "overall_accuracy_std": float(group["overall_accuracy"].std(ddof=0)),
        }
        summary_rows.append(row)
    return summary_rows


def write_markdown_report(path: Path, seed_df: pd.DataFrame, summary_rows: List[Dict[str, Any]]) -> None:
    """写入简要 markdown 汇总报告。"""
    lines: List[str] = []
    lines.append("# Formal Baseline Multi-seed Summary")
    lines.append("")
    lines.append(f"- generated_at: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- num_runs: {seed_df['run_subdir'].nunique()}")
    lines.append("")
    lines.append("## Split-level Mean/Std")
    lines.append("")
    lines.append("| split | runs | miou_mean | miou_std | mf1_mean | mf1_std | oa_mean | oa_std |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {split} | {num_runs} | {miou_mean:.6f} | {miou_std:.6f} | {mf1_mean:.6f} | {mf1_std:.6f} | {overall_accuracy_mean:.6f} | {overall_accuracy_std:.6f} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Per-run Records")
    lines.append("")
    lines.append("| run_subdir | seed | split | miou | mf1 | overall_accuracy |")
    lines.append("|---|---:|---|---:|---:|---:|")
    for _, item in seed_df.sort_values(["seed", "split"]).iterrows():
        lines.append(
            f"| {item['run_subdir']} | {int(item['seed'])} | {item['split']} | {float(item['miou']):.6f} | {float(item['mf1']):.6f} | {float(item['overall_accuracy']):.6f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """执行多 seed 结果聚合流程。"""
    args = parse_args()
    seed_frames: List[pd.DataFrame] = []
    for config in args.configs:
        seed_frames.append(load_seed_overall_metrics(config))
    seed_df = pd.concat(seed_frames, axis=0, ignore_index=True)

    if args.output_dir.strip():
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        output_dir = PROJECT_ROOT / "outputs" / "evaluation_reports" / f"{today}_baseline_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_summary_path = output_dir / "seed_summary.csv"
    split_summary_path = output_dir / "split_aggregate.csv"
    markdown_path = output_dir / "baseline_summary.md"

    seed_rows = seed_df.to_dict(orient="records")
    write_csv(path=seed_summary_path, rows=seed_rows, fieldnames=list(seed_df.columns))

    split_rows = build_split_summary(seed_df)
    write_csv(
        path=split_summary_path,
        rows=split_rows,
        fieldnames=[
            "split",
            "num_runs",
            "miou_mean",
            "miou_std",
            "mf1_mean",
            "mf1_std",
            "overall_accuracy_mean",
            "overall_accuracy_std",
        ],
    )
    write_markdown_report(markdown_path, seed_df=seed_df, summary_rows=split_rows)
    print(f"[OK] seed summary: {seed_summary_path}")
    print(f"[OK] split aggregate: {split_summary_path}")
    print(f"[OK] markdown report: {markdown_path}")


if __name__ == "__main__":
    main()
