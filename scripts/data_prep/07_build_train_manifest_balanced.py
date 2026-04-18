#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：07_build_train_manifest_balanced.py

功能：
1) 读取数据集 tiles_manifest.csv；
2) 仅对 final_split=train 执行“重复采样式”再平衡；
3) 保持 val/test split 完全不变；
4) 输出新的平衡清单与统计报告，供后续训练直接使用。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v2" / "manifests" / "tiles_manifest.csv"
DEFAULT_OUTPUT_MANIFEST = (
    PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v2" / "manifests" / "tiles_manifest_balanced_train.csv"
)
DEFAULT_REPORT_DIR = PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v2" / "manifests" / "reports_balanced"


def to_abs_path(path_ref: str) -> Path:
    p = Path(path_ref).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def parse_rare_class_ids(raw: str, num_classes: int) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        cls = int(text)
        if 1 <= cls <= num_classes:
            values.append(cls)
    return sorted(set(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build balanced train manifest for ViT segmentation dataset.")
    parser.add_argument("--config", default="", help="YAML 配置文件路径（可选）")
    parser.add_argument("--input-manifest", default=str(DEFAULT_INPUT_MANIFEST), help="输入 tiles_manifest.csv 路径")
    parser.add_argument("--output-manifest", default=str(DEFAULT_OUTPUT_MANIFEST), help="输出平衡后的 manifest 路径")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="报告输出目录")

    parser.add_argument("--num-classes", type=int, default=8, help="类别数量（默认 8）")
    parser.add_argument(
        "--method",
        choices=["auto_inverse_freq", "auto_sqrt_inverse_freq"],
        default="auto_sqrt_inverse_freq",
        help="类别稀有度权重计算方式",
    )
    parser.add_argument("--target-train-ratio", type=float, default=1.6, help="训练样本总量目标倍数（相对原 train 行数）")
    parser.add_argument("--min-multiplier", type=float, default=1.0, help="单 tile 最小重复倍率")
    parser.add_argument("--max-multiplier", type=float, default=4.0, help="单 tile 最大重复倍率")
    parser.add_argument("--weight-floor", type=float, default=0.5, help="类别权重下界（clip）")
    parser.add_argument("--weight-cap", type=float, default=3.0, help="类别权重上界（clip）")
    parser.add_argument("--rare-class-ids", default="6,7,8", help="需要额外增强关注的类别 ID（逗号分隔）")
    parser.add_argument("--rare-class-boost", type=float, default=0.35, help="稀有类密度附加权重系数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（用于随机舍入）")
    parser.add_argument("--no-overwrite", action="store_true", help="输出文件存在则报错，不覆盖")
    return parser


def load_yaml_config(path_ref: str | None) -> Dict[str, object]:
    if not path_ref:
        return {}
    path = to_abs_path(path_ref)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是映射(dict)：{path}")
    return payload


def require_columns(df: pd.DataFrame, columns: List[str], file_path: Path) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Manifest 缺失字段 {missing}：{file_path}")


def compute_class_weights(
    train_df: pd.DataFrame,
    class_cols: List[str],
    method: str,
    weight_floor: float,
    weight_cap: float,
) -> np.ndarray:
    class_pixels = train_df[class_cols].sum(axis=0).to_numpy(dtype=np.float64)
    class_pixels = np.clip(class_pixels, a_min=1.0, a_max=None)
    freq = class_pixels / np.clip(class_pixels.sum(), a_min=1.0, a_max=None)

    if method == "auto_inverse_freq":
        weights = 1.0 / np.clip(freq, a_min=1e-12, a_max=None)
    else:
        weights = 1.0 / np.sqrt(np.clip(freq, a_min=1e-12, a_max=None))

    weights = weights / max(float(weights.mean()), 1e-12)
    weights = np.clip(weights, a_min=weight_floor, a_max=weight_cap)
    return weights


def stochastic_round(values: np.ndarray, seed: int) -> np.ndarray:
    base = np.floor(values).astype(np.int64)
    frac = values - base
    rng = np.random.default_rng(seed)
    plus = (rng.random(len(values)) < frac).astype(np.int64)
    return base + plus


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_class_stats_csv(path: Path, before_pixels: np.ndarray, after_pixels: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    before_sum = max(float(before_pixels.sum()), 1.0)
    after_sum = max(float(after_pixels.sum()), 1.0)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "before_pixels", "before_share", "after_pixels", "after_share", "share_delta"],
        )
        writer.writeheader()
        for i in range(len(before_pixels)):
            before_share = float(before_pixels[i] / before_sum)
            after_share = float(after_pixels[i] / after_sum)
            writer.writerow(
                {
                    "class_id": i + 1,
                    "before_pixels": int(before_pixels[i]),
                    "before_share": f"{before_share:.6f}",
                    "after_pixels": int(after_pixels[i]),
                    "after_share": f"{after_share:.6f}",
                    "share_delta": f"{(after_share - before_share):.6f}",
                }
            )


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="YAML 配置文件路径（可选）")
    pre_args, remaining = pre_parser.parse_known_args()

    args = build_parser().parse_args(remaining)
    provided_flags = {token[2:].replace("-", "_") for token in remaining if token.startswith("--")}
    yaml_cfg = load_yaml_config(pre_args.config)
    for key, value in yaml_cfg.items():
        attr = str(key).replace("-", "_")
        if hasattr(args, attr) and attr not in provided_flags:
            setattr(args, attr, value)
    if pre_args.config:
        print(f"[config] loaded: {to_abs_path(pre_args.config)}")

    if args.num_classes <= 0:
        raise ValueError("--num-classes 必须大于 0")
    if args.target_train_ratio <= 0:
        raise ValueError("--target-train-ratio 必须大于 0")
    if args.min_multiplier <= 0 or args.max_multiplier <= 0 or args.min_multiplier > args.max_multiplier:
        raise ValueError("无效的倍率边界：请检查 --min-multiplier 与 --max-multiplier")
    if args.weight_floor <= 0 or args.weight_cap <= 0 or args.weight_floor > args.weight_cap:
        raise ValueError("无效的类别权重边界：请检查 --weight-floor 与 --weight-cap")
    if args.rare_class_boost < 0:
        raise ValueError("--rare-class-boost 不能小于 0")

    input_manifest = to_abs_path(args.input_manifest)
    output_manifest = to_abs_path(args.output_manifest)
    report_dir = to_abs_path(args.report_dir)
    if not input_manifest.exists():
        raise FileNotFoundError(f"输入 manifest 不存在：{input_manifest}")
    if args.no_overwrite and output_manifest.exists():
        raise FileExistsError(f"输出 manifest 已存在：{output_manifest}")

    df = pd.read_csv(input_manifest)
    require_columns(df, ["tile_id", "final_split"], input_manifest)

    class_cols = [f"class_{i}" for i in range(1, args.num_classes + 1)]
    require_columns(df, class_cols, input_manifest)

    train_df = df[df["final_split"] == "train"].copy().reset_index(drop=True)
    non_train_df = df[df["final_split"] != "train"].copy().reset_index(drop=True)
    if train_df.empty:
        raise ValueError("输入 manifest 中 train split 为空，无法构建平衡清单")

    tile_valid = train_df[class_cols].sum(axis=1).to_numpy(dtype=np.float64)
    tile_valid = np.clip(tile_valid, a_min=1.0, a_max=None)

    class_weights = compute_class_weights(
        train_df=train_df,
        class_cols=class_cols,
        method=args.method,
        weight_floor=float(args.weight_floor),
        weight_cap=float(args.weight_cap),
    )

    class_matrix = train_df[class_cols].to_numpy(dtype=np.float64)
    class_ratio = class_matrix / tile_valid[:, None]
    tile_score = (class_ratio * class_weights[None, :]).sum(axis=1)

    rare_ids = parse_rare_class_ids(args.rare_class_ids, num_classes=args.num_classes)
    if rare_ids:
        rare_cols = [f"class_{cid}" for cid in rare_ids]
        rare_density = train_df[rare_cols].sum(axis=1).to_numpy(dtype=np.float64) / tile_valid
        tile_score = tile_score * (1.0 + float(args.rare_class_boost) * rare_density)

    tile_score = tile_score / max(float(tile_score.mean()), 1e-12)
    expected = tile_score * float(args.target_train_ratio)
    expected = np.clip(expected, a_min=float(args.min_multiplier), a_max=float(args.max_multiplier))
    repeats = stochastic_round(expected, seed=int(args.seed))
    repeats = np.clip(repeats, a_min=1, a_max=None).astype(np.int64)

    train_with_meta = train_df.copy()
    train_with_meta["balanced_sampling_score"] = tile_score
    train_with_meta["balanced_expected_multiplier"] = expected
    train_with_meta["balanced_repeat_total"] = repeats

    repeated_rows: List[Dict[str, object]] = []
    for row in train_with_meta.to_dict(orient="records"):
        total = int(row["balanced_repeat_total"])
        for idx in range(total):
            out = dict(row)
            out["balanced_repeat_index"] = idx + 1
            repeated_rows.append(out)
    balanced_train_df = pd.DataFrame(repeated_rows)

    out_df = pd.concat([balanced_train_df, non_train_df], axis=0, ignore_index=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_manifest, index=False)

    report_dir.mkdir(parents=True, exist_ok=True)
    repeat_stats = (
        train_with_meta[
            [
                "tile_id",
                "pair_name",
                "source_subsplit",
                "unknown_ratio",
                "balanced_sampling_score",
                "balanced_expected_multiplier",
                "balanced_repeat_total",
            ]
            + [c for c in ["class_6", "class_7", "class_8"] if c in train_with_meta.columns]
        ]
        .sort_values(by=["balanced_repeat_total", "balanced_sampling_score"], ascending=[False, False])
        .reset_index(drop=True)
    )
    repeat_stats.to_csv(report_dir / "tile_repeat_stats.csv", index=False)

    before_pixels = train_df[class_cols].sum(axis=0).to_numpy(dtype=np.float64)
    after_pixels = balanced_train_df[class_cols].sum(axis=0).to_numpy(dtype=np.float64)
    write_class_stats_csv(report_dir / "class_distribution_before_after.csv", before_pixels=before_pixels, after_pixels=after_pixels)

    summary_rows = [
        {"metric": "input_manifest", "value": str(input_manifest)},
        {"metric": "output_manifest", "value": str(output_manifest)},
        {"metric": "train_rows_before", "value": int(len(train_df))},
        {"metric": "train_rows_after", "value": int(len(balanced_train_df))},
        {"metric": "train_expand_ratio", "value": f"{(len(balanced_train_df) / max(len(train_df), 1)):.6f}"},
        {"metric": "target_train_ratio", "value": float(args.target_train_ratio)},
        {"metric": "non_train_rows_kept", "value": int(len(non_train_df))},
        {"metric": "total_rows_before", "value": int(len(df))},
        {"metric": "total_rows_after", "value": int(len(out_df))},
        {"metric": "repeat_min", "value": int(repeats.min())},
        {"metric": "repeat_mean", "value": f"{float(repeats.mean()):.6f}"},
        {"metric": "repeat_max", "value": int(repeats.max())},
        {"metric": "rare_class_ids", "value": ",".join(str(x) for x in rare_ids)},
        {"metric": "method", "value": args.method},
        {"metric": "seed", "value": int(args.seed)},
    ]
    write_summary_csv(report_dir / "balanced_summary.csv", summary_rows)

    report_md = report_dir / "balanced_summary.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# Train Manifest 平衡采样报告\n\n")
        f.write(f"- 输入清单: `{input_manifest}`\n")
        f.write(f"- 输出清单: `{output_manifest}`\n")
        f.write(f"- train 行数: `{len(train_df)} -> {len(balanced_train_df)}`\n")
        f.write(f"- 扩增倍率: `{len(balanced_train_df) / max(len(train_df), 1):.4f}`\n")
        f.write(f"- 重复倍率范围: `{repeats.min()} ~ {repeats.max()}`\n")
        f.write(f"- 方法: `{args.method}`\n")
        f.write(f"- 稀有类增强: `{','.join(map(str, rare_ids))}` (boost={args.rare_class_boost})\n")
        f.write("\n## 输出文件\n\n")
        f.write(f"1. `{output_manifest}`\n")
        f.write(f"2. `{report_dir / 'balanced_summary.csv'}`\n")
        f.write(f"3. `{report_dir / 'class_distribution_before_after.csv'}`\n")
        f.write(f"4. `{report_dir / 'tile_repeat_stats.csv'}`\n")

    print("[OK] balanced manifest created")
    print(f"  - input : {input_manifest}")
    print(f"  - output: {output_manifest}")
    print(f"  - train rows: {len(train_df)} -> {len(balanced_train_df)}")
    print(f"  - reports: {report_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

