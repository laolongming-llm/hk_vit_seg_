"""Smoke Test 第 1 步：构建可复现的小规模子集清单（manifest）。

流程作用：
- 从配置中读取冻结后的完整 tiles manifest。
- 按固定随机种子为 train/val/test 各 split 抽样固定数量样本。
- 在训练前校验抽样结果对应的影像/标签文件是否存在。
- 将 smoke manifest 与 split 统计输出到运行目录。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import pandas as pd

from smoke_lib import (
    get_run_paths,
    load_config,
    read_manifest,
    resolve_project_path,
    validate_manifest_files,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，支持按需覆盖抽样种子和各 split 抽样数量。"""
    parser = argparse.ArgumentParser(
        description="Build a reproducible smoke subset manifest from the frozen tiles manifest."
    )
    parser.add_argument("--config", required=True, help="Path to smoke config YAML.")
    parser.add_argument("--seed", type=int, default=None, help="Override sampling seed.")
    parser.add_argument("--train", type=int, default=None, help="Override train sample count.")
    parser.add_argument("--val", type=int, default=None, help="Override val sample count.")
    parser.add_argument("--test-in-domain", type=int, default=None, dest="test_in_domain")
    parser.add_argument("--test-eco-holdout", type=int, default=None, dest="test_eco_holdout")
    return parser.parse_args()


def resolve_counts(cfg: Dict, args: argparse.Namespace) -> Dict[str, int]:
    """合并配置默认值与命令行覆盖值，得到最终的各 split 抽样数量。"""
    smoke_cfg = cfg.get("smoke", {})
    count_cfg = smoke_cfg.get("split_counts", {})
    defaults = {
        "train": 256,
        "val": 64,
        "test_in_domain": 32,
        "test_eco_holdout": 16,
    }
    counts = {
        split: int(count_cfg.get(split, defaults[split]))
        for split in defaults
    }
    for split in defaults:
        arg_value = getattr(args, split)
        if arg_value is not None:
            counts[split] = int(arg_value)
    return counts


def sample_split(df: pd.DataFrame, split_name: str, count: int, seed: int) -> pd.DataFrame:
    """从指定 split 中进行无放回抽样，并在样本不足时直接报错。"""
    split_df = df[df["final_split"] == split_name].copy()
    available = len(split_df)
    if count > available:
        raise ValueError(
            f"Requested {count} rows for split={split_name}, but only {available} available."
        )
    return split_df.sample(n=count, random_state=seed, replace=False)


def main() -> None:
    """执行 smoke 子集构建主流程并输出 manifest 与 split 统计结果。"""
    args = parse_args()
    cfg = load_config(args.config)
    config_path = Path(cfg["_meta"]["resolved_config_path"])

    smoke_cfg = cfg.get("smoke", {})
    seed = int(args.seed if args.seed is not None else smoke_cfg.get("subset_seed", 42))
    counts = resolve_counts(cfg, args)

    source_manifest_ref = smoke_cfg.get("source_manifest_path") or cfg["data"]["manifest_path"]
    source_manifest_path = resolve_project_path(source_manifest_ref, config_path)
    source_df = read_manifest(source_manifest_path)

    sampled_frames: List[pd.DataFrame] = []
    for split_name in ("train", "val", "test_in_domain", "test_eco_holdout"):
        sampled_frames.append(sample_split(source_df, split_name, counts[split_name], seed))

    sampled_df = pd.concat(sampled_frames, axis=0, ignore_index=True)
    sampled_df = sampled_df.sort_values(["final_split", "tile_id"]).reset_index(drop=True)

    data_cfg = cfg["data"]
    dataset_root = resolve_project_path(data_cfg["dataset_root"], config_path)
    missing_errors = validate_manifest_files(
        df=sampled_df,
        dataset_root=dataset_root,
        image_dirname=data_cfg["image_dirname"],
        label_dirname=data_cfg["label_dirname"],
        image_suffix=data_cfg["image_suffix"],
        label_suffix=data_cfg["label_suffix"],
    )
    if missing_errors:
        preview = "\n".join(missing_errors[:10])
        raise FileNotFoundError(
            f"Smoke subset has missing files ({len(missing_errors)}). First errors:\n{preview}"
        )

    run_paths = get_run_paths(cfg, config_path)
    manifests_dir = run_paths["manifests_dir"]
    manifests_dir.mkdir(parents=True, exist_ok=True)

    target_manifest_ref = smoke_cfg.get("subset_manifest_path") or cfg["data"]["manifest_path"]
    target_manifest_path = resolve_project_path(target_manifest_ref, config_path)
    target_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(target_manifest_path, index=False)

    split_counts_path = manifests_dir / "smoke_split_counts.csv"
    split_rows = []
    for split_name in ("train", "val", "test_in_domain", "test_eco_holdout"):
        split_rows.append(
            {
                "final_split": split_name,
                "requested_count": counts[split_name],
                "actual_count": int((sampled_df["final_split"] == split_name).sum()),
            }
        )
    write_csv(
        path=split_counts_path,
        rows=split_rows,
        fieldnames=["final_split", "requested_count", "actual_count"],
    )

    print(f"[OK] source manifest: {source_manifest_path}")
    print(f"[OK] smoke manifest:  {target_manifest_path}")
    print(f"[OK] split counts:    {split_counts_path}")
    print("[OK] selected rows:", len(sampled_df))


if __name__ == "__main__":
    main()
