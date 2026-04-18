"""Smoke Test 第 3 步：验证 checkpoint 可加载且可完成单批推理。

流程作用：
- 加载 smoke 训练产出的 checkpoint。
- 使用严格键匹配恢复模型权重。
- 在 train/val 样本上执行一次前向推理检查。
- 生成简要 load-check 报告，作为 smoke 验收证据。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from smoke_lib import (
    SegmentationTileDataset,
    build_model_from_config,
    get_run_paths,
    load_config,
    read_manifest,
    resolve_project_path,
)


def parse_args() -> argparse.Namespace:
    """解析 checkpoint 加载检查所需参数（配置与权重路径）。"""
    parser = argparse.ArgumentParser(
        description="Validate that a smoke checkpoint can be loaded and run for one batch."
    )
    parser.add_argument("--config", required=True, help="Path to smoke config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    return parser.parse_args()


def main() -> None:
    """执行 checkpoint 可用性检查并输出单批推理结果摘要。"""
    args = parse_args()
    cfg = load_config(args.config)
    config_path = Path(cfg["_meta"]["resolved_config_path"])

    data_cfg = cfg["data"]
    run_paths = get_run_paths(cfg, config_path)
    load_check_path = run_paths["metrics_dir"] / "load_check.txt"
    load_check_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = resolve_project_path(args.checkpoint, config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    manifest_path = resolve_project_path(data_cfg["manifest_path"], config_path)
    manifest_df = read_manifest(manifest_path)
    val_df = manifest_df[manifest_df["final_split"] == "val"].copy().reset_index(drop=True)
    if val_df.empty:
        val_df = manifest_df[manifest_df["final_split"] == "train"].copy().reset_index(drop=True)
    if val_df.empty:
        raise ValueError("Manifest has no train/val rows to run load check.")

    dataset_root = resolve_project_path(data_cfg["dataset_root"], config_path)
    dataset = SegmentationTileDataset(
        manifest_df=val_df.iloc[:1].copy(),
        dataset_root=dataset_root,
        image_dirname=data_cfg["image_dirname"],
        label_dirname=data_cfg["label_dirname"],
        image_suffix=data_cfg["image_suffix"],
        label_suffix=data_cfg["label_suffix"],
        num_classes=int(data_cfg["num_classes"]),
        ignore_index=int(data_cfg["ignore_index"]),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_config(cfg).to(device)

    payload = torch.load(checkpoint_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
        ckpt_epoch = payload.get("epoch")
    else:
        state_dict = payload
        ckpt_epoch = None
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    with torch.no_grad():
        batch = next(iter(loader))
        images = batch["image"].to(device)
        logits = model(images)
        preds = logits.argmax(dim=1)

    lines = [
        f"timestamp_utc={datetime.now(timezone.utc).isoformat()}",
        f"status=PASS",
        f"checkpoint={checkpoint_path}",
        f"checkpoint_epoch={ckpt_epoch}",
        f"device={device}",
        f"cuda_available={torch.cuda.is_available()}",
        f"logits_shape={tuple(logits.shape)}",
        f"pred_shape={tuple(preds.shape)}",
    ]
    load_check_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[OK] load check written: {load_check_path}")


if __name__ == "__main__":
    main()
