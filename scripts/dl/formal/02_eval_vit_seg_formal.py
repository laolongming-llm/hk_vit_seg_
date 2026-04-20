"""Formal 评估脚本（按 checkpoint 在多 split 输出指标）。

功能与作用：
1. 加载 formal 训练 checkpoint 并在指定 split 执行统一评估。
2. 输出 overall/per-class 指标与混淆矩阵文件，作为最终汇总输入。
3. 对 eco holdout 缺类场景给出 NA 标注，避免指标误读。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_lib import (
    SegmentationTileDataset,
    build_segmentation_criterion,
    build_model_from_config,
    compute_metrics_from_confusion,
    ensure_run_dirs,
    get_run_paths,
    load_checkpoint,
    load_config,
    read_manifest,
    resolve_project_path,
    setup_logger,
    update_confusion_matrix,
    validate_manifest_files,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    """解析评估命令行参数。"""
    parser = argparse.ArgumentParser(description="Evaluate formal checkpoint on configured splits.")
    parser.add_argument("--config", required=True, help="Path to formal config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint file.")
    parser.add_argument(
        "--splits",
        default="",
        help="Comma-separated split names. Empty means use config.evaluation.splits_to_eval.",
    )
    return parser.parse_args()


def build_dataloader(dataset: SegmentationTileDataset, loader_cfg: Dict[str, Any]) -> DataLoader:
    """创建评估 DataLoader。"""
    num_workers = int(loader_cfg.get("num_workers", 0))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    batch_size = int(loader_cfg.get("eval_batch_size", loader_cfg.get("batch_size", 2)))
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0,
    )


@torch.no_grad()
def evaluate_split(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, Any]:
    """执行单个 split 的评估，返回 loss、指标与混淆矩阵。"""
    model.eval()
    losses: List[float] = []
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Eval loss is NaN/Inf.")
        losses.append(float(loss.detach().cpu().item()))
        confusion = update_confusion_matrix(
            confusion=confusion,
            logits=logits.detach().cpu(),
            targets=labels.detach().cpu(),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
    metric = compute_metrics_from_confusion(confusion)
    metric["eval_loss"] = float(sum(losses) / max(1, len(losses)))
    metric["confusion_matrix"] = confusion.tolist()
    return metric


def resolve_splits(cfg: Dict[str, Any], user_splits: str) -> List[str]:
    """确定本次评估要执行的 split 列表。"""
    if user_splits.strip():
        return [item.strip() for item in user_splits.split(",") if item.strip()]
    return list(cfg.get("evaluation", {}).get("splits_to_eval", ["val", "test_in_domain", "test_eco_holdout"]))


def metric_to_cell(value: float, present: bool) -> str:
    """将 per-class 指标按缺类规则转成 CSV 输出单元格。"""
    if not present:
        return "NA"
    if math.isnan(value):
        return "NA"
    return f"{value:.8f}"


def write_confusion_matrix(path: Path, confusion: List[List[int]], num_classes: int) -> None:
    """将混淆矩阵按 CSV 写盘。"""
    rows: List[Dict[str, Any]] = []
    for row_idx, row_vals in enumerate(confusion):
        row: Dict[str, Any] = {"class_idx": row_idx}
        for col_idx, value in enumerate(row_vals):
            row[f"pred_{col_idx}"] = int(value)
        rows.append(row)
    write_csv(path=path, rows=rows, fieldnames=["class_idx"] + [f"pred_{i}" for i in range(num_classes)])


def main() -> None:
    """执行 formal checkpoint 多 split 评估。"""
    args = parse_args()
    cfg = load_config(args.config)
    config_path = Path(cfg["_meta"]["resolved_config_path"])

    data_cfg = cfg.get("data", {})
    loader_cfg = cfg.get("loader", {})
    eval_cfg = cfg.get("evaluation", {})
    loss_cfg = cfg.get("loss", {})
    run_paths = get_run_paths(cfg, config_path)
    ensure_run_dirs(run_paths)
    logger = setup_logger(run_paths["logs_dir"] / "eval.log", logger_name="formal_eval")

    splits = resolve_splits(cfg, args.splits)
    logger.info("Evaluation splits: %s", splits)

    checkpoint_path = resolve_project_path(args.checkpoint, config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = load_checkpoint(checkpoint_path, device=device)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    num_classes = int(data_cfg["num_classes"])
    ignore_index = int(data_cfg.get("ignore_index", 255))
    ignore_lum_ids = [int(x) for x in data_cfg.get("ignore_lum_ids", [])]
    logger.info("Label remap policy: ignore_lum_ids=%s -> ignore_index=%d", ignore_lum_ids, ignore_index)
    criterion = build_segmentation_criterion(
        loss_cfg=loss_cfg,
        ignore_index=ignore_index,
        class_weights=None,
    )

    manifest_path = resolve_project_path(data_cfg["manifest_path"], config_path)
    dataset_root = resolve_project_path(data_cfg["dataset_root"], config_path)
    manifest_df = read_manifest(manifest_path)

    overall_rows: List[Dict[str, Any]] = []
    per_class_rows: List[Dict[str, Any]] = []
    missing_policy = str(eval_cfg.get("eco_holdout_missing_class_policy", "mark_na_and_average_present_classes"))

    for split_name in splits:
        split_df = manifest_df[manifest_df["final_split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            logger.warning("Skip split=%s because no rows found in manifest.", split_name)
            continue

        if bool(data_cfg.get("validate_files_on_start", True)):
            errors = validate_manifest_files(
                df=split_df,
                dataset_root=dataset_root,
                image_dirname=data_cfg["image_dirname"],
                label_dirname=data_cfg["label_dirname"],
                image_suffix=data_cfg["image_suffix"],
                label_suffix=data_cfg["label_suffix"],
            )
            if errors:
                preview = "\n".join(errors[:10])
                raise FileNotFoundError(f"Missing files found in split={split_name} ({len(errors)}).\n{preview}")

        dataset = SegmentationTileDataset(
            manifest_df=split_df,
            dataset_root=dataset_root,
            image_dirname=data_cfg["image_dirname"],
            label_dirname=data_cfg["label_dirname"],
            image_suffix=data_cfg["image_suffix"],
            label_suffix=data_cfg["label_suffix"],
            num_classes=num_classes,
            ignore_index=ignore_index,
            ignore_lum_ids=ignore_lum_ids,
        )
        data_loader = build_dataloader(dataset, loader_cfg=loader_cfg)
        metric = evaluate_split(
            model=model,
            data_loader=data_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        overall_rows.append(
            {
                "split": split_name,
                "num_samples": len(split_df),
                "eval_loss": metric["eval_loss"],
                "miou": metric["miou"],
                "mf1": metric["mf1"],
                "overall_accuracy": metric["overall_accuracy"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(payload.get("epoch", -1)),
                "missing_class_policy": missing_policy if split_name == "test_eco_holdout" else "",
            }
        )

        for class_idx in range(num_classes):
            present = bool(metric["present_mask"][class_idx])
            iou_value = float(metric["per_class_iou"][class_idx])
            f1_value = float(metric["per_class_f1"][class_idx])
            per_class_rows.append(
                {
                    "split": split_name,
                    "class_idx": class_idx,
                    "lum_id": class_idx + 1,
                    "present_in_gt": int(present),
                    "gt_pixels": int(metric["gt_pixels_per_class"][class_idx]),
                    "pred_pixels": int(metric["pred_pixels_per_class"][class_idx]),
                    "tp_pixels": int(metric["tp_pixels_per_class"][class_idx]),
                    "iou": metric_to_cell(iou_value, present=present),
                    "f1": metric_to_cell(f1_value, present=present),
                }
            )

        confusion_path = run_paths["metrics_dir"] / f"confusion_matrix_{split_name}.csv"
        write_confusion_matrix(
            path=confusion_path,
            confusion=metric["confusion_matrix"],
            num_classes=num_classes,
        )
        logger.info(
            "split=%s eval_loss=%.6f miou=%.6f mf1=%.6f oa=%.6f",
            split_name,
            metric["eval_loss"],
            metric["miou"],
            metric["mf1"],
            metric["overall_accuracy"],
        )

    if not overall_rows:
        raise RuntimeError("No split evaluated. Please check splits and manifest.")

    write_csv(
        path=run_paths["metrics_dir"] / "overall_metrics.csv",
        rows=overall_rows,
        fieldnames=[
            "split",
            "num_samples",
            "eval_loss",
            "miou",
            "mf1",
            "overall_accuracy",
            "checkpoint",
            "checkpoint_epoch",
            "missing_class_policy",
        ],
    )
    write_csv(
        path=run_paths["metrics_dir"] / "per_class_metrics.csv",
        rows=per_class_rows,
        fieldnames=[
            "split",
            "class_idx",
            "lum_id",
            "present_in_gt",
            "gt_pixels",
            "pred_pixels",
            "tp_pixels",
            "iou",
            "f1",
        ],
    )
    logger.info("Formal evaluation completed successfully.")


if __name__ == "__main__":
    main()
