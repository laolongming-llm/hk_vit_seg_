"""Smoke Test 第 2 步：执行最小可用的端到端分割训练检查。

流程作用：
- 读取 smoke 配置与第 1 步生成的子集 manifest。
- 使用轻量分割模型进行少轮次训练（默认 2 epoch）。
- 在 smoke 验证集上评估并记录核心指标。
- 输出日志、配置快照、指标表、混淆矩阵与 checkpoint。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from smoke_lib import (
    SegmentationTileDataset,
    TrainState,
    build_model_from_config,
    compute_metrics_from_confusion,
    ensure_run_dirs,
    get_run_paths,
    load_config,
    read_manifest,
    resolve_project_path,
    save_checkpoint,
    save_yaml,
    set_global_seed,
    setup_logger,
    update_confusion_matrix,
    validate_manifest_files,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    """解析训练脚本参数，目前必需提供 smoke 配置文件路径。"""
    parser = argparse.ArgumentParser(description="Run a minimal smoke segmentation training loop.")
    parser.add_argument("--config", required=True, help="Path to smoke config YAML.")
    return parser.parse_args()


def _build_dataloader(dataset: SegmentationTileDataset, batch_size: int, shuffle: bool, cfg_loader: Dict[str, Any]) -> DataLoader:
    """根据配置构建 DataLoader，统一处理并行加载与 pin memory 等选项。"""
    num_workers = int(cfg_loader.get("num_workers", 0))
    persistent_workers = bool(cfg_loader.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(cfg_loader.get("pin_memory", True))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    ignore_index: int,
) -> Dict[str, Any]:
    """在验证集上执行前向评估，返回 loss、mIoU、总体精度和混淆矩阵。"""
    model.eval()
    losses: List[float] = []
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for batch in data_loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Validation loss is NaN/Inf.")
        losses.append(float(loss.detach().cpu().item()))
        confusion = update_confusion_matrix(
            confusion=confusion,
            logits=logits.detach().cpu(),
            targets=labels.detach().cpu(),
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

    metrics = compute_metrics_from_confusion(confusion)
    mean_loss = float(sum(losses) / max(1, len(losses)))
    metrics["val_loss"] = mean_loss
    metrics["confusion_matrix"] = confusion.tolist()
    return metrics


def main() -> None:
    """执行 smoke 训练主流程：读配置、训练、验证、保存指标与 checkpoint。"""
    args = parse_args()
    cfg = load_config(args.config)
    config_path = Path(cfg["_meta"]["resolved_config_path"])

    experiment_cfg = cfg.get("experiment", {})
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})
    output_cfg = cfg.get("output", {})
    loader_cfg = cfg.get("loader", {})
    optimizer_cfg = cfg.get("optimizer", {})
    freeze_cfg = cfg.get("freeze", {})

    run_paths = get_run_paths(cfg, config_path)
    ensure_run_dirs(run_paths)
    logger = setup_logger(run_paths["logs_dir"] / "train.log")

    save_yaml(run_paths["run_dir"] / "config_snapshot.yaml", cfg)
    logger.info("Config snapshot written to %s", run_paths["run_dir"] / "config_snapshot.yaml")

    if bool(freeze_cfg.get("verify_fingerprint_on_start", False)):
        fingerprint_path = resolve_project_path(freeze_cfg["dataset_fingerprint_path"], config_path)
        if not fingerprint_path.exists():
            raise FileNotFoundError(f"Fingerprint file missing: {fingerprint_path}")
        logger.info("Fingerprint check passed: %s", fingerprint_path)

    seed = int(experiment_cfg.get("seed", 42))
    set_global_seed(
        seed=seed,
        deterministic=bool(experiment_cfg.get("deterministic", True)),
        cudnn_benchmark=bool(experiment_cfg.get("cudnn_benchmark", False)),
    )

    dataset_root = resolve_project_path(data_cfg["dataset_root"], config_path)
    manifest_path = resolve_project_path(data_cfg["manifest_path"], config_path)
    manifest_df = read_manifest(manifest_path)
    logger.info("Loaded manifest rows: %d from %s", len(manifest_df), manifest_path)

    required_splits = ("train", "val")
    split_to_df = {
        split: manifest_df[manifest_df["final_split"] == split].copy().reset_index(drop=True)
        for split in required_splits
    }
    for split, df in split_to_df.items():
        if df.empty:
            raise ValueError(f"Manifest split '{split}' has no rows.")

    missing_errors = validate_manifest_files(
        df=pd.concat([split_to_df["train"], split_to_df["val"]], axis=0, ignore_index=True),
        dataset_root=dataset_root,
        image_dirname=data_cfg["image_dirname"],
        label_dirname=data_cfg["label_dirname"],
        image_suffix=data_cfg["image_suffix"],
        label_suffix=data_cfg["label_suffix"],
    )
    if missing_errors:
        preview = "\n".join(missing_errors[:10])
        raise FileNotFoundError(
            f"Found missing dataset files in smoke manifest ({len(missing_errors)}).\n{preview}"
        )

    num_classes = int(data_cfg["num_classes"])
    ignore_index = int(data_cfg["ignore_index"])

    train_dataset = SegmentationTileDataset(
        manifest_df=split_to_df["train"],
        dataset_root=dataset_root,
        image_dirname=data_cfg["image_dirname"],
        label_dirname=data_cfg["label_dirname"],
        image_suffix=data_cfg["image_suffix"],
        label_suffix=data_cfg["label_suffix"],
        num_classes=num_classes,
        ignore_index=ignore_index,
    )
    val_dataset = SegmentationTileDataset(
        manifest_df=split_to_df["val"],
        dataset_root=dataset_root,
        image_dirname=data_cfg["image_dirname"],
        label_dirname=data_cfg["label_dirname"],
        image_suffix=data_cfg["image_suffix"],
        label_suffix=data_cfg["label_suffix"],
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    train_loader = _build_dataloader(
        dataset=train_dataset,
        batch_size=int(loader_cfg.get("batch_size", 4)),
        shuffle=True,
        cfg_loader=loader_cfg,
    )
    val_loader = _build_dataloader(
        dataset=val_dataset,
        batch_size=int(loader_cfg.get("eval_batch_size", loader_cfg.get("batch_size", 4))),
        shuffle=False,
        cfg_loader=loader_cfg,
    )

    if len(train_loader) == 0:
        raise ValueError("Train dataloader is empty.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    logger.info("Device=%s | AMP=%s", device, amp_enabled)

    model = build_model_from_config(cfg).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg.get("lr", 1e-4)),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-2)),
        betas=tuple(optimizer_cfg.get("betas", [0.9, 0.999])),
        eps=float(optimizer_cfg.get("eps", 1e-8)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    max_epochs = int(train_cfg.get("max_epochs", 2))
    log_every = int(train_cfg.get("log_every_n_steps", 5))
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))

    total_steps = max_epochs * math.ceil(len(train_loader) / grad_accum_steps)
    current_step = 0

    val_metric_rows: List[Dict[str, Any]] = []
    best_val_miou = -1.0
    best_confusion_matrix: List[List[int]] | None = None

    logger.info("Training started | epochs=%d | train_batches=%d", max_epochs, len(train_loader))
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        optimizer.zero_grad(set_to_none=True)

        for batch_idx, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels) / grad_accum_steps

            if not torch.isfinite(loss):
                raise RuntimeError(f"Encountered NaN/Inf loss at epoch={epoch}, batch={batch_idx}.")

            scaler.scale(loss).backward()
            epoch_losses.append(float(loss.detach().cpu().item() * grad_accum_steps))

            if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                current_step += 1

                if current_step % log_every == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    logger.info(
                        "epoch=%d step=%d/%d train_loss=%.6f lr=%.6e",
                        epoch,
                        current_step,
                        total_steps,
                        epoch_losses[-1],
                        lr_now,
                    )

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        val_metrics = evaluate(
            model=model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        val_miou = float(val_metrics["miou"])
        val_loss = float(val_metrics["val_loss"])
        val_acc = float(val_metrics["overall_accuracy"])
        logger.info(
            "epoch=%d train_loss=%.6f val_loss=%.6f val_miou=%.6f val_acc=%.6f",
            epoch,
            train_loss,
            val_loss,
            val_miou,
            val_acc,
        )

        val_metric_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "val_overall_accuracy": val_acc,
            }
        )

        train_state = TrainState(epoch=epoch, best_val_miou=best_val_miou)
        save_checkpoint(run_paths["checkpoints_dir"] / "last.pth", model, optimizer, train_state, cfg)

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_confusion_matrix = val_metrics["confusion_matrix"]
            best_state = TrainState(epoch=epoch, best_val_miou=best_val_miou)
            save_checkpoint(run_paths["checkpoints_dir"] / "best.pth", model, optimizer, best_state, cfg)
            logger.info("Saved new best checkpoint at epoch=%d (val_miou=%.6f).", epoch, val_miou)

    metrics_path = run_paths["metrics_dir"] / "val_metrics.csv"
    write_csv(
        path=metrics_path,
        rows=val_metric_rows,
        fieldnames=["epoch", "train_loss", "val_loss", "val_miou", "val_overall_accuracy"],
    )
    logger.info("Validation metrics saved: %s", metrics_path)

    if best_confusion_matrix is not None:
        conf_rows = []
        for row_idx, row_vals in enumerate(best_confusion_matrix):
            row_dict = {"class_idx": row_idx}
            for col_idx, value in enumerate(row_vals):
                row_dict[f"pred_{col_idx}"] = value
            conf_rows.append(row_dict)
        confusion_path = run_paths["metrics_dir"] / "confusion_matrix_val.csv"
        write_csv(
            path=confusion_path,
            rows=conf_rows,
            fieldnames=["class_idx"] + [f"pred_{i}" for i in range(num_classes)],
        )
        logger.info("Confusion matrix saved: %s", confusion_path)

    logger.info("Smoke training completed successfully.")


if __name__ == "__main__":
    main()
