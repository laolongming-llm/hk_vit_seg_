"""Formal 训练入口脚本（单 seed / 单 run）。

功能与作用：
1. 按 baseline 配置执行完整训练流程（train + val + checkpoint）。
2. 启动时做数据指纹校验与环境信息落盘，保证结果可追溯。
3. 输出正式训练产物到 outputs/train_runs/formal/<run_subdir>/。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_lib import (
    SegmentationTileDataset,
    TrainState,
    build_segmentation_criterion,
    build_model_from_config,
    compute_metrics_from_confusion,
    ensure_run_dirs,
    gather_env_info,
    get_run_paths,
    load_checkpoint,
    load_config,
    read_manifest,
    resolve_class_lum_ids,
    resolve_project_path,
    save_checkpoint,
    save_yaml,
    set_global_seed,
    setup_logger,
    update_confusion_matrix,
    validate_manifest_files,
    verify_dataset_fingerprint,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="Run formal ViT baseline training.")
    parser.add_argument("--config", required=True, help="Path to formal training config YAML.")
    return parser.parse_args()


def build_dataloader(
    dataset: SegmentationTileDataset,
    batch_size: int,
    shuffle: bool,
    loader_cfg: Dict[str, Any],
    is_train: bool,
) -> DataLoader:
    """根据配置创建训练/验证 DataLoader。"""
    num_workers = int(loader_cfg.get("num_workers", 0))
    persistent_workers = bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    drop_last = bool(loader_cfg.get("drop_last_train", False)) if is_train else False
    prefetch_factor = loader_cfg.get("prefetch_factor", 2)
    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def build_poly_scheduler(
    optimizer: torch.optim.Optimizer,
    max_epochs: int,
    warmup_epochs: int,
    power: float,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """构建按 epoch 更新的 Poly 学习率调度器（含 warmup）。"""
    max_epochs = max(1, int(max_epochs))
    warmup_epochs = max(0, int(warmup_epochs))
    power = float(power)
    min_lr = float(min_lr)
    base_lr = float(base_lr)

    def lr_lambda(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            warmup_ratio = float(epoch + 1) / float(max(1, warmup_epochs))
            return max(min_lr / base_lr, warmup_ratio)
        progress_num = max(0, epoch - warmup_epochs + 1)
        progress_den = max(1, max_epochs - warmup_epochs)
        progress = min(1.0, float(progress_num) / float(progress_den))
        poly = (1.0 - progress) ** power
        lr = min_lr + (base_lr - min_lr) * poly
        return max(min_lr / base_lr, lr / base_lr)

    return torch.optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


def build_optimizer(
    model: nn.Module,
    optimizer_cfg: Dict[str, Any],
    logger: Any,
) -> torch.optim.Optimizer:
    """构建 AdamW 优化器，支持主干与解码头差分学习率。"""
    base_lr = float(optimizer_cfg.get("lr", 1e-4))
    weight_decay = float(optimizer_cfg.get("weight_decay", 1e-2))
    betas = tuple(optimizer_cfg.get("betas", [0.9, 0.999]))
    eps = float(optimizer_cfg.get("eps", 1e-8))

    backbone_lr_mult = float(optimizer_cfg.get("backbone_lr_mult", 1.0))
    decoder_lr_mult = float(
        optimizer_cfg.get("decoder_lr_mult", optimizer_cfg.get("head_lr_mult", 1.0))
    )
    use_diff_lr = (abs(backbone_lr_mult - 1.0) > 1e-12) or (abs(decoder_lr_mult - 1.0) > 1e-12)

    if use_diff_lr and hasattr(model, "backbone") and hasattr(model, "decoder"):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        decoder_params = [p for p in model.decoder.parameters() if p.requires_grad]
        taken = {id(p) for p in backbone_params + decoder_params}
        other_params = [p for p in model.parameters() if p.requires_grad and id(p) not in taken]

        param_groups: List[Dict[str, Any]] = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": base_lr * backbone_lr_mult})
        if decoder_params:
            param_groups.append({"params": decoder_params, "lr": base_lr * decoder_lr_mult})
        if other_params:
            param_groups.append({"params": other_params, "lr": base_lr})

        logger.info(
            "Optimizer param groups enabled: base_lr=%.3e backbone_lr_mult=%.3f decoder_lr_mult=%.3f groups=%d",
            base_lr,
            backbone_lr_mult,
            decoder_lr_mult,
            len(param_groups),
        )
        return torch.optim.AdamW(
            param_groups,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
        )

    logger.info("Optimizer single LR mode: lr=%.3e", base_lr)
    return torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )


def apply_backbone_freeze_policy(model: nn.Module, model_cfg: Dict[str, Any], logger: Any) -> None:
    """Apply optional backbone freezing / partial unfreezing policy.

    Supported model config keys:
    - model.freeze_backbone: bool
    - model.unfreeze_backbone_last_n_blocks: int
    - model.unfreeze_backbone_norm: bool (default: true)
    """
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))
    unfreeze_last_n_blocks = int(model_cfg.get("unfreeze_backbone_last_n_blocks", 0))
    unfreeze_backbone_norm = bool(model_cfg.get("unfreeze_backbone_norm", True))

    if unfreeze_last_n_blocks < 0:
        raise ValueError("model.unfreeze_backbone_last_n_blocks must be >= 0.")

    if not hasattr(model, "backbone"):
        if freeze_backbone or unfreeze_last_n_blocks > 0:
            raise ValueError("Model has no backbone attribute, cannot apply freeze policy.")
        return

    backbone = model.backbone
    applied_freeze = False
    unfrozen_blocks = 0
    total_blocks = 0

    # If partial unfreezing is requested, we first freeze whole backbone then unfreeze tail blocks.
    if freeze_backbone or unfreeze_last_n_blocks > 0:
        for p in backbone.parameters():
            p.requires_grad = False
        applied_freeze = True

    if unfreeze_last_n_blocks > 0:
        blocks = getattr(backbone, "blocks", None)
        if blocks is None:
            raise ValueError(
                "model.unfreeze_backbone_last_n_blocks is set, but backbone has no 'blocks' attribute."
            )
        try:
            total_blocks = len(blocks)  # type: ignore[arg-type]
        except TypeError as exc:  # pragma: no cover - defensive
            raise ValueError("backbone.blocks is not a sized container.") from exc

        if total_blocks <= 0:
            raise ValueError("backbone.blocks is empty, cannot partially unfreeze.")

        unfrozen_blocks = min(unfreeze_last_n_blocks, total_blocks)
        for blk in list(blocks)[-unfrozen_blocks:]:
            for p in blk.parameters():
                p.requires_grad = True

        if unfreeze_backbone_norm and hasattr(backbone, "norm"):
            norm_module = getattr(backbone, "norm")
            if isinstance(norm_module, nn.Module):
                for p in norm_module.parameters():
                    p.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    backbone_total_params = sum(p.numel() for p in backbone.parameters())
    backbone_trainable_params = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    logger.info(
        "Backbone freeze policy: freeze_backbone=%s unfreeze_last_n_blocks=%d/%d "
        "unfreeze_backbone_norm=%s applied_freeze=%s | "
        "trainable_params=%d/%d backbone_trainable=%d/%d",
        freeze_backbone,
        unfrozen_blocks,
        total_blocks,
        unfreeze_backbone_norm,
        applied_freeze,
        trainable_params,
        total_params,
        backbone_trainable_params,
        backbone_total_params,
    )


def resolve_best_metric(output_cfg: Dict[str, Any]) -> Dict[str, str]:
    """解析 best checkpoint 依据指标与比较方向。

    返回：
    - `name`: 配置原始名称（用于日志）
    - `metric_key`: evaluate 返回字典中的键
    - `mode`: `max` 或 `min`
    """
    raw_name = str(output_cfg.get("save_best_by", "val_miou")).strip().lower()
    alias_to_key = {
        "val_miou": "miou",
        "miou": "miou",
        "val_mf1": "mf1",
        "mf1": "mf1",
        "val_overall_accuracy": "overall_accuracy",
        "overall_accuracy": "overall_accuracy",
        "val_oa": "overall_accuracy",
        "oa": "overall_accuracy",
        "val_loss": "val_loss",
        "loss": "val_loss",
    }
    if raw_name not in alias_to_key:
        raise ValueError(
            "Unsupported output.save_best_by value: "
            f"'{raw_name}'. Supported: {sorted(alias_to_key.keys())}"
        )
    metric_key = alias_to_key[raw_name]
    mode = "min" if metric_key == "val_loss" else "max"
    return {"name": raw_name, "metric_key": metric_key, "mode": mode}


def resolve_class_weights(
    loss_cfg: Dict[str, Any],
    train_df: pd.DataFrame,
    class_lum_ids: List[int],
    device: torch.device,
    logger: Any,
) -> Optional[torch.Tensor]:
    """解析并构建类别权重张量。

    支持三种模式：
    1. `null`：不启用类别权重。
    2. `list[float]`：手动指定每个类别的权重（长度需等于 num_classes）。
    3. `str` 自动模式：
       - `auto_inverse_freq` / `auto_inv_freq`
       - `auto_sqrt_inverse_freq` / `auto_inv_sqrt_freq`

    自动模式基于训练集 manifest 中 `class_1..class_N` 像元计数计算，
    并做“均值归一化 + floor/cap 截断”，避免极端稀有类导致权重爆炸。
    注意：最终返回前会再次确保权重落在 [floor, cap]，避免 cap 失效。
    """
    class_weights_cfg = loss_cfg.get("class_weights", None)
    if class_weights_cfg is None:
        logger.info("Class weights: disabled (equal weights).")
        return None

    num_classes = len(class_lum_ids)

    if isinstance(class_weights_cfg, list):
        if len(class_weights_cfg) != num_classes:
            raise ValueError(
                f"loss.class_weights length mismatch: got {len(class_weights_cfg)}, expected {num_classes}."
            )
        weights = np.asarray([float(x) for x in class_weights_cfg], dtype=np.float32)
        if not np.all(np.isfinite(weights)):
            raise ValueError("loss.class_weights contains NaN/Inf.")
        if np.any(weights <= 0):
            raise ValueError("loss.class_weights must be > 0 for all classes.")
        logger.info("Class weights: manual list enabled.")
        return torch.tensor(weights, dtype=torch.float32, device=device)

    if isinstance(class_weights_cfg, str):
        mode = class_weights_cfg.strip().lower()
        valid_modes = {
            "auto_inverse_freq",
            "auto_inv_freq",
            "auto_sqrt_inverse_freq",
            "auto_inv_sqrt_freq",
        }
        if mode not in valid_modes:
            raise ValueError(
                "Unsupported loss.class_weights mode. "
                f"Got '{class_weights_cfg}', expected one of {sorted(valid_modes)}."
            )

        class_cols = [f"class_{lum_id}" for lum_id in class_lum_ids]
        missing_cols = [c for c in class_cols if c not in train_df.columns]
        if missing_cols:
            raise ValueError(
                f"Manifest missing class columns required for auto class weights: {missing_cols}"
            )

        class_pixels = train_df[class_cols].sum(axis=0).astype(np.float64).to_numpy()
        class_pixels = np.clip(class_pixels, a_min=1.0, a_max=None)
        if mode in {"auto_inverse_freq", "auto_inv_freq"}:
            weights = 1.0 / class_pixels
        else:
            weights = 1.0 / np.sqrt(class_pixels)

        # 先归一化到均值 1，再做上下限截断。
        weights = weights / max(float(weights.mean()), 1e-12)
        weight_floor = float(loss_cfg.get("class_weights_floor", 0.2))
        weight_cap = float(loss_cfg.get("class_weights_cap", 5.0))
        if weight_floor <= 0 or weight_cap <= 0 or weight_floor > weight_cap:
            raise ValueError(
                f"Invalid class weight bounds: floor={weight_floor}, cap={weight_cap}."
            )
        weights = np.clip(weights, a_min=weight_floor, a_max=weight_cap)

        # 可选：clip 后再做一次归一化；默认关闭以确保 cap/floor 严格生效。
        renorm_after_clip = bool(loss_cfg.get("class_weights_renorm_after_clip", False))
        if renorm_after_clip:
            weights = weights / max(float(weights.mean()), 1e-12)
            weights = np.clip(weights, a_min=weight_floor, a_max=weight_cap)

        logger.info(
            "Class weights: auto mode=%s | floor=%.3f cap=%.3f | renorm_after_clip=%s | mean=%.4f min=%.4f max=%.4f | weights=%s",
            mode,
            weight_floor,
            weight_cap,
            renorm_after_clip,
            float(weights.mean()),
            float(weights.min()),
            float(weights.max()),
            [round(float(x), 4) for x in weights.tolist()],
        )
        return torch.tensor(weights.astype(np.float32), dtype=torch.float32, device=device)

    raise ValueError(
        "loss.class_weights must be null, list[float], or supported auto mode string."
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
    """在验证集上评估并返回 loss/IoU/F1/OA 与混淆矩阵。"""
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

    metric_dict = compute_metrics_from_confusion(confusion)
    metric_dict["val_loss"] = float(sum(losses) / max(1, len(losses)))
    metric_dict["confusion_matrix"] = confusion.tolist()
    return metric_dict


def maybe_resume(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.cuda.amp.GradScaler,
    resume_path: Optional[Path],
    device: torch.device,
    logger: Any,
) -> TrainState:
    """如配置了 resume_from，则从 checkpoint 恢复训练状态。"""
    if resume_path is None:
        return TrainState(epoch=0, global_step=0, best_val_miou=-1.0, best_epoch=0)
    payload = load_checkpoint(resume_path, device=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    if "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    if "scheduler_state_dict" in payload:
        scheduler.load_state_dict(payload["scheduler_state_dict"])
    if "scaler_state_dict" in payload:
        scaler.load_state_dict(payload["scaler_state_dict"])
    state = TrainState(
        epoch=int(payload.get("epoch", 0)),
        global_step=int(payload.get("global_step", 0)),
        best_val_miou=float(payload.get("best_val_miou", -1.0)),
        best_epoch=int(payload.get("best_epoch", 0)),
    )
    logger.info("Resume from checkpoint: %s (epoch=%d)", resume_path, state.epoch)
    return state


def main() -> None:
    """执行正式训练主流程。"""
    args = parse_args()
    cfg = load_config(args.config)
    config_path = Path(cfg["_meta"]["resolved_config_path"])

    experiment_cfg = cfg.get("experiment", {})
    data_cfg = cfg.get("data", {})
    loader_cfg = cfg.get("loader", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("train", {})
    optimizer_cfg = cfg.get("optimizer", {})
    scheduler_cfg = cfg.get("scheduler", {})
    loss_cfg = cfg.get("loss", {})
    output_cfg = cfg.get("output", {})

    run_paths = get_run_paths(cfg, config_path)
    ensure_run_dirs(run_paths)
    logger = setup_logger(run_paths["logs_dir"] / "train.log", logger_name="formal_train")

    config_snapshot_path = run_paths["run_dir"] / "config_snapshot.yaml"
    save_yaml(config_snapshot_path, cfg)
    logger.info("Config snapshot saved: %s", config_snapshot_path)

    env_info = gather_env_info()
    env_info_path = run_paths["run_dir"] / "env_info.json"
    write_json(env_info_path, env_info)
    logger.info("Environment info saved: %s", env_info_path)

    fingerprint_result = verify_dataset_fingerprint(cfg, config_path)
    fingerprint_path = run_paths["run_dir"] / "fingerprint_check.json"
    write_json(fingerprint_path, fingerprint_result)
    logger.info("Fingerprint check result saved: %s", fingerprint_path)
    if fingerprint_result.get("enabled") and fingerprint_result.get("status") != "PASSED":
        raise RuntimeError(f"Fingerprint verification failed: {fingerprint_result.get('errors', [])}")

    seed = int(experiment_cfg.get("seed", 42))
    set_global_seed(
        seed=seed,
        deterministic=bool(experiment_cfg.get("deterministic", True)),
        cudnn_benchmark=bool(experiment_cfg.get("cudnn_benchmark", False)),
    )
    logger.info("Seed initialized: %d", seed)

    dataset_root = resolve_project_path(data_cfg["dataset_root"], config_path)
    manifest_path = resolve_project_path(data_cfg["manifest_path"], config_path)
    manifest_df = read_manifest(manifest_path)
    logger.info("Loaded manifest rows: %d | path=%s", len(manifest_df), manifest_path)

    train_df = manifest_df[manifest_df["final_split"] == "train"].copy().reset_index(drop=True)
    val_df = manifest_df[manifest_df["final_split"] == "val"].copy().reset_index(drop=True)
    if train_df.empty or val_df.empty:
        raise ValueError("Train or val split is empty in manifest.")

    if bool(data_cfg.get("validate_files_on_start", True)):
        missing_errors = validate_manifest_files(
            df=pd.concat([train_df, val_df], axis=0, ignore_index=True),
            dataset_root=dataset_root,
            image_dirname=data_cfg["image_dirname"],
            label_dirname=data_cfg["label_dirname"],
            image_suffix=data_cfg["image_suffix"],
            label_suffix=data_cfg["label_suffix"],
        )
        if missing_errors:
            preview = "\n".join(missing_errors[:10])
            raise FileNotFoundError(f"Missing dataset files ({len(missing_errors)}). First errors:\n{preview}")

    class_lum_ids = resolve_class_lum_ids(data_cfg)
    num_classes = len(class_lum_ids)
    ignore_index = int(data_cfg.get("ignore_index", 255))
    ignore_lum_ids = [int(x) for x in data_cfg.get("ignore_lum_ids", [])]
    logger.info(
        "Label remap policy: class_lum_ids=%s | ignore_lum_ids=%s -> ignore_index=%d",
        class_lum_ids,
        ignore_lum_ids,
        ignore_index,
    )
    train_dataset = SegmentationTileDataset(
        manifest_df=train_df,
        dataset_root=dataset_root,
        image_dirname=data_cfg["image_dirname"],
        label_dirname=data_cfg["label_dirname"],
        image_suffix=data_cfg["image_suffix"],
        label_suffix=data_cfg["label_suffix"],
        num_classes=num_classes,
        ignore_index=ignore_index,
        ignore_lum_ids=ignore_lum_ids,
        class_lum_ids=class_lum_ids,
        enable_augment=bool(data_cfg.get("augmentation", {}).get("enabled", False)),
        augment_cfg=dict(data_cfg.get("augmentation", {})),
    )
    val_dataset = SegmentationTileDataset(
        manifest_df=val_df,
        dataset_root=dataset_root,
        image_dirname=data_cfg["image_dirname"],
        label_dirname=data_cfg["label_dirname"],
        image_suffix=data_cfg["image_suffix"],
        label_suffix=data_cfg["label_suffix"],
        num_classes=num_classes,
        ignore_index=ignore_index,
        ignore_lum_ids=ignore_lum_ids,
        class_lum_ids=class_lum_ids,
        enable_augment=False,
        augment_cfg=None,
    )
    train_loader = build_dataloader(
        dataset=train_dataset,
        batch_size=int(loader_cfg.get("batch_size", 2)),
        shuffle=True,
        loader_cfg=loader_cfg,
        is_train=True,
    )
    val_loader = build_dataloader(
        dataset=val_dataset,
        batch_size=int(loader_cfg.get("eval_batch_size", loader_cfg.get("batch_size", 2))),
        shuffle=False,
        loader_cfg=loader_cfg,
        is_train=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    logger.info("Device=%s | AMP=%s | backbone=%s", device, amp_enabled, model_cfg.get("backbone"))
    logger.info("Train augmentation enabled: %s", bool(data_cfg.get("augmentation", {}).get("enabled", False)))

    model = build_model_from_config(cfg).to(device)
    apply_backbone_freeze_policy(model=model, model_cfg=model_cfg, logger=logger)
    class_weights_tensor = resolve_class_weights(
        loss_cfg=loss_cfg,
        train_df=train_df,
        class_lum_ids=class_lum_ids,
        device=device,
        logger=logger,
    )
    criterion = build_segmentation_criterion(
        loss_cfg=loss_cfg,
        ignore_index=int(loss_cfg.get("ignore_index", ignore_index)),
        class_weights=class_weights_tensor,
    )
    logger.info(
        "Loss config: type=%s label_smoothing=%.4f ce_weight=%.3f dice_weight=%.3f",
        str(loss_cfg.get("type", "cross_entropy")),
        float(loss_cfg.get("label_smoothing", 0.0)),
        float(loss_cfg.get("ce_weight", 1.0)),
        float(loss_cfg.get("dice_weight", 1.0)),
    )
    optimizer = build_optimizer(model=model, optimizer_cfg=optimizer_cfg, logger=logger)
    max_epochs = int(train_cfg.get("max_epochs", 80))
    scheduler = build_poly_scheduler(
        optimizer=optimizer,
        max_epochs=max_epochs,
        warmup_epochs=int(scheduler_cfg.get("warmup_epochs", 3)),
        power=float(scheduler_cfg.get("power", 0.9)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
        base_lr=float(optimizer_cfg.get("lr", 1e-4)),
    )
    # 兼容新旧 AMP API，避免 deprecation warning 干扰日志阅读。
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # pragma: no cover - 仅兼容旧版 torch
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    resume_from = experiment_cfg.get("resume_from")
    resume_path = resolve_project_path(resume_from, config_path) if resume_from else None
    state = maybe_resume(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        resume_path=resume_path,
        device=device,
        logger=logger,
    )

    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    log_every = max(1, int(train_cfg.get("log_every_n_steps", 20)))
    val_every = max(1, int(train_cfg.get("val_every_n_epochs", 1)))
    early_patience = max(1, int(train_cfg.get("early_stopping_patience", 12)))
    early_min_delta = max(0.0, float(train_cfg.get("early_stopping_min_delta", 0.0)))
    best_metric_cfg = resolve_best_metric(output_cfg)
    logger.info(
        "Best metric config: save_best_by=%s -> key=%s mode=%s",
        best_metric_cfg["name"],
        best_metric_cfg["metric_key"],
        best_metric_cfg["mode"],
    )
    logger.info("Early stopping config: patience=%d, min_delta=%.6f", early_patience, early_min_delta)

    # 兼容旧字段名 best_val_miou：这里保存“当前 best 指标值”（不一定是 mIoU）。
    best_score = float(state.best_val_miou)
    if state.epoch <= 0:
        best_score = float("-inf") if best_metric_cfg["mode"] == "max" else float("inf")
        state.best_val_miou = best_score

    val_rows: List[Dict[str, Any]] = []
    best_confusion: Optional[List[List[int]]] = None
    non_improve_epochs = 0

    logger.info("Training started: epochs=%d, train_batches=%d", max_epochs, len(train_loader))
    start_epoch = state.epoch + 1
    for epoch in range(start_epoch, max_epochs + 1):
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
                raise RuntimeError(f"Encountered NaN/Inf loss at epoch={epoch}, batch={batch_idx}")

            scaler.scale(loss).backward()
            epoch_losses.append(float(loss.detach().cpu().item() * grad_accum_steps))

            if batch_idx % grad_accum_steps == 0 or batch_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                state.global_step += 1

                if state.global_step % log_every == 0:
                    logger.info(
                        "epoch=%d step=%d train_loss=%.6f lr=%.6e",
                        epoch,
                        state.global_step,
                        epoch_losses[-1],
                        optimizer.param_groups[0]["lr"],
                    )

        scheduler.step()
        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))

        do_val = (epoch % val_every == 0) or (epoch == max_epochs)
        current_val_loss = float("nan")
        current_val_miou = float("nan")
        current_val_mf1 = float("nan")
        current_val_oa = float("nan")
        if do_val:
            val_metrics = evaluate(
                model=model,
                data_loader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            current_val_loss = float(val_metrics["val_loss"])
            current_val_miou = float(val_metrics["miou"])
            current_val_mf1 = float(val_metrics["mf1"])
            current_val_oa = float(val_metrics["overall_accuracy"])
            val_rows.append(
                {
                    "epoch": epoch,
                    "global_step": state.global_step,
                    "train_loss": train_loss,
                    "val_loss": current_val_loss,
                    "val_miou": current_val_miou,
                    "val_mf1": current_val_mf1,
                    "val_overall_accuracy": current_val_oa,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            logger.info(
                "epoch=%d train_loss=%.6f val_loss=%.6f val_miou=%.6f val_mf1=%.6f val_oa=%.6f",
                epoch,
                train_loss,
                current_val_loss,
                current_val_miou,
                current_val_mf1,
                current_val_oa,
            )

            current_metric_value = float(val_metrics[best_metric_cfg["metric_key"]])
            if best_metric_cfg["mode"] == "max":
                improve_threshold = best_score + early_min_delta
                improved = current_metric_value > improve_threshold
                weak_improve = (current_metric_value > best_score) and (not improved)
            else:
                improve_threshold = best_score - early_min_delta
                improved = current_metric_value < improve_threshold
                weak_improve = (current_metric_value < best_score) and (not improved)

            if improved:
                best_score = current_metric_value
                state.best_val_miou = best_score
                state.best_epoch = epoch
                best_confusion = val_metrics["confusion_matrix"]
                non_improve_epochs = 0
                save_checkpoint(
                    checkpoint_path=run_paths["checkpoints_dir"] / "best.pth",
                    model=model,
                    optimizer=optimizer,
                    train_state=TrainState(
                        epoch=epoch,
                        global_step=state.global_step,
                        best_val_miou=state.best_val_miou,
                        best_epoch=state.best_epoch,
                    ),
                    config=cfg,
                    scheduler=scheduler,
                    scaler=scaler,
                )
                logger.info(
                    "Best checkpoint updated at epoch=%d (%s=%.6f)",
                    epoch,
                    best_metric_cfg["name"],
                    current_metric_value,
                )
            else:
                non_improve_epochs += val_every
                if weak_improve:
                    delta = (
                        current_metric_value - best_score
                        if best_metric_cfg["mode"] == "max"
                        else best_score - current_metric_value
                    )
                    logger.info(
                        "%s improved by %.6f but below min_delta=%.6f; best remains %.6f",
                        best_metric_cfg["name"],
                        delta,
                        early_min_delta,
                        best_score,
                    )

        save_checkpoint(
            checkpoint_path=run_paths["checkpoints_dir"] / "last.pth",
            model=model,
            optimizer=optimizer,
            train_state=TrainState(
                epoch=epoch,
                global_step=state.global_step,
                best_val_miou=state.best_val_miou,
                best_epoch=state.best_epoch,
            ),
            config=cfg,
            scheduler=scheduler,
            scaler=scaler,
        )

        if val_rows:
            write_csv(
                path=run_paths["metrics_dir"] / "val_metrics.csv",
                rows=val_rows,
                fieldnames=[
                    "epoch",
                    "global_step",
                    "train_loss",
                    "val_loss",
                    "val_miou",
                    "val_mf1",
                    "val_overall_accuracy",
                    "lr",
                ],
            )

        if non_improve_epochs >= early_patience:
            logger.info(
                "Early stopping triggered at epoch=%d (no improvement for %d epochs).",
                epoch,
                non_improve_epochs,
            )
            break

    if best_confusion is not None:
        conf_rows: List[Dict[str, Any]] = []
        for row_idx, row_vals in enumerate(best_confusion):
            row: Dict[str, Any] = {"class_idx": row_idx, "lum_id": int(class_lum_ids[row_idx])}
            for col_idx, value in enumerate(row_vals):
                row[f"pred_lum_{class_lum_ids[col_idx]}"] = int(value)
            conf_rows.append(row)
        write_csv(
            path=run_paths["metrics_dir"] / "confusion_matrix_val.csv",
            rows=conf_rows,
            fieldnames=["class_idx", "lum_id"] + [f"pred_lum_{lum_id}" for lum_id in class_lum_ids],
        )
    logger.info("Formal training completed successfully.")


if __name__ == "__main__":
    main()
