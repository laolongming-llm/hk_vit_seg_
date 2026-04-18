"""Smoke Test 流程共享工具库。

流程作用：
- 提供以项目根目录为基准的路径与配置解析能力。
- 提供 manifest/文件校验与运行目录管理工具。
- 定义 smoke 使用的轻量分割模型与数据集加载器。
- 提供训练评估指标计算与 checkpoint 保存辅助函数。
"""

from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset


def _detect_project_root() -> Path:
    """自动探测项目根目录（优先以 .git 所在目录作为根）。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    # Fallback for environments where .git is unavailable.
    return current.parents[3]


PROJECT_ROOT = _detect_project_root()


def resolve_ref_path(path_ref: str | Path, config_base_dir: Path) -> Path:
    """解析路径引用：绝对路径直返；相对路径优先按项目根，再回退到配置目录。"""
    path_ref = Path(path_ref)
    if path_ref.is_absolute():
        return path_ref
    # Prefer project-root-relative resolution for all relative paths.
    candidate = (PROJECT_ROOT / path_ref).resolve()
    if candidate.exists():
        return candidate
    return (config_base_dir / path_ref).resolve()


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并配置字典，override 中同名键覆盖 base。"""
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_recursive(config_path: Path, stack: Optional[List[Path]] = None) -> Dict[str, Any]:
    """递归加载含 base_config 的配置，并检测循环引用。"""
    stack = stack or []
    resolved = config_path.resolve()
    if resolved in stack:
        chain = " -> ".join(str(p) for p in stack + [resolved])
        raise ValueError(f"Detected recursive base_config chain: {chain}")
    stack = stack + [resolved]

    with resolved.open("r", encoding="utf-8") as f:
        current_cfg = yaml.safe_load(f) or {}

    base_ref = current_cfg.get("base_config")
    if base_ref:
        base_path = resolve_ref_path(base_ref, resolved.parent)
        base_cfg = _load_config_recursive(base_path, stack=stack)
        current_cfg = dict(current_cfg)
        current_cfg.pop("base_config", None)
        merged = deep_merge_dict(base_cfg, current_cfg)
    else:
        merged = current_cfg

    merged.setdefault("_meta", {})
    merged["_meta"]["resolved_config_path"] = str(resolved)
    return merged


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """对外配置加载入口，返回已完成递归合并的配置。"""
    config_path = resolve_ref_path(config_path, PROJECT_ROOT)
    return _load_config_recursive(config_path)


def resolve_project_path(path_ref: str | Path, config_path: str | Path) -> Path:
    """结合当前配置位置解析项目内路径，供数据与输出路径统一调用。"""
    config_path = Path(config_path).resolve()
    return resolve_ref_path(path_ref, config_path.parent)


def get_run_paths(cfg: Dict[str, Any], config_path: str | Path) -> Dict[str, Path]:
    """根据配置生成本次运行的目录结构（logs/metrics/checkpoints/manifests）。"""
    output_cfg = cfg.get("output", {})
    experiment_cfg = cfg.get("experiment", {})
    root_dir = resolve_project_path(output_cfg.get("root_dir", "outputs/train_runs"), config_path)
    run_subdir = output_cfg.get("run_subdir") or experiment_cfg.get("run_name") or "smoke_run"
    run_dir = root_dir / run_subdir
    paths = {
        "root_dir": root_dir,
        "run_dir": run_dir,
        "logs_dir": run_dir / "logs",
        "metrics_dir": run_dir / "metrics",
        "checkpoints_dir": run_dir / "checkpoints",
        "manifests_dir": run_dir / "manifests",
    }
    return paths


def ensure_run_dirs(paths: Dict[str, Path]) -> None:
    """创建运行目录（忽略带后缀的文件路径项）。"""
    for path in paths.values():
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path) -> logging.Logger:
    """初始化训练日志器，同时输出到文件和终端。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("smoke_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_global_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """设置 Python/NumPy/PyTorch 随机种子，并配置 cuDNN 行为。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    """读取 manifest 并校验关键列是否存在。"""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    df = pd.read_csv(manifest_path)
    required = {"tile_id", "final_split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
    return df


def build_tile_paths(
    dataset_root: Path,
    image_dirname: str,
    label_dirname: str,
    image_suffix: str,
    label_suffix: str,
    split_name: str,
    tile_id: str,
) -> Tuple[Path, Path]:
    """按 split 与 tile_id 生成单个样本的影像与标签路径。"""
    image_path = dataset_root / image_dirname / split_name / f"{tile_id}{image_suffix}"
    label_path = dataset_root / label_dirname / split_name / f"{tile_id}{label_suffix}"
    return image_path, label_path


def validate_manifest_files(
    df: pd.DataFrame,
    dataset_root: Path,
    image_dirname: str,
    label_dirname: str,
    image_suffix: str,
    label_suffix: str,
) -> List[str]:
    """检查 manifest 对应文件是否齐全，返回缺失项错误列表。"""
    errors: List[str] = []
    for row in df.itertuples(index=False):
        split_name = str(getattr(row, "final_split"))
        tile_id = str(getattr(row, "tile_id"))
        image_path, label_path = build_tile_paths(
            dataset_root=dataset_root,
            image_dirname=image_dirname,
            label_dirname=label_dirname,
            image_suffix=image_suffix,
            label_suffix=label_suffix,
            split_name=split_name,
            tile_id=tile_id,
        )
        if not image_path.exists():
            errors.append(f"Missing image: {image_path}")
        if not label_path.exists():
            errors.append(f"Missing label: {label_path}")
    return errors


class TinySegNet(nn.Module):
    """用于 smoke 阶段的轻量分割网络，强调快速验证而非最终性能。"""
    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Dropout(p=dropout),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        x = self.encoder(x)
        x = self.decoder(x)
        if x.shape[-2:] != input_size:
            x = torch.nn.functional.interpolate(x, size=input_size, mode="bilinear", align_corners=False)
        return x


def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """从配置提取关键参数并实例化 smoke 模型。"""
    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    in_channels = int(data_cfg.get("in_channels", 3))
    num_classes = int(data_cfg.get("num_classes", 10))
    dropout = float(model_cfg.get("dropout", 0.1))
    return TinySegNet(in_channels=in_channels, num_classes=num_classes, dropout=dropout)


class SegmentationTileDataset(Dataset):
    """基于 manifest 的遥感分割数据集，按需读取影像与标签。"""
    def __init__(
        self,
        manifest_df: pd.DataFrame,
        dataset_root: Path,
        image_dirname: str,
        label_dirname: str,
        image_suffix: str,
        label_suffix: str,
        num_classes: int,
        ignore_index: int,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.image_dirname = image_dirname
        self.label_dirname = label_dirname
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """读取单样本并完成标签映射，返回训练所需张量字典。"""
        row = self.df.iloc[idx]
        tile_id = str(row["tile_id"])
        split_name = str(row["final_split"])
        image_path, label_path = build_tile_paths(
            dataset_root=self.dataset_root,
            image_dirname=self.image_dirname,
            label_dirname=self.label_dirname,
            image_suffix=self.image_suffix,
            label_suffix=self.label_suffix,
            split_name=split_name,
            tile_id=tile_id,
        )

        with rasterio.open(image_path) as src:
            image = src.read().astype(np.float32)
        with rasterio.open(label_path) as src:
            label = src.read(1).astype(np.int64)

        # Normalize common uint imagery range.
        if image.max() > 1.0:
            image = image / 255.0

        mapped_label = np.full_like(label, fill_value=self.ignore_index, dtype=np.int64)
        valid_mask = (label >= 1) & (label <= self.num_classes)
        mapped_label[valid_mask] = label[valid_mask] - 1

        return {
            "image": torch.from_numpy(image),
            "label": torch.from_numpy(mapped_label),
            "tile_id": tile_id,
            "split": split_name,
        }


def update_confusion_matrix(
    confusion: torch.Tensor,
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int,
) -> torch.Tensor:
    """按批次预测结果更新混淆矩阵，忽略 ignore_index 像素。"""
    preds = logits.argmax(dim=1)
    valid_mask = targets != ignore_index
    if valid_mask.sum() == 0:
        return confusion
    target_valid = targets[valid_mask]
    pred_valid = preds[valid_mask]
    encoded = target_valid * num_classes + pred_valid
    bincount = torch.bincount(encoded, minlength=num_classes * num_classes)
    confusion += bincount.reshape(num_classes, num_classes).cpu()
    return confusion


def compute_metrics_from_confusion(confusion: torch.Tensor) -> Dict[str, Any]:
    """由混淆矩阵计算每类 IoU、mIoU 与总体精度。"""
    confusion = confusion.to(torch.float64)
    tp = torch.diag(confusion)
    pos_gt = confusion.sum(dim=1)
    pos_pred = confusion.sum(dim=0)
    union = pos_gt + pos_pred - tp
    iou = torch.where(union > 0, tp / union, torch.nan)
    overall_acc = (tp.sum() / confusion.sum()).item() if confusion.sum() > 0 else 0.0
    miou = torch.nanmean(iou).item() if torch.isnan(iou).sum() < len(iou) else float("nan")
    return {
        "miou": miou,
        "overall_accuracy": overall_acc,
        "per_class_iou": iou.tolist(),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    """将行字典写入 CSV 文件（自动创建父目录）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """写入 JSON 文件（UTF-8，缩进格式）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """写入 YAML 文件并保留原始键顺序。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


@dataclass
class TrainState:
    """训练状态快照：当前 epoch 与最佳验证 mIoU。"""
    epoch: int
    best_val_miou: float


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_state: TrainState,
    config: Dict[str, Any],
) -> None:
    """保存训练 checkpoint，包含模型/优化器状态与关键训练元信息。"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": train_state.epoch,
        "best_val_miou": train_state.best_val_miou,
        "config": config,
    }
    torch.save(payload, checkpoint_path)
