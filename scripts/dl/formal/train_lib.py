"""Formal 训练公共工具库。

功能与作用：
1. 统一提供配置加载（含 base_config 递归合并）与项目路径解析。
2. 提供数据指纹校验、环境信息采集、日志与输出目录初始化。
3. 提供遥感分割数据集加载、ViT baseline 模型构建、指标计算与 checkpoint 管理。
4. 供 01/02/03/04/05 脚本复用，避免重复实现。
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset


def _detect_project_root() -> Path:
    """自动定位项目根目录（优先使用 .git 所在目录）。"""
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    # 兜底：scripts/dl/formal/train_lib.py -> 项目根目录通常是上 4 级。
    return current.parents[4]


PROJECT_ROOT = _detect_project_root()


def resolve_ref_path(path_ref: str | Path, config_base_dir: Path) -> Path:
    """解析路径引用：绝对路径直返；相对路径优先按项目根解析。"""
    path_ref = Path(path_ref)
    if path_ref.is_absolute():
        return path_ref
    candidate = (PROJECT_ROOT / path_ref).resolve()
    if candidate.exists():
        return candidate
    return (config_base_dir / path_ref).resolve()


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并两个字典，override 同名键覆盖 base。"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config_recursive(config_path: Path, stack: Optional[List[Path]] = None) -> Dict[str, Any]:
    """递归加载配置文件并处理 base_config 继承链。"""
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
    """加载正式训练配置（支持 base_config 继承与递归合并）。"""
    resolved = resolve_ref_path(config_path, PROJECT_ROOT)
    return _load_config_recursive(resolved)


def resolve_project_path(path_ref: str | Path, config_path: str | Path) -> Path:
    """Resolve runtime paths deterministically for training/evaluation."""
    config_path = Path(config_path).resolve()
    path_obj = Path(path_ref)
    if path_obj.is_absolute():
        return path_obj

    raw = str(path_ref).replace("\\", "/")
    if raw.startswith("./") or raw.startswith("../"):
        return (config_path.parent / path_obj).resolve()
    return (PROJECT_ROOT / path_obj).resolve()

def get_run_paths(cfg: Dict[str, Any], config_path: str | Path) -> Dict[str, Path]:
    """按配置生成 run 目录结构。"""
    output_cfg = cfg.get("output", {})
    experiment_cfg = cfg.get("experiment", {})
    root_dir = resolve_project_path(output_cfg.get("root_dir", "outputs/train_runs/formal"), config_path)
    run_subdir = output_cfg.get("run_subdir") or experiment_cfg.get("run_name") or "formal_run"
    run_dir = root_dir / run_subdir
    return {
        "root_dir": root_dir,
        "run_dir": run_dir,
        "logs_dir": run_dir / "logs",
        "metrics_dir": run_dir / "metrics",
        "checkpoints_dir": run_dir / "checkpoints",
        "figures_dir": run_dir / "figures",
    }


def ensure_run_dirs(paths: Dict[str, Path]) -> None:
    """创建 run 目录。"""
    for path in paths.values():
        if path.suffix:
            continue
        path.mkdir(parents=True, exist_ok=True)


def setup_logger(log_path: Path, logger_name: str) -> logging.Logger:
    """初始化日志器，同时输出到文件与终端。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def set_global_seed(seed: int, deterministic: bool = True, cudnn_benchmark: bool = False) -> None:
    """设置随机种子与 cuDNN 策略，增强训练复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)


def read_manifest(manifest_path: Path) -> pd.DataFrame:
    """读取 tiles manifest 并校验关键列。"""
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
    """根据 split 与 tile_id 构建影像/标签文件路径。"""
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
    """检查 manifest 对应样本文件是否存在，返回缺失列表。"""
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


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件 SHA256，用于数据指纹一致性校验。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_fingerprint(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    """校验数据冻结指纹关键目标是否完整且与当前文件哈希一致。"""
    freeze_cfg = cfg.get("freeze", {})
    result: Dict[str, Any] = {
        "enabled": bool(freeze_cfg.get("verify_fingerprint_on_start", False)),
        "status": "SKIPPED",
        "fingerprint_path": None,
        "checks": [],
        "errors": [],
    }
    if not result["enabled"]:
        return result

    fingerprint_path = resolve_project_path(freeze_cfg["dataset_fingerprint_path"], config_path)
    result["fingerprint_path"] = str(fingerprint_path)
    if not fingerprint_path.exists():
        result["status"] = "FAILED"
        result["errors"].append(f"Fingerprint file missing: {fingerprint_path}")
        return result

    df = pd.read_csv(fingerprint_path)
    required_targets = list(freeze_cfg.get("required_fingerprint_targets", []))
    for target in required_targets:
        row = df[df["target_path"] == target]
        if row.empty:
            result["errors"].append(f"Target missing in fingerprint: {target}")
            continue
        record = row.iloc[0].to_dict()
        path_obj = resolve_project_path(target, config_path)
        check_item = {
            "target_path": target,
            "path_exists": bool(path_obj.exists()),
            "file_exists_field": record.get("file_exists"),
            "sha256_match": True,
        }
        if not path_obj.exists():
            check_item["sha256_match"] = False
            result["checks"].append(check_item)
            result["errors"].append(f"Target file missing on disk: {path_obj}")
            continue
        expected_sha = str(record.get("sha256", "")).strip()
        if expected_sha and expected_sha.lower() != "nan":
            actual_sha = compute_sha256(path_obj)
            check_item["sha256_match"] = actual_sha == expected_sha
            if not check_item["sha256_match"]:
                result["errors"].append(f"SHA256 mismatch: {target}")
        result["checks"].append(check_item)

    result["status"] = "PASSED" if not result["errors"] else "FAILED"
    return result


def gather_env_info() -> Dict[str, Any]:
    """收集运行环境摘要信息，用于实验追溯。"""
    info: Dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cwd": str(Path.cwd()),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["torch_cuda_version"] = torch.version.cuda
    try:
        info["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        info["git_commit"] = None
    return info


class _ConvBNReLU(nn.Sequential):
    """常用 Conv-BN-ReLU 模块。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, dilation: int = 1):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class SimpleFPNHead(nn.Module):
    """轻量解码头：1x1 + 3x3 + 分类层。"""

    def __init__(self, in_channels: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNReLU(in_channels, 256, kernel_size=1, padding=0),
            _ConvBNReLU(256, 128, kernel_size=3, padding=1),
            nn.Dropout(p=float(dropout)),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASPPHead(nn.Module):
    """ASPP 解码头：用于增强多尺度感受野。"""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout: float,
        aspp_channels: int = 256,
        dilations: Tuple[int, int, int, int] = (1, 6, 12, 18),
    ):
        super().__init__()
        d1, d2, d3, d4 = [int(x) for x in dilations]
        self.branches = nn.ModuleList(
            [
                _ConvBNReLU(in_channels, aspp_channels, kernel_size=1, padding=0, dilation=d1),
                _ConvBNReLU(in_channels, aspp_channels, kernel_size=3, padding=d2, dilation=d2),
                _ConvBNReLU(in_channels, aspp_channels, kernel_size=3, padding=d3, dilation=d3),
                _ConvBNReLU(in_channels, aspp_channels, kernel_size=3, padding=d4, dilation=d4),
            ]
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # 注意：池化后特征是 [N, C, 1, 1]。
            # 当训练时出现 N=1（如最后一个不完整 batch）时，BatchNorm 会报错。
            # 这里改为 Conv + ReLU，避免依赖 batch 统计量。
            nn.Conv2d(in_channels, aspp_channels, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
        )
        fusion_channels = aspp_channels * (len(self.branches) + 1)
        self.project = nn.Sequential(
            _ConvBNReLU(fusion_channels, aspp_channels, kernel_size=1, padding=0),
            nn.Dropout(p=float(dropout)),
            nn.Conv2d(aspp_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_size = x.shape[-2:]
        feats = [branch(x) for branch in self.branches]
        pooled = self.image_pool(x)
        pooled = F.interpolate(pooled, size=spatial_size, mode="bilinear", align_corners=False)
        feats.append(pooled)
        fused = torch.cat(feats, dim=1)
        return self.project(fused)


class _ModelInitProgress:
    """Simple stage progress for model initialization without extra dependencies."""

    def __init__(self, total_steps: int = 4, enabled: bool = True):
        self.total_steps = max(1, int(total_steps))
        self.enabled = bool(enabled)
        self.current_step = 0
        self.stream = sys.stdout
        self._is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
        self._spinner_thread: Optional[threading.Thread] = None
        self._spinner_stop: Optional[threading.Event] = None
        self._spinner_label = ""
        self._spinner_start_ts = 0.0
        self._spinner_frames = ["|", "/", "-", "\\"]

    def _format_bar(self) -> str:
        width = 24
        filled = int(round(width * (self.current_step / float(self.total_steps))))
        filled = max(0, min(width, filled))
        return "[" + ("#" * filled) + ("-" * (width - filled)) + "]"

    def _write_line(self, text: str, carriage_return: bool = False) -> None:
        if carriage_return and self._is_tty:
            print(f"\r{text}", end="", file=self.stream, flush=True)
            return
        print(text, file=self.stream, flush=True)

    def start(self, label: str) -> None:
        if not self.enabled:
            return
        self._write_line(f"[model-init] {self._format_bar()} {self.current_step}/{self.total_steps} {label}")

    def step(self, label: str) -> None:
        if not self.enabled:
            return
        self.current_step = min(self.total_steps, self.current_step + 1)
        self._write_line(f"[model-init] {self._format_bar()} {self.current_step}/{self.total_steps} {label}")

    def start_spinner(self, label: str) -> None:
        if not self.enabled:
            return
        self.stop_spinner()
        self._spinner_label = label
        self._spinner_start_ts = time.time()
        self._spinner_stop = threading.Event()

        def _worker() -> None:
            frame_idx = 0
            last_non_tty_report_bucket = -1
            while self._spinner_stop and not self._spinner_stop.wait(0.2):
                elapsed = int(time.time() - self._spinner_start_ts)
                if self._is_tty:
                    frame = self._spinner_frames[frame_idx % len(self._spinner_frames)]
                    frame_idx += 1
                    line = (
                        f"[model-init] {self._format_bar()} {self.current_step}/{self.total_steps} "
                        f"{self._spinner_label} {frame} {elapsed}s"
                    )
                    self._write_line(line, carriage_return=True)
                else:
                    report_bucket = elapsed // 15
                    if report_bucket != last_non_tty_report_bucket:
                        last_non_tty_report_bucket = report_bucket
                        self._write_line(
                            f"[model-init] {self._format_bar()} {self.current_step}/{self.total_steps} "
                            f"{self._spinner_label} ... {elapsed}s"
                        )

        self._spinner_thread = threading.Thread(target=_worker, daemon=True)
        self._spinner_thread.start()

    def stop_spinner(self, done_label: Optional[str] = None) -> None:
        if self._spinner_stop is not None:
            self._spinner_stop.set()
        if self._spinner_thread is not None:
            self._spinner_thread.join(timeout=1.0)
        self._spinner_stop = None
        self._spinner_thread = None
        if not self.enabled:
            return
        if done_label:
            elapsed = int(max(0.0, time.time() - self._spinner_start_ts))
            self._write_line(
                f"[model-init] {self._format_bar()} {self.current_step}/{self.total_steps} "
                f"{done_label} ({elapsed}s)"
            )

    def finish(self, label: str) -> None:
        if not self.enabled:
            return
        self.stop_spinner()
        if self._is_tty:
            print(file=self.stream, flush=True)
        self._write_line(f"[model-init] {self._format_bar()} {self.current_step}/{self.total_steps} {label}")


def _env_flag(name: str, default: str = "0") -> bool:
    """Parse bool-like env var flags such as 1/0, true/false, on/off."""
    value = os.environ.get(name, default)
    if value is None:
        return False
    return str(value).strip().lower() not in {"0", "false", "off", "no", ""}


def _candidate_hf_hub_roots() -> List[Path]:
    """Collect candidate Hugging Face hub cache roots in priority order."""
    roots: List[Path] = []
    seen: set[str] = set()

    def _add(path_obj: Optional[Path]) -> None:
        if path_obj is None:
            return
        key = str(path_obj.resolve()) if path_obj.exists() else str(path_obj)
        if key in seen:
            return
        seen.add(key)
        roots.append(path_obj)

    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        _add(Path(hf_hub_cache))

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        _add(Path(hf_home) / "hub")

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        _add(Path(xdg_cache_home) / "huggingface" / "hub")

    _add(Path.home() / ".cache" / "huggingface" / "hub")
    # Project-local preferred fallback on Windows workstations.
    _add(Path(r"E:\cache\huggingface\hub"))
    return roots


def _resolve_timm_cache_dir() -> Optional[Path]:
    """Resolve cache dir passed to timm / huggingface_hub APIs."""
    for root in _candidate_hf_hub_roots():
        if root.exists():
            return root
    return None


def _find_local_hf_weight_file(hf_hub_id: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Find a complete local snapshot weight file for a HF repo id."""
    repo_id = str(hf_hub_id).strip()
    if not repo_id:
        return None, None, None
    repo_id = repo_id.split("@", 1)[0]
    repo_cache_name = f"models--{repo_id.replace('/', '--')}"

    preferred_names = [
        "model.safetensors",
        "pytorch_model.bin",
        "open_clip_pytorch_model.bin",
    ]
    preferred_exts = (".safetensors", ".bin", ".pth", ".pt")

    for hub_root in _candidate_hf_hub_roots():
        repo_dir = hub_root / repo_cache_name
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        ordered_snapshots: List[Path] = []
        main_ref = repo_dir / "refs" / "main"
        if main_ref.exists():
            try:
                main_sha = main_ref.read_text(encoding="utf-8").strip()
                if main_sha:
                    main_snapshot = snapshots_dir / main_sha
                    if main_snapshot.exists():
                        ordered_snapshots.append(main_snapshot)
            except Exception:
                pass

        other_snapshots = sorted(
            [p for p in snapshots_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for snap in other_snapshots:
            if snap not in ordered_snapshots:
                ordered_snapshots.append(snap)

        for snapshot_dir in ordered_snapshots:
            for name in preferred_names:
                candidate = snapshot_dir / name
                if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
                    return candidate, snapshot_dir, hub_root

            for candidate in sorted(snapshot_dir.iterdir()):
                if (
                    candidate.is_file()
                    and candidate.suffix.lower() in preferred_exts
                    and candidate.stat().st_size > 0
                ):
                    return candidate, snapshot_dir, hub_root

    return None, None, None


def _prefetch_hf_weights_if_possible(backbone_name: str, timm_module: Any) -> None:
    """Prefetch Hugging Face weights so terminal can show byte/speed progress via tqdm."""
    if not _env_flag("HK_VIT_SEG_HF_PREDOWNLOAD", "1"):
        return

    try:
        pretrained_cfg = timm_module.get_pretrained_cfg(backbone_name)
    except Exception:
        return
    hf_hub_id = getattr(pretrained_cfg, "hf_hub_id", None)
    if not hf_hub_id:
        return

    local_weight_file, local_snapshot_dir, local_hub_root = _find_local_hf_weight_file(str(hf_hub_id))
    if local_weight_file is not None:
        print(
            "[hf-download] local cache hit: "
            f"{local_weight_file} (snapshot={local_snapshot_dir}, cache={local_hub_root})",
            flush=True,
        )
        return

    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        from tqdm.auto import tqdm as _TqdmBase
    except Exception:
        print("[hf-download] huggingface_hub/tqdm unavailable, skip predownload.", flush=True)
        return

    token = os.environ.get("HF_TOKEN")
    max_workers = int(os.environ.get("HK_VIT_SEG_HF_MAX_WORKERS", "4"))
    force_download = _env_flag("HK_VIT_SEG_HF_FORCE_DOWNLOAD", "0")
    progress_every_seconds = float(os.environ.get("HK_VIT_SEG_HF_PROGRESS_EVERY_SECONDS", "1.0"))
    local_files_only = _env_flag("HF_HUB_OFFLINE", "0") or _env_flag("HK_VIT_SEG_HF_LOCAL_ONLY", "0")
    cache_dir = _resolve_timm_cache_dir()

    class _LineProgressTqdm(_TqdmBase):
        """Line-based progress output that keeps logs readable (no carriage-return overwrite)."""

        def __init__(self, *args: Any, **kwargs: Any):
            self._last_emit_ts = 0.0
            kwargs.setdefault("ascii", True)
            kwargs.setdefault("dynamic_ncols", False)
            super().__init__(*args, **kwargs)

        def display(self, msg: Optional[str] = None, pos: Optional[int] = None) -> None:  # type: ignore[override]
            _ = msg
            _ = pos
            now = time.time()
            if not hasattr(self, "_last_emit_ts"):
                self._last_emit_ts = 0.0
            if (self.total is not None) and (self.n < self.total):
                if (now - self._last_emit_ts) < progress_every_seconds:
                    return
            self._last_emit_ts = now

            rate = self.format_dict.get("rate", None)
            if rate is None or rate <= 0:
                rate_text = "n/a"
                eta_text = "n/a"
            else:
                rate_text = f"{self.format_sizeof(rate)}/s"
                if self.total is not None and self.total >= self.n:
                    eta_sec = int((self.total - self.n) / max(rate, 1e-12))
                    eta_text = f"{eta_sec}s"
                else:
                    eta_text = "n/a"

            if self.total is not None and self.total > 0:
                pct = 100.0 * (self.n / float(self.total))
                payload = (
                    f"{self.desc or 'download'} | {pct:6.2f}% | "
                    f"{self.format_sizeof(self.n)}/{self.format_sizeof(self.total)} | "
                    f"{rate_text} | eta {eta_text}"
                )
            else:
                payload = f"{self.desc or 'download'} | {self.format_sizeof(self.n)} | {rate_text}"
            print(f"[hf-progress] {payload}", flush=True)

    print(f"[hf-download] prefetch start: {hf_hub_id} (max_workers={max_workers})", flush=True)
    try:
        if local_files_only:
            print("[hf-download] local-only mode enabled and no cache hit, skip network prefetch.", flush=True)
            return
        api = HfApi(token=token)
        info = api.model_info(repo_id=hf_hub_id, files_metadata=True)
        all_files = [s.rfilename for s in (info.siblings or [])]
        preferred_weight = None
        if "model.safetensors" in all_files:
            preferred_weight = "model.safetensors"
        elif "pytorch_model.bin" in all_files:
            preferred_weight = "pytorch_model.bin"
        else:
            for name in all_files:
                if name.endswith((".safetensors", ".bin", ".pth", ".pt")):
                    preferred_weight = name
                    break

        if preferred_weight:
            print(f"[hf-download] downloading weight file: {preferred_weight}", flush=True)
            hf_hub_download(
                repo_id=hf_hub_id,
                filename=preferred_weight,
                token=token,
                force_download=force_download,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                tqdm_class=_LineProgressTqdm,
            )
            # Download lightweight metadata file for cache completeness when present.
            if "config.json" in all_files:
                hf_hub_download(
                    repo_id=hf_hub_id,
                    filename="config.json",
                    token=token,
                    force_download=False,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    tqdm_class=_LineProgressTqdm,
                )
        else:
            snapshot_download(
                repo_id=hf_hub_id,
                token=token,
                allow_patterns=["*.safetensors", "*.bin", "*.pth", "*.pt", "*.json"],
                force_download=force_download,
                max_workers=max_workers,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                tqdm_class=_LineProgressTqdm,
            )
        print("[hf-download] prefetch done.", flush=True)
    except Exception as exc:
        # Non-fatal: timm may still download from its own path as fallback.
        print(f"[hf-download] prefetch skipped due to error: {exc!r}", flush=True)


class TimmSegModel(nn.Module):
    """通用 timm 主干 + 可选解码头的分割模型。"""

    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        pretrained: bool,
        dropout: float,
        input_size: int,
        decoder_head: str = "simple_fpn_head",
        drop_path_rate: float = 0.0,
    ):
        super().__init__()
        try:
            import timm
        except Exception as exc:  # pragma: no cover - 依赖缺失时给清晰报错
            raise RuntimeError("timm is required for segmentation model.") from exc

        progress = _ModelInitProgress(
            total_steps=5,
            enabled=os.environ.get("HK_VIT_SEG_INIT_PROGRESS", "1").strip().lower()
            not in {"0", "false", "off"},
        )
        progress.start("start model initialization")

        create_kwargs: Dict[str, Any] = {
            "pretrained": pretrained,
            "num_classes": 0,
            "global_pool": "",
        }
        if float(drop_path_rate) > 0:
            create_kwargs["drop_path_rate"] = float(drop_path_rate)
        # ViT/DeiT/EVA 等 token-based 主干通常需要动态输入尺寸支持。
        if any(key in backbone_name for key in ("vit", "deit", "eva")):
            create_kwargs["img_size"] = int(input_size)
            create_kwargs["dynamic_img_size"] = True

        progress.step("backbone arguments prepared")
        cache_dir = _resolve_timm_cache_dir()
        if cache_dir is not None:
            create_kwargs["cache_dir"] = str(cache_dir)
        if pretrained:
            # Keep HuggingFace transfer progress visible when hub backend supports it.
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
            hf_hub_id = None
            try:
                pretrained_cfg = timm.get_pretrained_cfg(backbone_name)
                hf_hub_id = getattr(pretrained_cfg, "hf_hub_id", None)
            except Exception:
                hf_hub_id = None

            if hf_hub_id:
                local_weight_file, local_snapshot_dir, local_hub_root = _find_local_hf_weight_file(str(hf_hub_id))
                if local_weight_file is not None:
                    create_kwargs["pretrained_cfg_overlay"] = {
                        "source": "timm",
                        "file": str(local_weight_file),
                    }
                    print(
                        "[model-init] local pretrained override enabled: "
                        f"file={local_weight_file} (snapshot={local_snapshot_dir}, cache={local_hub_root})",
                        flush=True,
                    )
                elif _env_flag("HF_HUB_OFFLINE", "0"):
                    raise RuntimeError(
                        "HF_HUB_OFFLINE=1 but no local pretrained cache found for "
                        f"{hf_hub_id}. Disable offline mode or pre-download the weights."
                    )
            _prefetch_hf_weights_if_possible(backbone_name=backbone_name, timm_module=timm)
        progress.step("pretrained cache checked")
        progress.start_spinner("creating backbone (pretrained may download)")
        self.backbone = timm.create_model(backbone_name, **create_kwargs)
        progress.stop_spinner(done_label="backbone ready")
        progress.step("backbone created")
        self.decoder_head_name = str(decoder_head).strip().lower()
        embed_dim = int(getattr(self.backbone, "num_features", 768))

        if self.decoder_head_name in {"simple_fpn_head", "simple_head"}:
            self.decoder = SimpleFPNHead(in_channels=embed_dim, num_classes=num_classes, dropout=dropout)
        elif self.decoder_head_name in {"aspp_head", "deeplab_aspp_head"}:
            self.decoder = ASPPHead(in_channels=embed_dim, num_classes=num_classes, dropout=dropout)
        else:
            raise ValueError(
                "Unsupported model.decoder_head: "
                f"'{decoder_head}'. Supported: simple_fpn_head, aspp_head"
            )
        progress.step("decoder created")
        progress.step("model initialization completed")
        progress.finish("ready")

    def _resolve_patch_size(self) -> Tuple[int, int]:
        patch_embed = getattr(self.backbone, "patch_embed", None)
        patch_size = getattr(patch_embed, "patch_size", 16)
        if isinstance(patch_size, tuple):
            return int(patch_size[0]), int(patch_size[1])
        patch = int(patch_size)
        return patch, patch

    def _tokens_to_map(self, tokens: torch.Tensor, input_h: int, input_w: int) -> torch.Tensor:
        """将 token 序列恢复为二维特征图；若已是 4D 特征图则直接返回。"""
        if isinstance(tokens, (list, tuple)):
            if len(tokens) == 0:
                raise ValueError("Backbone returned empty feature list.")
            tokens = tokens[-1]
        if tokens.ndim == 4:
            return tokens
        if tokens.ndim != 3:
            raise ValueError(f"Unsupported feature shape from backbone: {tuple(tokens.shape)}")

        bsz, token_count, channels = tokens.shape
        patch_h, patch_w = self._resolve_patch_size()
        expected = max(1, input_h // patch_h) * max(1, input_w // patch_w)
        # timm 的部分主干（如 DINOv3）会输出多个前缀 token（cls + register tokens）。
        # 优先按 num_prefix_tokens（若存在）剥离；其次按 token_count-expected 自适应剥离。
        prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 0) or 0)
        if token_count != expected:
            candidate_prefix: List[int] = []
            if prefix_tokens > 0:
                candidate_prefix.append(prefix_tokens)
            if token_count > expected:
                candidate_prefix.append(token_count - expected)
            if token_count == expected + 1:
                candidate_prefix.append(1)

            seen = set()
            stripped = False
            for n_prefix in candidate_prefix:
                if n_prefix in seen:
                    continue
                seen.add(n_prefix)
                if n_prefix <= 0 or n_prefix >= token_count:
                    continue
                cand = tokens[:, n_prefix:, :]
                if cand.shape[1] == expected:
                    tokens = cand
                    stripped = True
                    break

            if (not stripped) and token_count != expected:
                # 容错：当主干输出 token 数与输入尺度推导不一致时，尝试按正方网格恢复。
                maybe_without_cls = token_count - 1
                grid = int(round(float(maybe_without_cls) ** 0.5))
                if grid * grid == maybe_without_cls:
                    tokens = tokens[:, 1:, :]
                else:
                    grid = int(round(float(token_count) ** 0.5))
                    if grid * grid == token_count:
                        pass
                    else:
                        raise ValueError(
                            "Token count mismatch and cannot infer 2D grid: "
                            f"token_count={token_count}, expected={expected}."
                        )

        spatial = int(round(float(tokens.shape[1]) ** 0.5))
        if spatial * spatial != tokens.shape[1]:
            raise ValueError(f"Token count {tokens.shape[1]} cannot form square feature map.")
        return tokens.transpose(1, 2).reshape(bsz, channels, spatial, spatial)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行前向推理并上采样回输入空间分辨率。"""
        input_h, input_w = x.shape[-2:]
        features = self.backbone.forward_features(x)
        feat_map = self._tokens_to_map(features, input_h=input_h, input_w=input_w)
        logits = self.decoder(feat_map)
        logits = F.interpolate(logits, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return logits


class SoftDiceLoss(nn.Module):
    """Soft Dice loss（忽略 ignore_index）。"""

    def __init__(self, ignore_index: int = 255, eps: float = 1e-6):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid_mask = targets != self.ignore_index
        if not torch.any(valid_mask):
            return logits.new_tensor(0.0)

        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))
        one_hot = F.one_hot(safe_targets, num_classes=probs.shape[1]).permute(0, 3, 1, 2).to(dtype=probs.dtype)

        valid_mask_f = valid_mask.unsqueeze(1).to(dtype=probs.dtype)
        probs = probs * valid_mask_f
        one_hot = one_hot * valid_mask_f

        intersection = torch.sum(probs * one_hot, dim=(0, 2, 3))
        pred_sum = torch.sum(probs, dim=(0, 2, 3))
        target_sum = torch.sum(one_hot, dim=(0, 2, 3))
        dice = (2.0 * intersection + self.eps) / (pred_sum + target_sum + self.eps)

        present_classes = target_sum > 0
        if torch.any(present_classes):
            return 1.0 - dice[present_classes].mean()
        return 1.0 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    """组合损失：CrossEntropy + SoftDice。"""

    def __init__(
        self,
        ignore_index: int = 255,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            ignore_index=int(ignore_index),
            label_smoothing=max(0.0, float(label_smoothing)),
            weight=class_weights,
        )
        self.dice = SoftDiceLoss(ignore_index=ignore_index)
        self.ce_weight = float(ce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def build_segmentation_criterion(
    loss_cfg: Dict[str, Any],
    ignore_index: int,
    class_weights: Optional[torch.Tensor] = None,
) -> nn.Module:
    """根据配置构建分割损失函数。"""
    loss_type = str(loss_cfg.get("type", "cross_entropy")).strip().lower()
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))

    if loss_type in {"cross_entropy", "ce"}:
        return nn.CrossEntropyLoss(
            ignore_index=int(ignore_index),
            label_smoothing=max(0.0, label_smoothing),
            weight=class_weights,
        )
    if loss_type in {"ce_dice", "cross_entropy_dice", "dice_ce"}:
        return CrossEntropyDiceLoss(
            ignore_index=int(ignore_index),
            label_smoothing=max(0.0, label_smoothing),
            class_weights=class_weights,
            ce_weight=float(loss_cfg.get("ce_weight", 1.0)),
            dice_weight=float(loss_cfg.get("dice_weight", 1.0)),
        )
    raise ValueError(
        "Unsupported loss.type: "
        f"'{loss_type}'. Supported: cross_entropy, ce_dice."
    )


def build_model_from_config(cfg: Dict[str, Any]) -> nn.Module:
    """根据配置构建正式训练模型。"""
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    architecture = str(model_cfg.get("architecture", "vit_seg_baseline")).lower()
    if architecture not in {"vit_seg_baseline", "timm_seg"}:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return TimmSegModel(
        backbone_name=str(model_cfg.get("backbone", "vit_base_patch16_224")),
        num_classes=int(data_cfg.get("num_classes", 10)),
        pretrained=bool(model_cfg.get("pretrained", True)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        input_size=int(data_cfg.get("patch_size", 512)),
        decoder_head=str(model_cfg.get("decoder_head", "simple_fpn_head")),
        drop_path_rate=float(model_cfg.get("drop_path_rate", 0.0)),
    )


class SegmentationTileDataset(Dataset):
    """基于 tile manifest 的遥感语义分割数据集。"""

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
        ignore_lum_ids: Optional[List[int]] = None,
        enable_augment: bool = False,
        augment_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.df = manifest_df.reset_index(drop=True)
        self.dataset_root = dataset_root
        self.image_dirname = image_dirname
        self.label_dirname = label_dirname
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        raw_ignore_lum_ids = [int(x) for x in (ignore_lum_ids or [])]
        self.ignore_lum_ids = sorted({x for x in raw_ignore_lum_ids if x >= 1})
        self._ignore_lum_ids_np = (
            np.asarray(self.ignore_lum_ids, dtype=np.int64) if self.ignore_lum_ids else None
        )
        self.enable_augment = bool(enable_augment)
        self.augment_cfg = augment_cfg or {}

    def __len__(self) -> int:
        return len(self.df)

    def _apply_train_augment(self, image: torch.Tensor, label: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """训练期增强（仅 train 启用）。

        说明：
        1. 空间变换同步作用于影像与标签，保持像元一一对应。
        2. 光谱/噪声增强仅作用于影像，不改变标签语义。
        3. 最终将影像裁剪到 [0, 1]，保持输入范围稳定。
        """
        cfg = self.augment_cfg

        if torch.rand(1).item() < float(cfg.get("hflip_prob", 0.5)):
            image = torch.flip(image, dims=[2])
            label = torch.flip(label, dims=[1])

        if torch.rand(1).item() < float(cfg.get("vflip_prob", 0.5)):
            image = torch.flip(image, dims=[1])
            label = torch.flip(label, dims=[0])

        if torch.rand(1).item() < float(cfg.get("rot90_prob", 0.5)):
            k = int(torch.randint(low=1, high=4, size=(1,)).item())
            image = torch.rot90(image, k=k, dims=[1, 2])
            label = torch.rot90(label, k=k, dims=[0, 1])

        if torch.rand(1).item() < float(cfg.get("color_jitter_prob", 0.5)):
            brightness_delta = float(cfg.get("brightness_delta", 0.10))
            contrast_delta = float(cfg.get("contrast_delta", 0.10))
            brightness_factor = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * brightness_delta
            contrast_factor = 1.0 + (2.0 * torch.rand(1).item() - 1.0) * contrast_delta

            channel_mean = image.mean(dim=(1, 2), keepdim=True)
            image = (image - channel_mean) * contrast_factor + channel_mean
            image = image * brightness_factor

        if torch.rand(1).item() < float(cfg.get("channel_scale_prob", 0.25)):
            channel_scale_delta = float(cfg.get("channel_scale_delta", 0.08))
            scale = 1.0 + (torch.rand((image.shape[0], 1, 1)) * 2.0 - 1.0) * channel_scale_delta
            image = image * scale

        if torch.rand(1).item() < float(cfg.get("gaussian_noise_prob", 0.20)):
            noise_std = float(cfg.get("gaussian_noise_std", 0.02))
            image = image + torch.randn_like(image) * noise_std

        image = torch.clamp(image, 0.0, 1.0)
        return image, label

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """读取单个 tile 样本并完成标签映射。"""
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

        if image.max() > 1.0:
            image = image / 255.0

        mapped_label = np.full_like(label, fill_value=self.ignore_index, dtype=np.int64)
        valid_mask = (label >= 1) & (label <= self.num_classes)
        if self._ignore_lum_ids_np is not None:
            valid_mask &= ~np.isin(label, self._ignore_lum_ids_np)
        mapped_label[valid_mask] = label[valid_mask] - 1

        image_tensor = torch.from_numpy(image)
        label_tensor = torch.from_numpy(mapped_label)
        if self.enable_augment:
            image_tensor, label_tensor = self._apply_train_augment(image=image_tensor, label=label_tensor)

        return {
            "image": image_tensor,
            "label": label_tensor,
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
    """根据当前 batch 预测更新混淆矩阵。"""
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
    """由混淆矩阵计算 mIoU、mF1、OA 与每类指标。"""
    confusion = confusion.to(torch.float64)
    tp = torch.diag(confusion)
    pos_gt = confusion.sum(dim=1)
    pos_pred = confusion.sum(dim=0)
    union = pos_gt + pos_pred - tp
    denom_f1 = 2.0 * tp + (pos_pred - tp) + (pos_gt - tp)

    iou = torch.where(union > 0, tp / union, torch.nan)
    f1 = torch.where(denom_f1 > 0, 2.0 * tp / denom_f1, torch.nan)
    present_mask = pos_gt > 0

    miou = torch.nanmean(iou[present_mask]).item() if present_mask.any() else float("nan")
    mf1 = torch.nanmean(f1[present_mask]).item() if present_mask.any() else float("nan")
    overall_acc = (tp.sum() / confusion.sum()).item() if confusion.sum() > 0 else 0.0

    return {
        "miou": float(miou),
        "mf1": float(mf1),
        "overall_accuracy": float(overall_acc),
        "per_class_iou": [float(x) for x in iou.tolist()],
        "per_class_f1": [float(x) for x in f1.tolist()],
        "gt_pixels_per_class": [int(x) for x in pos_gt.tolist()],
        "pred_pixels_per_class": [int(x) for x in pos_pred.tolist()],
        "tp_pixels_per_class": [int(x) for x in tp.tolist()],
        "present_mask": [bool(x) for x in present_mask.tolist()],
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    """写入 CSV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """写入 JSON 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    """写入 YAML 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


@dataclass
class TrainState:
    """训练状态快照。"""

    epoch: int
    global_step: int
    best_val_miou: float
    best_epoch: int


def save_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_state: TrainState,
    config: Dict[str, Any],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> None:
    """保存模型训练 checkpoint。"""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": train_state.epoch,
        "global_step": train_state.global_step,
        "best_val_miou": train_state.best_val_miou,
        "best_epoch": train_state.best_epoch,
        "config": config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """加载 checkpoint 并返回原始 payload。"""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid checkpoint payload type: {type(payload)}")
    return payload
