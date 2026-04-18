#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：05_vit_dataset_dry_run.py

功能：
1) 基于 1m 对齐影像-标签对，执行 ViT 数据集切片与 split 的 dry-run（不落盘影像/标签 patch）；
2) 按 11block 与 2swd 的既定空间策略进行 train/val/test 分配；
3) 执行 split 边界缓冲带剔除与 valid_ratio 过滤；
4) 输出 tiles_manifest/split_summary/eco_split_geometry/eco_split_class_stats；
5) 对照当前 AOI 阶段闸门输出通过性与调参建议。
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final
import yaml

try:
    from osgeo import gdal, osr
except Exception as import_exc:  # pragma: no cover
    gdal = None
    osr = None
    GDAL_IMPORT_ERROR = import_exc
else:
    GDAL_IMPORT_ERROR = None


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DEFAULT_IMAGERY_11BLOCK: Final[Path] = (
    PROJECT_ROOT / "data" / "interim" / "imagery_1m_aligned" / "tdop_11block_1m_aligned.tif"
)
DEFAULT_LABEL_11BLOCK: Final[Path] = (
    PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_11block_1m_aligned.tif"
)
DEFAULT_IMAGERY_2SWD: Final[Path] = (
    PROJECT_ROOT / "data" / "interim" / "imagery_1m_aligned" / "tdop_2swd_1m_aligned.tif"
)
DEFAULT_LABEL_2SWD: Final[Path] = (
    PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_2swd_1m_aligned.tif"
)
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v1" / "manifests"

DEFAULT_PATCH_SIZE: Final[int] = 512
# 2026-04-13 推荐参数组 A（在不扩 AOI 前提下）
DEFAULT_TRAIN_STRIDE: Final[int] = 128
DEFAULT_EVAL_STRIDE_11BLOCK: Final[int] = 256
DEFAULT_EVAL_STRIDE_2SWD_LEFT: Final[int] = 128
DEFAULT_BUFFER_M: Final[float] = 512.0

DEFAULT_LABEL_NODATA: Final[int] = 255
DEFAULT_VALID_RATIO_TRAIN: Final[float] = 0.2
DEFAULT_VALID_RATIO_VALTEST: Final[float] = 0.1
DEFAULT_VALID_RATIO_TRAIN_ECO: Final[float] = 0.3

DEFAULT_SPLIT11_TRAIN_RATIO: Final[float] = 0.70
DEFAULT_SPLIT11_VAL_RATIO: Final[float] = 0.15
DEFAULT_SPLIT2_LEFT_RATIO: Final[float] = 0.30
DEFAULT_SPLIT2_LEFT_VAL_RATIO: Final[float] = 0.50

DEFAULT_GATE_TRAIN: Final[int] = 2000
DEFAULT_GATE_VAL: Final[int] = 200
DEFAULT_GATE_TEST_IN_DOMAIN: Final[int] = 200
DEFAULT_GATE_TEST_ECO_HOLDOUT: Final[int] = 30

LUM_CLASSES: Final[list[int]] = list(range(1, 11))


@dataclass(frozen=True)
class RasterPair:
    pair_name: str
    imagery_path: Path
    label_path: Path


@dataclass
class SplitAccumulator:
    tile_count: int = 0
    pixel_count: int = 0
    valid_pixels: int = 0
    nodata_pixels: int = 0
    class_pixels: Counter[int] = field(default_factory=Counter)

    def update(self, class_counts: dict[int, int], patch_pixels: int, nodata_value: int) -> None:
        self.tile_count += 1
        self.pixel_count += patch_pixels
        nodata = int(class_counts.get(nodata_value, 0))
        self.nodata_pixels += nodata
        self.valid_pixels += patch_pixels - nodata
        for cls in LUM_CLASSES:
            self.class_pixels[cls] += int(class_counts.get(cls, 0))


def require_deps() -> None:
    if gdal is None or osr is None:
        detail = f"原始错误：{GDAL_IMPORT_ERROR}" if GDAL_IMPORT_ERROR else "未提供底层异常信息。"
        raise RuntimeError(
            "未检测到 GDAL Python 绑定（osgeo）。\n"
            "请在 GIS conda 环境中运行，例如：conda install -c conda-forge gdal\n"
            f"{detail}"
        )


def to_abs_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_yaml_config(path_ref: str | None) -> dict[str, object]:
    """读取 YAML 配置文件。"""
    if not path_ref:
        return {}
    cfg_path = to_abs_path(path_ref)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是映射(dict)：{cfg_path}")
    return payload


def approx_equal(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol


def spatial_ref_equal(wkt_a: str, wkt_b: str) -> bool:
    if not wkt_a or not wkt_b:
        return False
    srs_a = osr.SpatialReference()
    srs_b = osr.SpatialReference()
    if srs_a.ImportFromWkt(wkt_a) != 0 or srs_b.ImportFromWkt(wkt_b) != 0:
        return False
    return bool(srs_a.IsSame(srs_b))


def dataset_bounds(ds: gdal.Dataset) -> tuple[float, float, float, float]:
    gt = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    minx = gt[0]
    maxy = gt[3]
    maxx = minx + gt[1] * xsize
    miny = maxy + gt[5] * ysize
    if maxx < minx:
        minx, maxx = maxx, minx
    if maxy < miny:
        miny, maxy = maxy, miny
    return (minx, miny, maxx, maxy)


def validate_pair_alignment(pair: RasterPair) -> dict[str, object]:
    img_ds = gdal.Open(str(pair.imagery_path), gdal.GA_ReadOnly)
    if img_ds is None:
        raise RuntimeError(f"无法打开影像：{pair.imagery_path}")
    lab_ds = gdal.Open(str(pair.label_path), gdal.GA_ReadOnly)
    if lab_ds is None:
        img_ds = None
        raise RuntimeError(f"无法打开标签：{pair.label_path}")

    img_gt = img_ds.GetGeoTransform()
    lab_gt = lab_ds.GetGeoTransform()
    img_proj = img_ds.GetProjection()
    lab_proj = lab_ds.GetProjection()

    img_shape = (img_ds.RasterYSize, img_ds.RasterXSize)
    lab_shape = (lab_ds.RasterYSize, lab_ds.RasterXSize)
    if img_shape != lab_shape:
        raise RuntimeError(f"{pair.pair_name} 影像/标签 shape 不一致：{img_shape} vs {lab_shape}")

    if not spatial_ref_equal(img_proj, lab_proj):
        raise RuntimeError(f"{pair.pair_name} 影像/标签 CRS 不一致。")

    for i, (a, b) in enumerate(zip(img_gt, lab_gt)):
        if not approx_equal(float(a), float(b), tol=1e-8):
            raise RuntimeError(f"{pair.pair_name} 影像/标签 transform 不一致（index={i}）。")

    if img_gt[1] <= 0 or img_gt[5] >= 0 or not approx_equal(img_gt[2], 0.0) or not approx_equal(img_gt[4], 0.0):
        raise RuntimeError(f"{pair.pair_name} 当前仅支持 north-up 栅格（gt2=gt4=0, gt1>0, gt5<0）。")

    bounds = dataset_bounds(lab_ds)
    width_m = bounds[2] - bounds[0]
    height_m = bounds[3] - bounds[1]
    res = (float(lab_gt[1]), abs(float(lab_gt[5])))

    img_ds = None
    lab_ds = None
    return {
        "shape": lab_shape,
        "bounds": bounds,
        "resolution": res,
        "width_m": width_m,
        "height_m": height_m,
        "transform": lab_gt,
    }


def iter_windows(width: int, height: int, patch_size: int, stride: int):
    max_row = height - patch_size
    max_col = width - patch_size
    if max_row < 0 or max_col < 0:
        return
    for row_off in range(0, max_row + 1, stride):
        for col_off in range(0, max_col + 1, stride):
            yield row_off, col_off


def window_bounds(
    gt: tuple[float, float, float, float, float, float],
    row_off: int,
    col_off: int,
    patch_size: int,
) -> tuple[float, float, float, float]:
    minx = gt[0] + gt[1] * col_off
    maxx = gt[0] + gt[1] * (col_off + patch_size)
    maxy = gt[3] + gt[5] * row_off
    miny = gt[3] + gt[5] * (row_off + patch_size)
    if maxx < minx:
        minx, maxx = maxx, minx
    if maxy < miny:
        miny, maxy = maxy, miny
    return (minx, miny, maxx, maxy)


def bbox_intersects_vertical_buffer(minx: float, maxx: float, line_x: float, buffer_m: float) -> bool:
    return minx < (line_x + buffer_m) and maxx > (line_x - buffer_m)


def bbox_intersects_horizontal_buffer(miny: float, maxy: float, line_y: float, buffer_m: float) -> bool:
    return miny < (line_y + buffer_m) and maxy > (line_y - buffer_m)


def split_11block(
    center_x: float,
    bbox: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    train_ratio: float,
    val_ratio: float,
    buffer_m: float,
) -> tuple[str | None, str | None, str | None, dict[str, float]]:
    minx, _, maxx, _ = bounds
    width = maxx - minx
    x_line_1 = minx + train_ratio * width
    x_line_2 = minx + (train_ratio + val_ratio) * width

    if bbox_intersects_vertical_buffer(bbox[0], bbox[2], x_line_1, buffer_m):
        return None, None, "buffer_11block_xline1", {"x_line_1": x_line_1, "x_line_2": x_line_2}
    if bbox_intersects_vertical_buffer(bbox[0], bbox[2], x_line_2, buffer_m):
        return None, None, "buffer_11block_xline2", {"x_line_1": x_line_1, "x_line_2": x_line_2}

    if center_x <= x_line_1:
        return "train_11block", "train", None, {"x_line_1": x_line_1, "x_line_2": x_line_2}
    if center_x <= x_line_2:
        return "val_11block", "val", None, {"x_line_1": x_line_1, "x_line_2": x_line_2}
    return "test_in_domain", "test_in_domain", None, {"x_line_1": x_line_1, "x_line_2": x_line_2}


def split_2swd(
    center_x: float,
    center_y: float,
    bbox: tuple[float, float, float, float],
    bounds: tuple[float, float, float, float],
    left_ratio: float,
    left_val_ratio: float,
    buffer_m: float,
) -> tuple[str | None, str | None, str | None, dict[str, float]]:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    x_split = minx + left_ratio * width
    y_split = miny + left_val_ratio * height

    if bbox_intersects_vertical_buffer(bbox[0], bbox[2], x_split, buffer_m):
        return None, None, "buffer_2swd_xsplit", {"x_split": x_split, "y_split": y_split}

    if center_x > x_split:
        return "train_eco_support", "train", None, {"x_split": x_split, "y_split": y_split}

    if bbox_intersects_horizontal_buffer(bbox[1], bbox[3], y_split, buffer_m):
        return None, None, "buffer_2swd_ysplit", {"x_split": x_split, "y_split": y_split}

    if center_y <= y_split:
        return "val_eco", "val", None, {"x_split": x_split, "y_split": y_split}
    return "test_eco_holdout", "test_eco_holdout", None, {"x_split": x_split, "y_split": y_split}


def compute_class_counts(raw_bytes: bytes, nodata_value: int) -> tuple[dict[int, int], int, int, float, float]:
    pix_counter = Counter(raw_bytes)
    class_counts: dict[int, int] = {}
    for cls in LUM_CLASSES:
        class_counts[cls] = int(pix_counter.get(cls, 0))
    class_counts[nodata_value] = int(pix_counter.get(nodata_value, 0))

    total = int(len(raw_bytes))
    nodata_pixels = class_counts[nodata_value]
    valid_pixels = total - nodata_pixels
    valid_ratio = valid_pixels / total if total > 0 else 0.0
    unknown_ratio = nodata_pixels / total if total > 0 else 1.0
    return class_counts, valid_pixels, nodata_pixels, valid_ratio, unknown_ratio


def valid_ratio_threshold_for_tile(
    final_split: str,
    source_subsplit: str,
    valid_ratio_train: float,
    valid_ratio_valtest: float,
    valid_ratio_train_eco: float,
) -> float:
    if final_split == "train":
        if source_subsplit == "train_eco_support":
            return valid_ratio_train_eco
        return valid_ratio_train
    return valid_ratio_valtest


def write_tiles_manifest(rows: list[dict[str, object]], out_csv: Path) -> None:
    ensure_parent_dir(out_csv)
    headers = [
        "tile_id",
        "pair_name",
        "source_subsplit",
        "final_split",
        "row_off",
        "col_off",
        "patch_size",
        "stride",
        "window_minx",
        "window_miny",
        "window_maxx",
        "window_maxy",
        "center_x",
        "center_y",
        "valid_pixels",
        "nodata_pixels",
        "valid_ratio",
        "unknown_ratio",
    ] + [f"class_{cls}" for cls in LUM_CLASSES] + ["class_255"]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_split_summary(
    acc_map: dict[str, SplitAccumulator],
    out_csv: Path,
    gates: dict[str, int],
) -> None:
    ensure_parent_dir(out_csv)
    headers = [
        "split",
        "tile_count",
        "pixel_count",
        "valid_pixels",
        "nodata_pixels",
        "unknown_ratio",
        "gate_min",
        "gate_pass",
    ] + [f"class_{cls}" for cls in LUM_CLASSES]
    order = ["train", "val", "test_in_domain", "test_eco_holdout"]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for split in order:
            acc = acc_map.get(split, SplitAccumulator())
            unknown_ratio = (acc.nodata_pixels / acc.pixel_count) if acc.pixel_count > 0 else 0.0
            row = {
                "split": split,
                "tile_count": acc.tile_count,
                "pixel_count": acc.pixel_count,
                "valid_pixels": acc.valid_pixels,
                "nodata_pixels": acc.nodata_pixels,
                "unknown_ratio": f"{unknown_ratio:.6f}",
                "gate_min": gates.get(split, 0),
                "gate_pass": int(acc.tile_count >= gates.get(split, 0)),
            }
            for cls in LUM_CLASSES:
                row[f"class_{cls}"] = int(acc.class_pixels.get(cls, 0))
            w.writerow(row)


def write_eco_split_geometry(
    out_csv: Path,
    pair_2swd: RasterPair,
    bounds_2swd: tuple[float, float, float, float],
    x_split: float,
    y_split: float,
    args: argparse.Namespace,
) -> None:
    ensure_parent_dir(out_csv)
    headers = [
        "pair_name",
        "imagery_path",
        "label_path",
        "minx",
        "miny",
        "maxx",
        "maxy",
        "x_split",
        "y_split_left",
        "left_ratio",
        "right_ratio",
        "left_val_ratio",
        "left_test_ratio",
        "buffer_m",
        "patch_size",
        "train_stride",
        "eval_stride_11block",
        "eval_stride_2swd_left",
    ]
    minx, miny, maxx, maxy = bounds_2swd
    row = {
        "pair_name": pair_2swd.pair_name,
        "imagery_path": str(pair_2swd.imagery_path),
        "label_path": str(pair_2swd.label_path),
        "minx": f"{minx:.6f}",
        "miny": f"{miny:.6f}",
        "maxx": f"{maxx:.6f}",
        "maxy": f"{maxy:.6f}",
        "x_split": f"{x_split:.6f}",
        "y_split_left": f"{y_split:.6f}",
        "left_ratio": f"{args.split2_left_ratio:.6f}",
        "right_ratio": f"{(1.0 - args.split2_left_ratio):.6f}",
        "left_val_ratio": f"{args.split2_left_val_ratio:.6f}",
        "left_test_ratio": f"{(1.0 - args.split2_left_val_ratio):.6f}",
        "buffer_m": f"{args.buffer_m:.2f}",
        "patch_size": args.patch_size,
        "train_stride": args.train_stride,
        "eval_stride_11block": args.eval_stride_11block,
        "eval_stride_2swd_left": args.eval_stride_2swd_left,
    }
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerow(row)


def write_eco_split_class_stats(
    acc_map: dict[str, SplitAccumulator],
    out_csv: Path,
) -> None:
    ensure_parent_dir(out_csv)
    headers = [
        "source_subsplit",
        "final_split",
        "tile_count",
        "pixel_count",
        "valid_pixels",
        "nodata_pixels",
        "unknown_ratio",
    ] + [f"class_{cls}" for cls in LUM_CLASSES]
    order = ["train_eco_support", "val_eco", "test_eco_holdout"]
    final_map = {
        "train_eco_support": "train",
        "val_eco": "val",
        "test_eco_holdout": "test_eco_holdout",
    }
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for key in order:
            acc = acc_map.get(key, SplitAccumulator())
            unknown_ratio = (acc.nodata_pixels / acc.pixel_count) if acc.pixel_count > 0 else 0.0
            row = {
                "source_subsplit": key,
                "final_split": final_map[key],
                "tile_count": acc.tile_count,
                "pixel_count": acc.pixel_count,
                "valid_pixels": acc.valid_pixels,
                "nodata_pixels": acc.nodata_pixels,
                "unknown_ratio": f"{unknown_ratio:.6f}",
            }
            for cls in LUM_CLASSES:
                row[f"class_{cls}"] = int(acc.class_pixels.get(cls, 0))
            w.writerow(row)


def suggest_next_tuning(split_counts: dict[str, int], gates: dict[str, int]) -> list[str]:
    suggestions: list[str] = []
    if split_counts.get("test_eco_holdout", 0) < gates["test_eco_holdout"]:
        suggestions.append("test_eco_holdout 不足：优先下调 --eval-stride-2swd-left（例如 128 -> 96 或 64）。")
    if split_counts.get("val", 0) < gates["val"] or split_counts.get("test_in_domain", 0) < gates["test_in_domain"]:
        suggestions.append("val/test_in_domain 不足：下调 --eval-stride-11block（例如 256 -> 192 或 128）。")
    if split_counts.get("train", 0) < gates["train"]:
        suggestions.append("train 不足：下调 --train-stride（例如 128 -> 96），或将 --patch-size 调整为 384/256。")
    if not suggestions:
        suggestions.append("当前参数已满足 AOI 闸门；可进入落盘切片脚本实现。")
    return suggestions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ViT 数据集切片与 split dry-run（仅统计，不落盘 patch）")
    parser.add_argument("--config", default="", help="YAML 配置文件路径（可选）")
    parser.add_argument("--imagery-11block", default=str(DEFAULT_IMAGERY_11BLOCK), help="11block 1m 对齐影像")
    parser.add_argument("--label-11block", default=str(DEFAULT_LABEL_11BLOCK), help="11block 1m 对齐标签")
    parser.add_argument("--imagery-2swd", default=str(DEFAULT_IMAGERY_2SWD), help="2swd 1m 对齐影像")
    parser.add_argument("--label-2swd", default=str(DEFAULT_LABEL_2SWD), help="2swd 1m 对齐标签")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="CSV 输出目录")

    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="patch 尺寸（像元）")
    parser.add_argument("--train-stride", type=int, default=DEFAULT_TRAIN_STRIDE, help="训练 stride")
    parser.add_argument("--eval-stride-11block", type=int, default=DEFAULT_EVAL_STRIDE_11BLOCK, help="11block 验证/测试 stride")
    parser.add_argument("--eval-stride-2swd-left", type=int, default=DEFAULT_EVAL_STRIDE_2SWD_LEFT, help="2swd 左侧验证/测试 stride")
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M, help="split 边界缓冲带宽度（米）")
    parser.add_argument("--label-nodata", type=int, default=DEFAULT_LABEL_NODATA, help="标签 nodata 值")

    parser.add_argument("--valid-ratio-train", type=float, default=DEFAULT_VALID_RATIO_TRAIN, help="train 最低 valid_ratio")
    parser.add_argument("--valid-ratio-valtest", type=float, default=DEFAULT_VALID_RATIO_VALTEST, help="val/test 最低 valid_ratio")
    parser.add_argument("--valid-ratio-train-eco", type=float, default=DEFAULT_VALID_RATIO_TRAIN_ECO, help="train_eco_support 最低 valid_ratio")

    parser.add_argument("--split11-train-ratio", type=float, default=DEFAULT_SPLIT11_TRAIN_RATIO, help="11block train 比例")
    parser.add_argument("--split11-val-ratio", type=float, default=DEFAULT_SPLIT11_VAL_RATIO, help="11block val 比例")
    parser.add_argument("--split2-left-ratio", type=float, default=DEFAULT_SPLIT2_LEFT_RATIO, help="2swd 左侧比例")
    parser.add_argument("--split2-left-val-ratio", type=float, default=DEFAULT_SPLIT2_LEFT_VAL_RATIO, help="2swd 左侧内部 val 比例")

    parser.add_argument("--gate-train", type=int, default=DEFAULT_GATE_TRAIN, help="train 最低闸门")
    parser.add_argument("--gate-val", type=int, default=DEFAULT_GATE_VAL, help="val 最低闸门")
    parser.add_argument("--gate-test-in-domain", type=int, default=DEFAULT_GATE_TEST_IN_DOMAIN, help="test_in_domain 最低闸门")
    parser.add_argument("--gate-test-eco-holdout", type=int, default=DEFAULT_GATE_TEST_ECO_HOLDOUT, help="test_eco_holdout 最低闸门")

    parser.add_argument("--max-tiles", type=int, default=0, help="仅处理前 N 个候选窗口（0 表示不限制）")
    parser.add_argument("--no-overwrite", action="store_true", help="输出 CSV 已存在时不覆盖")
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="YAML 配置文件路径（可选）")
    pre_args, remaining = pre_parser.parse_known_args()

    args = build_parser().parse_args(remaining)
    require_deps()

    # 规则：命令行显式传入的参数优先，YAML 仅补充未显式给出的参数。
    provided_flags = {
        token[2:].replace("-", "_")
        for token in remaining
        if token.startswith("--")
    }
    yaml_cfg = load_yaml_config(pre_args.config)
    for key, value in yaml_cfg.items():
        attr = str(key).replace("-", "_")
        if hasattr(args, attr) and attr not in provided_flags:
            setattr(args, attr, value)
    if pre_args.config:
        print(f"[config] loaded: {to_abs_path(pre_args.config)}")

    if args.patch_size <= 0:
        raise ValueError("--patch-size 必须大于 0")
    if args.train_stride <= 0 or args.eval_stride_11block <= 0 or args.eval_stride_2swd_left <= 0:
        raise ValueError("stride 参数必须大于 0")
    if args.buffer_m < 0:
        raise ValueError("--buffer-m 不能小于 0")
    if not (0 <= args.label_nodata <= 255):
        raise ValueError("--label-nodata 必须位于 0~255")

    for ratio_name in [
        "valid_ratio_train",
        "valid_ratio_valtest",
        "valid_ratio_train_eco",
        "split11_train_ratio",
        "split11_val_ratio",
        "split2_left_ratio",
        "split2_left_val_ratio",
    ]:
        val = float(getattr(args, ratio_name))
        if val <= 0 or val >= 1:
            raise ValueError(f"--{ratio_name.replace('_', '-')} 必须位于 (0,1)")
    if args.split11_train_ratio + args.split11_val_ratio >= 1:
        raise ValueError("--split11-train-ratio + --split11-val-ratio 必须小于 1")

    pair_11block = RasterPair(
        pair_name="11block",
        imagery_path=to_abs_path(args.imagery_11block),
        label_path=to_abs_path(args.label_11block),
    )
    pair_2swd = RasterPair(
        pair_name="2swd",
        imagery_path=to_abs_path(args.imagery_2swd),
        label_path=to_abs_path(args.label_2swd),
    )
    output_dir = to_abs_path(args.output_dir)

    for p in [pair_11block.imagery_path, pair_11block.label_path, pair_2swd.imagery_path, pair_2swd.label_path]:
        if not p.exists():
            raise FileNotFoundError(f"输入文件不存在：{p}")

    out_tiles_manifest = output_dir / "tiles_manifest.csv"
    out_split_summary = output_dir / "split_summary.csv"
    out_eco_geom = output_dir / "eco_split_geometry.csv"
    out_eco_stats = output_dir / "eco_split_class_stats.csv"
    if args.no_overwrite:
        for out_path in [out_tiles_manifest, out_split_summary, out_eco_geom, out_eco_stats]:
            if out_path.exists():
                raise FileExistsError(f"输出文件已存在：{out_path}")

    meta_11 = validate_pair_alignment(pair_11block)
    meta_2 = validate_pair_alignment(pair_2swd)

    print("[meta] alignment check passed")
    print(f"  - 11block shape={meta_11['shape']}, res={meta_11['resolution']}, bounds={meta_11['bounds']}")
    print(f"  - 2swd    shape={meta_2['shape']}, res={meta_2['resolution']}, bounds={meta_2['bounds']}")
    print(
        "[params] "
        f"patch={args.patch_size}, train_stride={args.train_stride}, "
        f"eval_stride_11block={args.eval_stride_11block}, eval_stride_2swd_left={args.eval_stride_2swd_left}, "
        f"buffer={args.buffer_m}m"
    )

    gates = {
        "train": args.gate_train,
        "val": args.gate_val,
        "test_in_domain": args.gate_test_in_domain,
        "test_eco_holdout": args.gate_test_eco_holdout,
    }

    label_ds_map = {
        "11block": gdal.Open(str(pair_11block.label_path), gdal.GA_ReadOnly),
        "2swd": gdal.Open(str(pair_2swd.label_path), gdal.GA_ReadOnly),
    }
    for k, ds in label_ds_map.items():
        if ds is None:
            raise RuntimeError(f"无法打开标签数据集：{k}")

    accepted_rows: list[dict[str, object]] = []
    rejected_counter: Counter[str] = Counter()
    source_pass_counter: Counter[str] = Counter()
    split_acc: dict[str, SplitAccumulator] = defaultdict(SplitAccumulator)
    eco_acc: dict[str, SplitAccumulator] = defaultdict(SplitAccumulator)
    seen_windows: set[tuple[str, int, int, int]] = set()

    split_lines_11 = {
        "x_line_1": meta_11["bounds"][0] + args.split11_train_ratio * meta_11["width_m"],
        "x_line_2": meta_11["bounds"][0] + (args.split11_train_ratio + args.split11_val_ratio) * meta_11["width_m"],
    }
    split_lines_2 = {
        "x_split": meta_2["bounds"][0] + args.split2_left_ratio * meta_2["width_m"],
        "y_split": meta_2["bounds"][1] + args.split2_left_val_ratio * meta_2["height_m"],
    }

    def process_candidate(
        pair: RasterPair,
        row_off: int,
        col_off: int,
        stride: int,
        source_pass: str,
    ) -> None:
        ds = label_ds_map[pair.pair_name]
        band = ds.GetRasterBand(1)
        gt = ds.GetGeoTransform()
        bbox = window_bounds(gt, row_off=row_off, col_off=col_off, patch_size=args.patch_size)
        center_x = (bbox[0] + bbox[2]) * 0.5
        center_y = (bbox[1] + bbox[3]) * 0.5

        if pair.pair_name == "11block":
            source_subsplit, final_split, reject_reason, _ = split_11block(
                center_x=center_x,
                bbox=bbox,
                bounds=meta_11["bounds"],
                train_ratio=args.split11_train_ratio,
                val_ratio=args.split11_val_ratio,
                buffer_m=args.buffer_m,
            )
        else:
            source_subsplit, final_split, reject_reason, _ = split_2swd(
                center_x=center_x,
                center_y=center_y,
                bbox=bbox,
                bounds=meta_2["bounds"],
                left_ratio=args.split2_left_ratio,
                left_val_ratio=args.split2_left_val_ratio,
                buffer_m=args.buffer_m,
            )

        if reject_reason is not None:
            rejected_counter[reject_reason] += 1
            return
        if source_subsplit is None or final_split is None:
            rejected_counter["no_split_assignment"] += 1
            return
        if source_pass == "train" and final_split != "train":
            return
        if source_pass == "eval" and final_split not in {"val", "test_in_domain", "test_eco_holdout"}:
            return

        window_key = (pair.pair_name, row_off, col_off, args.patch_size)
        if window_key in seen_windows:
            rejected_counter["duplicate_window"] += 1
            return
        seen_windows.add(window_key)

        raw = band.ReadRaster(
            xoff=col_off,
            yoff=row_off,
            xsize=args.patch_size,
            ysize=args.patch_size,
            buf_xsize=args.patch_size,
            buf_ysize=args.patch_size,
            buf_type=gdal.GDT_Byte,
        )
        if raw is None:
            rejected_counter["read_raster_failed"] += 1
            return

        class_counts, valid_pixels, nodata_pixels, valid_ratio, unknown_ratio = compute_class_counts(
            raw_bytes=raw,
            nodata_value=args.label_nodata,
        )

        threshold = valid_ratio_threshold_for_tile(
            final_split=final_split,
            source_subsplit=source_subsplit,
            valid_ratio_train=args.valid_ratio_train,
            valid_ratio_valtest=args.valid_ratio_valtest,
            valid_ratio_train_eco=args.valid_ratio_train_eco,
        )

        if valid_pixels == 0:
            rejected_counter["all_nodata"] += 1
            return
        if valid_ratio < threshold:
            rejected_counter["valid_ratio_below_threshold"] += 1
            return

        patch_pixels = args.patch_size * args.patch_size
        tile_id = f"{pair.pair_name}_{final_split}_r{row_off:05d}_c{col_off:05d}"
        row = {
            "tile_id": tile_id,
            "pair_name": pair.pair_name,
            "source_subsplit": source_subsplit,
            "final_split": final_split,
            "row_off": row_off,
            "col_off": col_off,
            "patch_size": args.patch_size,
            "stride": stride,
            "window_minx": f"{bbox[0]:.3f}",
            "window_miny": f"{bbox[1]:.3f}",
            "window_maxx": f"{bbox[2]:.3f}",
            "window_maxy": f"{bbox[3]:.3f}",
            "center_x": f"{center_x:.3f}",
            "center_y": f"{center_y:.3f}",
            "valid_pixels": valid_pixels,
            "nodata_pixels": nodata_pixels,
            "valid_ratio": f"{valid_ratio:.6f}",
            "unknown_ratio": f"{unknown_ratio:.6f}",
            "class_255": class_counts.get(args.label_nodata, 0),
        }
        for cls in LUM_CLASSES:
            row[f"class_{cls}"] = class_counts.get(cls, 0)
        accepted_rows.append(row)

        split_acc[final_split].update(class_counts, patch_pixels=patch_pixels, nodata_value=args.label_nodata)
        if source_subsplit in {"train_eco_support", "val_eco", "test_eco_holdout"}:
            eco_acc[source_subsplit].update(class_counts, patch_pixels=patch_pixels, nodata_value=args.label_nodata)
        source_pass_counter[f"{pair.pair_name}:{source_pass}"] += 1

    total_candidates = 0
    max_tiles = int(args.max_tiles)

    # 11block: train pass
    ds_11 = label_ds_map["11block"]
    h11, w11 = ds_11.RasterYSize, ds_11.RasterXSize
    for r, c in iter_windows(w11, h11, args.patch_size, args.train_stride) or []:
        process_candidate(pair_11block, row_off=r, col_off=c, stride=args.train_stride, source_pass="train")
        total_candidates += 1
        if max_tiles > 0 and total_candidates >= max_tiles:
            break

    # 11block: eval pass
    if not (max_tiles > 0 and total_candidates >= max_tiles):
        for r, c in iter_windows(w11, h11, args.patch_size, args.eval_stride_11block) or []:
            process_candidate(pair_11block, row_off=r, col_off=c, stride=args.eval_stride_11block, source_pass="eval")
            total_candidates += 1
            if max_tiles > 0 and total_candidates >= max_tiles:
                break

    # 2swd: train pass
    if not (max_tiles > 0 and total_candidates >= max_tiles):
        ds_2 = label_ds_map["2swd"]
        h2, w2 = ds_2.RasterYSize, ds_2.RasterXSize
        for r, c in iter_windows(w2, h2, args.patch_size, args.train_stride) or []:
            process_candidate(pair_2swd, row_off=r, col_off=c, stride=args.train_stride, source_pass="train")
            total_candidates += 1
            if max_tiles > 0 and total_candidates >= max_tiles:
                break

    # 2swd: eval pass
    if not (max_tiles > 0 and total_candidates >= max_tiles):
        ds_2 = label_ds_map["2swd"]
        h2, w2 = ds_2.RasterYSize, ds_2.RasterXSize
        for r, c in iter_windows(w2, h2, args.patch_size, args.eval_stride_2swd_left) or []:
            process_candidate(pair_2swd, row_off=r, col_off=c, stride=args.eval_stride_2swd_left, source_pass="eval")
            total_candidates += 1
            if max_tiles > 0 and total_candidates >= max_tiles:
                break

    for ds in label_ds_map.values():
        ds = None

    accepted_rows.sort(key=lambda x: (x["final_split"], x["pair_name"], int(x["row_off"]), int(x["col_off"])))

    write_tiles_manifest(accepted_rows, out_tiles_manifest)
    write_split_summary(split_acc, out_split_summary, gates=gates)
    write_eco_split_geometry(
        out_csv=out_eco_geom,
        pair_2swd=pair_2swd,
        bounds_2swd=meta_2["bounds"],
        x_split=split_lines_2["x_split"],
        y_split=split_lines_2["y_split"],
        args=args,
    )
    write_eco_split_class_stats(eco_acc, out_eco_stats)

    split_counts = {
        "train": split_acc.get("train", SplitAccumulator()).tile_count,
        "val": split_acc.get("val", SplitAccumulator()).tile_count,
        "test_in_domain": split_acc.get("test_in_domain", SplitAccumulator()).tile_count,
        "test_eco_holdout": split_acc.get("test_eco_holdout", SplitAccumulator()).tile_count,
    }

    print("\n[result] split counts")
    print(f"  - train: {split_counts['train']} (gate>={gates['train']})")
    print(f"  - val: {split_counts['val']} (gate>={gates['val']})")
    print(f"  - test_in_domain: {split_counts['test_in_domain']} (gate>={gates['test_in_domain']})")
    print(f"  - test_eco_holdout: {split_counts['test_eco_holdout']} (gate>={gates['test_eco_holdout']})")
    print(
        "[result] gate pass = "
        f"{int(split_counts['train'] >= gates['train'] and split_counts['val'] >= gates['val'] and split_counts['test_in_domain'] >= gates['test_in_domain'] and split_counts['test_eco_holdout'] >= gates['test_eco_holdout'])}"
    )

    print("\n[split lines]")
    print(
        "  - 11block: "
        f"x_line_1={split_lines_11['x_line_1']:.3f}, "
        f"x_line_2={split_lines_11['x_line_2']:.3f}, buffer={args.buffer_m:.1f}m"
    )
    print(
        "  - 2swd: "
        f"x_split={split_lines_2['x_split']:.3f}, "
        f"y_split={split_lines_2['y_split']:.3f}, buffer={args.buffer_m:.1f}m"
    )

    print("\n[accept by pass]")
    for k in sorted(source_pass_counter):
        print(f"  - {k}: {source_pass_counter[k]}")

    print("\n[rejections]")
    if rejected_counter:
        for reason, count in rejected_counter.most_common():
            print(f"  - {reason}: {count}")
    else:
        print("  - none")

    print("\n[suggestions]")
    for s in suggest_next_tuning(split_counts=split_counts, gates=gates):
        print(f"  - {s}")

    print("\n[outputs]")
    print(f"  - {out_tiles_manifest}")
    print(f"  - {out_split_summary}")
    print(f"  - {out_eco_geom}")
    print(f"  - {out_eco_stats}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

