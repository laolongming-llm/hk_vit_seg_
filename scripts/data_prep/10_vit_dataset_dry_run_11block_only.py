#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：10_vit_dataset_dry_run_11block_only.py

功能：
1) 基于 11block 的 1m 对齐影像/标签执行 dry-run（仅生成清单，不落盘 patch）；
2) 在 11block 内完成 train/val/test 三分区；
3) 执行 split 边界缓冲带剔除与 valid_ratio 过滤；
4) 输出 tiles_manifest / split_summary / eco_split_geometry / eco_split_class_stats。
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
DEFAULT_OUTPUT_DIR: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v3_11block8" / "manifests"
)

DEFAULT_PATCH_SIZE: Final[int] = 512
DEFAULT_TRAIN_STRIDE: Final[int] = 128
DEFAULT_EVAL_STRIDE: Final[int] = 256
DEFAULT_BUFFER_M: Final[float] = 512.0

DEFAULT_LABEL_NODATA: Final[int] = 255
DEFAULT_VALID_RATIO_TRAIN: Final[float] = 0.2
DEFAULT_VALID_RATIO_VALTEST: Final[float] = 0.1

# 11block 内三分区：train / val / test
DEFAULT_SPLIT_TRAIN_RATIO: Final[float] = 0.70
DEFAULT_SPLIT_VAL_RATIO: Final[float] = 0.15

DEFAULT_GATE_TRAIN: Final[int] = 2000
DEFAULT_GATE_VAL: Final[int] = 200
DEFAULT_GATE_TEST: Final[int] = 200

LUM_CLASSES: Final[list[int]] = list(range(1, 9))
VALID_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")


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
    if gdal is not None and osr is not None:
        return
    detail = f"原始错误：{GDAL_IMPORT_ERROR}" if GDAL_IMPORT_ERROR else "未提供底层异常信息。"
    raise RuntimeError(
        "未检测到 GDAL Python 绑定（osgeo）。\n"
        "请在 GIS conda 环境运行，例如：conda install -c conda-forge gdal\n"
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
    if not path_ref:
        return {}
    cfg_path = to_abs_path(path_ref)
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须为映射(dict)：{cfg_path}")
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
            raise RuntimeError(f"{pair.pair_name} 影像/标签 transform 不一致(index={i})。")

    if img_gt[1] <= 0 or img_gt[5] >= 0 or not approx_equal(img_gt[2], 0.0) or not approx_equal(img_gt[4], 0.0):
        raise RuntimeError(f"{pair.pair_name} 当前仅支持 north-up 栅格。")

    bounds = dataset_bounds(lab_ds)
    width_m = bounds[2] - bounds[0]
    res = (float(lab_gt[1]), abs(float(lab_gt[5])))

    img_ds = None
    lab_ds = None
    return {
        "shape": lab_shape,
        "bounds": bounds,
        "resolution": res,
        "width_m": width_m,
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


def split_11block_threeway(
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
    return "test_11block", "test", None, {"x_line_1": x_line_1, "x_line_2": x_line_2}


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


def valid_ratio_threshold_for_tile(final_split: str, valid_ratio_train: float, valid_ratio_valtest: float) -> float:
    if final_split == "train":
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


def write_split_summary(acc_map: dict[str, SplitAccumulator], out_csv: Path, gates: dict[str, int]) -> None:
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

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for split in VALID_SPLITS:
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
    pair_11block: RasterPair,
    bounds_11block: tuple[float, float, float, float],
    split_lines: dict[str, float],
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
        "x_line_1",
        "x_line_2",
        "train_ratio",
        "val_ratio",
        "test_ratio",
        "buffer_m",
        "patch_size",
        "train_stride",
        "eval_stride",
    ]
    minx, miny, maxx, maxy = bounds_11block
    row = {
        "pair_name": pair_11block.pair_name,
        "imagery_path": str(pair_11block.imagery_path),
        "label_path": str(pair_11block.label_path),
        "minx": f"{minx:.6f}",
        "miny": f"{miny:.6f}",
        "maxx": f"{maxx:.6f}",
        "maxy": f"{maxy:.6f}",
        "x_line_1": f"{split_lines['x_line_1']:.6f}",
        "x_line_2": f"{split_lines['x_line_2']:.6f}",
        "train_ratio": f"{args.split_train_ratio:.6f}",
        "val_ratio": f"{args.split_val_ratio:.6f}",
        "test_ratio": f"{(1.0 - args.split_train_ratio - args.split_val_ratio):.6f}",
        "buffer_m": f"{args.buffer_m:.2f}",
        "patch_size": args.patch_size,
        "train_stride": args.train_stride,
        "eval_stride": args.eval_stride,
    }
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerow(row)


def write_eco_split_class_stats(split_acc: dict[str, SplitAccumulator], out_csv: Path) -> None:
    ensure_parent_dir(out_csv)
    headers = [
        "split",
        "tile_count",
        "pixel_count",
        "valid_pixels",
        "nodata_pixels",
        "unknown_ratio",
    ] + [f"class_{cls}" for cls in LUM_CLASSES]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for split in VALID_SPLITS:
            acc = split_acc.get(split, SplitAccumulator())
            unknown_ratio = (acc.nodata_pixels / acc.pixel_count) if acc.pixel_count > 0 else 0.0
            row = {
                "split": split,
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
    if split_counts.get("val", 0) < gates["val"] or split_counts.get("test", 0) < gates["test"]:
        suggestions.append("val/test 不足：下调 --eval-stride 或适当减小 --buffer-m。")
    if split_counts.get("train", 0) < gates["train"]:
        suggestions.append("train 不足：下调 --train-stride（例如 128 -> 96），或减小 --patch-size。")
    if not suggestions:
        suggestions.append("当前参数已满足三分区门槛，可进入导出 patch 阶段。")
    return suggestions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="11block-only 的 ViT 数据集 dry-run（仅统计，不落盘 patch）。")
    parser.add_argument("--config", default="", help="YAML 配置文件路径（可选）")
    parser.add_argument("--imagery-11block", default=str(DEFAULT_IMAGERY_11BLOCK), help="11block 1m 对齐影像")
    parser.add_argument("--label-11block", default=str(DEFAULT_LABEL_11BLOCK), help="11block 1m 对齐标签")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="CSV 输出目录")

    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE, help="patch 尺寸（像元）")
    parser.add_argument("--train-stride", type=int, default=DEFAULT_TRAIN_STRIDE, help="训练 stride")
    parser.add_argument("--eval-stride", type=int, default=DEFAULT_EVAL_STRIDE, help="验证/测试 stride")
    parser.add_argument("--buffer-m", type=float, default=DEFAULT_BUFFER_M, help="split 边界缓冲带宽度（米）")
    parser.add_argument("--label-nodata", type=int, default=DEFAULT_LABEL_NODATA, help="标签 nodata 值")

    parser.add_argument("--valid-ratio-train", type=float, default=DEFAULT_VALID_RATIO_TRAIN, help="train 最低 valid_ratio")
    parser.add_argument("--valid-ratio-valtest", type=float, default=DEFAULT_VALID_RATIO_VALTEST, help="val/test 最低 valid_ratio")

    parser.add_argument("--split-train-ratio", type=float, default=DEFAULT_SPLIT_TRAIN_RATIO, help="11block train 比例")
    parser.add_argument("--split-val-ratio", type=float, default=DEFAULT_SPLIT_VAL_RATIO, help="11block val 比例")

    parser.add_argument("--gate-train", type=int, default=DEFAULT_GATE_TRAIN, help="train 最低门槛")
    parser.add_argument("--gate-val", type=int, default=DEFAULT_GATE_VAL, help="val 最低门槛")
    parser.add_argument("--gate-test", type=int, default=DEFAULT_GATE_TEST, help="test 最低门槛")

    parser.add_argument("--max-tiles", type=int, default=0, help="仅处理前 N 个候选窗口（0 表示不限制）")
    parser.add_argument("--no-overwrite", action="store_true", help="输出 CSV 存在时不覆盖")
    return parser


def main() -> None:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", default="", help="YAML 配置文件路径（可选）")
    pre_args, remaining = pre_parser.parse_known_args()

    args = build_parser().parse_args(remaining)
    require_deps()

    provided_flags = {token[2:].replace("-", "_") for token in remaining if token.startswith("--")}
    yaml_cfg = load_yaml_config(pre_args.config)
    for key, value in yaml_cfg.items():
        attr = str(key).replace("-", "_")
        if hasattr(args, attr) and attr not in provided_flags:
            setattr(args, attr, value)
    if pre_args.config:
        print(f"[config] loaded: {to_abs_path(pre_args.config)}")

    if args.patch_size <= 0:
        raise ValueError("--patch-size 必须大于 0")
    if args.train_stride <= 0 or args.eval_stride <= 0:
        raise ValueError("stride 参数必须大于 0")
    if args.buffer_m < 0:
        raise ValueError("--buffer-m 不能小于 0")
    if not (0 <= args.label_nodata <= 255):
        raise ValueError("--label-nodata 必须位于 0~255")

    for ratio_name in ["valid_ratio_train", "valid_ratio_valtest", "split_train_ratio", "split_val_ratio"]:
        val = float(getattr(args, ratio_name))
        if val <= 0 or val >= 1:
            raise ValueError(f"--{ratio_name.replace('_', '-')} 必须位于 (0,1)")
    if args.split_train_ratio + args.split_val_ratio >= 1:
        raise ValueError("--split-train-ratio + --split-val-ratio 必须小于 1")

    pair_11block = RasterPair(
        pair_name="11block",
        imagery_path=to_abs_path(args.imagery_11block),
        label_path=to_abs_path(args.label_11block),
    )
    output_dir = to_abs_path(args.output_dir)

    for p in [pair_11block.imagery_path, pair_11block.label_path]:
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
    print("[meta] alignment check passed")
    print(f"  - 11block shape={meta_11['shape']}, res={meta_11['resolution']}, bounds={meta_11['bounds']}")
    print(
        "[params] "
        f"patch={args.patch_size}, train_stride={args.train_stride}, "
        f"eval_stride={args.eval_stride}, buffer={args.buffer_m}m"
    )

    gates = {
        "train": args.gate_train,
        "val": args.gate_val,
        "test": args.gate_test,
    }

    label_ds = gdal.Open(str(pair_11block.label_path), gdal.GA_ReadOnly)
    if label_ds is None:
        raise RuntimeError("无法打开标签数据集：11block")

    accepted_rows: list[dict[str, object]] = []
    rejected_counter: Counter[str] = Counter()
    source_pass_counter: Counter[str] = Counter()
    split_acc: dict[str, SplitAccumulator] = defaultdict(SplitAccumulator)
    seen_windows: set[tuple[str, int, int, int]] = set()

    split_lines = {
        "x_line_1": meta_11["bounds"][0] + args.split_train_ratio * meta_11["width_m"],
        "x_line_2": meta_11["bounds"][0] + (args.split_train_ratio + args.split_val_ratio) * meta_11["width_m"],
    }

    def process_candidate(row_off: int, col_off: int, stride: int, source_pass: str) -> None:
        band = label_ds.GetRasterBand(1)
        gt = label_ds.GetGeoTransform()
        bbox = window_bounds(gt, row_off=row_off, col_off=col_off, patch_size=args.patch_size)
        center_x = (bbox[0] + bbox[2]) * 0.5
        center_y = (bbox[1] + bbox[3]) * 0.5

        source_subsplit, final_split, reject_reason, _ = split_11block_threeway(
            center_x=center_x,
            bbox=bbox,
            bounds=meta_11["bounds"],
            train_ratio=args.split_train_ratio,
            val_ratio=args.split_val_ratio,
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
        if source_pass == "eval" and final_split not in {"val", "test"}:
            return

        window_key = (pair_11block.pair_name, row_off, col_off, args.patch_size)
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
            valid_ratio_train=args.valid_ratio_train,
            valid_ratio_valtest=args.valid_ratio_valtest,
        )

        if valid_pixels == 0:
            rejected_counter["all_nodata"] += 1
            return
        if valid_ratio < threshold:
            rejected_counter["valid_ratio_below_threshold"] += 1
            return

        patch_pixels = args.patch_size * args.patch_size
        tile_id = f"{pair_11block.pair_name}_{final_split}_r{row_off:05d}_c{col_off:05d}"
        row = {
            "tile_id": tile_id,
            "pair_name": pair_11block.pair_name,
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
        source_pass_counter[source_pass] += 1

    total_candidates = 0
    max_tiles = int(args.max_tiles)

    h11, w11 = label_ds.RasterYSize, label_ds.RasterXSize
    for r, c in iter_windows(w11, h11, args.patch_size, args.train_stride) or []:
        process_candidate(row_off=r, col_off=c, stride=args.train_stride, source_pass="train")
        total_candidates += 1
        if max_tiles > 0 and total_candidates >= max_tiles:
            break

    if not (max_tiles > 0 and total_candidates >= max_tiles):
        for r, c in iter_windows(w11, h11, args.patch_size, args.eval_stride) or []:
            process_candidate(row_off=r, col_off=c, stride=args.eval_stride, source_pass="eval")
            total_candidates += 1
            if max_tiles > 0 and total_candidates >= max_tiles:
                break

    label_ds = None

    accepted_rows.sort(key=lambda x: (x["final_split"], int(x["row_off"]), int(x["col_off"])))
    write_tiles_manifest(accepted_rows, out_tiles_manifest)
    write_split_summary(split_acc, out_split_summary, gates=gates)
    write_eco_split_geometry(
        out_csv=out_eco_geom,
        pair_11block=pair_11block,
        bounds_11block=meta_11["bounds"],
        split_lines=split_lines,
        args=args,
    )
    write_eco_split_class_stats(split_acc, out_eco_stats)

    split_counts = {
        "train": split_acc.get("train", SplitAccumulator()).tile_count,
        "val": split_acc.get("val", SplitAccumulator()).tile_count,
        "test": split_acc.get("test", SplitAccumulator()).tile_count,
    }

    print("\n[result] split counts")
    print(f"  - train: {split_counts['train']} (gate>={gates['train']})")
    print(f"  - val: {split_counts['val']} (gate>={gates['val']})")
    print(f"  - test: {split_counts['test']} (gate>={gates['test']})")
    print(
        "[result] gate pass = "
        f"{int(split_counts['train'] >= gates['train'] and split_counts['val'] >= gates['val'] and split_counts['test'] >= gates['test'])}"
    )

    print("\n[split lines]")
    print(
        f"  - x_line_1={split_lines['x_line_1']:.3f}, x_line_2={split_lines['x_line_2']:.3f}, "
        f"buffer={args.buffer_m:.1f}m"
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


