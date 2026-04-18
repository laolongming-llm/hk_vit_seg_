#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：09_remap_labels_11block_8class.py

功能：
1) 将 11block 的 10 类 LUM_ID 标签重映射为 8 类体系；
2) 保留 nodata=255；
3) 输出重映射统计报告，供后续数据集构建使用。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Final

try:
    from osgeo import gdal
except Exception as import_exc:  # pragma: no cover
    gdal = None
    GDAL_IMPORT_ERROR = import_exc
else:
    GDAL_IMPORT_ERROR = None

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_LABEL = (
    PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_11block_1m_aligned.tif"
)
DEFAULT_OUTPUT_LABEL = (
    PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_11block_1m_aligned_v3_8class.tif"
)
DEFAULT_REPORT_CSV = PROJECT_ROOT / "data" / "interim" / "qc" / "class_pixel_stats_11block8.csv"
DEFAULT_REPORT_MD = PROJECT_ROOT / "data" / "interim" / "qc" / "class_remap_report_11block8.md"

# 10类 -> 8类（鱼塘/红树林并入水体）
DEFAULT_MAPPING: Final[Dict[int, int]] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,   # fish_pond -> water
    8: 7,   # water -> water
    9: 8,   # mountainous -> mountainous
    10: 7,  # mangrove -> water
    255: 255,
}

NEW_CLASS_NAME: Final[Dict[int, str]] = {
    1: "building_land",
    2: "business_land",
    3: "industrial_land",
    4: "transport_land",
    5: "infrastructure_land",
    6: "agricultural_land",
    7: "water_body",
    8: "mountainous_land",
    255: "unknown_or_nodata",
}

GTIFF_CREATION_OPTIONS = [
    "COMPRESS=LZW",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "SPARSE_OK=TRUE",
    "NUM_THREADS=ALL_CPUS",
    "BIGTIFF=IF_SAFER",
]


def require_gdal() -> None:
    if gdal is not None:
        return
    detail = f"原始错误：{GDAL_IMPORT_ERROR}" if GDAL_IMPORT_ERROR else "未提供底层异常信息。"
    raise RuntimeError(
        "未检测到 GDAL Python 绑定（osgeo）。\n"
        "请在 GIS conda 环境中运行，例如：conda install -c conda-forge gdal\n"
        f"{detail}"
    )


def to_abs_path(path_ref: str) -> Path:
    p = Path(path_ref).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def parse_mapping(mapping_json: str) -> Dict[int, int]:
    payload = json.loads(mapping_json)
    if not isinstance(payload, dict):
        raise ValueError("--mapping-json 必须是对象字典，例如 {\"1\":1,...}")
    out: Dict[int, int] = {}
    for k, v in payload.items():
        out[int(k)] = int(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remap 11block labels from 10-class to 8-class.")
    parser.add_argument("--input-label", default=str(DEFAULT_INPUT_LABEL), help="输入标签栅格（10类）")
    parser.add_argument("--output-label", default=str(DEFAULT_OUTPUT_LABEL), help="输出标签栅格（8类）")
    parser.add_argument("--report-csv", default=str(DEFAULT_REPORT_CSV), help="输出统计 CSV")
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD), help="输出报告 Markdown")
    parser.add_argument(
        "--mapping-json",
        default="",
        help="可选：自定义映射 JSON（默认使用脚本内置映射）",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="输出存在时不覆盖")
    return parser


def write_report_csv(path: Path, counts: Dict[int, int], valid_total: int, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_value", "class_name", "pixel_count", "ratio_all", "ratio_valid", "is_nodata"],
        )
        writer.writeheader()
        for cls in [1, 2, 3, 4, 5, 6, 7, 8, 255]:
            c = int(counts.get(cls, 0))
            ratio_all = c / max(total, 1)
            ratio_valid = (c / max(valid_total, 1)) if cls != 255 else 0.0
            writer.writerow(
                {
                    "class_value": cls,
                    "class_name": NEW_CLASS_NAME.get(cls, f"class_{cls}"),
                    "pixel_count": c,
                    "ratio_all": f"{ratio_all:.8f}",
                    "ratio_valid": f"{ratio_valid:.8f}",
                    "is_nodata": int(cls == 255),
                }
            )


def write_report_md(path: Path, mapping: Dict[int, int], counts: Dict[int, int], total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = total - int(counts.get(255, 0))
    with path.open("w", encoding="utf-8") as f:
        f.write("# 11block 标签 10类->8类 重映射报告\n\n")
        f.write("## 映射规则\n\n")
        for k in sorted(mapping.keys()):
            f.write(f"- `{k} -> {mapping[k]}`\n")
        f.write("\n## 输出分布\n\n")
        f.write(f"- 总像元：`{total}`\n")
        f.write(f"- 有效像元：`{valid}`\n")
        f.write(f"- 无效像元(255)：`{counts.get(255,0)}`\n")
        f.write("\n| 类别 | 名称 | 像元数 | 占总像元 | 占有效像元 |\n")
        f.write("|---:|---|---:|---:|---:|\n")
        for cls in [1, 2, 3, 4, 5, 6, 7, 8]:
            c = int(counts.get(cls, 0))
            f.write(
                f"| {cls} | {NEW_CLASS_NAME.get(cls,'')} | {c} | {c/max(total,1):.6f} | {c/max(valid,1):.6f} |\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    require_gdal()

    input_label = to_abs_path(args.input_label)
    output_label = to_abs_path(args.output_label)
    report_csv = to_abs_path(args.report_csv)
    report_md = to_abs_path(args.report_md)

    if not input_label.exists():
        raise FileNotFoundError(f"输入标签不存在：{input_label}")
    if output_label.exists() and args.no_overwrite:
        raise FileExistsError(f"输出已存在：{output_label}")

    mapping = DEFAULT_MAPPING.copy()
    if args.mapping_json.strip():
        mapping = parse_mapping(args.mapping_json)

    src = gdal.Open(str(input_label), gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"无法打开输入标签：{input_label}")

    xsize = src.RasterXSize
    ysize = src.RasterYSize
    gt = src.GetGeoTransform()
    proj = src.GetProjection()

    output_label.parent.mkdir(parents=True, exist_ok=True)
    drv = gdal.GetDriverByName("GTiff")
    if output_label.exists():
        drv.Delete(str(output_label))

    dst = drv.Create(
        str(output_label),
        xsize,
        ysize,
        1,
        gdal.GDT_Byte,
        options=GTIFF_CREATION_OPTIONS,
    )
    if dst is None:
        raise RuntimeError("创建输出标签失败")

    dst.SetGeoTransform(gt)
    dst.SetProjection(proj)
    dst_band = dst.GetRasterBand(1)
    dst_band.SetNoDataValue(255)

    src_band = src.GetRasterBand(1)
    block_x, block_y = src_band.GetBlockSize()
    if block_x <= 0:
        block_x = 1024
    if block_y <= 0:
        block_y = 1024

    counts: Dict[int, int] = {k: 0 for k in [1, 2, 3, 4, 5, 6, 7, 8, 255]}

    for yoff in range(0, ysize, block_y):
        ywin = min(block_y, ysize - yoff)
        for xoff in range(0, xsize, block_x):
            xwin = min(block_x, xsize - xoff)
            arr = src_band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xwin, win_ysize=ywin)
            if arr is None:
                raise RuntimeError("读取输入栅格失败")

            out = arr.copy()
            # 默认兜底到 255，避免出现非法值漏映射。
            out.fill(255)
            for old_cls, new_cls in mapping.items():
                out[arr == old_cls] = new_cls

            dst_band.WriteArray(out, xoff=xoff, yoff=yoff)

            # 统计
            unique, unique_counts = __import__("numpy").unique(out, return_counts=True)
            for cls, c in zip(unique.tolist(), unique_counts.tolist()):
                if cls not in counts:
                    counts[cls] = 0
                counts[cls] += int(c)

    dst_band.FlushCache()
    dst.FlushCache()
    dst = None
    src = None

    total = xsize * ysize
    valid_total = total - int(counts.get(255, 0))
    write_report_csv(report_csv, counts=counts, valid_total=valid_total, total=total)
    write_report_md(report_md, mapping=mapping, counts=counts, total=total)

    print("[OK] label remap completed")
    print(f"  - input : {input_label}")
    print(f"  - output: {output_label}")
    print(f"  - report: {report_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

