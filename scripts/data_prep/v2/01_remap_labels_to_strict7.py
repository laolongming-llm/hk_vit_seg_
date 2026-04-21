#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Remap 11block labels to strict 7-class schema.

Default mapping (8-class source -> strict7 target):
1->1, 2->2, 3->3, 4->4, 5->5, 6->255(ignore), 7->6, 8->7, 255->255
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Final

import numpy as np

try:
    from osgeo import gdal
except Exception as import_exc:  # pragma: no cover
    gdal = None
    GDAL_IMPORT_ERROR = import_exc
else:
    GDAL_IMPORT_ERROR = None


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]

DEFAULT_INPUT_LABEL = (
    PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_11block_1m_aligned.tif"
)
DEFAULT_OUTPUT_LABEL = (
    PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_11block_1m_aligned_v2_strict7.tif"
)
DEFAULT_REPORT_CSV = PROJECT_ROOT / "data" / "interim" / "qc" / "class_pixel_stats_11block_strict7.csv"
DEFAULT_REPORT_MD = PROJECT_ROOT / "data" / "interim" / "qc" / "class_remap_report_11block_strict7.md"

DEFAULT_MAPPING: Final[Dict[int, int]] = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 255,  # agriculture -> ignore
    7: 6,    # water
    8: 7,    # mountainous
    255: 255,
}

TARGET_CLASS_NAME: Final[Dict[int, str]] = {
    1: "building_land",
    2: "business_land",
    3: "industrial_land",
    4: "transport_land",
    5: "infrastructure_land",
    6: "water_body",
    7: "mountainous_land",
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
    detail = f"original error: {GDAL_IMPORT_ERROR}" if GDAL_IMPORT_ERROR else "no low-level error info."
    raise RuntimeError(
        "GDAL Python bindings (osgeo) are not available.\n"
        "Please run in GIS conda env, e.g. conda install -c conda-forge gdal\n"
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
        raise ValueError("--mapping-json must be an object, e.g. {\"1\":1,...}.")
    out: Dict[int, int] = {}
    for k, v in payload.items():
        out[int(k)] = int(v)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remap 11block labels to strict7 schema.")
    parser.add_argument("--input-label", default=str(DEFAULT_INPUT_LABEL), help="Input label raster path")
    parser.add_argument("--output-label", default=str(DEFAULT_OUTPUT_LABEL), help="Output remapped label raster path")
    parser.add_argument("--report-csv", default=str(DEFAULT_REPORT_CSV), help="Output report CSV")
    parser.add_argument("--report-md", default=str(DEFAULT_REPORT_MD), help="Output report Markdown")
    parser.add_argument(
        "--mapping-json",
        default="",
        help="Optional custom mapping JSON. Unmapped source values fallback to 255.",
    )
    parser.add_argument("--no-overwrite", action="store_true", help="Fail if output exists")
    return parser


def write_report_csv(path: Path, counts: Dict[int, int], valid_total: int, total: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_value", "class_name", "pixel_count", "ratio_all", "ratio_valid", "is_nodata"],
        )
        writer.writeheader()
        for cls in [1, 2, 3, 4, 5, 6, 7, 255]:
            c = int(counts.get(cls, 0))
            ratio_all = c / max(total, 1)
            ratio_valid = (c / max(valid_total, 1)) if cls != 255 else 0.0
            writer.writerow(
                {
                    "class_value": cls,
                    "class_name": TARGET_CLASS_NAME.get(cls, f"class_{cls}"),
                    "pixel_count": c,
                    "ratio_all": f"{ratio_all:.8f}",
                    "ratio_valid": f"{ratio_valid:.8f}",
                    "is_nodata": int(cls == 255),
                }
            )


def write_report_md(
    path: Path,
    mapping: Dict[int, int],
    source_counts: Dict[int, int],
    target_counts: Dict[int, int],
    total: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = total - int(target_counts.get(255, 0))
    with path.open("w", encoding="utf-8") as f:
        f.write("# 11block Label Remap Report (strict7)\n\n")
        f.write("## Mapping\n\n")
        for k in sorted(mapping.keys()):
            f.write(f"- `{k} -> {mapping[k]}`\n")

        f.write("\n## Source Distribution (top values)\n\n")
        f.write("| value | pixels |\n")
        f.write("|---:|---:|\n")
        for cls, cnt in sorted(source_counts.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"| {cls} | {cnt} |\n")

        f.write("\n## Target Distribution\n\n")
        f.write(f"- total_pixels: `{total}`\n")
        f.write(f"- valid_pixels: `{valid}`\n")
        f.write(f"- nodata_pixels(255): `{target_counts.get(255, 0)}`\n\n")
        f.write("| class | name | pixels | ratio_all | ratio_valid |\n")
        f.write("|---:|---|---:|---:|---:|\n")
        for cls in [1, 2, 3, 4, 5, 6, 7]:
            c = int(target_counts.get(cls, 0))
            f.write(
                f"| {cls} | {TARGET_CLASS_NAME.get(cls,'')} | {c} | "
                f"{c / max(total, 1):.6f} | {c / max(valid, 1):.6f} |\n"
            )


def main() -> None:
    args = build_parser().parse_args()
    require_gdal()

    input_label = to_abs_path(args.input_label)
    output_label = to_abs_path(args.output_label)
    report_csv = to_abs_path(args.report_csv)
    report_md = to_abs_path(args.report_md)

    if not input_label.exists():
        raise FileNotFoundError(f"input label not found: {input_label}")
    if output_label.exists() and args.no_overwrite:
        raise FileExistsError(f"output already exists: {output_label}")

    mapping = DEFAULT_MAPPING.copy()
    if args.mapping_json.strip():
        mapping = parse_mapping(args.mapping_json)

    src = gdal.Open(str(input_label), gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"cannot open input label: {input_label}")

    xsize = src.RasterXSize
    ysize = src.RasterYSize
    gt = src.GetGeoTransform()
    proj = src.GetProjection()
    src_band = src.GetRasterBand(1)

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
        raise RuntimeError("failed to create output raster.")
    dst.SetGeoTransform(gt)
    dst.SetProjection(proj)
    dst_band = dst.GetRasterBand(1)
    dst_band.SetNoDataValue(255)

    block_x, block_y = src_band.GetBlockSize()
    if block_x <= 0:
        block_x = 1024
    if block_y <= 0:
        block_y = 1024

    source_counts: Dict[int, int] = {}
    target_counts: Dict[int, int] = {k: 0 for k in [1, 2, 3, 4, 5, 6, 7, 255]}

    for yoff in range(0, ysize, block_y):
        ywin = min(block_y, ysize - yoff)
        for xoff in range(0, xsize, block_x):
            xwin = min(block_x, xsize - xoff)
            arr = src_band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xwin, win_ysize=ywin)
            if arr is None:
                raise RuntimeError("failed reading source block.")

            # source stats
            u_src, c_src = np.unique(arr, return_counts=True)
            for cls, c in zip(u_src.tolist(), c_src.tolist()):
                source_counts[int(cls)] = source_counts.get(int(cls), 0) + int(c)

            out = np.full(arr.shape, fill_value=255, dtype=np.uint8)
            for old_cls, new_cls in mapping.items():
                out[arr == old_cls] = np.uint8(new_cls)

            dst_band.WriteArray(out, xoff=xoff, yoff=yoff)

            u_dst, c_dst = np.unique(out, return_counts=True)
            for cls, c in zip(u_dst.tolist(), c_dst.tolist()):
                if int(cls) not in target_counts:
                    target_counts[int(cls)] = 0
                target_counts[int(cls)] += int(c)

    dst_band.FlushCache()
    dst.FlushCache()
    dst = None
    src = None

    total = xsize * ysize
    valid_total = total - int(target_counts.get(255, 0))
    write_report_csv(report_csv, counts=target_counts, valid_total=valid_total, total=total)
    write_report_md(
        report_md,
        mapping=mapping,
        source_counts=source_counts,
        target_counts=target_counts,
        total=total,
    )

    print("[OK] strict7 remap completed")
    print(f"  - input : {input_label}")
    print(f"  - output: {output_label}")
    print(f"  - report: {report_csv}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

