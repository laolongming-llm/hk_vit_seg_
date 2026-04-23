#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Create a GIS-ready compact classified raster for strict7 visualization.

Input class values:
- 1..7: strict7 classes
- 255 : nodata/unknown

Output class values:
- 0   : nodata/unknown
- 1..7: strict7 classes (unchanged)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from osgeo import gdal, osr
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "osgeo.gdal is required. Please run this script in your GIS/conda environment."
    ) from exc


STRICT7_COLORS = {
    0: (255, 255, 255, 255),  # 未分类/空白
    1: (228, 26, 28, 255),    # 建筑用地
    2: (255, 127, 0, 255),    # 商业用地
    3: (152, 78, 163, 255),   # 工业用地
    4: (255, 217, 47, 255),   # 交通用地
    5: (55, 126, 184, 255),   # 基础设施/公共服务用地
    6: (31, 120, 180, 255),   # 水体
    7: (51, 160, 44, 255),    # 山地/林地
}

STRICT7_CATEGORY_NAMES = [
    "0 未分类/空白",
    "1 建筑用地",
    "2 商业用地",
    "3 工业用地",
    "4 交通用地",
    "5 基础设施/公共服务用地",
    "6 水体",
    "7 山地/林地",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create compact strict7 GIS-ready classified raster.")
    parser.add_argument("--input", required=True, help="Input classified GeoTIFF path.")
    parser.add_argument("--output", required=True, help="Output GIS-ready GeoTIFF path.")
    parser.add_argument(
        "--force-epsg",
        type=int,
        default=2326,
        help="Force output CRS to this EPSG code (default: 2326).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if exists.")
    return parser.parse_args()


def remap_block(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[arr == 255] = 0
    out[(arr < 0) | (arr > 7)] = 0
    return out.astype(np.uint8)


def main() -> None:
    args = parse_args()
    in_path = Path(args.input).resolve()
    out_path = Path(args.output).resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if args.overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output exists: {out_path}")

    src = gdal.Open(str(in_path), gdal.GA_ReadOnly)
    if src is None:
        raise RuntimeError(f"Cannot open input: {in_path}")

    xsize = src.RasterXSize
    ysize = src.RasterYSize
    gt = src.GetGeoTransform()
    src_band = src.GetRasterBand(1)
    block_x, block_y = src_band.GetBlockSize()
    if block_x <= 0 or block_y <= 0:
        block_x, block_y = 512, 512

    drv = gdal.GetDriverByName("GTiff")
    dst = drv.Create(
        str(out_path),
        xsize,
        ysize,
        1,
        gdal.GDT_Byte,
        options=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "BIGTIFF=IF_SAFER",
        ],
    )
    if dst is None:
        raise RuntimeError(f"Cannot create output: {out_path}")

    dst.SetGeoTransform(gt)
    if args.force_epsg > 0:
        srs = osr.SpatialReference()
        if srs.ImportFromEPSG(int(args.force_epsg)) != 0:
            raise RuntimeError(f"Failed to import EPSG:{args.force_epsg}")
        dst.SetProjection(srs.ExportToWkt())
    else:
        dst.SetProjection(src.GetProjection())

    dst_band = dst.GetRasterBand(1)
    dst_band.SetDescription("LUM_ID_STRICT7_GIS_READY")
    dst_band.SetNoDataValue(0)

    ct = gdal.ColorTable()
    for idx in range(8):
        ct.SetColorEntry(idx, STRICT7_COLORS[idx])
    dst_band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    dst_band.SetRasterColorTable(ct)
    category_names = [""] * 256
    for idx, name in enumerate(STRICT7_CATEGORY_NAMES):
        category_names[idx] = name
    dst_band.SetCategoryNames(category_names)

    for y_off in range(0, ysize, block_y):
        win_h = min(block_y, ysize - y_off)
        for x_off in range(0, xsize, block_x):
            win_w = min(block_x, xsize - x_off)
            data = src_band.ReadAsArray(x_off, y_off, win_w, win_h)
            if data is None:
                raise RuntimeError(f"Read failed at window x={x_off}, y={y_off}")
            mapped = remap_block(data)
            dst_band.WriteArray(mapped, xoff=x_off, yoff=y_off)

    dst_band.FlushCache()
    dst.FlushCache()
    src = None
    dst = None
    print(f"[OK] GIS-ready strict7 raster written: {out_path}")


if __name__ == "__main__":
    main()
