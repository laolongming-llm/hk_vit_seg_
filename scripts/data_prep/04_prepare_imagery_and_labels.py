#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：04_prepare_imagery_and_labels.py

功能：
1) 按“主块(11*) + 离散块(2SWD)”分组拼接 TDOP 0.25m 影像（不使用 VRT）；
2) 按需为源影像补写/确认 EPSG:2326；
3) 以 LUM_ID 1m 栅格为参考网格，导出每个分组的 1m 标签与 1m 对齐影像；
4) 可选按 AOI（包络框）继续导出影像-标签子集；
5) 输出配准质检 CSV 与类别像元统计 CSV。
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

try:
    from osgeo import gdal, ogr, osr
except Exception as import_exc:  # pragma: no cover
    gdal = None
    ogr = None
    osr = None
    GDAL_IMPORT_ERROR = import_exc
else:
    GDAL_IMPORT_ERROR = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    from lumid_style import apply_lumid_style_to_raster, write_style_sidecars_for_raster
except Exception:
    apply_lumid_style_to_raster = None
    write_style_sidecars_for_raster = None


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ===== 默认输入 =====
DEFAULT_IMAGERY_ROOT: Final[Path] = PROJECT_ROOT / "data" / "raw" / "imagery"
DEFAULT_MAIN_GLOB: Final[str] = "TDOP_TIFF_11*/*.tif"
DEFAULT_SECONDARY_GLOB: Final[str] = "TDOP_TIFF_2SWD/*.tif"
DEFAULT_REFERENCE_LABEL: Final[Path] = PROJECT_ROOT / "data" / "interim" / "masks_full" / "hk_landuse_LUMID.tif"

# ===== 默认输出根目录 =====
DEFAULT_OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / "data" / "interim"

# ===== 产物路径（相对 OUTPUT_ROOT）=====
REL_MOSAIC_MAIN: Final[Path] = Path("imagery_mosaic") / "tdop_11block_0p25m.tif"
REL_MOSAIC_SECONDARY: Final[Path] = Path("imagery_mosaic") / "tdop_2swd_0p25m.tif"
REL_IMG_MAIN_1M: Final[Path] = Path("imagery_1m_aligned") / "tdop_11block_1m_aligned.tif"
REL_IMG_SECONDARY_1M: Final[Path] = Path("imagery_1m_aligned") / "tdop_2swd_1m_aligned.tif"
REL_LABEL_MAIN_1M: Final[Path] = Path("labels_1m_aligned") / "lumid_11block_1m_aligned.tif"
REL_LABEL_SECONDARY_1M: Final[Path] = Path("labels_1m_aligned") / "lumid_2swd_1m_aligned.tif"
REL_QC_ALIGNMENT: Final[Path] = Path("qc") / "imagery_label_alignment_report.csv"
REL_QC_CLASS_STATS: Final[Path] = Path("qc") / "class_pixel_stats.csv"
REL_AOI_OUTPUT_DIR: Final[Path] = Path("aoi_pairs")

# ===== 处理参数 =====
DEFAULT_TARGET_EPSG: Final[int] = 2326
DEFAULT_MOSAIC_RES: Final[float] = 0.25
DEFAULT_ALIGNED_RES: Final[float] = 1.0
DEFAULT_IMAGERY_NODATA: Final[int] = 0
DEFAULT_LABEL_NODATA: Final[int] = 255
DEFAULT_IMAGERY_RESAMPLE: Final[str] = "average"

# ===== 栅格读块参数 =====
DEFAULT_BLOCK_SIZE: Final[int] = 2048

GTIFF_CREATION_OPTIONS: Final[list[str]] = [
    "COMPRESS=LZW",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "SPARSE_OK=TRUE",
    "NUM_THREADS=ALL_CPUS",
    "BIGTIFF=IF_SAFER",
]

LUM_CLASS_NAME: Final[dict[int, str]] = {
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


@dataclass(frozen=True)
class RasterPair:
    pair_name: str
    pair_type: str
    imagery_path: Path
    label_path: Path


@dataclass(frozen=True)
class GroupSpec:
    group_name: str
    input_glob: str
    mosaic_rel: Path
    imagery_1m_rel: Path
    label_1m_rel: Path


def require_gdal() -> None:
    if gdal is not None and ogr is not None and osr is not None:
        return
    detail = f"原始错误：{GDAL_IMPORT_ERROR}" if GDAL_IMPORT_ERROR else "未提供底层异常信息。"
    raise RuntimeError(
        "未检测到 GDAL/OGR Python 绑定（osgeo）。\n"
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


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def cleanup_raster(path: Path) -> None:
    remove_if_exists(path)
    remove_if_exists(Path(str(path) + ".aux.xml"))
    remove_if_exists(Path(str(path) + ".ovr"))


def prepare_output(path: Path, overwrite: bool) -> None:
    ensure_parent_dir(path)
    if not path.exists():
        return
    if not overwrite:
        raise FileExistsError(f"输出文件已存在: {path}")
    cleanup_raster(path)


def approx_equal(a: float, b: float, tol: float = 1e-8) -> bool:
    return abs(a - b) <= tol


def sanitize_name(name: str, fallback: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return safe if safe else fallback


def list_group_tifs(imagery_root: Path, pattern: str) -> list[Path]:
    files = sorted(p.resolve() for p in imagery_root.glob(pattern) if p.is_file())
    files = [p for p in files if p.suffix.lower() == ".tif"]
    if not files:
        raise FileNotFoundError(
            f"未匹配到 TIFF 文件：imagery_root={imagery_root}, pattern={pattern}"
        )
    return files


def build_epsg_srs(epsg: int):
    srs = osr.SpatialReference()
    err = srs.ImportFromEPSG(int(epsg))
    if err != 0:
        raise RuntimeError(f"无法构建 EPSG:{epsg} 的空间参考。")
    return srs


def get_epsg_code_from_wkt(wkt: str | None) -> int | None:
    if not wkt:
        return None
    srs = osr.SpatialReference()
    if srs.ImportFromWkt(wkt) != 0:
        return None
    auth = srs.GetAuthorityCode(None)
    if auth is None:
        return None
    try:
        return int(auth)
    except ValueError:
        return None


def assign_epsg_inplace(
    tif_paths: list[Path],
    target_epsg: int,
) -> tuple[int, int]:
    target_srs = build_epsg_srs(target_epsg)
    target_wkt = target_srs.ExportToWkt()

    updated = 0
    unchanged = 0
    for tif in tif_paths:
        ds = gdal.Open(str(tif), gdal.GA_Update)
        if ds is None:
            raise RuntimeError(f"无法以可写模式打开影像：{tif}")

        proj = ds.GetProjection()
        current_epsg = get_epsg_code_from_wkt(proj)
        if current_epsg == target_epsg:
            unchanged += 1
            ds = None
            continue

        # 若已带其它明确 EPSG，默认不强改，避免误写错投影
        if current_epsg is not None and current_epsg != target_epsg:
            ds = None
            raise RuntimeError(
                f"影像 {tif} 的 EPSG={current_epsg}，与目标 EPSG:{target_epsg} 不一致。"
                "请先人工核对后再处理。"
            )

        ds.SetProjection(target_wkt)
        ds.FlushCache()
        ds = None
        updated += 1

    return updated, unchanged


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


def bounds_intersection(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    minx = max(a[0], b[0])
    miny = max(a[1], b[1])
    maxx = min(a[2], b[2])
    maxy = min(a[3], b[3])
    if minx >= maxx or miny >= maxy:
        return None
    return (minx, miny, maxx, maxy)


def mosaic_group(
    input_tifs: list[Path],
    output_tif: Path,
    target_epsg: int,
    resolution: float,
    nodata: int,
    overwrite: bool,
) -> None:
    prepare_output(output_tif, overwrite=overwrite)

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        srcSRS=f"EPSG:{target_epsg}",
        dstSRS=f"EPSG:{target_epsg}",
        xRes=resolution,
        yRes=resolution,
        targetAlignedPixels=True,
        resampleAlg="near",
        dstNodata=nodata,
        multithread=True,
        creationOptions=GTIFF_CREATION_OPTIONS,
    )
    ds = gdal.Warp(str(output_tif), [str(p) for p in input_tifs], options=warp_opts)
    if ds is None:
        raise RuntimeError(f"影像拼接失败：{output_tif}")
    ds = None


def bounds_to_window_on_reference(
    ref_ds: gdal.Dataset,
    bounds: tuple[float, float, float, float],
) -> tuple[int, int, int, int] | None:
    gt = ref_ds.GetGeoTransform()
    px_w = gt[1]
    px_h = abs(gt[5])
    if px_w <= 0 or px_h <= 0:
        raise RuntimeError("参考标签的像元大小非法。")

    x0 = gt[0]
    y0 = gt[3]

    minx, miny, maxx, maxy = bounds

    # 基于参考网格求窗口，保证输出与参考像元边界严格一致
    col0 = math.floor((minx - x0) / px_w + 1e-10)
    col1 = math.ceil((maxx - x0) / px_w - 1e-10)
    row0 = math.floor((y0 - maxy) / px_h + 1e-10)
    row1 = math.ceil((y0 - miny) / px_h - 1e-10)

    col0 = max(0, min(col0, ref_ds.RasterXSize))
    col1 = max(0, min(col1, ref_ds.RasterXSize))
    row0 = max(0, min(row0, ref_ds.RasterYSize))
    row1 = max(0, min(row1, ref_ds.RasterYSize))

    if col1 <= col0 or row1 <= row0:
        return None

    return (col0, row0, col1 - col0, row1 - row0)


def window_to_bounds_on_reference(
    ref_ds: gdal.Dataset,
    window: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    gt = ref_ds.GetGeoTransform()
    col0, row0, width, height = window
    minx = gt[0] + gt[1] * col0
    maxy = gt[3] + gt[5] * row0
    maxx = gt[0] + gt[1] * (col0 + width)
    miny = gt[3] + gt[5] * (row0 + height)
    if maxx < minx:
        minx, maxx = maxx, minx
    if maxy < miny:
        miny, maxy = maxy, miny
    return (minx, miny, maxx, maxy)


def translate_window(
    src_tif: Path,
    dst_tif: Path,
    window: tuple[int, int, int, int],
    overwrite: bool,
    nodata: int | None = None,
) -> None:
    prepare_output(dst_tif, overwrite=overwrite)
    col0, row0, width, height = window
    translate_opts = gdal.TranslateOptions(
        format="GTiff",
        srcWin=[col0, row0, width, height],
        creationOptions=GTIFF_CREATION_OPTIONS,
    )
    ds = gdal.Translate(str(dst_tif), str(src_tif), options=translate_opts)
    if ds is None:
        raise RuntimeError(f"窗口裁剪失败：{dst_tif}")
    ds = None

    if nodata is not None:
        ds_update = gdal.Open(str(dst_tif), gdal.GA_Update)
        if ds_update is not None:
            band = ds_update.GetRasterBand(1)
            band.SetNoDataValue(nodata)
            band.FlushCache()
            ds_update.FlushCache()
            ds_update = None


def align_imagery_to_label_grid(
    src_mosaic_tif: Path,
    ref_label_tif: Path,
    dst_tif: Path,
    imagery_resample: str,
    imagery_nodata: int,
    overwrite: bool,
) -> None:
    prepare_output(dst_tif, overwrite=overwrite)

    label_ds = gdal.Open(str(ref_label_tif), gdal.GA_ReadOnly)
    if label_ds is None:
        raise RuntimeError(f"无法打开参考标签：{ref_label_tif}")

    bounds = dataset_bounds(label_ds)
    width = label_ds.RasterXSize
    height = label_ds.RasterYSize
    proj = label_ds.GetProjection()
    label_ds = None

    warp_opts = gdal.WarpOptions(
        format="GTiff",
        dstSRS=proj if proj else None,
        outputBounds=[bounds[0], bounds[1], bounds[2], bounds[3]],
        width=width,
        height=height,
        resampleAlg=imagery_resample,
        dstNodata=imagery_nodata,
        multithread=True,
        creationOptions=GTIFF_CREATION_OPTIONS,
    )
    ds = gdal.Warp(str(dst_tif), str(src_mosaic_tif), options=warp_opts)
    if ds is None:
        raise RuntimeError(f"影像对齐失败：{dst_tif}")
    ds = None


def spatial_ref_equal(wkt_a: str, wkt_b: str) -> bool:
    if not wkt_a or not wkt_b:
        return False
    srs_a = osr.SpatialReference()
    srs_b = osr.SpatialReference()
    if srs_a.ImportFromWkt(wkt_a) != 0 or srs_b.ImportFromWkt(wkt_b) != 0:
        return False
    return bool(srs_a.IsSame(srs_b))


def raster_meta(path: Path) -> dict[str, object]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开栅格：{path}")

    gt = ds.GetGeoTransform()
    bounds = dataset_bounds(ds)
    proj = ds.GetProjection()
    epsg = get_epsg_code_from_wkt(proj)
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue() if band is not None else None
    dtype = gdal.GetDataTypeName(band.DataType) if band is not None else "UNKNOWN"
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    bands = ds.RasterCount
    ds = None
    return {
        "path": str(path),
        "epsg": epsg,
        "projection": proj,
        "transform": gt,
        "resolution": (gt[1], abs(gt[5])),
        "shape": (ysize, xsize),
        "width": xsize,
        "height": ysize,
        "bounds": bounds,
        "nodata": nodata,
        "dtype": dtype,
        "bands": bands,
    }


def format_tuple(values: tuple[float, ...] | tuple[int, ...]) -> str:
    return "(" + ", ".join(str(v) for v in values) + ")"


def write_alignment_report(pairs: list[RasterPair], out_csv: Path) -> None:
    ensure_parent_dir(out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_name",
                "pair_type",
                "imagery_path",
                "label_path",
                "overall_pass",
                "crs_match",
                "resolution_match",
                "transform_match",
                "shape_match",
                "bounds_match",
                "imagery_shape",
                "label_shape",
                "imagery_resolution",
                "label_resolution",
                "imagery_bounds",
                "label_bounds",
            ]
        )

        for pair in pairs:
            img = raster_meta(pair.imagery_path)
            lbl = raster_meta(pair.label_path)

            crs_match = spatial_ref_equal(
                str(img["projection"] or ""),
                str(lbl["projection"] or ""),
            )
            img_res = img["resolution"]
            lbl_res = lbl["resolution"]
            resolution_match = approx_equal(float(img_res[0]), float(lbl_res[0])) and approx_equal(
                float(img_res[1]), float(lbl_res[1])
            )

            img_gt = img["transform"]
            lbl_gt = lbl["transform"]
            transform_match = all(approx_equal(float(a), float(b)) for a, b in zip(img_gt, lbl_gt))

            shape_match = img["shape"] == lbl["shape"]

            img_bounds = img["bounds"]
            lbl_bounds = lbl["bounds"]
            bounds_match = all(approx_equal(float(a), float(b)) for a, b in zip(img_bounds, lbl_bounds))

            overall = all([crs_match, resolution_match, transform_match, shape_match, bounds_match])

            w.writerow(
                [
                    pair.pair_name,
                    pair.pair_type,
                    str(pair.imagery_path),
                    str(pair.label_path),
                    int(overall),
                    int(crs_match),
                    int(resolution_match),
                    int(transform_match),
                    int(shape_match),
                    int(bounds_match),
                    format_tuple(img["shape"]),
                    format_tuple(lbl["shape"]),
                    format_tuple(img_res),
                    format_tuple(lbl_res),
                    format_tuple(img_bounds),
                    format_tuple(lbl_bounds),
                ]
            )


def count_label_pixels(label_tif: Path, block_size: int = DEFAULT_BLOCK_SIZE) -> tuple[dict[int, int], int]:
    ds = gdal.Open(str(label_tif), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"无法打开标签栅格：{label_tif}")
    band = ds.GetRasterBand(1)
    if band is None:
        ds = None
        raise RuntimeError(f"标签栅格无波段：{label_tif}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    data_type = band.DataType
    counts: Counter[int] = Counter()

    for yoff in range(0, height, block_size):
        ysize = min(block_size, height - yoff)
        for xoff in range(0, width, block_size):
            xsize = min(block_size, width - xoff)

            if data_type == gdal.GDT_Byte:
                raw = band.ReadRaster(
                    xoff=xoff,
                    yoff=yoff,
                    xsize=xsize,
                    ysize=ysize,
                    buf_xsize=xsize,
                    buf_ysize=ysize,
                    buf_type=gdal.GDT_Byte,
                )
                if raw is None:
                    ds = None
                    raise RuntimeError(f"读取标签块失败：{label_tif}")
                counts.update(raw)
            else:
                arr = band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
                if arr is None:
                    ds = None
                    raise RuntimeError(f"读取标签块失败：{label_tif}")
                if np is None:
                    ds = None
                    raise RuntimeError(
                        "标签不是 Byte 类型，且当前环境缺少 numpy，无法统计类别像元。"
                    )
                vals, freq = np.unique(arr, return_counts=True)
                for v, c in zip(vals.tolist(), freq.tolist()):
                    counts[int(v)] += int(c)

    ds = None
    return dict(counts), width * height


def write_class_stats(
    pairs: list[RasterPair],
    out_csv: Path,
    label_nodata: int,
) -> None:
    ensure_parent_dir(out_csv)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "pair_name",
                "pair_type",
                "label_path",
                "class_value",
                "class_name",
                "pixel_count",
                "ratio_all",
                "ratio_valid",
                "is_nodata",
            ]
        )

        for pair in pairs:
            counts, total = count_label_pixels(pair.label_path)
            nodata_count = counts.get(label_nodata, 0)
            valid_total = total - nodata_count

            present_classes = {int(v) for v, c in counts.items() if c > 0}
            forced = set(range(1, 11)) | {label_nodata}
            to_report = sorted(present_classes | forced)

            for cls in to_report:
                c = int(counts.get(cls, 0))
                ratio_all = c / total if total > 0 else 0.0
                if cls == label_nodata:
                    ratio_valid = 0.0
                else:
                    ratio_valid = c / valid_total if valid_total > 0 else 0.0
                w.writerow(
                    [
                        pair.pair_name,
                        pair.pair_type,
                        str(pair.label_path),
                        cls,
                        LUM_CLASS_NAME.get(cls, f"class_{cls}"),
                        c,
                        f"{ratio_all:.8f}",
                        f"{ratio_valid:.8f}",
                        int(cls == label_nodata),
                    ]
                )


def load_aoi_envelopes(
    aoi_vector: Path,
    aoi_layer: str | None,
    aoi_name_field: str | None,
    target_epsg: int,
) -> list[tuple[str, tuple[float, float, float, float]]]:
    ds = ogr.Open(str(aoi_vector), 0)
    if ds is None:
        raise RuntimeError(f"无法打开 AOI 矢量：{aoi_vector}")

    layer = ds.GetLayerByName(aoi_layer) if aoi_layer else ds.GetLayer(0)
    if layer is None:
        ds = None
        raise RuntimeError(f"AOI 图层不存在：{aoi_layer or '<first layer>'}")

    src_srs = layer.GetSpatialRef()
    dst_srs = build_epsg_srs(target_epsg)
    coord_tx = None
    if src_srs is not None and not bool(src_srs.IsSame(dst_srs)):
        coord_tx = osr.CoordinateTransformation(src_srs, dst_srs)

    name_field_idx = -1
    if aoi_name_field:
        name_field_idx = layer.GetLayerDefn().GetFieldIndex(aoi_name_field)

    envelopes: list[tuple[str, tuple[float, float, float, float]]] = []
    for feat in layer:
        fid = int(feat.GetFID())
        geom = feat.GetGeometryRef()
        if geom is None or geom.IsEmpty():
            continue

        g = geom.Clone()
        if coord_tx is not None:
            g.Transform(coord_tx)
        minx, maxx, miny, maxy = g.GetEnvelope()

        raw_name = None
        if name_field_idx >= 0:
            raw_name = feat.GetField(name_field_idx)
        if raw_name is None or str(raw_name).strip() == "":
            raw_name = f"aoi_{fid}"
        safe_name = sanitize_name(str(raw_name), fallback=f"aoi_{fid}")
        envelopes.append((safe_name, (minx, miny, maxx, maxy)))

    ds = None
    if not envelopes:
        raise RuntimeError("AOI 矢量中未读取到有效几何。")
    return envelopes


def export_aoi_bbox_pairs(
    base_pairs: list[RasterPair],
    aoi_envelopes: list[tuple[str, tuple[float, float, float, float]]],
    output_root: Path,
    label_nodata: int,
    overwrite: bool,
) -> list[RasterPair]:
    exported: list[RasterPair] = []

    for pair in base_pairs:
        label_ds = gdal.Open(str(pair.label_path), gdal.GA_ReadOnly)
        if label_ds is None:
            raise RuntimeError(f"无法打开标签：{pair.label_path}")
        label_bounds = dataset_bounds(label_ds)

        for aoi_name, aoi_bounds in aoi_envelopes:
            inter = bounds_intersection(label_bounds, aoi_bounds)
            if inter is None:
                continue

            window = bounds_to_window_on_reference(label_ds, inter)
            if window is None:
                continue

            subdir = output_root / aoi_name
            label_out = subdir / f"{pair.label_path.stem}_{aoi_name}.tif"
            imagery_out = subdir / f"{pair.imagery_path.stem}_{aoi_name}.tif"

            translate_window(pair.label_path, label_out, window, overwrite=overwrite, nodata=label_nodata)
            translate_window(pair.imagery_path, imagery_out, window, overwrite=overwrite, nodata=None)

            exported.append(
                RasterPair(
                    pair_name=f"{pair.pair_name}:{aoi_name}",
                    pair_type="aoi_bbox",
                    imagery_path=imagery_out,
                    label_path=label_out,
                )
            )

        label_ds = None

    return exported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TDOP 影像拼接、1m 配准与标签对齐导出")
    parser.add_argument("--imagery-root", default=str(DEFAULT_IMAGERY_ROOT), help="TDOP 影像根目录")
    parser.add_argument("--main-glob", default=DEFAULT_MAIN_GLOB, help="主块影像匹配模式（相对 imagery-root）")
    parser.add_argument("--secondary-glob", default=DEFAULT_SECONDARY_GLOB, help="离散块影像匹配模式（相对 imagery-root）")
    parser.add_argument("--reference-label", default=str(DEFAULT_REFERENCE_LABEL), help="参考 LUM_ID 1m 栅格路径")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="输出根目录（默认 data/interim）")

    parser.add_argument("--target-epsg", type=int, default=DEFAULT_TARGET_EPSG, help="目标 EPSG（默认 2326）")
    parser.add_argument("--mosaic-resolution", type=float, default=DEFAULT_MOSAIC_RES, help="拼接分辨率（默认 0.25）")
    parser.add_argument("--aligned-resolution", type=float, default=DEFAULT_ALIGNED_RES, help="目标对齐分辨率（默认 1.0）")
    parser.add_argument("--imagery-resample", default=DEFAULT_IMAGERY_RESAMPLE, help="影像下采样方法（默认 average）")
    parser.add_argument("--imagery-nodata", type=int, default=DEFAULT_IMAGERY_NODATA, help="影像 nodata（默认 0）")
    parser.add_argument("--label-nodata", type=int, default=DEFAULT_LABEL_NODATA, help="标签 nodata（默认 255）")

    parser.add_argument(
        "--assign-crs-inplace",
        dest="assign_crs_inplace",
        action="store_true",
        help="按需在源 TIFF 上补写 EPSG（默认开启）",
    )
    parser.add_argument(
        "--no-assign-crs-inplace",
        dest="assign_crs_inplace",
        action="store_false",
        help="不改写源 TIFF 投影，仅在输出中统一 CRS",
    )
    parser.set_defaults(assign_crs_inplace=True)

    parser.add_argument("--aoi-vector", default=None, help="可选 AOI 边界矢量路径（用于额外导出 AOI 子集）")
    parser.add_argument("--aoi-layer", default=None, help="AOI 图层名（默认第一个图层）")
    parser.add_argument("--aoi-name-field", default="name", help="AOI 名称字段（默认 name）")

    parser.add_argument("--no-lumid-style", action="store_true", help="不写入 LUM_ID 色表")
    parser.add_argument("--no-style-sidecar", action="store_true", help="不导出同名 qml/clr 样式文件")
    parser.add_argument("--dry-run", action="store_true", help="仅打印将要执行的处理，不落盘")
    parser.add_argument("--no-overwrite", action="store_true", help="若输出已存在则报错")
    return parser


def main() -> None:
    require_gdal()
    gdal.UseExceptions()

    args = build_parser().parse_args()

    imagery_root = to_abs_path(args.imagery_root)
    reference_label = to_abs_path(args.reference_label)
    output_root = to_abs_path(args.output_root)
    overwrite = not args.no_overwrite

    if not imagery_root.exists():
        raise FileNotFoundError(f"影像根目录不存在：{imagery_root}")
    if not reference_label.exists():
        raise FileNotFoundError(f"参考标签不存在：{reference_label}")
    if args.mosaic_resolution <= 0 or args.aligned_resolution <= 0:
        raise ValueError("--mosaic-resolution 与 --aligned-resolution 必须大于 0。")

    groups = [
        GroupSpec(
            group_name="11block",
            input_glob=args.main_glob,
            mosaic_rel=REL_MOSAIC_MAIN,
            imagery_1m_rel=REL_IMG_MAIN_1M,
            label_1m_rel=REL_LABEL_MAIN_1M,
        ),
        GroupSpec(
            group_name="2swd",
            input_glob=args.secondary_glob,
            mosaic_rel=REL_MOSAIC_SECONDARY,
            imagery_1m_rel=REL_IMG_SECONDARY_1M,
            label_1m_rel=REL_LABEL_SECONDARY_1M,
        ),
    ]

    group_files: dict[str, list[Path]] = {}
    for g in groups:
        files = list_group_tifs(imagery_root, g.input_glob)
        group_files[g.group_name] = files
        print(f"[scan] {g.group_name}: matched {len(files)} tif(s)")

    all_input_tifs = sorted({p for paths in group_files.values() for p in paths})

    if args.dry_run:
        print("\n[dry-run] 输入影像：")
        for p in all_input_tifs:
            print(f"  - {p}")
        print("[dry-run] 输出根目录：", output_root)
        return

    if args.assign_crs_inplace:
        updated, unchanged = assign_epsg_inplace(all_input_tifs, target_epsg=args.target_epsg)
        print(f"[crs] EPSG:{args.target_epsg} in-place 处理完成：updated={updated}, unchanged={unchanged}")

    ref_ds = gdal.Open(str(reference_label), gdal.GA_ReadOnly)
    if ref_ds is None:
        raise RuntimeError(f"无法打开参考标签：{reference_label}")
    ref_gt = ref_ds.GetGeoTransform()
    ref_res = (ref_gt[1], abs(ref_gt[5]))
    if not (approx_equal(ref_res[0], args.aligned_resolution) and approx_equal(ref_res[1], args.aligned_resolution)):
        raise RuntimeError(
            "参考标签分辨率与 --aligned-resolution 不一致。\n"
            f"参考标签分辨率: {ref_res}, 参数: {args.aligned_resolution}"
        )
    ref_bounds = dataset_bounds(ref_ds)
    ref_ds = None

    base_pairs: list[RasterPair] = []

    for g in groups:
        print(f"\n[run] processing group: {g.group_name}")
        input_tifs = group_files[g.group_name]
        mosaic_tif = output_root / g.mosaic_rel
        label_tif = output_root / g.label_1m_rel
        imagery_1m_tif = output_root / g.imagery_1m_rel

        print(f"  - mosaic 0.25m -> {mosaic_tif}")
        mosaic_group(
            input_tifs=input_tifs,
            output_tif=mosaic_tif,
            target_epsg=args.target_epsg,
            resolution=args.mosaic_resolution,
            nodata=args.imagery_nodata,
            overwrite=overwrite,
        )

        mosaic_ds = gdal.Open(str(mosaic_tif), gdal.GA_ReadOnly)
        if mosaic_ds is None:
            raise RuntimeError(f"无法打开拼接输出：{mosaic_tif}")
        mosaic_bounds = dataset_bounds(mosaic_ds)
        mosaic_ds = None

        inter = bounds_intersection(mosaic_bounds, ref_bounds)
        if inter is None:
            raise RuntimeError(
                f"分组 {g.group_name} 与参考标签无空间交集，无法导出 1m 标签。"
            )

        ref_ds = gdal.Open(str(reference_label), gdal.GA_ReadOnly)
        if ref_ds is None:
            raise RuntimeError(f"无法打开参考标签：{reference_label}")
        window = bounds_to_window_on_reference(ref_ds, inter)
        if window is None:
            ref_ds = None
            raise RuntimeError(f"分组 {g.group_name} 交集无法映射到参考标签网格。")
        snapped_bounds = window_to_bounds_on_reference(ref_ds, window)
        ref_ds = None

        print(f"  - label 1m aligned -> {label_tif}")
        translate_window(
            src_tif=reference_label,
            dst_tif=label_tif,
            window=window,
            overwrite=overwrite,
            nodata=args.label_nodata,
        )

        print(f"  - imagery 1m aligned -> {imagery_1m_tif}")
        align_imagery_to_label_grid(
            src_mosaic_tif=mosaic_tif,
            ref_label_tif=label_tif,
            dst_tif=imagery_1m_tif,
            imagery_resample=args.imagery_resample,
            imagery_nodata=args.imagery_nodata,
            overwrite=overwrite,
        )

        print(
            "  - snapped bounds: "
            f"minx={snapped_bounds[0]}, miny={snapped_bounds[1]}, "
            f"maxx={snapped_bounds[2]}, maxy={snapped_bounds[3]}"
        )

        base_pairs.append(
            RasterPair(
                pair_name=g.group_name,
                pair_type="base",
                imagery_path=imagery_1m_tif,
                label_path=label_tif,
            )
        )

    all_pairs = list(base_pairs)

    if args.aoi_vector:
        aoi_vector = to_abs_path(args.aoi_vector)
        if not aoi_vector.exists():
            raise FileNotFoundError(f"AOI 矢量不存在：{aoi_vector}")
        aoi_envelopes = load_aoi_envelopes(
            aoi_vector=aoi_vector,
            aoi_layer=args.aoi_layer,
            aoi_name_field=args.aoi_name_field,
            target_epsg=args.target_epsg,
        )
        print(f"\n[aoi] loaded {len(aoi_envelopes)} AOI envelope(s)")
        aoi_pairs = export_aoi_bbox_pairs(
            base_pairs=base_pairs,
            aoi_envelopes=aoi_envelopes,
            output_root=output_root / REL_AOI_OUTPUT_DIR,
            label_nodata=args.label_nodata,
            overwrite=overwrite,
        )
        print(f"[aoi] exported {len(aoi_pairs)} imagery-label AOI pair(s)")
        all_pairs.extend(aoi_pairs)

    styled_label_count = 0
    sidecar_count = 0
    if not args.no_lumid_style:
        if apply_lumid_style_to_raster is None:
            raise RuntimeError("无法导入 lumid_style，无法写入 LUM_ID 色表。")
        label_paths = sorted({pair.label_path for pair in all_pairs})
        for label_path in label_paths:
            apply_lumid_style_to_raster(label_path, label_nodata=args.label_nodata)
            styled_label_count += 1
            if not args.no_style_sidecar:
                if write_style_sidecars_for_raster is None:
                    raise RuntimeError("无法导入 lumid_style，无法导出 qml/clr 样式文件。")
                write_style_sidecars_for_raster(label_path)
                sidecar_count += 1

    qc_alignment = output_root / REL_QC_ALIGNMENT
    qc_class_stats = output_root / REL_QC_CLASS_STATS

    write_alignment_report(all_pairs, qc_alignment)
    write_class_stats(all_pairs, qc_class_stats, label_nodata=args.label_nodata)

    print("\n[done] 输出完成：")
    print(f"  - {output_root / REL_MOSAIC_MAIN}")
    print(f"  - {output_root / REL_MOSAIC_SECONDARY}")
    print(f"  - {output_root / REL_IMG_MAIN_1M}")
    print(f"  - {output_root / REL_IMG_SECONDARY_1M}")
    print(f"  - {output_root / REL_LABEL_MAIN_1M}")
    print(f"  - {output_root / REL_LABEL_SECONDARY_1M}")
    print(f"  - {qc_alignment}")
    print(f"  - {qc_class_stats}")
    if not args.no_lumid_style:
        print(f"  - LUM_ID 色表已写入标签数: {styled_label_count}")
        if not args.no_style_sidecar:
            print(f"  - qml/clr 样式对数量: {sidecar_count}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

