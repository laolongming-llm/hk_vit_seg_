#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：03_rasterize_LUMID_classes.py

功能：
- 将带 `LUM_ID` 的 multipolygons 矢量栅格化为模型输入栅格；
- 按区域 `minx,miny,maxx,maxy` 构造矩形范围；
- 分类像元写入对应 `LUM_ID`；
- 未分类/空白像元统一赋值为 255。

说明：
- 输出栅格默认 Byte 类型；
- 若启用目标投影，将先将矢量重投影到目标 EPSG 后再栅格化；
- 默认在非 context 面内按“强语义非叶子面 -> 叶子面”顺序烧录；
- 可选开启 context 补洞：仅在主层仍为 `nodata` 的像元位置填充 context 面结果。
"""

from __future__ import annotations

import argparse
import math
import sys
import uuid
from pathlib import Path
from typing import Final

try:
    from osgeo import gdal, ogr
except Exception as import_exc:  # pragma: no cover
    gdal = None
    ogr = None
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

# ===== 可配置项（路径）=====
# 默认输入：重分类后的 GPKG
DEFAULT_INPUT_GPKG: Final[Path] = PROJECT_ROOT / "data" / "interim" / "cleaned_vectors" / "hk_multipolygons_hydro_reclass.gpkg"
# 默认输出：LUM_ID 栅格
DEFAULT_OUTPUT_TIF: Final[Path] = PROJECT_ROOT / "data" / "interim" / "masks_full" / "hk_landuse_LUMID.tif"
# 默认临时目录：重投影中间文件
DEFAULT_TMP_DIR: Final[Path] = PROJECT_ROOT / "data" / "interim" / "temp" / "rasterize"

# ===== 可配置项（字段与栅格参数）=====
# 图层名
DEFAULT_LAYER_NAME: Final[str] = "multipolygons"
# 分类字段
DEFAULT_LUM_FIELD: Final[str] = "LUM_ID"
# 上下文字段
DEFAULT_CONTEXT_FIELD: Final[str] = "is_context_polygon"
# 叶子面字段
DEFAULT_LEAF_FIELD: Final[str] = "is_leaf_polygon"
# 强语义非叶子面准入字段
DEFAULT_NONLEAF_ALLOW_FIELD: Final[str] = "is_non_leaf_allowed"
# 像元分辨率（目标坐标单位，EPSG:2326 下为米）
DEFAULT_PIXEL_SIZE: Final[float] = 1
# 目标投影
DEFAULT_TARGET_EPSG: Final[int] = 2326
# 未分类/空白像元值
DEFAULT_NODATA: Final[int] = 255

# ===== 可配置项（性能与资源）=====
# GDAL 内部缓存大小（MB）
DEFAULT_GDAL_CACHE_MB: Final[int] = 1024
# 允许的最大像元总数（超出则默认报错，可用 --allow-huge-raster 强制继续）
DEFAULT_MAX_PIXELS: Final[int] = 20_000_000_000
# context 补洞使用内存栅格的最大像元阈值（超出后改用临时 GTiff）
DEFAULT_CONTEXT_MEM_MAX_PIXELS: Final[int] = 150_000_000
# 分块处理时默认窗口大小（像元）
DEFAULT_WINDOW_SIZE: Final[int] = 2048


def require_gdal() -> None:
    if gdal is not None and ogr is not None:
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


def get_layer_epsg(layer: ogr.Layer) -> int | None:
    srs = layer.GetSpatialRef()
    if srs is None:
        return None
    code = srs.GetAuthorityCode(None)
    if code is None:
        return None
    try:
        return int(code)
    except ValueError:
        return None


def cleanup_gpkg(path: Path) -> None:
    for p in [
        path,
        Path(str(path) + "-wal"),
        Path(str(path) + "-shm"),
        Path(str(path) + "-journal"),
    ]:
        if p.exists():
            p.unlink()


def prepare_vector(
    input_gpkg: Path,
    layer_name: str,
    target_epsg: int | None,
    tmp_dir: Path,
    overwrite: bool,
) -> tuple[Path, Path | None]:
    """按需重投影矢量，返回可用于栅格化的路径。"""
    require_gdal()

    src_ds = ogr.Open(str(input_gpkg), 0)
    if src_ds is None:
        raise RuntimeError(f"无法打开输入 GPKG: {input_gpkg}")

    layer = src_ds.GetLayerByName(layer_name)
    if layer is None:
        raise RuntimeError(f"输入中不存在图层: {layer_name}")

    src_epsg = get_layer_epsg(layer)
    src_ds = None

    if target_epsg is None or (src_epsg is not None and src_epsg == target_epsg):
        return input_gpkg, None

    tmp_dir.mkdir(parents=True, exist_ok=True)
    reproj_path = tmp_dir / f"{input_gpkg.stem}_epsg{target_epsg}_{uuid.uuid4().hex[:8]}.gpkg"

    if reproj_path.exists():
        if not overwrite:
            raise FileExistsError(f"临时重投影文件已存在: {reproj_path}")
        cleanup_gpkg(reproj_path)

    options = [
        "-f",
        "GPKG",
        "-t_srs",
        f"EPSG:{target_epsg}",
        "-nln",
        layer_name,
        "-sql",
        f"SELECT * FROM {layer_name}",
        "-dialect",
        "SQLITE",
        "-dsco",
        "VERSION=1.4",
        "-lco",
        "SPATIAL_INDEX=YES",
        "-progress",
    ]

    ds = gdal.VectorTranslate(
        destNameOrDestDS=str(reproj_path),
        srcDS=str(input_gpkg),
        options=gdal.VectorTranslateOptions(options=options),
    )
    if ds is None:
        raise RuntimeError("重投影失败。")
    ds = None

    return reproj_path, reproj_path


def estimate_raster_size(width: int, height: int, bytes_per_pixel: int = 1) -> tuple[int, float]:
    pixels = width * height
    raw_gib = (pixels * bytes_per_pixel) / (1024**3)
    return pixels, raw_gib


def iter_windows(width: int, height: int, window_size: int):
    for yoff in range(0, height, window_size):
        ysize = min(window_size, height - yoff)
        for xoff in range(0, width, window_size):
            xsize = min(window_size, width - xoff)
            yield xoff, yoff, xsize, ysize


def windowed_context_fill(
    main_band,
    context_band,
    nodata: int,
    width: int,
    height: int,
    window_size: int,
) -> int:
    filled = 0
    for xoff, yoff, xsize, ysize in iter_windows(width, height, window_size):
        main_arr = main_band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
        context_arr = context_band.ReadAsArray(xoff=xoff, yoff=yoff, win_xsize=xsize, win_ysize=ysize)
        if main_arr is None or context_arr is None:
            raise RuntimeError("窗口读取栅格数组失败（context fill）。")

        fill_mask = (main_arr == nodata) & (context_arr != nodata)
        c = int(fill_mask.sum())
        if c > 0:
            main_arr[fill_mask] = context_arr[fill_mask]
            main_band.WriteArray(main_arr, xoff=xoff, yoff=yoff)
            filled += c
    return filled


def layer_has_field(layer: ogr.Layer, field_name: str) -> bool:
    return layer.GetLayerDefn().GetFieldIndex(field_name) >= 0


def sql_count(ds_vec: ogr.DataSource, sql: str) -> int:
    lyr = ds_vec.ExecuteSQL(sql, dialect="SQLITE")
    if lyr is None:
        return -1
    feat = lyr.GetNextFeature()
    value = feat.GetField(0) if feat is not None else -1
    ds_vec.ReleaseResultSet(lyr)
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def rasterize_sql_layer(
    ds_vec: ogr.DataSource,
    sql: str,
    target_ds: gdal.Dataset,
    lum_field: str,
    all_touched: bool,
) -> int:
    layer = ds_vec.ExecuteSQL(sql, dialect="SQLITE")
    if layer is None:
        raise RuntimeError("构建栅格化图层失败")

    options = [f"ATTRIBUTE={lum_field}"]
    if all_touched:
        options.append("ALL_TOUCHED=TRUE")

    err = gdal.RasterizeLayer(target_ds, [1], layer, options=options)
    ds_vec.ReleaseResultSet(layer)
    return err


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="将 LUM_ID 分类矢量栅格化")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT_GPKG), help="输入重分类 GPKG")
    parser.add_argument("--output", "-o", default=str(DEFAULT_OUTPUT_TIF), help="输出栅格路径(.tif)")
    parser.add_argument("--layer", default=DEFAULT_LAYER_NAME, help="输入图层名，默认 multipolygons")
    parser.add_argument("--lum-field", default=DEFAULT_LUM_FIELD, help="分类字段名，默认 LUM_ID")
    parser.add_argument("--context-field", default=DEFAULT_CONTEXT_FIELD, help="上下文面字段名，默认 is_context_polygon")
    parser.add_argument("--leaf-field", default=DEFAULT_LEAF_FIELD, help="叶子面字段名，默认 is_leaf_polygon")
    parser.add_argument(
        "--nonleaf-allow-field",
        default=DEFAULT_NONLEAF_ALLOW_FIELD,
        help="强语义非叶子面准入字段名，默认 is_non_leaf_allowed",
    )
    parser.add_argument("--pixel-size", type=float, default=DEFAULT_PIXEL_SIZE, help="像元分辨率（目标坐标单位）")
    parser.add_argument("--target-epsg", type=int, default=DEFAULT_TARGET_EPSG, help="目标 EPSG，默认 2326")
    parser.add_argument("--nodata", type=int, default=DEFAULT_NODATA, help="空白/未分类像元值，默认 255")
    parser.add_argument("--all-touched", action="store_true", help="启用 ALL_TOUCHED=TRUE")
    parser.add_argument("--context-fill", action="store_true", help="启用 context 面补洞（仅填充主层 nodata 像元）")
    parser.add_argument("--gdal-cache-mb", type=int, default=DEFAULT_GDAL_CACHE_MB, help="GDAL 缓存大小（MB）")
    parser.add_argument(
        "--max-pixels",
        type=int,
        default=DEFAULT_MAX_PIXELS,
        help="允许的最大像元总数（超出时默认报错）",
    )
    parser.add_argument("--allow-huge-raster", action="store_true", help="像元总数超阈值时仍继续执行")
    parser.add_argument(
        "--context-mem-max-pixels",
        type=int,
        default=DEFAULT_CONTEXT_MEM_MAX_PIXELS,
        help="context 补洞使用内存栅格的最大像元阈值，超出则用临时 GTiff",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=DEFAULT_WINDOW_SIZE,
        help="context 补洞分块窗口大小（像元）",
    )
    parser.add_argument("--tmp-dir", default=str(DEFAULT_TMP_DIR), help="临时目录")
    parser.add_argument("--keep-temp", action="store_true", help="保留重投影临时文件")
    parser.add_argument("--no-lumid-style", action="store_true", help="不写入 LUM_ID 色表")
    parser.add_argument("--no-style-sidecar", action="store_true", help="不导出同名 qml/clr 样式文件")
    parser.add_argument("--no-overwrite", action="store_true", help="输出存在时不覆盖")
    return parser


def main() -> None:
    require_gdal()
    gdal.UseExceptions()

    args = build_parser().parse_args()

    input_gpkg = to_abs_path(args.input)
    output_tif = to_abs_path(args.output)
    tmp_dir = to_abs_path(args.tmp_dir)
    overwrite = not args.no_overwrite

    if not input_gpkg.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_gpkg}")

    if args.pixel_size <= 0:
        raise ValueError("--pixel-size 必须大于 0")

    if not (0 <= args.nodata <= 255):
        raise ValueError("--nodata 必须位于 0~255（Byte）")
    if args.gdal_cache_mb <= 0:
        raise ValueError("--gdal-cache-mb 必须大于 0")
    if args.max_pixels <= 0:
        raise ValueError("--max-pixels 必须大于 0")
    if args.context_mem_max_pixels <= 0:
        raise ValueError("--context-mem-max-pixels 必须大于 0")
    if args.window_size <= 0:
        raise ValueError("--window-size 必须大于 0")

    gdal.SetCacheMax(int(args.gdal_cache_mb) * 1024 * 1024)

    work_gpkg, temp_gpkg = prepare_vector(
        input_gpkg=input_gpkg,
        layer_name=args.layer,
        target_epsg=args.target_epsg,
        tmp_dir=tmp_dir,
        overwrite=overwrite,
    )

    ds_vec = ogr.Open(str(work_gpkg), 0)
    if ds_vec is None:
        raise RuntimeError(f"无法打开矢量数据: {work_gpkg}")

    layer = ds_vec.GetLayerByName(args.layer)
    if layer is None:
        raise RuntimeError(f"图层不存在: {args.layer}")

    context_field_exists = layer_has_field(layer, args.context_field)
    leaf_field_exists = layer_has_field(layer, args.leaf_field)
    nonleaf_allow_field_exists = layer_has_field(layer, args.nonleaf_allow_field)
    if not context_field_exists:
        print(f"[WARN] 图层中未找到上下文字段 `{args.context_field}`，将不区分 context/non-context。")
    if not leaf_field_exists or not nonleaf_allow_field_exists:
        print(
            f"[WARN] 未找到层级门控字段 (`{args.leaf_field}`/`{args.nonleaf_allow_field}`)，"
            "将回退为普通主层烧录。"
        )

    minx, maxx, miny, maxy = layer.GetExtent()

    # 以 minx/miny 与 maxx/maxy 构造矩形范围
    width = max(1, math.ceil((maxx - minx) / args.pixel_size))
    height = max(1, math.ceil((maxy - miny) / args.pixel_size))
    total_pixels, raw_gib = estimate_raster_size(width, height, bytes_per_pixel=1)

    if total_pixels > args.max_pixels and not args.allow_huge_raster:
        raise RuntimeError(
            "像元总数超出安全阈值。\n"
            f"当前像元数: {total_pixels:,}，约 {raw_gib:.2f} GiB (1-band uint8 原始)\n"
            f"阈值: {args.max_pixels:,}\n"
            "建议调大 --pixel-size 或分区域处理；如确认继续，可加 --allow-huge-raster。"
        )

    output_tif.parent.mkdir(parents=True, exist_ok=True)
    if output_tif.exists():
        if not overwrite:
            raise FileExistsError(f"输出文件已存在: {output_tif}")
        output_tif.unlink()

    driver = gdal.GetDriverByName("GTiff")
    if driver is None:
        raise RuntimeError("未找到 GTiff 驱动")

    raster_ds = driver.Create(
        str(output_tif),
        width,
        height,
        1,
        gdal.GDT_Byte,
        options=[
            "COMPRESS=LZW",
            "TILED=YES",
            "BLOCKXSIZE=512",
            "BLOCKYSIZE=512",
            "SPARSE_OK=TRUE",
            "NUM_THREADS=ALL_CPUS",
            "BIGTIFF=IF_SAFER",
        ],
    )
    if raster_ds is None:
        raise RuntimeError("创建输出栅格失败")

    geotransform = (minx, args.pixel_size, 0.0, maxy, 0.0, -args.pixel_size)
    raster_ds.SetGeoTransform(geotransform)

    srs = layer.GetSpatialRef()
    if srs is not None:
        raster_ds.SetProjection(srs.ExportToWkt())

    band = raster_ds.GetRasterBand(1)
    band.SetNoDataValue(args.nodata)
    band.Fill(args.nodata)

    base_where = f"COALESCE({args.lum_field}, 255) <> 255"
    if context_field_exists:
        base_where = f"{base_where} AND COALESCE({args.context_field}, 0) = 0"
        context_where = (
            f"COALESCE({args.lum_field}, 255) <> 255 "
            f"AND COALESCE({args.context_field}, 0) = 1"
        )
    else:
        context_where = "1=0"

    hierarchy_fields_ready = leaf_field_exists and nonleaf_allow_field_exists
    if hierarchy_fields_ready:
        strong_nonleaf_where = (
            f"{base_where} "
            f"AND COALESCE({args.leaf_field}, 1) = 0 "
            f"AND COALESCE({args.nonleaf_allow_field}, 0) = 1"
        )
        leaf_where = f"{base_where} AND COALESCE({args.leaf_field}, 1) = 1"
    else:
        strong_nonleaf_where = "1=0"
        leaf_where = base_where

    strong_nonleaf_count = sql_count(ds_vec, f"SELECT COUNT(*) FROM {args.layer} WHERE {strong_nonleaf_where}")
    leaf_count = sql_count(ds_vec, f"SELECT COUNT(*) FROM {args.layer} WHERE {leaf_where}")
    main_count = max(strong_nonleaf_count, 0) + max(leaf_count, 0)

    # 先烧录强语义非叶子面，再由叶子面覆盖，保证“细碎面优先”
    if strong_nonleaf_count > 0:
        strong_nonleaf_sql = f"SELECT * FROM {args.layer} WHERE {strong_nonleaf_where} ORDER BY fid ASC"
        err = rasterize_sql_layer(ds_vec, strong_nonleaf_sql, raster_ds, args.lum_field, args.all_touched)
        if err != 0:
            raise RuntimeError(f"强语义非叶子面栅格化失败，错误码: {err}")

    if leaf_count > 0:
        leaf_sql = f"SELECT * FROM {args.layer} WHERE {leaf_where} ORDER BY fid ASC"
        err = rasterize_sql_layer(ds_vec, leaf_sql, raster_ds, args.lum_field, args.all_touched)
        if err != 0:
            raise RuntimeError(f"叶子面栅格化失败，错误码: {err}")

    filled_context_pixels = 0
    context_backend = "disabled"
    if args.context_fill:
        if np is None:
            raise RuntimeError("当前环境缺少 numpy，无法执行 --context-fill。")

        if context_field_exists:
            context_count = sql_count(ds_vec, f"SELECT COUNT(*) FROM {args.layer} WHERE {context_where}")
            if context_count > 0:
                use_mem_context = total_pixels <= args.context_mem_max_pixels
                context_backend = "MEM" if use_mem_context else "GTiffTemp"
                context_ds = None
                context_tmp_tif: Path | None = None
                if use_mem_context:
                    mem_driver = gdal.GetDriverByName("MEM")
                    context_ds = mem_driver.Create("", width, height, 1, gdal.GDT_Byte)
                else:
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    context_tmp_tif = tmp_dir / f"{output_tif.stem}_context_{uuid.uuid4().hex[:8]}.tif"
                    context_ds = driver.Create(
                        str(context_tmp_tif),
                        width,
                        height,
                        1,
                        gdal.GDT_Byte,
                        options=[
                            "COMPRESS=LZW",
                            "TILED=YES",
                            "BLOCKXSIZE=512",
                            "BLOCKYSIZE=512",
                            "SPARSE_OK=TRUE",
                            "NUM_THREADS=ALL_CPUS",
                            "BIGTIFF=IF_SAFER",
                        ],
                    )
                    if context_ds is None:
                        raise RuntimeError("创建 context 临时栅格失败。")

                context_ds.SetGeoTransform(geotransform)
                if srs is not None:
                    context_ds.SetProjection(srs.ExportToWkt())

                context_band = context_ds.GetRasterBand(1)
                context_band.SetNoDataValue(args.nodata)
                context_band.Fill(args.nodata)

                context_sql = f"SELECT * FROM {args.layer} WHERE {context_where} ORDER BY fid ASC"
                err = rasterize_sql_layer(ds_vec, context_sql, context_ds, args.lum_field, args.all_touched)
                if err != 0:
                    raise RuntimeError(f"context 补洞层栅格化失败，错误码: {err}")

                filled_context_pixels = windowed_context_fill(
                    main_band=band,
                    context_band=context_band,
                    nodata=args.nodata,
                    width=width,
                    height=height,
                    window_size=args.window_size,
                )

                context_ds = None
                if context_tmp_tif is not None and context_tmp_tif.exists():
                    context_tmp_tif.unlink()
            else:
                context_backend = "skipped(no_context_features)"
        else:
            print(f"[WARN] --context-fill 已启用，但字段 `{args.context_field}` 不存在，跳过补洞。")
            context_backend = "skipped(no_context_field)"

    band.FlushCache()
    raster_ds.FlushCache()
    raster_ds = None
    ds_vec = None

    if temp_gpkg is not None and not args.keep_temp:
        cleanup_gpkg(temp_gpkg)

    style_qml_path = None
    style_clr_path = None
    if not args.no_lumid_style:
        if apply_lumid_style_to_raster is None:
            raise RuntimeError("无法导入 lumid_style，无法写入 LUM_ID 色表。")
        apply_lumid_style_to_raster(output_tif, label_nodata=args.nodata)
        if not args.no_style_sidecar:
            if write_style_sidecars_for_raster is None:
                raise RuntimeError("无法导入 lumid_style，无法导出 qml/clr 样式文件。")
            style_qml_path, style_clr_path = write_style_sidecars_for_raster(output_tif)

    print(f"输出栅格: {output_tif}")
    if not args.no_lumid_style:
        print("LUM_ID 色表: embedded")
        if style_qml_path is not None and style_clr_path is not None:
            print(f"样式文件(QGIS): {style_qml_path}")
            print(f"样式文件(CLR): {style_clr_path}")
    if main_count >= 0:
        print(f"主层参与栅格化要素数: {main_count}")
    if hierarchy_fields_ready:
        print(f"强语义非叶子面参与数: {max(strong_nonleaf_count, 0)}")
        print(f"叶子面参与数: {max(leaf_count, 0)}")
    if args.context_fill:
        print(f"context 补洞像元数: {filled_context_pixels}")
        print(f"context 补洞后端: {context_backend}")
    print(f"范围 minx/miny/maxx/maxy: {minx}, {miny}, {maxx}, {maxy}")
    print(f"栅格尺寸: {width} x {height}")
    print(f"像元总数: {total_pixels:,} (~{raw_gib:.2f} GiB raw uint8)")
    print(f"像元分辨率: {args.pixel_size}")
    print(f"GDAL 缓存: {args.gdal_cache_mb} MB")
    print(f"未分类/空白值: {args.nodata}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

