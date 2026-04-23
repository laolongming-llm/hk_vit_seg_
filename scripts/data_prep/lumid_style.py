#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LUM_ID 分类色表与样式工具。
用途：
1) 给分类栅格写入 GDAL ColorTable + CategoryNames；
2) 导出 QGIS `.qml` 样式文件；
3) 导出通用 `.clr` 色表文本（可供 ENVI 等软件导入）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# value, english_name, chinese_name, hex, (r,g,b)
LUM_ID_STYLE_8CLASS: Final[list[tuple[int, str, str, str, tuple[int, int, int]]]] = [
    (1, "building_land", "建筑用地", "#E41A1C", (228, 26, 28)),
    (2, "business_land", "商业用地", "#FF7F00", (255, 127, 0)),
    (3, "industrial_land", "工业用地", "#984EA3", (152, 78, 163)),
    (4, "transport_land", "交通用地", "#FFD92F", (255, 217, 47)),
    (5, "infrastructure_land", "基础设施/公共服务用地", "#377EB8", (55, 126, 184)),
    (6, "agricultural_land", "农业用地", "#A6D854", (166, 216, 84)),
    (7, "water_body", "水体", "#1F78B4", (31, 120, 180)),
    (8, "mountainous_land", "山地/林地", "#33A02C", (51, 160, 44)),
    (255, "unknown", "未分类/空白", "#FFFFFF", (255, 255, 255)),
]

# strict7 体系：
# 1..5 保持不变；原 7(水体)->6；原 8(山地/林地)->7；255 为 unknown
LUM_ID_STYLE_STRICT7: Final[list[tuple[int, str, str, str, tuple[int, int, int]]]] = [
    (1, "building_land", "建筑用地", "#E41A1C", (228, 26, 28)),
    (2, "business_land", "商业用地", "#FF7F00", (255, 127, 0)),
    (3, "industrial_land", "工业用地", "#984EA3", (152, 78, 163)),
    (4, "transport_land", "交通用地", "#FFD92F", (255, 217, 47)),
    (5, "infrastructure_land", "基础设施/公共服务用地", "#377EB8", (55, 126, 184)),
    (6, "water_body", "水体", "#1F78B4", (31, 120, 180)),
    (7, "mountainous_land", "山地/林地", "#33A02C", (51, 160, 44)),
    (255, "unknown", "未分类/空白", "#FFFFFF", (255, 255, 255)),
]

# Backward compatibility: historical default style (8-class LUM_ID).
LUM_ID_STYLE: Final[list[tuple[int, str, str, str, tuple[int, int, int]]]] = LUM_ID_STYLE_8CLASS


def resolve_style_entries(
    style_profile: str = "lum8",
) -> list[tuple[int, str, str, str, tuple[int, int, int]]]:
    profile = (style_profile or "lum8").strip().lower()
    if profile in {"lum8", "lum_id", "lumid", "8class", "default"}:
        return list(LUM_ID_STYLE_8CLASS)
    if profile in {"strict7", "v2_strict7", "7class"}:
        return list(LUM_ID_STYLE_STRICT7)
    raise ValueError(f"Unknown style profile: {style_profile}")


def _normalize_style_entries(
    style_entries: list[tuple[int, str, str, str, tuple[int, int, int]]] | None,
) -> list[tuple[int, str, str, str, tuple[int, int, int]]]:
    return list(style_entries) if style_entries is not None else list(LUM_ID_STYLE)


def _hex_to_lower(value: str) -> str:
    return value.lower()


def apply_lumid_style_to_raster(
    raster_path: Path,
    label_nodata: int = 255,
    style_entries: list[tuple[int, str, str, str, tuple[int, int, int]]] | None = None,
) -> None:
    """
    将 LUM_ID 色表写入栅格，便于 QGIS/ENVI 直接按类别上色显示。
    """
    try:
        from osgeo import gdal
    except Exception as exc:
        raise RuntimeError(
            "apply_lumid_style_to_raster 需要 osgeo.gdal；当前环境未安装 GDAL。"
        ) from exc

    ds = gdal.Open(str(raster_path), gdal.GA_Update)
    if ds is None:
        raise RuntimeError(f"无法打开栅格（写入色表失败）：{raster_path}")

    band = ds.GetRasterBand(1)
    if band is None:
        ds = None
        raise RuntimeError(f"栅格无波段（写入色表失败）：{raster_path}")

    # 仅对 Byte 分类栅格应用标准色表
    if band.DataType != gdal.GDT_Byte:
        ds = None
        raise RuntimeError(
            f"当前栅格不是 Byte 类型，无法按 LUM_ID 标准色表写入：{raster_path}"
        )

    ct = gdal.ColorTable()
    # 对未定义类别给透明黑，避免误导显示
    for idx in range(256):
        ct.SetColorEntry(idx, (0, 0, 0, 0))

    category_names = [""] * 256
    entries = _normalize_style_entries(style_entries)
    for value, en_name, zh_name, _, (r, g, b) in entries:
        ct.SetColorEntry(int(value), (int(r), int(g), int(b), 255))
        category_names[int(value)] = f"{value} {en_name} ({zh_name})"

    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)
    band.SetRasterColorTable(ct)
    band.SetCategoryNames(category_names)
    band.SetNoDataValue(label_nodata)
    band.SetDescription("LUM_ID")
    band.FlushCache()
    ds.FlushCache()
    ds = None


def write_qgis_qml(
    qml_path: Path,
    label_field_name: str = "LUM_ID",
    style_entries: list[tuple[int, str, str, str, tuple[int, int, int]]] | None = None,
) -> None:
    """
    导出 QGIS 调色板样式文件（可直接“加载样式”）。
    """
    qml_path.parent.mkdir(parents=True, exist_ok=True)

    entries = _normalize_style_entries(style_entries)
    values = [int(item[0]) for item in entries]
    vmin = min(values) if values else 0
    vmax = max(values) if values else 255

    lines: list[str] = [
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
        "<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>",
        "<qgis styleCategories=\"Symbology\" version=\"3.34.0\">",
        "  <pipe>",
        (
            "    <rasterrenderer alphaBand=\"-1\" band=\"1\" "
            f"classificationMax=\"{vmax}\" classificationMin=\"{vmin}\" opacity=\"1\" type=\"paletted\">"
        ),
        "      <colorPalette>",
    ]
    for value, en_name, zh_name, hex_color, _ in entries:
        label = f"{value} {en_name} ({zh_name})"
        lines.append(
            f"        <paletteEntry alpha=\"255\" color=\"{_hex_to_lower(hex_color)}\" label=\"{label}\" value=\"{value}\"/>"
        )
    lines.extend(
        [
            "      </colorPalette>",
            "    </rasterrenderer>",
            "    <brightnesscontrast brightness=\"0\" contrast=\"0\" gamma=\"1\"/>",
            "    <huesaturation colorizeBlue=\"128\" colorizeGreen=\"128\" colorizeOn=\"0\" colorizeRed=\"255\" colorizeStrength=\"100\" grayscaleMode=\"0\" invertColors=\"0\" saturation=\"0\"/>",
            "    <rasterresampler maxOversampling=\"2\">",
            "      <bilinearRasterResampler/>",
            "      <cubicRasterResampler/>",
            "    </rasterresampler>",
            "  </pipe>",
            f"  <customproperties><property key=\"lumid_field\" value=\"{label_field_name}\"/></customproperties>",
            "</qgis>",
            "",
        ]
    )
    qml_path.write_text("\n".join(lines), encoding="utf-8")


def write_clr(
    clr_path: Path,
    style_entries: list[tuple[int, str, str, str, tuple[int, int, int]]] | None = None,
) -> None:
    """
    导出通用 clr 色表文本（value R G B label）。
    """
    clr_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# value R G B label"]
    entries = _normalize_style_entries(style_entries)
    for value, en_name, zh_name, _, (r, g, b) in entries:
        lines.append(f"{value} {r} {g} {b} {en_name}({zh_name})")
    lines.append("")
    clr_path.write_text("\n".join(lines), encoding="utf-8")


def write_style_sidecars_for_raster(
    raster_path: Path,
    style_entries: list[tuple[int, str, str, str, tuple[int, int, int]]] | None = None,
) -> tuple[Path, Path]:
    """
    为指定栅格生成同名 `.qml` 与 `.clr` 文件。
    """
    qml_path = raster_path.with_suffix(".qml")
    clr_path = raster_path.with_suffix(".clr")
    write_qgis_qml(qml_path, style_entries=style_entries)
    write_clr(clr_path, style_entries=style_entries)
    return qml_path, clr_path
