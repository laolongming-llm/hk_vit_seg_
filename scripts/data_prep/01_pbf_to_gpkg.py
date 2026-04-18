#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：01_pbf_to_gpkg.py

目标：
- 将香港地区 OSM `.osm.pbf` 数据转换为 GeoPackage（`.gpkg`）主工作库；
- 在转换阶段尽量保留原始 OSM 标签信息，支撑后续 QGIS 人工清洗、类别映射与语义分割标签构建；
- 从主库中派生导出“香港行政边界”与清洗后的 `multipolygons` 面，用于后续制图与标注处理。

设计原则：
- 优先使用 GDAL/OGR 官方 OSM Driver；
- 默认路径与项目目录结构对齐，可参数化覆盖；
- 强化错误提示与处理，保证流程可复现、可维护；
- 默认不做重投影、不做几何简化，以最大限度保留原始信息。
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Final

try:
    from osgeo import gdal
except Exception as import_exc:  # pragma: no cover
    gdal = None
    GDAL_IMPORT_ERROR = import_exc
else:
    GDAL_IMPORT_ERROR = None

# 项目级默认路径（可通过命令行参数覆盖）
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PBF: Final[Path] = PROJECT_ROOT / "data" / "raw" / "osm" / "hong-kong-260330.osm.pbf"
DEFAULT_OUTPUT_DIR: Final[Path] = PROJECT_ROOT / "data" / "interim" / "gpkg"
DEFAULT_TMP_DIR: Final[Path] = PROJECT_ROOT / "data" / "interim" / "temp" / "gdal_osm"
DEFAULT_DROP_FIDS: Final[tuple[int, ...]] = (1588, 79386, 111736)
DEFAULT_HK_BOUNDARY_NAME: Final[str] = "香港 Hong Kong"

EXPECTED_OSM_LAYERS: Final[tuple[str, ...]] = (
    "points",
    "lines",
    "multilinestrings",
    "multipolygons",
    "other_relations",
)


def require_gdal() -> None:
    """确保 GDAL Python 绑定可用，并在缺失时给出可执行提示。"""
    if gdal is not None:
        return

    detail = f"原始错误：{GDAL_IMPORT_ERROR}" if GDAL_IMPORT_ERROR else "未提供底层异常信息。"
    raise RuntimeError(
        "未检测到 GDAL Python 绑定（osgeo）。\n"
        "请先安装 GDAL 后再执行转换，例如：\n"
        "  conda install -c conda-forge gdal\n"
        f"{detail}"
    )


def to_abs_path(path_str: str) -> Path:
    """将路径解析为绝对路径；若为相对路径，则以项目根目录为基准。"""
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def sql_quote_literal(value: str) -> str:
    """将字符串安全转为 SQL 单引号字面量。"""
    return "'" + value.replace("'", "''") + "'"


def infer_output_gpkg_path(input_pbf: Path, output_arg: str) -> Path:
    """支持 `--output` 传目录或文件路径；目录模式下自动推断输出文件名。"""
    output_path = to_abs_path(output_arg)

    if output_path.suffix.lower() == ".gpkg":
        return output_path

    input_name = input_pbf.name
    if input_name.endswith(".osm.pbf"):
        stem = input_name[: -len(".osm.pbf")]
    else:
        stem = input_pbf.stem

    return output_path / f"{stem}.gpkg"


def infer_hk_boundary_output_path(main_output_gpkg: Path, user_value: str | None) -> Path:
    """推断香港行政边界导出路径。"""
    if user_value:
        p = to_abs_path(user_value)
        return p if p.suffix.lower() == ".gpkg" else p / f"{main_output_gpkg.stem}_hk_boundary.gpkg"
    return main_output_gpkg.with_name(f"{main_output_gpkg.stem}_hk_boundary.gpkg")


def infer_clean_output_path(main_output_gpkg: Path, user_value: str | None) -> Path:
    """推断 multipolygons clean 导出路径。"""
    if user_value:
        p = to_abs_path(user_value)
        return p if p.suffix.lower() == ".gpkg" else p / f"{main_output_gpkg.stem}_multipolygons_clean.gpkg"
    return main_output_gpkg.with_name(f"{main_output_gpkg.stem}_multipolygons_clean.gpkg")


def parse_drop_fids(value: str) -> list[int]:
    """解析 `--drop-fids`，格式示例：`1588,79386,111736`。"""
    cleaned = value.strip()
    if not cleaned:
        return []

    parsed: list[int] = []
    for token in cleaned.split(","):
        t = token.strip()
        if not t:
            continue
        try:
            fid = int(t)
        except ValueError as exc:
            raise ValueError(f"--drop-fids 含非法值：{t}") from exc
        parsed.append(fid)

    return list(dict.fromkeys(parsed))


def ensure_osm_driver_available() -> None:
    """显式检查 OSM Driver，避免运行中期才报错。"""
    require_gdal()
    if gdal.GetDriverByName("OSM") is None:
        raise RuntimeError(
            "当前 GDAL 环境未启用 OSM Driver，无法读取 .osm.pbf。\n"
            "请检查 GDAL 安装（建议使用 conda-forge 或 OSGeo4W 提供的完整 GDAL）。"
        )


def find_default_osmconf(explicit_path: str | None = None) -> Path:
    """查找 Windows 环境下 GDAL 自带 `osmconf.ini`。"""
    require_gdal()

    if explicit_path:
        p = to_abs_path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"指定的 osmconf.ini 不存在: {p}")
        return p

    candidates: list[Path] = []

    gdal_data = os.environ.get("GDAL_DATA") or gdal.GetConfigOption("GDAL_DATA")
    if gdal_data:
        candidates.append(Path(gdal_data) / "osmconf.ini")

    candidates.extend(
        [
            Path(sys.prefix) / "Library" / "share" / "gdal" / "osmconf.ini",
            Path(sys.prefix) / "Library" / "share" / "gdal" / "data" / "osmconf.ini",
            Path(sys.prefix) / "share" / "gdal" / "osmconf.ini",
        ]
    )

    try:
        import osgeo  # noqa: F401

        osgeo_dir = Path(sys.modules["osgeo"].__file__).resolve().parent
        candidates.extend(
            [
                osgeo_dir / "data" / "osmconf.ini",
                osgeo_dir.parent / "share" / "gdal" / "osmconf.ini",
            ]
        )
    except Exception:
        pass

    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "未能自动找到 GDAL 自带的 osmconf.ini。\n"
        "请使用 --base-osmconf 显式指定其路径。"
    )


def _set_or_append_config(text: str, key: str, value: str, replace_all: bool = False) -> str:
    """将 `key=value` 写入配置文本；若不存在则追加。"""
    pattern = rf"(?mi)^\s*#?\s*{re.escape(key)}\s*=.*$"
    count_limit = 0 if replace_all else 1
    new_text, count = re.subn(pattern, f"{key}={value}", text, count=count_limit)

    if count == 0:
        if not new_text.endswith("\n"):
            new_text += "\n"
        new_text += f"{key}={value}\n"

    return new_text


def patch_osmconf(base_conf: Path, out_conf: Path, prefer_json_tags: bool = True) -> Path:
    """
    基于默认 `osmconf.ini` 生成“高保真保 tag”版本。

    关键策略：
    - `report_all_tags=yes`：尽可能上报所有 OSM 标签；
    - `all_tags=yes`：在图层中保留标签集合；
    - `other_tags=no`：避免与 `all_tags` 重复冗余；
    - `tags_format=json`（可选）：在新版本 GDAL 中更易后处理。
    """
    text = base_conf.read_text(encoding="utf-8", errors="ignore")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = _set_or_append_config(text, "report_all_tags", "yes", replace_all=False)
    text = _set_or_append_config(text, "all_tags", "yes", replace_all=True)
    text = _set_or_append_config(text, "other_tags", "no", replace_all=True)

    if prefer_json_tags:
        text = _set_or_append_config(text, "tags_format", "json", replace_all=False)

    out_conf.parent.mkdir(parents=True, exist_ok=True)
    out_conf.write_text(text, encoding="utf-8")
    return out_conf


def cleanup_existing_gpkg(output_gpkg: Path) -> None:
    """清理已有 GeoPackage 及常见 sidecar 文件。"""
    candidates = [
        output_gpkg,
        Path(str(output_gpkg) + "-wal"),
        Path(str(output_gpkg) + "-shm"),
        Path(str(output_gpkg) + "-journal"),
    ]
    for p in candidates:
        if p.exists():
            p.unlink()


def convert_pbf_to_gpkg(
    input_pbf: Path,
    output_gpkg: Path,
    custom_osmconf: Path,
    max_tmp_mb: int = 4096,
    tmp_dir: Path | None = None,
    overwrite: bool = True,
    prefer_json_tags: bool = True,
) -> None:
    """使用 GDAL VectorTranslate 将 `.osm.pbf` 转为 `.gpkg`。"""
    require_gdal()
    gdal.UseExceptions()
    ensure_osm_driver_available()

    if not input_pbf.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_pbf}")

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    if output_gpkg.exists():
        if not overwrite:
            raise FileExistsError(
                f"输出文件已存在: {output_gpkg}\n"
                "如需覆盖，请移除 --no-overwrite 或手动删除旧文件。"
            )
        cleanup_existing_gpkg(output_gpkg)

    if tmp_dir is None:
        tmp_dir = DEFAULT_TMP_DIR
    tmp_dir.mkdir(parents=True, exist_ok=True)

    gdal_version_num = int(gdal.VersionInfo("VERSION_NUM"))
    supports_json_tags = gdal_version_num >= 3070000

    options = [
        "-f",
        "GPKG",
        "-oo",
        f"CONFIG_FILE={custom_osmconf}",
        "-oo",
        "INTERLEAVED_READING=YES",
        "-oo",
        "COMPRESS_NODES=YES",
        "-oo",
        f"MAX_TMPFILE_SIZE={max_tmp_mb}",
        "-dsco",
        "VERSION=1.4",
        "-lco",
        "SPATIAL_INDEX=YES",
        "-progress",
    ]

    if prefer_json_tags and supports_json_tags:
        options.extend(["-oo", "TAGS_FORMAT=JSON"])

    print("=" * 72)
    print("开始执行 OSM PBF -> GPKG 转换")
    print(f"输入 PBF:      {input_pbf}")
    print(f"输出 GPKG:     {output_gpkg}")
    print(f"osmconf 配置:  {custom_osmconf}")
    print(f"临时目录:       {tmp_dir}")
    print(f"GDAL 版本:      {gdal.VersionInfo('--version')}")
    print("=" * 72)

    with gdal.config_options(
        {
            "CPL_TMPDIR": str(tmp_dir),
            "OGR_INTERLEAVED_READING": "YES",
        }
    ):
        ds = gdal.VectorTranslate(
            destNameOrDestDS=str(output_gpkg),
            srcDS=str(input_pbf),
            options=gdal.VectorTranslateOptions(options=options),
        )

    if ds is None:
        raise RuntimeError("GDAL VectorTranslate 返回 None，转换失败。")

    ds = None
    print("转换完成。")


def export_multipolygon_subset(
    source_gpkg: Path,
    output_gpkg: Path,
    sql: str,
    out_layer_name: str = "multipolygons",
    overwrite: bool = True,
) -> None:
    """从主库中按 SQL 导出 multipolygons 子集到独立 GPKG。"""
    require_gdal()
    gdal.UseExceptions()

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)

    if output_gpkg.exists():
        if not overwrite:
            raise FileExistsError(f"输出文件已存在: {output_gpkg}")
        cleanup_existing_gpkg(output_gpkg)

    options = [
        "-f",
        "GPKG",
        "-dialect",
        "SQLITE",
        "-sql",
        sql,
        "-nln",
        out_layer_name,
        "-dsco",
        "VERSION=1.4",
        "-lco",
        "SPATIAL_INDEX=YES",
        "-progress",
    ]

    ds = gdal.VectorTranslate(
        destNameOrDestDS=str(output_gpkg),
        srcDS=str(source_gpkg),
        options=gdal.VectorTranslateOptions(options=options),
    )
    if ds is None:
        raise RuntimeError(f"子集导出失败: {output_gpkg}")
    ds = None


def summarize_output_layers(
    output_gpkg: Path,
    title: str = "输出图层概览",
    check_expected_layers: bool = False,
) -> None:
    """输出图层摘要，便于快速检查。"""
    require_gdal()

    info = gdal.VectorInfo(str(output_gpkg), format="json", deserialize=True)
    layers = info.get("layers", []) if isinstance(info, dict) else []

    if not layers:
        print(f"[警告] {output_gpkg.name} 未读取到图层信息，请在 QGIS 中手动核验。")
        return

    print(f"\n{title}：{output_gpkg.name}")
    observed: list[str] = []
    for layer in layers:
        name = layer.get("name", "UNKNOWN")
        geom = layer.get("geometryType", "UNKNOWN")
        feat_count = layer.get("featureCount", "UNKNOWN")
        observed.append(name)
        print(f"  - 图层: {name:<18} 几何类型: {geom:<20} 要素数: {feat_count}")

    if check_expected_layers:
        missing = [name for name in EXPECTED_OSM_LAYERS if name not in observed]
        if missing:
            print("\n[提示] 以下预期 OSM 图层未出现在主库中：")
            for name in missing:
                print(f"  - {name}")


def run_postprocess_exports(
    main_gpkg: Path,
    hk_boundary_output: Path,
    clean_output: Path,
    drop_fids: list[int],
    hk_boundary_name: str,
    overwrite: bool,
    export_hk_boundary: bool,
    export_clean: bool,
) -> None:
    """执行香港行政边界提取与 clean multipolygons 导出。"""
    fid_clause = ",".join(str(fid) for fid in drop_fids)

    if export_hk_boundary:
        hk_boundary_sql = (
            'SELECT * FROM multipolygons '
            f'WHERE COALESCE("name", \'\') = {sql_quote_literal(hk_boundary_name)}'
        )
        print("\n开始导出香港行政边界图层...")
        export_multipolygon_subset(
            source_gpkg=main_gpkg,
            output_gpkg=hk_boundary_output,
            sql=hk_boundary_sql,
            out_layer_name="multipolygons",
            overwrite=overwrite,
        )
        summarize_output_layers(hk_boundary_output, title="香港行政边界导出结果")

    if export_clean:
        clean_sql = (
            'SELECT * FROM multipolygons '
            'WHERE COALESCE("type", \'\') <> \'boundary\''
        )
        if fid_clause:
            clean_sql += f" AND fid NOT IN ({fid_clause})"

        print("\n开始导出 clean multipolygons 图层...")
        export_multipolygon_subset(
            source_gpkg=main_gpkg,
            output_gpkg=clean_output,
            sql=clean_sql,
            out_layer_name="multipolygons",
            overwrite=overwrite,
        )
        summarize_output_layers(clean_output, title="Multipolygons Clean 导出结果")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 OSM .osm.pbf 高保真转换为 GeoPackage（.gpkg），并导出香港边界与 clean multipolygons"
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(DEFAULT_INPUT_PBF),
        help=f"输入 .osm.pbf 路径（默认：{DEFAULT_INPUT_PBF}）",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(DEFAULT_OUTPUT_DIR),
        help=(
            "主输出路径，可为 .gpkg 文件或目录（默认目录："
            f"{DEFAULT_OUTPUT_DIR}）。目录模式下自动命名为 <pbf文件名>.gpkg"
        ),
    )
    parser.add_argument(
        "--hk-boundary-output",
        "--boundary-output",
        dest="hk_boundary_output",
        default=None,
        help="香港行政边界导出路径（可为 .gpkg 或目录，默认同主输出目录自动命名）",
    )
    parser.add_argument(
        "--hk-boundary-name",
        default=DEFAULT_HK_BOUNDARY_NAME,
        help=f"香港行政边界匹配名称（默认：{DEFAULT_HK_BOUNDARY_NAME}）",
    )
    parser.add_argument(
        "--multipolygons-clean-output",
        default=None,
        help="clean multipolygons 导出路径（可为 .gpkg 或目录，默认同主输出目录自动命名）",
    )
    parser.add_argument(
        "--drop-fids",
        default=",".join(str(fid) for fid in DEFAULT_DROP_FIDS),
        help="从 clean multipolygons 中剔除的 GPKG fid 列表（逗号分隔）",
    )
    parser.add_argument(
        "--no-hk-boundary-export",
        "--no-boundary-export",
        dest="no_hk_boundary_export",
        action="store_true",
        help="禁用香港行政边界导出",
    )
    parser.add_argument(
        "--no-clean-export",
        action="store_true",
        help="禁用 clean multipolygons 导出",
    )
    parser.add_argument(
        "--base-osmconf",
        default=None,
        help="可选：显式指定 GDAL 默认 osmconf.ini 的路径",
    )
    parser.add_argument(
        "--tmp-dir",
        default=str(DEFAULT_TMP_DIR),
        help=f"临时目录（默认：{DEFAULT_TMP_DIR}）",
    )
    parser.add_argument(
        "--max-tmp-mb",
        type=int,
        default=4096,
        help="GDAL OSM 临时数据库阈值（MB），默认 4096",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="若设置该参数，输出文件已存在时将直接报错而非覆盖",
    )
    parser.add_argument(
        "--no-json-tags",
        action="store_true",
        help="禁用 tags_format=json（仅在特殊兼容性场景建议使用）",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_pbf = to_abs_path(args.input)
    main_output_gpkg = infer_output_gpkg_path(input_pbf, args.output)
    hk_boundary_output_gpkg = infer_hk_boundary_output_path(main_output_gpkg, args.hk_boundary_output)
    clean_output_gpkg = infer_clean_output_path(main_output_gpkg, args.multipolygons_clean_output)
    tmp_dir = to_abs_path(args.tmp_dir)

    if not input_pbf.exists():
        raise FileNotFoundError(
            f"输入 PBF 不存在: {input_pbf}\n"
            "请检查 data/raw/osm 下是否已放置 hong-kong-260330.osm.pbf，"
            "或通过 --input 指定其他文件。"
        )

    drop_fids = parse_drop_fids(args.drop_fids)
    overwrite = not args.no_overwrite
    prefer_json_tags = not args.no_json_tags
    export_hk_boundary = not args.no_hk_boundary_export
    export_clean = not args.no_clean_export

    base_conf = find_default_osmconf(args.base_osmconf)
    custom_conf = main_output_gpkg.parent / "osmconf_hk_keep_all_tags.ini"

    patch_osmconf(
        base_conf=base_conf,
        out_conf=custom_conf,
        prefer_json_tags=prefer_json_tags,
    )

    convert_pbf_to_gpkg(
        input_pbf=input_pbf,
        output_gpkg=main_output_gpkg,
        custom_osmconf=custom_conf,
        max_tmp_mb=args.max_tmp_mb,
        tmp_dir=tmp_dir,
        overwrite=overwrite,
        prefer_json_tags=prefer_json_tags,
    )

    summarize_output_layers(main_output_gpkg, title="主库图层概览", check_expected_layers=True)

    run_postprocess_exports(
        main_gpkg=main_output_gpkg,
        hk_boundary_output=hk_boundary_output_gpkg,
        clean_output=clean_output_gpkg,
        drop_fids=drop_fids,
        hk_boundary_name=args.hk_boundary_name,
        overwrite=overwrite,
        export_hk_boundary=export_hk_boundary,
        export_clean=export_clean,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

