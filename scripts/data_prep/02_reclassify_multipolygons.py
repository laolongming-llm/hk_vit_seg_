#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：02_reclassify_multipolygons.py

功能：
1) 读取 hong-kong-260330_multipolygons_clean.gpkg；
2) 按规则将 multipolygons 重分类到水动力模型 8 类；
3) 输出带 LUM_ID 的重分类 GPKG；
4) 输出 unknown(=255) 子集 GPKG；
5) 输出分类审计统计 CSV。

说明：
- LUM_ID: 1~8 为有效地物分类；255 为未分类/空白。
- LUM_ID 对应地物类型：
  - 1: building_land（建筑用地） *
  - 2: business_land（商业用地）
  - 3: industrial_land（工业用地） *
  - 4: transport_land（交通用地）
  - 5: infrastructure_land（基础设施/公共服务用地） *
  - 6: agricultural_land（农业用地）
  - 7: water_body（水体，含鱼塘与红树林）
  - 8: mountainous_land（山地/自然地）
- 本脚本仅处理矢量重分类，不进行栅格化。
"""

from __future__ import annotations

import argparse
import csv
import struct
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from typing import Final

try:
    from osgeo import ogr
except Exception:
    ogr = None

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[2]

# ===== 可配置项（路径）=====
# 输入 clean multipolygons GPKG
DEFAULT_INPUT_GPKG: Final[Path] = PROJECT_ROOT / "data" / "interim" / "gpkg" / "hong-kong-260330_multipolygons_clean.gpkg"
# 输出重分类 GPKG
DEFAULT_RECLASS_GPKG: Final[Path] = PROJECT_ROOT / "data" / "interim" / "cleaned_vectors" / "hk_multipolygons_hydro_reclass.gpkg"
# 输出 unknown 子集 GPKG
DEFAULT_UNKNOWN_GPKG: Final[Path] = PROJECT_ROOT / "data" / "interim" / "cleaned_vectors" / "hk_multipolygons_unknown.gpkg"
# 输出审计 CSV
DEFAULT_AUDIT_CSV: Final[Path] = PROJECT_ROOT / "data" / "interim" / "cleaned_vectors" / "hk_mapping_audit.csv"

# ===== 可配置项（表结构）=====
TABLE_NAME: Final[str] = "multipolygons"
RTREE_TABLE_NAME: Final[str] = f"rtree_{TABLE_NAME}_geom"

# ===== 可配置项（层级判定）=====
# bbox_exact: 候选 bbox + 精确几何 contains；bbox: 仅 bbox
DEFAULT_HIERARCHY_MODE: Final[str] = "bbox_exact"
# 精确 contains 默认处理全部候选父面（0=全部）
DEFAULT_EXACT_PARENT_LIMIT: Final[int] = 0
# 精确 contains 阶段进度打印间隔（父面数）
EXACT_PROGRESS_EVERY: Final[int] = 2000
# 子面几何对象缓存上限（超出后清空缓存以控制内存）
EXACT_GEOM_CACHE_CAP: Final[int] = 8000
# SQLite IN 子句分块大小（批量回写）
SQLITE_IN_CHUNK_SIZE: Final[int] = 900

# ===== 可配置项（规则词表）=====
# wetland 子类型中归入水体类（LUM_ID=7）的集合（如红树林）
WETLAND_TO_MANGROVE_SUBTYPES: Final[tuple[str, ...]] = ("mangrove",)
# wetland 子类型中归入自然地（LUM_ID=8）的集合（不含红树林）
WETLAND_TO_NATURAL_LAND_SUBTYPES: Final[tuple[str, ...]] = ("saltmarsh", "swamp", "reedbed")
# 交通类中保留的 amenity
TRANSPORT_AMENITY_VALUES: Final[tuple[str, ...]] = ("bus_station", "ferry_terminal")
# 基础设施类 amenity（含停车相关）
INFRA_AMENITY_VALUES: Final[tuple[str, ...]] = (
    "parking",
    "parking_space",
    "bicycle_parking",
    "motorcycle_parking",
    "waste_transfer_station",
    "recycling",
    "fuel",
    "fire_station",
    "police",
    "grave_yard",
    "school",
    "college",
    "university",
    "hospital",
    "clinic",
    "prison",
    "courthouse",
    "townhall",
    "library",
)

CORE_EMPTY_EXPR: Final[str] = (
    "COALESCE(landuse,'')='' AND COALESCE(natural,'')='' AND COALESCE(building,'')='' "
    "AND COALESCE(amenity,'')='' AND COALESCE(leisure,'')='' AND COALESCE(man_made,'')='' "
    "AND COALESCE(shop,'')='' AND COALESCE(office,'')='' AND COALESCE(tourism,'')='' "
    "AND COALESCE(sport,'')='' AND COALESCE(aeroway,'')='' AND COALESCE(military,'')='' "
    "AND COALESCE(craft,'')='' AND COALESCE(historic,'')='' AND COALESCE(geological,'')=''"
)


def sqlite_st_is_empty(geom_blob: object) -> int:
    """
    兼容 sqlite3 直连 GPKG 时缺失的 ST_IsEmpty。
    仅用于触发器条件判断：1=empty, 0=not empty。
    """
    if geom_blob is None:
        return 1

    if isinstance(geom_blob, memoryview):
        geom_blob = geom_blob.tobytes()

    if not isinstance(geom_blob, (bytes, bytearray)):
        return 0

    if len(geom_blob) == 0:
        return 1

    # GeoPackageBinary header: [0:2]="GP", [3]=flags, bit 4(0x10)=empty
    if len(geom_blob) >= 4 and geom_blob[0:2] == b"GP":
        return 1 if (geom_blob[3] & 0x10) else 0

    return 0


def gpkg_bounds(geom_blob: object) -> tuple[float, float, float, float] | None:
    if geom_blob is None:
        return None

    if isinstance(geom_blob, memoryview):
        geom_blob = geom_blob.tobytes()

    if not isinstance(geom_blob, (bytes, bytearray)):
        return None

    if len(geom_blob) < 8 or geom_blob[0:2] != b"GP":
        return None

    flags = geom_blob[3]
    envelope_indicator = (flags >> 1) & 0x07
    if envelope_indicator == 0:
        return None

    envelope_size_map = {1: 4, 2: 6, 3: 6, 4: 8}
    n_doubles = envelope_size_map.get(envelope_indicator)
    if n_doubles is None:
        return None

    endian = "<" if (flags & 0x01) else ">"
    envelope_bytes = 8 * n_doubles
    if len(geom_blob) < 8 + envelope_bytes:
        return None

    vals = struct.unpack_from(f"{endian}{n_doubles}d", geom_blob, 8)
    minx, maxx, miny, maxy = vals[0], vals[1], vals[2], vals[3]
    return (minx, maxx, miny, maxy)


def gpkg_blob_to_wkb(geom_blob: object) -> bytes | None:
    if geom_blob is None:
        return None
    if isinstance(geom_blob, memoryview):
        geom_blob = geom_blob.tobytes()
    if not isinstance(geom_blob, (bytes, bytearray)):
        return None
    if len(geom_blob) == 0:
        return None
    if geom_blob[0:2] != b"GP":
        return bytes(geom_blob)
    if len(geom_blob) < 8:
        return None

    flags = geom_blob[3]
    envelope_indicator = (flags >> 1) & 0x07
    envelope_size_map = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}
    n_doubles = envelope_size_map.get(envelope_indicator)
    if n_doubles is None:
        return None
    wkb_offset = 8 + 8 * n_doubles
    if wkb_offset >= len(geom_blob):
        return None
    return bytes(geom_blob[wkb_offset:])


def gpkg_blob_to_ogr_geometry(geom_blob: object):
    if ogr is None:
        return None
    wkb = gpkg_blob_to_wkb(geom_blob)
    if wkb is None:
        return None
    try:
        geom = ogr.CreateGeometryFromWkb(wkb)
    except Exception:
        return None
    return geom


def sqlite_st_minx(geom_blob: object) -> float | None:
    bounds = gpkg_bounds(geom_blob)
    return None if bounds is None else bounds[0]


def sqlite_st_maxx(geom_blob: object) -> float | None:
    bounds = gpkg_bounds(geom_blob)
    return None if bounds is None else bounds[1]


def sqlite_st_miny(geom_blob: object) -> float | None:
    bounds = gpkg_bounds(geom_blob)
    return None if bounds is None else bounds[2]


def sqlite_st_maxy(geom_blob: object) -> float | None:
    bounds = gpkg_bounds(geom_blob)
    return None if bounds is None else bounds[3]


def connect_gpkg(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.create_function("ST_IsEmpty", 1, sqlite_st_is_empty)
    conn.create_function("ST_MinX", 1, sqlite_st_minx)
    conn.create_function("ST_MaxX", 1, sqlite_st_maxx)
    conn.create_function("ST_MinY", 1, sqlite_st_miny)
    conn.create_function("ST_MaxY", 1, sqlite_st_maxy)
    return conn


def to_abs_path(path_str: str) -> Path:
    p = Path(path_str).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def sql_text_in(field_name: str, values: tuple[str, ...]) -> str:
    quoted = ",".join([f"'{v}'" for v in values])
    return f"COALESCE({field_name},'') IN ({quoted})"


def wetland_other_tags_cond(subtypes: tuple[str, ...]) -> str:
    return " OR ".join(
        [f"instr(COALESCE(other_tags,''), '\"wetland\"=>\"{subtype}\"') > 0" for subtype in subtypes]
    )


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if not overwrite:
            raise FileExistsError(f"输出文件已存在: {dst}")
        dst.unlink()
    shutil.copy2(src, dst)


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(r[1] == column for r in rows)


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1;",
        (table,),
    ).fetchone()
    return row is not None


def ensure_columns(conn: sqlite3.Connection) -> None:
    cols = [
        ("LUM_ID", "INTEGER"),
        ("LUM_NAME", "TEXT"),
        ("rule_id", "TEXT"),
        ("rule_level", "TEXT"),
        ("is_context_polygon", "INTEGER"),
        ("confidence", "TEXT"),
        ("child_count", "INTEGER"),
        ("is_leaf_polygon", "INTEGER"),
        ("is_strong_non_leaf_candidate", "INTEGER"),
        ("is_non_leaf_allowed", "INTEGER"),
    ]
    for col_name, col_type in cols:
        if not column_exists(conn, TABLE_NAME, col_name):
            conn.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN {col_name} {col_type};")


def init_classification_fields(conn: sqlite3.Connection) -> None:
    # 统一初始化为 unknown(255)
    conn.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET LUM_ID = 255,
            LUM_NAME = 'unknown',
            rule_id = 'UNMAPPED',
            rule_level = 'unmapped',
            confidence = 'low',
            child_count = 0,
            is_leaf_polygon = 1,
            is_strong_non_leaf_candidate = 0,
            is_non_leaf_allowed = 0;
        """
    )

    # 标记上下文面：边界/区域语义字段非空，且核心地物字段全空
    conn.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET is_context_polygon = CASE
            WHEN (
                COALESCE(boundary,'') <> ''
                OR COALESCE(admin_level,'') <> ''
                OR COALESCE(place,'') <> ''
                OR COALESCE(name,'') <> ''
            ) AND {CORE_EMPTY_EXPR} THEN 1
            ELSE 0
        END;
        """
    )


def compute_hierarchy_flags(
    conn: sqlite3.Connection,
    hierarchy_mode: str = DEFAULT_HIERARCHY_MODE,
    exact_parent_limit: int = DEFAULT_EXACT_PARENT_LIMIT,
) -> tuple[int, int, int]:
    if not table_exists(conn, RTREE_TABLE_NAME):
        conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET child_count = 0,
                is_leaf_polygon = 1;
            """
        )
        return (0, 0, 0)

    # 阶段1：基于 RTree 包络框包含关系计算 child_count（候选）
    conn.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET child_count = (
            SELECT COUNT(b.id)
            FROM {RTREE_TABLE_NAME} AS a
            JOIN {RTREE_TABLE_NAME} AS b
              ON b.id != a.id
             AND b.minx >= a.minx
             AND b.maxx <= a.maxx
             AND b.miny >= a.miny
             AND b.maxy <= a.maxy
            WHERE a.id = {TABLE_NAME}.fid
        );
        """
    )
    conn.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET is_leaf_polygon = CASE WHEN COALESCE(child_count, 0) = 0 THEN 1 ELSE 0 END;
        """
    )
    bbox_non_leaf = conn.execute(
        f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE COALESCE(child_count,0) > 0;"
    ).fetchone()[0]

    if hierarchy_mode != "bbox_exact":
        return (bbox_non_leaf, 0, 0)

    if ogr is None:
        print("[WARN] 未检测到 osgeo.ogr，层级判定回退为 bbox-only。")
        return (bbox_non_leaf, 0, 0)

    # 阶段2：对候选父面做精确几何 contains 判定，去除 bbox 误判
    parent_sql = (
        f"SELECT t.fid, t.geom, r.minx, r.maxx, r.miny, r.maxy "
        f"FROM {TABLE_NAME} AS t "
        f"JOIN {RTREE_TABLE_NAME} AS r ON r.id=t.fid "
        f"WHERE COALESCE(t.child_count,0) > 0 "
        f"ORDER BY t.fid"
    )
    if exact_parent_limit > 0:
        parent_sql += f" LIMIT {int(exact_parent_limit)}"

    candidate_sql = (
        f"SELECT c.fid, c.geom "
        f"FROM {TABLE_NAME} AS c "
        f"JOIN {RTREE_TABLE_NAME} AS rc ON rc.id=c.fid "
        f"WHERE c.fid <> ? "
        f"AND rc.minx >= ? AND rc.maxx <= ? "
        f"AND rc.miny >= ? AND rc.maxy <= ?"
    )

    no_exact_child_fids: list[int] = []
    child_geom_cache: dict[int, object] = {}
    processed = 0
    t0 = time.time()

    for fid, parent_blob, minx, maxx, miny, maxy in conn.execute(parent_sql):
        processed += 1
        parent_geom = gpkg_blob_to_ogr_geometry(parent_blob)
        if parent_geom is None or parent_geom.IsEmpty():
            no_exact_child_fids.append(int(fid))
            continue

        exact_found = False
        for child_fid, child_blob in conn.execute(candidate_sql, (fid, minx, maxx, miny, maxy)):
            child_geom = child_geom_cache.get(int(child_fid))
            if child_geom is None:
                child_geom = gpkg_blob_to_ogr_geometry(child_blob)
                if child_geom is None or child_geom.IsEmpty():
                    continue
                child_geom_cache[int(child_fid)] = child_geom
                if len(child_geom_cache) > EXACT_GEOM_CACHE_CAP:
                    child_geom_cache.clear()
            try:
                if parent_geom.Contains(child_geom):
                    exact_found = True
                    break
            except Exception:
                continue

        if not exact_found:
            no_exact_child_fids.append(int(fid))

        if processed % EXACT_PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            print(
                f"[hierarchy] exact contains refined {processed} parents, "
                f"temporary no-child={len(no_exact_child_fids)}, elapsed={elapsed:.1f}s"
            )

    # 批量回写：bbox 误判为“有子面”的父面重置为叶子面
    for i in range(0, len(no_exact_child_fids), SQLITE_IN_CHUNK_SIZE):
        chunk = no_exact_child_fids[i : i + SQLITE_IN_CHUNK_SIZE]
        placeholders = ",".join(["?"] * len(chunk))
        conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET child_count = 0,
                is_leaf_polygon = 1
            WHERE fid IN ({placeholders});
            """,
            chunk,
        )

    if exact_parent_limit > 0 and exact_parent_limit < bbox_non_leaf:
        print(
            f"[WARN] exact contains 仅处理前 {exact_parent_limit} 个父面，"
            "其余仍保留 bbox 判定结果。"
        )

    return (bbox_non_leaf, len(no_exact_child_fids), processed)


def mark_non_leaf_participation(conn: sqlite3.Connection) -> int:
    conn.execute(
        f"""
        UPDATE {TABLE_NAME}
        SET is_strong_non_leaf_candidate = 0,
            is_non_leaf_allowed = CASE WHEN COALESCE(rule_level,'')='manual_override' THEN 1 ELSE 0 END;
        """
    )

    wetland_natural_land_cond = wetland_other_tags_cond(WETLAND_TO_NATURAL_LAND_SUBTYPES)
    wetland_mangrove_cond = wetland_other_tags_cond(WETLAND_TO_MANGROVE_SUBTYPES)

    strong_mountain_non_leaf = (
        "COALESCE(LUM_ID,255)=8 "
        "AND COALESCE(rule_level,'')='tag_direct' "
        "AND COALESCE(confidence,'')='high' "
        "AND ("
        "COALESCE(natural,'') IN ('wood','grassland','scrub','bare_rock','heath','scree','rock','cliff','shingle','slope','fell','landslide') "
        "OR COALESCE(landuse,'') IN ('forest','grass','greenfield') "
        f"OR ({wetland_natural_land_cond})"
        ")"
    )
    strong_water_non_leaf = (
        "COALESCE(LUM_ID,255)=7 "
        "AND COALESCE(rule_level,'')='tag_direct' "
        "AND COALESCE(confidence,'')='high' "
        "AND ("
        "COALESCE(natural,'') IN ('water','bay','strait','reef','shoal','mud') "
        "OR (COALESCE(natural,'')='wetland' AND NOT ("
        f"{wetland_mangrove_cond} OR "
        f"{wetland_natural_land_cond}"
        "))"
        ")"
    )

    sql = f"""
    UPDATE {TABLE_NAME}
    SET is_strong_non_leaf_candidate = 1,
        is_non_leaf_allowed = 1
    WHERE COALESCE(is_leaf_polygon, 1) = 0
      AND COALESCE(is_context_polygon, 0) = 0
      AND (
        ({strong_mountain_non_leaf})
        OR ({strong_water_non_leaf})
      );
    """
    cur = conn.execute(sql)
    return cur.rowcount if cur.rowcount is not None else 0


def apply_hierarchy_gate(conn: sqlite3.Connection) -> int:
    # 非叶子面默认不参与，只有强语义白名单或人工强制规则可保留
    sql = f"""
    UPDATE {TABLE_NAME}
    SET LUM_ID = 255,
        LUM_NAME = 'unknown',
        rule_id = 'NON_LEAF_EXCLUDED',
        rule_level = 'hierarchy_gate',
        confidence = 'low'
    WHERE COALESCE(is_leaf_polygon, 1) = 0
      AND COALESCE(is_non_leaf_allowed, 0) = 0
      AND COALESCE(rule_level, '') <> 'manual_override'
      AND COALESCE(LUM_ID, 255) <> 255;
    """
    cur = conn.execute(sql)
    return cur.rowcount if cur.rowcount is not None else 0


def apply_rule(
    conn: sqlite3.Connection,
    lum_id: int,
    lum_name: str,
    rule_id: str,
    rule_level: str,
    confidence: str,
    condition_sql: str,
) -> int:
    sql = f"""
    UPDATE {TABLE_NAME}
    SET LUM_ID = ?,
        LUM_NAME = ?,
        rule_id = ?,
        rule_level = ?,
        confidence = ?
    WHERE COALESCE(LUM_ID, 255) = 255
      AND ({condition_sql});
    """
    cur = conn.execute(sql, (lum_id, lum_name, rule_id, rule_level, confidence))
    return cur.rowcount if cur.rowcount is not None else 0


def apply_force_rule(
    conn: sqlite3.Connection,
    lum_id: int,
    lum_name: str,
    rule_id: str,
    rule_level: str,
    confidence: str,
    condition_sql: str,
) -> int:
    # 强制规则：覆盖已有分类结果（用于人工确认的特例修正）
    sql = f"""
    UPDATE {TABLE_NAME}
    SET LUM_ID = ?,
        LUM_NAME = ?,
        rule_id = ?,
        rule_level = ?,
        confidence = ?
    WHERE ({condition_sql});
    """
    cur = conn.execute(sql, (lum_id, lum_name, rule_id, rule_level, confidence))
    return cur.rowcount if cur.rowcount is not None else 0


def run_rules(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    stats: list[tuple[str, int]] = []

    # 严格鱼塘规则：other_tags 仅包含 "water"=>"pond"，直接并入水体类(LUM_ID=7)
    fish_pond_only_pond_cond = (
        "COALESCE(natural,'')='water' "
        "AND REPLACE(TRIM(COALESCE(other_tags,'')), ' ', '') = '\"water\"=>\"pond\"'"
    )
    stats.append(
        (
            "FISHPOND_ONLY_WATER_POND",
            apply_rule(
                conn,
                7,
                "water_body",
                "FISHPOND_ONLY_WATER_POND",
                "tag_indirect",
                "high",
                fish_pond_only_pond_cond,
            ),
        )
    )

    # 既有鱼塘规则：natural=water 且 other_tags 含 water=pond（可带 aquaculture=yes），并入水体类
    fish_pond_cond = (
        "COALESCE(natural,'')='water' AND ("
        "(instr(COALESCE(other_tags,''), '\"aquaculture\"=>\"yes\"') > 0 "
        "AND instr(COALESCE(other_tags,''), '\"water\"=>\"pond\"') > 0) "
        "OR instr(COALESCE(other_tags,''), '\"water\"=>\"pond\"') > 0"
        ")"
    )
    stats.append(
        (
            "FISHPOND_OTHER_TAG",
            apply_rule(
                conn,
                7,
                "water_body",
                "FISHPOND_OTHER_TAG",
                "tag_indirect",
                "high",
                fish_pond_cond,
            ),
        )
    )

    mangrove_cond = (
        "COALESCE(natural,'')='mangrove' OR ("
        "COALESCE(natural,'')='wetland' AND ("
        f"{wetland_other_tags_cond(WETLAND_TO_MANGROVE_SUBTYPES)}"
        ")"
        ")"
    )
    stats.append(
        (
            "MANGROVE_TAG",
            apply_rule(
                conn,
                7,
                "water_body",
                "MANGROVE_TAG",
                "tag_direct",
                "high",
                mangrove_cond,
            ),
        )
    )

    wetland_natural_land_cond = (
        "COALESCE(natural,'')='wetland' AND ("
        f"{wetland_other_tags_cond(WETLAND_TO_NATURAL_LAND_SUBTYPES)}"
        ")"
    )
    stats.append(
        (
            "WETLAND_NATURAL_LAND",
            apply_rule(
                conn,
                8,
                "mountainous_land",
                "WETLAND_NATURAL_LAND",
                "tag_direct",
                "high",
                wetland_natural_land_cond,
            ),
        )
    )

    water_cond = (
        "COALESCE(natural,'') IN ('water','bay','strait','wetland','reef','shoal','mud')"
    )
    stats.append(
        (
            "WATER_NATURAL",
            apply_rule(
                conn,
                7,
                "water_body",
                "WATER_NATURAL",
                "tag_direct",
                "high",
                water_cond,
            ),
        )
    )

    industrial_cond = (
        "COALESCE(landuse,'') IN ('industrial','brownfield','landfill') "
        "OR COALESCE(building,'') IN ('industrial','warehouse')"
    )
    stats.append(
        (
            "INDUSTRIAL_TAG",
            apply_rule(
                conn,
                3,
                "industrial_land",
                "INDUSTRIAL_TAG",
                "tag_direct",
                "high",
                industrial_cond,
            ),
        )
    )

    business_cond = (
        "COALESCE(landuse,'') IN ('commercial','retail') "
        "OR COALESCE(shop,'')<>'' "
        "OR COALESCE(office,'')<>'' "
        "OR COALESCE(tourism,'') IN ('hotel','hostel','guest_house','motel','apartment') "
        "OR COALESCE(building,'') IN ('commercial','retail','office','hotel')"
    )
    stats.append(
        (
            "BUSINESS_TAG",
            apply_rule(
                conn,
                2,
                "business_land",
                "BUSINESS_TAG",
                "tag_direct",
                "high",
                business_cond,
            ),
        )
    )

    transport_cond = (
        "COALESCE(aeroway,'')<>'' "
        "OR COALESCE(landuse,'') IN ('highway','railway') "
        f"OR {sql_text_in('amenity', TRANSPORT_AMENITY_VALUES)} "
        "OR COALESCE(man_made,'') IN ('bridge','pier','quay','container_terminal')"
    )
    stats.append(
        (
            "TRANSPORT_TAG",
            apply_rule(
                conn,
                4,
                "transport_land",
                "TRANSPORT_TAG",
                "tag_direct",
                "high",
                transport_cond,
            ),
        )
    )

    infrastructure_cond = (
        "COALESCE(man_made,'') IN ('reservoir_covered','pumping_station','wastewater_plant','water_works','storage_tank','water_tower') "
        f"OR {sql_text_in('amenity', INFRA_AMENITY_VALUES)} "
        "OR COALESCE(landuse,'') IN ('cemetery','military','institutional','religious','education')"
    )
    stats.append(
        (
            "INFRA_TAG",
            apply_rule(
                conn,
                5,
                "infrastructure_land",
                "INFRA_TAG",
                "tag_direct",
                "high",
                infrastructure_cond,
            ),
        )
    )

    agricultural_cond = (
        "COALESCE(landuse,'') IN ('farmland','farmyard','orchard','greenhouse_horticulture','plant_nursery','allotments')"
    )
    stats.append(
        (
            "AGRI_TAG",
            apply_rule(
                conn,
                6,
                "agricultural_land",
                "AGRI_TAG",
                "tag_direct",
                "high",
                agricultural_cond,
            ),
        )
    )

    mountainous_cond = (
        "COALESCE(natural,'') IN ('wood','grassland','scrub','bare_rock','heath','scree','rock','cliff','shingle','slope','fell','landslide') "
        "OR COALESCE(landuse,'') IN ('forest','grass','greenfield')"
    )
    stats.append(
        (
            "MOUNTAIN_TAG",
            apply_rule(
                conn,
                8,
                "mountainous_land",
                "MOUNTAIN_TAG",
                "tag_direct",
                "high",
                mountainous_cond,
            ),
        )
    )

    # 利用 other_tags 补识别 building:part
    building_part_cond = "instr(COALESCE(other_tags,'') , 'building:part') > 0"
    stats.append(
        (
            "BUILDING_PART_FALLBACK",
            apply_rule(
                conn,
                1,
                "building_land",
                "BUILDING_PART_FALLBACK",
                "tag_indirect",
                "medium",
                building_part_cond,
            ),
        )
    )

    building_cond = "COALESCE(building,'')<>''"
    stats.append(
        (
            "BUILDING_TAG",
            apply_rule(
                conn,
                1,
                "building_land",
                "BUILDING_TAG",
                "tag_direct",
                "high",
                building_cond,
            ),
        )
    )

    # name 兜底规则（仅处理仍 unknown 且非上下文面的对象）
    business_name_cond = (
        "("
        "LOWER(COALESCE(name,'')) LIKE '%mall%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%shopping%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%plaza%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%galleria%' "
        "OR COALESCE(name,'') LIKE '%商場%' "
        "OR COALESCE(name,'') LIKE '%廣場%'"
        ") AND COALESCE(is_context_polygon, 0) = 0"
    )
    stats.append(
        (
            "BUSINESS_NAME",
            apply_rule(
                conn,
                2,
                "business_land",
                "BUSINESS_NAME",
                "name_fallback",
                "medium",
                business_name_cond,
            ),
        )
    )

    transport_name_cond = (
        "("
        "LOWER(COALESCE(name,'')) LIKE '%terminal%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%station%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%apron%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%airport%' "
        "OR COALESCE(name,'') LIKE '%碼頭%' "
        "OR COALESCE(name,'') LIKE '%站%' "
        "OR COALESCE(name,'') LIKE '%停機坪%' "
        "OR COALESCE(name,'') LIKE '%機場%'"
        ") AND COALESCE(is_context_polygon, 0) = 0"
    )
    stats.append(
        (
            "TRANSPORT_NAME",
            apply_rule(
                conn,
                4,
                "transport_land",
                "TRANSPORT_NAME",
                "name_fallback",
                "medium",
                transport_name_cond,
            ),
        )
    )

    water_name_cond = (
        "("
        "LOWER(COALESCE(name,'')) LIKE '%harbour%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%bay%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%channel%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%strait%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%reservoir%' "
        "OR COALESCE(name,'') LIKE '%港%' "
        "OR COALESCE(name,'') LIKE '%灣%' "
        "OR COALESCE(name,'') LIKE '%海峽%' "
        "OR COALESCE(name,'') LIKE '%水庫%'"
        ") AND COALESCE(is_context_polygon, 0) = 0"
    )
    stats.append(
        (
            "WATER_NAME",
            apply_rule(
                conn,
                7,
                "water_body",
                "WATER_NAME",
                "name_fallback",
                "low",
                water_name_cond,
            ),
        )
    )

    mountain_name_cond = (
        "("
        "LOWER(COALESCE(name,'')) LIKE '%island%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%islet%' "
        "OR LOWER(COALESCE(name,'')) LIKE '%archipelago%' "
        "OR COALESCE(name,'') LIKE '%島%' "
        "OR COALESCE(name,'') LIKE '%洲%' "
        "OR COALESCE(place,'') IN ('island','islet','archipelago','peninsula')"
        ") AND COALESCE(is_context_polygon, 0) = 0"
    )
    stats.append(
        (
            "MOUNTAIN_NAME_PLACE",
            apply_rule(
                conn,
                8,
                "mountainous_land",
                "MOUNTAIN_NAME_PLACE",
                "context_fill",
                "low",
                mountain_name_cond,
            ),
        )
    )

    # 人工确认特例：fid=64864 强制归类为 water_body(LUM_ID=7)
    stats.append(
        (
            "MANUAL_FID_64864_WATER",
            apply_force_rule(
                conn,
                7,
                "water_body",
                "MANUAL_FID_64864_WATER",
                "manual_override",
                "high",
                "fid = 64864",
            ),
        )
    )

    return stats


def write_audit_csv(conn: sqlite3.Connection, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0]

    class_rows = conn.execute(
        f"""
        SELECT LUM_ID, LUM_NAME, COUNT(*) AS n
        FROM {TABLE_NAME}
        GROUP BY LUM_ID, LUM_NAME
        ORDER BY LUM_ID;
        """
    ).fetchall()

    rule_rows = conn.execute(
        f"""
        SELECT rule_id, rule_level, confidence, COUNT(*) AS n
        FROM {TABLE_NAME}
        GROUP BY rule_id, rule_level, confidence
        ORDER BY n DESC;
        """
    ).fetchall()

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["section", "key1", "key2", "key3", "count", "ratio"])
        for lum_id, lum_name, n in class_rows:
            ratio = (n / total) if total else 0.0
            w.writerow(["class", lum_id, lum_name, "", n, f"{ratio:.6f}"])

        for rule_id, rule_level, confidence, n in rule_rows:
            ratio = (n / total) if total else 0.0
            w.writerow(["rule", rule_id, rule_level, confidence, n, f"{ratio:.6f}"])


def export_unknown_subset(reclass_gpkg: Path, unknown_gpkg: Path, overwrite: bool) -> None:
    copy_file(reclass_gpkg, unknown_gpkg, overwrite=overwrite)
    conn = connect_gpkg(unknown_gpkg)
    try:
        conn.execute(f"DELETE FROM {TABLE_NAME} WHERE COALESCE(LUM_ID, 255) <> 255;")
        conn.commit()
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="重分类 multipolygons 并输出 LUM_ID 结果")
    parser.add_argument("--input", "-i", default=str(DEFAULT_INPUT_GPKG), help="输入 clean multipolygons gpkg")
    parser.add_argument("--output", "-o", default=str(DEFAULT_RECLASS_GPKG), help="输出重分类 gpkg")
    parser.add_argument("--unknown-output", default=str(DEFAULT_UNKNOWN_GPKG), help="输出 unknown 子集 gpkg")
    parser.add_argument("--audit-csv", default=str(DEFAULT_AUDIT_CSV), help="输出审计统计 csv")
    parser.add_argument(
        "--hierarchy-mode",
        choices=["bbox_exact", "bbox"],
        default=DEFAULT_HIERARCHY_MODE,
        help="层级判定方式：bbox_exact=候选bbox+精确contains，bbox=仅包络框",
    )
    parser.add_argument(
        "--exact-parent-limit",
        type=int,
        default=DEFAULT_EXACT_PARENT_LIMIT,
        help="仅调试用：限制精确contains处理的候选父面数量（0=全部）",
    )
    parser.add_argument("--no-unknown-output", action="store_true", help="不导出 unknown 子集 gpkg")
    parser.add_argument("--no-audit-csv", action="store_true", help="不输出审计 csv")
    parser.add_argument("--no-overwrite", action="store_true", help="若输出存在则报错，不覆盖")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    input_gpkg = to_abs_path(args.input)
    output_gpkg = to_abs_path(args.output)
    unknown_gpkg = to_abs_path(args.unknown_output)
    audit_csv = to_abs_path(args.audit_csv)
    overwrite = not args.no_overwrite

    if not input_gpkg.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_gpkg}")
    if args.exact_parent_limit < 0:
        raise ValueError("--exact-parent-limit 不能小于 0")

    copy_file(input_gpkg, output_gpkg, overwrite=overwrite)

    conn = connect_gpkg(output_gpkg)
    try:
        ensure_columns(conn)
        init_classification_fields(conn)
        bbox_non_leaf, exact_refined_to_leaf, exact_processed = compute_hierarchy_flags(
            conn,
            hierarchy_mode=args.hierarchy_mode,
            exact_parent_limit=args.exact_parent_limit,
        )
        stats = run_rules(conn)
        strong_non_leaf_allowed = mark_non_leaf_participation(conn)
        hierarchy_excluded = apply_hierarchy_gate(conn)
        conn.commit()

        total = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME};").fetchone()[0]
        unknown = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE COALESCE(LUM_ID,255)=255;").fetchone()[0]
        leaf_count = conn.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE COALESCE(is_leaf_polygon,1)=1;").fetchone()[0]
        non_leaf_count = total - leaf_count
        non_leaf_allowed = conn.execute(
            f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE COALESCE(is_leaf_polygon,1)=0 AND COALESCE(is_non_leaf_allowed,0)=1;"
        ).fetchone()[0]

        print(f"总要素数: {total}")
        print(f"叶子面数: {leaf_count}")
        print(f"非叶子面数: {non_leaf_count}")
        print(f"bbox 非叶子候选数: {bbox_non_leaf}")
        if args.hierarchy_mode == "bbox_exact":
            print(
                f"exact contains 已处理父面数: {exact_processed}，"
                f"由候选回退为叶子面数: {exact_refined_to_leaf}"
            )
        print(f"强语义非叶子面准入数: {non_leaf_allowed} (本轮新增标记 {strong_non_leaf_allowed})")
        print(f"层级门控排除数(置为unknown): {hierarchy_excluded}")
        print(f"未分类要素数(LUM_ID=255): {unknown}")
        print("规则命中统计(增量):")
        for rule_name, n in stats:
            print(f"  - {rule_name:<24} {n}")

        if not args.no_audit_csv:
            write_audit_csv(conn, audit_csv)
            print(f"已输出审计 CSV: {audit_csv}")

    finally:
        conn.close()

    if not args.no_unknown_output:
        export_unknown_subset(output_gpkg, unknown_gpkg, overwrite=overwrite)
        print(f"已输出 unknown 子集: {unknown_gpkg}")

    print(f"已输出重分类 GPKG: {output_gpkg}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

