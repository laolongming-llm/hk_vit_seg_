#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：06_export_vit_dataset_tiles.py

功能：
1) 读取 dry-run 生成的 tiles_manifest.csv；
2) 按 split 将影像/标签窗口导出为实体 patch；
3) 输出 export_report.csv，记录导出统计与来源信息。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass
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

DEFAULT_TILES_MANIFEST: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v1" / "manifests_dryrun_tuned_b256" / "tiles_manifest.csv"
)
DEFAULT_MANIFESTS_DIR: Final[Path] = (
    PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v1" / "manifests_dryrun_tuned_b256"
)
DEFAULT_OUTPUT_ROOT: Final[Path] = PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v1"

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

GTIFF_CREATION_OPTIONS: Final[list[str]] = [
    "COMPRESS=LZW",
    "TILED=YES",
    "BLOCKXSIZE=512",
    "BLOCKYSIZE=512",
    "SPARSE_OK=TRUE",
    "NUM_THREADS=ALL_CPUS",
    "BIGTIFF=IF_SAFER",
]

VALID_SPLITS: Final[set[str]] = {"train", "val", "test_in_domain", "test_eco_holdout"}


@dataclass(frozen=True)
class RasterPair:
    pair_name: str
    imagery_path: Path
    label_path: Path


def require_deps() -> None:
    if gdal is not None and osr is not None:
        return
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


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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


def log(msg: str) -> None:
    print(msg, flush=True)


def format_eta(seconds: float) -> str:
    if seconds < 0 or not (seconds < float("inf")):
        return "--:--:--"
    sec = int(seconds)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


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


def validate_pair_alignment(pair: RasterPair) -> tuple[int, int]:
    img_ds = gdal.Open(str(pair.imagery_path), gdal.GA_ReadOnly)
    if img_ds is None:
        raise RuntimeError(f"无法打开影像：{pair.imagery_path}")
    lab_ds = gdal.Open(str(pair.label_path), gdal.GA_ReadOnly)
    if lab_ds is None:
        img_ds = None
        raise RuntimeError(f"无法打开标签：{pair.label_path}")

    if (img_ds.RasterYSize, img_ds.RasterXSize) != (lab_ds.RasterYSize, lab_ds.RasterXSize):
        raise RuntimeError(f"{pair.pair_name} 影像/标签 shape 不一致。")
    if not spatial_ref_equal(img_ds.GetProjection(), lab_ds.GetProjection()):
        raise RuntimeError(f"{pair.pair_name} 影像/标签 CRS 不一致。")

    img_gt = img_ds.GetGeoTransform()
    lab_gt = lab_ds.GetGeoTransform()
    for i, (a, b) in enumerate(zip(img_gt, lab_gt)):
        if not approx_equal(float(a), float(b), tol=1e-8):
            raise RuntimeError(f"{pair.pair_name} 影像/标签 transform 不一致（index={i}）。")

    ysize = lab_ds.RasterYSize
    xsize = lab_ds.RasterXSize
    img_ds = None
    lab_ds = None
    return ysize, xsize


def read_tiles_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"tiles_manifest 为空：{path}")
    required = {"tile_id", "pair_name", "final_split", "row_off", "col_off", "patch_size"}
    headers = set(rows[0].keys())
    missing = required - headers
    if missing:
        raise RuntimeError(f"tiles_manifest 缺少必要字段：{sorted(missing)}")
    return rows


def export_patch(
    src_path: Path,
    dst_path: Path,
    row_off: int,
    col_off: int,
    patch_size: int,
    overwrite: bool,
) -> None:
    ensure_dir(dst_path.parent)
    if dst_path.exists():
        if not overwrite:
            return
        dst_path.unlink()
    ds = gdal.Translate(
        str(dst_path),
        str(src_path),
        options=gdal.TranslateOptions(
            format="GTiff",
            srcWin=[col_off, row_off, patch_size, patch_size],
            creationOptions=GTIFF_CREATION_OPTIONS,
        ),
    )
    if ds is None:
        raise RuntimeError(f"导出失败：{dst_path}")
    ds = None


def write_export_report(
    out_csv: Path,
    counters: Counter[str],
    total_rows: int,
    exported_by_split: Counter[str],
    tiles_manifest_path: Path,
    output_root: Path,
) -> None:
    ensure_dir(out_csv.parent)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["tiles_manifest", str(tiles_manifest_path)])
        w.writerow(["output_root", str(output_root)])
        w.writerow(["total_rows_in_manifest", total_rows])
        w.writerow(["exported", counters["exported"]])
        w.writerow(["skipped_exists", counters["skipped_exists"]])
        w.writerow(["failed", counters["failed"]])
        for split in sorted(VALID_SPLITS):
            w.writerow([f"exported_{split}", exported_by_split.get(split, 0)])


def copy_support_manifests(src_dir: Path, dst_dir: Path) -> None:
    ensure_dir(dst_dir)
    for name in ["split_summary.csv", "eco_split_geometry.csv", "eco_split_class_stats.csv"]:
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="按 tiles_manifest 导出 ViT 数据集 patch")
    parser.add_argument("--config", default="", help="YAML 配置文件路径（可选）")
    parser.add_argument("--tiles-manifest", default=str(DEFAULT_TILES_MANIFEST), help="dry-run 生成的 tiles_manifest.csv")
    parser.add_argument("--manifests-dir", default=str(DEFAULT_MANIFESTS_DIR), help="dry-run manifests 目录（用于拷贝辅助清单）")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="数据集输出根目录")

    parser.add_argument("--imagery-11block", default=str(DEFAULT_IMAGERY_11BLOCK), help="11block 影像路径")
    parser.add_argument("--label-11block", default=str(DEFAULT_LABEL_11BLOCK), help="11block 标签路径")
    parser.add_argument("--imagery-2swd", default=str(DEFAULT_IMAGERY_2SWD), help="2swd 影像路径")
    parser.add_argument("--label-2swd", default=str(DEFAULT_LABEL_2SWD), help="2swd 标签路径")

    parser.add_argument("--max-tiles", type=int, default=0, help="仅导出前 N 个 tile（0 表示全部）")
    parser.add_argument("--copy-manifests", action="store_true", help="将 split_summary/eco_* 从 dry-run 目录拷贝到输出 manifests")
    parser.add_argument("--no-overwrite", action="store_true", help="输出 patch 已存在则跳过")
    parser.add_argument("--progress-every", type=int, default=50, help="每处理 N 个 tile 输出一次进度")
    parser.add_argument("--progress-seconds", type=float, default=8.0, help="每隔 N 秒至少输出一次进度")
    parser.add_argument("--show-gdal-warnings", action="store_true", help="显示 GDAL/PROJ 告警（默认静默）")
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
        log(f"[config] loaded: {to_abs_path(pre_args.config)}")

    tiles_manifest = to_abs_path(args.tiles_manifest)
    manifests_dir = to_abs_path(args.manifests_dir)
    output_root = to_abs_path(args.output_root)
    overwrite = not args.no_overwrite
    max_tiles = int(args.max_tiles)

    if not tiles_manifest.exists():
        raise FileNotFoundError(f"tiles_manifest 不存在：{tiles_manifest}")
    if args.progress_every <= 0:
        raise ValueError("--progress-every 必须大于 0")
    if args.progress_seconds <= 0:
        raise ValueError("--progress-seconds 必须大于 0")

    if not args.show_gdal_warnings:
        # 默认静默 GDAL/PROJ 告警，避免淹没实时进度信息
        gdal.PushErrorHandler("CPLQuietErrorHandler")

    pair_map = {
        "11block": RasterPair(
            pair_name="11block",
            imagery_path=to_abs_path(args.imagery_11block),
            label_path=to_abs_path(args.label_11block),
        ),
        "2swd": RasterPair(
            pair_name="2swd",
            imagery_path=to_abs_path(args.imagery_2swd),
            label_path=to_abs_path(args.label_2swd),
        ),
    }
    for pair in pair_map.values():
        if not pair.imagery_path.exists() or not pair.label_path.exists():
            raise FileNotFoundError(f"{pair.pair_name} 输入不存在：{pair.imagery_path} / {pair.label_path}")
        validate_pair_alignment(pair)

    rows = read_tiles_manifest(tiles_manifest)
    total_rows = len(rows)
    if max_tiles > 0:
        rows = rows[:max_tiles]
    run_total = len(rows)
    log(f"[start] manifest rows={total_rows}, rows_to_process={run_total}, output_root={output_root}")

    images_root = output_root / "images"
    labels_root = output_root / "labels"
    manifests_out = output_root / "manifests"
    ensure_dir(images_root)
    ensure_dir(labels_root)
    ensure_dir(manifests_out)

    # 固化本次使用的 tiles manifest
    shutil.copy2(tiles_manifest, manifests_out / "tiles_manifest.csv")
    if args.copy_manifests:
        copy_support_manifests(manifests_dir, manifests_out)

    counters: Counter[str] = Counter()
    exported_by_split: Counter[str] = Counter()
    start_ts = time.monotonic()
    last_report_ts = start_ts
    last_report_idx = 0

    def maybe_report(processed: int, force: bool = False) -> None:
        nonlocal last_report_ts, last_report_idx
        if run_total <= 0 or processed <= 0:
            return
        now = time.monotonic()
        due_count = (processed - last_report_idx) >= args.progress_every
        due_time = (now - last_report_ts) >= args.progress_seconds
        if not (force or due_count or due_time):
            return

        elapsed = max(now - start_ts, 1e-9)
        rate = processed / elapsed
        remain = max(run_total - processed, 0)
        eta = remain / rate if rate > 0 else float("inf")
        pct = (processed / run_total) * 100.0
        log(
            "[progress] "
            f"{processed}/{run_total} ({pct:.2f}%) | "
            f"exported={counters['exported']} skipped={counters['skipped_exists']} failed={counters['failed']} | "
            f"rate={rate:.2f} tile/s eta={format_eta(eta)}"
        )
        last_report_ts = now
        last_report_idx = processed

    for idx, row in enumerate(rows, start=1):
        pair_name = row["pair_name"].strip()
        final_split = row["final_split"].strip()
        tile_id = row["tile_id"].strip()

        if pair_name not in pair_map:
            counters["failed"] += 1
            log(f"[WARN] 未知 pair_name: {pair_name} (tile_id={tile_id})")
            maybe_report(idx)
            continue
        if final_split not in VALID_SPLITS:
            counters["failed"] += 1
            log(f"[WARN] 未知 split: {final_split} (tile_id={tile_id})")
            maybe_report(idx)
            continue

        try:
            row_off = int(row["row_off"])
            col_off = int(row["col_off"])
            patch_size = int(row["patch_size"])
        except Exception:
            counters["failed"] += 1
            log(f"[WARN] 坐标字段解析失败: tile_id={tile_id}")
            maybe_report(idx)
            continue

        pair = pair_map[pair_name]
        img_out = images_root / final_split / f"{tile_id}.tif"
        lbl_out = labels_root / final_split / f"{tile_id}.tif"

        if (img_out.exists() or lbl_out.exists()) and not overwrite:
            counters["skipped_exists"] += 1
            maybe_report(idx)
            continue

        try:
            export_patch(
                src_path=pair.imagery_path,
                dst_path=img_out,
                row_off=row_off,
                col_off=col_off,
                patch_size=patch_size,
                overwrite=overwrite,
            )
            export_patch(
                src_path=pair.label_path,
                dst_path=lbl_out,
                row_off=row_off,
                col_off=col_off,
                patch_size=patch_size,
                overwrite=overwrite,
            )
        except Exception as exc:
            counters["failed"] += 1
            log(f"[WARN] 导出失败 tile_id={tile_id}: {exc}")
            maybe_report(idx)
            continue

        counters["exported"] += 1
        exported_by_split[final_split] += 1
        maybe_report(idx)

    maybe_report(run_total, force=True)

    export_report = manifests_out / "export_report.csv"
    write_export_report(
        out_csv=export_report,
        counters=counters,
        total_rows=total_rows,
        exported_by_split=exported_by_split,
        tiles_manifest_path=tiles_manifest,
        output_root=output_root,
    )

    log("[done] 导出完成")
    log(f"  - total rows in manifest: {total_rows}")
    log(f"  - rows processed: {len(rows)}")
    log(f"  - exported: {counters['exported']}")
    log(f"  - skipped_exists: {counters['skipped_exists']}")
    log(f"  - failed: {counters['failed']}")
    for split in sorted(VALID_SPLITS):
        log(f"  - exported_{split}: {exported_by_split.get(split, 0)}")
    log(f"  - export report: {export_report}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)

