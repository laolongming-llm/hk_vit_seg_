"""Mosaic exported prediction tiles with confidence-based fusion (stage B)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.windows import Window

from train_lib import setup_logger, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[3]

LAYER_SUFFIX_MAP: Dict[str, str] = {
    "pred_all_pixels": "_pred_all_pixels.tif",
    "pred_masked": "_pred_masked.tif",
}


@dataclass(frozen=True)
class TilePair:
    split: str
    tile_id: str
    pred_path: Path
    conf_path: Path
    bounds: Tuple[float, float, float, float]  # left, bottom, right, top
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mosaic prediction tiles with max-confidence fusion.")
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Run directory that contains predictions/<split>/ and logs/metrics dirs.",
    )
    parser.add_argument(
        "--splits",
        default="train,val,test",
        help="Comma-separated splits to include, e.g. train,val,test.",
    )
    parser.add_argument(
        "--layer",
        default="pred_all_pixels",
        choices=["pred_all_pixels", "pred_masked"],
        help="Prediction layer to mosaic.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=["per_split", "all_splits", "both"],
        help="Mosaic per split only, all splits only, or both.",
    )
    parser.add_argument(
        "--fusion",
        default="max_conf",
        choices=["max_conf"],
        help="Fusion strategy. Stage B uses max confidence only.",
    )
    parser.add_argument(
        "--max-tiles-per-split",
        type=int,
        default=0,
        help="Optional debug cap per split (0 means all tiles).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output mosaic directory. Empty means <run-dir>/mosaics.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def parse_splits(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def resolve_run_dir(run_dir_arg: str) -> Path:
    p = Path(run_dir_arg).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def infer_tile_id(path: Path, suffix: str) -> str:
    name = path.name
    if not name.endswith(suffix):
        raise ValueError(f"Unexpected file name suffix: {path.name} (expect {suffix})")
    return name[: -len(suffix)]


def normalize_crs_for_qgis(crs_obj: Any) -> Any:
    if crs_obj is None:
        return None
    try:
        epsg = crs_obj.to_epsg()
    except Exception:
        epsg = None
    if epsg:
        return CRS.from_epsg(int(epsg))

    crs_text = str(crs_obj).lower()
    wkt_text = ""
    try:
        wkt_text = str(crs_obj.to_wkt()).lower()
    except Exception:
        wkt_text = ""
    if ("hong kong 1980 grid system" in crs_text) or ("hong kong 1980 grid system" in wkt_text):
        return CRS.from_epsg(2326)
    return crs_obj


def collect_tile_pairs(
    run_dir: Path,
    splits: Sequence[str],
    layer: str,
    max_tiles_per_split: int,
    logger: Any,
) -> Dict[str, List[TilePair]]:
    pred_root = run_dir / "predictions"
    suffix = LAYER_SUFFIX_MAP[layer]
    out: Dict[str, List[TilePair]] = {}

    for split in splits:
        split_dir = pred_root / split
        if not split_dir.exists():
            logger.warning("Skip split=%s: directory missing: %s", split, split_dir)
            out[split] = []
            continue

        pred_paths = sorted(split_dir.glob(f"*{suffix}"))
        if max_tiles_per_split > 0:
            pred_paths = pred_paths[:max_tiles_per_split]

        pairs: List[TilePair] = []
        for pred_path in pred_paths:
            tile_id = infer_tile_id(pred_path, suffix)
            conf_path = split_dir / f"{tile_id}_confidence.tif"
            if not conf_path.exists():
                raise FileNotFoundError(f"Confidence tile not found for {pred_path.name}: {conf_path}")

            with rasterio.open(pred_path) as ds:
                bounds = ds.bounds
                pairs.append(
                    TilePair(
                        split=split,
                        tile_id=tile_id,
                        pred_path=pred_path,
                        conf_path=conf_path,
                        bounds=(float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)),
                        width=int(ds.width),
                        height=int(ds.height),
                    )
                )

        out[split] = pairs
        logger.info("Collected split=%s tiles=%d", split, len(pairs))
    return out


def validate_profile_consistency(pairs: Sequence[TilePair]) -> Dict[str, Any]:
    if not pairs:
        raise ValueError("No tile pairs to mosaic.")

    with rasterio.open(pairs[0].pred_path) as ds0:
        normalized_crs = normalize_crs_for_qgis(ds0.crs)
        try:
            pred_colormap = ds0.colormap(1)
        except ValueError:
            pred_colormap = None
        pred_legend = ds0.tags(1).get("legend", "")
        base = {
            "crs": normalized_crs,
            "res": ds0.res,
            "dtype": ds0.dtypes[0],
            "count": ds0.count,
            "colormap": pred_colormap,
            "legend": pred_legend,
        }

    for p in pairs[1:]:
        with rasterio.open(p.pred_path) as ds:
            if normalize_crs_for_qgis(ds.crs) != base["crs"]:
                raise ValueError(f"CRS mismatch: {p.pred_path}")
            if tuple(ds.res) != tuple(base["res"]):
                raise ValueError(f"Resolution mismatch: {p.pred_path}")
            if ds.dtypes[0] != base["dtype"]:
                raise ValueError(f"Dtype mismatch: {p.pred_path}")
            if ds.count != base["count"]:
                raise ValueError(f"Band count mismatch: {p.pred_path}")
    return base


def compute_mosaic_grid(pairs: Sequence[TilePair], res_x: float, res_y: float) -> Dict[str, Any]:
    left = min(x.bounds[0] for x in pairs)
    bottom = min(x.bounds[1] for x in pairs)
    right = max(x.bounds[2] for x in pairs)
    top = max(x.bounds[3] for x in pairs)

    width = int(round((right - left) / res_x))
    height = int(round((top - bottom) / res_y))
    transform = rasterio.transform.from_origin(left, top, res_x, res_y)
    return {
        "left": left,
        "bottom": bottom,
        "right": right,
        "top": top,
        "width": width,
        "height": height,
        "transform": transform,
    }


def make_window(bounds: Tuple[float, float, float, float], grid: Dict[str, Any], res_x: float, res_y: float) -> Window:
    left, _bottom, _right, top = bounds
    col_off = int(round((left - float(grid["left"])) / res_x))
    row_off = int(round((float(grid["top"]) - top) / res_y))
    # width/height are resolved from tile bounds to minimize floating drift
    win_width = int(round((bounds[2] - bounds[0]) / res_x))
    win_height = int(round((bounds[3] - bounds[1]) / res_y))
    return Window(col_off=col_off, row_off=row_off, width=win_width, height=win_height)


def mosaic_with_max_conf(
    pairs: Sequence[TilePair],
    out_pred_path: Path,
    out_conf_path: Path,
    overwrite: bool,
    logger: Any,
) -> Dict[str, Any]:
    base = validate_profile_consistency(pairs)
    res_x, res_y = [float(x) for x in base["res"]]
    grid = compute_mosaic_grid(pairs, res_x=res_x, res_y=res_y)

    if out_pred_path.exists() and (not overwrite):
        raise FileExistsError(f"Output exists (pred): {out_pred_path}")
    if out_conf_path.exists() and (not overwrite):
        raise FileExistsError(f"Output exists (conf): {out_conf_path}")

    out_pred_path.parent.mkdir(parents=True, exist_ok=True)
    out_conf_path.parent.mkdir(parents=True, exist_ok=True)

    # For uint8 class map, 255 is reserved as unknown/empty in this project.
    pred_nodata = 255
    conf_nodata = -1.0

    pred_profile = {
        "driver": "GTiff",
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "count": 1,
        "dtype": "uint8",
        "crs": base["crs"],
        "transform": grid["transform"],
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": float(pred_nodata),
    }
    conf_profile = {
        "driver": "GTiff",
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "count": 1,
        "dtype": "float32",
        "crs": base["crs"],
        "transform": grid["transform"],
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": float(conf_nodata),
    }

    logger.info(
        "Mosaic grid: width=%d height=%d bounds=(%.3f, %.3f, %.3f, %.3f)",
        int(grid["width"]),
        int(grid["height"]),
        float(grid["left"]),
        float(grid["bottom"]),
        float(grid["right"]),
        float(grid["top"]),
    )

    update_total = 0
    pair_total = len(pairs)
    canvas_h = int(grid["height"])
    canvas_w = int(grid["width"])
    pred_canvas = np.full((canvas_h, canvas_w), fill_value=np.uint8(pred_nodata), dtype=np.uint8)
    conf_canvas = np.full((canvas_h, canvas_w), fill_value=np.float32(conf_nodata), dtype=np.float32)

    for idx, item in enumerate(pairs, start=1):
        with rasterio.open(item.pred_path) as pred_src, rasterio.open(item.conf_path) as conf_src:
            pred_arr = pred_src.read(1)
            conf_arr = conf_src.read(1).astype(np.float32)

        win = make_window(item.bounds, grid=grid, res_x=res_x, res_y=res_y)

        # Window can be off by 1 due to float accumulation; reconcile with real tile shape.
        expected_h, expected_w = pred_arr.shape
        win = Window(col_off=int(win.col_off), row_off=int(win.row_off), width=expected_w, height=expected_h)

        r0 = int(win.row_off)
        r1 = r0 + int(win.height)
        c0 = int(win.col_off)
        c1 = c0 + int(win.width)

        cur_conf = conf_canvas[r0:r1, c0:c1]
        cur_pred = pred_canvas[r0:r1, c0:c1]

        # Stage B rule: choose label from tile that has larger confidence.
        # Tie uses >= to make later tiles override earlier ones deterministically.
        update_mask = conf_arr >= cur_conf
        if np.any(update_mask):
            cur_conf[update_mask] = conf_arr[update_mask]
            cur_pred[update_mask] = pred_arr[update_mask]
            update_total += int(np.count_nonzero(update_mask))

        if idx % 200 == 0 or idx == pair_total:
            logger.info("Mosaic progress: %d/%d tiles processed", idx, pair_total)

    with rasterio.open(out_pred_path, "w", **pred_profile) as pred_dst:
        pred_dst.write(pred_canvas, 1)
        pred_dst.set_band_description(1, "LUM_ID_PRED_MOSAIC")
        if base.get("legend"):
            pred_dst.update_tags(1, legend=str(base["legend"]))
        if base.get("colormap"):
            pred_dst.write_colormap(1, base["colormap"])
    with rasterio.open(out_conf_path, "w", **conf_profile) as conf_dst:
        conf_dst.write(conf_canvas, 1)
        conf_dst.set_band_description(1, "PRED_CONFIDENCE_MOSAIC_MAX")

    return {
        "tile_count": int(pair_total),
        "update_pixels": int(update_total),
        "width": int(grid["width"]),
        "height": int(grid["height"]),
        "left": float(grid["left"]),
        "bottom": float(grid["bottom"]),
        "right": float(grid["right"]),
        "top": float(grid["top"]),
        "pred_output": str(out_pred_path),
        "conf_output": str(out_conf_path),
    }


def run_scope(
    scope_name: str,
    pairs: Sequence[TilePair],
    output_dir: Path,
    layer: str,
    overwrite: bool,
    logger: Any,
) -> Dict[str, Any]:
    pred_name = f"mosaic_{scope_name}_{layer}_max_conf.tif"
    conf_name = f"mosaic_{scope_name}_{layer}_max_conf_confidence.tif"
    pred_path = output_dir / pred_name
    conf_path = output_dir / conf_name

    logger.info("Start mosaic scope=%s tiles=%d layer=%s", scope_name, len(pairs), layer)
    result = mosaic_with_max_conf(
        pairs=pairs,
        out_pred_path=pred_path,
        out_conf_path=conf_path,
        overwrite=overwrite,
        logger=logger,
    )
    result["scope"] = scope_name
    result["layer"] = layer
    result["fusion"] = "max_conf"
    return result


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    output_dir = (
        (Path(args.output_dir).expanduser().resolve())
        if args.output_dir.strip()
        else (run_dir / "mosaics").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = run_dir / "logs"
    metrics_dir = run_dir / "metrics"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(logs_dir / "mosaic_predictions.log", logger_name="formal_mosaic_predictions")

    splits = parse_splits(args.splits)
    logger.info(
        "Mosaic args: run_dir=%s splits=%s layer=%s mode=%s fusion=%s max_tiles_per_split=%d output_dir=%s",
        run_dir,
        splits,
        args.layer,
        args.mode,
        args.fusion,
        int(args.max_tiles_per_split),
        output_dir,
    )

    by_split = collect_tile_pairs(
        run_dir=run_dir,
        splits=splits,
        layer=args.layer,
        max_tiles_per_split=int(args.max_tiles_per_split),
        logger=logger,
    )

    results: List[Dict[str, Any]] = []
    if args.mode in {"per_split", "both"}:
        for split in splits:
            pairs = by_split.get(split, [])
            if not pairs:
                logger.warning("Skip per_split mosaic: split=%s has no tiles.", split)
                continue
            results.append(
                run_scope(
                    scope_name=split,
                    pairs=pairs,
                    output_dir=output_dir,
                    layer=args.layer,
                    overwrite=bool(args.overwrite),
                    logger=logger,
                )
            )

    if args.mode in {"all_splits", "both"}:
        merged: List[TilePair] = []
        for split in splits:
            merged.extend(by_split.get(split, []))
        if merged:
            # Deterministic ordering for tie behavior.
            merged.sort(key=lambda x: (x.split, x.tile_id))
            results.append(
                run_scope(
                    scope_name="all_splits",
                    pairs=merged,
                    output_dir=output_dir,
                    layer=args.layer,
                    overwrite=bool(args.overwrite),
                    logger=logger,
                )
            )
        else:
            logger.warning("Skip all_splits mosaic: no tiles collected.")

    if not results:
        logger.warning("No mosaic outputs generated.")
        return

    report_path = metrics_dir / "mosaic_report.csv"
    write_csv(
        path=report_path,
        rows=results,
        fieldnames=[
            "scope",
            "layer",
            "fusion",
            "tile_count",
            "update_pixels",
            "width",
            "height",
            "left",
            "bottom",
            "right",
            "top",
            "pred_output",
            "conf_output",
        ],
    )
    logger.info("Mosaic report saved: %s", report_path)
    logger.info("Prediction mosaic completed successfully.")


if __name__ == "__main__":
    main()
