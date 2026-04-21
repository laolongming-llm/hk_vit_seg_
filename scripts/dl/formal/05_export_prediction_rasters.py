"""Formal prediction raster export script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from train_lib import (
    SegmentationTileDataset,
    build_model_from_config,
    build_tile_paths,
    ensure_run_dirs,
    get_run_paths,
    load_checkpoint,
    load_config,
    read_manifest,
    resolve_class_lum_ids,
    resolve_project_path,
    setup_logger,
    validate_manifest_files,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data_prep.lumid_style import LUM_ID_STYLE, write_style_sidecars_for_raster


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export formal prediction rasters with LUM_ID color mapping.")
    parser.add_argument("--config", required=True, help="Path to formal config YAML.")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint.")
    parser.add_argument(
        "--splits",
        default="",
        help="Comma-separated split names. Empty means config.evaluation.splits_to_eval.",
    )
    parser.add_argument(
        "--max-tiles-per-split",
        type=int,
        default=0,
        help="Optional debug cap per split (0 means export all).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tif files.")
    parser.add_argument(
        "--skip-sidecars",
        action="store_true",
        help="Skip .qml/.clr generation (default is write sidecars for prediction rasters).",
    )
    return parser.parse_args()


def resolve_splits(cfg: Dict[str, Any], user_splits: str) -> List[str]:
    if user_splits.strip():
        return [item.strip() for item in user_splits.split(",") if item.strip()]
    return list(cfg.get("evaluation", {}).get("splits_to_eval", ["val", "test_in_domain", "test_eco_holdout"]))


def build_eval_dataloader(dataset: SegmentationTileDataset, loader_cfg: Dict[str, Any]) -> DataLoader:
    num_workers = int(loader_cfg.get("num_workers", 0))
    eval_batch_size = int(loader_cfg.get("eval_batch_size", loader_cfg.get("batch_size", 2)))
    return DataLoader(
        dataset=dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=False,
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0,
    )


def build_lumid_colormap() -> Dict[int, tuple[int, int, int, int]]:
    colormap: Dict[int, tuple[int, int, int, int]] = {idx: (0, 0, 0, 0) for idx in range(256)}
    for value, _, _, _, (r, g, b) in LUM_ID_STYLE:
        colormap[int(value)] = (int(r), int(g), int(b), 255)
    return colormap


def _write_prediction_raster(
    out_path: Path,
    data: np.ndarray,
    profile: Dict[str, Any],
    colormap: Dict[int, tuple[int, int, int, int]],
    band_desc: str,
    legend: str,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.set_band_description(1, band_desc)
        dst.update_tags(1, legend=legend)
        dst.write_colormap(1, colormap)


def _write_confidence_raster(out_path: Path, confidence: np.ndarray, profile: Dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(confidence.astype(np.float32), 1)
        dst.set_band_description(1, "PRED_CONFIDENCE_MAX_SOFTMAX")
        dst.update_tags(
            1,
            legend="Per-pixel max-softmax confidence (0~1, higher means more confident).",
        )


def _maybe_write_sidecars(raster_path: Path, write_sidecars: bool, overwrite: bool) -> None:
    if not write_sidecars:
        return
    qml_path = raster_path.with_suffix(".qml")
    clr_path = raster_path.with_suffix(".clr")
    if overwrite or (not qml_path.exists()) or (not clr_path.exists()):
        write_style_sidecars_for_raster(raster_path)


def _map_pred_class_to_lumid(
    pred_class_map: np.ndarray,
    class_lum_ids: List[int],
    ignore_lum_ids: List[int],
    ignore_index: int,
) -> np.ndarray:
    if pred_class_map.size == 0:
        return pred_class_map.astype(np.uint8)

    max_pred = int(pred_class_map.max())
    if max_pred >= len(class_lum_ids):
        raise ValueError(
            "Prediction class index out of mapping range: "
            f"max_pred={max_pred}, len(class_lum_ids)={len(class_lum_ids)}"
        )

    class_to_lum = np.asarray(class_lum_ids, dtype=np.uint16)
    pred_lumid = class_to_lum[pred_class_map.astype(np.int64)]

    if ignore_lum_ids:
        ignore_lum_np = np.asarray(ignore_lum_ids, dtype=np.uint16)
        pred_lumid[np.isin(pred_lumid, ignore_lum_np)] = np.uint16(ignore_index)

    return pred_lumid.astype(np.uint8)


def export_one_prediction(
    out_dir: Path,
    tile_id: str,
    pred_class_map: np.ndarray,
    confidence_map: np.ndarray,
    label_path: Path,
    colormap: Dict[int, tuple[int, int, int, int]],
    overwrite: bool,
    write_sidecars: bool,
    ignore_index: int,
    class_lum_ids: List[int],
    ignore_lum_ids: List[int],
) -> Dict[str, Any]:
    with rasterio.open(label_path) as label_src:
        label_arr = label_src.read(1)
        base_profile = label_src.profile.copy()

    pred_lumid_all = _map_pred_class_to_lumid(
        pred_class_map=pred_class_map,
        class_lum_ids=class_lum_ids,
        ignore_lum_ids=ignore_lum_ids,
        ignore_index=ignore_index,
    )

    pred_lumid_masked = pred_lumid_all.copy()
    pred_lumid_masked[label_arr == ignore_index] = np.uint8(ignore_index)

    active_lum_text = ",".join(str(x) for x in class_lum_ids)
    ignored_lum_text = ",".join(str(x) for x in ignore_lum_ids) if ignore_lum_ids else "none"

    masked_path = out_dir / f"{tile_id}_pred_masked.tif"
    all_pixels_path = out_dir / f"{tile_id}_pred_all_pixels.tif"
    confidence_path = out_dir / f"{tile_id}_confidence.tif"

    profile_masked = base_profile.copy()
    profile_masked.update(
        count=1,
        dtype="uint8",
        nodata=float(ignore_index),
        compress=profile_masked.get("compress") or "lzw",
    )

    profile_all = base_profile.copy()
    profile_all.update(
        count=1,
        dtype="uint8",
        nodata=None,
        compress=profile_all.get("compress") or "lzw",
    )

    profile_conf = base_profile.copy()
    profile_conf.update(
        count=1,
        dtype="float32",
        nodata=None,
        compress=profile_conf.get("compress") or "lzw",
    )

    masked_written = 0
    all_pixels_written = 0
    confidence_written = 0

    if overwrite or (not masked_path.exists()):
        _write_prediction_raster(
            out_path=masked_path,
            data=pred_lumid_masked,
            profile=profile_masked,
            colormap=colormap,
            band_desc="LUM_ID_PRED_MASKED",
            legend=(
                f"LUM_ID prediction with label-mask ({ignore_index}=unknown). "
                f"active_lum_ids=[{active_lum_text}] ignored_lum_ids=[{ignored_lum_text}]"
            ),
        )
        _maybe_write_sidecars(masked_path, write_sidecars=write_sidecars, overwrite=overwrite)
        masked_written = 1

    if overwrite or (not all_pixels_path.exists()):
        _write_prediction_raster(
            out_path=all_pixels_path,
            data=pred_lumid_all,
            profile=profile_all,
            colormap=colormap,
            band_desc="LUM_ID_PRED_ALL_PIXELS",
            legend=(
                f"LUM_ID prediction for all pixels. active_lum_ids=[{active_lum_text}] "
                f"ignored_lum_ids=[{ignored_lum_text}] {ignore_index}=unknown"
            ),
        )
        _maybe_write_sidecars(all_pixels_path, write_sidecars=write_sidecars, overwrite=overwrite)
        all_pixels_written = 1

    if overwrite or (not confidence_path.exists()):
        _write_confidence_raster(out_path=confidence_path, confidence=confidence_map, profile=profile_conf)
        confidence_written = 1

    tile_status = "exported" if (masked_written or all_pixels_written or confidence_written) else "skipped_exists"
    return {
        "tile_status": tile_status,
        "pred_masked_written": masked_written,
        "pred_all_pixels_written": all_pixels_written,
        "confidence_written": confidence_written,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    config_path = Path(cfg["_meta"]["resolved_config_path"])

    data_cfg = cfg.get("data", {})
    loader_cfg = cfg.get("loader", {})
    class_lum_ids = resolve_class_lum_ids(data_cfg)
    num_classes = len(class_lum_ids)
    ignore_lum_ids = [int(x) for x in data_cfg.get("ignore_lum_ids", [])]
    ignore_index = int(data_cfg.get("ignore_index", 255))

    run_paths = get_run_paths(cfg, config_path)
    ensure_run_dirs(run_paths)
    logger = setup_logger(run_paths["logs_dir"] / "export_predictions.log", logger_name="formal_export_predictions")

    splits = resolve_splits(cfg, args.splits)
    logger.info("Prediction export splits: %s", splits)
    logger.info(
        "Label remap policy: class_lum_ids=%s | ignore_lum_ids=%s -> ignore_index=%d",
        class_lum_ids,
        ignore_lum_ids,
        ignore_index,
    )

    checkpoint_path = resolve_project_path(args.checkpoint, config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device=device)
    model = build_model_from_config(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    logger.info("Loaded checkpoint: %s", checkpoint_path)

    dataset_root = resolve_project_path(data_cfg["dataset_root"], config_path)
    manifest_path = resolve_project_path(data_cfg["manifest_path"], config_path)
    manifest_df = read_manifest(manifest_path)

    colormap = build_lumid_colormap()
    pred_root = run_paths["run_dir"] / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)

    summary_rows: List[Dict[str, Any]] = []
    for split_name in splits:
        split_df = manifest_df[manifest_df["final_split"] == split_name].copy().reset_index(drop=True)
        if split_df.empty:
            logger.warning("Skip split=%s: no rows in manifest.", split_name)
            summary_rows.append(
                {
                    "split": split_name,
                    "requested_tiles": 0,
                    "exported_tiles": 0,
                    "skipped_exists": 0,
                    "pred_masked_written": 0,
                    "pred_all_pixels_written": 0,
                    "confidence_written": 0,
                    "status": "empty_split",
                }
            )
            continue

        if args.max_tiles_per_split > 0:
            split_df = split_df.iloc[: args.max_tiles_per_split].copy().reset_index(drop=True)

        errors = validate_manifest_files(
            df=split_df,
            dataset_root=dataset_root,
            image_dirname=data_cfg["image_dirname"],
            label_dirname=data_cfg["label_dirname"],
            image_suffix=data_cfg["image_suffix"],
            label_suffix=data_cfg["label_suffix"],
        )
        if errors:
            preview = "\n".join(errors[:10])
            raise FileNotFoundError(f"Missing files in split={split_name} ({len(errors)}).\n{preview}")

        dataset = SegmentationTileDataset(
            manifest_df=split_df,
            dataset_root=dataset_root,
            image_dirname=data_cfg["image_dirname"],
            label_dirname=data_cfg["label_dirname"],
            image_suffix=data_cfg["image_suffix"],
            label_suffix=data_cfg["label_suffix"],
            num_classes=num_classes,
            ignore_index=ignore_index,
            ignore_lum_ids=ignore_lum_ids,
            class_lum_ids=class_lum_ids,
        )
        data_loader = build_eval_dataloader(dataset, loader_cfg=loader_cfg)

        exported_count = 0
        skipped_count = 0
        masked_written_total = 0
        all_pixels_written_total = 0
        confidence_written_total = 0
        total = 0

        for batch in data_loader:
            images = batch["image"].to(device, non_blocking=True)
            with torch.no_grad():
                logits = model(images)
                probs = torch.softmax(logits, dim=1)
                confidence, pred = torch.max(probs, dim=1)
                pred = pred.detach().cpu().numpy().astype(np.uint8)
                confidence = confidence.detach().cpu().numpy().astype(np.float32)

            tile_ids = list(batch["tile_id"])
            batch_splits = list(batch["split"])
            for idx in range(len(tile_ids)):
                tile_id = str(tile_ids[idx])
                item_split = str(batch_splits[idx])
                _, label_path = build_tile_paths(
                    dataset_root=dataset_root,
                    image_dirname=data_cfg["image_dirname"],
                    label_dirname=data_cfg["label_dirname"],
                    image_suffix=data_cfg["image_suffix"],
                    label_suffix=data_cfg["label_suffix"],
                    split_name=item_split,
                    tile_id=tile_id,
                )

                out_dir = pred_root / item_split
                export_result = export_one_prediction(
                    out_dir=out_dir,
                    tile_id=tile_id,
                    pred_class_map=pred[idx],
                    confidence_map=confidence[idx],
                    label_path=label_path,
                    colormap=colormap,
                    overwrite=bool(args.overwrite),
                    write_sidecars=not bool(args.skip_sidecars),
                    ignore_index=ignore_index,
                    class_lum_ids=class_lum_ids,
                    ignore_lum_ids=ignore_lum_ids,
                )

                total += 1
                masked_written_total += int(export_result["pred_masked_written"])
                all_pixels_written_total += int(export_result["pred_all_pixels_written"])
                confidence_written_total += int(export_result["confidence_written"])

                if str(export_result["tile_status"]) == "exported":
                    exported_count += 1
                else:
                    skipped_count += 1

            logger.info(
                "split=%s processed=%d/%d exported_tiles=%d skipped_tiles=%d masked_written=%d all_pixels_written=%d confidence_written=%d",
                split_name,
                total,
                len(split_df),
                exported_count,
                skipped_count,
                masked_written_total,
                all_pixels_written_total,
                confidence_written_total,
            )

        summary_rows.append(
            {
                "split": split_name,
                "requested_tiles": int(len(split_df)),
                "exported_tiles": int(exported_count),
                "skipped_exists": int(skipped_count),
                "pred_masked_written": int(masked_written_total),
                "pred_all_pixels_written": int(all_pixels_written_total),
                "confidence_written": int(confidence_written_total),
                "status": "ok",
            }
        )

    summary_path = run_paths["metrics_dir"] / "prediction_export_summary.csv"
    write_csv(
        path=summary_path,
        rows=summary_rows,
        fieldnames=[
            "split",
            "requested_tiles",
            "exported_tiles",
            "skipped_exists",
            "pred_masked_written",
            "pred_all_pixels_written",
            "confidence_written",
            "status",
        ],
    )
    logger.info("Prediction export summary: %s", summary_path)
    logger.info("Prediction raster export completed successfully.")


if __name__ == "__main__":
    main()
