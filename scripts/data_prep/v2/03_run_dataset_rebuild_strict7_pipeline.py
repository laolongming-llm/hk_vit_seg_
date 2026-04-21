#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run strict7 dataset rebuild pipeline by reusing existing data_prep scripts.

Pipeline:
1) remap labels to strict7               (v2/01)
2) dry-run (11block-only)                (data_prep/10)
3) patch manifests to strict7 schema     (v2/02)
4) export tiles                          (data_prep/11)
5) build balanced train manifest         (data_prep/07)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DRYRUN_CFG = PROJECT_ROOT / "configs" / "dataset_build_v3_11block8" / "dry_run_11block8.yaml"
DEFAULT_EXPORT_CFG = PROJECT_ROOT / "configs" / "dataset_build_v3_11block8" / "export_11block8.yaml"
DEFAULT_BALANCE_CFG = PROJECT_ROOT / "configs" / "dataset_build_v3_11block8" / "balance_train_manifest_11block8.yaml"

DEFAULT_INPUT_LABEL = PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_11block_1m_aligned.tif"
DEFAULT_REMAPPED_LABEL = (
    PROJECT_ROOT / "data" / "interim" / "labels_1m_aligned" / "lumid_11block_1m_aligned_v2_strict7.tif"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v2_11block7"
DEFAULT_MANIFEST_DIR = DEFAULT_OUTPUT_ROOT / "manifests"


def to_abs_path(path_ref: str | Path) -> Path:
    p = Path(path_ref).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def run_cmd(cmd: List[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild strict7 dataset by reusing existing scripts.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable for child scripts")

    parser.add_argument("--dry-run-config", default=str(DEFAULT_DRYRUN_CFG), help="Config for script 10")
    parser.add_argument("--export-config", default=str(DEFAULT_EXPORT_CFG), help="Config for script 11")
    parser.add_argument("--balance-config", default=str(DEFAULT_BALANCE_CFG), help="Config for script 07")

    parser.add_argument("--input-label", default=str(DEFAULT_INPUT_LABEL), help="Source label raster")
    parser.add_argument("--remapped-label", default=str(DEFAULT_REMAPPED_LABEL), help="Remapped strict7 label raster")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Dataset output root")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR), help="Dry-run manifest directory")

    parser.add_argument("--skip-remap", action="store_true", help="Skip remap stage")
    parser.add_argument("--skip-dry-run", action="store_true", help="Skip dry-run stage")
    parser.add_argument("--skip-patch", action="store_true", help="Skip manifest patch stage")
    parser.add_argument("--skip-export", action="store_true", help="Skip export stage")
    parser.add_argument("--skip-balance", action="store_true", help="Skip balance stage")

    parser.add_argument(
        "--allow-nonzero-drop-class",
        action="store_true",
        help="Allow class_8 non-zero when patching manifests",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    python_exe = args.python_exe
    dryrun_cfg = to_abs_path(args.dry_run_config)
    export_cfg = to_abs_path(args.export_config)
    balance_cfg = to_abs_path(args.balance_config)
    input_label = to_abs_path(args.input_label)
    remapped_label = to_abs_path(args.remapped_label)
    output_root = to_abs_path(args.output_root)
    manifest_dir = to_abs_path(args.manifest_dir)

    remap_script = PROJECT_ROOT / "scripts" / "data_prep" / "v2" / "01_remap_labels_to_strict7.py"
    patch_script = PROJECT_ROOT / "scripts" / "data_prep" / "v2" / "02_patch_manifests_to_strict7.py"
    dryrun_script = PROJECT_ROOT / "scripts" / "data_prep" / "10_vit_dataset_dry_run_11block_only.py"
    export_script = PROJECT_ROOT / "scripts" / "data_prep" / "11_export_vit_dataset_tiles_11block_only.py"
    balance_script = PROJECT_ROOT / "scripts" / "data_prep" / "07_build_train_manifest_balanced.py"

    print("[INFO] strict7 dataset rebuild pipeline starts", flush=True)
    print(f"[INFO] output_root={output_root}", flush=True)
    print(f"[INFO] manifest_dir={manifest_dir}", flush=True)

    if not args.skip_remap:
        run_cmd(
            [
                python_exe,
                str(remap_script),
                "--input-label",
                str(input_label),
                "--output-label",
                str(remapped_label),
            ]
        )

    if not args.skip_dry_run:
        run_cmd(
            [
                python_exe,
                str(dryrun_script),
                "--config",
                str(dryrun_cfg),
                "--label-11block",
                str(remapped_label),
                "--output-dir",
                str(manifest_dir),
            ]
        )

    if not args.skip_patch:
        patch_cmd = [
            python_exe,
            str(patch_script),
            "--manifest-dir",
            str(manifest_dir),
            "--drop-class-id",
            "8",
            "--num-classes",
            "7",
        ]
        if args.allow_nonzero_drop_class:
            patch_cmd.append("--allow-nonzero-drop-class")
        run_cmd(patch_cmd)

    if not args.skip_export:
        run_cmd(
            [
                python_exe,
                str(export_script),
                "--config",
                str(export_cfg),
                "--label-11block",
                str(remapped_label),
                "--tiles-manifest",
                str(manifest_dir / "tiles_manifest.csv"),
                "--manifests-dir",
                str(manifest_dir),
                "--output-root",
                str(output_root),
            ]
        )

    if not args.skip_balance:
        run_cmd(
            [
                python_exe,
                str(balance_script),
                "--config",
                str(balance_cfg),
                "--input-manifest",
                str(manifest_dir / "tiles_manifest.csv"),
                "--output-manifest",
                str(manifest_dir / "tiles_manifest_balanced_train.csv"),
                "--report-dir",
                str(manifest_dir / "reports_balanced"),
                "--num-classes",
                "7",
                "--rare-class-ids",
                "6,7",
            ]
        )

    print("[OK] strict7 dataset rebuild pipeline completed", flush=True)


if __name__ == "__main__":
    main()

