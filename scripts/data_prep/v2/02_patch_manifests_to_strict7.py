#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Patch dry-run manifests from 8-class schema to strict7 schema.

Main behavior:
1) Verify dropped class column (default class_8) is all zeros (strict mode).
2) Drop dropped class column in:
   - tiles_manifest.csv
   - split_summary.csv
   - eco_split_class_stats.csv (if exists)
3) Write a patch report.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Final, List, Tuple


PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_DIR: Final[Path] = PROJECT_ROOT / "data" / "processed" / "vit_dataset" / "v2_11block7" / "manifests"


def to_abs_path(path_ref: str) -> Path:
    p = Path(path_ref).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch dry-run manifests to strict7 schema.")
    parser.add_argument("--manifest-dir", default=str(DEFAULT_MANIFEST_DIR), help="Directory containing manifests")
    parser.add_argument("--drop-class-id", type=int, default=8, help="Class column to drop (default: class_8)")
    parser.add_argument("--num-classes", type=int, default=7, help="Target class count (default: 7)")
    parser.add_argument(
        "--allow-nonzero-drop-class",
        action="store_true",
        help="Allow non-zero values in dropped class column (not recommended)",
    )
    parser.add_argument(
        "--report-md",
        default="",
        help="Optional report markdown path. Default: <manifest-dir>/strict7_patch_report.md",
    )
    return parser


def _read_csv(csv_path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    return rows, fieldnames


def _write_csv(csv_path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return 0.0


def _drop_class_column(
    csv_path: Path,
    drop_col: str,
    required_cols: List[str],
    strict_zero: bool,
) -> tuple[int, int]:
    if not csv_path.exists():
        return (0, 0)

    rows, fieldnames = _read_csv(csv_path)
    missing = [c for c in required_cols if c not in fieldnames]
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    checked_nonzero = 0
    dropped = 0
    if drop_col in fieldnames:
        checked_nonzero = int(sum(_safe_float((r.get(drop_col) or "0").strip()) for r in rows))
        if strict_zero and checked_nonzero != 0:
            raise ValueError(
                f"{csv_path} has non-zero values in {drop_col}: sum={checked_nonzero}. "
                "Please verify remap stage first."
            )
        new_fieldnames = [c for c in fieldnames if c != drop_col]
        for r in rows:
            r.pop(drop_col, None)
        fieldnames = new_fieldnames
        dropped = 1

    _write_csv(csv_path, rows, fieldnames)
    return (checked_nonzero, dropped)


def main() -> None:
    args = build_parser().parse_args()
    manifest_dir = to_abs_path(args.manifest_dir)
    if not manifest_dir.exists():
        raise FileNotFoundError(f"manifest dir not found: {manifest_dir}")

    drop_col = f"class_{int(args.drop_class_id)}"
    strict_zero = not bool(args.allow_nonzero_drop_class)

    tiles_manifest = manifest_dir / "tiles_manifest.csv"
    split_summary = manifest_dir / "split_summary.csv"
    eco_stats = manifest_dir / "eco_split_class_stats.csv"

    required_tiles = ["tile_id", "pair_name", "final_split", "class_255"]
    # split_summary / eco_split_class_stats usually do not contain class_255.
    required_summary = ["split", "tile_count"] if (manifest_dir / "split_summary.csv").exists() else []

    checks: List[str] = []
    total_drop_col_sum = 0
    total_files_dropped = 0

    for csv_path, required in [
        (tiles_manifest, required_tiles),
        (split_summary, required_summary),
        (eco_stats, required_summary),
    ]:
        if not csv_path.exists():
            continue
        csum, dropped = _drop_class_column(
            csv_path=csv_path,
            drop_col=drop_col,
            required_cols=required,
            strict_zero=strict_zero,
        )
        total_drop_col_sum += csum
        total_files_dropped += dropped
        checks.append(f"- {csv_path.name}: drop_col_sum={csum}, dropped={bool(dropped)}")

    # Basic target-class sanity check on tiles_manifest
    if tiles_manifest.exists():
        _, fieldnames = _read_csv(tiles_manifest)
        expected_cols = [f"class_{i}" for i in range(1, int(args.num_classes) + 1)]
        missing_expected = [c for c in expected_cols if c not in fieldnames]
        if missing_expected:
            raise ValueError(f"{tiles_manifest} missing expected target class columns: {missing_expected}")
        if drop_col in fieldnames:
            raise ValueError(f"{tiles_manifest} still contains dropped column: {drop_col}")

    report_path = to_abs_path(args.report_md) if args.report_md else (manifest_dir / "strict7_patch_report.md")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Strict7 Manifest Patch Report",
                "",
                f"- manifest_dir: `{manifest_dir}`",
                f"- dropped_column: `{drop_col}`",
                f"- strict_zero_check: `{strict_zero}`",
                f"- total_drop_col_sum: `{total_drop_col_sum}`",
                f"- files_with_dropped_column: `{total_files_dropped}`",
                "",
                "## Per-file checks",
                "",
                *checks,
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("[OK] strict7 manifest patch completed")
    print(f"  - manifest_dir: {manifest_dir}")
    print(f"  - dropped_column: {drop_col}")
    print(f"  - total_drop_col_sum: {total_drop_col_sum}")
    print(f"  - report: {report_path}")


if __name__ == "__main__":
    main()
