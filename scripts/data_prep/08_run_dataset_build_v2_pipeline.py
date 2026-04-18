#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
脚本名称：08_run_dataset_build_v2_pipeline.py

功能：
1) 读取 dry-run / export / balanced 三份 YAML 参数；
2) 顺序执行 scripts/data_prep/05、scripts/data_prep/06、scripts/data_prep/07；
3) 统一输出每一步命令，便于复现与排错。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DRYRUN_CFG = PROJECT_ROOT / "configs" / "dataset_build_v2" / "dry_run_candidate_d.yaml"
DEFAULT_EXPORT_CFG = PROJECT_ROOT / "configs" / "dataset_build_v2" / "export_from_candidate_d.yaml"
DEFAULT_BALANCE_CFG = PROJECT_ROOT / "configs" / "dataset_build_v2" / "balance_train_manifest_v2.yaml"


def to_abs_path(path_ref: str | Path) -> Path:
    p = Path(path_ref).expanduser()
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p.resolve()


def load_yaml(path_ref: str | Path) -> Dict[str, object]:
    path = to_abs_path(path_ref)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"配置文件顶层必须是映射(dict)：{path}")
    return payload


def build_cli_args(arg_map: Dict[str, object]) -> List[str]:
    cli: List[str] = []
    for key, value in arg_map.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli.append(flag)
            continue
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            cli.append(flag)
            cli.extend(str(v) for v in value)
            continue
        cli.extend([flag, str(value)])
    return cli


def run_stage(python_exe: str, script_path: Path, config_path: Path, stage_name: str) -> None:
    arg_map = load_yaml(config_path)
    cmd = [python_exe, str(script_path)] + build_cli_args(arg_map)
    print(f"\n[STAGE] {stage_name}")
    print(f"[CFG] {config_path}")
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dataset v2 build pipeline with YAML configs.")
    parser.add_argument("--dry-run-config", default=str(DEFAULT_DRYRUN_CFG), help="05 dry-run 的 YAML 参数文件")
    parser.add_argument("--export-config", default=str(DEFAULT_EXPORT_CFG), help="06 export 的 YAML 参数文件")
    parser.add_argument("--balance-config", default=str(DEFAULT_BALANCE_CFG), help="07 balanced 的 YAML 参数文件")
    parser.add_argument("--skip-dry-run", action="store_true", help="跳过 dry-run 阶段")
    parser.add_argument("--skip-export", action="store_true", help="跳过 export 阶段")
    parser.add_argument("--skip-balance", action="store_true", help="跳过 balanced 阶段")
    parser.add_argument("--python-exe", default=sys.executable, help="用于执行子脚本的 Python 可执行文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dryrun_script = PROJECT_ROOT / "scripts" / "data_prep" / "05_vit_dataset_dry_run.py"
    export_script = PROJECT_ROOT / "scripts" / "data_prep" / "06_export_vit_dataset_tiles.py"
    balance_script = PROJECT_ROOT / "scripts" / "data_prep" / "07_build_train_manifest_balanced.py"

    if not args.skip_dry_run:
        run_stage(
            python_exe=args.python_exe,
            script_path=dryrun_script,
            config_path=to_abs_path(args.dry_run_config),
            stage_name="dry-run",
        )
    if not args.skip_export:
        run_stage(
            python_exe=args.python_exe,
            script_path=export_script,
            config_path=to_abs_path(args.export_config),
            stage_name="export",
        )
    if not args.skip_balance:
        run_stage(
            python_exe=args.python_exe,
            script_path=balance_script,
            config_path=to_abs_path(args.balance_config),
            stage_name="build_train_manifest_balanced",
        )

    print("\n[OK] dataset v2 pipeline completed.")


if __name__ == "__main__":
    main()


