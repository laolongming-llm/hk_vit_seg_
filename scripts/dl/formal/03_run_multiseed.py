"""Formal 多 seed 串行调度脚本。

功能与作用：
1. 按固定顺序（默认 42 -> 3407 -> 2026）串行执行正式训练。
2. 每个 seed 训练完成后自动触发评估（可通过参数关闭）。
3. 将多次子进程执行日志统一输出，便于长任务追踪。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

from train_lib import get_run_paths, load_config


def parse_args() -> argparse.Namespace:
    """解析多 seed 调度参数。"""
    parser = argparse.ArgumentParser(description="Run formal training/evaluation sequentially for multiple seeds.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/baseline_v1/train_vit_seg_baseline_v1_seed42.yaml",
            "configs/baseline_v1/train_vit_seg_baseline_v1_seed3407.yaml",
            "configs/baseline_v1/train_vit_seg_baseline_v1_seed2026.yaml",
        ],
        help="Config paths in execution order.",
    )
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation step after each training run.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable to use.")
    return parser.parse_args()


def run_command(cmd: List[str]) -> None:
    """执行命令并在失败时直接抛错中断。"""
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def resolve_best_checkpoint(config_path: str) -> Path:
    """根据配置定位最佳 checkpoint，不存在则回退 last checkpoint。"""
    cfg = load_config(config_path)
    resolved_config = Path(cfg["_meta"]["resolved_config_path"])
    run_paths = get_run_paths(cfg, resolved_config)
    best_path = run_paths["checkpoints_dir"] / "best.pth"
    if best_path.exists():
        return best_path
    last_path = run_paths["checkpoints_dir"] / "last.pth"
    if last_path.exists():
        return last_path
    raise FileNotFoundError(f"Neither best.pth nor last.pth exists in {run_paths['checkpoints_dir']}")


def main() -> None:
    """按顺序执行多 seed 训练与评估。"""
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    train_script = script_dir / "01_train_vit_seg_formal.py"
    eval_script = script_dir / "02_eval_vit_seg_formal.py"

    for idx, config in enumerate(args.configs, start=1):
        print(f"\n=== [Seed Job {idx}/{len(args.configs)}] config={config} ===")
        run_command([args.python_exe, str(train_script), "--config", config])
        if args.skip_eval:
            continue
        checkpoint = resolve_best_checkpoint(config)
        run_command([args.python_exe, str(eval_script), "--config", config, "--checkpoint", str(checkpoint)])

    print("\n[OK] Multi-seed sequential pipeline completed.")


if __name__ == "__main__":
    main()

