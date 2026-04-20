# baseline_v4 配置说明

本目录用于 `baseline_v4` 训练配置。相对 `baseline_v3` 的关键变化：
- 保留 `data.ignore_lum_ids: [6]`（农业用地映射到 `ignore_index=255`）；
- 关闭预训练权重加载（`model.pretrained: false`）；
- 提供 3 个种子配置（42 / 3407 / 2026）。

## 文件说明

- `train_vit_seg_baseline_v4_base.yaml`：公共基础配置。
- `train_vit_seg_baseline_v4_seed42.yaml`：seed=42。
- `train_vit_seg_baseline_v4_seed3407.yaml`：seed=3407。
- `train_vit_seg_baseline_v4_seed2026.yaml`：seed=2026。

## 单种子训练命令（项目根目录执行）

```powershell
# seed42
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v4/train_vit_seg_baseline_v4_seed42.yaml

# seed3407
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v4/train_vit_seg_baseline_v4_seed3407.yaml

# seed2026
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v4/train_vit_seg_baseline_v4_seed2026.yaml
```

## 单种子评估命令（示例：seed42）

```powershell
python scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v4/train_vit_seg_baseline_v4_seed42.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v4_seed42/checkpoints/best.pth
```

## 多种子串行运行（03 脚本）

```powershell
python scripts/dl/formal/03_run_multiseed.py `
  --configs `
  configs/baseline_v4/train_vit_seg_baseline_v4_seed42.yaml `
  configs/baseline_v4/train_vit_seg_baseline_v4_seed3407.yaml `
  configs/baseline_v4/train_vit_seg_baseline_v4_seed2026.yaml
```

说明：
- 默认会对每个 seed 先训练再评估；
- 如只训练不评估，可追加 `--skip-eval`。

## 多种子结果合并（04 脚本）

```powershell
python scripts/dl/formal/04_aggregate_multiseed_results.py `
  --configs `
  configs/baseline_v4/train_vit_seg_baseline_v4_seed42.yaml `
  configs/baseline_v4/train_vit_seg_baseline_v4_seed3407.yaml `
  configs/baseline_v4/train_vit_seg_baseline_v4_seed2026.yaml `
  --output-dir outputs/evaluation_reports/baseline_v4_multiseed
```

合并输出文件：
- `seed_summary.csv`：每个 seed、每个 split 的原始汇总；
- `split_aggregate.csv`：按 split 统计的均值和标准差；
- `baseline_summary.md`：便于汇报的 Markdown 摘要。

## 训练输出目录

- `outputs/train_runs/formal/baseline_v4_seed42/`
- `outputs/train_runs/formal/baseline_v4_seed3407/`
- `outputs/train_runs/formal/baseline_v4_seed2026/`
