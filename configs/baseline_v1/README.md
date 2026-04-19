# baseline_v1 配置说明（v3_11block8）

本目录提供正式训练（formal）最小可用配置，已对齐当前数据集：

- 数据根目录：`data/processed/vit_dataset/v3_11block8`
- split 口径：`train / val / test`
- manifest：`tiles_manifest_balanced_train.csv`（仅 train 被重采样，val/test 保持不变）

## 文件说明

- `train_vit_seg_baseline_v1_base.yaml`：基础配置（模型、数据、训练超参）
- `train_vit_seg_baseline_v1_seed42.yaml`：seed=42 覆盖配置
- `train_vit_seg_baseline_v1_seed3407.yaml`：seed=3407 覆盖配置
- `train_vit_seg_baseline_v1_seed2026.yaml`：seed=2026 覆盖配置

## 推荐执行

单 seed（先验证）：

```powershell
E:\Programfiles\miniconda\envs\vit-seg\python.exe scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v1/train_vit_seg_baseline_v1_seed42.yaml
```

训练后评估（val + test）：

```powershell
E:\Programfiles\miniconda\envs\vit-seg\python.exe scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v1/train_vit_seg_baseline_v1_seed42.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v1_seed42/checkpoints/best.pth
```

多 seed 串行（默认即本目录 3 个 seed 配置）：

```powershell
E:\Programfiles\miniconda\envs\vit-seg\python.exe scripts/dl/formal/03_run_multiseed.py `
  --python-exe E:\Programfiles\miniconda\envs\vit-seg\python.exe
```

结果聚合：

```powershell
E:\Programfiles\miniconda\envs\vit-seg\python.exe scripts/dl/formal/04_aggregate_multiseed_results.py
```

## 备注

- `model.pretrained` 默认设为 `false`，避免首次运行因外网下载权重导致中断。若环境稳定联网且需要更高起点，可改为 `true`。
- 若显存不足，优先降低 `loader.batch_size`，或将 `model.backbone` 换为更小主干。
