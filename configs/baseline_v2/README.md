# baseline_v2 配置说明（单配置版）

本目录用于放置 `baseline_v2` 的单文件训练配置（不拆多 seed）。

## 文件说明

- `train_vit_seg_baseline_v2_single.yaml`：从 `hk_flood_vit_seg` 迁移的单配置版本。
- 当前 `run_name/run_subdir` 已固定为 `vit_seg_baseline_v2_single`。

## 运行前检查

- 本配置已对齐当前项目数据路径：
  - `data.dataset_root: data/processed/vit_dataset/v3_11block8`
  - `data.manifest_path: data/processed/vit_dataset/v3_11block8/manifests/tiles_manifest_balanced_train.csv`
  - `data.num_classes: 8`
- 当前 `freeze.verify_fingerprint_on_start: false`，默认跳过指纹校验以便直接启动训练。

## 训练命令（项目根目录执行）

```powershell
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v2/train_vit_seg_baseline_v2_single.yaml
```

## 评估命令（项目根目录执行）

```powershell
python scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v2/train_vit_seg_baseline_v2_single.yaml `
  --checkpoint outputs/train_runs/formal/vit_seg_baseline_v2_single/checkpoints/best.pth
```

## 主要输出路径

- 训练日志：`outputs/train_runs/formal/vit_seg_baseline_v2_single/logs/train.log`
- 验证指标：`outputs/train_runs/formal/vit_seg_baseline_v2_single/metrics/val_metrics.csv`
- 最优权重：`outputs/train_runs/formal/vit_seg_baseline_v2_single/checkpoints/best.pth`
- 最后权重：`outputs/train_runs/formal/vit_seg_baseline_v2_single/checkpoints/last.pth`
