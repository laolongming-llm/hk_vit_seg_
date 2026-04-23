# baseline_v5 配置说明

本目录用于“部分解冻”实验：
- 保留预训练权重（`pretrained: true`）
- 先冻结 backbone，再只解冻最后几层 block
- 同时训练 ASPP 头
- 训练数据切换为 `v2_11block7`（严格 7 类）

## 文件

- `train_vit_seg_baseline_v5_base.yaml`：v5 主配置（严格 7 类 + 部分解冻策略）。
- `train_vit_seg_baseline_v5_single.yaml`：基座配置（seed42）。
- `train_vit_seg_baseline_v5_seed3407.yaml`：基于 single 覆盖种子为 3407。
- `train_vit_seg_baseline_v5_seed2026.yaml`：基于 single 覆盖种子为 2026。
- `export_v5_fullcoverage.yaml`：05 导出专用配置（非 balanced manifest，覆盖 train/val/test）。

## 关键配置

- `data.dataset_root: data/processed/vit_dataset/v2_11block7`
- `data.manifest_path: data/processed/vit_dataset/v2_11block7/manifests/tiles_manifest_balanced_train.csv`
- `data.class_lum_ids: [1,2,3,4,5,6,7]`
- `data.ignore_lum_ids: []`
- `model.freeze_backbone: true`
- `model.unfreeze_backbone_last_n_blocks: 2`
- `model.unfreeze_backbone_norm: true`
- `optimizer.backbone_lr_mult: 0.2`
- `optimizer.decoder_lr_mult: 1.0`

## 训练命令（项目根目录执行）

```powershell
# seed42
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v5/train_vit_seg_baseline_v5_single.yaml

# seed3407
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v5/train_vit_seg_baseline_v5_seed3407.yaml

# seed2026
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v5/train_vit_seg_baseline_v5_seed2026.yaml
```

## 测试命令（项目根目录执行）

```powershell
# seed42
python scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v5/train_vit_seg_baseline_v5_single.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed42/checkpoints/best.pth

# seed3407
python scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v5/train_vit_seg_baseline_v5_seed3407.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed3407/checkpoints/best.pth

# seed2026
python scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v5/train_vit_seg_baseline_v5_seed2026.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed2026/checkpoints/best.pth
```

## 预测导出命令（05，项目根目录执行）

说明：
- 为尽量覆盖当前数据集可用区域，导出配置使用 `tiles_manifest.csv`（非 balanced）。
- `--splits` 在 PowerShell 下需整体加引号（如 `"train,val,test"`）。

```powershell
# 全量导出（train+val+test）
python scripts/dl/formal/05_export_prediction_rasters.py `
  --config configs/baseline_v5/export_v5_fullcoverage.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed42/checkpoints/best.pth `
  --splits "train,val,test"
```

```powershell
# 快速冒烟（每个 split 仅导出前 20 个 tile）
python scripts/dl/formal/05_export_prediction_rasters.py `
  --config configs/baseline_v5/export_v5_fullcoverage.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed42/checkpoints/best.pth `
  --splits "train,val,test" `
  --max-tiles-per-split 20
```

## 切片拼接命令（06，阶段 B：max_conf 融合）

说明：
- 06 脚本会将 05 的预测切片按“逐像元最大置信度优先”进行融合拼接。
- 默认建议先拼接 `pred_all_pixels`。

```powershell
# 全量拼接（按 split + 全部 split 各输出一套）
python scripts/dl/formal/06_mosaic_prediction_tiles.py `
  --run-dir outputs/train_runs/formal/baseline_v5_export_fullcoverage_seed42 `
  --splits "train,val,test" `
  --layer pred_all_pixels `
  --mode both `
  --fusion max_conf `
  --overwrite
```

```powershell
# 快速冒烟（每个 split 仅取 20 个 tile）
python scripts/dl/formal/06_mosaic_prediction_tiles.py `
  --run-dir outputs/train_runs/formal/baseline_v5_export_fullcoverage_seed42 `
  --splits "train,val,test" `
  --layer pred_all_pixels `
  --mode both `
  --fusion max_conf `
  --max-tiles-per-split 20 `
  --output-dir outputs/train_runs/formal/baseline_v5_export_fullcoverage_seed42/mosaics_smoke `
  --overwrite
```
