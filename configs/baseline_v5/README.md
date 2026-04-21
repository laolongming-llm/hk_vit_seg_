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
