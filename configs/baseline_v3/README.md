# baseline_v3 配置说明

本目录用于 `baseline_v3` 训练配置。相对 `baseline_v1` 的关键变化是：
- 新增 `data.ignore_lum_ids: [6]`，将农业用地（lum_id=6）映射为 `ignore_index=255`；
- 保持 `vit_large_patch16_dinov3.sat493m` + `pretrained: true`；
- 提供 3 个种子配置（42 / 3407 / 2026）。

## 文件说明

- `train_vit_seg_baseline_v3_base.yaml`：公共基础配置。
- `train_vit_seg_baseline_v3_seed42.yaml`：seed=42。
- `train_vit_seg_baseline_v3_seed3407.yaml`：seed=3407。
- `train_vit_seg_baseline_v3_seed2026.yaml`：seed=2026。

## 训练命令（在项目根目录执行）

```powershell
# seed42
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v3/train_vit_seg_baseline_v3_seed42.yaml

# seed3407
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v3/train_vit_seg_baseline_v3_seed3407.yaml

# seed2026
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v3/train_vit_seg_baseline_v3_seed2026.yaml
```

## 评估命令（示例：seed42）

```powershell
python scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v3/train_vit_seg_baseline_v3_seed42.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v3_seed42/checkpoints/best.pth
```

## 输出目录

- `outputs/train_runs/formal/baseline_v3_seed42/`
- `outputs/train_runs/formal/baseline_v3_seed3407/`
- `outputs/train_runs/formal/baseline_v3_seed2026/`
