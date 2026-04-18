# smoke_v1 配置说明（v3_11block8）

本目录用于 `scripts/dl/smoke/` 三步流程：

1. 构建 smoke 子集 manifest  
2. 运行轻量训练（TinySegNet）  
3. 检查 checkpoint 可加载与单批推理

## 配置文件

- `smoke_tinyseg_v3_11block8_seed42.yaml`

## 说明

- 当前 v3 数据集 split 为 `train/val/test`。  
- `01_build_smoke_subset_manifest.py` 脚本仍保留旧字段 `test_in_domain/test_eco_holdout`，因此本配置中将这两个计数设为 `0`，只抽样 `train/val`，满足 smoke 训练与验证需求。

## 推荐执行顺序

```powershell
# 1) 生成 smoke 子集清单
E:\Programfiles\miniconda\envs\vit-seg\python.exe scripts/dl/smoke/01_build_smoke_subset_manifest.py `
  --config configs/smoke_v1/smoke_tinyseg_v3_11block8_seed42.yaml

# 2) 运行 smoke 训练
E:\Programfiles\miniconda\envs\vit-seg\python.exe scripts/dl/smoke/02_train_vit_seg_smoke.py `
  --config configs/smoke_v1/smoke_tinyseg_v3_11block8_seed42.yaml

# 3) checkpoint 加载检查（默认用 best.pth）
E:\Programfiles\miniconda\envs\vit-seg\python.exe scripts/dl/smoke/03_smoke_checkpoint_load_check.py `
  --config configs/smoke_v1/smoke_tinyseg_v3_11block8_seed42.yaml `
  --checkpoint outputs/train_runs/smoke/v3_11block8_smoke_seed42/checkpoints/best.pth
```
