# baseline_v4 配置说明

本目录保留一版单配置，用于“保留预训练权重 + 冻结 backbone + 训练 ASPP 头”的实验。

## 文件

- `train_vit_seg_baseline_v4_single.yaml`

## 关键点

- `model.pretrained: true`：保留预训练初始化。
- `model.freeze_backbone: true`：硬冻结 backbone（`requires_grad=False`）。
- `optimizer.decoder_lr_mult: 1.0`：正常训练 ASPP 解码头。

## 训练命令（项目根目录执行）

```powershell
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v4/train_vit_seg_baseline_v4_single.yaml
```

## 测试命令（项目根目录执行）

```powershell
python scripts/dl/formal/02_eval_vit_seg_formal.py `
  --config configs/baseline_v4/train_vit_seg_baseline_v4_single.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v4_freeze_seed42/checkpoints/best.pth
```
