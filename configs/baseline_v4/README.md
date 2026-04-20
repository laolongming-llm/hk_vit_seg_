# baseline_v4 配置说明

本目录保留一版单配置，用于“保留预训练权重 + 重点训练 ASPP 头”的实验。

## 文件

- `train_vit_seg_baseline_v4_single.yaml`

## 关键点

- `model.pretrained: true`：不关闭预训练权重。
- `optimizer.backbone_lr_mult: 0.0`：将主干学习率设为 0（近似冻结主干）。
- `optimizer.decoder_lr_mult: 1.0`：正常训练 ASPP 解码头。

## 训练命令（项目根目录执行）

```powershell
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v4/train_vit_seg_baseline_v4_single.yaml
```
