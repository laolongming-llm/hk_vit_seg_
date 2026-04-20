# baseline_v5 配置说明

本目录用于“部分解冻”实验：
- 保留预训练权重（`pretrained: true`）
- 先冻结 backbone，再只解冻最后几层 block
- 同时训练 ASPP 头

## 文件

- `train_vit_seg_baseline_v5_single.yaml`

## 关键配置

- `model.freeze_backbone: true`
- `model.unfreeze_backbone_last_n_blocks: 2`
- `model.unfreeze_backbone_norm: true`
- `optimizer.backbone_lr_mult: 0.2`
- `optimizer.decoder_lr_mult: 1.0`

说明：
- 这版是“稳健起步”的部分解冻参数；
- 若效果稳定可继续尝试 `unfreeze_backbone_last_n_blocks: 3/4` 做对比。

## 训练命令（项目根目录执行）

```powershell
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/baseline_v5/train_vit_seg_baseline_v5_single.yaml
```
