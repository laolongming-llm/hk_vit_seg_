# deit3_v1 配置说明（预留）

本目录用于后续尝试 **DeiT3** 主干，目录结构与 `baseline_v1` 对齐：

- `train_deit3_v1_base.yaml`
- `train_deit3_v1_seed42.yaml`
- `train_deit3_v1_seed3407.yaml`
- `train_deit3_v1_seed2026.yaml`

## 当前建议

- 先跑你已经确认的 `baseline_v1`（`vit_large_patch16_224` 方案）。
- `deit3_v1` 作为后续对照实验配置预留。

## 重要注意（运行前提）

在当前 `scripts/dl/formal/train_lib.py` 下，`deit3_*` 主干通常会对输入尺寸进行断言（如 `Input height (512) doesn't match model (224)`）。

也就是说，这套配置**已建好但默认不保证可直接跑通**。要实际训练 DeiT3，建议先完成以下任一改造：

1. 在模型构建逻辑里为 `deit3_*` 启用动态输入尺寸（类似 ViT 的 `dynamic_img_size` 处理）。
2. 或在数据管线中统一把输入 resize 到主干要求尺寸（不推荐直接牺牲 512 空间信息）。

## 计划中的运行方式（完成兼容后）

请先激活你自己的 Python/Conda 环境，并在**项目根目录**执行。

```powershell
python scripts/dl/formal/01_train_vit_seg_formal.py `
  --config configs/deit3_v1/train_deit3_v1_seed42.yaml
```
