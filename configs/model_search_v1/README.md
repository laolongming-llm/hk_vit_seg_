# model_search_v1 配置说明

本目录用于“主干 + 分割头 + 训练策略”快速对照实验。

统一约定：
1. 基于 `configs/anti_overfit_v4/train_vit_seg_anti_overfit_v4_data_v2_adapt_seed42.yaml` 继承。
2. 数据集与 split 不变，只改变模型结构与训练策略。
3. 每个配置都提供独立 `run_name/run_subdir`，避免结果互相覆盖。

建议执行顺序：
1. `train_model_search_v1_vit_simple_ce_dice_seed42.yaml`
2. `train_model_search_v1_vit_aspp_ce_dice_seed42.yaml`
3. `train_model_search_v1_convnext_aspp_ce_dice_seed42.yaml`
