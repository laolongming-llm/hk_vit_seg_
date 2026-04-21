# 数据预处理 v2（Strict7 重构）

本目录提供一套“严格 7 类”数据集重构流程，在尽量复用 `scripts/data_prep/` 既有脚本的前提下完成重构。

## 目标

重构 11block 数据集，使标签真正变为 7 类：

- 保留：`1,2,3,4,5`
- 重编号：`7->6`，`8->7`
- 忽略：`6->255`

## 脚本说明

- `01_remap_labels_to_strict7.py`
  - 将 11block 标签栅格重映射为 strict7 体系。
- `02_patch_manifests_to_strict7.py`
  - 对 dry-run 产出的清单做 strict7 补丁（删除 `class_8`，并进行安全校验）。
- `03_run_dataset_rebuild_strict7_pipeline.py`
  - 总控脚本，复用现有 `10/11/07` 脚本串联完整流程。

## 一键运行

在项目根目录执行：

```powershell
python scripts/data_prep/v2/03_run_dataset_rebuild_strict7_pipeline.py
```

默认输出：

- 重映射标签：
  - `data/interim/labels_1m_aligned/lumid_11block_1m_aligned_v2_strict7.tif`
- 数据集根目录：
  - `data/processed/vit_dataset/v2_11block7`
- 清单目录：
  - `data/processed/vit_dataset/v2_11block7/manifests`

## 可选参数

跳过部分阶段：

```powershell
python scripts/data_prep/v2/03_run_dataset_rebuild_strict7_pipeline.py `
  --skip-remap `
  --skip-balance
```

指定自定义配置/路径：

```powershell
python scripts/data_prep/v2/03_run_dataset_rebuild_strict7_pipeline.py `
  --dry-run-config configs/dataset_build_v3_11block8/dry_run_11block8.yaml `
  --export-config configs/dataset_build_v3_11block8/export_11block8.yaml `
  --balance-config configs/dataset_build_v3_11block8/balance_train_manifest_11block8.yaml `
  --output-root data/processed/vit_dataset/v2_11block7 `
  --manifest-dir data/processed/vit_dataset/v2_11block7/manifests
```

