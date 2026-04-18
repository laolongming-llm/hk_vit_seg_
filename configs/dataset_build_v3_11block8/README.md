# dataset_build_v3_11block8

该目录用于“11block-only + 8类”数据集构建流程。

- `dry_run_11block8.yaml`：供 `scripts/data_prep/10_vit_dataset_dry_run_11block_only.py` 使用。
- `export_11block8.yaml`：供 `scripts/data_prep/11_export_vit_dataset_tiles_11block_only.py` 使用。
- `balance_train_manifest_11block8.yaml`：供 `scripts/data_prep/07_build_train_manifest_balanced.py` 使用。

推荐一键执行：

```bash
python scripts/data_prep/12_run_dataset_build_v3_pipeline.py
```

