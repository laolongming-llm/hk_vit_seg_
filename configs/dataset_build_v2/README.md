# Dataset Build V2 Configs

本目录用于数据侧 v2 重构流程的参数管理，覆盖三步：

1. `05_vit_dataset_dry_run.py`
2. `06_export_vit_dataset_tiles.py`
3. `07_build_train_manifest_balanced.py`

推荐直接使用（默认走 candidate_d）：

```powershell
python scripts/data_prep/08_run_dataset_build_v2_pipeline.py --python-exe python
```

可选配置：

1. `dry_run_candidate_a.yaml`
2. `dry_run_candidate_b.yaml`
3. `dry_run_candidate_c.yaml`
4. `dry_run_candidate_d.yaml`（当前推荐）
5. `export_from_candidate_a.yaml`
6. `export_from_candidate_b.yaml`
7. `export_from_candidate_c.yaml`
8. `export_from_candidate_d.yaml`（当前推荐）
9. `balance_train_manifest_v2.yaml`

