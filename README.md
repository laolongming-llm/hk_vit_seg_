# HK ViT Seg (Hong Kong LUM Classification)

本项目用于香港区域遥感影像的地类语义分割（LUM 分类），核心模型为 ViT 主干分割框架（`timm_seg`）。

当前已确定的**最终主线方案**：
- 训练配置：`configs/baseline_v5`
- 主实验数据：`data/processed/vit_dataset/v2_11block7`（strict7）
- 最终成果输出：`outputs/train_runs/formal/baseline_v5_export_inference_fullcoverage_nosplit_seed42`

## 1. 项目目标

- 基于 ViT 分割模型完成香港区域地类分类。
- 在课程论文场景下形成可复现的训练、测试、聚合与全域出图流程。
- 统一为 strict7 类别体系，避免极低占比类别对训练与评估的干扰。

## 2. 项目数据与方法说明（补充）

### 2.1 数据层级与当前主线数据

本项目数据按 `raw -> interim -> processed -> outputs` 分层组织：

- `data/raw/`
  - `osm/hong-kong-260330.osm.pbf`（OSM 原始矢量数据）
  - `imagery/TDOP_TIFF_11*/*.tif`、`imagery/TDOP_TIFF_2SWD/*.tif`（原始遥感影像）
- `data/interim/`
  - OSM 清洗与重分类中间产物（`gpkg/`、`cleaned_vectors/`）
  - 栅格标签与影像对齐产物（`masks_full/`、`imagery_1m_aligned/`、`labels_1m_aligned/`）
- `data/processed/vit_dataset/`
  - `v3_11block8`：11block-only 8 类数据集（中间版本）
  - `v2_11block7`：strict7 训练数据集（最终训练主线）
  - `v2_11block7_inference_full_nosplit`：全覆盖推理数据集（最终出图主线）

当前最终主线：

- 训练数据：`data/processed/vit_dataset/v2_11block7`
- 推理出图数据：`data/processed/vit_dataset/v2_11block7_inference_full_nosplit`

### 2.2 方法流程（从原始数据到成果图）

完整方法链路分为 4 阶段：

1. 原始数据预处理：OSM 转换、规则重分类、栅格化、影像标签对齐（`scripts/data_prep/01~04`）。  
2. 数据集构建：11block 8 类切片与 balanced manifest（`10/11/07` 或 `12` 一键）。  
3. strict7 重构：将类别体系重映射为 7 类训练口径（`scripts/data_prep/v2/03` 一键）。  
4. 模型训练与制图：`baseline_v5` 三种子训练/测试/聚合 + 05/06/07 全域导出与 GIS 化。

### 2.3 原始数据预处理（01-04）

首次从原始数据构建时，按以下顺序执行（项目根目录）：

```powershell
python scripts/data_prep/01_pbf_to_gpkg.py
python scripts/data_prep/02_reclassify_multipolygons.py
python scripts/data_prep/03_rasterize_LUMID_classes.py
python scripts/data_prep/04_prepare_imagery_and_labels.py
```

说明：

- 以上命令默认读取仓库约定路径（`data/raw/osm`、`data/raw/imagery`）。  
- `01~04` 脚本默认会在 `data/interim` 下生成后续数据集构建所需中间产物。  
- 若你的原始数据路径不同，可通过各脚本 `--help` 查看并覆盖默认参数。

### 2.4 数据集构建（含 strict7 最终口径）

1) 先构建 11block-only 8 类数据集（中间版本）：

```powershell
python scripts/data_prep/12_run_dataset_build_v3_pipeline.py
```

2) 再重构 strict7 训练数据集（最终训练口径）：

```powershell
python scripts/data_prep/v2/03_run_dataset_rebuild_strict7_pipeline.py
```

3) 构建“全覆盖无缝优先”推理数据集（最终出图口径）：

```powershell
python scripts/data_prep/10_vit_dataset_dry_run_11block_only.py --config configs/dataset_build_v5_inference/dry_run_11block_strict7_fullcoverage_nosplit.yaml
python scripts/data_prep/11_export_vit_dataset_tiles_11block_only.py --config configs/dataset_build_v5_inference/export_tiles_11block_strict7_fullcoverage_nosplit.yaml
```

## 3. 最终采用的配置与结果概览

### 3.1 训练配置（最终）

- 主配置：`configs/baseline_v5/train_vit_seg_baseline_v5_base.yaml`
- 主干：`vit_large_patch16_dinov3.sat493m`（`pretrained: true`）
- 策略：冻结 backbone 后部分解冻（最后 2 个 block + norm）
- 分割头：`aspp_head`
- 类别体系：strict7（`num_classes: 7`，类别 id 为 `1..7`）

### 3.2 三种子聚合结果（strict7）

来源：`outputs/evaluation_reports/2026-04-22_baseline_v5_strict7/split_aggregate.csv`

- `test`: mIoU `0.462248 ± 0.002271`，mF1 `0.547154 ± 0.003675`，OA `0.955822 ± 0.000571`
- `val`: mIoU `0.477676 ± 0.002354`，mF1 `0.566607 ± 0.003656`，OA `0.946044 ± 0.000355`

### 3.3 全域成果（最终交付目录）

`outputs/train_runs/formal/baseline_v5_export_inference_fullcoverage_nosplit_seed42/mosaics/`

关键文件：
- `mosaic_all_splits_pred_all_pixels_max_conf.tif`：全域拼接分类图（max_conf 融合）
- `mosaic_all_splits_pred_all_pixels_max_conf_confidence.tif`：全域置信度图（最大 softmax）
- `mosaic_all_splits_pred_all_pixels_max_conf_strict7_gis.tif`：GIS 友好版（0..7 + 调色板 + 类别名）
- `.qml/.clr`：QGIS/通用样式文件

## 4. 目录结构（核心）

```text
configs/
  baseline_v5/                         # 最终训练与导出配置
  dataset_build_v5_inference/          # 全覆盖推理数据集构建配置（nosplit）

scripts/
  data_prep/v2/                        # strict7 数据集重构流程
  data_prep/01~04                      # 原始数据预处理流程
  data_prep/10~12                      # 数据集切片与一键构建流程
  dl/formal/
    01_train_vit_seg_formal.py         # 训练
    02_eval_vit_seg_formal.py          # 测试
    04_aggregate_multiseed_results.py  # 多种子聚合
    05_export_prediction_rasters.py    # 导出预测切片
    06_mosaic_prediction_tiles.py      # 切片拼接（max_conf）
    07_make_gis_ready_classified_raster.py # GIS 友好分类栅格
```

## 5. 运行环境

本仓库命令均按“**在项目根目录执行**”编写，环境由研究者自行激活。

推荐参考环境文件：
- `environment.dl.gpu.yml`（Python 3.10 + PyTorch 2.5.1 + CUDA 12.1）

说明：
- `requirements.txt` 已提供，建议优先使用 conda 环境文件（`environment.dl.gpu.yml`）以获得更稳定的 GIS/GPU 依赖安装体验。
- 若需下载 `timm` 预训练权重，建议配置 `HF_TOKEN` 以提升 Hugging Face 下载稳定性。

## 6. 复现流程（最终主线）

### 6.1 strict7 训练数据集重构

```powershell
python scripts/data_prep/v2/03_run_dataset_rebuild_strict7_pipeline.py
```

输出根目录：
- `data/processed/vit_dataset/v2_11block7`

### 6.2 baseline_v5 三种子训练

```powershell
python scripts/dl/formal/01_train_vit_seg_formal.py --config configs/baseline_v5/train_vit_seg_baseline_v5_single.yaml
python scripts/dl/formal/01_train_vit_seg_formal.py --config configs/baseline_v5/train_vit_seg_baseline_v5_seed3407.yaml
python scripts/dl/formal/01_train_vit_seg_formal.py --config configs/baseline_v5/train_vit_seg_baseline_v5_seed2026.yaml
```

### 6.3 三种子测试

```powershell
python scripts/dl/formal/02_eval_vit_seg_formal.py --config configs/baseline_v5/train_vit_seg_baseline_v5_single.yaml --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed42/checkpoints/best.pth
python scripts/dl/formal/02_eval_vit_seg_formal.py --config configs/baseline_v5/train_vit_seg_baseline_v5_seed3407.yaml --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed3407/checkpoints/best.pth
python scripts/dl/formal/02_eval_vit_seg_formal.py --config configs/baseline_v5/train_vit_seg_baseline_v5_seed2026.yaml --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed2026/checkpoints/best.pth
```

### 6.4 多种子聚合

```powershell
python scripts/dl/formal/04_aggregate_multiseed_results.py `
  --configs `
  configs/baseline_v5/train_vit_seg_baseline_v5_single.yaml `
  configs/baseline_v5/train_vit_seg_baseline_v5_seed3407.yaml `
  configs/baseline_v5/train_vit_seg_baseline_v5_seed2026.yaml `
  --output-dir outputs/evaluation_reports/2026-04-22_baseline_v5_strict7
```

### 6.5 构建“全覆盖无缝优先”推理数据集（nosplit）

```powershell
python scripts/data_prep/10_vit_dataset_dry_run_11block_only.py --config configs/dataset_build_v5_inference/dry_run_11block_strict7_fullcoverage_nosplit.yaml
python scripts/data_prep/11_export_vit_dataset_tiles_11block_only.py --config configs/dataset_build_v5_inference/export_tiles_11block_strict7_fullcoverage_nosplit.yaml
```

输出根目录：
- `data/processed/vit_dataset/v2_11block7_inference_full_nosplit`

### 6.6 导出预测切片（05）

```powershell
python scripts/dl/formal/05_export_prediction_rasters.py `
  --config configs/baseline_v5/export_v5_inference_fullcoverage_nosplit.yaml `
  --checkpoint outputs/train_runs/formal/baseline_v5_partial_unfreeze_v2_strict7_seed42/checkpoints/best.pth `
  --splits "train,val,test"
```

### 6.7 拼接全域图（06，max_conf）

```powershell
python scripts/dl/formal/06_mosaic_prediction_tiles.py `
  --run-dir outputs/train_runs/formal/baseline_v5_export_inference_fullcoverage_nosplit_seed42 `
  --splits "train,val,test" `
  --layer pred_all_pixels `
  --mode both `
  --fusion max_conf `
  --overwrite
```

### 6.8 生成 GIS 友好分类栅格（07）

```powershell
python scripts/dl/formal/07_make_gis_ready_classified_raster.py `
  --input outputs/train_runs/formal/baseline_v5_export_inference_fullcoverage_nosplit_seed42/mosaics/mosaic_all_splits_pred_all_pixels_max_conf.tif `
  --output outputs/train_runs/formal/baseline_v5_export_inference_fullcoverage_nosplit_seed42/mosaics/mosaic_all_splits_pred_all_pixels_max_conf_strict7_gis.tif `
  --force-epsg 2326 `
  --overwrite
```

> `--force-epsg 2326` 用于规避 QGIS 中 “No transform available between Hong Kong 1980 Grid System and Custom CRS” 的常见 CRS 识别问题。

## 7. strict7 分类体系

分类体系明细表（含中文名称与颜色）：
- `classification_system_v2_strict7.xlsx`

当前标签定义（GIS 展示）：

| 值 | 类别 | 颜色 (RGB) |
|---|---|---|
| 0 | 未分类/空白 | (255, 255, 255) |
| 1 | 建筑用地 | (228, 26, 28) |
| 2 | 商业用地 | (255, 127, 0) |
| 3 | 工业用地 | (152, 78, 163) |
| 4 | 交通用地 | (255, 217, 47) |
| 5 | 基础设施/公共服务用地 | (55, 126, 184) |
| 6 | 水体 | (31, 120, 180) |
| 7 | 山地/林地 | (51, 160, 44) |

## 8. 注意事项

- 本仓库 `.gitignore` 默认忽略 `data/`、`outputs/`、`analysis_plans/` 等本地产物目录；推送到远程仓库时通常不包含大体量结果文件。
- 训练与推理请优先使用 `best.pth`。
- 全域图中的置信度建议与分类图配合阅读，低置信度区域优先人工复核。
