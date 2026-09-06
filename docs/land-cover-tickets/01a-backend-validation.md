# 1a 后端真实数据验收记录（2026-09-06）

## POST → GET → render

- 数据：数据库影像 `image_id=28`（`2025_cut.tif`），模型 `model_id=8`，用户 `user_id=1`。
- 源网格：26,426 × 28,311，3 波段 `uint8`，EPSG:4528，分辨率约 0.8 m，NoData=0，源块 128 × 128。
- 设备：NVIDIA GeForce RTX 5060 Ti；`deeplabv3` Conda 环境；真实 PyTorch 权重与 PostgreSQL，未替换推理器。
- 命令：`python scripts/verify_identification_flow.py --user-id 1 --image-id 28 --model-id 8 --timeout 900`。
- 结果：POST 返回耗时 0.089 s；稳定结果 ID `7713703d946e480fa11a2229c661b184`；轮询 GET 达到 `SUCCEEDED` 后，render 返回 256 × 256 RGBA PNG；端到端 165.972 s。
- 数据库生成指标：总计 162.988 s；源读取 7.543 s；模型推理 136.744 s；压缩写入及完整落盘校验 12.030 s；模型装载 0.617 s；5106/5106 有效 tile；峰值 RSS 4,043,915,264 B；峰值 GPU 分配 468,366,336 B。

完整像素校验计入 `compressed_write_seconds`；已存在结果的复用检查只验证文件与网格元数据，不重复扫描全部像素。

## 320612DOM 整景面积

- 数据：`E:\change_detection\局科研数据\2025\影像\32061201通州区\320612DOM.tif`，136,067 × 55,649，EPSG:4528，0.8 m。
- 命令：`python scripts/measure_valid_area.py <path> --tile-size 2048`。
- 方法：2048 像元窗口顺序读取三波段 masked 数据；NoData 独立排除；每个窗口调用共享 `pixel_area_m2`，不物化整景面积数组。
- 结果：有效像元 2,465,692,559；地表面积 1,577,623,151.650362 m²（157,762.315165 ha）；耗时 81.012 s。
- 东西边缘单像元面积分别为 0.639945534792704 m² 和 0.6395569505460511 m²；跨 EPSG 推荐适用范围的东缘仍可计算。

## 自动化验证

- 全量：145 passed，1 skipped；跳过项是默认关闭的真实 PostgreSQL 集成测试。
- 显式 PostgreSQL 集成：1 passed（2.46 s）。
- 限制：当前生成契约要求源影像至少三波段，前三波段为 `uint8`，存在 CRS；旋转或错切面积网格在本任务中明确不支持。前端仓的现有识别页面接线需单独完成跨仓验收。
