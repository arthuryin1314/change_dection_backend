# #54：无 Torch 后端验收与运行

本 ticket 将 Web 后端与进程内 DeepLab 推理解耦。新建推理请求暂时不可用，历史识别、历史变化结果及其已有栅格仍可读取。真实模型推理要等后续 Worker 接入。

## 运行环境

本机已创建独立 conda 环境 `change-detection-web`（Python 3.11），路径为 `E:\anaconda3\envs\change-detection-web`。没有安装 `torch`、`torchvision`；原 `deeplabv3` 环境未修改。

从后端目录运行：

```powershell
cd E:\change_detection\change_detection_backend
conda run --no-capture-output -n change-detection-web python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

继续使用后端原有 `.env` 中的数据库、JWT 和 GeoServer 配置。若已有进程占用 8000 端口，应先停止旧后端。启动成功后访问 `http://127.0.0.1:8000/docs`。

其他机器可在后端目录用 `conda env create -f environment-web.yml` 创建环境，再用 `conda run -n change-detection-web python -m pip install -r requirements-dev.txt` 安装测试依赖。无需安装 CUDA。

依赖清单同时约束了 bcrypt 的兼容版本范围；新环境已验证使用 bcrypt 4.0.1。原先不设上限会安装 bcrypt 5，导致现有 passlib 1.7.4 在普通密码验证时失败。

## 一条命令完成验收

```powershell
cd E:\change_detection\change_detection_backend
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
conda run --no-capture-output -n change-detection-web python scripts/verify_torch_free_backend.py
```

脚本执行以下检查：

1. 确认 Torch 和 torchvision 均未安装。
2. 使用 `.env`（或当前进程 `DATABASE_URL`）中的数据库账号，创建随机命名的隔离数据库并启用 PostGIS。
3. 在独立文件目录中运行真实 FastAPI lifespan，验证启动、OpenAPI 和正常关闭。
4. 只向隔离库应用仓库迁移，运行全部测试，包括两个 PostgreSQL 集成测试文件及历史迁移测试。
5. 删除本轮隔离库与临时文件，以 pytest 的退出码结束。

账号需要创建数据库和启用 PostGIS 的权限，PostgreSQL 需要支持 `DROP DATABASE ... WITH (FORCE)`（13 或以上）。本机已通过此流程。脚本不迁移或清空 `.env` 指向的业务数据库；异常时也会尝试清理隔离库。强制终止脚本可能留下以 `torch_free_check_` 开头的临时库。

如只检查本 ticket 的 HTTP 行为、应用导入和登录依赖，可运行：

```powershell
conda run --no-capture-output -n change-detection-web python -m pytest tests/test_backend_without_torch.py -q -p no:cacheprovider
```

## 手动验收

启动新后端，使用原有账号登录。在 Swagger 或浏览器开发者工具中检查 HTTP 状态和 JSON 响应。以下 POST 均使用合法参数和已登录身份验收。关闭的新建变化及重试入口可能在请求校验完成前就返回 503，因此不能用这两个入口判断请求参数是否合法。

| 请求 | 预期 |
| --- | --- |
| `POST /api/segment` | HTTP 503；`data.error_code = INFERENCE_UNAVAILABLE`；提示“推理服务暂不可用” |
| `POST /api/identification-results` | 同上；不会创建新的识别任务 |
| `POST /api/change-results` | HTTP 503；`data.error_code = CHANGE_DETECTION_DISABLED`；提示“第一阶段暂不支持新建变化检测任务” |
| `POST /api/change-results/{request_id}/retry` | 同上，不能通过重试绕过关闭的新建入口 |
| `GET /api/identification-results/history` | 返回自己的历史识别列表 |
| `GET /api/identification-results/history/{result_id}` | 已有结果详情、面积、影像及模型快照正常返回 |
| `GET /api/change-results/history` | 返回自己的历史变化列表 |
| `GET /api/change-results/history/{result_id}` | 已有矩阵、统计范围及来源信息正常返回 |

响应仍沿用 `{code, message, data}` 结构，例如：

```json
{"code":503,"message":"推理服务暂不可用","data":{"error_code":"INFERENCE_UNAVAILABLE"}}
```

手动打开一条已有历史识别图和变化结果报告，确认仍能展示。历史文件本身缺失时，接口仍按原来的缺失状态处理；本 ticket 不补算或恢复丢失的历史文件。

## 权重与验证记录

删除后端的 `weights/` 和 `model_data/` 前，已验证以下归档：

`模型库/地物识别模型库/地物识别模型库/0. DeeplabV3-plus/production_weights/best_epoch_weights.pth`

SHA256：`7bd7e4e82257e14c999ac28b1d3d52e307dc465b29461843910d1a9e747c474e`。

后端旧模型源码及其专用测试已移除。数据库中的模型权重路径字段、历史上传权重和历史模型身份保持原样；这部分迁移属于后续 ticket。测试以假运行时验证保留下来的生成、取消、结果复用和编排行为，不执行真实模型推理。

2026-10-24 在 Windows 的新环境中执行完整验收：288 项测试通过，无跳过；真实应用启动和关闭通过。类型检查覆盖运行时接口和生成服务，并分别按 Windows、Linux 平台检查通过。`pip check` 未发现依赖冲突。第三方库的弃用提示仍存在，未为消除提示而扩大本 ticket 的修改范围。
