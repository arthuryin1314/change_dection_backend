# plan.md — 上传模型 API

## 功能概述

实现 `POST /api/models/upload` 接口，接收前端 multipart/form-data 请求（含 `model_name`、`model_type`、`framework`、`weight` 文件、`model_file` 文件、`description` 六个字段），验证 JWT 身份后将文件分块写入磁盘，元数据写入新建的 `model_info` 表，返回创建结果。

## 技术决策

- **文件接收**：FastAPI `File` + `UploadFile`，`Form` 接收文本字段（multipart/form-data 不能用 JSON Body）
- **文件写入**：分块读取（8192 字节/块），防止大模型权重文件（数百 MB）一次性 `read()` 导致 OOM
- **文件命名**：只取客户端文件名的后缀，以 `{uuid4().hex}{suffix}` 生成纯净磁盘文件名；原始文件名仅存入数据库，不参与路径拼接，防止目录遍历攻击
- **文件存储**：`weight` 文件存 `uploads/weights/{user_id}/`，`model_file` 存 `uploads/model_files/{user_id}/`，与现有 `uploads/images/` 同级
- **数据库**：新增 `model_info` 表，复用现有 `AsyncSession` + `get_db` 依赖
- **认证**：复用现有 `get_current_user` 依赖，与其他受保护路由一致

## 前置条件

- 依赖的接口：无（全新）
- 认证工具 `utils/get_user_by_token.py` 中的 `get_current_user` 已就绪
- 响应工具 `utils/response.py` 中的 `success_response` / `error_response` 已就绪
- 坐标系：本模块无地理数据，不涉及坐标系转换

---

## Steps

### Step 1：新建 ORM 模型 `models/ml_models.py`

**目标**：在数据库定义 `model_info` 表，字段与前端 form 对应

**修改文件**：
- `models/ml_models.py`（新增）

**具体操作**：
1. 创建 `MLModel` 类继承 `Base`，表名 `model_info`
2. 字段：
   - `id`（Integer，主键，自增）
   - `user_id`（BigInteger，外键 → `user_info.id`，`ondelete="CASCADE"`）
   - `model_name`（String 255，nullable=False）
   - `model_type`（String 50，枚举值：地物识别 / 变化检测 / 目标提取）
   - `framework`（String 50，枚举值：PyTorch / TensorFlow / PaddlePaddle / ONNX）
   - `weight_path`（Text，磁盘存储路径）
   - `weight_original_name`（String 255，客户端原始文件名，仅供展示）
   - `model_file_path`（Text，磁盘存储路径）
   - `model_file_original_name`（String 255，客户端原始文件名，仅供展示）
   - `description`（Text）
   - `upload_time`（DateTime，`default=datetime.now`）
3. 与 `User` 表建立外键关联，`ondelete="CASCADE"`

**预期效果**：文件无报错，可被其他模块 import

**验收标准**：
- [x] `from models.ml_models import MLModel` 无 ImportError
- [x] 字段与前端 form 一一对应，原始文件名字段存在

**风险点**：
- ⚠️ 数据库表需要在 Step 6 通过 `create_all` 创建，ORM 模型必须在 Step 6 之前定义完毕

---

### Step 2：新建 Schema `schemas/ml_models.py`

**目标**：定义上传响应的 Pydantic Schema（请求字段用 `Form` + `File` 在 router 直接声明，无需 BaseModel）

**修改文件**：
- `schemas/ml_models.py`（新增）

**具体操作**：
1. 定义 `MLModelResponse(BaseModel)`，字段：
   - `id`、`user_id`、`model_name`、`model_type`、`framework`
   - `weight_path`、`weight_original_name`
   - `model_file_path`、`model_file_original_name`
   - `description`、`upload_time`
2. 加 `model_config = ConfigDict(from_attributes=True)`，与 `ImageResponse` 保持一致

**预期效果**：response schema 可被 router import

**验收标准**：
- [x] `from schemas.ml_models import MLModelResponse` 无 ImportError

---

### Step 3：新建 CRUD `crud/ml_models.py`

**目标**：封装数据库写入逻辑，与 `crud/images.py` 风格一致

**修改文件**：
- `crud/ml_models.py`（新增）

**具体操作**：
1. 实现函数签名：
   ```python
   async def create_ml_model(
       db: AsyncSession,
       user_id: int,
       model_name: str,
       model_type: str,
       framework: str,
       weight_path: str,
       weight_original_name: str,
       model_file_path: str,
       model_file_original_name: str,
       description: str,
   ) -> MLModel
   ```
2. 函数体：实例化 `MLModel`，`db.add()`，`await db.flush()`，`await db.refresh()`，返回对象
3. 不在此处 `commit`（由 router 层统一 commit，与现有模式一致）

**预期效果**：函数可被 router 调用

**验收标准**：
- [x] `from crud.ml_models import create_ml_model` 无 ImportError

---

### Step 4.1：新建文件存储工具 `utils/file_storage.py`

**目标**：封装安全的异步分块文件写入函数，与路由逻辑解耦

**修改文件**：
- `utils/file_storage.py`（新增）

**具体操作**：
1. 实现 `async def save_upload_file(upload_file: UploadFile, dest_dir: Path) -> Path`
2. 安全文件名：只取 `upload_file.filename` 的后缀（`Path(upload_file.filename).suffix`），生成 `{uuid4().hex}{suffix}` 作为磁盘文件名，**不拼接原始文件名**
3. 创建目录：`dest_dir.mkdir(parents=True, exist_ok=True)`
4. 分块写入（8192 字节/块）：
   ```python
   async with aiofiles.open(save_path, 'wb') as f:
       while chunk := await upload_file.read(8192):
           await f.write(chunk)
   ```
5. 返回最终保存的绝对路径 `save_path`
6. 若写入过程中抛出异常，捕获后执行 `save_path.unlink(missing_ok=True)` 清理半写文件，再 re-raise

**预期效果**：调用后磁盘出现以 UUID 命名的文件，原始文件名不出现在路径中

**验收标准**：
- [x] `from utils.file_storage import save_upload_file` 无 ImportError
- [x] 写入 100MB 文件时进程内存无明显峰值
- [x] 传入含 `../` 的文件名不会在目标目录之外生成文件

**风险点**：
- ⚠️ `aiofiles` 需确认已在 `requirements.txt` 中，否则在此步骤前补充

---

### Step 4.2：新建路由 `router/ml_models.py`

**目标**：实现 `POST /api/models/upload` 端点，调用 Step 4.1 的工具完成文件保存，调用 Step 3 的 CRUD 完成 DB 写入

**修改文件**：
- `router/ml_models.py`（新增）

**具体操作**：
1. `router = APIRouter(prefix='/api/models', tags=['models'])`
2. 定义 `POST /upload` 端点，参数列表：
   ```python
   model_name: str = Form(...)
   model_type: str = Form(...)
   framework: str = Form(...)
   weight: UploadFile = File(...)
   model_file: UploadFile = File(...)
   description: str = Form(...)
   current_user = Depends(get_current_user)
   db: AsyncSession = Depends(get_db)
   ```
3. 文件扩展名校验（使用 `Path(f.filename).suffix.lower()`）：
   - `weight` 必须在 `{.pth, .pt, .h5, .onnx, .pdparams}`
   - `model_file` 必须在 `{.py, .zip}`
   - 不合法时直接 `return error_response(400, "不支持的文件类型")`
4. 调用 `save_upload_file(weight, Path("uploads/weights") / str(current_user.id))` 保存权重文件，记录返回路径
5. 在 `try` 块中调用 `save_upload_file(model_file, ...)` 保存模型文件；若此步失败，在 `except` 中清理第 4 步已保存的权重文件
6. 在同一 `try` 块中调用 `create_ml_model(db, ..., weight_original_name=weight.filename, model_file_original_name=model_file.filename, ...)`
7. `await db.commit()`
8. 返回 `success_response(message="上传成功", data=MLModelResponse.model_validate(db_record).model_dump())`
9. `except` 块：`await db.rollback()`，清理已写入的两个磁盘文件（`Path(path).unlink(missing_ok=True)`），`return error_response(500, "上传失败")`

**预期效果**：
- 合法请求 → `uploads/weights/{user_id}/` 出现 UUID 命名文件，`model_info` 表新增记录，返回 200
- 非法扩展名 → 返回 400，无文件写入
- DB 失败 → 返回 500，磁盘文件已清理

**验收标准**：
- [ ] Swagger UI 发送合法请求，返回 200 + 模型数据
- [x] 发送非法文件类型，返回 400
- [ ] 不携带 token，返回 401
- [x] `uploads/weights/{user_id}/` 目录下文件名为纯 UUID 格式，无原始文件名路径字符

**风险点**：
- ⚠️ multipart/form-data + `UploadFile` 不能与 Pydantic BaseModel 混用，文本字段必须用 `Form(...)`
- ⚠️ 两个文件的保存顺序决定了异常回滚的清理顺序，须严格先保存权重、再保存模型文件

---

### Step 5：注册路由到 `main.py`

**目标**：将新 router 挂载到 FastAPI app

**修改文件**：
- `main.py`（修改）

**具体操作**：
1. 在 import 区加 `from router import ml_models`
2. 在 `app.include_router(segment.router)` 之后加 `app.include_router(ml_models.router)`

**预期效果**：`http://localhost:8000/docs` 中出现 `models` 标签和 `/api/models/upload` 端点

**验收标准**：
- [x] 应用启动无报错
- [x] Swagger UI 能看到 `/api/models/upload` POST 端点

---

### Step 6：建表（DDL）

**目标**：在数据库中创建 `model_info` 表

**修改文件**：
- `main.py`（修改 lifespan）

**具体操作**：
1. 在 lifespan 函数顶部补充以下导入和建表逻辑（在 `load_model` 之前执行）：
   ```python
   from config.db_config import async_engine
   from models.Base import Base
   from models import ml_models  # 注册到 metadata

   async with async_engine.begin() as conn:
       await conn.run_sync(Base.metadata.create_all)
   ```
2. `async with async_engine.begin()` 是标准的 asyncpg 异步写法，`run_sync` 在连接线程中执行同步 DDL，不阻塞事件循环

**预期效果**：首次启动后 `model_info` 表自动创建，重复启动不报错

**验收标准**：
- [x] 数据库中存在 `model_info` 表，字段与 ORM 定义一致（含 `weight_original_name`、`model_file_original_name`）
- [x] 重复启动无 `DuplicateTable` 报错

**风险点**：
- ⚠️ 若后续引入 Alembic 迁移工具，需移除此处的 `create_all` 改用 Alembic 管理

---

## 整体风险清单

| 风险 | 影响 | 缓解方案 |
|------|------|----------|
| 大文件一次性 `read()` 导致 OOM | FastAPI 进程崩溃 | `utils/file_storage.py` 强制 8192 字节分块读写 |
| 客户端文件名含 `../` 路径穿越 | 覆盖系统文件 | 磁盘文件名仅用 `uuid4().hex + suffix`，原始名只存 DB |
| multipart/form-data 与 JSON Body 混用 | 422 Unprocessable Entity | 文本字段全部用 `Form(...)`，不用 Pydantic BaseModel 作请求体 |
| 权重文件写入成功但模型文件或 DB 写入失败 | 磁盘脏文件残留 | 异常时 `Path(path).unlink(missing_ok=True)` 清理所有已保存文件 |
| `aiofiles` 未安装 | ImportError | Step 4.1 前检查 `requirements.txt`，如缺失则添加 |
| `model_info` 表不存在 | 500 启动或请求报错 | Step 6 在 lifespan 中 `create_all` 自动建表 |

---

## 注意事项（给执行者的提示）

- 每次只执行一个 Step，执行完等待审查后再继续
- 每个 Step 修改文件严格控制在 1~3 个以内；Step 4 已拆分为 4.1 和 4.2，不得合并执行
- 不要自行添加模型列表、删除等功能，严格按 plan 执行
- `Form` 参数和 `File` 参数必须同时出现在函数签名，不能用 `BaseModel` 包裹
- `db.commit()` 只在 router 层调用，CRUD 层只 `flush` + `refresh`
- 文件存储路径使用 `pathlib.Path` 操作，禁止字符串拼接路径
