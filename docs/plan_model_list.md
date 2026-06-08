# plan.md — 实现模型列表接口

## 功能概述

实现 `GET /api/models/list` 接口，返回当前登录用户上传的模型分页列表，支持按模型名称关键字搜索。前端模型仓库页面（`modelDatabase/index.vue`）调用此接口替换现有的静态 tableData，表格展示模型名称、类型、框架、权重文件名、模型文件名、描述和更新日期。

---

## 技术决策

- **CRUD 层**：使用 SQLAlchemy 异步 `select` + `where` + `offset/limit` 实现分页查询，`func.count` 获取总数，避免两次全表扫描。
- **字段映射**：`weight_file_path` / `model_file_path` 为磁盘完整路径，列表接口通过 `Path(...).name` 提取文件名返回，保持前端表格字段名（`weight`、`model_file`）不变。
- **日期格式**：`updated_time`（`datetime`）序列化为 `YYYY-MM-DD` 字符串字段 `update_date`，匹配前端静态数据的格式。
- **分页参数**：接受 `page`（从 1 开始）和 `pageSize`，转换为 SQL `offset = (page-1)*pageSize`。

---

## 前置条件

- `POST /api/models/upload` 已就绪（Steps 1-6 已完成）
- `MLModel` ORM、`get_current_user`、`success_response` 均已存在
- 坐标系约定：本接口无地图数据，不涉及坐标系

---

## Steps

### Step 1：在 `schemas/ml_models.py` 中添加列表响应 Schema

**目标**：定义列表单条记录 `MLModelListItem` 和分页响应 `MLModelListResponse`，将路径字段映射为文件名、时间映射为日期字符串。

**修改文件**：
- `schemas/ml_models.py`（修改）

**具体操作**：
1. 在文件顶部补充导入：`from pathlib import Path`（若已有则跳过）
2. 在文件末尾新增 `MLModelListItem(BaseModel)`，加 `model_config = ConfigDict(from_attributes=True)`，字段包括：
   - `id: int`
   - `model_name: str`
   - `model_type: str`
   - `framework: str`
   - `weight: str`（文件名）
   - `model_file: str`（文件名）
   - `description: Optional[str]`
   - `update_date: Optional[str]`（`YYYY-MM-DD`）
3. 在 `MLModelListItem` 上添加 `@model_validator(mode="before") @classmethod` 进行字段转换（使用 Pydantic V2 方式，**不使用** `from_orm`）：
   - 若输入为 ORM 对象（非 dict），先调用 `model.__dict__` 或直接用属性读取
   - 推荐方式：在 router/Step 3 中构造 item 时，手动传入转换后的字段值，即 `MLModelListItem(id=obj.id, ..., weight=Path(obj.weight_file_path).name, model_file=Path(obj.model_file_path).name, update_date=obj.updated_time.strftime("%Y-%m-%d") if obj.updated_time else None)`，保持 Schema 纯净无转换逻辑
4. 新增 `MLModelListResponse(BaseModel)`，字段：
   - `items: list[MLModelListItem]`
   - `total: int`
   - `page: int`
   - `pageSize: int`

**预期效果**：Schema 可正常 import，在 router 中通过关键字参数构造 `MLModelListItem`，返回含文件名和日期字符串的对象。不调用任何 V1 风格的 `from_orm`。

**验收标准**：
- [x] `from schemas.ml_models import MLModelListItem, MLModelListResponse` 无报错
- [x] `MLModelListItem` 包含 `weight`、`model_file`、`update_date` 字段，无 `weight_file_path` / `model_file_path`

**风险点**：
- ⚠️ `weight_file_path` 为空字符串时 `Path("").name` 返回 `""`，需确保上传时路径非空（现有逻辑已保证）

---

### Step 2：在 `crud/ml_models.py` 中添加分页查询函数

**目标**：实现 `get_ml_models` 异步函数，按 `user_id` 过滤，支持 `keyword` 模糊搜索，返回 `(items, total)` 元组。

**修改文件**：
- `crud/ml_models.py`（修改）

**具体操作**：
1. 新增导入：`from sqlalchemy import func, select`（`select` 已存在则跳过）
2. 添加函数签名：
   ```python
   async def get_ml_models(
       db: AsyncSession,
       user_id: int,
       page: int,
       page_size: int,
       keyword: str = "",
   ) -> tuple[list[MLModel], int]:
   ```
3. 构建基础查询：`stmt = select(MLModel).where(MLModel.user_id == user_id)`
4. 若 `keyword` 非空，追加 `.where(MLModel.model_name.ilike(f"%{keyword}%"))`
5. 执行 count 查询：`total = await db.scalar(select(func.count()).select_from(stmt.subquery()))`
6. 执行分页查询：`stmt = stmt.order_by(MLModel.upload_time.desc()).offset((page-1)*page_size).limit(page_size)`，`result = await db.execute(stmt)`，`items = result.scalars().all()`
7. 返回 `(list(items), total)`

**预期效果**：函数可正确返回分页数据和总数；`keyword=""` 时返回该用户全部记录。

**验收标准**：
- [x] `from crud.ml_models import get_ml_models` 无报错
- [x] 函数签名和返回类型正确

**风险点**：
- ⚠️ `page` 从 1 开始，offset 计算为 `(page-1)*page_size`，若前端传 0 会导致负 offset，路由层需做参数校验（`ge=1`）

---

### Step 3：在 `router/ml_models.py` 中添加 GET /list 端点

**目标**：注册 `GET /api/models/list` 路由，接收分页和搜索参数，调用 CRUD 函数，返回标准分页响应。

**修改文件**：
- `router/ml_models.py`（修改）

**具体操作**：
1. 在文件顶部补充导入：`from schemas.ml_models import MLModelListResponse, MLModelListItem` 和 `from crud.ml_models import get_ml_models`
2. 导入 `Query`：`from fastapi import Query`（与现有 import 合并）
3. 在 `upload_model` 端点之后新增：
   ```python
   @router.get("/list", summary="获取模型列表")
   async def list_models(
       page: int = Query(default=1, ge=1),
       pageSize: int = Query(default=10, ge=1, le=100),
       keyword: str = Query(default=""),
       current_user=Depends(get_current_user),
       db: AsyncSession = Depends(get_db),
   ):
   ```
4. 函数体：
   - 调用 `items, total = await get_ml_models(db, current_user.id, page, pageSize, keyword)`
   - 构建 `MLModelListResponse`，手动转换每条记录（Pydantic V2，**不使用** `from_orm`）：
     ```python
     MLModelListResponse(
         items=[
             MLModelListItem(
                 id=i.id,
                 model_name=i.model_name,
                 model_type=i.model_type,
                 framework=i.framework,
                 weight=Path(i.weight_file_path).name,
                 model_file=Path(i.model_file_path).name,
                 description=i.description,
                 update_date=i.updated_time.strftime("%Y-%m-%d") if i.updated_time else None,
             )
             for i in items
         ],
         total=total,
         page=page,
         pageSize=pageSize,
     )
     ```
   - 返回 `success_response(data=response.model_dump())`
5. 异常用 `try/except` 包裹，`logger.exception` 记录，返回 `error_response(500, "获取模型列表失败")`

**预期效果**：
- `GET /api/models/list?page=1&pageSize=10` 返回 `{"code": 200, "data": {"items": [...], "total": N, "page": 1, "pageSize": 10}}`
- 未登录或 token 失效时返回 401

**验收标准**：
- [x] FastAPI `/docs` 中出现 `GET /api/models/list` 端点
- [x] 携带有效 token 请求返回 200 和正确 JSON 结构
- [x] `keyword` 过滤生效，只返回名称包含关键字的记录
- [x] 只返回当前登录用户的模型，不返回其他用户的数据

**风险点**：
- ⚠️ `MLModelListItem.from_orm` 在 `weight_file_path` 为绝对路径时依赖 `Path.name`，Windows 路径（反斜杠）需确认 `Path` 可正确解析（Python `pathlib.Path` 在 Windows 上处理反斜杠正常）

---

## 整体风险清单

| 风险 | 影响 | 缓解方案 |
|------|------|----------|
| 前端 `tableData` 仍为静态数据，未调用 `getModelList` | 接口实现后页面不更新 | 本 plan 完成后需前端 `onMounted` 调用接口并替换 `tableData` |
| `page=0` 导致负 offset | 查询报错 | Query 参数加 `ge=1` 约束 |
| 数据库 `updated_time` 为 `None`（旧记录未更新） | `update_date` 显示 `null` | Schema 中 `Optional[str]`，前端显示兜底为 `—` |
| 大量模型时全表 count 性能 | 响应慢 | 当前数据量小可接受；未来可加 `user_id` 索引 |

---

## 注意事项（给 Codex 的特别提示）

- 每次只执行一个 Step，执行完等待 Claude Code 审查后再继续。
- 每个 Step 修改范围严格控制在 1 个文件内。
- Step 1 和 Step 2 无依赖关系，可并行执行；Step 3 依赖 Step 1 和 Step 2 完成。
- 不要修改现有的 `upload_model` 端点和 `MLModelResponse` schema。
- 不要自行扩展为「管理员可见全部模型」等功能，严格按 `user_id` 过滤。
- 遇到 plan 中未覆盖的情况，停下来报告，不要自行决策。
