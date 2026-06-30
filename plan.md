# plan.md

## 功能概述

为模型库的"编辑"和"删除"功能实现完整的后端接口，并对接前端按钮。完成后，用户可在模型列表中点击"编辑"弹出表单修改模型元数据（名称、类型、框架、描述），点击"删除"经确认后删除数据库记录及磁盘文件。

## 技术决策

- 编辑接口使用 `PUT /api/models/{model_id}`，接收 JSON body，仅更新元数据字段，不涉及文件替换
- 删除接口使用 `DELETE /api/models/{model_id}`，删除数据库记录同时清理磁盘上的 weight 和 model_file
- 两个接口均通过 `current_user` 依赖校验所有权（user_id 匹配），防止越权操作
- 前端编辑弹窗复用现有 Element Plus `el-dialog` 风格，与上传弹窗保持一致

## 前置条件

- `POST /api/models/upload` 和 `GET /api/models/list` 接口已就绪
- ORM 模型 `MLModel`（`models/ml_models.py`）字段已确认：id, user_id, model_name, model_type, framework, weight_file_path, model_file_path, description, upload_time, updated_time
- 坐标系约定：本功能无地图操作，不涉及坐标系

---

## Steps

### Step 1：新增编辑请求 Schema

**目标**：在 `schemas/ml_models.py` 中添加 `MLModelUpdateRequest`，定义编辑接口的请求体结构。

**修改文件**：
- `schemas/ml_models.py`（修改）

**具体操作**：
1. 在文件末尾追加 `MLModelUpdateRequest(BaseModel)`，包含 4 个 `Optional` 字段：`model_name`、`model_type`、`framework`、`description`
2. 所有字段设为 `Optional[str] = None`，允许部分更新（PATCH 语义）

**预期效果**：`from schemas.ml_models import MLModelUpdateRequest` 可正常导入，无报错

**验收标准**：
- [x] `python -c "from schemas.ml_models import MLModelUpdateRequest; print('ok')"` 输出 ok
- [x] 字段均为 Optional，不传时为 None

**风险点**：
- ⚠️ `exclude_none=True` 会过滤掉 `None` 值，导致前端无法将字段清空为 null。前后端约定：**若要清空 description，前端传空字符串 `""` 而非 `null`**，空字符串不会被过滤，可正常写入数据库。

---

### Step 2：新增 CRUD 函数（查单条、更新、删除）

**目标**：在 `crud/ml_models.py` 中实现 `get_ml_model_by_id`、`update_ml_model`、`delete_ml_model` 三个异步函数。

**修改文件**：
- `crud/ml_models.py`（修改）

**具体操作**：
1. 新增 `get_ml_model_by_id(db, model_id, user_id) -> Optional[MLModel]`：通过 id + user_id 查单条记录（防止越权），查不到直接 `return None`
2. 新增 `update_ml_model(db, model_id, user_id, **kwargs) -> Optional[MLModel]`：
   - 先调用 `get_ml_model_by_id`；若返回 `None` 则直接 `return None`，**不执行后续 setattr**（防止对 None 调用 setattr 抛 AttributeError）
   - 对 kwargs 中的每个字段逐一 `setattr(record, key, value)`
   - `await db.flush()` + `await db.refresh(record)` 后返回更新后的对象
3. 新增 `delete_ml_model(db, model_id, user_id) -> Optional[tuple[str, str]]`：
   - 先调用 `get_ml_model_by_id`；若返回 `None` 则直接 `return None`
   - **提前提取路径字符串为局部变量**：`w_path = record.weight_file_path`，`m_path = record.model_file_path`
   - 执行 `await db.delete(record)`
   - 返回 `(w_path, m_path)` 这个普通字符串元组（不返回 ORM 对象）
   - **绝对不能在 delete 后再访问 record 的任何属性**，否则触发 `ObjectDeletedError`

**预期效果**：三个函数可被 router 导入并调用，查不到记录时返回 `None`

**验收标准**：
- [x] `from crud.ml_models import get_ml_model_by_id, update_ml_model, delete_ml_model` 无报错
- [x] `get_ml_model_by_id` 在 user_id 不匹配时返回 None（防越权）
- [x] `delete_ml_model` 返回类型为 `tuple[str, str]` 而非 ORM 对象

**风险点**：
- ⚠️ `update_ml_model` 不能直接 `db.commit()`，需由 router 层控制事务（与现有 upload 接口一致）
- ⚠️ `delete_ml_model` 必须在 `db.delete()` 前提取路径，commit 后 ORM 实例进入 detached/deleted 状态，此时访问任何列属性均会抛 `ObjectDeletedError`

---

### Step 3：新增 PUT 和 DELETE 路由端点

**目标**：在 `router/ml_models.py` 中新增 `PUT /api/models/{model_id}` 和 `DELETE /api/models/{model_id}` 两个端点。

**修改文件**：
- `router/ml_models.py`（修改）

**具体操作**：
1. 顶部 import 补充：`from crud.ml_models import update_ml_model, delete_ml_model, get_ml_model_by_id` 和 `from schemas.ml_models import MLModelUpdateRequest`
2. 新增 `PUT /{model_id}` 端点：
   - 接收 `model_id: int`（路径参数）+ `body: MLModelUpdateRequest`（JSON）
   - 调用 `update_ml_model(db, model_id, current_user.id, **body.model_dump(exclude_none=True))`
   - 返回值为 `None` 时返回 `error_response(404, "模型不存在")`
   - 成功后 `await db.commit()`，返回 `success_response(message="更新成功")`
3. 新增 `DELETE /{model_id}` 端点，**执行顺序严格按如下步骤**：
   - Step 3a：调用 `delete_ml_model(db, model_id, current_user.id)`，返回值为 `None` 时返回 `error_response(404, "模型不存在")`
   - Step 3b：此时拿到 `(w_path, m_path)` 字符串元组（路径已在 CRUD 层提前提取，**不再访问 ORM 对象**）
   - Step 3c：`await db.commit()`（数据库记录正式删除）
   - Step 3d：清理磁盘文件，**必须做非空判断再转 Path**：
     ```python
     if w_path:
         _safe_unlink(Path(w_path))
     if m_path:
         _safe_unlink(Path(m_path))
     ```
   - Step 3e：返回 `success_response(message="删除成功")`

**预期效果**：
- `PUT /api/models/1` 传 `{"model_name": "新名称"}` 返回 `{"code": 200, "message": "更新成功"}`
- `DELETE /api/models/1` 返回 `{"code": 200, "message": "删除成功"}`，磁盘文件被清除

**验收标准**：
- [x] FastAPI 文档（`/docs`）出现新的两个路由
- [x] 请求他人的模型 ID 返回 404
- [x] 删除后磁盘对应文件不存在
- [x] 路径为 None 时不抛 TypeError，安全跳过

**风险点**：
- ⚠️ **不得**在 `db.commit()` 之后访问原 ORM record 的任何属性，否则抛 `ObjectDeletedError`；路径字符串必须在 CRUD 层 `db.delete()` 前提取完毕
- ⚠️ `Path(None)` 直接抛 `TypeError`；Step 3d 的 `if w_path / if m_path` 判断不可省略
- ⚠️ 先 commit 再删文件：若文件删除失败，数据库记录已删，文件成为孤儿。可接受（`_safe_unlink` 吞异常），比反过来（文件已删但数据库未删）更安全
- ⚠️ **业务冲突风险**：若该模型正被后台推理任务引用（如 DeepLab 推理进行中），删除其权重文件会导致任务崩溃。当前 plan 不实现此校验（需先确认推理任务表结构），但 Codex 在实现时若发现存在推理任务关联表，**必须停下来报告，不得自行删除**

---

### Step 4：前端新增 API 函数

**目标**：在 `src/api/model.js` 中新增 `updateModel` 和 `deleteModel` 两个函数。

**修改文件**：
- `E:\change_detection\change_detection\src\api\model.js`（修改）

**具体操作**：
1. 追加 `updateModel(id, data)` — `PUT /models/{id}`，body 为 JSON 对象
2. 追加 `deleteModel(id)` — `DELETE /models/{id}`

**预期效果**：`import { updateModel, deleteModel } from '@/api/model'` 可正常使用

**验收标准**：
- [x] 两个函数导出成功，无 lint 报错

**风险点**：
- ⚠️ 无

---

### Step 5：前端对接编辑弹窗和删除确认

**目标**：在 `src/views/modelDatabase/index.vue` 中为"编辑"和"删除"按钮绑定完整交互逻辑。

**修改文件**：
- `E:\change_detection\change_detection\src\views\modelDatabase\index.vue`（修改）

**具体操作**：
1. `<script setup>` 中追加：
   - 导入 `updateModel, deleteModel` 和 `ElMessageBox`
   - 新增 `editDialogVisible ref`、`editForm ref`（含 id、model_name、model_type、framework、description）、`isEditing ref`、`editFormRef ref`
   - 新增 `handleEdit(row)` 函数：将 row 数据填入 editForm，打开编辑弹窗
   - 新增 `handleEditSubmit(formEl)` 函数：校验表单 → 调用 `updateModel` → 成功后关闭弹窗并 `fetchModels()`
   - 新增 `handleDelete(row)` 函数：`ElMessageBox.confirm` 确认 → 调用 `deleteModel` → 成功后 `fetchModels()`
2. 操作列按钮绑定：`@click="handleEdit(scope.row)"` 和 `@click="handleDelete(scope.row)"`（需将 `#default` 改为 `#default="scope"`）
3. 在 `</el-dialog>` 后追加编辑弹窗，结构与上传弹窗一致，但仅含 model_name、model_type、framework、description 四个表单项（无文件上传）

**预期效果**：
- 点击"编辑"：弹出对话框，预填当前行数据，修改后提交成功并刷新列表
- 点击"删除"：弹出确认框，确认后删除并刷新列表

**验收标准**：
- [x] 编辑弹窗预填数据正确
- [x] 编辑提交后列表数据更新
- [x] 删除确认后行消失，取消则不删除
- [x] 操作越权（非本人模型）时前端显示错误提示

**风险点**：
- ⚠️ `el-table` 的 `#default` 插槽若不传 scope 参数则无法访问 row，必须改为 `#default="scope"` 并使用 `scope.row`
- ℹ️ 无需对空描述做特殊处理：Element Plus 的 `el-input` 在用户清空内容后，`v-model` 绑定值天然为空字符串 `""`，与 Step 1 约定的清空规则天然契合，不需要额外的 null → "" 转换逻辑

---

## 整体风险清单

| 风险 | 影响 | 缓解方案 |
|------|------|----------|
| 删除后磁盘文件清理失败 | 文件孤儿占用存储 | `_safe_unlink` 吞异常，数据库已删即视为成功；后续可加定期清理任务 |
| 编辑时 model_type 传入非法值 | 数据库存入脏数据 | 后端可在 schema 加 `Literal` 类型约束，与前端 options 保持一致 |
| 并发删除同一条记录 | 第二次返回 404 | 已通过先查后删处理，属于正常行为 |
| 前端编辑弹窗 scope 未绑定 | 点击编辑无法获取 row 数据 | Step 5 中明确要求改 `#default="scope"` |

---

## 注意事项（给 Codex 的特别提示）

- 每次只执行一个 Step，执行完等待 Claude Code 审查后再继续
- 每个 Step 修改严格控制在 1~3 个核心文件内
- 不要自行扩展功能（如文件替换、批量删除），严格按 plan 执行
- 遇到 plan 未覆盖的情况，停下来报告，不要自行决策
- Step 2 的 `delete_ml_model` 返回值为 `tuple[str, str]`（路径字符串），**不是 ORM 对象**；Step 3 路由层直接解包使用，commit 后不再访问任何 ORM 属性
- Step 3 清理磁盘文件前必须做非空判断（`if w_path: _safe_unlink(Path(w_path))`），`Path(None)` 会直接抛 `TypeError`
- 前端若需清空 description，传空字符串 `""` 而非 `null`，否则被 `exclude_none=True` 过滤后不会更新数据库
- Step 5 中编辑表单校验规则复用 `modelFormRules` 中 model_name、model_type、framework、description 对应的规则
