# plan_edit_with_files.md

## 功能概述

在已完成的编辑/删除后端接口基础上，升级编辑功能以支持文件替换。用户在编辑弹窗中除了可以修改元数据（名称、类型、框架、描述），还能看到当前文件名并可选择性替换权重文件和/或模型文件，一次提交完成所有更新。

## 技术决策

- 方案 A：将 `PUT /api/models/{model_id}` 从 JSON body 改为 `multipart/form-data`，文本字段改为 `Form(None)`，文件字段为可选 `UploadFile = File(None)`
- 文件替换策略：先保存新文件 → commit 更新数据库 → 删除旧文件；异常时回滚数据库并清理已保存的新文件
- 前端 FormData：直接传 FormData 对象，依赖 axios 自动设置 Content-Type（含 boundary），不手动指定 header

## 已完成的前置工作

- Step 1 ✅ `schemas/ml_models.py`：`MLModelUpdateRequest` 已存在（无需修改，改为 Form 后此 Schema 不再用作请求体，但字段名仍作参考）
- Step 2 ✅ `crud/ml_models.py`：`update_ml_model(**kwargs)` 的 kwargs 机制天然支持传入 `weight_file_path` / `model_file_path`，无需修改
- DELETE 端点 ✅ 已完成，不在本次改动范围内

---

## Steps

### Step 1：改造 PUT 端点为 multipart/form-data

**目标**：修改 `router/ml_models.py` 中已有的 `PUT /{model_id}` 端点，从 JSON body 改为 multipart/form-data，支持可选文件替换。

**修改文件**：
- `router/ml_models.py`（修改）

**具体操作**：

1. 在文件顶部常量区新增文件大小上限（紧跟 `MODEL_FILE_EXTENSIONS` 之后）：
   ```python
   MAX_WEIGHT_SIZE = 1 * 1024 * 1024 * 1024   # 1 GB，遥感大模型权重上限
   MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024     # 100 MB，模型脚本/zip 上限
   ```

2. 将 PUT 端点函数签名替换为 Form + File 参数（删除原有 `body: MLModelUpdateRequest`）：
   ```python
   model_name: Optional[str] = Form(None),
   model_type: Optional[str] = Form(None),
   framework:  Optional[str] = Form(None),
   description: Optional[str] = Form(None),
   weight: Optional[UploadFile] = File(None),
   model_file: Optional[UploadFile] = File(None),
   ```

2. 函数体按以下**严格顺序**实现：

   **2a. 查记录，提前提取旧文件路径**
   ```python
   record = await get_ml_model_by_id(db, model_id, current_user.id)
   if record is None:
       return error_response(404, "模型不存在")

   # 仅在用户传了新文件时才需要清理旧文件，提前用局部变量保存
   old_w_path = record.weight_file_path if weight else None
   old_m_path = record.model_file_path if model_file else None
   ```
   > ⚠️ 必须在此处提取，不能在 commit 后再访问 ORM 属性（ObjectDeletedError 风险）

   **2b. 构建 update_kwargs，初始化新文件路径局部变量**
   ```python
   update_kwargs = {}
   if model_name is not None: update_kwargs["model_name"] = model_name
   if model_type is not None: update_kwargs["model_type"] = model_type
   if framework  is not None: update_kwargs["framework"]  = framework
   if description is not None: update_kwargs["description"] = description
   new_w_path: Optional[Path] = None   # 用于异常时回滚清理
   new_m_path: Optional[Path] = None
   ```

   **2c. 在 try 块内保存新文件、提交、清理旧文件**
   ```python
   try:
       if weight:
           if _get_suffix(weight) not in WEIGHT_EXTENSIONS:
               return error_response(400, "不支持的权重文件类型")
           if weight.size and weight.size > MAX_WEIGHT_SIZE:
               return error_response(400, "权重文件超过大小限制（最大 1 GB）")
       if model_file:
           if _get_suffix(model_file) not in MODEL_FILE_EXTENSIONS:
               return error_response(400, "不支持的模型文件类型")
           if model_file.size and model_file.size > MAX_MODEL_FILE_SIZE:
               return error_response(400, "模型文件超过大小限制（最大 100 MB）")
   ```
   > ⚠️ 大小校验必须在 `save_upload_file` 前执行；`UploadFile.size` 在某些客户端不发送 Content-Length 时可能为 None，故用 `and` 保护，None 时跳过校验（可接受）

   继续 try 块（接上方校验后）：
   ```python
       if weight:
           new_w_path = await save_upload_file(
               weight, BASE_DIR / "uploads" / "weights" / str(current_user.id)
           )
           update_kwargs["weight_file_path"] = str(new_w_path)

       if model_file:
           if _get_suffix(model_file) not in MODEL_FILE_EXTENSIONS:
               return error_response(400, "不支持的模型文件类型")
           new_m_path = await save_upload_file(
               model_file, BASE_DIR / "uploads" / "model_files" / str(current_user.id)
           )
           update_kwargs["model_file_path"] = str(new_m_path)

       if not update_kwargs:
           return success_response(message="无改动")

       await update_ml_model(db, model_id, current_user.id, **update_kwargs)
       await db.commit()

       # commit 成功后才删旧文件
       if old_w_path: _safe_unlink(Path(old_w_path))
       if old_m_path: _safe_unlink(Path(old_m_path))

       return success_response(message="更新成功")

   except Exception as e:
       logger.exception("更新模型失败: %s", e)
       await db.rollback()
       # 回滚时清理本次已保存的新文件
       _safe_unlink(new_w_path)
       _safe_unlink(new_m_path)
       return error_response(500, "更新失败")
   ```

**预期效果**：
- 只传文本字段 → 仅更新元数据，文件不变
- 只传新文件 → 仅替换文件路径，文本字段不变
- 同时传 → 一次请求完成全部更新

**验收标准**：
- [x] `/docs` 中 PUT 端点显示为 multipart/form-data 参数
- [x] 只传 model_name，数据库 weight_file_path 不变
- [x] 传新 weight 文件后，数据库 weight_file_path 更新，旧文件从磁盘消失
- [x] 传入他人 model_id 返回 404
- [x] 文件扩展名非法返回 400

**风险点**：
- ⚠️ 扩展名校验失败时直接 return，此时 new_w_path / new_m_path 均为 None，无需 rollback，`_safe_unlink(None)` 也安全
- ⚠️ `old_w_path` 必须在 `save_upload_file` 前提取，不能在 commit 后访问 ORM 对象

---

### Step 2：修改前端 updateModel API 函数

**目标**：将 `src/api/model.js` 中的 `updateModel` 改为发送 FormData，与后端 multipart 接口对齐。

**修改文件**：
- `E:\change_detection\change_detection\src\api\model.js`（修改）

**具体操作**：
1. 修改 `updateModel` 函数，参数改为接收 FormData，**不手动设置 Content-Type**（让 axios 自动处理，手动设置会丢失 boundary 导致解析失败）：
   ```js
   export function updateModel(id, formData) {
     return request({
       url: `/models/${id}`,
       method: 'put',
       data: formData,
     })
   }
   ```

**验收标准**：
- [x] 浏览器 Network 面板中 PUT 请求 Content-Type 为 `multipart/form-data; boundary=...`（有 boundary）

**风险点**：
- ⚠️ 若 axios 实例在拦截器中统一设置了 `Content-Type: application/json`，需在拦截器中判断 data 为 FormData 时跳过，否则会覆盖 boundary

---

### Step 3.1：新增响应式状态与预填/重置函数（Script 逻辑层）

**目标**：仅在 `<script setup>` 中补充 import、声明编辑所需的 ref，并实现 `handleEdit`、`closeEditDialog` 两个函数。**不触碰 `<template>`**。

**修改文件**：
- `E:\change_detection\change_detection\src\views\modelDatabase\index.vue`（修改）

**具体操作**：
1. 补充 import（在现有 import 行追加）：
   ```js
   import { getModelList, uploadModel as uploadModelApi, updateModel, deleteModel } from '@/api/model'
   import { ElMessage, ElMessageBox } from 'element-plus'
   ```
2. 在现有 ref 声明区末尾追加：
   ```js
   const editDialogVisible = ref(false)
   const editFormRef = ref(null)
   const isEditing = ref(false)
   const editWeightFileList = ref([])
   const editModelFileList = ref([])
   const editForm = ref({
     id: null,
     model_name: '',
     model_type: '',
     framework: '',
     description: '',
     current_weight: '',     // 只读展示当前文件名，不参与表单校验
     current_model_file: '', // 同上
   })
   ```
3. 在现有函数区末尾追加 `handleEdit` 和 `closeEditDialog`：
   ```js
   function handleEdit(row) {
     Object.assign(editForm.value, {
       id: row.id,
       model_name: row.model_name,
       model_type: row.model_type,
       framework: row.framework,
       description: row.description,
       current_weight: row.weight,
       current_model_file: row.model_file,
     })
     editWeightFileList.value = []
     editModelFileList.value = []
     editDialogVisible.value = true
   }

   function closeEditDialog(formEl) {
     formEl?.resetFields()
     editWeightFileList.value = []
     editModelFileList.value = []
     editDialogVisible.value = false
   }
   ```

**预期效果**：`console.log(editForm.value)` 在点击编辑后输出对应行数据，`editDialogVisible` 变为 true

**验收标准**：
- [x] 页面无 console error，现有功能（列表、上传）不受影响
- [x] 点击编辑后 `editForm.value` 包含正确的行数据（可用 Vue Devtools 验证）

**风险点**：
- ⚠️ 此步骤**只改 `<script>` 区域**，不动 `<template>`；若发现需要改模板，停下来报告

---

### Step 3.2：构建编辑弹窗模板（Template 视图层）

**目标**：在 `<template>` 中绑定操作列按钮，并在上传弹窗的 `</el-dialog>` 之后追加编辑弹窗结构。**不新增或修改任何函数逻辑**。

**修改文件**：
- `E:\change_detection\change_detection\src\views\modelDatabase\index.vue`（修改）

**具体操作**：
1. 将操作列的 `#default` 改为 `#default="scope"`，绑定按钮点击：
   ```html
   <template #default="scope">
     <el-button size="small" @click="handleEdit(scope.row)">编辑</el-button>
     <el-button size="small" type="danger" @click="handleDelete(scope.row)">删除</el-button>
   </template>
   ```
2. 在上传弹窗 `</el-dialog>` 之后追加编辑弹窗：
   - `v-model="editDialogVisible"`，title="编辑模型"，width="500"
   - `el-form` 绑定 `ref="editFormRef"`，`:model="editForm"`
   - 四个表单项：model_name、model_type、framework、description（校验规则同上传弹窗；description 的 min 改为 0 允许空）
   - 权重文件区：
     - `el-form-item` label="模型权重"（**不加 required**）
     - 提示文字：`<div class="current-file">当前文件：{{ editForm.current_weight }}</div>`
     - `el-upload` 绑定 `v-model:file-list="editWeightFileList"`，`:auto-upload="false"`，`:limit="1"`，`accept=".pth,.pt,.h5,.onnx,.pdparams"`，`:on-exceed="handleUploadExceed"`（复用现有函数）
   - 模型文件区：同上，绑定 `editModelFileList`，`accept=".py,.zip"`
   - footer：取消按钮调用 `closeEditDialog(editFormRef)`，确认按钮调用 `handleEditSubmit(editFormRef)`（函数在 Step 3.3 实现，此步骤仅绑定名称）

**预期效果**：点击编辑后弹窗打开，预填数据正确，文件区显示当前文件名；文件上传 UI 可交互但提交按钮暂不生效（Step 3.3 才实现）

**验收标准**：
- [x] 编辑弹窗正常渲染，样式无错位
- [x] 四个文本字段预填正确
- [x] 权重/模型文件区显示当前文件名
- [x] el-upload 可选择文件（不自动上传）
- [x] 现有上传弹窗、列表、分页功能不受影响

**风险点**：
- ⚠️ `#default` 必须改为 `#default="scope"`，否则 `scope.row` 为 undefined
- ⚠️ 编辑弹窗的文件 `el-form-item` **不加** `prop` 或校验规则（文件非必填），避免影响表单 validate

---

### Step 3.3：实现提交与删除逻辑（API 交互层）

**目标**：在 `<script setup>` 中实现 `handleEditSubmit` 和 `handleDelete` 两个函数，完成 FormData 组装、接口调用和异常处理。

**修改文件**：
- `E:\change_detection\change_detection\src\views\modelDatabase\index.vue`（修改）

**具体操作**：在 `closeEditDialog` 函数之后追加：
```js
async function handleEditSubmit(formEl) {
  if (!formEl || isEditing.value) return
  const valid = await formEl.validate().catch(() => false)
  if (!valid) return

  const formData = new FormData()
  formData.append('model_name', editForm.value.model_name)
  formData.append('model_type', editForm.value.model_type)
  formData.append('framework', editForm.value.framework)
  formData.append('description', editForm.value.description ?? '')
  if (editWeightFileList.value.length) {
    formData.append('weight', editWeightFileList.value[0].raw)
  }
  if (editModelFileList.value.length) {
    formData.append('model_file', editModelFileList.value[0].raw)
  }

  isEditing.value = true
  try {
    const res = await updateModel(editForm.value.id, formData)
    if (res.data.code === 200) {
      ElMessage.success('更新成功')
      closeEditDialog(editFormRef.value)
      fetchModels()
    } else {
      ElMessage.error(res.data.message || '更新失败')
    }
  } finally {
    isEditing.value = false
  }
}

async function handleDelete(row) {
  try {
    await ElMessageBox.confirm(`确认删除模型「${row.model_name}」？`, '提示', {
      type: 'warning',
    })
    const res = await deleteModel(row.id)
    if (res.data.code === 200) {
      ElMessage.success('删除成功')
      fetchModels()
    } else {
      ElMessage.error(res.data.message || '删除失败')
    }
  } catch {
    // 用户点取消，不处理
  }
}
```

**预期效果**：
- 仅改文字提交 → 后端只更新元数据，文件名不变
- 选择新文件提交 → 表格刷新后文件名更新
- 删除确认 → 行消失；取消 → 不删除

**验收标准**：
- [x] 编辑提交后 Network 面板 PUT 请求 Content-Type 含 boundary
- [x] 仅修改描述后，weight 文件名在列表中不变
- [x] 替换权重文件后，列表中权重文件名更新
- [x] 删除确认后行消失，取消无变化
- [x] 操作他人模型（404）时 ElMessage 显示"模型不存在"

**风险点**：
- ⚠️ `formData.append('description', editForm.value.description ?? '')` 中 `?? ''` 确保空描述传 `""` 而非 `"null"`
- ℹ️ el-input 清空后 v-model 天然为 `""`，无需额外转换
- ⚠️ FormData.append 的 key（`weight`、`model_file`）必须与后端 Form 参数名完全一致

---

## 整体风险清单

| 风险 | 影响 | 缓解方案 |
|------|------|----------|
| old_w_path 在 commit 后访问 | ObjectDeletedError | Step 1 的 2a 要求在 get_ml_model_by_id 后立即提取为局部变量 |
| new_w_path 未初始化时 except 报错 | 500 掩盖真实异常 | 初始化为 None，_safe_unlink 已处理 None |
| axios 拦截器覆盖 Content-Type | 请求解析失败 | Step 2 不手动设置 header，若拦截器有问题需在拦截器判断 FormData |
| 文件保存成功但 commit 失败 | 磁盘孤儿文件 | except 中 _safe_unlink(new_w_path / new_m_path) 清理 |
| 大体积权重文件耗尽磁盘或超时 | 服务崩溃 / 504 | Step 1 已加 MAX_WEIGHT_SIZE(1GB) / MAX_MODEL_FILE_SIZE(100MB) 校验 |
| 上传目录不存在导致 FileNotFoundError | ✅ 已解决 | `save_upload_file` 第 12 行已有 `dest_dir.mkdir(parents=True, exist_ok=True)` |

---

## 注意事项（给 Codex 的特别提示）

- 每次只执行一个 Step，执行完等待 Claude Code 审查后再继续
- Step 1 中 `old_w_path` / `old_m_path` 必须在 `save_upload_file` 前提取，不能在 commit 后访问 ORM 属性
- Step 2 中直接传 FormData 给 axios，不要手动设置 Content-Type header
- Step 3 中文件字段为非必填，校验规则不得加 required；FormData.append 的 key 名称必须与后端 Form 参数名完全一致（`weight`、`model_file`）
- 遇到 plan 未覆盖的情况，停下来报告，不要自行决策
