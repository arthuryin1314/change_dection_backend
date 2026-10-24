# #55：系统模型目录、Schema 迁移与权限边界

本票将模型库改为所有登录用户共享的系统目录。普通用户只读，管理员只编辑名称、描述和标签。旧模型目录按本次确认清空；已有识别结果、变化结果、名称快照、分类文件和面积保留。上传、验证、生产版本切换、下线和真实推理由后续 ticket 交付。

## 1. 执行数据库迁移

先停止后端，在当前应用数据库中执行完整文件：

`migrations/010_system_model_catalog.sql`

本机配置的数据库是 `localhost` 上的 `change_detection`。可以在数据库工具的 SQL 编辑器中打开此文件并全部执行；也可以在后端目录使用 psql（将 `数据库账号` 替换为自己的账号）：

```powershell
psql -h localhost -d change_detection -U 数据库账号 -v ON_ERROR_STOP=1 -f migrations/010_system_model_catalog.sql
```

脚本有事务边界，失败会回滚。它会：

- 给 `user_info` 添加 `is_admin`，现有用户默认都是普通用户。
- 创建 `system_models` 和 `system_model_versions`，保存逻辑模型、版本、生产版本指针、代码/配置哈希、权重哈希和环境镜像摘要。
- 解除历史识别和变化结果对旧模型行的引用，然后 `DELETE FROM model_library`。不使用 `TRUNCATE CASCADE`。
- 保留历史结果快照和磁盘文件。旧模型上传文件也不通过 SQL 删除。

新系统目录为空，旧模型不会转换或重新导入。重复执行不会清空新系统目录。旧的 `model_library` 空表暂留，供历史外键和尚未迁移的内部运行接口兼容；新目录接口不读取它。

应在已经完成前序 `001`–`009` 迁移的数据库上执行；尤其 `007` 负责补齐可恢复的历史名称快照。应用启动的 `create_all()` 不会给已有用户表补列，因此需要先运行本迁移，再启动新后端。

## 2. 设置管理员

本票不自动选择或提升任何账号。先查询账号，再将准备作为管理员的账号 ID 填入 UPDATE：

```sql
SELECT id, username, is_admin FROM user_info ORDER BY id;

-- 将 123 换成上一步确认的管理员账号 ID。
UPDATE user_info SET is_admin = true WHERE id = 123
RETURNING id, username, is_admin;
```

更新应返回恰好一行。保留另一个 `is_admin = false` 的账号进行普通用户验收。权限在每次请求时读取数据库，不依赖前端保存的身份；修改权限后刷新模型库即可。

## 3. 启动与初始验收

后端：

```powershell
cd E:\change_detection\change_detection_backend
conda run --no-capture-output -n change-detection-web python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

前端，在另一个终端：

```powershell
cd E:\change_detection\change_detection
npm run dev
```

普通用户登录后进入“模型仓库”（`/modelDatabase`）。初次应显示“系统模型库”，总数 0、暂无模型，不出现上传、编辑、删除按钮。管理员也不会看到上传或删除入口。

打开“地物识别历史”，旧结果仍存在，列表和详情显示“原模型不可用”。原有文件完整时分类图和面积仍可查看；不能重新执行。已有变化历史和报告仍能读取。文件原先缺失的历史记录不会被迁移脚本修复。

## 4. 有数据的目录与编辑验收

因为上传属于后续票，本票刚迁移后的空目录没有可编辑记录。在**测试数据库**执行 `docs/sql/ticket-55-demo-catalog.sql`，创建两个合成目录条目及版本，以便现在验收筛选、展示和编辑；这些条目没有实际模型资产，也不能推理。正式目录可以等后续上传功能完成后用真实模型复验。

记录脚本返回的模型 ID，再逐项检查：

| 操作 | 预期 |
| --- | --- |
| 两个不同普通账号打开模型库 | 都看到相同两个模型，与上传者或账号无关 |
| 选择“地物识别”或“变化检测” | 只展示对应模型；切换类型或搜索时回到第一页 |
| 搜索模型名称 | 返回服务端筛选结果，总数与分页对应筛选后的集合 |
| 查看模型信息 | 展示类型、版本、状态、输入说明、波段及 CPU/内存/GPU 需求；不显示权重路径、代码路径或资产哈希 |
| 普通用户查看目录 | 没有编辑、上传、删除按钮 |
| 管理员查看目录 | 显示“目录管理”列和“编辑”入口 |
| 管理员修改名称、描述和逗号分隔的标签并保存 | 保存成功、列表刷新；另一普通账号刷新后看到更新 |
| 模拟保存失败（例如弹窗打开后撤销管理员权限） | 显示失败原因、保留输入、停止加载，允许取消或在权限恢复后重试 |
| 技术可用的合成版本 | 目录显示“技术可用”，业务能力仍为“暂不可调用”，不会出现在业务选择器里 |

编辑前后执行下面的查询，确认只修改目录元数据，生产版本指针及三个资产身份字段保持不变：

```sql
SELECT m.id, m.model_name, m.description, m.tags, m.production_version_id,
       v.id AS version_id, v.version, v.code_config_sha256,
       v.weight_sha256, v.image_digest
FROM system_models m
LEFT JOIN system_model_versions v ON v.model_id = m.id
ORDER BY m.id, v.id;
```

## 5. 绕过前端验证后端权限

在已登录页面打开浏览器开发者工具的 Console。将 `模型ID` 替换成测试条目的真实整数 ID；这里只从当前页面读取已登录状态，不打印 token：

```javascript
const modelId = 模型ID
const session = JSON.parse(localStorage.getItem('users'))
const response = await fetch(`/api/models/${modelId}`, {
  method: 'PUT',
  headers: {
    Authorization: `Bearer ${session.token}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ model_name: '验收修改名称', description: '验收描述', tags: ['验收'] }),
})
console.log(response.status, await response.json())
```

- 普通账号：HTTP 403，不修改记录。
- 管理员：HTTP 200，元数据更新。
- 管理员在 JSON 中额外添加 `model_type`、`production_version_id`、`version` 或 `weight_file_path`：HTTP 422，不能通过目录编辑改变版本或资产。
- 管理员编辑不存在的模型 ID：HTTP 404。

也可用同样的 Authorization 请求以下旧入口，确认不能绕过新目录：

| 入口 | 普通用户 | 管理员 |
| --- | --- | --- |
| `POST /api/models/upload` | 403 | 503，上传尚未开放 |
| `DELETE /api/models/{id}` | 403 | 405，不允许物理删除 |

上述写入口即使没有提供表单，也应先对普通用户返回 403。新建识别/变化任务继续保持 #54 的不可用响应，不能用历史快照中的旧模型 ID 重新执行。

## 自动验证

后端在无 Torch 环境执行完整验收，包含真实 PostgreSQL 迁移、权限、目录查询、历史读取及应用启动：

```powershell
cd E:\change_detection\change_detection_backend
conda run --no-capture-output -n change-detection-web python scripts/verify_torch_free_backend.py
```

脚本创建并最终删除随机命名的验收数据库，不修改 `.env` 指向的业务库。迁移测试额外覆盖重复执行不会删除新目录、版本名称唯一，以及生产版本不能指向另一个逻辑模型。

前端：

```powershell
cd E:\change_detection\change_detection
node --import jiti/register --test --test-concurrency=1 tests/*.test.mjs
npm run type-check
npx eslint .
npx oxlint .
npm run build
```

页面测试保留真实目录、筛选、编辑及历史详情组件，仅替换网络和底层 UI/地图适配器。
