# 快速开始指南

## 🚀 立即开始使用影像上传功能

### 第一步：安装依赖
```powershell
pip install python-multipart
```

### 第二步：启动服务器
```powershell
cd E:\change_detection\change_detection_backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

看到以下信息表示启动成功：
```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 第三步：测试API

#### 方法1：使用Swagger UI（推荐新手）
1. 打开浏览器访问：http://localhost:8000/docs
2. 找到 `/images/upload` 接口
3. 点击 "Try it out"
4. 填写表单参数并上传文件
5. 点击 "Execute"

#### 方法2：使用测试HTML页面
1. 双击打开文件：`E:\change_detection\change_detection_backend\test_upload.html`
2. 填写表单
3. 选择文件
4. 点击上传
5. 查看影像列表

#### 方法3：前端代码调用
```javascript
// 在你的前端代码中
const formData = new FormData();
formData.append('image_name', '小明图');
formData.append('resolution', 1);
formData.append('capture_date', '2025-03-03');
formData.append('satellite', 'swot');
formData.append('type', 'dom');
formData.append('region_code', '320612001');
formData.append('image_file', imageFile); // File对象
// 可选：添加shapefile
formData.append('shp_files', shpFile1);
formData.append('shp_files', shpFile2);

// 上传
const response = await fetch('http://localhost:8000/images/upload', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);

// 获取列表
const listResponse = await fetch('http://localhost:8000/images/list');
const images = await listResponse.json();
console.log(images);
```

## 📋 可用的API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/images/upload` | POST | 上传影像 |
| `/images/list` | GET | 获取所有影像 |
| `/images/{id}` | GET | 获取单个影像 |

## 🎯 前端表格渲染示例

```javascript
// 获取影像列表
const images = await fetch('http://localhost:8000/images/list').then(r => r.json());

// 渲染到表格（Vue示例）
<template>
  <el-table :data="images">
    <el-table-column prop="id" label="ID" />
    <el-table-column prop="image_name" label="影像名称" />
    <el-table-column prop="resolution" label="分辨率" />
    <el-table-column prop="capture_date" label="拍摄日期" />
    <el-table-column prop="satellite" label="卫星" />
    <el-table-column prop="image_type" label="类型" />
    <el-table-column prop="region_code" label="区域代码" />
    <el-table-column prop="upload_time" label="上传时间">
      <template #default="{ row }">
        {{ new Date(row.upload_time).toLocaleString() }}
      </template>
    </el-table-column>
  </el-table>
</template>

<script setup>
import { ref, onMounted } from 'vue';

const images = ref([]);

const loadImages = async () => {
  const response = await fetch('http://localhost:8000/images/list');
  images.value = await response.json();
};

onMounted(loadImages);
</script>
```

## ✅ 验证清单

- [ ] 服务器正常启动（无错误信息）
- [ ] 访问 http://localhost:8000/docs 可以看到API文档
- [ ] 使用Swagger UI测试上传成功
- [ ] 使用 `/images/list` 可以获取到数据
- [ ] 前端可以正常调用接口

## 📁 文件保存位置

上传的文件会保存在：
- 影像文件：`E:\change_detection\change_detection_backend\uploads\images\`
- Shapefile：`E:\change_detection\change_detection_backend\uploads\shapefiles\`

## ❓ 常见问题

### Q: 上传时报错 "405 Method Not Allowed"
A: 检查是否使用了 `POST` 方法，不是 `GET`

### Q: 上传时报错 "422 Unprocessable Entity"
A: 检查所有必填字段是否都已填写，特别注意日期格式要是 `YYYY-MM-DD`

### Q: CORS错误
A: 确保前端地址在 `main.py` 的 `origins` 列表中，或临时改为 `allow_origins=["*"]`

### Q: 找不到 python-multipart
A: 运行 `pip install python-multipart`

## 📚 更多信息

- 完整API文档：查看 `API_IMAGES_README.md`
- 实现总结：查看 `IMPLEMENTATION_SUMMARY.md`
- HTTP测试：查看 `test_images_api.http`

---

**祝你使用愉快！** 🎉

如有问题，请检查服务器日志输出，通常会有详细的错误信息。

