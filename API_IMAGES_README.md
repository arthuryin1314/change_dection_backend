# 影像管理API文档

## 功能概述

后端提供两个主要接口：
1. **上传影像**：上传影像文件和相关的shapefile边界文件
2. **获取影像列表**：获取所有已上传的影像记录

## API接口

### 1. 上传影像

**接口地址**：`POST /images/upload`

**请求方式**：multipart/form-data

**请求参数**：

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| image_name | string | 是 | 影像名称 |
| resolution | float | 是 | 影像分辨率（米） |
| capture_date | date | 是 | 拍摄日期（格式：YYYY-MM-DD） |
| satellite | string | 是 | 卫星名称 |
| type | string | 是 | 影像类型（如：dom, dsm等） |
| region_code | string | 是 | 区域代码 |
| image_file | file | 是 | 影像文件 |
| shp_files | file[] | 否 | Shapefile文件（.shp, .dbf, .prj等，可多选） |

**响应示例**：
```json
{
  "code": 200,
  "message": "影像上传成功",
  "data": {
    "id": 1,
    "image_name": "小明图",
    "img_path": "uploads/images/320612001_小明图_test.tif"
  }
}
```

### 2. 获取影像列表

**接口地址**：`GET /images/list`

**请求方式**：GET

**响应示例**：
```json
[
  {
    "id": 1,
    "image_name": "小明图",
    "resolution": 1.0,
    "capture_date": "2025-03-03",
    "satellite": "swot",
    "image_type": "dom",
    "region_code": "320612001",
    "img_path": "uploads/images/320612001_小明图_test.tif",
    "upload_time": "2026-03-10T10:30:00"
  }
]
```

### 3. 根据ID获取影像

**接口地址**：`GET /images/{image_id}`

**请求方式**：GET

**路径参数**：
- `image_id`：影像ID

**响应示例**：
```json
{
  "id": 1,
  "image_name": "小明图",
  "resolution": 1.0,
  "capture_date": "2025-03-03",
  "satellite": "swot",
  "image_type": "dom",
  "region_code": "320612001",
  "img_path": "uploads/images/320612001_小明图_test.tif",
  "upload_time": "2026-03-10T10:30:00"
}
```

## 前端集成示例

### Vue.js 示例

```javascript
// 上传影像
async function uploadImage(formData) {
  try {
    const response = await fetch('http://localhost:8000/images/upload', {
      method: 'POST',
      body: formData
    });
    
    const result = await response.json();
    
    if (response.ok) {
      console.log('上传成功:', result.data);
      return result.data;
    } else {
      console.error('上传失败:', result.detail);
      throw new Error(result.detail);
    }
  } catch (error) {
    console.error('网络错误:', error);
    throw error;
  }
}

// 获取影像列表
async function getImagesList() {
  try {
    const response = await fetch('http://localhost:8000/images/list');
    const images = await response.json();
    return images;
  } catch (error) {
    console.error('获取列表失败:', error);
    throw error;
  }
}

// 使用示例
const formData = new FormData();
formData.append('image_name', '小明图');
formData.append('resolution', 1);
formData.append('capture_date', '2025-03-03');
formData.append('satellite', 'swot');
formData.append('type', 'dom');
formData.append('region_code', '320612001');
formData.append('image_file', fileInput.files[0]);

// 如果有shapefile
const shpFiles = shpFileInput.files;
for (let i = 0; i < shpFiles.length; i++) {
  formData.append('shp_files', shpFiles[i]);
}

await uploadImage(formData);
const images = await getImagesList();
```

### React 示例

```jsx
import { useState } from 'react';

function ImageUpload() {
  const [images, setImages] = useState([]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    
    try {
      const response = await fetch('http://localhost:8000/images/upload', {
        method: 'POST',
        body: formData
      });
      
      const result = await response.json();
      
      if (response.ok) {
        alert('上传成功！');
        loadImages();
      } else {
        alert('上传失败: ' + result.detail);
      }
    } catch (error) {
      alert('网络错误: ' + error.message);
    }
  };
  
  const loadImages = async () => {
    try {
      const response = await fetch('http://localhost:8000/images/list');
      const data = await response.json();
      setImages(data);
    } catch (error) {
      console.error('加载失败:', error);
    }
  };
  
  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input name="image_name" placeholder="影像名称" required />
        <input name="resolution" type="number" step="0.01" placeholder="分辨率" required />
        <input name="capture_date" type="date" required />
        <input name="satellite" placeholder="卫星名称" required />
        <select name="type" required>
          <option value="">请选择类型</option>
          <option value="dom">DOM</option>
          <option value="dsm">DSM</option>
        </select>
        <input name="region_code" placeholder="区域代码" required />
        <input name="image_file" type="file" required />
        <input name="shp_files" type="file" multiple />
        <button type="submit">上传</button>
      </form>
      
      <button onClick={loadImages}>刷新列表</button>
      
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>名称</th>
            <th>分辨率</th>
            <th>日期</th>
            <th>卫星</th>
            <th>类型</th>
            <th>区域代码</th>
          </tr>
        </thead>
        <tbody>
          {images.map(img => (
            <tr key={img.id}>
              <td>{img.id}</td>
              <td>{img.image_name}</td>
              <td>{img.resolution}</td>
              <td>{img.capture_date}</td>
              <td>{img.satellite}</td>
              <td>{img.image_type}</td>
              <td>{img.region_code}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

## 文件存储结构

```
uploads/
├── images/                     # 影像文件存储目录
│   └── {region_code}_{image_name}_{filename}
└── shapefiles/                 # Shapefile存储目录
    └── {region_code}_{image_name}/
        ├── *.shp
        ├── *.dbf
        └── *.prj
```

## 注意事项

1. **文件大小限制**：根据服务器配置，可能需要调整上传文件大小限制
2. **文件类型**：建议对上传的文件类型进行验证
3. **CORS配置**：确保前端域名在 `main.py` 的 CORS 配置中
4. **数据库**：确保数据库表已正确创建
5. **权限**：确保 `uploads` 目录具有写入权限

## 测试

### 使用提供的HTML测试页面
1. 在浏览器中打开 `test_upload.html`
2. 填写表单信息
3. 选择影像文件和shapefile文件（可选）
4. 点击上传
5. 查看影像列表

### 使用HTTP客户端
参考 `test_images_api.http` 文件中的示例请求

## 启动服务

```bash
cd E:\change_detection\change_detection_backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

服务启动后访问：
- API文档：http://localhost:8000/docs
- 测试页面：file:///{path}/test_upload.html

