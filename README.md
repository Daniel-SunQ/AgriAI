# 农作物病害检测与实时监控系统

本项目基于 Flask + YOLOv8 实现农作物病害检测、实时监控与统计分析，支持图片上传、病害识别、分区监控、数据统计等功能，适用于温室、农田等场景的智能化管理。

---

## 目录结构

```
.
├── app.py                  # Flask主程序
├── requirements.txt        # 依赖包列表
├── data/                   # 模型、数据库与检测结果
│   ├── best.pt             # 训练好的YOLO模型
│   ├── best.mlpackage/     # 其它模型格式（如有）
│   ├── detections.db       # 检测结果数据库
│   └── ...                 # 其它数据文件
├── static/                 # 静态资源
│   ├── css/style.css       # 样式表
│   ├── uploads/            # 上传图片
│   └── detected/           # 检测后图片
├── templates/              # 前端页面模板
│   ├── base.html
│   ├── index.html
│   ├── monitor.html
│   └── statistics.html
└── ...
```

---

## 安装与运行

### 1. 安装依赖

建议使用虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 准备模型

将你训练好的 YOLOv8 模型（如 `best.pt` 或 `best.mlpackage`）放入 `data/` 目录，并在 `app.py` 中配置正确的模型路径。

### 3. 启动服务

```bash
python app.py
```

浏览器访问：[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 功能介绍

- **病害检测**：上传作物图片，自动识别病害类别与位置。
- **实时监控**：分区展示各区域、各植株的健康状态，异常自动高亮。
- **统计分析**：统计各区域病害分布、异常数量、健康比例等。
- **历史记录**：所有检测结果自动存入数据库，便于追溯与分析。

---

## 主要技术栈

- 后端：Flask
- 前端：Bootstrap、原生JS
- 目标检测：YOLOv8（ultralytics）
- 数据库：SQLite

---

## 依赖环境

- Python 3.8+
- Flask
- ultralytics
- opencv-python
- 其它依赖见 requirements.txt

---

## 注意事项

- 请确保 `data/`、`static/uploads/`、`static/detected/` 目录存在且有写权限。
- 数据库文件 `detections.db` 会自动生成，无需手动创建。
- 模型文件需与训练时类别顺序一致，否则检测结果类别会错乱。

---

## 联系与支持

如有问题或建议，欢迎联系作者或提交 issue。 