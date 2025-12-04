# 安装说明

## 环境要求

- Python 3.8 或更高版本
- pip（Python包管理器）

## 快速安装

### 1. 克隆项目

```bash
git clone <项目地址>
cd welding_ml_project
```

### 2. 创建虚拟环境（推荐）

```bash
# 使用 venv
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### 3. 安装依赖

```bash
# 安装核心依赖（必需）
pip install pandas numpy scikit-learn joblib

# 或安装所有依赖（包括可选）
pip install -r requirements.txt
```

### 4. 安装可选依赖

```bash
# 如果需要XGBoost模型
pip install xgboost

# 如果需要可视化
pip install matplotlib seaborn

# 如果需要Jupyter Notebook
pip install jupyter
```

## 验证安装

```bash
# 运行系统测试
python scripts/test_system.py

# 运行演示
python demo.py
```

## 目录结构

安装完成后，项目目录结构如下：

```
welding_ml_project/
├── configs/          # 配置文件
├── models/           # 模型定义
├── data/             # 数据处理
├── scripts/          # 可执行脚本
├── saved_models/     # 保存的模型
├── results/          # 评估结果
├── logs/             # 日志文件
├── dataset/          # 数据文件
├── requirements.txt  # 依赖列表
├── README.md         # 项目说明
├── USAGE.md          # 使用说明
├── INSTALL.md        # 本文件
└── .gitignore        # Git忽略文件
```

## 常见问题

### 1. 导入错误

```bash
# 确保在项目根目录运行
cd /path/to/welding_ml_project

# 确保虚拟环境已激活
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
```

### 2. 编码错误

项目数据文件使用GBK编码，代码已自动处理。如果仍有问题：

```bash
# 检查文件编码
file -i dataset/data.csv

# 或转换编码（如果需要）
iconv -f gbk -t utf-8 dataset/data.csv > dataset/data_utf8.csv
```

### 3. 内存不足

```bash
# 减少训练数据量
python scripts/train.py --team 甲 --model random_forest --max_samples 1000
```

### 4. 缺少依赖

```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt
```

## 开发环境设置

### VS Code 配置

创建 `.vscode/settings.json`：

```json
{
    "python.defaultInterpreterPath": "venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/.mypy_cache": true
    }
}
```

### PyCharm 配置

1. 打开项目
2. 设置Python解释器为虚拟环境中的Python
3. 安装项目依赖

## 生产环境部署

### Docker 部署（示例）

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "run.py"]
```

### 系统服务（Linux）

创建服务文件 `/etc/systemd/system/welding-ml.service`：

```ini
[Unit]
Description=Welding ML Service
After=network.target

[Service]
Type=simple
User=welding
WorkingDirectory=/opt/welding_ml_project
ExecStart=/opt/welding_ml_project/venv/bin/python run.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## 更新项目

```bash
# 更新代码
git pull

# 更新依赖
pip install --upgrade -r requirements.txt

# 清理缓存
rm -rf __pycache__/
rm -rf saved_models/*
rm -rf results/*
rm -rf logs/*
```

## 联系方式

如有安装问题，请参考文档或联系项目维护者。

---

**版本**: 1.0.0  
**最后更新**: 2024-12-04  
**作者**: 焊机参数机器学习项目组