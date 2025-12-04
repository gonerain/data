"""
检查项目结构完整性
"""

import os
import sys
from pathlib import Path

print("=" * 70)
print("焊机参数机器学习项目结构检查")
print("=" * 70)
print()

# 必需的文件和目录
required_items = [
    # 目录
    ("configs", True, "配置目录"),
    ("models", True, "模型目录"),
    ("data", True, "数据目录"),
    ("scripts", True, "脚本目录"),
    ("saved_models", True, "模型保存目录"),
    ("results", True, "结果目录"),
    ("logs", True, "日志目录"),
    ("dataset", True, "数据集目录"),
    
    # 配置文件
    ("configs/config.py", False, "主配置文件"),
    
    # 模型文件
    ("models/base_model.py", False, "基础模型类"),
    ("models/random_forest_model.py", False, "随机森林模型"),
    ("models/model_factory.py", False, "模型工厂"),
    
    # 数据处理
    ("data/data_loader.py", False, "数据加载器"),
    
    # 主要脚本
    ("scripts/train.py", False, "训练脚本"),
    ("scripts/evaluate.py", False, "评估脚本"),
    ("scripts/predict.py", False, "预测脚本"),
    ("scripts/test_system.py", False, "测试脚本"),
    
    # 文档（README_CN.md 和 README.md 二选一即可）
    ("README.md", False, "项目说明"),
    # ("README_CN.md", False, "中文说明"),  # 可选
    ("USAGE.md", False, "使用说明"),
    ("INSTALL.md", False, "安装说明"),
    
    # 其他文件
    ("requirements.txt", False, "依赖列表"),
    (".gitignore", False, "Git忽略文件"),
    ("run.py", False, "启动脚本"),
    ("demo.py", False, "演示脚本"),
]

# 检查每个项目
all_passed = True
missing_items = []

for item_path, is_dir, description in required_items:
    path = Path(item_path)
    exists = path.exists()
    is_correct_type = (is_dir and path.is_dir()) or (not is_dir and path.is_file())
    
    if exists and is_correct_type:
        status = "[OK]"
    else:
        # 对于可选文件，不标记为缺失
        if "README_CN.md" in item_path:
            status = "[OPTIONAL]"
            continue  # 跳过可选文件
        status = "[MISSING]"
        all_passed = False
        missing_items.append((item_path, description, is_dir))
    
    type_str = "目录" if is_dir else "文件"
    print(f"{status} {type_str}: {item_path} ({description})")

print()

# 检查数据集文件
dataset_files = [
    ("dataset/data.csv", "主要数据文件"),
    ("dataset/test_data.csv", "测试数据文件"),
    ("dataset/all_grades.csv", "牌号数据文件"),
]

print("数据集文件检查:")
for file_path, description in dataset_files:
    path = Path(file_path)
    if path.exists() and path.is_file():
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  [OK] {file_path} ({description}) - {size_mb:.2f} MB")
    else:
        print(f"  [MISSING] {file_path} ({description}) - 文件不存在")
        all_passed = False
        missing_items.append((file_path, description, False))

print()

# 检查.gitkeep文件
gitkeep_files = [
    "saved_models/.gitkeep",
    "results/.gitkeep",
    "logs/.gitkeep",
]

print(".gitkeep文件检查:")
for gitkeep_path in gitkeep_files:
    path = Path(gitkeep_path)
    if path.exists() and path.is_file():
        print(f"  [OK] {gitkeep_path}")
    else:
        print(f"  [MISSING] {gitkeep_path} - 建议创建此文件")

print()

# 显示结果
if all_passed:
    print("=" * 70)
    print("所有检查通过! 项目结构完整。")
    print("=" * 70)
    
    # 显示下一步建议
    print("\n下一步建议:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 测试系统: python scripts/test_system.py")
    print("3. 查看演示: python demo.py")
    print("4. 开始使用: python run.py")
    
else:
    print("=" * 70)
    print("发现缺失的项目:")
    print("=" * 70)
    
    for item_path, description, is_dir in missing_items:
        type_str = "目录" if is_dir else "文件"
        print(f"- {type_str}: {item_path} ({description})")
    
    print("\n修复建议:")
    print("1. 确保已正确克隆或下载项目")
    print("2. 检查文件路径和权限")
    print("3. 重新运行项目设置脚本")
    
    # 提供修复命令
    print("\n创建缺失目录的命令:")
    for item_path, description, is_dir in missing_items:
        if is_dir:
            print(f"  mkdir -p {item_path}")
    
print()

# 检查Python环境
print("Python环境检查:")
try:
    import pandas
    print(f"  [OK] pandas {pandas.__version__}")
except ImportError:
    print("  [MISSING] pandas - 未安装")
    all_passed = False

try:
    import numpy
    print(f"  [OK] numpy {numpy.__version__}")
except ImportError:
    print("  [MISSING] numpy - 未安装")
    all_passed = False

try:
    import sklearn
    print(f"  [OK] scikit-learn {sklearn.__version__}")
except ImportError:
    print("  [MISSING] scikit-learn - 未安装")
    all_passed = False

try:
    import joblib
    print(f"  [OK] joblib {joblib.__version__}")
except ImportError:
    print("  [MISSING] joblib - 未安装")
    all_passed = False

print()
print("=" * 70)
print("检查完成!")
print("=" * 70)