"""
演示如何使用新的焊机参数机器学习系统
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append('.')

print("=" * 70)
print("焊机参数机器学习系统演示")
print("=" * 70)
print()

# 演示1: 使用配置
print("1. 使用配置系统")
print("-" * 50)
from configs.config import get_team_config

config = get_team_config('甲')
print(f"创建甲班配置:")
print(f"  班组: {config.data.team}")
print(f"  模型类型: {config.model.model_type}")
print(f"  最大样本数: {'全部' if config.data.max_samples is None else config.data.max_samples}")
print()

# 演示2: 数据加载
print("2. 数据加载演示")
print("-" * 50)
from data.data_loader import DataLoader

# 使用小数据集演示
test_config = config.data
test_config.max_samples = 100
test_config.verbose = False

loader = DataLoader(test_config)
df = loader.load_data()
print(f"加载数据: {len(df)} 条记录")
print(f"列名: {list(df.columns)[:5]}...")
print()

# 演示3: 模型工厂
print("3. 模型工厂演示")
print("-" * 50)
from models.model_factory import ModelFactory

available_models = ModelFactory.get_available_models()
print(f"可用模型类型: {available_models}")

for model_type in available_models:
    description = ModelFactory.get_model_description(model_type)
    print(f"  {model_type}: {description}")
print()

# 演示4: 训练流程
print("4. 训练流程演示")
print("-" * 50)
print("使用小数据集进行演示训练...")

# 修改配置使用小数据集
demo_config = get_team_config('甲')
demo_config.data.max_samples = 200
demo_config.data.verbose = False
demo_config.verbose = False
demo_config.save_model = False
demo_config.evaluate_model = True

from data.data_loader import DataLoader
from models.model_factory import ModelFactory

# 加载数据
loader = DataLoader(demo_config.data)
X_train, X_test, y_train, y_test, metadata = loader.get_data()

print(f"数据加载完成:")
print(f"  训练样本: {X_train.shape[0]}")
print(f"  测试样本: {X_test.shape[0]}")
print(f"  特征维度: {X_train.shape[1]}")
print(f"  目标维度: {y_train.shape[1]}")
print()

# 创建和训练模型
model = ModelFactory.create_model(demo_config)
print(f"创建模型: {model.__class__.__name__}")
model.fit(X_train, y_train)
print("模型训练完成")
print()

# 评估模型
results = model.evaluate(X_test, y_test, demo_config.data.target_columns)

print("模型评估结果:")
for target_name in demo_config.data.target_columns:
    if target_name in results:
        metrics = results[target_name]
        print(f"  {target_name}: R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")  # 改为R2

if 'overall' in results:
    overall = results['overall']
    print(f"  总体: R2={overall['r2']:.4f}, MAE={overall['mae']:.4f}")  # 改为R2
print()

# 演示5: 特征重要性
print("5. 特征重要性演示")
print("-" * 50)
importance = model.get_feature_importance()
if importance is not None:
    feature_names = metadata['feature_names']
    print("特征重要性排名:")
    for i, (name, imp) in enumerate(zip(feature_names, importance), 1):
        print(f"  {i}. {name}: {imp:.4f}")
else:
    print("该模型不支持特征重要性分析")
print()

# 演示6: 使用示例
print("6. 使用示例")
print("-" * 50)
print("命令行使用:")
print("  训练模型: python scripts/train.py --team 甲 --model random_forest --max_samples 2000")
print("  比较模型: python scripts/evaluate.py compare_models --team 甲")
print("  预测: python scripts/predict.py interactive --model saved_models/模型文件.joblib")
print()
print("Python代码使用:")
print("  from scripts.train import train_model")
print("  from configs.config import get_team_config")
print("  config = get_team_config('甲')")
print("  train_model(config)")
print()

print("=" * 70)
print("演示完成!")
print("详细用法请参考 USAGE.md 文档")
print("=" * 70)