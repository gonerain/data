"""
测试新系统功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.append('.')

print("=" * 70)
print("测试焊机参数机器学习系统")
print("=" * 70)

# 测试1: 检查配置文件
print("\n[1/4] 测试配置文件...")
try:
    from configs.config import TrainingConfig, get_team_config
    config = get_team_config('甲')
    print(f"  [OK] 配置文件加载成功")
    print(f"    班组: {config.data.team}")
    print(f"    模型类型: {config.model.model_type}")
except Exception as e:
    print(f"  [ERROR] 配置文件错误: {e}")
    sys.exit(1)

# 测试2: 检查数据加载器
print("\n[2/4] 测试数据加载器...")
try:
    from data.data_loader import DataLoader
    
    # 创建测试配置
    test_config = config.data
    test_config.max_samples = 100  # 只测试100条数据
    test_config.verbose = False
    
    loader = DataLoader(test_config)
    
    # 测试数据加载
    df = loader.load_data()
    print(f"  [OK] 数据加载成功，读取 {len(df)} 条数据")
    
    # 测试数据预处理
    df_processed = loader.preprocess_data(df)
    print(f"  [OK] 数据预处理成功，处理 {len(df_processed)} 条数据")
    
    # 测试特征准备
    X, y, metadata = loader.prepare_features(df_processed)
    print(f"  [OK] 特征准备成功，特征维度: {X.shape}")
    
    # 测试数据划分
    X_train, X_test, y_train, y_test = loader.split_data(X, y)
    print(f"  [OK] 数据划分成功，训练集: {X_train.shape[0]}, 测试集: {X_test.shape[0]}")
    
except Exception as e:
    print(f"  [ERROR] 数据加载器错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 检查模型工厂
print("\n[3/4] 测试模型工厂...")
try:
    from models.model_factory import ModelFactory
    
    available_models = ModelFactory.get_available_models()
    print(f"  [OK] 可用模型: {available_models}")
    
    # 测试创建随机森林模型
    rf_model = ModelFactory.create_model(config)
    print(f"  [OK] 随机森林模型创建成功")
    
    # 测试模型信息
    model_info = rf_model.get_model_info()
    print(f"    模型类型: {model_info['model_type']}")
    
except Exception as e:
    print(f"  [ERROR] 模型工厂错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试4: 检查训练流程
print("\n[4/4] 测试训练流程...")
try:
    # 使用小数据集测试
    test_config = get_team_config('甲')
    test_config.data.max_samples = 200
    test_config.data.verbose = False
    test_config.verbose = False
    test_config.save_model = False
    test_config.evaluate_model = True
    
    from data.data_loader import DataLoader
    from models.model_factory import ModelFactory
    
    # 加载数据
    loader = DataLoader(test_config.data)
    X_train, X_test, y_train, y_test, metadata = loader.get_data()
    
    # 创建和训练模型
    model = ModelFactory.create_model(test_config)
    model.fit(X_train, y_train)
    
    # 评估模型
    results = model.evaluate(X_test, y_test, test_config.data.target_columns)
    
    print(f"  [OK] 训练流程成功")
    print(f"    训练样本: {X_train.shape[0]}")
    print(f"    测试样本: {X_test.shape[0]}")
    
    if 'overall' in results:
        overall = results['overall']
        print(f"    总体R2: {overall['r2']:.4f}")
    
    # 测试特征重要性
    importance = model.get_feature_importance()
    if importance is not None:
        print(f"  [OK] 特征重要性计算成功")
        print(f"    特征数量: {len(importance)}")
    
except Exception as e:
    print(f"  [ERROR] 训练流程错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("所有测试通过! 系统功能正常")
print("=" * 70)

print("\n下一步:")
print("1. 训练完整模型: python scripts/train.py --team 甲 --model random_forest")
print("2. 比较模型性能: python scripts/evaluate.py compare_models --team 甲")
print("3. 使用模型预测: python scripts/predict.py interactive --model saved_models/...")
print("\n详细用法请参考 USAGE.md")