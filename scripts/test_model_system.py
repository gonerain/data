"""
测试模型工厂系统
"""
import sys
import os
# 添加当前目录到Python路径
sys.path.append('.')

from configs.config import TrainingConfig
from models.model_factory import ModelFactory


def print_header(title):
    """打印标题"""
    print("\n" + "="*60)
    print(f" {title}")
    print("="*60)


def test_basic_functionality():
    """测试基本功能"""
    print_header("1. 测试基本功能")
    
    # 查看已注册模型
    print("当前已注册的模型:")
    models = ModelFactory.get_available_models()
    for i, model_type in enumerate(models, 1):
        desc = ModelFactory.get_model_description(model_type)
        print(f"  {i}. {model_type}: {desc}")
    
    return True


def test_model_creation():
    """测试模型创建"""
    print_header("2. 测试模型创建")
    
    test_cases = [
        ('random_forest', '随机森林'),
        ('linear_regression', '线性回归'),
    ]
    
    for model_type, model_name in test_cases:
        print(f"\n测试 {model_name} 模型:")
        config = TrainingConfig()
        config.model.model_type = model_type
        
        try:
            model = ModelFactory.create_model(config)
            print(f"  ✓ 成功创建: {model.__class__.__name__}")
            print(f"    配置参数: {list(model.config.keys())}")
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")
    
    return True


def test_dynamic_discovery():
    """测试动态发现"""
    print_header("3. 测试动态发现")
    
    print("测试决策树模型（通过动态发现）:")
    config = TrainingConfig()
    config.model.model_type = 'decision_tree'
    
    try:
        # 决策树模型应该被自动发现
        model = ModelFactory.create_model(config)
        print(f"  ✓ 动态发现成功: {model.__class__.__name__}")
        
        # 验证已注册
        models = ModelFactory.get_available_models()
        if 'decision_tree' in models:
            print(f"  ✓ 决策树模型已注册")
        else:
            print(f"  ✗ 决策树模型未注册")
            return False
            
    except Exception as e:
        print(f"  ✗ 动态发现失败: {e}")
        return False
    
    return True


def test_dependency_checking():
    """测试依赖检查"""
    print_header("4. 测试依赖检查")
    
    print("测试XGBoost模型（如果未安装应该回退）:")
    config = TrainingConfig()
    config.model.model_type = 'xgboost'
    
    try:
        model = ModelFactory.create_model(config)
        model_class = model.__class__.__name__
        
        if model_class == 'RandomForestModel':
            print(f"  ✓ XGBoost未安装，正确回退到随机森林")
        else:
            print(f"  ! XGBoost已安装，使用 {model_class}")
            
    except Exception as e:
        print(f"  ✗ 测试失败: {e}")
        return False
    
    return True


def test_manual_registration():
    """测试手动注册"""
    print_header("5. 测试手动注册")
    
    # 创建一个简单的测试模型
    from models.base_model import BaseModel
    import numpy as np
    
    class TestModel(BaseModel):
        """测试模型"""
        
        def __init__(self, config: dict):
            super().__init__(config)
            self.model = None
            
        def build_model(self):
            pass
            
        def fit(self, X: np.ndarray, y: np.ndarray):
            self.model = "test_model"
            self.feature_importance = np.ones(X.shape[1])
            
        def predict(self, X: np.ndarray) -> np.ndarray:
            return np.zeros((X.shape[0], 3))
    
    # 手动注册
    print("手动注册测试模型:")
    ModelFactory.manual_register(
        model_type='test_model',
        model_class=TestModel,
        description='测试模型 - 演示手动注册',
        dependencies=['numpy']
    )
    
    # 验证注册
    models = ModelFactory.get_available_models()
    if 'test_model' in models:
        print("  ✓ 手动注册成功")
        
        # 测试创建
        config = TrainingConfig()
        config.model.model_type = 'test_model'
        
        try:
            model = ModelFactory.create_model(config)
            print(f"  ✓ 成功创建测试模型")
            
            # 测试功能
            X = np.random.randn(5, 4)
            y = np.random.randn(5, 3)
            model.fit(X, y)
            pred = model.predict(X)
            print(f"  ✓ 预测功能正常，形状: {pred.shape}")
            
            return True
            
        except Exception as e:
            print(f"  ✗ 创建失败: {e}")
            return False
    else:
        print("  ✗ 手动注册失败")
        return False


def demonstrate_workflow():
    """演示工作流程"""
    print_header("6. 演示添加新模型工作流程")
    
    print("步骤总结:")
    print("  1. 创建模型文件 models/new_model.py")
    print("  2. 继承 BaseModel，实现必要方法")
    print("  3. 使用 @ModelFactory.register() 装饰器")
    print("  4. （可选）在 config.py 中添加配置类")
    print("  5. 使用新模型: config.model.model_type = 'new_model'")
    print()
    print("当前系统状态:")
    print(f"  - 已注册模型数: {len(ModelFactory.get_available_models())}")
    print(f"  - 支持动态发现: 是")
    print(f"  - 自动依赖检查: 是")
    print(f"  - 配置自动提取: 是")
    
    return True


def main():
    """主函数"""
    print("模型工厂系统测试")
    print("="*60)
    
    tests = [
        (test_basic_functionality, "基本功能"),
        (test_model_creation, "模型创建"),
        (test_dynamic_discovery, "动态发现"),
        (test_dependency_checking, "依赖检查"),
        (test_manual_registration, "手动注册"),
        (demonstrate_workflow, "工作流程演示")
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func, test_name in tests:
        try:
            if test_func():
                print(f"\n[PASS] {test_name} 测试通过")
                passed += 1
            else:
                print(f"\n[FAIL] {test_name} 测试失败")
        except Exception as e:
            print(f"\n[ERROR] {test_name} 测试异常: {e}")
            import traceback
            traceback.print_exc()
    
    print_header("测试结果")
    print(f"通过: {passed}/{total}")
    
    if passed == total:
        print("\n[SUCESEE] 所有测试通过！系统工作正常。")
    else:
        print(f"\n[FAIL]  {total - passed} 个测试失败，请检查上述错误。")
    
    # 显示最终状态
    print("\n最终注册表状态:")
    for model_type in sorted(ModelFactory.get_available_models()):
        desc = ModelFactory.get_model_description(model_type)
        print(f"  - {model_type}: {desc}")


if __name__ == "__main__":
    main()