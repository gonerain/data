# 如何添加新模型 - 动态注册系统指南

## 概述

本项目使用动态注册模型工厂系统，使得添加新模型变得非常简单。系统支持三种添加方式：

1. **装饰器注册**（推荐）：使用 `@ModelFactory.register` 装饰器
2. **手动注册**：使用 `ModelFactory.manual_register()` 方法
3. **自动发现**：遵循命名规范，系统自动发现

## 方法一：使用装饰器（推荐，2步）

### 步骤1：创建模型文件

在 `models/` 目录下创建新模型文件，例如 `models/svm_model.py`：

```python
"""
支持向量机模型
"""

import numpy as np
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from models.base_model import BaseModel
from models.model_factory import ModelFactory


@ModelFactory.register(
    model_type='svm',                    # 模型类型标识符
    config_section='svm',                # 配置文件中对应的节名
    description='支持向量机回归 - 适合小样本，非线性问题',  # 模型描述
    dependencies=['sklearn']             # 依赖包列表
)
class SVMModel(BaseModel):
    """支持向量机回归模型"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        """构建SVM模型"""
        svm_config = self.config.get('svm', {})
        
        base_model = SVR(
            kernel=svm_config.get('kernel', 'rbf'),
            C=svm_config.get('C', 1.0),
            epsilon=svm_config.get('epsilon', 0.1)
        )
        
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        self.model.fit(X, y)
        self.feature_importance = None  # SVM没有特征重要性
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
```

### 步骤2：添加配置（可选）

如果需要自定义参数，在 `configs/config.py` 中添加：

```python
# 在合适的位置添加配置类
@dataclass
class SVMConfig:
    """SVM配置"""
    kernel: str = 'rbf'
    C: float = 1.0
    epsilon: float = 0.1

# 在 TrainingConfig 中添加
@dataclass
class TrainingConfig:
    # ... 其他配置 ...
    svm: SVMConfig = field(default_factory=SVMConfig)  # 添加这行
```

### 完成！

现在就可以使用新模型了：

```python
from configs.config import TrainingConfig
from models.model_factory import ModelFactory

# 创建配置
config = TrainingConfig()
config.model.model_type = 'svm'  # 使用新模型

# 创建模型
model = ModelFactory.create_model(config)
print(f"模型类型: {model.__class__.__name__}")
```

## 方法二：手动注册（3步）

### 步骤1：创建模型类（不装饰）

```python
# models/custom_model.py
"""
自定义模型
"""

import numpy as np
from models.base_model import BaseModel

class CustomModel(BaseModel):
    """自定义模型"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        
    def build_model(self):
        pass
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model = "custom"
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.ones((X.shape[0], 3))
```

### 步骤2：在需要的地方注册

```python
from models.model_factory import ModelFactory
from models.custom_model import CustomModel

# 手动注册
ModelFactory.manual_register(
    model_type='custom',
    model_class=CustomModel,
    description='自定义模型',
    dependencies=['numpy']
)
```

### 步骤3：使用

```python
config.model.model_type = 'custom'
model = ModelFactory.create_model(config)
```

## 方法三：动态发现（1步，最简单）

如果遵循命名规范，只需创建文件即可：

1. 创建 `models/{model_type}_model.py` 文件
2. 类名必须是 `{ModelType}Model` 格式（如 `SVMModel`、`LightGBMModel`）
3. 继承 `BaseModel`

系统会自动发现并注册！

## 验证新模型

创建测试文件验证：

```python
# test_new_model.py
"""
测试新模型
"""

from configs.config import TrainingConfig
from models.model_factory import ModelFactory

print("=== 测试新模型 ===\n")

# 1. 查看所有可用模型
print("1. 当前可用模型:")
for model_type in ModelFactory.get_available_models():
    desc = ModelFactory.get_model_description(model_type)
    print(f"   - {model_type}: {desc}")

# 2. 测试新模型
print("\n2. 测试新模型 'svm':")
config = TrainingConfig()
config.model.model_type = 'svm'

try:
    model = ModelFactory.create_model(config)
    print(f"   ✅ 成功创建: {model.__class__.__name__}")
    
    # 测试基本功能
    import numpy as np
    X = np.random.randn(10, 4)
    y = np.random.randn(10, 3)
    
    model.fit(X, y)
    pred = model.predict(X)
    print(f"   ✅ 预测形状: {pred.shape}")
    
except Exception as e:
    print(f"   ❌ 失败: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 测试完成 ===")
```

## 现有模型示例

### 随机森林模型
- 文件：`models/random_forest_model.py`
- 类型：`random_forest`
- 配置节：`random_forest`

### 线性回归模型
- 文件：`models/linear_regression_model.py`
- 类型：`linear_regression`
- 配置节：`linear_regression`

## 系统特性

1. **自动依赖检查**：如果缺少依赖，自动回退到随机森林
2. **配置自动提取**：从配置文件中自动提取模型参数
3. **动态发现**：支持插件式添加模型
4. **统一接口**：所有模型通过相同方式创建和使用
5. **类型安全**：所有模型继承 `BaseModel`，确保接口一致

## 注意事项

1. 所有模型必须继承 `BaseModel` 并实现必要方法
2. 模型类型标识符必须唯一
3. 配置节名应与模型类型一致（或通过 `config_section` 指定）
4. 依赖包列表用于检查是否安装，缺失时会回退到随机森林

## 快速开始模板

```python
# models/template_model.py
"""
{模型名称}模型
"""

import numpy as np
# 导入需要的库

from models.base_model import BaseModel
from models.model_factory import ModelFactory


@ModelFactory.register(
    model_type='{model_type}',
    config_section='{config_section}',
    description='{模型描述}',
    dependencies=['{依赖包1}', '{依赖包2}']
)
class {ModelType}Model(BaseModel):
    """{模型名称}模型"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        """构建模型"""
        # 从 config 中获取参数
        model_config = self.config.get('{config_section}', {})
        
        # 创建模型实例
        # self.model = ...
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        # self.model.fit(X, y)
        # self.feature_importance = ...
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        # return self.model.predict(X)
        pass
```

## 支持的命令

```python
# 获取所有可用模型
models = ModelFactory.get_available_models()

# 获取模型描述
desc = ModelFactory.get_model_description('random_forest')

# 创建模型
model = ModelFactory.create_model(config)

# 手动注册模型
ModelFactory.manual_register(...)

# 取消注册模型
ModelFactory.unregister('model_type')

# 清空注册表
ModelFactory.clear_registry()
```

## 故障排除

1. **模型未注册错误**：检查是否使用了正确的 `model_type`
2. **依赖缺失**：安装缺失的包或检查 `dependencies` 列表
3. **配置提取失败**：检查 `config_section` 是否与配置文件中的节名一致
4. **动态发现失败**：检查文件名和类名是否符合规范

---

通过这个动态注册系统，你可以轻松地扩展模型库，无需修改核心代码。