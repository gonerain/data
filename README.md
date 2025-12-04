# 焊机参数机器学习系统

## 项目概述

本项目基于焊机历史数据，使用机器学习方法预测焊机参数。系统可以根据前后带钢的厚度和牌号，预测最佳的电流、速度和压力参数。

**核心特性**：
- 🔧 **动态模型注册系统** - 插件式架构，轻松添加新模型
- 📊 **多模型支持** - 随机森林、线性回归、决策树等
- 🏭 **班组分析** - 支持甲、乙、丙、丁四个班组
- ⚡ **高性能** - 并行计算，快速训练和预测
- 🔄 **可扩展** - 模块化设计，易于维护和扩展

## 数据特征

### 输入特征（4个）：
1. 前行带钢厚度（5）
2. 后行带钢厚度（6）
3. 前行带钢内部牌号
4. 后行带钢内部牌号

### 预测目标（3个）：
1. 电流（7）
2. 速度（9）
3. 压力（8）

### 可选班组：
- 甲班
- 乙班
- 丙班
- 丁班

## 系统架构

### 动态注册模型工厂

本项目采用**插件式架构**，通过动态注册模型工厂实现高度可扩展性：

```
models/
├── base_model.py          # 基础模型抽象类
├── model_factory.py       # ★ 动态注册模型工厂（核心）
├── random_forest_model.py # 随机森林模型
├── linear_regression_model.py # 线性回归模型
└── example_model.py       # 决策树示例模型
```

### 核心特性：
1. **无需修改核心代码** - 添加新模型不影响现有系统
2. **自动注册** - 使用装饰器的模型自动注册
3. **动态发现** - 符合命名规范的模型自动发现
4. **依赖检查** - 缺失依赖时自动回退到随机森林
5. **配置驱动** - 所有参数通过配置文件管理

## 文件结构

```
.
├── dataset/               # 数据文件
│   ├── data.csv          # 原始数据文件
│   ├── test_data.csv     # 测试数据
│   └── all_grades.csv    # 所有牌号数据
├── configs/              # 配置文件
│   └── config.py         # 训练和模型配置
├── models/               # 模型实现
│   ├── base_model.py     # 基础模型抽象类
│   ├── model_factory.py  # 动态注册模型工厂 ★
│   ├── random_forest_model.py
│   ├── linear_regression_model.py
│   └── example_model.py
├── scripts/              # 功能脚本
│   ├── train.py          # 训练脚本
│   ├── evaluate.py       # 评估脚本
│   ├── predict.py        # 预测脚本
│   └── test_system.py    # 系统测试
├── saved_models/         # 保存的模型
├── results/              # 结果输出
├── logs/                 # 日志文件
├── run.py                # 主启动脚本
├── ADD_NEW_MODEL.md      # 添加新模型指南
└── README.md             # 本说明文件
```

## 快速开始

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 启动系统
```bash
python run.py
```

### 3. 使用命令行
```bash
# 训练模型
python scripts/train.py --team 甲 --model random_forest --max_samples 2000

# 比较模型
python scripts/evaluate.py compare_models --team 甲

# 交互式预测
python scripts/predict.py interactive
```

## 添加新模型

系统支持三种方式添加新模型：

### 方法1：使用装饰器（推荐）
```python
# models/new_model.py
@ModelFactory.register(
    model_type='new_model',
    config_section='new_model',
    description='模型描述',
    dependencies=['sklearn']
)
class NewModel(BaseModel):
    # 实现模型
    pass
```

### 方法2：手动注册
```python
ModelFactory.manual_register(
    model_type='new_model',
    model_class=NewModel,
    description='模型描述',
    dependencies=['sklearn']
)
```

### 方法3：动态发现
1. 创建 `models/{model_type}_model.py`
2. 类名必须是 `{ModelType}Model`
3. 继承 `BaseModel`

**详细指南请查看 [ADD_NEW_MODEL.md](ADD_NEW_MODEL.md)**

## 支持的模型

### 当前已实现：
1. **随机森林回归** (`random_forest`)
   - 集成学习，抗过拟合能力强
   - 默认配置：100棵树，最大深度10

2. **线性回归** (`linear_regression`)
   - 简单快速，适合线性关系
   - 多输出回归

3. **决策树回归** (`decision_tree`)
   - 简单易懂，可解释性强
   - 示例模型，演示如何添加新模型

### 计划支持：
- XGBoost回归
- 梯度提升回归
- 支持向量机回归
- 神经网络

## 模型性能

基于甲班2000条数据的测试结果：

| 参数 | MSE | MAE | R² |
|------|-----|-----|----|
| 电流（7） | 0.3337 | 0.3779 | 0.9272 |
| 速度（9） | 0.2581 | 0.3214 | 0.9520 |
| 压力（8） | 0.8213 | 0.5758 | 0.9665 |

## 示例预测

输入：
- 前行带钢厚度: 1.00
- 后行带钢厚度: 1.00
- 前行带钢牌号: M3A43(外板）
- 后行带钢牌号: M3A43(外板）

输出：
- 电流（7）: 17.52
- 速度（9）: 12.61
- 压力（8）: 13.23

## 配置系统

### 模型配置示例
```python
# configs/config.py
@dataclass
class RandomForestConfig:
    n_estimators: int = 100      # 树的数量
    max_depth: int = 10          # 最大深度
    min_samples_split: int = 5   # 分裂最小样本数
    min_samples_leaf: int = 2    # 叶节点最小样本数
    random_state: int = 42       # 随机种子
    n_jobs: int = -1             # 使用所有CPU核心
```

### 训练配置
```python
config = TrainingConfig()
config.data.team = '甲'           # 选择班组
config.model.model_type = 'random_forest'  # 选择模型
config.data.max_samples = 2000    # 最大样本数
```

## 技术细节

### 使用的算法
- 随机森林回归 (Random Forest Regressor)
- 线性回归 (Linear Regression)
- 决策树回归 (Decision Tree Regressor)
- 多输出回归 (MultiOutputRegressor)

### 数据预处理
1. GBK编码读取CSV文件
2. 缺失值处理
3. 分类特征编码 (Label Encoding)
4. 数值特征标准化 (Standard Scaling)

### 模型评估
- 均方误差 (MSE)
- 平均绝对误差 (MAE)
- 决定系数 (R²)
- 特征重要性分析

## 系统测试

运行系统测试确保一切正常：
```bash
python scripts/test_system.py
```

## 注意事项

1. 数据文件使用GBK编码
2. 首次运行需要安装依赖
3. 完整数据集较大，训练可能需要较长时间
4. 建议从2000条数据开始测试

## 依赖安装

```bash
# 基础依赖
pip install pandas numpy scikit-learn joblib

# 可选依赖（用于特定模型）
pip install xgboost lightgbm
```

## 未来改进方向

1. **更多算法**：添加XGBoost、LightGBM、神经网络等
2. **特征工程**：自动特征生成和选择
3. **自动化**：自动超参数调优
4. **可视化**：训练过程可视化，结果图表
5. **部署**：Web API接口，实时预测
6. **监控**：模型性能监控和自动重训练

## 开发指南

### 添加新模型的完整流程：
1. 阅读 [ADD_NEW_MODEL.md](ADD_NEW_MODEL.md)
2. 创建模型文件 `models/new_model.py`
3. 使用装饰器注册模型
4. （可选）在 `configs/config.py` 中添加配置类
5. 测试新模型
6. 提交代码

### 代码规范：
- 所有模型必须继承 `BaseModel`
- 使用类型提示
- 添加文档字符串
- 遵循PEP8编码规范

## 故障排除

### 常见问题：
1. **导入错误**：确保在项目根目录运行脚本
2. **模型未注册**：检查是否使用了正确的 `model_type`
3. **依赖缺失**：安装缺失的包或检查 `dependencies` 列表
4. **内存不足**：减少 `max_samples` 或使用更简单的模型

### 调试建议：
```bash
# 检查模型注册状态
python -c "from models.model_factory import ModelFactory; print(ModelFactory.get_available_models())"

# 测试数据加载
python scripts/test_data_loading.py

# 运行简单测试
python -c "from configs.config import TrainingConfig; from models.model_factory import ModelFactory; config=TrainingConfig(); config.model.model_type='random_forest'; model=ModelFactory.create_model(config); print('OK')"
```

## 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

MIT License

## 作者

焊机参数机器学习项目团队

---

**提示**：详细的使用指南和API文档请查看项目中的其他文档文件。