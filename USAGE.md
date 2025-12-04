# 焊机参数机器学习系统 - 使用说明

## 系统架构

```
.
├── config.py              # 配置文件
├── data_loader.py         # 数据加载和预处理
├── models/                # 模型定义
│   ├── base_model.py     # 基础模型类
│   ├── random_forest_model.py  # 随机森林模型
│   └── model_factory.py  # 模型工厂
├── train.py              # 统一训练脚本
├── evaluate.py           # 评估和比较脚本
├── predict.py            # 预测脚本
├── saved_models/         # 保存的模型（自动创建）
├── results/              # 评估结果（自动创建）
└── logs/                 # 日志文件（自动创建）
```

## 安装依赖

```bash
pip install pandas numpy scikit-learn joblib
# 可选: 如果需要其他模型
pip install xgboost
```

## 快速开始

### 1. 训练模型

训练甲班随机森林模型（使用2000条数据）：
```bash
python train.py --team 甲 --model random_forest --max_samples 2000
```

训练乙班XGBoost模型（使用所有数据）：
```bash
python train.py --team 乙 --model xgboost --max_samples 0
```

### 2. 比较不同模型

比较甲班的不同模型：
```bash
python evaluate.py compare_models --team 甲
```

比较不同班组的随机森林模型：
```bash
python evaluate.py compare_teams --model random_forest
```

### 3. 使用模型预测

交互式预测：
```bash
python predict.py interactive --model saved_models/welding_model_甲_random_forest_20241204_143022.joblib
```

批量预测：
```bash
python predict.py batch --model saved_models/模型文件.joblib --input 输入文件.csv --output 输出文件.csv
```

列出所有模型：
```bash
python predict.py list
```

## 详细用法

### 训练脚本 (train.py)

```bash
# 基本用法
python train.py --team 甲 --model random_forest

# 使用所有数据
python train.py --team 乙 --model xgboost --max_samples 0

# 指定实验名称
python train.py --team 丙 --model gradient_boosting --experiment "丙班梯度提升实验"

# 关闭详细输出
python train.py --team 丁 --model random_forest --verbose False
```

**参数说明：**
- `--team`: 选择班组（甲、乙、丙、丁）
- `--model`: 选择模型类型（random_forest、xgboost、gradient_boosting）
- `--max_samples`: 最大样本数（0表示使用所有数据）
- `--experiment`: 实验名称
- `--verbose`: 是否显示详细输出

### 评估脚本 (evaluate.py)

```bash
# 比较不同模型
python evaluate.py compare_models --team 甲 --max_samples 2000

# 比较不同班组
python evaluate.py compare_teams --model random_forest --max_samples 2000
```

### 预测脚本 (predict.py)

```bash
# 交互式预测
python predict.py interactive --model saved_models/模型文件.joblib

# 批量预测
python predict.py batch --model saved_models/模型文件.joblib --input data/test.csv --output data/predictions.csv

# 列出所有模型
python predict.py list --dir saved_models
```

## 配置文件说明

系统使用 `config.py` 管理所有配置，主要配置类：

1. **DataConfig**: 数据配置
   - `data_path`: 数据文件路径
   - `team`: 班组选择
   - `max_samples`: 最大样本数
   - `feature_columns`: 特征列
   - `target_columns`: 目标列

2. **ModelConfig**: 模型配置
   - `model_type`: 模型类型
   - `model_save_path`: 模型保存路径
   - `model_name`: 模型名称

3. **具体模型配置**:
   - `RandomForestConfig`: 随机森林参数
   - `XGBoostConfig`: XGBoost参数
   - `GradientBoostingConfig`: 梯度提升参数

## 自定义配置

### 创建自定义配置

```python
# custom_config.py
from config import TrainingConfig, DataConfig, ModelConfig, RandomForestConfig

# 创建自定义配置
custom_config = TrainingConfig(
    experiment_name="自定义实验",
    data=DataConfig(
        team="甲",
        max_samples=5000,
        test_size=0.3
    ),
    model=ModelConfig(
        model_type="random_forest",
        model_name="custom_model"
    ),
    random_forest=RandomForestConfig(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10
    )
)
```

### 在代码中使用自定义配置

```python
from train import train_model
from custom_config import custom_config

# 使用自定义配置训练
model, results, metadata = train_model(custom_config)
```

## 模型输出

### 训练输出

训练完成后会生成：
1. **模型文件**: `saved_models/welding_model_甲_random_forest_20241204_143022.joblib`
2. **特征重要性文件**: `saved_models/welding_model_甲_random_forest_20241204_143022_importance.json`
3. **评估结果**: 控制台显示R²、MSE、MAE等指标

### 特征重要性示例

```json
[
  {
    "feature": "后行带钢厚度（6）",
    "importance": 0.4521
  },
  {
    "feature": "前行带钢厚度（5）",
    "importance": 0.3215
  },
  {
    "feature": "后行带钢内部牌号",
    "importance": 0.1234
  },
  {
    "feature": "前行带钢内部牌号",
    "importance": 0.1030
  }
]
```

## 扩展系统

### 添加新模型

1. 在 `models/` 目录下创建新模型类
2. 在 `models/model_factory.py` 中添加模型创建逻辑
3. 在 `config.py` 中添加对应的配置类

### 添加新特征

1. 修改 `config.py` 中的 `DataConfig.feature_columns`
2. 更新 `data_loader.py` 中的特征处理逻辑

## 故障排除

### 常见问题

1. **编码错误**: 确保数据文件使用GBK编码
2. **内存不足**: 减少 `max_samples` 参数值
3. **模型加载失败**: 检查模型文件路径和格式
4. **预测错误**: 检查输入特征是否完整

### 调试模式

```bash
# 启用详细输出
python train.py --team 甲 --model random_forest --verbose True
```

## 性能优化建议

1. **数据层面**:
   - 使用 `max_samples` 控制训练数据量
   - 考虑特征选择减少维度
   - 使用数据采样平衡类别

2. **模型层面**:
   - 调整模型超参数
   - 使用更高效的算法（如LightGBM）
   - 考虑模型集成

3. **系统层面**:
   - 使用GPU加速（如果支持）
   - 并行处理多个班组
   - 缓存预处理结果

## 联系方式

如有问题或建议，请提交Issue或联系项目维护者。

---

**版本**: 1.0.0  
**最后更新**: 2024-12-04  
**作者**: 焊机参数机器学习项目组