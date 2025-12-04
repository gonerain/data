# 焊机参数机器学习系统

## 项目概述

本项目基于焊机历史数据，使用机器学习方法预测焊机参数。系统可以根据前后带钢的厚度和牌号，预测最佳的电流、速度和压力参数。

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

## 文件结构

```
.
├── dataset/
│   ├── data.csv          # 原始数据文件
│   ├── test_data.csv     # 测试数据
│   └── all_grades.csv    # 所有牌号数据
├── welding_ml_model.py   # 完整机器学习模型
├── train_model_simple.py # 简化版训练脚本
├── use_model.py          # 模型使用脚本
├── test_data_loading.py  # 数据测试脚本
└── README_CN.md          # 本说明文件
```

## 使用方法

### 1. 数据测试
首先测试数据加载是否正确：
```bash
python test_data_loading.py
```

### 2. 训练模型
训练简化版模型（使用甲班数据）：
```bash
python train_model_simple.py
```

### 3. 使用模型进行预测
```bash
python use_model.py
```

### 4. 训练完整模型
如果要训练完整模型（使用所有数据）：
```bash
python welding_ml_model.py
```

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

## 扩展功能

### 训练不同班组的模型
修改 `train_model_simple.py` 中的 `selected_team` 变量：
```python
# 选择班组
selected_team = '甲'  # 可改为 '乙', '丙', '丁'
```

### 使用完整数据集
修改 `train_model_simple.py`，移除数据限制：
```python
# 注释掉这行以使用所有数据
# if len(df_team) > 2000:
#     df_team = df_team.head(2000)
#     print(f"使用前2000条数据进行训练")
```

## 技术细节

### 使用的算法
- 随机森林回归 (Random Forest Regressor)
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

## 注意事项

1. 数据文件使用GBK编码
2. 模型训练需要scikit-learn、pandas、numpy等库
3. 首次运行需要安装依赖：
   ```bash
   pip install pandas numpy scikit-learn joblib
   ```
4. 完整数据集较大，训练可能需要较长时间

## 未来改进方向

1. 尝试其他机器学习算法（如梯度提升、神经网络）
2. 添加特征工程（如厚度差异、牌号相似度）
3. 实现模型集成
4. 开发Web界面
5. 添加实时数据更新和在线学习功能

## 作者

焊机参数机器学习项目

## 许可证

MIT License