"""简化版模型训练"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("简化版焊机参数机器学习模型")
print("=" * 60)

# 选择班组
selected_team = '甲'
print(f"选择班组: {selected_team}")

# 加载数据（只加载部分数据以避免内存问题）
try:
    print("正在加载数据...")
    df = pd.read_csv('dataset/data.csv', encoding='gbk')
    print(f"总数据量: {len(df)} 条")
    
    # 过滤指定班组的数据
    df_team = df[df['班组'] == selected_team].copy()
    print(f"班组 {selected_team} 的数据量: {len(df_team)} 条")
    
    # 为了测试，只取前2000条数据
    if len(df_team) > 2000:
        df_team = df_team.head(2000)
        print(f"使用前2000条数据进行训练")
    
    # 处理缺失值
    df_team = df_team.replace('#N/A', np.nan)
    
    # 定义特征和目标列
    feature_columns = ['前行带钢厚度（5）', '后行带钢厚度（6）', '前行带钢内部牌号', '后行带钢内部牌号']
    target_columns = ['电流（7）', '速度（9）', '压力（8）']
    
    # 删除缺失值
    df_team = df_team.dropna(subset=feature_columns + target_columns)
    print(f"清理后数据量: {len(df_team)} 条")
    
    # 转换数据类型
    for col in ['前行带钢厚度（5）', '后行带钢厚度（6）']:
        df_team[col] = pd.to_numeric(df_team[col], errors='coerce')
    
    for col in ['电流（7）', '速度（9）', '压力（8）']:
        df_team[col] = pd.to_numeric(df_team[col], errors='coerce')
    
    # 再次删除转换失败的行
    df_team = df_team.dropna(subset=feature_columns + target_columns)
    print(f"最终数据量: {len(df_team)} 条")
    
    # 准备特征
    print("\n准备特征...")
    
    # 提取数值特征
    X_numeric = df_team[['前行带钢厚度（5）', '后行带钢厚度（6）']].values
    
    # 对分类特征进行编码
    label_encoders = {}
    X_encoded = []
    
    for col in ['前行带钢内部牌号', '后行带钢内部牌号']:
        le = LabelEncoder()
        encoded = le.fit_transform(df_team[col].astype(str))
        X_encoded.append(encoded.reshape(-1, 1))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} 个类别")
    
    X_encoded = np.hstack(X_encoded)
    
    # 合并所有特征
    X = np.hstack([X_numeric, X_encoded])
    
    # 提取目标变量
    y = df_team[target_columns].values
    
    print(f"特征维度: {X.shape}")
    print(f"目标维度: {y.shape}")
    
    # 划分训练集和测试集
    print("\n划分数据集...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建并训练模型
    print("\n训练模型...")
    base_model = RandomForestRegressor(
        n_estimators=50,  # 减少树的数量以加快训练
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model = MultiOutputRegressor(base_model)
    model.fit(X_train_scaled, y_train)
    
    # 评估模型
    print("\n模型评估:")
    y_pred = model.predict(X_test_scaled)
    
    for i, target_name in enumerate(target_columns):
        mse = mean_squared_error(y_test[:, i], y_pred[:, i])
        mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
        r2 = r2_score(y_test[:, i], y_pred[:, i])
        
        print(f"\n{target_name}:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R2: {r2:.4f}")  # 改为R2避免编码问题
    
    # 保存模型
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'team': selected_team,
        'feature_columns': feature_columns,
        'target_columns': target_columns
    }
    
    model_file = f'welding_model_{selected_team}_simple.joblib'
    joblib.dump(model_data, model_file)
    print(f"\n模型已保存到: {model_file}")
    
    # 示例预测
    print("\n" + "=" * 60)
    print("示例预测:")
    print("=" * 60)
    
    # 从测试集中取一个示例
    sample_idx = 0
    sample_features = X_test[sample_idx]
    sample_target = y_test[sample_idx]
    
    # 获取原始牌号
    front_brand = label_encoders['前行带钢内部牌号'].inverse_transform([int(sample_features[2])])[0]
    back_brand = label_encoders['后行带钢内部牌号'].inverse_transform([int(sample_features[3])])[0]
    
    print(f"输入参数:")
    print(f"  前行带钢厚度: {sample_features[0]:.2f}")
    print(f"  后行带钢厚度: {sample_features[1]:.2f}")
    print(f"  前行带钢牌号: {front_brand}")
    print(f"  后行带钢牌号: {back_brand}")
    
    # 准备输入数据
    input_features = sample_features.reshape(1, -1)
    input_scaled = scaler.transform(input_features)
    
    # 预测
    prediction = model.predict(input_scaled)[0]
    
    print(f"\n预测结果:")
    for i, param in enumerate(target_columns):
        print(f"  {param}: {prediction[i]:.2f}")
    
    print(f"\n实际值:")
    for i, param in enumerate(target_columns):
        print(f"  {param}: {sample_target[i]:.2f}")
    
    print("\n" + "=" * 60)
    print("训练完成!")
    print("=" * 60)
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()