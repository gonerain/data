"""训练所有班组的模型"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

def train_team_model(team, max_samples=2000):
    """训练指定班组的模型"""
    print(f"\n{'='*60}")
    print(f"训练 {team} 班模型")
    print(f"{'='*60}")
    
    try:
        # 加载数据
        print("正在加载数据...")
        df = pd.read_csv('dataset/data.csv', encoding='gbk')
        
        # 过滤指定班组的数据
        df_team = df[df['班组'] == team].copy()
        print(f"班组 {team} 的数据量: {len(df_team)} 条")
        
        # 限制样本数量
        if len(df_team) > max_samples:
            df_team = df_team.head(max_samples)
            print(f"使用前{max_samples}条数据进行训练")
        
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
        
        if len(df_team) < 100:
            print(f"警告: {team}班数据量不足 ({len(df_team)}条)，跳过训练")
            return None
        
        # 准备特征
        print("准备特征...")
        
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
        print("划分数据集...")
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
        print("训练模型...")
        base_model = RandomForestRegressor(
            n_estimators=50,
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
        
        results = {}
        for i, target_name in enumerate(target_columns):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            results[target_name] = {'MSE': mse, 'MAE': mae, 'R2': r2}
            
            print(f"\n{target_name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R2: {r2:.4f}")
        
        # 保存模型
        model_data = {
            'model': model,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'team': team,
            'feature_columns': feature_columns,
            'target_columns': target_columns,
            'results': results
        }
        
        model_file = f'welding_model_{team}.joblib'
        joblib.dump(model_data, model_file)
        print(f"\n模型已保存到: {model_file}")
        
        return model_data
        
    except Exception as e:
        print(f"训练{team}班模型时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_teams():
    """比较不同班组的模型性能"""
    print("=" * 60)
    print("班组模型性能比较")
    print("=" * 60)
    
    teams = ['甲', '乙', '丙', '丁']
    all_results = {}
    
    for team in teams:
        model_file = f'welding_model_{team}.joblib'
        if os.path.exists(model_file):
            try:
                model_data = joblib.load(model_file)
                results = model_data.get('results', {})
                all_results[team] = results
                
                print(f"\n{team}班:")
                for target, metrics in results.items():
                    print(f"  {target}: R2={metrics['R2']:.4f}, MAE={metrics['MAE']:.4f}")
            except Exception as e:
                print(f"加载{team}班模型时出错: {e}")
        else:
            print(f"\n{team}班: 模型文件不存在")
    
    return all_results

def main():
    """主函数"""
    print("=" * 60)
    print("焊机参数机器学习 - 多班组训练")
    print("=" * 60)
    
    # 选择要训练的班组
    teams_to_train = ['甲', '乙', '丙', '丁']
    
    print(f"将训练以下班组: {teams_to_train}")
    print(f"每个班组最多使用2000条数据")
    
    # 训练所有班组
    trained_models = {}
    for team in teams_to_train:
        model_data = train_team_model(team)
        if model_data:
            trained_models[team] = model_data
    
    # 比较性能
    if trained_models:
        compare_teams()
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"成功训练了 {len(trained_models)} 个班组模型")
        
        # 创建使用说明
        print("\n使用说明:")
        print("1. 使用 use_model.py 加载特定班组的模型进行预测")
        print("2. 模型文件格式: welding_model_<班组>.joblib")
        print("3. 示例: welding_model_甲.joblib")
    else:
        print("\n没有成功训练任何模型")

if __name__ == "__main__":
    main()