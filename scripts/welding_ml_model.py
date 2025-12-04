"""
焊机参数机器学习模型
根据前后带钢厚度和牌号预测电流、速度、压力参数
"""

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

class WeldingParameterPredictor:
    def __init__(self, team='甲'):
        """
        初始化焊机参数预测器
        
        Parameters:
        -----------
        team : str
            选择的班组，可选值：'甲', '乙', '丙', '丁'
        """
        self.team = team
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.model = None
        self.feature_columns = [
            '前行带钢厚度（5）',
            '后行带钢厚度（6）',
            '前行带钢内部牌号',
            '后行带钢内部牌号'
        ]
        self.target_columns = ['电流（7）', '速度（9）', '压力（8）']
        
    def load_and_preprocess_data(self, filepath='dataset/data.csv'):
        """
        加载和预处理数据
        
        Parameters:
        -----------
        filepath : str
            CSV文件路径
            
        Returns:
        --------
        pandas.DataFrame
            预处理后的数据
        """
        print(f"正在加载数据从: {filepath}")
        print(f"选择班组: {self.team}")
        
        # 使用GBK编码读取数据
        try:
            df = pd.read_csv(filepath, encoding='gbk')
        except UnicodeDecodeError:
            # 如果GBK失败，尝试其他编码
            df = pd.read_csv(filepath, encoding='utf-8')
        
        # 过滤指定班组的数据
        df_team = df[df['班组'] == self.team].copy()
        print(f"班组 {self.team} 的数据量: {len(df_team)} 条")
        
        # 处理缺失值
        df_team = df_team.replace('#N/A', np.nan)
        df_team = df_team.dropna(subset=self.feature_columns + self.target_columns)
        print(f"清理后数据量: {len(df_team)} 条")
        
        # 转换数据类型
        for col in ['前行带钢厚度（5）', '后行带钢厚度（6）']:
            df_team[col] = pd.to_numeric(df_team[col], errors='coerce')
        
        for col in ['电流（7）', '速度（9）', '压力（8）']:
            df_team[col] = pd.to_numeric(df_team[col], errors='coerce')
        
        # 再次删除转换失败的行
        df_team = df_team.dropna(subset=self.feature_columns + self.target_columns)
        print(f"最终数据量: {len(df_team)} 条")
        
        return df_team
    
    def prepare_features(self, df):
        """
        准备特征数据
        
        Parameters:
        -----------
        df : pandas.DataFrame
            原始数据
            
        Returns:
        --------
        tuple
            (X_features, X_numeric, X_encoded, y)
        """
        # 提取数值特征
        X_numeric = df[['前行带钢厚度（5）', '后行带钢厚度（6）']].values
        
        # 对分类特征进行编码
        X_encoded = []
        for i, col in enumerate(['前行带钢内部牌号', '后行带钢内部牌号']):
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str))
            X_encoded.append(encoded.reshape(-1, 1))
            self.label_encoders[col] = le
        
        X_encoded = np.hstack(X_encoded)
        
        # 合并所有特征
        X_features = np.hstack([X_numeric, X_encoded])
        
        # 提取目标变量
        y = df[self.target_columns].values
        
        return X_features, y
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        """
        训练机器学习模型
        
        Parameters:
        -----------
        X : numpy.ndarray
            特征数据
        y : numpy.ndarray
            目标数据
        test_size : float
            测试集比例
        random_state : int
            随机种子
        """
        print("\n开始训练模型...")
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"训练集大小: {X_train.shape[0]}")
        print(f"测试集大小: {X_test.shape[0]}")
        
        # 标准化特征
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 创建并训练模型
        # 使用MultiOutputRegressor包装RandomForestRegressor来处理多输出回归
        base_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.model = MultiOutputRegressor(base_model)
        self.model.fit(X_train_scaled, y_train)
        
        # 评估模型
        print("\n模型评估:")
        y_pred = self.model.predict(X_test_scaled)
        
        for i, target_name in enumerate(self.target_columns):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            print(f"\n{target_name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
        
        return X_test_scaled, y_test, y_pred
    
    def predict(self, front_thickness, back_thickness, front_brand, back_brand):
        """
        使用训练好的模型进行预测
        
        Parameters:
        -----------
        front_thickness : float
            前行带钢厚度
        back_thickness : float
            后行带钢厚度
        front_brand : str
            前行带钢内部牌号
        back_brand : str
            后行带钢内部牌号
            
        Returns:
        --------
        dict
            预测结果
        """
        if self.model is None:
            raise ValueError("模型尚未训练，请先调用train_model方法")
        
        # 准备输入数据
        numeric_features = np.array([[front_thickness, back_thickness]])
        
        # 编码分类特征
        encoded_features = []
        for col, value in zip(['前行带钢内部牌号', '后行带钢内部牌号'], 
                            [front_brand, back_brand]):
            le = self.label_encoders.get(col)
            if le is None:
                raise ValueError(f"未找到{col}的编码器")
            
            try:
                encoded = le.transform([str(value)])
                encoded_features.append(encoded)
            except ValueError:
                # 如果遇到新的类别，使用最常见的类别
                print(f"警告: 牌号 '{value}' 不在训练数据中，使用默认值")
                encoded = le.transform([le.classes_[0]])
                encoded_features.append(encoded)
        
        encoded_features = np.array(encoded_features).reshape(1, -1)
        
        # 合并特征
        features = np.hstack([numeric_features, encoded_features])
        
        # 标准化
        features_scaled = self.scaler.transform(features)
        
        # 预测
        predictions = self.model.predict(features_scaled)[0]
        
        return {
            '电流（7）': predictions[0],
            '速度（9）': predictions[1],
            '压力（8）': predictions[2]
        }
    
    def save_model(self, filepath='welding_model.joblib'):
        """
        保存模型到文件
        
        Parameters:
        -----------
        filepath : str
            模型保存路径
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'team': self.team,
            'feature_columns': self.feature_columns,
            'target_columns': self.target_columns
        }
        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath='welding_model.joblib'):
        """
        从文件加载模型
        
        Parameters:
        -----------
        filepath : str
            模型文件路径
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.team = model_data['team']
        self.feature_columns = model_data['feature_columns']
        self.target_columns = model_data['target_columns']
        print(f"模型已从 {filepath} 加载")
        print(f"模型训练班组: {self.team}")

def main():
    """主函数"""
    print("=" * 60)
    print("焊机参数机器学习模型")
    print("=" * 60)
    
    # 选择班组
    teams = ['甲', '乙', '丙', '丁']
    print(f"可选班组: {teams}")
    
    # 这里可以选择不同的班组进行训练
    selected_team = '甲'  # 可以修改为其他班组
    
    # 创建预测器
    predictor = WeldingParameterPredictor(team=selected_team)
    
    # 加载和预处理数据
    df = predictor.load_and_preprocess_data('dataset/data.csv')
    
    # 准备特征
    X, y = predictor.prepare_features(df)
    
    # 训练模型
    X_test, y_test, y_pred = predictor.train_model(X, y)
    
    # 保存模型
    predictor.save_model(f'welding_model_{selected_team}.joblib')
    
    # 示例预测
    print("\n" + "=" * 60)
    print("示例预测:")
    print("=" * 60)
    
    # 从数据中取一个示例
    sample = df.iloc[0]
    front_thickness = sample['前行带钢厚度（5）']
    back_thickness = sample['后行带钢厚度（6）']
    front_brand = sample['前行带钢内部牌号']
    back_brand = sample['后行带钢内部牌号']
    
    print(f"输入参数:")
    print(f"  前行带钢厚度: {front_thickness}")
    print(f"  后行带钢厚度: {back_thickness}")
    print(f"  前行带钢牌号: {front_brand}")
    print(f"  后行带钢牌号: {back_brand}")
    
    try:
        prediction = predictor.predict(front_thickness, back_thickness, 
                                     front_brand, back_brand)
        print(f"\n预测结果:")
        for param, value in prediction.items():
            print(f"  {param}: {value:.2f}")
        
        print(f"\n实际值:")
        print(f"  电流（7）: {sample['电流（7）']}")
        print(f"  速度（9）: {sample['速度（9）']}")
        print(f"  压力（8）: {sample['压力（8）']}")
    except Exception as e:
        print(f"预测时出错: {e}")

if __name__ == "__main__":
    main()