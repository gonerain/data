"""
数据加载和预处理模块
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from configs.config import DataConfig


class DataLoader:
    """数据加载器"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        if self.config.verbose:
            print(f"正在加载数据从: {self.config.data_path}")
            print(f"选择班组: {self.config.team}")
        
        # 读取数据
        try:
            df = pd.read_csv(self.config.data_path, encoding=self.config.encoding)
        except UnicodeDecodeError:
            # 如果指定编码失败，尝试utf-8
            df = pd.read_csv(self.config.data_path, encoding='utf-8')
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """预处理数据"""
        # 过滤指定班组的数据
        df_team = df[df['班组'] == self.config.team].copy()
        
        if self.config.verbose:
            print(f"班组 {self.config.team} 的数据量: {len(df_team)} 条")
        
        # 限制样本数量
        if self.config.max_samples is not None and len(df_team) > self.config.max_samples:
            df_team = df_team.head(self.config.max_samples)
            if self.config.verbose:
                print(f"使用前{self.config.max_samples}条数据进行训练")
        
        # 处理缺失值
        df_team = df_team.replace('#N/A', np.nan)
        
        # 删除缺失值
        df_team = df_team.dropna(subset=self.config.feature_columns + self.config.target_columns)
        
        if self.config.verbose:
            print(f"清理后数据量: {len(df_team)} 条")
        
        # 转换数据类型
        numeric_features = ['前行带钢厚度（5）', '后行带钢厚度（6）']
        for col in numeric_features:
            if col in df_team.columns:
                df_team[col] = pd.to_numeric(df_team[col], errors='coerce')
        
        for col in self.config.target_columns:
            if col in df_team.columns:
                df_team[col] = pd.to_numeric(df_team[col], errors='coerce')
        
        # 再次删除转换失败的行
        df_team = df_team.dropna(subset=self.config.feature_columns + self.config.target_columns)
        
        if self.config.verbose:
            print(f"最终数据量: {len(df_team)} 条")
            
            # 显示数据统计
            print("\n数据统计:")
            for col in self.config.feature_columns + self.config.target_columns:
                if col in df_team.columns and df_team[col].dtype in [np.float64, np.int64]:
                    print(f"  {col}: min={df_team[col].min():.2f}, max={df_team[col].max():.2f}, mean={df_team[col].mean():.2f}")
        
        return df_team
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """准备特征数据"""
        if self.config.verbose:
            print("\n准备特征...")
        
        # 提取数值特征
        numeric_cols = ['前行带钢厚度（5）', '后行带钢厚度（6）']
        X_numeric = df[numeric_cols].values
        
        # 对分类特征进行编码
        X_encoded = []
        categorical_cols = ['前行带钢内部牌号', '后行带钢内部牌号']
        
        for col in categorical_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str))
            X_encoded.append(encoded.reshape(-1, 1))
            self.label_encoders[col] = le
            
            if self.config.verbose:
                print(f"  {col}: {len(le.classes_)} 个类别")
        
        X_encoded = np.hstack(X_encoded)
        
        # 合并所有特征
        X = np.hstack([X_numeric, X_encoded])
        
        # 提取目标变量
        y = df[self.config.target_columns].values
        
        if self.config.verbose:
            print(f"特征维度: {X.shape}")
            print(f"目标维度: {y.shape}")
        
        # 准备元数据
        metadata = {
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'label_encoders': self.label_encoders,
            'feature_names': numeric_cols + categorical_cols
        }
        
        return X, y, metadata
    
    def split_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """划分训练集和测试集"""
        if self.config.verbose:
            print("\n划分数据集...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state
        )
        
        if self.config.verbose:
            print(f"训练集大小: {X_train.shape[0]}")
            print(f"测试集大小: {X_test.shape[0]}")
        
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """标准化特征"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        return X_train_scaled, X_test_scaled
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """获取处理后的数据"""
        # 加载数据
        df = self.load_data()
        
        # 预处理
        df_processed = self.preprocess_data(df)
        
        # 准备特征
        X, y, metadata = self.prepare_features(df_processed)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # 标准化特征
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        metadata['scaler'] = self.scaler
        metadata['data_info'] = {
            'team': self.config.team,
            'total_samples': len(df_processed),
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0]
        }
        
        return X_train_scaled, X_test_scaled, y_train, y_test, metadata


def load_and_preprocess_data(config: DataConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """快速加载和预处理数据"""
    loader = DataLoader(config)
    return loader.get_data()