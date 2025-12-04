"""
基础模型类
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class BaseModel(ABC):
    """基础模型抽象类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.feature_importance = None
        
    @abstractmethod
    def build_model(self):
        """构建模型"""
        pass
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        pass
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                 target_names: list = None) -> Dict[str, Any]:
        """评估模型"""
        y_pred = self.predict(X_test)
        
        results = {}
        n_targets = y_test.shape[1]
        
        if target_names is None:
            target_names = [f'target_{i}' for i in range(n_targets)]
        
        for i in range(n_targets):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            results[target_names[i]] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        # 总体评估
        results['overall'] = {
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return results
    
    def get_feature_importance(self) -> np.ndarray:
        """获取特征重要性"""
        return self.feature_importance
    
    def save_model(self, filepath: str):
        """保存模型"""
        import joblib
        model_data = {
            'model': self.model,
            'config': self.config,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        
    def load_model(self, filepath: str):
        """加载模型"""
        import joblib
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.config = model_data['config']
        self.feature_importance = model_data.get('feature_importance')
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': self.__class__.__name__,
            'config': self.config
        }