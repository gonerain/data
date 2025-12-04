"""
随机森林模型
"""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from models.base_model import BaseModel
from models.model_factory import ModelFactory


@ModelFactory.register(
    model_type='random_forest',
    config_section='random_forest',
    description='随机森林回归 - 集成学习，抗过拟合能力强',
    dependencies=['sklearn']
)
class RandomForestModel(BaseModel):
    """随机森林模型"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        """构建随机森林模型"""
        rf_config = self.config.get('random_forest', {})
        
        base_model = RandomForestRegressor(
            n_estimators=rf_config.get('n_estimators', 100),
            max_depth=rf_config.get('max_depth', 10),
            min_samples_split=rf_config.get('min_samples_split', 5),
            min_samples_leaf=rf_config.get('min_samples_leaf', 2),
            random_state=rf_config.get('random_state', 42),
            n_jobs=rf_config.get('n_jobs', -1)
        )
        
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        self.model.fit(X, y)
        
        # 计算特征重要性（取所有目标模型的平均值）
        importances = []
        for estimator in self.model.estimators_:
            importances.append(estimator.feature_importances_)
        
        self.feature_importance = np.mean(importances, axis=0)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def get_model_params(self) -> dict:
        """获取模型参数"""
        return self.model.estimators_[0].get_params()