"""
示例模型 - 演示如何添加新模型
"""

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.multioutput import MultiOutputRegressor

from models.base_model import BaseModel
from models.model_factory import ModelFactory


@ModelFactory.register(
    model_type='decision_tree',
    config_section='decision_tree',
    description='决策树回归 - 简单易懂，可解释性强',
    dependencies=['sklearn']
)
class DecisionTreeModel(BaseModel):
    """决策树回归模型"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        """构建决策树模型"""
        dt_config = self.config.get('decision_tree', {})
        
        base_model = DecisionTreeRegressor(
            max_depth=dt_config.get('max_depth', 10),
            min_samples_split=dt_config.get('min_samples_split', 5),
            min_samples_leaf=dt_config.get('min_samples_leaf', 2),
            random_state=dt_config.get('random_state', 42)
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
    
    def get_tree_depth(self):
        """获取决策树深度"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        depths = []
        for estimator in self.model.estimators_:
            depths.append(estimator.tree_.max_depth)
        
        return depths