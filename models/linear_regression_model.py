"""
线性回归模型
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor

from models.base_model import BaseModel
from models.model_factory import ModelFactory


@ModelFactory.register(
    model_type='linear_regression',
    config_section='linear_regression',
    description='线性回归 - 简单快速，适合线性关系',
    dependencies=['sklearn']
)
class LinearRegressionModel(BaseModel):
    """线性回归模型"""
    
    def __init__(self, config: dict):
        super().__init__(config)
        self.build_model()
        
    def build_model(self):
        """构建线性回归模型"""
        lr_config = self.config.get('linear_regression', {})
        
        base_model = LinearRegression(
            fit_intercept=lr_config.get('fit_intercept', True),
            copy_X=lr_config.get('copy_X', True),
            n_jobs=lr_config.get('n_jobs', -1)
        )
        
        self.model = MultiOutputRegressor(base_model)
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """训练模型"""
        self.model.fit(X, y)
        
        # 线性回归没有特征重要性，设为None
        self.feature_importance = None
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        return self.model.predict(X)
    
    def get_coefficients(self):
        """获取回归系数"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        coefs = []
        for estimator in self.model.estimators_:
            coefs.append(estimator.coef_)
        
        return np.array(coefs)
    
    def get_intercepts(self):
        """获取截距"""
        if self.model is None:
            raise ValueError("模型尚未训练")
        
        intercepts = []
        for estimator in self.model.estimators_:
            intercepts.append(estimator.intercept_)
        
        return np.array(intercepts)
