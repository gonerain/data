"""
模型工厂
"""

from typing import Dict, Any

from models.random_forest_model import RandomForestModel
from configs.config import (
    TrainingConfig, 
    RandomForestConfig, 
    XGBoostConfig, 
    GradientBoostingConfig
)


class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(config: TrainingConfig):
        """创建模型"""
        model_type = config.model.model_type
        
        # 准备模型配置
        model_config = {
            'model_type': model_type,
            'model_name': config.model.model_name
        }
        
        if model_type == 'random_forest':
            model_config['random_forest'] = {
                'n_estimators': config.random_forest.n_estimators,
                'max_depth': config.random_forest.max_depth,
                'min_samples_split': config.random_forest.min_samples_split,
                'min_samples_leaf': config.random_forest.min_samples_leaf,
                'random_state': config.random_forest.random_state,
                'n_jobs': config.random_forest.n_jobs
            }
            return RandomForestModel(model_config)
            
        elif model_type == 'xgboost':
            # 检查是否安装了xgboost
            try:
                from models.xgboost_model import XGBoostModel
                model_config['xgboost'] = {
                    'n_estimators': config.xgboost.n_estimators,
                    'max_depth': config.xgboost.max_depth,
                    'learning_rate': config.xgboost.learning_rate,
                    'subsample': config.xgboost.subsample,
                    'colsample_bytree': config.xgboost.colsample_bytree,
                    'random_state': config.xgboost.random_state,
                    'n_jobs': config.xgboost.n_jobs
                }
                return XGBoostModel(model_config)
            except ImportError:
                print("警告: XGBoost未安装，使用随机森林替代")
                print("安装命令: pip install xgboost")
                model_config['random_forest'] = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
                return RandomForestModel(model_config)
                
        elif model_type == 'gradient_boosting':
            # 检查是否安装了梯度提升
            try:
                from models.gradient_boosting_model import GradientBoostingModel
                model_config['gradient_boosting'] = {
                    'n_estimators': config.gradient_boosting.n_estimators,
                    'max_depth': config.gradient_boosting.max_depth,
                    'learning_rate': config.gradient_boosting.learning_rate,
                    'subsample': config.gradient_boosting.subsample,
                    'random_state': config.gradient_boosting.random_state
                }
                return GradientBoostingModel(model_config)
            except ImportError:
                print("警告: 使用随机森林替代梯度提升")
                model_config['random_forest'] = {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
                return RandomForestModel(model_config)
                
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
    
    @staticmethod
    def get_available_models() -> list:
        """获取可用的模型类型"""
        return ['random_forest', 'xgboost', 'gradient_boosting']
    
    @staticmethod
    def get_model_description(model_type: str) -> str:
        """获取模型描述"""
        descriptions = {
            'random_forest': '随机森林回归 - 集成学习，抗过拟合能力强',
            'xgboost': 'XGBoost回归 - 梯度提升树，竞赛常用算法',
            'gradient_boosting': '梯度提升回归 - 逐步优化，预测精度高'
        }
        return descriptions.get(model_type, '未知模型类型')