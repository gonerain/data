"""
训练配置文件
可以创建多个配置文件，通过train.py指定使用哪个配置
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class DataConfig:
    """数据配置"""
    # 数据文件路径
    data_path: str = 'dataset/data.csv'
    encoding: str = 'gbk'
    
    # 班组选择
    team: str = '甲'  # 可选: '甲', '乙', '丙', '丁'
    
    # 数据限制
    max_samples: Optional[int] = 2000  # None表示使用所有数据
    test_size: float = 0.2
    random_state: int = 42
    verbose: bool = True
    
    # 特征列
    feature_columns: List[str] = None
    target_columns: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                '前行带钢厚度（5）',
                '后行带钢厚度（6）',
                '前行带钢内部牌号',
                '后行带钢内部牌号'
            ]
        if self.target_columns is None:
            self.target_columns = ['电流（7）', '速度（9）', '压力（8）']

@dataclass
class ModelConfig:
    """模型配置"""
    # 模型类型
    model_type: str = 'random_forest'  # 可选: 'random_forest', 'xgboost', 'gradient_boosting'
    
    # 模型保存路径
    model_save_path: str = 'saved_models'
    model_name: str = 'welding_model'
    
    # 是否保存特征重要性
    save_feature_importance: bool = True

@dataclass
class RandomForestConfig:
    """随机森林配置"""
    n_estimators: int = 100
    max_depth: int = 10
    min_samples_split: int = 5
    min_samples_leaf: int = 2
    random_state: int = 42
    n_jobs: int = -1

@dataclass
class XGBoostConfig:
    """XGBoost配置"""
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42
    n_jobs: int = -1

@dataclass
class GradientBoostingConfig:
    """梯度提升配置"""
    n_estimators: int = 100
    max_depth: int = 5
    learning_rate: float = 0.1
    subsample: float = 0.8
    random_state: int = 42

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    experiment_name: str = 'welding_experiment'
    
    # 数据配置
    data: DataConfig = field(default_factory=DataConfig)
    
    # 模型配置
    model: ModelConfig = field(default_factory=ModelConfig)
    
    # 具体模型配置
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    xgboost: XGBoostConfig = field(default_factory=XGBoostConfig)
    gradient_boosting: GradientBoostingConfig = field(default_factory=GradientBoostingConfig)
    
    # 训练配置
    save_model: bool = True
    evaluate_model: bool = True
    verbose: bool = True
    
    # 日志配置
    log_dir: str = 'logs'
    save_logs: bool = True

# 预定义配置
def get_team_config(team: str = '甲') -> TrainingConfig:
    """获取班组配置"""
    config = TrainingConfig()
    config.data.team = team
    config.model.model_name = f'welding_model_{team}'
    config.experiment_name = f'welding_{team}_team'
    return config

def get_full_data_config() -> TrainingConfig:
    """获取完整数据配置"""
    config = TrainingConfig()
    config.data.max_samples = None  # 使用所有数据
    return config

def get_random_forest_config() -> TrainingConfig:
    """获取随机森林配置"""
    config = TrainingConfig()
    config.model.model_type = 'random_forest'
    return config

def get_xgboost_config() -> TrainingConfig:
    """获取XGBoost配置"""
    config = TrainingConfig()
    config.model.model_type = 'xgboost'
    return config

def get_gradient_boosting_config() -> TrainingConfig:
    """获取梯度提升配置"""
    config = TrainingConfig()
    config.model.model_type = 'gradient_boosting'
    return config