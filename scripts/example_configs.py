"""
示例配置文件
展示如何创建和使用不同的配置
"""

from configs.config import (
    TrainingConfig, 
    DataConfig, 
    ModelConfig,
    RandomForestConfig, 
    XGBoostConfig, 
    GradientBoostingConfig
)

# ============================================================================
# 示例1: 甲班随机森林配置
# ============================================================================

def get_team_a_rf_config() -> TrainingConfig:
    """甲班随机森林配置"""
    return TrainingConfig(
        experiment_name="甲班随机森林实验",
        data=DataConfig(
            team="甲",
            max_samples=3000,  # 使用3000条数据
            test_size=0.25     # 25%测试集
        ),
        model=ModelConfig(
            model_type="random_forest",
            model_name="team_a_rf",
            model_save_path="saved_models/team_a"
        ),
        random_forest=RandomForestConfig(
            n_estimators=150,    # 增加树的数量
            max_depth=12,        # 增加深度
            min_samples_split=3, # 减少分裂所需样本
            min_samples_leaf=1   # 减少叶节点样本
        )
    )

# ============================================================================
# 示例2: 乙班XGBoost配置
# ============================================================================

def get_team_b_xgb_config() -> TrainingConfig:
    """乙班XGBoost配置"""
    return TrainingConfig(
        experiment_name="乙班XGBoost实验",
        data=DataConfig(
            team="乙",
            max_samples=None,  # 使用所有数据
            test_size=0.2
        ),
        model=ModelConfig(
            model_type="xgboost",
            model_name="team_b_xgb",
            model_save_path="saved_models/team_b"
        ),
        xgboost=XGBoostConfig(
            n_estimators=200,      # 更多树
            max_depth=8,           # 中等深度
            learning_rate=0.05,    # 较小学习率
            subsample=0.7,         # 子采样
            colsample_bytree=0.8   # 特征采样
        )
    )

# ============================================================================
# 示例3: 丙班梯度提升配置
# ============================================================================

def get_team_c_gb_config() -> TrainingConfig:
    """丙班梯度提升配置"""
    return TrainingConfig(
        experiment_name="丙班梯度提升实验",
        data=DataConfig(
            team="丙",
            max_samples=5000,
            test_size=0.15  # 15%测试集
        ),
        model=ModelConfig(
            model_type="gradient_boosting",
            model_name="team_c_gb",
            model_save_path="saved_models/team_c"
        ),
        gradient_boosting=GradientBoostingConfig(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8
        )
    )

# ============================================================================
# 示例4: 丁班对比实验配置
# ============================================================================

def get_team_d_comparison_config(model_type: str = "random_forest") -> TrainingConfig:
    """丁班对比实验配置"""
    config = TrainingConfig(
        experiment_name=f"丁班{model_type}对比实验",
        data=DataConfig(
            team="丁",
            max_samples=4000,
            test_size=0.2
        ),
        model=ModelConfig(
            model_type=model_type,
            model_name=f"team_d_{model_type}",
            model_save_path="saved_models/comparison"
        )
    )
    
    # 根据模型类型设置具体参数
    if model_type == "random_forest":
        config.random_forest = RandomForestConfig(
            n_estimators=100,
            max_depth=10
        )
    elif model_type == "xgboost":
        config.xgboost = XGBoostConfig(
            n_estimators=100,
            max_depth=6
        )
    elif model_type == "gradient_boosting":
        config.gradient_boosting = GradientBoostingConfig(
            n_estimators=100,
            max_depth=5
        )
    
    return config

# ============================================================================
# 示例5: 生产环境配置
# ============================================================================

def get_production_config(team: str) -> TrainingConfig:
    """生产环境配置"""
    return TrainingConfig(
        experiment_name=f"{team}班生产模型",
        data=DataConfig(
            team=team,
            max_samples=None,  # 使用所有数据
            test_size=0.1      # 10%测试集
        ),
        model=ModelConfig(
            model_type="random_forest",  # 生产环境使用稳定的随机森林
            model_name=f"production_{team}",
            model_save_path="production_models"
        ),
        random_forest=RandomForestConfig(
            n_estimators=200,    # 更多树提高稳定性
            max_depth=15,        # 适当深度
            min_samples_split=10, # 防止过拟合
            min_samples_leaf=5    # 防止过拟合
        ),
        save_model=True,
        evaluate_model=True,
        verbose=False  # 生产环境关闭详细输出
    )

# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    """展示如何使用这些配置"""
    
    # 1. 获取甲班随机森林配置
    config_a = get_team_a_rf_config()
    print(f"甲班配置: {config_a.experiment_name}")
    print(f"  样本数: {config_a.data.max_samples}")
    print(f"  模型: {config_a.model.model_type}")
    print()
    
    # 2. 获取乙班XGBoost配置
    config_b = get_team_b_xgb_config()
    print(f"乙班配置: {config_b.experiment_name}")
    print(f"  样本数: {'全部' if config_b.data.max_samples is None else config_b.data.max_samples}")
    print(f"  模型: {config_b.model.model_type}")
    print()
    
    # 3. 获取生产环境配置
    for team in ["甲", "乙", "丙", "丁"]:
        prod_config = get_production_config(team)
        print(f"{team}班生产配置: {prod_config.experiment_name}")
    
    print("\n配置示例完成!")
    print("可以在train.py中使用这些配置:")
    print("  from scripts.example_configs import get_team_a_rf_config")
    print("  config = get_team_a_rf_config()")
    print("  # 然后使用config进行训练")