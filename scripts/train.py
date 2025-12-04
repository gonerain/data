"""
统一训练脚本
用法: python train.py --config config_name --team 甲 --model random_forest
"""

import argparse
import os
import sys
from datetime import datetime
import json

# 添加当前目录到Python路径
sys.path.append('.')

from configs.config import TrainingConfig, get_team_config
from data.data_loader import DataLoader
from models.model_factory import ModelFactory


def setup_directories(config: TrainingConfig):
    """设置目录"""
    os.makedirs(config.model.model_save_path, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)


def train_model(config: TrainingConfig):
    """训练模型"""
    print("=" * 70)
    print(f"焊机参数机器学习训练")
    print(f"实验名称: {config.experiment_name}")
    print(f"训练时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # 设置目录
    setup_directories(config)
    
    # 创建数据加载器
    data_loader = DataLoader(config.data)
    
    # 加载和预处理数据
    print("\n[1/4] 加载和预处理数据...")
    X_train, X_test, y_train, y_test, metadata = data_loader.get_data()
    
    # 创建模型
    print(f"\n[2/4] 创建模型 ({config.model.model_type})...")
    model = ModelFactory.create_model(config)
    
    model_info = model.get_model_info()
    print(f"  模型类型: {model_info['model_type']}")
    print(f"  模型参数: {json.dumps(model_info['config'], indent=2, default=str)}")
    
    # 训练模型
    print(f"\n[3/4] 训练模型...")
    model.fit(X_train, y_train)
    
    # 评估模型
    if config.evaluate_model:
        print(f"\n[4/4] 评估模型...")
        target_names = config.data.target_columns
        results = model.evaluate(X_test, y_test, target_names)
        
        print("\n评估结果:")
        print("-" * 50)
        
        # 打印每个目标的评估结果
        for target_name in target_names:
            if target_name in results:
                metrics = results[target_name]
                print(f"{target_name}:")
                print(f"  MSE: {metrics['mse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  R²:  {metrics['r2']:.4f}")
                print()
        
        # 打印总体评估结果
        if 'overall' in results:
            overall = results['overall']
            print(f"总体评估:")
            print(f"  MSE: {overall['mse']:.4f}")
            print(f"  MAE: {overall['mae']:.4f}")
            print(f"  R²:  {overall['r2']:.4f}")
        
        print("-" * 50)
    
    # 保存模型
    if config.save_model:
        model_filename = f"{config.model.model_name}_{config.model.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        model_path = os.path.join(config.model.model_save_path, model_filename)
        
        # 保存完整的模型数据
        import joblib
        full_model_data = {
            'model': model.model,
            'config': config,
            'metadata': metadata,
            'results': results if config.evaluate_model else None,
            'feature_importance': model.get_feature_importance(),
            'training_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        joblib.dump(full_model_data, model_path)
        print(f"\n模型已保存到: {model_path}")
        
        # 保存特征重要性
        if config.model.save_feature_importance and model.feature_importance is not None:
            feature_names = metadata['feature_names']
            importance_data = []
            
            for name, importance in zip(feature_names, model.feature_importance):
                importance_data.append({
                    'feature': name,
                    'importance': float(importance)
                })
            
            # 按重要性排序
            importance_data.sort(key=lambda x: x['importance'], reverse=True)
            
            importance_file = model_path.replace('.joblib', '_importance.json')
            with open(importance_file, 'w', encoding='utf-8') as f:
                json.dump(importance_data, f, ensure_ascii=False, indent=2)
            
            print(f"特征重要性已保存到: {importance_file}")
            
            # 打印特征重要性
            print("\n特征重要性排名:")
            print("-" * 50)
            for i, item in enumerate(importance_data[:5], 1):
                print(f"{i}. {item['feature']}: {item['importance']:.4f}")
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    
    return model, results, metadata


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练焊机参数机器学习模型')
    
    parser.add_argument('--team', type=str, default='甲', 
                       choices=['甲', '乙', '丙', '丁'],
                       help='选择班组 (默认: 甲)')
    
    parser.add_argument('--model', type=str, default='random_forest',
                       choices=ModelFactory.get_available_models(),
                       help='选择模型类型 (默认: random_forest)')
    
    parser.add_argument('--max_samples', type=int, default=2000,
                       help='最大样本数 (默认: 2000, 0表示使用所有数据)')
    
    parser.add_argument('--experiment', type=str, default=None,
                       help='实验名称')
    
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='显示详细输出')
    
    args = parser.parse_args()
    
    # 创建配置
    config = get_team_config(args.team)
    config.model.model_type = args.model
    
    if args.max_samples == 0:
        config.data.max_samples = None
    else:
        config.data.max_samples = args.max_samples
    
    if args.experiment:
        config.experiment_name = args.experiment
    
    config.verbose = args.verbose
    
    # 显示配置信息
    print("训练配置:")
    print(f"  班组: {config.data.team}")
    print(f"  模型: {config.model.model_type}")
    print(f"  最大样本数: {'全部' if config.data.max_samples is None else config.data.max_samples}")
    print(f"  实验名称: {config.experiment_name}")
    print()
    
    # 开始训练
    try:
        train_model(config)
    except Exception as e:
        print(f"训练过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()