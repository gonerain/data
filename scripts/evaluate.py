"""
模型评估和比较脚本
"""

import argparse
import os
import sys
import json
import pandas as pd
from datetime import datetime

# 添加当前目录到Python路径
sys.path.append('.')

from configs.config import TrainingConfig, get_team_config
from data.data_loader import DataLoader
from models.model_factory import ModelFactory


def evaluate_single_model(config: TrainingConfig, model_type: str = None):
    """评估单个模型"""
    if model_type:
        config.model.model_type = model_type
    
    print(f"\n评估 {config.data.team}班 - {config.model.model_type} 模型")
    print("-" * 50)
    
    # 创建数据加载器
    data_loader = DataLoader(config.data)
    
    # 加载数据
    X_train, X_test, y_train, y_test, metadata = data_loader.get_data()
    
    # 创建模型
    model = ModelFactory.create_model(config)
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 评估模型
    results = model.evaluate(X_test, y_test, config.data.target_columns)
    
    # 显示结果
    for target_name in config.data.target_columns:
        if target_name in results:
            metrics = results[target_name]
            print(f"{target_name}: R²={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}")
    
    if 'overall' in results:
        overall = results['overall']
        print(f"总体: R²={overall['r2']:.4f}, MAE={overall['mae']:.4f}")
    
    return {
        'team': config.data.team,
        'model_type': config.model.model_type,
        'results': results,
        'feature_importance': model.get_feature_importance().tolist() if model.get_feature_importance() is not None else None
    }


def compare_models(config: TrainingConfig, model_types: list = None):
    """比较多个模型"""
    if model_types is None:
        model_types = ModelFactory.get_available_models()
    
    print("=" * 70)
    print(f"模型比较 - {config.data.team}班")
    print(f"比较时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = []
    
    for model_type in model_types:
        try:
            # 临时修改模型类型
            current_config = TrainingConfig(**config.__dict__)
            current_config.model.model_type = model_type
            
            # 评估模型
            result = evaluate_single_model(current_config)
            all_results.append(result)
            
        except Exception as e:
            print(f"评估 {model_type} 模型时出错: {e}")
    
    # 显示比较结果
    print("\n" + "=" * 70)
    print("模型比较结果")
    print("=" * 70)
    
    comparison_data = []
    for result in all_results:
        model_info = {
            '模型类型': result['model_type'],
            '班组': result['team']
        }
        
        # 添加每个目标的R²分数
        for target_name in config.data.target_columns:
            if target_name in result['results']:
                model_info[target_name + '_R2'] = result['results'][target_name]['r2']
                model_info[target_name + '_MAE'] = result['results'][target_name]['mae']
        
        # 添加总体R²分数
        if 'overall' in result['results']:
            model_info['总体_R2'] = result['results']['overall']['r2']
            model_info['总体_MAE'] = result['results']['overall']['mae']
        
        comparison_data.append(model_info)
    
    # 创建DataFrame并显示
    df = pd.DataFrame(comparison_data)
    
    # 设置显示格式
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n模型性能比较:")
    print(df.to_string(index=False))
    
    # 找出最佳模型
    if '总体_R2' in df.columns:
        best_model_idx = df['总体_R2'].idxmax()
        best_model = df.iloc[best_model_idx]
        
        print("\n" + "-" * 50)
        print(f"最佳模型: {best_model['模型类型']}")
        print(f"总体R²: {best_model['总体_R2']:.4f}")
        print(f"总体MAE: {best_model['总体_MAE']:.4f}")
        print("-" * 50)
    
    # 保存比较结果
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/model_comparison_{config.data.team}_{timestamp}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'comparison_time': timestamp,
            'team': config.data.team,
            'results': all_results,
            'comparison_table': df.to_dict('records')
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n比较结果已保存到: {result_file}")
    
    return all_results, df


def compare_teams(model_type: str = 'random_forest', max_samples: int = 2000):
    """比较不同班组的模型性能"""
    teams = ['甲', '乙', '丙', '丁']
    
    print("=" * 70)
    print(f"班组比较 - {model_type} 模型")
    print(f"比较时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    all_results = []
    
    for team in teams:
        try:
            config = get_team_config(team)
            config.model.model_type = model_type
            config.data.max_samples = max_samples
            
            print(f"\n评估 {team}班...")
            
            # 创建数据加载器
            data_loader = DataLoader(config.data)
            
            # 加载数据
            X_train, X_test, y_train, y_test, metadata = data_loader.get_data()
            
            # 创建模型
            model = ModelFactory.create_model(config)
            
            # 训练模型
            model.fit(X_train, y_train)
            
            # 评估模型
            results = model.evaluate(X_test, y_test, config.data.target_columns)
            
            all_results.append({
                'team': team,
                'model_type': model_type,
                'results': results,
                'data_info': metadata['data_info']
            })
            
            # 显示结果
            if 'overall' in results:
                overall = results['overall']
                print(f"  {team}班: R²={overall['r2']:.4f}, MAE={overall['mae']:.4f}, 样本数={metadata['data_info']['total_samples']}")
            
        except Exception as e:
            print(f"评估 {team}班时出错: {e}")
    
    # 显示比较结果
    print("\n" + "=" * 70)
    print("班组比较结果")
    print("=" * 70)
    
    comparison_data = []
    for result in all_results:
        team_info = {
            '班组': result['team'],
            '模型类型': result['model_type'],
            '样本数': result['data_info']['total_samples']
        }
        
        if 'overall' in result['results']:
            overall = result['results']['overall']
            team_info['总体_R2'] = overall['r2']
            team_info['总体_MAE'] = overall['mae']
        
        comparison_data.append(team_info)
    
    # 创建DataFrame并显示
    df = pd.DataFrame(comparison_data)
    
    # 设置显示格式
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    print("\n班组性能比较:")
    print(df.to_string(index=False))
    
    # 找出最佳班组
    if '总体_R2' in df.columns:
        best_team_idx = df['总体_R2'].idxmax()
        best_team = df.iloc[best_team_idx]
        
        print("\n" + "-" * 50)
        print(f"最佳班组: {best_team['班组']}")
        print(f"总体R²: {best_team['总体_R2']:.4f}")
        print(f"样本数: {best_team['样本数']}")
        print("-" * 50)
    
    # 保存比较结果
    os.makedirs('results', exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f'results/team_comparison_{model_type}_{timestamp}.json'
    
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'comparison_time': timestamp,
            'model_type': model_type,
            'results': all_results,
            'comparison_table': df.to_dict('records')
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n比较结果已保存到: {result_file}")
    
    return all_results, df


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='评估和比较模型')
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 模型比较命令
    model_parser = subparsers.add_parser('compare_models', help='比较不同模型')
    model_parser.add_argument('--team', type=str, default='甲', 
                            choices=['甲', '乙', '丙', '丁'],
                            help='选择班组 (默认: 甲)')
    model_parser.add_argument('--max_samples', type=int, default=2000,
                            help='最大样本数 (默认: 2000)')
    
    # 班组比较命令
    team_parser = subparsers.add_parser('compare_teams', help='比较不同班组')
    team_parser.add_argument('--model', type=str, default='random_forest',
                           choices=ModelFactory.get_available_models(),
                           help='选择模型类型 (默认: random_forest)')
    team_parser.add_argument('--max_samples', type=int, default=2000,
                           help='最大样本数 (默认: 2000)')
    
    args = parser.parse_args()
    
    if args.command == 'compare_models':
        config = get_team_config(args.team)
        config.data.max_samples = args.max_samples
        compare_models(config)
        
    elif args.command == 'compare_teams':
        compare_teams(args.model, args.max_samples)
        
    else:
        # 默认执行模型比较
        config = get_team_config('甲')
        compare_models(config)


if __name__ == "__main__":
    main()