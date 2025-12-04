"""
预测脚本
"""

import argparse
import os
import sys
import json
import numpy as np
import joblib

# 添加当前目录到Python路径
sys.path.append('.')

from configs.config import TrainingConfig


def load_model(model_path: str):
    """加载模型"""
    if not os.path.exists(model_path):
        print(f"错误: 模型文件 {model_path} 不存在")
        return None
    
    try:
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None


def predict_single(model_data: dict, features: dict) -> dict:
    """单个预测"""
    model = model_data['model']
    metadata = model_data['metadata']
    scaler = metadata['scaler']
    label_encoders = metadata['label_encoders']
    
    # 准备数值特征
    numeric_features = []
    numeric_cols = ['前行带钢厚度（5）', '后行带钢厚度（6）']
    
    for col in numeric_cols:
        if col in features:
            try:
                numeric_features.append(float(features[col]))
            except ValueError:
                print(f"错误: {col} 必须是数值")
                return None
        else:
            print(f"错误: 缺少特征 {col}")
            return None
    
    numeric_features = np.array([numeric_features])
    
    # 准备分类特征
    encoded_features = []
    categorical_cols = ['前行带钢内部牌号', '后行带钢内部牌号']
    
    for col in categorical_cols:
        if col in features:
            value = str(features[col])
            le = label_encoders.get(col)
            
            if le is None:
                print(f"错误: 未找到 {col} 的编码器")
                return None
            
            try:
                encoded = le.transform([value])
                encoded_features.append(encoded)
            except ValueError:
                # 处理未知类别
                print(f"警告: 牌号 '{value}' 不在训练数据中")
                print(f"可用牌号: {le.classes_[:5].tolist()}...")
                # 使用最常见的类别
                encoded = le.transform([le.classes_[0]])
                encoded_features.append(encoded)
        else:
            print(f"错误: 缺少特征 {col}")
            return None
    
    encoded_features = np.array(encoded_features).reshape(1, -1)
    
    # 合并特征
    X = np.hstack([numeric_features, encoded_features])
    
    # 标准化
    X_scaled = scaler.transform(X)
    
    # 预测
    predictions = model.predict(X_scaled)[0]
    
    # 获取目标列名
    config = model_data.get('config')
    if config and hasattr(config, 'data'):
        target_columns = config.data.target_columns
    else:
        target_columns = ['电流（7）', '速度（9）', '压力（8）']
    
    # 构建结果
    result = {}
    for i, col in enumerate(target_columns):
        result[col] = float(predictions[i])
    
    return result


def interactive_predict(model_path: str):
    """交互式预测"""
    model_data = load_model(model_path)
    if model_data is None:
        return
    
    config = model_data.get('config', TrainingConfig())
    metadata = model_data.get('metadata', {})
    
    print("=" * 70)
    print("焊机参数预测系统")
    print("=" * 70)
    
    if hasattr(config, 'data') and hasattr(config.data, 'team'):
        print(f"模型班组: {config.data.team}")
    
    if 'data_info' in metadata:
        print(f"训练样本数: {metadata['data_info']['total_samples']}")
    
    print()
    
    # 显示可用牌号
    if 'label_encoders' in metadata:
        label_encoders = metadata['label_encoders']
        print("可用牌号示例:")
        for col, le in label_encoders.items():
            print(f"  {col}: {le.classes_[:5].tolist()}...")
    
    print("\n" + "=" * 70)
    print("开始预测 (输入q退出)")
    print("=" * 70)
    
    while True:
        print("\n请输入参数:")
        
        try:
            features = {}
            
            # 获取输入
            front_thickness = input("前行带钢厚度: ").strip()
            if front_thickness.lower() == 'q':
                break
            
            back_thickness = input("后行带钢厚度: ").strip()
            if back_thickness.lower() == 'q':
                break
            
            front_brand = input("前行带钢牌号: ").strip()
            if front_brand.lower() == 'q':
                break
            
            back_brand = input("后行带钢牌号: ").strip()
            if back_brand.lower() == 'q':
                break
            
            features['前行带钢厚度（5）'] = front_thickness
            features['后行带钢厚度（6）'] = back_thickness
            features['前行带钢内部牌号'] = front_brand
            features['后行带钢内部牌号'] = back_brand
            
            # 预测
            result = predict_single(model_data, features)
            
            if result:
                print("\n" + "-" * 50)
                print("预测结果:")
                print("-" * 50)
                
                for param, value in result.items():
                    print(f"  {param}: {value:.2f}")
                
                # 显示建议
                print("\n建议参数:")
                if '电流（7）' in result:
                    print(f"  电流: {result['电流（7）']:.1f} A")
                if '速度（9）' in result:
                    print(f"  速度: {result['速度（9）']:.1f} m/min")
                if '压力（8）' in result:
                    print(f"  压力: {result['压力（8）']:.1f} MPa")
                
                print("-" * 50)
            
        except KeyboardInterrupt:
            print("\n\n中断预测")
            break
        except Exception as e:
            print(f"预测时出错: {e}")
    
    print("\n感谢使用焊机参数预测系统!")


def batch_predict(model_path: str, input_file: str, output_file: str):
    """批量预测"""
    model_data = load_model(model_path)
    if model_data is None:
        return
    
    # 读取输入文件
    try:
        import pandas as pd
        df = pd.read_csv(input_file)
        print(f"读取 {len(df)} 条数据")
    except Exception as e:
        print(f"读取输入文件时出错: {e}")
        return
    
    # 检查必要列
    required_cols = ['前行带钢厚度（5）', '后行带钢厚度（6）', 
                    '前行带钢内部牌号', '后行带钢内部牌号']
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"错误: 输入文件缺少列: {missing_cols}")
        return
    
    # 批量预测
    predictions = []
    for idx, row in df.iterrows():
        features = row[required_cols].to_dict()
        
        try:
            result = predict_single(model_data, features)
            if result:
                predictions.append(result)
            else:
                predictions.append({})
        except Exception as e:
            print(f"第 {idx+1} 行预测时出错: {e}")
            predictions.append({})
        
        if (idx + 1) % 100 == 0:
            print(f"已处理 {idx+1} 条数据")
    
    # 合并结果
    results_df = pd.DataFrame(predictions)
    output_df = pd.concat([df, results_df], axis=1)
    
    # 保存结果
    output_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n预测结果已保存到: {output_file}")
    print(f"总处理数据: {len(output_df)} 条")


def list_models(models_dir: str = 'saved_models'):
    """列出所有模型"""
    if not os.path.exists(models_dir):
        print(f"模型目录 {models_dir} 不存在")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not model_files:
        print("未找到模型文件")
        return
    
    print("=" * 70)
    print("可用模型列表")
    print("=" * 70)
    
    for i, model_file in enumerate(sorted(model_files), 1):
        model_path = os.path.join(models_dir, model_file)
        
        try:
            model_data = joblib.load(model_path)
            config = model_data.get('config', {})
            
            team = '未知'
            model_type = '未知'
            
            if hasattr(config, 'data') and hasattr(config.data, 'team'):
                team = config.data.team
            
            if hasattr(config, 'model') and hasattr(config.model, 'model_type'):
                model_type = config.model.model_type
            
            file_size = os.path.getsize(model_path) / 1024 / 1024  # MB
            
            print(f"{i}. {model_file}")
            print(f"   班组: {team}, 模型: {model_type}, 大小: {file_size:.2f} MB")
            
            if 'training_time' in model_data:
                print(f"   训练时间: {model_data['training_time']}")
            
            print()
            
        except Exception as e:
            print(f"{i}. {model_file} (加载失败: {e})")
            print()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='焊机参数预测')
    
    subparsers = parser.add_subparsers(dest='command', help='子命令')
    
    # 交互式预测命令
    interactive_parser = subparsers.add_parser('interactive', help='交互式预测')
    interactive_parser.add_argument('--model', type=str, required=True,
                                  help='模型文件路径')
    
    # 批量预测命令
    batch_parser = subparsers.add_parser('batch', help='批量预测')
    batch_parser.add_argument('--model', type=str, required=True,
                            help='模型文件路径')
    batch_parser.add_argument('--input', type=str, required=True,
                            help='输入CSV文件路径')
    batch_parser.add_argument('--output', type=str, required=True,
                            help='输出CSV文件路径')
    
    # 列出模型命令
    list_parser = subparsers.add_parser('list', help='列出所有模型')
    list_parser.add_argument('--dir', type=str, default='saved_models',
                           help='模型目录 (默认: saved_models)')
    
    args = parser.parse_args()
    
    if args.command == 'interactive':
        interactive_predict(args.model)
        
    elif args.command == 'batch':
        batch_predict(args.model, args.input, args.output)
        
    elif args.command == 'list':
        list_models(args.dir)
        
    else:
        # 默认显示帮助
        parser.print_help()


if __name__ == "__main__":
    main()