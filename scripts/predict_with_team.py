"""选择班组进行预测"""

import joblib
import numpy as np
import os

print("=" * 60)
print("焊机参数预测系统 - 多班组支持")
print("=" * 60)

# 检查可用的模型
available_models = []
for team in ['甲', '乙', '丙', '丁']:
    model_file = f'welding_model_{team}.joblib'
    if os.path.exists(model_file):
        available_models.append(team)

if not available_models:
    print("未找到任何训练好的模型!")
    print("请先运行以下命令训练模型:")
    print("  python train_model_simple.py    # 训练甲班模型")
    print("  python train_all_teams.py       # 训练所有班组模型")
    exit(1)

print(f"可用的班组模型: {available_models}")
print("\n选择要使用的班组模型:")
for i, team in enumerate(available_models, 1):
    print(f"  {i}. {team}班")

# 选择模型
try:
    choice = input("\n请输入选择 (1-{}): ".format(len(available_models))).strip()
    choice_idx = int(choice) - 1
    
    if choice_idx < 0 or choice_idx >= len(available_models):
        print("选择无效!")
        exit(1)
        
    selected_team = available_models[choice_idx]
    model_file = f'welding_model_{selected_team}.joblib'
    
    print(f"\n正在加载 {selected_team}班 模型...")
    
    # 加载模型
    model_data = joblib.load(model_file)
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    team = model_data['team']
    feature_columns = model_data['feature_columns']
    target_columns = model_data['target_columns']
    
    print(f"模型加载成功!")
    print(f"训练班组: {team}")
    
    # 显示模型信息
    if 'results' in model_data:
        print("\n模型性能:")
        results = model_data['results']
        for target, metrics in results.items():
            print(f"  {target}: R2={metrics['R2']:.4f}")
    
    # 显示可用牌号
    print("\n可用牌号示例:")
    front_brands = label_encoders['前行带钢内部牌号'].classes_
    back_brands = label_encoders['后行带钢内部牌号'].classes_
    
    print(f"前行带钢牌号 (前5个): {front_brands[:5].tolist()}")
    print(f"后行带钢牌号 (前5个): {back_brands[:5].tolist()}")
    
    # 交互式预测
    print("\n" + "=" * 60)
    print(f"开始预测 - {team}班 (输入q退出)")
    print("=" * 60)
    
    while True:
        print("\n请输入参数:")
        
        try:
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
            
            # 转换为数值
            front_thickness = float(front_thickness)
            back_thickness = float(back_thickness)
            
            # 准备特征
            numeric_features = np.array([[front_thickness, back_thickness]])
            
            # 编码分类特征
            encoded_features = []
            brand_inputs = [front_brand, back_brand]
            brand_types = ['前行带钢内部牌号', '后行带钢内部牌号']
            
            for i, (col, value) in enumerate(zip(brand_types, brand_inputs)):
                le = label_encoders.get(col)
                if le is None:
                    print(f"错误: 未找到{col}的编码器")
                    continue
                
                try:
                    encoded = le.transform([str(value)])
                    encoded_features.append(encoded)
                except ValueError:
                    # 如果遇到新的类别
                    print(f"\n警告: 牌号 '{value}' 不在训练数据中")
                    print(f"可用牌号示例: {le.classes_[:5].tolist()}")
                    print("请重新输入或输入q退出")
                    encoded_features = []
                    break
            
            if len(encoded_features) != 2:
                continue
                
            encoded_features = np.array(encoded_features).reshape(1, -1)
            
            # 合并特征
            features = np.hstack([numeric_features, encoded_features])
            
            # 标准化
            features_scaled = scaler.transform(features)
            
            # 预测
            predictions = model.predict(features_scaled)[0]
            
            # 显示结果
            print("\n" + "-" * 40)
            print(f"预测结果 ({team}班):")
            print("-" * 40)
            for i, param in enumerate(target_columns):
                print(f"  {param}: {predictions[i]:.2f}")
            print("-" * 40)
            
            # 显示建议
            print("\n建议参数:")
            print(f"  电流: {predictions[0]:.1f} A")
            print(f"  速度: {predictions[1]:.1f} m/min")
            print(f"  压力: {predictions[2]:.1f} MPa")
            
        except ValueError as e:
            print(f"输入错误: {e}")
        except Exception as e:
            print(f"预测错误: {e}")
    
    print("\n感谢使用焊机参数预测系统!")
    
except ValueError:
    print("输入无效!")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()