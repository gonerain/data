"""使用训练好的模型进行预测"""

import joblib
import numpy as np

print("=" * 60)
print("焊机参数预测系统")
print("=" * 60)

# 加载模型
try:
    model_file = 'welding_model_甲_simple.joblib'
    print(f"正在加载模型: {model_file}")
    
    model_data = joblib.load(model_file)
    model = model_data['model']
    scaler = model_data['scaler']
    label_encoders = model_data['label_encoders']
    team = model_data['team']
    feature_columns = model_data['feature_columns']
    target_columns = model_data['target_columns']
    
    print(f"模型训练班组: {team}")
    print(f"特征列: {feature_columns}")
    print(f"目标列: {target_columns}")
    
    # 显示可用的牌号
    print("\n可用牌号示例:")
    front_brands = label_encoders['前行带钢内部牌号'].classes_
    back_brands = label_encoders['后行带钢内部牌号'].classes_
    
    print(f"前行带钢牌号 (前10个): {front_brands[:10].tolist()}")
    print(f"后行带钢牌号 (前10个): {back_brands[:10].tolist()}")
    
    # 交互式预测
    print("\n" + "=" * 60)
    print("开始预测 (输入q退出)")
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
            for col, value in zip(['前行带钢内部牌号', '后行带钢内部牌号'], 
                                [front_brand, back_brand]):
                le = label_encoders.get(col)
                if le is None:
                    print(f"错误: 未找到{col}的编码器")
                    continue
                
                try:
                    encoded = le.transform([str(value)])
                    encoded_features.append(encoded)
                except ValueError:
                    # 如果遇到新的类别
                    print(f"警告: 牌号 '{value}' 不在训练数据中")
                    print(f"可用牌号: {le.classes_[:5].tolist()}...")
                    encoded_features.append([0])  # 使用默认值
            
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
            print("预测结果:")
            print("-" * 40)
            for i, param in enumerate(target_columns):
                print(f"  {param}: {predictions[i]:.2f}")
            print("-" * 40)
            
        except ValueError as e:
            print(f"输入错误: {e}")
        except Exception as e:
            print(f"预测错误: {e}")
    
    print("\n感谢使用焊机参数预测系统!")
    
except FileNotFoundError:
    print(f"错误: 模型文件 {model_file} 未找到")
    print("请先运行 train_model_simple.py 训练模型")
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()