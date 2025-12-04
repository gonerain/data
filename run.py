"""
焊机参数机器学习系统 - 启动脚本
简化用户操作
"""

import sys
import os
import subprocess
from pathlib import Path

import models.model_factory as model_factory

# 添加当前目录到Python路径
sys.path.append('.')

print("=" * 70)
print("焊机参数机器学习系统")
print("=" * 70)
print()

# 检查必要目录
required_dirs = ['configs', 'models', 'data', 'scripts', 'saved_models', 'results', 'logs']
for dir_name in required_dirs:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        print(f"创建目录: {dir_name}")

print()

# 主菜单
while True:
    print("请选择操作:")
    print("  1. 训练模型")
    print("  2. 比较模型")
    print("  3. 使用模型预测")
    print("  4. 系统测试")
    print("  5. 查看帮助")
    print("  0. 退出")
    print()
    
    choice = input("请输入选择 (0-5): ").strip()
    
    if choice == '0':
        print("\n感谢使用，再见!")
        break
        
    elif choice == '1':
        print("\n训练模型选项:")
        print("  1. 快速训练 (甲班，随机森林，2000条数据)")
        print("  2. 自定义训练")
        print("  0. 返回主菜单")
        
        train_choice = input("\n请输入选择: ").strip()
        
        if train_choice == '1':
            print("\n开始快速训练...")
            cmd = [sys.executable, "scripts/train.py", "--team", "甲", "--model", "random_forest", "--max_samples", "2000"]
            subprocess.run(cmd)
            
        elif train_choice == '2':
            print("\n自定义训练:")
            team = input("班组 (甲/乙/丙/丁): ").strip() or "甲"
            # model = input("模型类型 (random_forest/xgboost/gradient_boosting): ").strip() or "random_forest"
            model_list = model_factory.ModelFactory.get_available_models()
            model = input(f"模型类型 {model_list}: ").strip() or "random_forest"
            samples = input("最大样本数 (0表示全部): ").strip() or "2000"
            
            cmd = [sys.executable, "scripts/train.py", "--team", team, "--model", model, "--max_samples", samples]
            subprocess.run(cmd)
            
    elif choice == '2':
        print("\n比较模型选项:")
        print("  1. 比较不同模型 (同一班组)")
        print("  2. 比较不同班组 (同一模型)")
        print("  0. 返回主菜单")
        
        compare_choice = input("\n请输入选择: ").strip()
        
        if compare_choice == '1':
            team = input("班组 (甲/乙/丙/丁): ").strip() or "甲"
            samples = input("最大样本数: ").strip() or "2000"
            
            cmd = [sys.executable, "scripts/evaluate.py", "compare_models", "--team", team, "--max_samples", samples]
            subprocess.run(cmd)
            
        elif compare_choice == '2':
            model = input("模型类型 (random_forest/xgboost/gradient_boosting): ").strip() or "random_forest"
            samples = input("最大样本数: ").strip() or "2000"
            
            cmd = [sys.executable, "scripts/evaluate.py", "compare_teams", "--model", model, "--max_samples", samples]
            subprocess.run(cmd)
            
    elif choice == '3':
        print("\n预测选项:")
        print("  1. 交互式预测")
        print("  2. 批量预测")
        print("  3. 列出所有模型")
        print("  0. 返回主菜单")
        
        predict_choice = input("\n请输入选择: ").strip()
        
        if predict_choice == '1':
            # 查找可用的模型
            model_files = list(Path("saved_models").glob("*.joblib"))
            
            if not model_files:
                print("\n未找到训练好的模型，请先训练模型。")
                continue
                
            print("\n可用的模型:")
            for i, model_file in enumerate(model_files, 1):
                print(f"  {i}. {model_file.name}")
            
            model_choice = input("\n选择模型 (输入编号): ").strip()
            
            try:
                idx = int(model_choice) - 1
                if 0 <= idx < len(model_files):
                    model_path = model_files[idx]
                    cmd = [sys.executable, "scripts/predict.py", "interactive", "--model", str(model_path)]
                    subprocess.run(cmd)
                else:
                    print("选择无效")
            except ValueError:
                print("输入无效")
                
        elif predict_choice == '2':
            model_path = input("模型文件路径: ").strip()
            input_file = input("输入CSV文件路径: ").strip()
            output_file = input("输出CSV文件路径: ").strip()
            
            if not all([model_path, input_file, output_file]):
                print("所有参数都必须填写")
                continue
                
            cmd = [sys.executable, "scripts/predict.py", "batch", "--model", model_path, "--input", input_file, "--output", output_file]
            subprocess.run(cmd)
            
        elif predict_choice == '3':
            cmd = [sys.executable, "scripts/predict.py", "list"]
            subprocess.run(cmd)
            
    elif choice == '4':
        print("\n开始系统测试...")
        cmd = [sys.executable, "scripts/test_system.py"]
        subprocess.run(cmd)
        
    elif choice == '5':
        print("\n" + "=" * 70)
        print("帮助信息")
        print("=" * 70)
        print("\n直接使用命令行:")
        print("  训练模型: python scripts/train.py --team 甲 --model random_forest")
        print("  比较模型: python scripts/evaluate.py compare_models --team 甲")
        print("  预测: python scripts/predict.py interactive --model saved_models/模型文件.joblib")
        print("\n详细文档请查看 USAGE.md 和 README_CN.md")
        print("=" * 70)
        
    else:
        print("\n选择无效，请重新输入")
    
    print("\n" + "-" * 50 + "\n")