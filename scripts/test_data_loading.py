"""测试数据加载"""

import pandas as pd
import numpy as np

# 测试数据加载
print("测试数据加载...")

try:
    # 尝试不同的编码方式
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"尝试编码: {encoding}")
            # 读取前1000行数据
            df = pd.read_csv('dataset/data.csv', nrows=1000, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取 {len(df)} 行数据")
            print(f"列名: {list(df.columns)}")
            break
        except UnicodeDecodeError:
            continue
    else:
        print("所有编码尝试都失败了")
        exit(1)
    
    # 查看班组分布
    print("\n班组分布:")
    print(df['班组'].value_counts())
    
    # 查看数据类型
    print("\n数据类型:")
    print(df.dtypes)
    
    # 查看数值列的基本统计
    numeric_cols = ['前行带钢厚度（5）', '后行带钢厚度（6）', '电流（7）', '速度（9）', '压力（8）']
    print("\n数值列统计:")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"{col}: min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}")
    
    # 查看牌号种类
    print("\n前行带钢内部牌号种类:")
    print(df['前行带钢内部牌号'].value_counts().head(10))
    
    print("\n后行带钢内部牌号种类:")
    print(df['后行带钢内部牌号'].value_counts().head(10))
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
