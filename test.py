import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 创建数据
error_thresh = [0.6, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
Max_inter = [200,500, 1000, 1500, 2000,2500]

# 性能数据矩阵（6行×5列）
performance_data = [
    [0.491, 0.566, 0.687, 0.710, 0.691, 0.602],
    [0.534, 0.667, 0.733, 0.728, 0.668, 0.615],
    [0.726, 0.861, 0.893, 0.914, 0.779, 0.630],
    [0.769, 0.891, 0.924, 0.902, 0.684, 0.601],
    [0.621, 0.723, 0.833, 0.772, 0.667, 0.642],
    [0.510, 0.648, 0.714, 0.554, 0.524, 0.603],
    [0.478, 0.553, 0.652, 0.715, 0.513, 0.547]
]

# 创建DataFrame
df = pd.DataFrame(performance_data,
                  index=[f'k={k}' for k in error_thresh],
                  columns=[f'ρ={rou}' for rou in Max_inter])

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制热力图
heatmap = sns.heatmap(df,
                      annot=True,
                      fmt=".3f",
                      cmap="OrRd",  # 反向的红黄蓝配色，红色表示高值RdYlBu_r
                      linewidths=0.5,
                      linecolor='white',
                      ax=ax,
                      cbar_kws={'label': 'Performance Score (OP.)',
                               'shrink': 0.8})

# 自定义样式
ax.set_title('Performance Score (OP.)',
             fontsize=16, fontweight='light', pad=20)
ax.set_xlabel('Max Edit Inter', fontsize=12, fontweight='light')
ax.set_ylabel('Error Threshold', fontsize=12, fontweight='light')

# 旋转x轴标签以便更好地显示
plt.setp(ax.get_xticklabels(), rotation=45, ha='center')
plt.setp(ax.get_yticklabels(), rotation=0)

# 添加网格线（可选）
ax.grid(False)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存高分辨率图像
fig.savefig('performance_analysis_heatmap.pdf', dpi=300, bbox_inches='tight')