import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 创建数据
k_values = [2, 3, 4, 5, 6, 10]
rou_values = [0.1, 0.2, 0.3, 0.5, 0.8,0.98]

# 性能数据矩阵（6行×5列）
performance_data = [
    [0.754, 0.924, 0.877, 0.767, 0.584, 0.432],
    [0.736, 0.893, 0.853, 0.732, 0.543, 0.449],
    [0.712, 0.861, 0.820, 0.743, 0.587, 0.381],
    [0.696, 0.792, 0.761, 0.705, 0.639, 0.329],
    [0.642, 0.723, 0.693, 0.682, 0.667, 0.426],
    [0.631, 0.679, 0.675, 0.663, 0.712, 0.464]
]

# 创建DataFrame
df = pd.DataFrame(performance_data,
                  index=[f'k={k}' for k in k_values],
                  columns=[f'rou={rou}' for rou in rou_values])

# 创建图形
fig, ax = plt.subplots(figsize=(12, 8))

# 绘制热力图
heatmap = sns.heatmap(df,
                      annot=True,
                      fmt=".3f",
                      cmap="RdYlBu_r",  # 反向的红黄蓝配色，红色表示高值
                      linewidths=0.5,
                      linecolor='white',
                      ax=ax,
                      cbar_kws={'label': 'Performance Score (OP)',
                               'shrink': 0.8})

# 自定义样式
ax.set_title('Model Performance Heatmap\nby k and rou Parameters',
             fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('rou Parameter Values', fontsize=12, fontweight='bold')
ax.set_ylabel('k Parameter Values', fontsize=12, fontweight='bold')

# 旋转x轴标签以便更好地显示
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.setp(ax.get_yticklabels(), rotation=0)

# 添加网格线（可选）
ax.grid(False)

# 调整布局
plt.tight_layout()

# 显示图形
plt.show()

# 保存高分辨率图像
# fig.savefig('performance_analysis_heatmap.png', dpi=300, bbox_inches='tight')