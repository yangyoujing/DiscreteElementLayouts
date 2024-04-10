import matplotlib.pyplot as plt
import numpy as np

# 创建一个示例图形，假设图形是一个正方形
original_points = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])

# 指定缩放因子
scale_x = 1.5  # x轴方向的缩放因子
scale_y = 1.5  # y轴方向的缩放因子

# 计算缩放后的点坐标
scaled_points = original_points * [scale_x, scale_y]

# 绘制原始图形
plt.plot(original_points[:, 0], original_points[:, 1], label='原始图形', marker='o')
# 绘制缩放后的图形
plt.plot(scaled_points[:, 0], scaled_points[:, 1], label='缩放后的图形', marker='x')

plt.axis('equal')  # 设置坐标轴比例相等，以保持图形不变形
plt.legend()
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('各向异性缩放示例')
plt.grid(True)
plt.show()
