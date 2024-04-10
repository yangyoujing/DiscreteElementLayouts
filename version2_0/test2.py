import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# 输入数据
data = np.array([[483.84902211, 380.87456586],
                 [368.03708803, 371.31487273],
                 [288.12971848, 349.47179213],
                 [186.70960287, 318.14054354],
                 [123.73355159, 285.95418122],
                 [64.23914709, 248.79998477],
                 [0.0, 184.35488156]])

# 按照x坐标排序
data = data[data[:, 0].argsort()]

# 拆分输入数据为x和y坐标
x = data[:, 0]
y = data[:, 1]

# 使用样条插值拟合数据
spline = CubicSpline(x, y)

# 生成插值曲线的数据
x_interp = np.linspace(min(x), max(x), 100)
y_interp = spline(x_interp)

# 绘制插值曲线和原始数据
plt.plot(x_interp, y_interp, label='Interpolated Curve')
plt.scatter(x, y, color='red', label='Original Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Spline Interpolation')
plt.legend()
plt.grid(True)
plt.show()

# 打印插值后的x和y坐标
for i in range(len(x_interp)):
    print(f"Point {i+1}: X = {x_interp[i]}, Y = {y_interp[i]}")

print()