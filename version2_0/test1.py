import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors

# 多边形轮廓坐标
polygon_points = np.array([[19.88722985610366, 0.0], [250.61116424761713, 49.09329983126372], [151.2657275106758, 346.4095951858908], [69.40622988343239, 321.2177796475589], [115.6733187418431, 119.74540476035327], [0.0, 90.9737235782668], [19.88722985610366, 0.0]])


# 计算角度
def calculate_angle(B, A, C):
    # 计算向量 BA 和 BC
    BA = A - B
    BC = C - B

    # 检查向量长度是否为零
    if np.linalg.norm(BA) == 0 or np.linalg.norm(BC) == 0:
        return np.nan

    # 计算点积
    dot_product = np.dot(BA, BC)

    # 计算向量 BA 和 BC 之间的夹角（弧度）
    cos_angle = dot_product / (np.linalg.norm(BA) * np.linalg.norm(BC))
    angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))

    # 将弧度转换为角度并返回
    angle_deg = np.degrees(angle_rad)
    return angle_deg


# 划分轮廓边界
edges = []
current_edge = []
current_edge.append(polygon_points[0])
current_edge.append(polygon_points[1])
for i in range(2,len(polygon_points)):
    angle = calculate_angle(polygon_points[(i -1)% len(polygon_points)],
                            polygon_points[(i - 2) % len(polygon_points)],
                            polygon_points[i % len(polygon_points)])
    print("angle:", angle)
    if np.isnan(angle) or angle >= 140:
        current_edge.append(polygon_points[i])
    else:
        print("current_edge:", current_edge)
        edges.append(np.array(current_edge))
        temp=current_edge[-1]
        current_edge = []
        current_edge.append(temp)
        current_edge.append(polygon_points[i])

edges.append(np.array(current_edge))
print("edges:", edges)
# 绘制每个边界
for i, edge in enumerate(edges):
    # 排序多边形边界上的点，确保x值严格递增
    edge = edge[np.argsort(edge[:, 0])]
    plt.plot(edge[:, 0], edge[:, 1], marker='o', linestyle='-', label=f'Edge {i + 1}')

    # 拟合曲线
    x = edge[:, 0]
    y = edge[:, 1]
    spline = CubicSpline(x, y)
    x_interp = np.linspace(min(x), max(x), 100)
    y_interp = spline(x_interp)

    # 计算原始边界的颜色
    orig_color = plt.gca().lines[-1].get_color()

    # 计算互补颜色
    orig_rgb = mcolors.hex2color(mcolors.to_hex(orig_color))
    comp_rgb = [1 - orig_rgb[0], 1 - orig_rgb[1], 1 - orig_rgb[2]]
    comp_color = mcolors.to_hex(comp_rgb)

    # 绘制拟合曲线，并使用互补颜色
    # plt.plot(x_interp, y_interp, color=comp_color)
# 标记每个点
for i, edge in enumerate(edges):
    for j, point in enumerate(edge):
        plt.text(point[0], point[1], f'{j}', fontsize=5, ha='right')

# 设置图形标题和标签
plt.title('Edges with Points')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)
plt.axis('equal')
plt.legend()
plt.show()
