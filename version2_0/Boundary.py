# -*- codeing=utf-8 -*-
# @Time:2024/3/18 4:07
# @Author: 杨又菁
# @File:Boundary.py
# @Software:PyCharm
import copy
import math

import numpy as np
from scipy.interpolate import CubicSpline


class BoundaryOut:
    def __init__(self, contour=None):
        self.contour = contour

    # 获取轮廓
    def get_contour(self):
        return self.contour

    def get_len(self):
        return len(self.contour)

    # 计算角度
    def calculate_angle(self, B, A, C):
        # 计算向量 BA 和 BC
        BA = [A[i] - B[i] for i in range(len(A))]
        BC = [C[i] - B[i] for i in range(len(C))]

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

    # 把最外面的边界分割成小的边界(返回的是坐标)
    def split_to_segments(self):
        # 划分轮廓边界
        edges = []
        current_edge = []
        current_edge.append(self.contour[0])
        current_edge.append(self.contour[1])
        for i in range(2, len(self.contour)):
            angle = self.calculate_angle(self.contour[(i - 1) % len(self.contour)],
                                         self.contour[(i - 2) % len(self.contour)],
                                         self.contour[i % len(self.contour)])
            # print("angle:", angle)
            if np.isnan(angle) or angle >= 140:
                current_edge.append(self.contour[i])
            else:
                # print("current_edge:", current_edge)
                edges.append(np.array(current_edge))
                temp = current_edge[-1]
                current_edge = []
                current_edge.append(temp)
                current_edge.append(self.contour[i])

        edges.append(np.array(current_edge))
        return edges



    def distance(self, point1, point2):
        """计算两个坐标点之间的欧几里得距离"""
        return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


class BoundarySegment(BoundaryOut):
    total_num = 0

    def __init__(self, num, contour=None):
        BoundarySegment.total_num += 1
        super().__init__(contour)
        self.num = num
        self.direction = None
        self.elements = []
        # 角度需要重新考虑如何计算，因为加入了曲线
        # if self.contour is not None:
        #     self.direction = (contour[1][0] - contour[0][0], contour[1][1] - contour[0][1])

    # 该边界有哪些元素
    def enroll_P(self, element):
        self.elements.append(element)
        print("已加入元素:", element)

    def __del__(self):
        self.total_num -= 1

    def find_start_point(self, points):
        """找到乱序坐标点中的起点"""
        start_point = points[0]
        for point in points:
            if point[0] < start_point[0] or (point[0] == start_point[0] and point[1] < start_point[1]):
                start_point = point
        return start_point

    def connect_points(self, points, sort_by_x=True):
        """连接乱序坐标点形成曲线"""
        sorted_points = []
        if sort_by_x:
            s_points = sorted(points, key=lambda x: x[0])
        else:
            s_points = sorted(points, key=lambda x: x[1])
        current_point = self.find_start_point(s_points)
        sorted_points.append(current_point)
        # 用 filter() 函数过滤掉要移除的元素
        s_points = list(filter(lambda x: not np.array_equal(x, current_point), s_points))

        while s_points:
            min_dist = float('inf')
            next_point = None
            for i in range(len(s_points)):
                dist = self.distance(current_point, s_points[i])
                if dist < min_dist:
                    min_dist = dist
                    next_point = s_points[i]
            sorted_points.append(next_point)
            s_points.remove(next_point)
            current_point = next_point

        return sorted_points

    def length(self):
        """计算曲线的长度"""
        # 连接坐标点形成曲线（按照x排列）
        connected_points_x = self.connect_points(self.contour, sort_by_x=True)
        length_x = sum(
            self.distance(connected_points_x[i], connected_points_x[i + 1]) for i in range(len(connected_points_x) - 1))

        # 连接坐标点形成曲线（按照y排列）
        connected_points_y = self.connect_points(self.contour, sort_by_x=False)
        length_y = sum(
            self.distance(connected_points_y[i], connected_points_y[i + 1]) for i in range(len(connected_points_y) - 1))

        # 选择较短的曲线长度
        if length_x < length_y:
            return length_x
        else:
            return length_y

    # 用曲线拟合边界段
    def fit_curve(self, segment):
        # 按照x坐标排序
        data = segment[segment[:, 0].argsort()]

        # 拆分输入数据为x和y坐标
        x = data[:, 0]
        y = data[:, 1]

        # 使用样条插值拟合数据
        spline = CubicSpline(x, y)

        # 生成插值曲线的数据
        x_interp = np.linspace(min(x), max(x), 100)  # 这里可以根据需要调整插值点数
        y_interp = spline(x_interp)

        # 把x_interp和y_interp合并成新的数据数组
        new_segment = np.column_stack((x_interp, y_interp))
        return new_segment

    def addsegment(self, add_p):
        n = len(self.contour)
        length = add_p[-1]
        add_p[-1] = 1
        new_segments = []
        current_length = 0
        segment = []
        j = 0
        for i in range(len(add_p)):
            l = add_p[i] * length
            while j < n - 1:
                segment.append(self.contour[j])
                current_length += self.distance(self.contour[j], self.contour[j + 1])
                # print("current length", current_length, "l:", l)
                if current_length > l:
                    start_point = self.contour[j]
                    end_point = self.contour[j + 1]
                    nl = current_length - l
                    #  计算新的点
                    original_distance = self.distance(start_point, end_point)
                    ratio = nl / original_distance
                    new_point = (end_point[0] + ratio * (start_point[0] - end_point[0]),
                                 end_point[1] + ratio * (start_point[1] - end_point[1]))
                    print("new_point:", new_point)
                    if add_p[i] != 1:
                        segment.append(new_point)
                    else:
                        segment.append(self.contour[-1])
                    new_segments.append(segment)
                    segment = [new_point]
                    j += 1
                    break
                j += 1

        print("self.contour:", self.contour, "new_segments:", new_segments)

        return new_segments


class BoundaryInner(BoundaryOut):
    total_num = 0

    def __init__(self, num, contour=None):
        BoundarySegment.total_num += 1
        super().__init__(contour)
        self.num = num
        self.direction = None
        self.elements = []
        if self.contour is not None:
            self.direction = (contour[1][0] - contour[0][0], contour[1][1] - contour[0][1])

    # 该边界有哪些元素
    def enroll_P(self, element):
        self.elements.append(element)
        print("已加入元素:", element)

    def __del__(self):
        self.total_num -= 1
