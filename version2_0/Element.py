# -*- codeing=utf-8 -*-
# @Time:2024/3/18 4:07
# @Author: 杨又菁
# @File:Element.py
# @Software:PyCharm


import numpy as np


class Element:
    def __init__(self, num, contour=None):
        self.num = num
        self.contour = contour
        # 固定的，可重复的，空的
        self.ele_type = "notset"
        # 中心元素、边界元素，两者都有
        self.bdy_type = "notset"
        # 南北朝向
        self.dir_type = "notset"
        # 所属边界
        self.boundarySegment = []
        # 所属中心线
        self.boundaryInner = None
        self.neighbor_left = None
        self.neighbor_right = None
        self.floor = -1

    def get_contour(self):
        return self.contour

    # 该元素所属边界S
    def belong_Seg(self, boundary):
        self.boundarySegment.append(boundary)

    # 该元素所属中间线
    def belong_In(self, boundary):
        self.boundaryInner = boundary

    # 判断该元素的类型
    # fixed = 2, repeatable = 1, empty = 0
    def judge_ele_type(self):
        if len(self.boundarySegment) == 2:
            self.ele_type = "fixed"
        elif len(self.boundarySegment) == 1:
            self.ele_type = "repeatable"
        else:
            self.ele_type = "empty"

    # 判断该元素的位置
    # 边界元素outer, 内部元素inner, 两者都是both
    def judge_bdy_type(self):
        if self.boundaryInner is None:
            self.bdy_type = "outer"
        elif len(self.boundarySegment) == 0:
            self.bdy_type = "inner"
        else:
            self.bdy_type = "both"

    def cal_min_bounding_box(self):
        min_x = min(point[0] for point in self.contour)
        min_y = min(point[1] for point in self.contour)
        max_x = max(point[0] for point in self.contour)
        max_y = max(point[1] for point in self.contour)
        return [(min_x, min_y), (max_x, max_y)]

    # 判断该元素的朝向
    # 南，北
    def judge_dir_type(self):
        min_bounding_box = self.cal_min_bounding_box()
        long = min_bounding_box[1][0] - min_bounding_box[0][0]
        width = min_bounding_box[1][1] - min_bounding_box[0][1]
        if width > long:
            self.dir_type = "north/south"
        else:
            self.dir_type = "east/west"

    # 给曲线细化，增加坐标点
    def refine_curve(self,target_num_points):
        num_segments = len(self.contour) - 1
        segment_length = [np.linalg.norm(np.array(self.contour[i + 1]) - np.array(self.contour[i])) for i in range(num_segments)]
        total_length = sum(segment_length)
        target_segment_length = total_length / (target_num_points - 1)

        refined_points = [self.contour[0]]
        current_length = 0

        for i in range(num_segments):
            while current_length < target_segment_length:
                t = (current_length - target_segment_length * i) / segment_length[i]
                new_point = self.bezier_curve(np.array(self.contour[i]), np.array(self.contour[i + 1]), np.array(self.contour[i + 2]), np.array(self.contour[i + 3]), t)
                refined_points.append(new_point)
                current_length += np.linalg.norm(np.array(new_point) - np.array(refined_points[-2]))
            current_length -= target_segment_length

        refined_points.append(self.contour[-1])

        return refined_points

    def bezier_curve(self,p0, p1, p2, p3, t):
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
