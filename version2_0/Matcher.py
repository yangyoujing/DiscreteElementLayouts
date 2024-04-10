# -*- codeing=utf-8 -*-
# @Time:2024/3/18 5:54
# @Author: 杨又菁
# @File:Matcher.py
# @Software:PyCharm

# 首先比较面积，面积不得相差超过20%
# 再把两个小区的坐标移到坐标原点
# 最后比较重合的面积，重合的面积越大越好
import json
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

class NeighborhoodMatcher:
    def __init__(self, similar_neibh):
        self.similar_neibh = similar_neibh

    def find_most_similar(self, neighborhood_data, area_threshold=0.2):
        most_similar_neighborhood = None
        max_overlap_area = 0
        i=0
        for neighborhood_info in neighborhood_data:
            i+=1
            neighborhood_boundary = neighborhood_info["boundary"]
            if self.is_similar_area(neighborhood_boundary, area_threshold) :
                overlap_area = self.calculate_overlap_area(neighborhood_boundary)
                if overlap_area > max_overlap_area and neighborhood_info["_id"]!="61f4d725ea20fd79e35c328d":
                    max_overlap_area = overlap_area
                    most_similar_neighborhood_id = neighborhood_info["_id"]


        return most_similar_neighborhood_id

    def is_similar_area(self, other_neighborhood, area_threshold):
        known_area = self.calculate_area(self.similar_neibh)
        # print("Known area:", known_area)
        other_area = self.calculate_area(other_neighborhood)
        # print("Other area:", other_area)
        area_difference = abs(known_area - other_area) / known_area
        return area_difference <= area_threshold

    def calculate_area(self, neighborhood):
            # 计算多边形面积，使用 Shoelace formula
        x = [point[0] for point in neighborhood]
        y = [point[1] for point in neighborhood]
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))

    def calculate_overlap_area(self, other_neighborhood):
        poly1 = Polygon([(float(coord[0]), float(coord[1])) for coord in self.similar_neibh])
        poly2 = Polygon([(float(coord[0]), float(coord[1])) for coord in other_neighborhood])

        if not poly1.intersects(poly2) :
            return 0
        else:
            intersection = poly1.intersection(poly2)
            return intersection.area