# -*- codeing=utf-8 -*-
# @Time:2023/10/8 22:29
# @Author: 杨又菁
# @File:Elements.py
# @Software:PyCharm

import numpy as np


class Element:
    def __init__(self, num,contour=None ):
        self.num = num
        self.contour = contour
        self.ele_type = "notset"
        self.boundary = []
        self.neighbor_left = None
        self.neighbor_right = None

    def get_contour(self):
        return self.contour

    # 该元素所属边界S
    def belong_S(self, boundary):
        self.boundary.append(boundary)

    # 判断该元素的类型
    # fixed = 2, repeatable = 1, empty = 0
    def judge_type(self):
        if len(self.boundary) == 2:
            self.ele_type = "fixed"
        elif len(self.boundary) == 1:
            self.ele_type = "repeatable"
        else:
            self.ele_type = "empty"
