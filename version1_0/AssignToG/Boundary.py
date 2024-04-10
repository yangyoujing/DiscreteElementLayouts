# -*- codeing=utf-8 -*-
# @Time:2023/10/8 22:30
# @Author: 杨又菁
# @File:Boundary.py
# @Software:PyCharm
import numpy as np


class BoundaryOut:
    def __init__(self, contour=None):
        self.contour = contour

    # 获取轮廓
    def get_contour(self):
        return self.contour

    def get_len(self):
        return len(self.contour)

    # 把最外面的边界分割成小的边界
    def split_to_segments(self):
        S_temp = []
        # print(self.contour)
        for i in range(len(self.contour) - 1):
            Si = []
            line_head = []
            line_head.append(self.contour[i][0])
            line_head.append(self.contour[i][1])
            # print(type(line_segment))
            Si.append(np.array(line_head))
            line_tail = []
            line_tail.append(self.contour[i + 1][0])
            line_tail.append(self.contour[i + 1][1])
            Si.append(np.array(line_tail))
            S_temp.append(np.array(Si))
            # 绘制近似的轮廓
        Si = []
        line_head = []
        line_head.append(self.contour[-1][0])
        line_head.append(self.contour[-1][1])
        # print(type(line_segment))
        Si.append(np.array(line_head))
        line_tail = []
        line_tail.append(self.contour[0][0])
        line_tail.append(self.contour[0][1])
        Si.append(np.array(line_tail))
        S_temp.append(np.array(Si))
        # print("S:========", S)
        # print(type(S))
        S_temp = np.array(S_temp)
        return S_temp


class BoundarySegment(BoundaryOut):
    total_num = 0

    def __init__(self, num,contour=None):
        BoundarySegment.total_num += 1
        super().__init__(contour)
        self.num = num
        self.direction = None
        self.elements=[]
        if self.contour is not None:
            self.direction=(contour[1][0]-contour[0][0],contour[1][1]-contour[0][1])

    # 该边界有哪些元素
    def enroll_P(self,element):
        self.elements.append(element)
        print("已加入元素",element)

    def __del__(self):
        self.total_num -= 1
