# -*- coding: utf-8 -*- 
# @Time : 2022/2/28 21:35 
# @Author : zzd 
# @File : douglas.py 
# @desc:    Douglas_Peucker算法

import math
import numpy as np
import cv2


class Point:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]

# 执行道格拉斯-普克算法，用于对给定的切割后的点列表进行抽稀，找出关键点。
# 通过计算点到连线的距离，选择距离足够远的点作为关键点，并进行递归调用以找到更多的关键点。
# 最终，将关键点添加到端点列表中

#道格拉斯-普克算法（Douglas-Peucker algorithm）
# 是一种用于抽稀（曲线简化）的算法。它可以对曲线或折线进行抽稀，保留曲线的主要形状特征，同时减少点的数量
def contour_extract(contour):
    endpoint_list = []
    point_list = []
    # 将轮廓contour中的每个点，转换为Point对象，并将其添加到point_list列表中
    for p in contour:
        point_list.append(Point(p[0]))
    # 寻找左上方的点
    left_up_index = __search_left_up_point_index(point_list)
    # 寻找与该点距离最远的点
    maxlength_index = __search_maxlength_index(point_list[left_up_index], point_list)
    # 将point_list切割成两段，同时保留切割端点
    cut_list1, cut_list2 = __cut_point_list(left_up_index, maxlength_index, point_list)
    endpoint_list.append(point_list[left_up_index])
    endpoint_list.append(point_list[maxlength_index])
    dougals_peucker(cut_list1, endpoint_list)
    dougals_peucker(cut_list2, endpoint_list)


    endpoint_index_list = []
    for i in range(len(point_list)):
        if point_list[i] in endpoint_list:
            endpoint_index_list.append(i)

    t = contour[endpoint_index_list[0]]
    for i in range(len(endpoint_index_list) - 1):
        t = np.concatenate((t, contour[endpoint_index_list[i + 1]]), axis=1)
        # t = np.append(contour[endpoint_index_list[i + 1]])

    t = t.reshape((len(endpoint_index_list), 1, 2))

    return t



# dougals_peucker算法，递归实现
def dougals_peucker(cut_list, endpoint_list):
    max_length = 0
    index = -1
    for i in range(len(cut_list)):
        if i != 0 and i != len(cut_list) - 1:
            length = __get_distance_from_point_to_line(cut_list[i], cut_list[0], cut_list[-1])
            if length > max_length:
                max_length = length
                index = i
    if max_length > 2:
        endpoint_list.append(cut_list[index])
        cut_list1 = cut_list[0:index]
        cut_list1.append(cut_list[index])
        cut_list2 = cut_list[index:]
        dougals_peucker(cut_list1, endpoint_list)
        dougals_peucker(cut_list2, endpoint_list)


# 点到另外两点组成的直线的距离
def __get_distance_from_point_to_line(point, line_point1, line_point2):
    # 对于两点坐标为同一点时,返回点与点的距离
    if line_point1 == line_point2:
        return math.hypot(point.x - line_point1.x, point.y - line_point1.y)
    # 计算直线的三个参数
    A = line_point2.y - line_point1.y
    B = line_point1.x - line_point2.x
    C = (line_point1.y - line_point2.y) * line_point1.x + \
        (line_point2.x - line_point1.x) * line_point1.y
    # 根据点到直线的距离公式计算距离
    distance = np.abs(A * point.x + B * point.y + C) / (np.sqrt(A ** 2 + B ** 2))
    return distance


# 将point_list切割成两段，同时保留切割端点
# 如果index2大于index1，则切割方式为从index1到index2的子列表，
# 并将index2处的点添加到子列表末尾。
# 同时，将index2之后的点和index1之前的点合并为另一段子列表，并将index1处的点添加到该子列表末尾
def __cut_point_list(index1, index2, point_list):
    if index2 > index1:
        cut_list1 = point_list[index1:index2]
        cut_list1.append(point_list[index2])
        cut_list2 = point_list[index2:] + point_list[0:index1]
        cut_list2.append(point_list[index1])
    elif index2 < index1:
        index1, index2 = index2, index1
        cut_list1 = point_list[index1:index2]
        cut_list1.append(point_list[index2])
        cut_list2 = point_list[index2:] + point_list[0:index1]
        cut_list2.append(point_list[index1])
    else:
        print()
    return cut_list1, cut_list2


# 寻找与点距离最远的点
def __search_maxlength_index(target_point, point_list):
    maxlength = 0
    index = -1
    for i in range(len(point_list)):
        item = point_list[i]
        length = math.hypot(item.x - target_point.x, item.y - target_point.y)
        if length > maxlength:
            maxlength = length
            index = i
    return index


# 寻找左上的点，opencv生成的坐标，左上方是原点
def __search_left_up_point_index(point_list):
    index = 0
    min_x = math.inf
    min_y = math.inf
    for i in range(len(point_list)):
        if point_list[i].x < min_x:
            min_x = point_list[i].x
            min_y = point_list[i].y
            index = i
        elif point_list[i].x == min_x:
            if point_list[i].y < min_y:
                min_y = point_list[i].y
                index = i
    return index
