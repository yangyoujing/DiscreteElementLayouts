# -*- coding: utf-8 -*- 
# @Time : 2022/3/1 19:58 
# @Author : zzd 
# @File : coordinate_trans.py 
# @desc:    坐标转化

import math
import numpy as np
import pandas as pd
import re

# 用于将提取的点列表转换为CAD坐标系中的点列表
def trans(extracted_list, base_point, pixel_base_point, scale):
    cad_points_list = []
    for i in range(len(extracted_list)):
        cad_points_item = []
        for j in range(len(extracted_list[i])):
            x, y = extracted_list[i][j][0]
            cad_point = __get_cad_point([x, y], base_point, pixel_base_point, scale)
            cad_points_item.append([cad_point])
        cad_points_item = np.array(cad_points_item).reshape((len(cad_points_item), 1, 2))
        cad_points_list.append(cad_points_item)
    return cad_points_list

def magic(cad_points_list):
    point_info_read = pd.read_csv("./road_point.csv")
    point_data = []
    for i in range(len(point_info_read)):
        item = point_info_read.iloc[i]
        point_data.append([np.float32(item['x']), np.float32(item['y'])])
    # s = point_info_read['pos']
    # s = s.tolist()
    # point_data = []
    # for i in range(len(s)):
    #     pattern = r'@(.*?)@'
    #     pos_pattern = re.findall(pattern, s[i])[0]
    #     temp = pos_pattern.split(',')
    #     for j in range(len(temp)):
    #         temp[j] = np.float32(temp[j])
    #     point_data.append(temp)

    for i in range(len(cad_points_list)):
        l = []
        for j in range(len(cad_points_list[i])):
            item = cad_points_list[i][j]
            item = __search_closed(point_data, item[0])
            l.append(item)
        cad_points_list[i] = np.array(l, dtype=np.float32).reshape((-1,1,2))
    return cad_points_list

# 用于在给定的点列表中查找最接近的点
def __search_closed(point_data, point):
    m = 1000
    index = 0
    for i in range(len(point_data)):
        s = __cal_length(point_data[i], point)
        if s < m:
            index = i
            m = s;

    if m != 1000 and m < 3:
        return point_data[index]
    return point.tolist()

# 获取cad中真实的点坐标
def __get_cad_point(point, base_point, pixel_base_point, scale):
    pixel_x_length = point[0] - pixel_base_point[0]
    pixel_y_length = point[1] - pixel_base_point[1]
    cad_x_length = pixel_x_length * scale
    cad_y_length = -pixel_y_length * scale
    new_point = [base_point[0] + cad_x_length, base_point[1] + cad_y_length]
    return new_point


# 计算两点之间的距离
def __cal_length(point1, point2):
    return math.hypot(point1[0] - point2[0], point1[1] - point2[1])
