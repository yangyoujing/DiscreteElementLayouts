# -*- codeing=utf-8 -*-
# @Time:2023/9/23 16:21
# @Author: 杨又菁
# @File:main.py
# @Software:PyCharm
import sys
sys.path.append('../AssignToG')
import csv
import numpy as np
from Boundary import *
from Elements import *

import cv2

# img = cv2.imread('2.png')


# # 把csv中的一行轮廓提取出来
# def csv_row_to_pointlist(row):
#     # print(row)
#     temp = row[1].split(',')
#     segment = []
#     i = 0
#     # print("temp_len:", len(temp))
#     while i < len(temp) - 1:
#         point = []
#         point.append(round(float(temp[i])))
#         point.append(round(float(temp[i + 1])))
#         segment.append(point)
#         i = i + 2
#         # print("type(P_segment):",type(P_segment))
#     # print("[np.array(segment)]:_______________", [np.array(segment)])
#     cv2.drawContours(img, [np.array(segment)], -1, (0, 255, 255), 1)
#     return np.array(segment)


def csv_row_from_elements(row, Element):
    # print(row)
    Element.ele_type = row[1]
    Element.boundary = row[2]
    Element.neighbor_left = row[3]
    Element.neighbor_right = row[4]
    temp = row[5].split(',')
    # print("读取元素坐标后",temp)
    segment = []
    i = 0
    # print("temp_len:", len(temp))
    while i < len(temp) - 1:
        point = []
        point.append(round(float(temp[i])))
        point.append(round(float(temp[i + 1])))
        segment.append(point)
        i = i + 2
    # cv2.drawContours(img, [np.array(segment)], -1, (0, 255, 255), 1)
    Element.contour = segment
    return Element


def csv_row_from_boundaryout(row, BoundaryOut):
    print(row)
    temp = row[0].split(',')
    segment = []
    i = 0
    print("temp_len:", len(temp))
    while i < len(temp) - 1:
        point = []
        point.append(round(float(temp[i])))
        point.append(round(float(temp[i + 1])))
        segment.append(point)
        i = i + 2
    # cv2.drawContours(img, [np.array(segment)], -1, (0, 255, 255), 1)
    BoundaryOut.contour = segment
    print("BoundaryOut.contour:",BoundaryOut.contour)
    return BoundaryOut


# 计算MVC权重
def calculate_mvc_weights(point, polygon):
    n = len(polygon)
    weights = np.zeros(n)

    for i in range(n):
        # 计算当前多边形边的两个顶点
        vi = polygon[i]
        vi_min1 = polygon[(i - 1 + n) % n]
        vi_plus1 = polygon[(i + 1) % n]
        vi_p = (vi - point) / np.linalg.norm(vi - point)  # 计算点 p 到顶点 vi 的归一化向量
        # vi_p = vi - point

        # 计算 ang1 和 ang2
        # print("vi_min1 - point, vi_p",vi_min1 - point, vi_p)
        ang1 = getAngle(vi_min1 - point, vi_p)
        # print("ang1:",ang1)
        # print("vi_p, vi_plus1 - point",vi_p, vi_plus1 - point)
        ang2 = getAngle(vi_p, vi_plus1 - point)
        # print("ang2:", ang2)

        # 计算 t1 和 t2
        t1 = np.tan(ang1 * 0.5)
        t2 = np.tan(ang2 * 0.5)

        # 计算权重
        weights[i] = (t1 + t2) / np.linalg.norm(vi - point)
        # print("weights[",i,"]:",weights[i])

        # 归一化权重
    sumweights = np.sum(weights)
    weights /= sumweights

    return weights


def getAngle(A, B):
    return np.arctan2(np.cross(A, B), np.dot(A, B))


with open('../CSV/target_building_2.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)
    Target_B = csv_row_from_boundaryout(next(csv_reader), BoundaryOut())
    print("Target_B.contour:",Target_B.contour)

# 读取elements
with open('../CSV/elements_2.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)

    org_segments = []
    i = 0
    for row in csv_reader:
        temp_element = Element(i)
        temp_element = csv_row_from_elements(row, temp_element)
        # print("temp_element.contour:",temp_element.contour)
        org_segments.append(temp_element)
        i += 1

    # print("org_segments:",org_segments)

# 读取boundaryout
with open('../CSV/boundaryout_2.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)
    B = csv_row_from_boundaryout(next(csv_reader), BoundaryOut())

if __name__ == '__main__':
    segmenter_contour = np.ones((1000, 1200, 3), dtype=np.uint8) * 0

    # 提取B，最外面的边界

    # print("Target_B:]]]]]]]]]]]]]]]]]]]]]", Target_B)
    # print("B:]]]]]]]]]]]]]]]]]]]]]", B)
    # cv2.drawContours(segmenter_contour, [np.array(Target_B)], -1, (0, 255, 255), 1)
    # cv2.drawContours(segmenter_contour, [np.array(B)], -1, (0, 255, 255), 1)

    # 假设有源轮廓和目标轮廓的点坐标
    s_contour = np.array(B.contour)
    t_contour = np.array(Target_B.contour)
    # print("s_contour:",s_contour)
    # print("t_contour:", t_contour)


    # 进行轮廓分割,把目标轮廓B分为一小段的S
    S_temp = np.array(Target_B.split_to_segments())
    S = []
    for i in range(B.get_len()):
        S.append(BoundarySegment(i, S_temp[i]))

    # 把分段边界S信息写入target_boundarysegment.csv
    sort_data = []
    for i in range(len(S)):
        sort_temp = []
        sort_temp.append(S[i].num)  # 编号
        # 包含哪些元素
        flatted = np.array(S[i].elements).flatten()
        temp_str = ','.join(str(x) for x in flatted)
        sort_temp.append(temp_str)
        # 边界方向
        flatted = np.array(S[i].direction).flatten()
        temp_str = ','.join(str(x) for x in flatted)
        sort_temp.append(temp_str)
        # 坐标
        flatted = np.array(S[i].contour).flatten()
        temp_str = ','.join(str(x) for x in flatted)
        sort_temp.append(temp_str)
        sort_data.append(sort_temp)

    # 写入csv文件
    with open('../CSV/target_boundarysegment_2.csv', 'w') as file:
        csvwriter = csv.writer(file, lineterminator='\n')
        csvwriter.writerows(sort_data)




    # test数据
    # t_contour=[[689,539],[689,476],[549,384],[299,384],[299,539]]
    # t_contour = [[689, 539], [689, 476], [689, 384], [299, 384], [299, 539]]
    # t_contour = [[989,739],[989,576],[549,484],[299,384],[299,739]]
    cv2.drawContours(segmenter_contour, [np.array(t_contour)], -1, (0, 255, 255), 1)

    P_changed = []
    # print("len(org_segments):",len(org_segments))
    # print(org_segments)
    for i in range(len(org_segments)):
        # i=0
        # 初始化映射轮廓
        # print("org_segments[",i,"].contour:",org_segments[i].contour)
        temp = Element(i)
        temp.ele_type = org_segments[i].ele_type
        temp.boundary = org_segments[i].boundary
        temp.neighbor_left = org_segments[i].neighbor_left
        temp.neighbor_right = org_segments[i].neighbor_right

        mapped_contour_P = np.empty_like(np.array(org_segments[i].contour))
        # print("org_segments[",i,"].contour:",org_segments[i].contour)
        for j in range(len(org_segments[i].contour)):
            source_point = np.array(org_segments[i].contour[j])
            # print("source_point:",source_point)
            weights = calculate_mvc_weights(source_point, s_contour)

            # print("weights", weights)

            # 计算映射坐标
            mapped_point = np.dot(weights, t_contour)

            # 存储映射点坐标
            mapped_contour_P[j] = mapped_point

        temp.contour = mapped_contour_P
        P_changed.append(temp)
        # 打印映射轮廓
        print("mapped_contour_P:\n", [np.array(mapped_contour_P)])
        cv2.drawContours(segmenter_contour, [np.array(mapped_contour_P)], -1, (0, 255, 255), 1)

    # 把元素信息写入segments_changed.csv
    sort_data = []
    for i in range(len(P_changed)):
        sort_temp = []
        sort_temp.append(P_changed[i].num)  # 编号
        sort_temp.append(P_changed[i].ele_type)  # 类型
        sort_temp.append(P_changed[i].boundary)  # 所属边界
        sort_temp.append(P_changed[i].neighbor_left)  # 左邻居
        sort_temp.append(P_changed[i].neighbor_right)  # 右邻居
        # 坐标
        flatted = np.array(P_changed[i].contour).flatten()
        temp_str = ','.join(str(x) for x in flatted)
        sort_temp.append(temp_str)
        sort_data.append(sort_temp)

    # 写入csv文件
    with open('../CSV/elements_changed_2.csv', 'w') as file:
        csvwriter = csv.writer(file, lineterminator='\n')
        csvwriter.writerows(sort_data)

    cv2.imshow('Segmented Contour', segmenter_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
