# -*- codeing=utf-8 -*-
# @Time:2023/9/24 13:36
# @Author: 杨又菁
# @File:main.py
# @Software:PyCharm
import csv
import sys
sys.path.append('../AssignToG')
import cv2
import numpy as np
from Elements import *


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


def AnisotropicScaling(factor, element):
    print(np.array(element.contour).astype(np.float32))
    hull=cv2.convexHull(np.array(element.contour).astype(np.float32))


    # 计算凸包的中心点
    M = cv2.moments(hull)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    # 遍历凸包的每个点，根据中心点和缩放因子计算新的坐标
    scaled_hull = []
    for point in hull:
        x, y = point[0]
        new_x = int(cX + (x - cX) * factor)
        new_y = int(cY + (y - cY) * factor)
        scaled_hull.append([[new_x, new_y]])
    element.contour =scaled_hull

    # # 指定缩放因子
    # scale_x = factor  # x轴方向的缩放因子
    # scale_y = factor  # y轴方向的缩放因子
    #
    # # 计算缩放后的点坐标
    # scaled_points = ([scale_x, scale_y] * np.array(element.contour)).astype(np.int32)
    # print("scaled_points", scaled_points)
    # element.contour=scaled_points

    return element


segmenter_contour = np.ones((1000, 1200, 3), dtype=np.uint8) * 0

# 读取elements
with open('elements.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)

    org_segments = []
    i = 0
    for row in csv_reader:
        temp_element = Element(i)
        temp_element = csv_row_from_elements(row, temp_element)
        # print("temp_element.contour:",temp_element.contour)
        cv2.drawContours(segmenter_contour, np.array([temp_element.contour]), -1, (0, 255, 255), 1)
        org_segments.append(temp_element)
        i += 1

if __name__ == '__main__':


    # 创建一个示例图形，假设图形是一个正方形
    # org_segments=[[341, 243], [371, 243], [371, 222], [341, 223]]

    #   cv2.drawContours(segmenter_contour, np.array([original_points]), -1, (0, 255, 255), 1)
    scale_segments = []
    for element in org_segments:
        temp=AnisotropicScaling(1.5,element)
        scale_segments.append(temp)
        cv2.drawContours(segmenter_contour, np.array([temp.contour]), -1, (0, 255, 255), 1)

    # 把元素信息写入scale_segments.csv
    sort_data = []
    for i in range(len(scale_segments)):
        sort_temp = []
        sort_temp.append(scale_segments[i].num)  # 编号
        sort_temp.append(scale_segments[i].ele_type)  # 类型
        sort_temp.append(scale_segments[i].boundary)  # 所属边界
        sort_temp.append(scale_segments[i].neighbor_left)  # 左邻居
        sort_temp.append(scale_segments[i].neighbor_right)  # 右邻居
        # 坐标
        flatted = np.array(scale_segments[i].contour).flatten()
        temp_str = ','.join(str(x) for x in flatted)
        sort_temp.append(temp_str)
        sort_data.append(sort_temp)

    # 写入csv文件
    with open('../CSV/scale_segments.csv', 'w') as file:
        csvwriter = csv.writer(file, lineterminator='\n')
        csvwriter.writerows(sort_data)


    cv2.imshow('Segmented Contour', segmenter_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
