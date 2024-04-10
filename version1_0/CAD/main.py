# -*- codeing=utf-8 -*-
# @Time:2023/9/16 11:43
# @Author: 杨又菁
# @File:main.py
# @Software:PyCharm

import cv2
import numpy as np
import douglas
import coordinate_trans
import math
import csv

if __name__ == '__main__':

    # 图像处理部分

    print("开始")
    # 读取图像
    # img = cv2.imread('3.png')  # 读取源block
    img = cv2.imread('PNG/4.png') # 读取目标block

    # 将图像转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 转换为二值图像
    ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # 100可改
    # 提取轮廓
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # print(contours)

    # 通过点的个数排除不合格的轮廓
    i = len(contours) - 1
    # while i < len(contours):
    #     if len(contours[i]) < 10 or len(contours[i]) > 800:
    #         del contours[i]
    #         i = i - 1
    #     i = i + 1

    print(i)

    # 过滤掉重叠的轮廓

    # 定义距离阈值
    distance_threshold = 0.001
    # 过滤相近的轮廓
    filtered_contours = []
    for j, contour_j in enumerate(contours):
        keep_contour = True
        for k, contour_k in enumerate(contours):
            if j != k:
                distance = cv2.matchShapes(contour_j, contour_k, cv2.CONTOURS_MATCH_I1, 0)
                if distance < distance_threshold:
                    keep_contour = False
                    break
        if keep_contour:
            filtered_contours.append((contour_j))

    print("长度", len(filtered_contours))
    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

    # 在图像上绘制轮廓并显示图像

    img_temp = np.zeros((1121, 1121, 3))  # 画布大小

    i = len(filtered_contours) - 1

    # 循环画出全部
    # temp = [filtered_contours[i]]
    # for i in range(len(filtered_contours)):
    #     cv2.drawContours(img_temp, filtered_contours, i, (0, 0, 255), 1)
    #     cv2.imshow("img_temp", img_temp)
    #     cv2.waitKey(0)

    # 一个一个画出来
    # cv2.drawContours(img_temp, filtered_contours, -1, (0, 0, 255), 1)
    # cv2.imshow("img_temp", img_temp)
    # cv2.waitKey(0)

    # 等待用户按下任意键退出

    # 比例尺部分
    # 根据基准点和基准长度，以及对应的像素点和像素长度，计算出一个比例尺，以便在后续的处理中将像素值转换为实际世界的长度单位
    base_point1 = [493.506, 252.796]
    base_point2 = [503.4143, 234.4849]
    base_length = 20.82

    # pixel_base_point1 = [55, 591]
    # pixel_base_point2 = [107, 612]
    pixel_base_point1 = [355, 593]
    pixel_base_point2 = [372, 624]
    pixel_length = math.hypot(pixel_base_point1[0] - pixel_base_point2[0], pixel_base_point1[1] - pixel_base_point2[1])

    scale = base_length / pixel_length

    # 循环遍历contours列表中的每个轮廓，并通过调用douglas.contour_extract()函数对每个轮廓进行处理
    extracted_list = []
    j = 0
    while j < len(filtered_contours):
        print("=================================")

        if len(filtered_contours[j]) > 500:
            del filtered_contours[j]
            j = j - 1
        j = j + 1

    # 一个一个画出来
    # cv2.drawContours(img_temp, filtered_contours, -1, (0, 0, 255), 1)
    # cv2.imshow("img_temp", img_temp)
    # cv2.waitKey(0)

    # 循环画出全部
    temp = [filtered_contours[i]]
    for i in range(len(filtered_contours)):
        cv2.drawContours(img_temp, filtered_contours, i, (0, 0, 255), 1)
        print("filtered_contours_______________",filtered_contours)
        cv2.imshow("img_temp", img_temp)
        cv2.waitKey(0)

    for j in range(len(filtered_contours)):
        extracted = douglas.contour_extract(filtered_contours[j])
        extracted_list.append(extracted)
    print(extracted_list)

    cad_points_list = coordinate_trans.trans(extracted_list, base_point1, pixel_base_point1, scale)

    print("cad_points_list:",cad_points_list)
    # 将cad_points_list中的数据进行处理和转换，以便后续使用或存储
    sort_data = []
    for i in range(len(cad_points_list)):
        # print("cad_points_list",cad_points_list[i])
        sort_temp = [len(cad_points_list[i])]
        flatted = cad_points_list[i].flatten()
        temp_str = ','.join(str(x) for x in flatted)
        # a2 = np.array([float(x) for x in str.split(',')])
        sort_temp.append(temp_str)
        sort_data.append(sort_temp)

    # 将数据存储为CSV文件，以便后续的读取、处理或分析
    # with open('input_building.csv', 'w') as file:
    #     csvwriter = csv.writer(file, lineterminator='\n')
    #     csvwriter.writerows(sort_data)
    #     # csvwriter.writerows(cad_points_list)

    with open('../CSV/target_building_4.csv', 'w') as file:
        csvwriter = csv.writer(file, lineterminator='\n')
        csvwriter.writerows(sort_data)
        # csvwriter.writerows(cad_points_list)

    print()
