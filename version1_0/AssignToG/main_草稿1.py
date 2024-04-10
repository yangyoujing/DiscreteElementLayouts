# -*- codeing=utf-8 -*-
# @Time:2023/9/18 0:37
# @Author: 杨又菁
# @File:main_草稿1.py.py
# @Software:PyCharm
import csv
from scipy.spatial.distance import cdist
import cv2
import numpy as np

img = cv2.imread('3.png')


# def split_contour_into_segments(contour, epsilon):
#     # 进行轮廓近似
#     approx = cv2.approxPolyDP(contour, epsilon, closed=True)
#
#     print("contours:++++++++++++++",approx)
#     print("contour[0][0][0]",type([approx][0][0][0]))
#     # 绘制近似的轮廓
#     cv2.drawContours(img, [approx], -1, (0, 255, 0), 2)
#
#     return approx


# def get_boundary_B():
#     # 读取图像
#
#     # 将图像转换为灰度图
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 转换为二值图像
#     ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)  # 100可改
#     # 提取轮廓
#     contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     return contours


# def read_input_buildings():


if __name__ == '__main__':
    B = get_boundary_B()
    if B:
        contour_of_interest = B[0]
    # 进行轮廓分割
    epsilon = 0.02 * cv2.arcLength(contour_of_interest, True)  # 调整epsilon值以控制近似精度
    approx = split_contour_into_segments(contour_of_interest, epsilon)
    # print(type(approx))
    S = []
    for i in range(len(approx) - 1):
        line_segment = []
        line_segment.append(approx[i][0])
        line_segment.append(approx[i + 1][0])
        # print(type(line_segment))
        S.append(np.array(line_segment))
        # 绘制近似的轮廓
    line_segment = []
    line_segment.append(approx[len(approx) - 1][0])
    line_segment.append(approx[0][0])
    S.append(np.array(line_segment))
    print("S:========",S)
    print(type(S))
    S=np.array(S)
    print(type(S[0]))

    # print("approx:",approx)
    # print(approx[0])
    # print(approx[0][0])
    # print(S[0])

    # P的集合
    P = []
    # 打开CSV文件
    with open('input_building.csv', 'r') as csv_file:
        # 创建CSV读取器对象
        csv_reader = csv.reader(csv_file)

        # 遍历CSV文件的每一行
        for row in csv_reader:
            # # 每一行都作为一个列表返回
            # i=len(row)
            # G_segment=[i]
            # for j in range(i):
            #     G_segment[j]=row[0][0]
            print(row)
            temp = row[1].split(',')
            # print(temp)
            # print(temp[0])
            # print(temp[1])
            # print(len(temp))
            # print(row[0])
            P_segment = []
            i = 0
            print("temp_len:",len(temp))
            while (i < len(temp) - 1):
                point = []
                point.append(round(float(temp[i])))
                point.append(round(float(temp[i + 1])))
                P_segment.append(point)
                i = i + 2
                # print("type(P_segment):",type(P_segment))
            print("[np.array(P_segment)]:_______________",[np.array(P_segment)])
            cv2.drawContours(img, [np.array(P_segment)], -1, (0, 255, 255), 1)
            P.append(np.array(P_segment))

            print(len(P))

            #把P画出来看对不对
    P=np.array(P)

    # print("P:============",P)
    # cv2.drawContours(img, P, -1, (122, 122, 122), 1)

    # 计算Hausdorff距离
    print(type(P))
    print(type(S[0][0]))
    # 计算P中的点到S中的所有点的距离
    distances_P_to_S = cdist(P[0], S[0], 'euclidean')  # 这里使用欧几里得距离

    # 计算S中的点到P中的所有点的距离
    distances_S_to_P = cdist(S[0], P[0], 'euclidean')

    # 计算Hausdorff距离
    hausdorff_distance = max(distances_P_to_S.max(axis=1).max(), distances_S_to_P.max(axis=1).max())

    print("Hausdorff距离:", hausdorff_distance)





    cv2.imshow('Segmented Contour', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
