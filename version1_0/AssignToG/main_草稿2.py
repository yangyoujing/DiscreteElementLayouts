# -*- coding: utf-8 -*- 
# @Time : 2023/9/20 17:38 
# @Author : zzd 
# @File : main_草稿2.py
# @desc:
import csv
from scipy.spatial.distance import cdist
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff

img = cv2.imread('3.png')
# 打开csv文件
with open('input_building.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)


    # 把csv中的一行轮廓提取出来
    def csv_row_to_pointlist(row):
        # print(row)
        temp = row[1].split(',')
        segment = []
        i = 0
        # print("temp_len:", len(temp))
        while (i < len(temp) - 1):
            point = []
            point.append(round(float(temp[i])))
            point.append(round(float(temp[i + 1])))
            segment.append(point)
            i = i + 2
            # print("type(P_segment):",type(P_segment))
        # print("[np.array(segment)]:_______________", [np.array(segment)])
        cv2.drawContours(img, [np.array(segment)], -1, (0, 255, 255), 1)
        return np.array(segment)


    #
    # def Hausdorff_distance(x, y):
    #     distances_P_to_S = cdist(x, y, 'euclidean')  # 这里使用欧几里得距离
    #
    #     # 计算S中的点到P中的所有点的距离
    #     # distances_S_to_P = cdist(y, x, 'euclidean')
    #
    #     # 计算Hausdorff距离
    #     # hausdorff_distance = max(distances_P_to_S.max(axis=1).max(), distances_S_to_P.max(axis=1).max())
    #     hausdorff_distance = distances_P_to_S.max(axis=1).max()
    #     print("Hausdorff距离:", hausdorff_distance)
    #     return hausdorff_distance

    # 计算两个坐标之间的插值坐标（对轮廓进行细化）
    def insert_point(point1, point2, num):
        x1, y1 = point1
        x2, y2 = point2
        # 计算线段的长度
        # length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        # 使用插值生成中间点坐标
        intermediate_points = []
        for t in np.linspace(0, 1, num):
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            intermediate_points.append([round(x), round(y)])

        return intermediate_points


    if __name__ == '__main__':
        segmenter_contour = np.ones((1000, 1200, 3), dtype=np.uint8) * 0

        # 提取B，最外面的边界
        B = csv_row_to_pointlist(next(csv_reader))
        # print("B:]]]]]]]]]]]]]]]]]]]]]",B)
        cv2.drawContours(segmenter_contour, [np.array(B)], -1, (0, 255, 255), 1)

        # 进行轮廓分割,把B分为一小段的S

        S_temp = []
        for i in range(len(B) - 1):
            Si = []
            line_head = []
            line_head.append(B[i][0])
            line_head.append(B[i][1])
            # print(type(line_segment))
            Si.append(np.array(line_head))
            line_tail = []
            line_tail.append(B[i + 1][0])
            line_tail.append(B[i + 1][1])
            Si.append(np.array(line_tail))
            S_temp.append(np.array(Si))
            # 绘制近似的轮廓
        Si = []
        line_head = []
        line_head.append(B[i][0])
        line_head.append(B[i][1])
        # print(type(line_segment))
        Si.append(np.array(line_head))
        line_tail = []
        line_tail.append(B[0][0])
        line_tail.append(B[0][1])
        Si.append(np.array(line_tail))
        S_temp.append(np.array(Si))
        # print("S:========", S)
        # print(type(S))
        S_temp = np.array(S_temp)
        # print(type(S[0]))

        # 对S进行插值

        # 指定插值点的数量
        num_points = 100  # 例如，生成10个中间点

        S = np.empty((len(S_temp), num_points, 2), dtype=int)
        for i in range(len(S_temp)):

            # 创建一个5*num_points*2的S数组

            # print(S)
            for j in range(num_points):
                # print("S_temp[i]",S_temp[i])
                # print("intermediate_points[j]", intermediate_points[j])
                # print("len(S[i])", len(S[i]))
                S[i][j] = insert_point(S_temp[i][0], S_temp[i][1], num_points)[j]
        #         print("S[",i,"][",j,"]:",S[i][j] )
        print("S:===============", S)
        print("end")

        # P的集合
        # 对P进行插值
        P_temp = []
        for row in csv_reader:
            P_temp.append(csv_row_to_pointlist(row))

        # 对P进行插值
        P_insert_points = 20

        P = np.empty((len(P_temp), P_insert_points * len(P_temp), 2), dtype=int)
        for i in range(len(P_temp)):
            print("len(P_temp):",len(P_temp))
            for j in range(P_insert_points):
                P[i][j] = insert_point(P_temp[i][0],P_temp[i][1],P_insert_points)[j]
        #         print("S[",i,"][",j,"]:",S[i][j] )
        # print("S:===============",S)
        # print("end")

        print("P:==================", P)
        cv2.drawContours(segmenter_contour, np.array(P), -1, (0, 255, 255), 1)

        # 计算Hausdorff距离

        # print(type(P))
        # print(type(S[0][0]))
        # 计算P中的点到S中的所有点的距离
        # print("P[0]:",P[0])
        # print("P[0]:", type(P[0]))
        # print("S[0]:", S[0])
        # print("S[0]:", type(S[0]))
        # print(S)

        # P里面元素到边界段S的映射，P_to_S中，数组内的第一个元素表示类型：fixed,repeatable
        P_to_S = []
        for i in range(len(P)):
            print("i:~~~~~~~~~~~~~~~~~", i)
            segment = ["type"]
            count = 0
            for j in range(len(S)):
                print("j:~~~~~~~~~~~~~~~~~", j)
                print("P[i]：", P[i])
                print("S[i]：", S[j])
                hausdorff_distance = directed_hausdorff(P[i], S[j])[0]
                if hausdorff_distance <= 50:
                    count = count + 1
                    print(hausdorff_distance)
                    segment.append(j)
                else:
                    print(hausdorff_distance)

            if len(segment) > 1:
                P_to_S.append(segment)
            if count == 1:
                segment[0] = "repeatable"
            else:
                segment[0] = "fixed"

        print("P_to_S:", P_to_S)

        cv2.imshow('Segmented Contour', segmenter_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
