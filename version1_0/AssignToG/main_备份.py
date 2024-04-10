# -*- codeing=utf-8 -*-
# @Time:2023/9/23 11:45
# @Author: 杨又菁
# @File:main.py
# @Software:PyCharm

import csv
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

    def bezier_curve(p0, p1, p2, p3, t):
        return (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3


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
        line_head.append(B[-1][0])
        line_head.append(B[-1][1])
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
        # print("S_temp:!!!!!!!!!!!!!!!!!!!!!!!!!!!",S_temp)
        # print(type(S[0]))

        # 对S进行插值

        # 指定插值点的数量
        num_points = 100  # 例如，生成10个中间点

        t=np.linspace(0,1,num_points)

        #生成细化后的轮廓点
        S=[]
        for i in range(len(S_temp)):
            # print(S_temp)
            point_temp=[]
            # print("S_temp[i][0],S_temp[i][1]:",S_temp[i][0],S_temp[i][1])
            p0,p1=S_temp[i][0],S_temp[i][1]
            for t_val in t:
                # print("bezier_curve(p0,p0,p1,p1,t_val):",bezier_curve(p0,p0,p1,p1,t_val))
                point_temp.append(bezier_curve(p0,p0,p1,p1,t_val))
            S.append(point_temp)
            # print("S:",S)

        S=np.array(S)



        # print("S:===============", S)
        # print("end")

        # P的集合
        P=[]
        P_temp = []
        for row in csv_reader:
            P_temp.append(csv_row_to_pointlist(row))

        print("===================",P_temp)
        # 对P进行插值

        # 指定插值点的数量
        num_points = 100  # 例如，生成10个中间点

        t = np.linspace(0, 1, num_points)

        # 生成细化后的轮廓点
        for i in range(len(P_temp)):
            # print("P_temp",P_temp)
            point_temp = []
            # print("S_temp[i][0],S_temp[i][1]:",S_temp[i][0],S_temp[i][1])
            for j in range(len(P_temp[i])-1):
                p0, p1 = P_temp[i][j], P_temp[i][j+1]
                for t_val in t:
                    point_temp.append(bezier_curve(p0, p0, p1, p1, t_val))

            p0, p1 = P_temp[i][-1], P_temp[i][0]
            for t_val in t:
                point_temp.append(bezier_curve(p0, p0, p1, p1, t_val))
            print("point_temp:",point_temp)
            # point_temp=np.array(point_temp)
            P.append(np.array(point_temp))

        print("P:", np.array(P,))
        cv2.drawContours(segmenter_contour, np.array(P_temp), -1, (0, 255, 255), 1)

        # 计算Hausdorff距离

        # print(type(P))
        # print(type(S[0][0]))
        # 计算P中的点到S中的所有点的距离
        # print("P[0]:",P[0])
        # print("P[0]:", type(P[0]))
        # print("S[0]:", S[0])
        # print("S[0]:", type(S[0]))
        # print(S)

        # P里面元素到边界段S的映射，P_to_S中：
        P_to_S = []

        # P中元素的类型：
        # fixed=2,repeatable=1,empty=0
        P_type=[]

        for i in range(len(P)):
            print("i:~~~~~~~~~~~~~~~~~", i)
            # print("P[i]：", P[i])
            segment = []
            count = 0
            for j in range(len(S)):
                print("j:~~~~~~~~~~~~~~~~~", j)

                # print("S[j]：", S[j])
                hausdorff_distance = directed_hausdorff(P[i], S[j])[0]
                if hausdorff_distance <= 65:
                    count = count + 1
                    print(hausdorff_distance)
                    segment.append(j)
                else:
                    print(hausdorff_distance)

            if len(segment) > 0:
                P_to_S.append(segment)
            if count == 1:
                P_type.append(1)
            else:
                P_type.append(2)


        print("P_to_S:", P_to_S)
        print(P_type)

        # 把元素的轮廓放进Si对应的G中
        G=[[] for _ in range(len(S_temp))]
        for i in range(len(P_temp)):
            for j in range(len(P_to_S[i])):
                print("int(P_to_S[i][j]):",int(P_to_S[i][j]))
                print("P_temp[i]:",P_temp[i])
                G[int(P_to_S[i][j])].append(P_temp[i])
                print("G:", G,i)

        print("G:",G)


        cv2.imshow('Segmented Contour', segmenter_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # sort_data = []
        # for i in range(len(G)):
        #     for j in range(len(G[i])):
        #         # print("cad_points_list",cad_points_list[i])
        #         sort_temp = [len(G[i][j])]
        #         flatted = np.array(G[i][j]).flatten()
        #         temp_str = ','.join(str(x) for x in flatted)
        #         # a2 = np.array([float(x) for x in str.split(',')])
        #         sort_temp.append(temp_str)
        #         sort_data.append(sort_temp)
        #
        #
        #
        # #写入csv文件
        # with open('group_G.csv', 'w') as file:
        #     csvwriter = csv.writer(file, lineterminator='\n')
        #     csvwriter.writerows(sort_data)



