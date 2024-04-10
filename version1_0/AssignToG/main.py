# -*- codeing=utf-8 -*-
# @Time:2023/9/23 11:45
# @Author: 杨又菁
# @File:main.py
# @Software:PyCharm
import copy
import csv
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff
from Boundary import *
from Elements import *

img = cv2.imread('3.png')
# 打开csv文件
with open('../CSV/input_building_2.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)


    # 把csv中的一行轮廓提取出来
    def csv_row_to_pointlist(row):
        print(row)
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
        return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


    if __name__ == '__main__':
        segmenter_contour = np.ones((1000, 1200, 3), dtype=np.uint8) * 0

        # 提取B，最外面的边界
        B = BoundaryOut(csv_row_to_pointlist(next(csv_reader)))

        # print("B:]]]]]]]]]]]]]]]]]]]]]",B)
        cv2.drawContours(segmenter_contour, [np.array(B.get_contour())], -1, (0, 255, 255), 1)
        # print("[np.array(B.get_contour())]",[np.array(B.get_contour())])

        # 进行轮廓分割,把B分为一小段的S
        S_temp = np.array(B.split_to_segments())
        S = []
        for i in range(B.get_len()):
            S.append(BoundarySegment(i, S_temp[i]))
        org_S=copy.deepcopy(S)
        # 把segments打印出来
        # for i in range(len(segments)):
        #     print("segments[i].contour", segments[i].contour)

        # 对S进行插值

        # 指定插值点的数量
        num_points = 100  # 例如，生成10个中间点

        t = np.linspace(0, 1, num_points)

        # 生成细化后的轮廓点
        for i in range(len(S)):
            point_temp = []
            p0, p1 = S[i].contour[0], S[i].contour[1]
            for t_val in t:
                point_temp.append(bezier_curve(p0, p0, p1, p1, t_val))
            S[i].contour = np.array(point_temp)
            print("segments[i].contour", S[i].contour)

        # P的集合
        P = []
        i = 0
        for row in csv_reader:
            P.append(Element(i, csv_row_to_pointlist(row)))
            i += 1
        for i in range(len(P)):
            cv2.drawContours(segmenter_contour, [np.array(P[i].get_contour())], -1, (0, 255, 255), 1)
            print("P[i].get_contour()", P[i].get_contour())

        original_P_contour=[]
        for element in P:
            original_P_contour.append(element.contour)
        # 对P进行插值

        # 指定插值点的数量
        num_points = 100  # 例如，生成10个中间点

        t = np.linspace(0, 1, num_points)

        # 生成细化后的轮廓点
        for i in range(len(P)):
            # print("P_temp",P_temp)
            point_temp = []
            # print("S_temp[i][0],S_temp[i][1]:",S_temp[i][0],S_temp[i][1])
            for j in range(len(P[i].get_contour()) - 1):
                p0, p1 = P[i].get_contour()[j], P[i].get_contour()[j + 1]
                for t_val in t:
                    point_temp.append(bezier_curve(p0, p0, p1, p1, t_val))

            p0, p1 = P[i].get_contour()[-1], P[i].get_contour()[0]
            for t_val in t:
                point_temp.append(bezier_curve(p0, p0, p1, p1, t_val))
            print("point_temp:", point_temp)
            P[i].contour = np.array(point_temp)

        # 根据hausdorff距离来将元素分到对应的边界

        for i in range(len(P)):
            print("i:~~~~~~~~~~~~~~~~~", i)
            # print("P[i]：", P[i])
            for j in range(len(S)):
                print("j:~~~~~~~~~~~~~~~~~", j)
                # print("S[j]：", S[j])
                hausdorff_distance = directed_hausdorff(P[i].contour, S[j].contour)[0]
                if hausdorff_distance <= 90:  #70可以修改
                    P[i].belong_S(S[j].num)
                    org_S[j].enroll_P(P[i].num)
                    print(hausdorff_distance)

        # 处理P中元素的类型：
        # fixed=2,repeatable=1,empty=0
        for i in range(len(P)):
            P[i].judge_type()
            # 添加元素的前后邻居
            # if len(P[(i - 1) % len(P)].boundary.intersection(P[i].boundary)) > 0:
            #     P[i].neighbor_left = P[(i - 1) % len(P)].num
            # if len(P[(i + 1) % len(P)].boundary.intersection(P[i].boundary)) > 0:
            #     P[i].neighbor_right = P[(i + 1) % len(P)].num
            if len([value for value in P[(i - 1) % len(P)].boundary if value in P[i].boundary]) > 0:
                P[i].neighbor_left = P[(i - 1) % len(P)].num
            if len([value for value in P[(i + 1) % len(P)].boundary if value in P[i].boundary]) > 0:
                P[i].neighbor_right = P[(i + 1) % len(P)].num

        #
        for i in range(len(P)):
            print("元素", P[i].num, "的类型是：", P[i].ele_type, "  所属的边界有：", P[i].boundary)

        for i in range(len(S)):
            print("边界", org_S[i].num, "中有哪些元素：", org_S[i].elements)

        cv2.imshow('Segmented Contour', segmenter_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # sort_data = []
        # for i in range(len(P)):
        #     for j in range(len(G[i])):
        #         # print("cad_points_list",cad_points_list[i])
        #         sort_temp = [len(G[i][j])]
        #         flatted = np.array(G[i][j]).flatten()
        #         temp_str = ','.join(str(x) for x in flatted)
        #         # a2 = np.array([float(x) for x in str.split(',')])
        #         sort_temp.append(temp_str)
        #         sort_data.append(sort_temp)

        # 把元素信息写入segments.csv
        sort_data = []
        for i in range(len(P)):
            sort_temp = []
            sort_temp.append(P[i].num)  # 编号
            sort_temp.append(P[i].ele_type)  # 类型
            # 所属边界
            flatted = np.array(P[i].boundary).flatten()
            temp_str = ','.join(str(x) for x in flatted)
            sort_temp.append(temp_str)

            sort_temp.append(P[i].neighbor_left)  # 左邻居
            sort_temp.append(P[i].neighbor_right)  # 右邻居
            # 坐标
            flatted = np.array(original_P_contour[i]).flatten()
            temp_str = ','.join(str(x) for x in flatted)
            sort_temp.append(temp_str)
            sort_data.append(sort_temp)

        # 写入csv文件
        with open('../CSV/elements_2.csv', 'w') as file:
            csvwriter = csv.writer(file, lineterminator='\n')
            csvwriter.writerows(sort_data)

        # 把最外轮廓信息写入boundaryout.csv
        sort_data = []
        sort_temp = []
        # 坐标
        flatted = np.array(B.contour).flatten()
        temp_str = ','.join(str(x) for x in flatted)
        sort_temp.append(temp_str)
        sort_data.append(sort_temp)

        # 写入csv文件
        with open('../CSV/boundaryout_2.csv', 'w') as file:
            csvwriter = csv.writer(file, lineterminator='\n')
            csvwriter.writerows(sort_data)

        # 把分段边界S信息写入boundarysegment.csv
        sort_data = []
        for i in range(len(org_S)):
            sort_temp = []
            sort_temp.append(S[i].num)  # 编号
            # 包含哪些元素
            flatted = np.array(org_S[i].elements).flatten()
            temp_str = ','.join(str(x) for x in flatted)
            sort_temp.append(temp_str)
            # 边界方向
            flatted = np.array(org_S[i].direction).flatten()
            temp_str = ','.join(str(x) for x in flatted)
            sort_temp.append(temp_str)
            # 坐标
            flatted = np.array(org_S[i].contour).flatten()
            temp_str = ','.join(str(x) for x in flatted)
            sort_temp.append(temp_str)
            sort_data.append(sort_temp)

        # 写入csv文件
        with open('../CSV/boundarysegment_2.csv', 'w') as file:
            csvwriter = csv.writer(file, lineterminator='\n')
            csvwriter.writerows(sort_data)
