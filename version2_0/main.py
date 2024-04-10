# -*- codeing=utf-8 -*-
# @Time:2024/3/18 5:30
# @Author: 杨又菁
# @File:main.py
# @Software:PyCharm


import json
import matplotlib.pyplot as plt
from Matcher import *
from ShapeMatcher import *
from scipy.spatial.distance import directed_hausdorff
from Boundary import *
from BuildingMatcher import *
from SegmentsMatcher import *
import numpy as np
from Element import *
import cv2

# 目标小区的轮廓坐标
# target_neighborhood = [[123.53294144757092, 384.160605462268], [142.0923202354461, 331.17746670590714],
#                        [142.08544256351888, 314.60469290288165], [144.8718948122114, 269.8047721697949],
#                        [136.4233750347048, 250.00864866701886], [129.32657785713673, 239.37245193729177],
#                        [128.82800431735814, 226.82865376072004], [120.75962947495282, 211.85565539589152],
#                        [87.9886629730463, 178.7750335629098], [0.0, 129.8469779463485], [55.68909960612655, 0.0],
#                        [322.86909129843116, 102.96619065711275], [193.9663555137813, 416.19540413701907],
#                        [123.53294144757092, 384.160605462268]]

# 从 JSON 文件中加载数据
with open('ReCo_json.json', 'r') as file:
    neighborhood_data = json.load(file)

Elements = []
i = 0
for neighborhood in neighborhood_data:
    _id = neighborhood['_id']
    if _id == "61f4d725ea20fd79e35c328d":
        target_neighborhood = neighborhood['boundary']
        buildings = neighborhood['buildings']
        for building in buildings:
            temp = Element(i, building["coords"])
            temp.floor = building["floor"]
            Elements.append(temp)
            i += 1

# for i in Elements:
#     print(i.num,":floor:",i.floor,"coords:",i.contour)


# 根据形状上下文进行的形状匹配
neighborhood_matcher = ShapeMatch(target_neighborhood)
# most_similar_neighborhood = neighborhood_matcher.find_most_similar(neighborhood_data,_id)
# print("形状最相近的小区id:", most_similar_neighborhood)
# # 根据面积重叠进行的形状匹配
# nei=NeighborhoodMatcher(target_neighborhood)
# most_similar_neighborhood1=nei.find_most_similar(neighborhood_data,_id)
# print("形状最相近的小区id1:", most_similar_neighborhood1)

# 运行时间太长，假设已经得出
most_similar_neighborhood = "61fc13cfaecb8d42542212ce"

# def rotate_points(boundary, angle):
#     # 将列表转换为 NumPy 数组
#     points = np.array(boundary)
#
#     # 计算旋转矩阵
#     rotation_matrix = cv2.getRotationMatrix2D(center=(0, 0), angle=angle, scale=1.0)
#
#     # 将边界点转换为齐次坐标表示
#     boundary_homogeneous = np.ones((len(boundary), 3))
#     boundary_homogeneous[:, :2] = boundary
#
#     # 应用旋转矩阵
#     rotated_boundary = np.dot(rotation_matrix, boundary_homogeneous.T).T
#
#     # 将结果转换回笛卡尔坐标表示
#     rotated_boundary_cartesian = rotated_boundary[:, :2]
#
#     # 找到旋转后的边界点中的最小 x 和 y 坐标
#     min_x = np.min(rotated_boundary_cartesian[:, 0])
#     min_y = np.min(rotated_boundary_cartesian[:, 1])
#
#     # 调整坐标，使得最小的坐标都为 0
#     rotated_boundary_cartesian[:, 0] -= min_x
#     rotated_boundary_cartesian[:, 1] -= min_y
#
#     return rotated_boundary_cartesian
#
#
# def flip_points(boundary):
#     # 将列表转换为 NumPy 数组
#     points = np.array(boundary)
#
#     # 计算水平翻转矩阵
#     flip_matrix = np.array([[-1, 0, 0],
#                             [0, 1, 0],
#                             [0, 0, 1]])
#
#     # 将边界点转换为齐次坐标表示
#     boundary_homogeneous = np.ones((len(boundary), 3))
#     boundary_homogeneous[:, :2] = boundary
#
#     # 应用翻转矩阵
#     flipped_boundary = np.dot(flip_matrix, boundary_homogeneous.T).T
#
#     # 将结果转换回笛卡尔坐标表示
#     flipped_boundary_cartesian = flipped_boundary[:, :2]
#
#     # 找到翻转后的边界点中的最小 x 和 y 坐标
#     min_x = np.min(flipped_boundary_cartesian[:, 0])
#     min_y = np.min(flipped_boundary_cartesian[:, 1])
#
#     # 调整坐标，使得最小的坐标都为 0
#     flipped_boundary_cartesian[:, 0] -= min_x
#     flipped_boundary_cartesian[:, 1] -= min_y
#
#     return flipped_boundary_cartesian

# 对目标小区的轮廓进行分段，分成边界
t_boundaryOut = BoundaryOut(target_neighborhood)
t_boundarySegments = t_boundaryOut.split_to_segments()
t_segments = []  # 记录所有的边界段，里面存了边界上的坐标
for i in range(len(t_boundarySegments)):
    temp_segment = BoundarySegment(i, t_boundarySegments[i])
    t_segments.append(temp_segment)

# # 打印出边界段的信息，看是否正确
# for i in t_segments:
#     print("t_Segment", i.num, ":", i.contour)

# 根据小区id，从数据集中获取对应的小区轮廓坐标
for neighborhood in neighborhood_data:
    if neighborhood['_id'] == most_similar_neighborhood:
        print("目标小区id:", neighborhood['_id'], "目标小区轮廓坐标:", neighborhood['boundary'])
        o_boundaryOut = BoundaryOut(neighborhood['boundary'])
        # 画出仿射变换
        # neighborhood_matcher.context_shape_match(neighborhood['boundary'], visual_num=50, N=100, angle=5,
        #                                         distance=[0, 0.125, 0.25, 0.5, 1.0, 2.0])

        # 接下来尝试旋转使距离函数最小
        # 使neighborhood['boundary']进行旋转，并生成旋转后的边界
        # d=neighborhood_matcher.distance_function(neighborhood['boundary'])
        # print("初始距离函数值:", d)
        # rotated_boundary_cartesian = neighborhood['boundary']
        # for angle in range(0, 360, 10):
        #     rotated_boundary = rotate_points(neighborhood['boundary'], angle)
        #     temp=neighborhood_matcher.distance_function(rotated_boundary)
        #
        #     print("angle:", angle, "distance:", temp)
        #     if temp<d:
        #         print("旋转角度:", angle, "新的距离函数值:", temp, "小于初始值:", d)
        #         d=temp
        #         rotated_boundary_cartesian = rotated_boundary
        # # 水平翻转
        # flipped_boundary = flip_points(rotated_boundary_cartesian)
        # print("水平翻转一次")
        # # 再水平旋转
        # for angle in range(0, 360, 10):
        #     rotated_boundary = rotate_points(flipped_boundary, angle)
        #     temp = neighborhood_matcher.distance_function(rotated_boundary)
        #
        #     print("angle:", angle, "distance:", temp)
        #     if temp < d:
        #         print("旋转角度:", angle, "新的距离函数值:", temp, "小于初始值:", d)
        #         d = temp
        #         rotated_boundary_cartesian = rotated_boundary
        #     # neighborhood_matcher.context_shape_match(rotated_boundary, visual_num=50, N=100, angle=5,
        #     #                                         distance=[0, 0.125, 0.25, 0.5, 1.0, 2.0])
        #
        # # 画出仿射变换
        # neighborhood_matcher.context_shape_match(rotated_boundary_cartesian, visual_num=10, N=100, angle=5,
        #                                          distance=[0, 0.125, 0.25, 0.5, 1.0, 2.0])
        # 尝试结果：效果并没有更好

        o_boundarySegments = o_boundaryOut.split_to_segments()
        o_segments = []  # 记录所有的边界段，里面存了边界上的坐标
        for i in range(len(o_boundarySegments)):
            temp_segment = BoundarySegment(i, o_boundarySegments[i])
            o_segments.append(temp_segment)

        # # 打印出边界段的信息，看是否正确
        # for i in o_segments:
        #     print("o_Segment", i.num, ":", i.contour)

        # 比较目标边界段与原边界段的个数，哪个少了就增加哪个

        # 将o_segments与t_segments进行匹配
        # segmentsMatcher=SegmentsMatch(o_segments, t_segments)
        segmentsMatcher = SegmentsMatch(t_segments, o_segments)
        for i in range(len(t_segments)):
            print(t_segments[i].num, ":", t_segments[i].contour)
        match, o_segments, t_segments = segmentsMatcher.Match()

        _o_segments=copy.deepcopy(o_segments)

        print("match:", match)
        for i in range(len(o_segments)):
            print(i, ":", o_segments[i].contour)
        for i in range(len(t_segments)):
            print(i, ":", t_segments[i].contour)

        # 画出增加线段后的图
        # # 绘图
        # fig, ax = plt.subplots()

        # # 绘制线段
        # for segment in o_segments:
        #     contour = segment.contour
        #     x_values = [point[0] for point in contour]
        #     y_values = [point[1] for point in contour]
        #     ax.plot(x_values, y_values, 'b-')  # 绘制线段
        #     ax.plot(x_values[0], y_values[0], 'ro')  # 红色点标出起点
        #     ax.plot(x_values[-1], y_values[-1], 'ro')  # 红色点标出终点
        #
        # plt.gca().set_aspect('equal', adjustable='box')  # 设置x和y轴的比例相等
        # plt.show()

        segmenter_contour = np.ones((1000, 1200, 3), dtype=np.uint8) * 0
        cv2.drawContours(segmenter_contour, [np.round(target_neighborhood).astype(int)], -1, (0, 255, 255), 1)
        # 把Elements的信息匹配到对应的o_segments中
        for i in range(len(Elements)):
            # print("Elements[i].contour:",Elements[i].contour)
            cv2.drawContours(segmenter_contour, [np.round(Elements[i].contour).astype(int)], -1, (0, 255, 255), 1)

            # 获取轮廓的中心点坐标
            M = cv2.moments(np.round(Elements[i].contour).astype(int))
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            else:
                cX, cY = 0, 0

            # 在图像上绘制 Element[i].num 对应的数字
            cv2.putText(segmenter_contour, str(Elements[i].num), (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

        # 对Element进行插值

        # 指定插值点的数量
        num_points = 1  # 例如，生成10个中间点

        # 生成细化后的轮廓点
        for i in range(len(Elements)):
            Elements[i].contour=Elements[i].refine_curve(num_points)

        # 根据hausdorff距离来将元素分到对应的边界

        for i in range(len(Elements)):
            # print("i:~~~~~~~~~~~~~~~~~", i)
            # print("P[i]：", P[i])
            for j in range(len(o_segments)):
                # print("j:~~~~~~~~~~~~~~~~~", j)
                # print("S[j]：", S[j])
                hausdorff_distance = directed_hausdorff(Elements[i].contour, o_segments[j].contour)[0]
                if hausdorff_distance <= 60:  # 70可以修改
                    Elements[i].belong_Seg(o_segments[j].num)
                    _o_segments[j].enroll_P(Elements[i].num)
                    print(hausdorff_distance)

        # 处理P中元素的类型：
        # fixed=2,repeatable=1,empty=0
        for i in range(len(Elements)):
            Elements[i].judge_ele_type()
            # 添加元素的前后邻居
            # if len(P[(i - 1) % len(P)].boundary.intersection(P[i].boundary)) > 0:
            #     P[i].neighbor_left = P[(i - 1) % len(P)].num
            # if len(P[(i + 1) % len(P)].boundary.intersection(P[i].boundary)) > 0:
            #     P[i].neighbor_right = P[(i + 1) % len(P)].num
            # if len([value for value in Elements[(i - 1) % len(Elements)].contour if value in Elements[i].contour]) > 0:
            #     Elements[i].neighbor_left = Elements[(i - 1) % len(Elements)].num
            # if len([value for value in Elements[(i + 1) % len(Elements)].contour if value in Elements[i].contour]) > 0:
            #     Elements[i].neighbor_right = Elements[(i + 1) % len(Elements)].num

        #
        for i in range(len(Elements)):
            print("元素", Elements[i].num, "的类型是：", Elements[i].ele_type, "  所属的边界有：", Elements[i].boundarySegment)

        for i in range(len(o_segments)):
            print("边界", _o_segments[i].num, "中有哪些元素：", _o_segments[i].elements)

        cv2.imshow('Segmented Contour', segmenter_contour)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
