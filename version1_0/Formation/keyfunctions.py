# -*- codeing=utf-8 -*-
# @Time:2023/10/13 14:19
# @Author: 杨又菁
# @File:keyfunctions.py
# @Software:PyCharm
import sys
sys.path.append('../AssignToG')
import math
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Point
from Boundary import *
from Elements import *
from shapely.geometry import LineString, Polygon
from math import cos, sin
from shapely.validation import make_valid
import numpy as np
import copy
from itertools import combinations


def Area(Element1, Element2):
    poly1 = Polygon(np.array(Element1.contour)).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(np.array(Element2.contour)).convex_hull
    # print("第一个元素是：", Element1.num)
    # print("第二个元素是：", Element2.num)
    # 计算p1的面积
    # area_p1 = cv2.contourArea(np.array(Element1.contour))
    area_p1 = poly1.area
    # print("p1的面积为：", area_p1)

    # # 计算轮廓的交集
    # intersection = cv2.bitwise_and(np.array(Element1.contour), np.array(Element2.contour))
    # # 计算交集的面积
    # intersection_area = cv2.contourArea(intersection)

    if not poly1.intersects(poly2):
        intersection_area = 0  # 如果两四边形不相交
    else:
        intersection_area = poly1.intersection(poly2).area  # 相交面积

    # print("重合部分的面积是：", intersection_area)
    # intersection_area =cv2.countNonZero(intersection)

    result = area_p1 - intersection_area
    # print("p1的面积减去重合的面积等于：", result)

    return result


# 相邻的更倾向不同的
def tendto_different(Element1, Element2):
    a = 0.05
    poly1 = Polygon(np.array(Element1.contour)).convex_hull  # Polygon：多边形对象
    poly2 = Polygon(np.array(Element2.contour)).convex_hull
    # 如果相邻两个元素都是空白元素
    if Element1.ele_type == "empty" and Element2.ele_type == "empty":
        print("两个都是空元素")
        return a
    elif poly1.equals(poly2):
        print("放置的两个元素相同：", 1 - a)
        return 1 - a
    elif not poly1.equals(poly2):
        print("两个元素不相同")
        return 0

    print("啥也不是？？？？？")


# 元素P投影到边界S上的长度
def projector_distance(Element, BoundarySegment):
    polygon = Polygon(Element.contour)
    line = LineString(BoundarySegment.contour)

    # 初始化投影点列表
    projection_points = []

    # 遍历多边形的所有顶点，计算它们到线段的投影点
    for vertex in polygon.exterior.coords:
        point = Point(vertex)
        projection = line.interpolate(line.project(point))
        projection_points.append(projection)

    # 计算投影点中的最远两个点之间的距离
    if len(projection_points) < 2:
        return 0  # 如果没有足够的投影点，距离为0
    else:
        max_distance = 0
        for pair in combinations(projection_points, 2):
            distance = pair[0].distance(pair[1])
            if distance > max_distance:
                max_distance = distance
        return max_distance


# 计算两点之间的长度
def point_distance(point1, point2):
    return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)


# 计算当前离散位置
def discrete_position(Element, BoundarySegment):
    quadrilateral = Polygon(Element.contour)
    line = LineString(BoundarySegment.contour)

    # 获取直线的起点和终点
    # start_point, end_point = line.boundary

    # 初始化四边形的位置（中心点）
    current_position = quadrilateral.centroid

    # # 计算直线的方向向量
    # direction_vector = [end_point.x - start_point.x, end_point.y - start_point.y]
    #
    # # 计算直线的长度
    # line_length = line.length
    #
    # # 计算直线的当前离散位置

    # # 中心点投影到边界上的坐标
    # _, projection_point = nearest_points(line, current_position)
    # position = (projection_point[0] - start_point.x) * line_length / direction_vector[0]

    return [current_position.x, current_position.y]


# 将Element从当前离散位置沿着边界方向移动一个步长
def move_interval(Element, BoundarySegment_contour):
    # print("移动之前的坐标：",Element.contour)
    # 设置步长
    step_size = 1

    line = LineString(BoundarySegment_contour)
    # 获取直线的起点和终点
    start_point, end_point = BoundarySegment_contour[0], BoundarySegment_contour[1]
    # 计算直线的长度
    line_length = line.length

    # 计算直线的方向向量,并标准化
    t = math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2)
    direction_vector = [(end_point[0] - start_point[0]) / t, (end_point[1] - start_point[1]) / t]
    # print(direction_vector[0],direction_vector[1])

    temp_contour = []
    for i in range(len(Element.contour)):
        next_x = Element.contour[i][0] + direction_vector[0] * step_size
        next_y = Element.contour[i][1] + direction_vector[1] * step_size
        # next_position = [round(next_x), round(next_y)]
        next_position = [next_x, next_y]
        # print("next_position:",next_position)

        # 检查是否四边形已经超出直线的终点
        # _, projection_point = nearest_points(line, Point(next_position))
        # if projection_point.x > end_point[0]:
        #     return

        temp_contour.append(next_position)
    # print("1111111111111111移动之后的坐标：",temp_contour)
    result_element = copy.deepcopy(Element)
    result_element.contour = temp_contour
    return result_element


# 判断两个Element或者边界是否相交
def is_intersect(Element1, Element2_or_BoundarySegment):
    quadrilateral1 = Polygon(Element1.contour)
    if isinstance(Element2_or_BoundarySegment,Element):
        quadrilateral2 = Polygon(Element2_or_BoundarySegment.contour)
        if quadrilateral1.intersection(quadrilateral2):
            # print("相交")
            return 1
        else:
            # print("不相交")
            return 0
    else:
        line_string = LineString(Element2_or_BoundarySegment.contour)
        if quadrilateral1.intersection(line_string):
            # print("相交")
            return 1
        else:
            # print("不相交")
            return 0


# 根据离散位置求Element坐标(暂定为中心坐标平行)
def position_to_contour(Element, BoundarySegment, position):
    contour_to_tuples = [tuple(sublist) for sublist in Element.contour]
    quadrilateral = Polygon(contour_to_tuples)
    line = LineString(BoundarySegment.contour)

    # 初始化四边形的位置（中心点）
    central_point = quadrilateral.centroid
    # print("初始化四边形的位置（中心点）:",central_point )
    moved_distance_x = position[0] - central_point.x
    moved_distance_y = position[1] - central_point.y
    # print("moved_distance_x:",moved_distance_x,"moved_distance_y:",moved_distance_y)
    temp_contour = []
    # print("len(Element.contour):",len(Element.contour))
    for i in range(len(Element.contour)):
        moved_x = Element.contour[i][0] + moved_distance_x
        moved_y = Element.contour[i][1] + moved_distance_y
        next_position = [round(moved_x), round(moved_y)]
        # print("next_position :",next_position )

        temp_contour.append(next_position)

    result_element = copy.deepcopy(Element)
    result_element.contour = temp_contour
    # print("移动到起始端点时element坐标Element.contour = temp_contour:",Element.contour)
    # if is_out_boundary(Element, BoundarySegment.contour):
    #     return
    # else:
    #     return Element

    return result_element


# 判断Element的轮廓是否超出了边界的范围
def is_out_boundary(Element, BoundarySegment_contour):
    # 去除重复的点
    temp = Element.contour[0]
    cleaned_Element = []
    i = 1
    cleaned_Element.append(temp)
    while i < len(Element.contour):
        if temp[0] == Element.contour[i][0] and temp[1] == Element.contour[i][1]:
            i += 1
        else:
            temp = Element.contour[i]
            cleaned_Element.append(temp)
            i += 1
    # print("cleaned_Element:",cleaned_Element)

    temp = BoundarySegment_contour[0]
    cleaned_BoundarySegment = []
    i = 1
    cleaned_BoundarySegment.append(temp)
    while i < len(BoundarySegment_contour):
        if temp[0] == BoundarySegment_contour[i][0] and temp[1] == BoundarySegment_contour[i][1]:
            i += 1
        else:
            temp = BoundarySegment_contour[i]
            cleaned_BoundarySegment.append(temp)
            i += 1
    # print("cleaned_BoundarySegment:",cleaned_BoundarySegment)

    # print("Element.contour:",Element.contour)
    contour_to_tuples = [tuple(sublist) for sublist in cleaned_Element]
    # print("contour_to_tuples:",contour_to_tuples)
    quadrilateral = Polygon(contour_to_tuples)
    line = LineString(cleaned_BoundarySegment)

    # 检查多边形是否有效
    if not quadrilateral.is_valid:
        print("多边形无效////////////////////")
        # 尝试修复多边形
        quadrilateral = make_valid(quadrilateral)

    # 初始化一个标志来表示是否有投影点在线段外面
    outside_flag = False

    # 判断是否相交
    if quadrilateral.intersection(line):
        outside_flag = True

    # # 获取Element的最小外接矩形
    # # 计算多边形的边界框
    # polygon_bbox = quadrilateral.bounds
    # print("polygon_bbox:",polygon_bbox)
    # polygon_test=Polygon(polygon_bbox)
    # print("成功")
    # 获取线段的两个端点
    start_point = Point(BoundarySegment_contour[0])
    end_point = Point(BoundarySegment_contour[1])

    # 计算线段的长度
    line_length = point_distance(start_point, end_point)

    # 计算矩形的四个顶点在线段上的投影点
    for vertex in quadrilateral.exterior.coords:
        vertex_point = Point(vertex)  # 将点坐标转换为 Shapely Point 对象
        projection = line.interpolate(line.project(vertex_point))
        distance_to_start = projection.distance(start_point)
        distance_to_end = projection.distance(end_point)

        if distance_to_start < 0 or distance_to_end < 0 or distance_to_start > line_length or distance_to_end > line_length:
            outside_flag = True
            break

    # 判断是否有投影点在线段外面
    if outside_flag:
        # print("矩形的投影点中有点在线段外面")
        return 1
    else:
        # print("矩形的投影点都在线段范围内")
        return 0

def is_out_block(Element, BoundarySegment_contour):
    polygon1=Polygon(Element.contour)
    polygon2 = Polygon(BoundarySegment_contour)
    if polygon1.within(polygon2):
        return False
    else:
        return True



# 已知一个Element坐标，计算下一个Element坐标与之不相交的初始位置
def next_not_intersect(Element1, Element2, BoundarySegment):
    # 先算出Element1的中心坐标
    position = discrete_position(Element1, BoundarySegment)
    # print("前一个元素的中心坐标：",position)
    # print("前一个元素的轮廓坐标为：",Element1.contour)
    # 将Element2移动到该中心坐标
    temp_Element = position_to_contour(Element2, BoundarySegment, position)
    # print("移动后的中心坐标：",discrete_position(temp_Element, BoundarySegment))
    # print("将Element2移动到该中心坐标:",temp_Element.contour)
    while is_intersect(Element1, temp_Element):
        temp_Element = move_interval(temp_Element, BoundarySegment.contour)
        # print("移动了一个步长")
        # print("将Element2移动到与1不相交的位置:", temp_Element.contour)
    # 把temp_Element的contour中的每个数字四舍五入
    for i in range(len(temp_Element.contour)):
        temp_Element.contour[i]=[round(temp_Element.contour[i][0]),round(temp_Element.contour[i][1])]

    if is_intersect(Element1, temp_Element):
        temp_Element = move_interval(temp_Element, BoundarySegment.contour)
        # print("移动了一个步长")
        # print("将Element2移动到与1不相交的位置:", temp_Element.contour)
    # 把temp_Element的contour中的每个数字四舍五入
        for i in range(len(temp_Element.contour)):
            temp_Element.contour[i]=[round(temp_Element.contour[i][0]),round(temp_Element.contour[i][1])]


    # 判断是否超出了线段的边界

    return temp_Element


# 已知元素原来的坐标和原来的边界，目标边界，将元素进行旋转
def rotate_Element(Element, BoundarySegment1, BoundarySegment2):
    # if Element.ele_type == "fixed":
    #     print("len(Element.ele_type)==fixed")
    #     return Element

    quadrilateral = Polygon(Element.contour)
    o_line = LineString(BoundarySegment1.contour)
    t_line = LineString(BoundarySegment2.contour)
    # print("BoundarySegment1.contour:",BoundarySegment1.contour)
    # print("旋转前的多边形坐标：", Element.contour)
    # 计算两条边界的夹角
    magnitude_A = math.sqrt(sum(a * a for a in BoundarySegment1.direction))
    magnitude_B = math.sqrt(sum(b * b for b in BoundarySegment2.direction))

    # 计算点积
    dot_product = sum(a * b for a, b in zip(BoundarySegment1.direction, BoundarySegment2.direction))

    # 计算夹角
    angle_difference = math.acos(dot_product / (magnitude_A * magnitude_B))
    # print("两条边界的夹角为：", math.degrees(angle_difference))

    # 获取多边形的质心
    centroid = quadrilateral.centroid
    # print("centroid :", centroid.x, centroid.y)
    # 计算质心投影到边界上的对应点
    # 计算点到线段起点的向量
    point_vector = [centroid.x - BoundarySegment1.contour[0][0], centroid.y - BoundarySegment1.contour[0][1]]

    # 计算点在线段上的投影点
    dot_product = point_vector[0] * BoundarySegment1.direction[0] + point_vector[1] * BoundarySegment1.direction[1]
    length_squared = BoundarySegment1.direction[0] ** 2 + BoundarySegment1.direction[1] ** 2
    projection_scalar = dot_product / length_squared
    # print("projection_scalar:",projection_scalar)
    projection_point = [
        BoundarySegment1.contour[0][0] + projection_scalar * BoundarySegment1.direction[0],
        BoundarySegment1.contour[0][1] + projection_scalar * BoundarySegment1.direction[1]
    ]
    # print("点在线段上的投影点:", projection_point)

    # 创建一个新的旋转后的多边形
    rotated_polygon = []
    for x, y in quadrilateral.exterior.coords:
        # 计算Element轮廓上的点到投影点的向量
        point_to_projection = [x - projection_point[0], y - projection_point[1]]
        # angle = math.atan2(point_to_projection[1], point_to_projection[0]) + angle_difference
        angle = angle_difference
        # print("旋转的弧度为：",angle)

        vector1 = np.array(BoundarySegment1.direction)
        vector2 = np.array(BoundarySegment2.direction)

        cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        angle_test = np.arccos(cosine_angle)
        cross_product = np.cross(vector1, vector2)

        # 判断旋转方向
        if cross_product < 0:
            # 顺时针旋转
            clockwise_angle = -angle_test
        else:
            # 逆时针旋转
            clockwise_angle = angle_test

        # print("·····················旋转的角度为：", math.degrees(clockwise_angle))

        # 计算旋转矩阵的元素
        cos_theta = cos(clockwise_angle)
        sin_theta = sin(clockwise_angle)
        # print("cos_theta,sin_theta:", cos_theta, sin_theta)

        # 计算旋转后的新点坐标
        new_x = (x - projection_point[0]) * cos_theta - (y - projection_point[1]) * sin_theta + projection_point[
            0]
        new_y = (x - projection_point[0]) * sin_theta + (y - projection_point[1]) * cos_theta + projection_point[1]

        rotated_polygon.append([round(new_x), round(new_y)])

    # Element.contour = list([rotated_polygon.exterior.coords])
    result_element = copy.deepcopy(Element)
    result_element.contour = rotated_polygon
    # print("旋转后的多边形坐标：", Element.contour)

    return result_element


# 判断hausdorff距离是否小于65
def hausdorff_distance_65(Element, BoundarySegment_contour):
    hausdorff_distance = directed_hausdorff(Element.contour, BoundarySegment_contour)[0]
    if hausdorff_distance < 65:
        return 1
    else:
        return 0


# element上所有点到边界的距离大于等于13
def min_distance_to_boundary(Element, BoundarySegment_contour):
    e_contour = np.array(Element.contour)
    b_contour = np.array(BoundarySegment_contour)
    # 计算边界距离阈值
    threshold_distance = 17

    # 遍历元素上的每个点，并检查它们到外边界线段的距离
    for point in e_contour:
        x, y = point
        # 计算点到线段的距离
        distance_to_boundary = np.linalg.norm(
            np.cross(b_contour[1] - b_contour[0], b_contour[0] - point)) / np.linalg.norm(
            b_contour[1] - b_contour[0])

        if distance_to_boundary < threshold_distance:
            return 1

    return 0


#  已知旋转后的元素坐标，和目标边界，求该元素位于初始位置时的坐标
def start_Element(Element, BoundarySegment,block_contour):
    line = LineString(BoundarySegment.contour)
    line_length = line.length
    start_point = BoundarySegment.contour[0]
    end_point = BoundarySegment.contour[1]
    # print("start_point:",start_point)
    # print("end_point:",end_point)

    direction = BoundarySegment.direction
    # print("direction :",direction )
    # 暂定与边界的距离为40
    distance = 40

    # 将direction标准化
    t = distance / math.sqrt(direction[0] ** 2 + direction[1] ** 2)
    new_start_point = [start_point[0] + direction[0] * t,
                       start_point[1] + direction[1] * t]
    new_end_point = [end_point[0] + direction[0] * t,
                     end_point[1] + direction[1] * t]
    limited_contour = [new_start_point, new_end_point]
    # print("new_start_point:",new_start_point)
    # print("new_end_point:",new_end_point)
    # print("limited_contour:",limited_contour)

    # 将元素的中心位置移动到边界的起始点
    temp_Element1 = position_to_contour(Element, BoundarySegment, new_start_point)
    # print("temp_Element1 :",temp_Element1.contour )
    # 将元素沿与边界垂直的方向移动
    vertical_direction = [direction[1], -direction[0]]
    while is_out_boundary(temp_Element1, limited_contour) or min_distance_to_boundary(temp_Element1, limited_contour):
        # while hausdorff_distance_65(temp_Element1, limited_contour):  hausdorff_distance距离不太行，考虑用所有点距离直线的距离大于等于20
        print("循环一次")
        # while min_distance_to_boundary(temp_Element1, limited_contour):
        temp_Element1 = move_interval(temp_Element1, [[0, 0], [vertical_direction[0], vertical_direction[1]]])

    for i in range(len(temp_Element1.contour)):
        temp_Element1.contour[i][0]=round(temp_Element1.contour[i][0])
        temp_Element1.contour[i][1] = round(temp_Element1.contour[i][1])


    return temp_Element1


# 计算元素P和元素P‘在边界S方向上的最右投影点之间的距离
def distance_P_and_P__(Element1, Element2, BoundarySegment):
    polygon1 = Polygon(Element1.contour)
    polygon2 = Polygon(Element2.contour)
    line = LineString(BoundarySegment.contour)
    start_point = Point(BoundarySegment.contour[0][0], BoundarySegment.contour[0][1])

    # 初始化投影点列表
    projection_points1 = []
    projection_points2 = []

    # 遍历多边形的所有顶点，计算它们到线段的投影点
    for vertex in polygon1.exterior.coords:
        point = Point(vertex)
        projection = line.interpolate(line.project(point))
        projection_points1.append(projection)

    for vertex in polygon2.exterior.coords:
        point = Point(vertex)
        projection = line.interpolate(line.project(point))
        projection_points2.append(projection)

    # 计算投影点中距离边界起始点最远的距离
    max_distance1 = 0
    max_distance2 = 0
    for point1 in projection_points1:
        distance = start_point.distance(point1)
        if distance > max_distance1:
            max_distance1 = distance
    for point2 in projection_points2:
        distance = start_point.distance(point2)
        if distance > max_distance2:
            max_distance2 = distance
    result = max_distance1 - max_distance2

    print("x1-x2的距离：", result)
    return result

# 判断边界的元素是否要保留（是否超出边界、是否与已确定的元素有重叠）
def is_element_remain(Element,block_contour,result_elements):
    if is_out_block(Element,block_contour):
        return 0
    for e in result_elements:
        if is_intersect(Element,e):
            return 0
    print("可以存")
    return 1
