# -*- codeing=utf-8 -*-
# @Time:2023/10/12 20:25
# @Author: 杨又菁
# @File:main.py
# @Software:PyCharm


import csv
import numpy as np
import sys
sys.path.append('../AssignToG')
from Boundary import *
from Elements import *
from keyfunctions import *
import copy

import cv2


# 读取轮廓信息
def csv_row_from_elements(row, Element):
    # print(row)
    Element.ele_type = row[1]
    t1 = row[2].split(',')
    print(t1)
    if not  t1 == ['']:
        Element.boundary = [int(item) for item in t1]
    # Element.boundary = row[2]
    # print("Element.boundary:",Element.boundary)
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


# 读取边界信息
def csv_row_from_boundarysegment(row, BoundarySegment):
    # print(row)
    t1 = row[1].split(',')
    if t1[0] == '':
        BoundarySegment.elements = []
    else:
        BoundarySegment.elements = [int(item) for item in t1]
        print("BoundarySegment.elements", BoundarySegment.elements)
    t2 = row[2].split(',')
    BoundarySegment.direction = [float(t2[0]), float(t2[1])]
    t3 = row[3].split(',')
    # print("读取元素坐标后",temp)
    segment = []
    i = 0
    # print("temp_len:", len(temp))
    while i < len(t3) - 1:
        point = []
        point.append(round(float(t3[i])))
        point.append(round(float(t3[i + 1])))
        segment.append(point)
        i = i + 2
    # cv2.drawContours(img, [np.array(segment)], -1, (0, 255, 255), 1)
    BoundarySegment.contour = segment
    return BoundarySegment


# 创建画布
segmenter_contour = np.ones((1000, 1200, 3), dtype=np.uint8) * 0

# 读取elements
with open('../CSV/elements.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)

    org_elements = []
    i = 0
    for row in csv_reader:
        temp_element = Element(i)
        temp_element = csv_row_from_elements(row, temp_element)
        # print("temp_element.contour:",temp_element.contour)
        org_elements.append(temp_element)
        # print("temp_element：",temp_element.contour)
        i += 1
        # cv2.drawContours(segmenter_contour, [np.array(temp_element.contour)], -1, (0, 255, 255), 1)
    # print("org_segments:",org_segments)

# 读取elements
with open('../CSV/elements_changed.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)

    changed_elements = []
    i = 0
    for row in csv_reader:
        temp_element = Element(i)
        temp_element = csv_row_from_elements(row, temp_element)
        # print("temp_element.contour:",temp_element.contour)
        changed_elements.append(temp_element)
        i += 1
        # print("elements_changed:",[np.array(temp_element.contour)])
        # cv2.drawContours(segmenter_contour, [np.array(temp_element.contour)], -1, (0, 255, 255), 1)
    # print("org_segments:",org_segments)

# 读取target_boundarysegment
with open('../CSV/target_boundarysegment.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    block_contour=[]
    csv_reader = csv.reader(csv_file)
    target_bs = []
    i = 0
    for row in csv_reader:
        temp = BoundarySegment(i)
        temp = csv_row_from_boundarysegment(row, temp)
        # print("temp_element.contour:",temp_element.contour)
        target_bs.append(temp)
        block_contour.append(temp.contour[0])
        i += 1
        print("[np.array(temp.contour)]:", [np.array(temp.contour)])
        cv2.drawContours(segmenter_contour, [np.array(temp.contour)], -1, (0, 255, 255), 1)
    # print("org_segments:",target_bs)
    print("block_contour:",block_contour)


# 读取boundarysegment
with open('../CSV/boundarysegment.csv', 'r') as csv_file:
    # 创建CSV读取器对象
    csv_reader = csv.reader(csv_file)

    org_bs = []
    i = 0
    for row in csv_reader:
        temp = BoundarySegment(i)
        temp = csv_row_from_boundarysegment(row, temp)
        # print("temp_element.contour:",temp_element.contour)
        org_bs.append(temp)
        i += 1
        # cv2.drawContours(segmenter_contour, [np.array(temp.contour)], -1, (0, 255, 255), 1)

if __name__ == '__main__':
    # block_contour=[[540.8349814963058,553.712927487172],[740.8349814963058,485.991896839374],[602.448527563849,399.42675331566704],[326.85337675367964,399.42675331566704],[326.85337675367964,554.3018060145442]]
    result_elements=[]
    l=0
    count_org_element = [0] * len(org_elements)  # 统计元素被使用了几次，限定fixed不能重复使用
    # 确定第一个组
    for group in target_bs:
        # group=target_bs[1]
        r_group_elements = []  # 存储该组有哪些元素

        # group = target_bs[4]
        # 找出目标边界对应的原始边界
        org = org_bs[group.num]
        print("group.num:", group.num)

        # 计算该组的元素个数M
        sum = 0
        for i in org.elements:
            # print(org.contour)
            sum += projector_distance(org_elements[i], org)
        arg = sum / len(org.elements)
        M = round((math.sqrt((group.contour[1][0] - group.contour[0][0]) ** 2 + (
                    group.contour[1][1] - group.contour[0][1]) ** 2) - 33 * 2) / arg)
        print("该组需要排列多少个元素：", M)

        # 确定第一个元素
        p = 0  # 记录P’移动到哪里了
        element_P__ = changed_elements[org.elements[p]]
        # cv2.drawContours(segmenter_contour, [np.array(element_P__.contour)], -1, (255, 0, 255), 1)

        # i=org.elements[0]
        min_area = float('inf')
        first_e = copy.deepcopy(org.elements[0])
        print("first_e:", first_e)
        num_e = 0


        # Flag = 0
        while M > 0:
            # 确定第一个元素
            if num_e == 0:
                # for i in range(len(org_elements)):
                for i in org.elements:
                    # fixed只能用一次
                    print("if count_org_element[i] == 1 and org_elements[i].ele_type == fixed:",count_org_element[i],org_elements[i].ele_type)
                    if count_org_element[i] == 1 and org_elements[i].ele_type == "fixed":
                        continue
                    # 旋转
                    rotated_element = rotate_Element(org_elements[i], org, group)
                    print("旋转完毕：",i,"`org:",org.num,group.num)
                    # rotated_element = rotate_Element(org_elements[i], org_bs[org_elements[i].boundary[0]], group)

                    # print("[np.array(rotated_element.contour[:-1])]:",[np.array(rotated_element.contour[:-1])])
                    # cv2.drawContours(segmenter_contour, [np.array(rotated_element.contour)], -1, (0, 255, 255), 1)

                    # 移动中心点到边界起始处
                    temp_element = start_Element(rotated_element, group,block_contour)
                    # print("移动到起始点temp_element.contour:",[np.array(temp_element.contour)])
                    # cv2.drawContours(segmenter_contour, [np.array(temp_element.contour)], -1, (0, 255, 255), 1)

                    # 与P‘相交面积最大的
                    # area = Area(temp_element, element_P__)  # 导致选取元素尽量小
                    # area = Area(element_P__,temp_element)  #会导致选取元素尽可能得大，覆盖原来的元素
                    area = Area(temp_element, element_P__) * 0.2 + Area(element_P__, temp_element) * 0.8
                    if area < min_area:
                        first_e = temp_element.num
                        min_area = area
                        min_element = copy.deepcopy(temp_element)
                    # L.append(min_area)
                # print("=======================选取元素编号：", first_e)
                print("-----------------------min_element:", min_element.num)
                # cv2.drawContours(segmenter_contour, [np.array(org_elements[first_e].contour)], -1, (0, 255, 255), 1)

                # cv2.drawContours(segmenter_contour, [np.array(min_element.contour)], -1, (100, 100, 100), 1)

            # 确定后面的元素
            if num_e != 0:
                print("num_e:", num_e)
                # while num_e != 1 and len(org.elements) > 1:
                min_area = float('inf')
                # 跟哪个P‘进行比较，确定P’
                print("上一个元素投影长度：", projector_distance(r_group_elements[num_e - 1], group))
                print("最右端距离相差多少：",
                      distance_P_and_P__(changed_elements[org.elements[p]], r_group_elements[num_e - 1], group))
                if distance_P_and_P__(changed_elements[org.elements[p]], r_group_elements[num_e - 1],
                                      group) < projector_distance(r_group_elements[num_e - 1], group) and p < len(
                        org.elements)-1:
                    print("p向后移了一位:", p)
                    p += 1
                element_P__ = changed_elements[org.elements[p]]
                # cv2.drawContours(segmenter_contour, [np.array(element_P__.contour)], -1, (255, 0, 255), 1)

                for j in org.elements:
                    print("j:", j)
                    # fixed只能用一次
                    print("if count_org_element[j] == 1 and org_elements[j].ele_type == fixed:", count_org_element[j],
                          org_elements[j].ele_type)
                    if count_org_element[j] == 1 and org_elements[j].ele_type == "fixed":
                        continue
                    # 将元素进行旋转，调整角度
                    rotated_element = rotate_Element(org_elements[j], org, group)
                    # cv2.drawContours(segmenter_contour, [np.array(rotated_element.contour)], -1, (0, 255, 255), 1)
                    print("旋转完毕")
                    # 找到元素的下一个位置
                    next_element = next_not_intersect(r_group_elements[num_e - 1], rotated_element, group)
                    # next_element = next_not_intersect(r_group_elements[num_e - 1], org_elements[j], group)
                    print("找到了下一个的位置：", next_element.contour)
                    # area = Area(next_element, element_P__)
                    area = Area(next_element, element_P__) * 0.5 + Area(element_P__, next_element) * 0.5
                    lm = tendto_different(r_group_elements[num_e - 1], next_element)
                    if area + lm < min_area:
                        next_e = next_element.num
                        min_area = area + lm
                        min_element = copy.deepcopy(next_element)

                # cv2.drawContours(segmenter_contour, [np.array(min_element.contour)], -1, (200, 200, 200), 5)
                print("min_element.num:", min_element.num)


            r_group_elements.append(min_element)

            # 确定了下一个元素以后存起来
            # cv2.drawContours(segmenter_contour, [np.array(min_element.contour)], -1, (255, 255, 255), 1)
            if is_element_remain(min_element,block_contour,result_elements):
                print("存了")
                count_org_element[min_element.num] += 1  # 元素被使用次数+1
                print("count_org_element:",count_org_element)
                r_element = copy.deepcopy(min_element)
                r_element.num=l
                r_element.boundary.clear()
                r_element.boundary.append(group.num)
                group.elements.append(r_element.num)
                result_elements.append(r_element)
                l+=1
                cv2.drawContours(segmenter_contour, [np.array(r_element.contour)], -1, (255, 255, 255), 1)

            M -= 1
            # print("放进去的元素的坐标为：",r_group_elements[num_e].contour)
            # cv2.drawContours(segmenter_contour, [np.array(r_group_elements[num_e].contour)], -1, (255, 255, 255), 1)
            num_e += 1






    cv2.imshow('Segmented Contour', segmenter_contour)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
