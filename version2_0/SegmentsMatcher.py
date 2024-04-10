# -*- codeing=utf-8 -*-
# @Time:2024/3/26 6:52
# @Author: 杨又菁
# @File:SegmentsMatcher.py
# @Software:PyCharm
from Boundary import *
import numpy as np
from scipy.optimize import linear_sum_assignment


class SegmentsMatch:
    def __init__(self, segments1, segments2):
        self.segments1 = segments1
        self.segments2 = segments2

    def Match(self):
        vertices1 = []
        vertices2 = []
        num_segment1 = len(self.segments1)
        num_segment2 = len(self.segments2)
        for segment1 in self.segments1:
            vertices1.append(segment1.contour[0])
        vertices1 = np.array(vertices1)
        for segment2 in self.segments2:
            vertices2.append(segment2.contour[0])
        vertices2 = np.array(vertices2)
        print("vertices1:", vertices1)
        print("vertices2:", vertices2)

        # 计算顶点之间的距离矩阵
        distance_matrix = np.linalg.norm(vertices1[:, np.newaxis, :] - vertices2, axis=-1)

        # 使用匈牙利算法寻找最优匹配
        row_ind, col_ind = linear_sum_assignment(distance_matrix)

        # 匹配结果
        matches = [(row, col) for row, col in zip(row_ind, col_ind)]

        # match中没有出现的顶点个数，就是要增加的段的个数
        # 暂时没考虑既增加又减少的情况，只考虑增加多段的情况
        if num_segment1 - len(matches) > 0:  # segment1的数量多，需要增加segment2的个数
            # 找到多出来的顶点
            extra1 = []
            for i in range(num_segment1):
                if not any(item[0] == i for item in matches):
                    extra1.append(i)
            # 找到segment1中与这些顶点相关的边，以及应该在segment2中的哪些顶点中增加边
            add_mun1 = 0
            # 加在哪里
            segments_to_add = []
            for i in extra1:
                if (i - 1) % num_segment1 not in extra1:
                    temp_start = (i - 1) % num_segment1
                    add_mun1 += 1
                    if (i + 1) % num_segment1 not in extra1:
                        temp_end = (i + 1) % num_segment1
                        segments_to_add.append((temp_start, temp_end))
            print("segments_to_add:", segments_to_add)
            # 比如这里返回的是[(3,5)]，说明3到5中间的段在segment2中是一段，需要拆分
            # 进行拆分的具体步骤
            segments_need_add = []
            for segment in segments_to_add:
                temp = (0, 0)  # 初始化临时变量
                for match in matches:
                    if match[0] == segment[0]:
                        temp = (match[1], temp[1])  # 更新temp的第一个元素
                    if match[0] == segment[1]:
                        temp = (temp[0], match[1])  # 更新temp的第二个元素
                segments_need_add.append(temp)
            print("segments_need_add:", segments_need_add)
            self.add_segments(segments_to_add, segments_need_add, 2)
        if num_segment2 - len(matches) > 0:  # segment2的数量多，需要增加segment1的个数
            # 找到多出来的顶点
            extra2 = []
            for i in range(num_segment2):
                if not any(item[1] == i for item in matches):
                    extra2.append(i)
            # 找到segment2中与这些顶点相关的边，以及应该在segment1中的哪些顶点中增加边
            add_mun2 = 0
            # 加在哪里
            segments_to_add = []
            print("extra2:", extra2)
            for i in extra2:
                if (i - 1) % num_segment2 not in extra2:
                    temp_start = (i - 1) % num_segment2
                    add_mun2 += 1
                    if (i + 1) % num_segment2 not in extra2:
                        temp_end = (i + 1) % num_segment2
                        segments_to_add.append((temp_start, temp_end))
            print("segments_to_add:", segments_to_add)
            # 比如这里返回的是[(3,5)]，说明3到5中间的段在segment2中是一段，需要拆分
            # 进行拆分的具体步骤
            segments_need_add = []
            for segment in segments_to_add:
                temp = (0, 0)  # 初始化临时变量
                for match in matches:
                    if match[1] == segment[0]:
                        temp = (match[0], temp[1])  # 更新temp的第一个元素
                    if match[1] == segment[1]:
                        temp = (temp[0], match[0])  # 更新temp的第二个元素
                segments_need_add.append(temp)
            print("segments_need_add:", segments_need_add)
            self.add_segments(segments_to_add, segments_need_add, 1)
        # print("self.segments1:", self.segments1,"self.segments2:", self.segments2)
        return matches,self.segments1,self.segments2

    def add_segments(self, segments_to_add, segments_need_add, num):
        # 一一对应，在segments_need_add中根据segments_to_add来加
        add_num=0
        for idx,i in enumerate(segments_to_add):
            print("i:",i,"idx:",idx,"num:",num)

            if num == 1:  # 1要增加,设置num的原因，识别是1还是2要增加
                print("num==1")
                current_num = 0
                add_p = []
                start = i[0]
                end = i[1]
                print("start:", start, "end:", end)
                for j in range(start, end):
                    current_num += self.segments2[j].length()
                    add_p.append(current_num)
                for k in range(len(add_p) - 1):
                    add_p[k] /= add_p[-1]
                print("add_p:", add_p)
                # print("toatl_length", add_p[-1], "add_p:", add_p)
                # print("self.segments2[segments_need_add[idx][0]].contour:",self.segments2[segments_need_add[idx][0]].contour)
                # 更新segments1，加入插入的段
                insert_segments = self.segments1[segments_need_add[idx][0]].addsegment(add_p)
                for segment in range(len(insert_segments)):
                    # print("insert_segments[segment]:",insert_segments[segment])
                    temp = BoundarySegment(0, insert_segments[segment])
                    # print("self.segments2:",self.segments2)
                    if segment == 0:
                        self.segments1[idx + add_num].contour = insert_segments[segment]
                    else:
                        self.segments1.insert(idx + add_num + segment, temp)
                add_num += len(add_p) - 1

                # 调整segments中segment的num序号
                for i in range(len(self.segments1)):
                    self.segments1[i].num = i
            if num == 2:  # 2要增加,设置num的原因，识别是1还是2要增加
                print("num==2")
                current_num = 0
                add_p = []
                start = i[0]
                end = i[1]
                print("start:",start,"end:",end)
                for j in range(start, end):
                    current_num += self.segments1[j].length()
                    add_p.append(current_num)
                for k in range(len(add_p)-1):
                    add_p[k] /= add_p[-1]
                print("add_p:",add_p)
                print("toatl_length", add_p[-1], "add_p:", add_p)
                # print("self.segments2[segments_need_add[idx][0]].contour:",self.segments2[segments_need_add[idx][0]].contour)
                # 更新segments2，加入插入的段
                insert_segments=self.segments2[segments_need_add[idx][0]].addsegment(add_p)
                for segment in range(len(insert_segments)):
                    # print("insert_segments[segment]:",insert_segments[segment])
                    temp=BoundarySegment(0,insert_segments[segment])
                    # print("self.segments2:",self.segments2)
                    if segment==0:
                        self.segments2[idx+add_num].contour=insert_segments[segment]
                    else:
                        self.segments2.insert(idx+add_num+segment,temp)
                add_num+=len(add_p)-1

                # 调整segments中segment的num序号
                for i in range(len(self.segments2)):
                    self.segments2[i].num=i
