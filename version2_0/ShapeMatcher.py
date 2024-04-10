# -*- codeing=utf-8 -*-
# @Time:2024/3/25 17:04
# @Author: 杨又菁
# @File:ShapeMatcher.py
# @Software:PyCharm

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.optimize import linear_sum_assignment
import random
from tqdm import tqdm


class ShapeMatch:
    def __init__(self, similer_neibh):
        self.similer_neibh = similer_neibh

    def find_most_similar(self, neighborhood_data,_id):
        most_similar_neighborhood = None
        min_distance = float('inf')
        i = 0
        for neighborhood in tqdm(neighborhood_data, desc="Processing neighborhoods"):
            i += 1
            neighborhood_boundary = neighborhood["boundary"]
            if self.calculate_area(self.similer_neibh) * 0.5 < self.calculate_area(
                    neighborhood_boundary) < self.calculate_area(self.similer_neibh) * 1.5:
                distance = self.distance_function(np.array(neighborhood_boundary))
                if distance < min_distance and neighborhood["_id"] != _id:
                    min_distance = distance
                    most_similar_neighborhood_id = neighborhood["_id"]
        print("min_distance:", min_distance)
        return most_similar_neighborhood_id

    def calculate_area(self, neighborhood):
        # 计算多边形面积，使用 Shoelace formula
        x = [point[0] for point in neighborhood]
        y = [point[1] for point in neighborhood]
        return 0.5 * abs(sum(x[i] * y[i + 1] - x[i + 1] * y[i] for i in range(-1, len(x) - 1)))

    # 计算形状匹配的距离
    def distance_function(self, neighborhood_boundary):
        N = 100
        angle = 12

        # 获取样本点
        sample_points1 = self.Jitendra_Sample(np.array(self.similer_neibh))
        sample_points2 = self.Jitendra_Sample(np.array(neighborhood_boundary))
        histogram_feature1 = self.Shape_Context(sample_points1, angle)
        histogram_feature2 = self.Shape_Context(sample_points2, angle)

        # 代价矩阵
        # cost_matrix = Cost_function(histogram_feature1, histogram_feature2)
        cost_matrix = 0.5 * self.Cost_function_Shape_Context(histogram_feature1,
                                                             histogram_feature2) + 0.5 * self.Cost_function_Local_Appearance(
            np.array(self.similer_neibh), np.array(neighborhood_boundary), sample_points1, sample_points2)

        # 匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # 提取匹配的值
        matched_values = cost_matrix[row_ind, col_ind]
        # 计算匹配总代价
        total_cost = np.sum(matched_values)
        distance = total_cost
        # 归一化匹配总代价
        # distance = total_cost / np.sum(cost_matrix)
        # print("匹配总代价:", distance)

        # match = np.array([[x, y] for x, y in zip(row_ind, col_ind)])

        # 可视化
        # drawmatch_(img1=img1, img2=img2, points1=sample_points1, points2=sample_points2, match=match,
        #            visual_num=visual_num)

        return distance

    # 从给定的样本点集中进行采样
    def Jitendra_Sample(self, neighborhood_boundary, N=100, k=3):
        """
        Jitendra’s sampling
        points: 样本点，shape为[I, 2]，其中I为点的个数
        N: 采样的点的数目
        k: 阈值
        """
        # 增加样本点个数
        points = self.sample_on_boundary(neighborhood_boundary, 400)
        # 样本点个数
        # I = neighborhood_boundary.shape[0]
        I = 400
        # 首先需要对样本点进行乱序
        points = np.random.permutation(points)

        NStart = min(k * N, I)
        # 阈值
        if I > k * N:
            NStart_sample_points = points[:NStart]
        # 计算欧式距离矩阵
        dist = np.sqrt(
            np.sum(
                np.square(NStart_sample_points.reshape((1, NStart, 2)) - NStart_sample_points.reshape((NStart, 1, 2))),
                axis=-1)) + np.eye(NStart, dtype=int) * 999999999999

        # 迭代，删到只剩N个点
        for num in range(NStart - N):
            # 将该点删去，实现是设置为很大
            # i = np.where(dist == np.min(dist))[0][0]
            j = np.where(dist == np.min(dist))[1][0]
            dist[j, :] = 999999999999;
            dist[:, j] = 999999999999

        # 获取序列，注意去重
        i = np.unique(np.where(dist < 999999999999)[0])
        sample_points = NStart_sample_points[i]
        return sample_points

    # 增加样本点
    def sample_on_boundary(self, points, num_samples):
        # 计算边界点之间的距离
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        total_length = np.sum(distances)

        # 计算每段的采样间隔
        interval_length = total_length / num_samples

        sampled_points = []
        current_distance = 0
        current_segment = 0

        # 遍历每个边界点之间的距离
        for distance in distances:
            # 如果当前间隔长度小于剩余的距离
            while current_distance + interval_length < distance:
                # 计算采样点的位置
                ratio = current_distance / distance
                new_point = points[current_segment] + ratio * (
                        points[current_segment + 1] - points[current_segment])
                sampled_points.append(new_point)
                current_distance += interval_length

            # 跳转到下一段
            current_distance -= distance
            current_segment += 1

        return np.array(sampled_points)

    def Shape_Context(self, points, angle=12, distance=[0, 0.125, 0.25, 0.5, 1.0, 2.0]):
        """
        形状上下文直方图矩阵的构建
        points: 输入的采样点 shape[N,2]
        angle:  划分的角度区域个数
        distance: 划分的距离区域
        """
        # 计算欧式距离矩阵
        N = points.shape[0]
        dist = np.sqrt(np.sum(np.square(points.reshape((1, N, 2)) - points.reshape((N, 1, 2))), axis=-1))

        # 距离均值
        mean_dist = np.sum(dist) / (N * N - N)
        # 除以均值，减少缩放敏感性
        dist = np.log(dist / mean_dist + 0.000000000001) + np.eye(N, dtype=int) * 999
        # print(dist)

        # 角度计算
        theta = np.arctan((points[:, 1].reshape(1, N) - points[:, 1].reshape(N, 1)) / (
                points[:, 0].reshape(1, N) - points[:, 0].reshape(N, 1) + 0.000000000001)) / math.pi + (
                        (points[:, 0].reshape(1, N) - points[:, 0].reshape(N, 1)) < 0).astype(int) + 0.5  # range(0, 2)

        histogram_feature = np.zeros((N, angle, len(distance)))

        for i in range(angle):
            # angle range
            angle_matrix = (theta > (2 / angle * i)) * (theta <= (2 / angle * (i + 1)))
            for j in range(1, len(distance)):
                distance_matrix = (dist < distance[j]) * (dist > distance[j - 1])

                histogram_feature[:, i, j - 1] = np.sum(angle_matrix * distance_matrix, axis=1)
        return histogram_feature

    def imshow(self, img):
        """
        jupyter 内部可视化图像
        """
        plt.axis('off')
        # 3通道图像应从BGR转RGB再显示
        if len(img.shape) == 3:
            img = img[:, :, ::-1]  # transform image to rgb
        # 灰度图应转为RGB
        else:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        plt.imshow(img)
        plt.show()

    def visualization(self, img_path, sample_points, index):
        """
        可视化
        """
        for i in range(len(index)):
            img = cv2.imread(img_path)
            img = self.Edge_detection(img)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for point in sample_points:
                cv2.circle(img, (point[1], point[0]), 2, [0, 255, 0], 4)
            cv2.circle(img, (sample_points[i][1], sample_points[i][0]), 4, [255, 0, 0], 6)
            plt.subplot(2, len(index), i + 1)
            plt.axis('off')
            plt.imshow(img)

            plt.subplot(2, len(index), i + 1 + len(index))
            plt.axis('off')
            plt.imshow(self.histogram_feature[index[i]].astype(np.uint8))
        plt.show()

    def Cost_function_Shape_Context(self, histogram1, histogram2):
        """
        代价矩阵
        histogram1: N1*A*D
        histogram2: N2*A*D
        """
        A = histogram1.shape[1]
        D = histogram1.shape[2]
        N1 = histogram1.shape[0]
        N2 = histogram2.shape[0]
        assert histogram1.shape[1] == histogram2.shape[1]
        assert histogram1.shape[2] == histogram2.shape[2]
        cost = 0.5 * np.sum(np.sum(
            np.square(
                histogram1.reshape((N1, 1, A, D)) - histogram2.reshape((1, N2, A, D))) / (
                    histogram1.reshape((N1, 1, A, D)) + histogram2.reshape((1, N2, A, D)) + 0.000000001)
            , axis=-1), axis=-1)
        return cost

    def Cost_function_Local_Appearance(self, points1, points2, sample_points1, sample_points2):
        """
        Local Appearance
        """

        N1 = sample_points1.shape[0]
        N2 = sample_points2.shape[0]

        # 模拟生成图像，这里假设图像大小为 256x256
        img_shape = (2000, 2000)

        # 生成空白图像
        img1gray = np.zeros(img_shape, dtype=np.uint8)
        img2gray = np.zeros(img_shape, dtype=np.uint8)

        # 在图像上标记样本点
        for point in sample_points1:
            img1gray[int(point[0]), int(point[1])] = 255

        for point in sample_points2:
            img2gray[int(point[0]), int(point[1])] = 255

        sobel1_x = cv2.Sobel(img1gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel1_y = cv2.Sobel(img1gray, cv2.CV_64F, 0, 1, ksize=3)

        sobel2_x = cv2.Sobel(img2gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel2_y = cv2.Sobel(img2gray, cv2.CV_64F, 0, 1, ksize=3)

        cos1 = sobel1_x[sample_points1[:, 0].astype(int), sample_points1[:, 1].astype(int)]
        cos2 = sobel2_x[sample_points2[:, 0].astype(int), sample_points2[:, 1].astype(int)]
        sin1 = sobel1_y[sample_points1[:, 0].astype(int), sample_points1[:, 1].astype(int)]
        sin2 = sobel2_y[sample_points2[:, 0].astype(int), sample_points2[:, 1].astype(int)]

        cost = 0.5 * np.sqrt(
            np.square(cos2.reshape(1, N2) - cos1.reshape(N1, 1)) + np.square(sin2.reshape(1, N2) - sin1.reshape(N1, 1))
        )
        return cost

    def drawmatch(self, img_path1, img_path2, points1, points2, match, visual_num=20):
        """
        可视化匹配图像
        img_path1: 图像1的路径
        img_path2: 图像2的路径
        points1: 图像1的点坐标 (N1,2)
        points2: 图像2的点坐标 (N2,2)
        match: 匹配点索引对 (N3,2)
        visual_num: 可视化的点数目，随机抽点
        """
        img1 = cv2.imread(img_path1);
        img2 = cv2.imread(img_path2)

        # image2高度与image1一致对齐：
        img2 = cv2.resize(img2, (int(img1.shape[0] / img2.shape[0] * img2.shape[1]), img1.shape[0]))
        points2 = points2 * img1.shape[0] / img2.shape[0]
        points2[:, 1] += img1.shape[1]
        points2 = points2.astype(np.int32)

        # 拼接图像
        new_img = np.zeros([img1.shape[0], img1.shape[1] + img2.shape[1], 3])
        new_img[:, :img1.shape[1]] = img1
        new_img[:, img1.shape[1]:] = img2

        # 随机采样点
        match = np.random.permutation(match)
        for i in range(visual_num):
            # 获取匹配的特征点在两个图像中的索引
            idx1, idx2 = match[i]
            # 随机颜色
            color = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
            # 提取匹配特征点的坐标
            point1_img1 = points1[idx1]
            point2_img2 = points2[idx2]
            # 在图像1中绘制特征点
            new_img = cv2.circle(new_img, (int(point1_img1[1]), int(point1_img1[0])), int(img1.shape[0] / 100),
                                 color, int(img1.shape[0] / 200), 8, 0)
            # 在图像2中绘制特征点
            new_img = cv2.circle(new_img, (int(point2_img2[1]), int(point2_img2[0])), int(img2.shape[0] / 100),
                                 color, int(img2.shape[0] / 200), 8, 0)
            # 在两个特征点之间绘制连线
            new_img = cv2.line(new_img, (int(point1_img1[1]), int(point1_img1[0])),
                               (int(point2_img2[1]), int(point2_img2[0])), color, 1)

        new_img = new_img.astype(np.uint8)

        self.imshow(new_img)
        cv2.imwrite("result.jpg", new_img)

    def context_shape_match(self, neighborhood_boundary,visual_num=30, N=100, angle=12,
                            distance=[0, 0.125, 0.25, 0.5, 1.0, 2.0]):
        points1=self.similer_neibh
        points2=neighborhood_boundary
        # 获取样本点
        sample_points1 = self.Jitendra_Sample(np.array(self.similer_neibh))
        sample_points2 = self.Jitendra_Sample(np.array(neighborhood_boundary))
        histogram_feature1 = self.Shape_Context(sample_points1, angle)
        histogram_feature2 = self.Shape_Context(sample_points2, angle)

        # 代价矩阵
        # cost_matrix = Cost_function(histogram_feature1, histogram_feature2)
        cost_matrix = 0.5 * self.Cost_function_Shape_Context(histogram_feature1,
                                                             histogram_feature2) + 0.5 * self.Cost_function_Local_Appearance(
            np.array(self.similer_neibh), np.array(neighborhood_boundary), sample_points1, sample_points2)

        # 匈牙利匹配
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        # 提取匹配的值
        match = np.array([[x, y] for x, y in zip(row_ind, col_ind)])

        # 可视化
        # 创建一个空白图像
        # 假设图像大小为 500x500
        img_shape = (1000, 1000, 3)
        img1 = np.ones(img_shape, dtype=np.uint8) * 255  # 创建一个全白的图像
        # 将样本点连接起来，只绘制相邻点之间的线条
        for i in range(len(points1) - 1):
            pt1 = (int(points1[i][1]), int(points1[i][0]))  # 转换为整数坐标
            pt2 = (int(points1[i + 1][1]), int(points1[i + 1][0]))  # 转换为整数坐标
            cv2.line(img1, pt1, pt2, color=(0, 0, 255), thickness=5)
        # 保存图像为 jpg 文件
        cv2.imwrite("points1_contour.jpg", img1)

        # 创建一个空白图像
        img2 = np.ones(img_shape, dtype=np.uint8) * 255  # 创建一个全白的图像
        # 将样本点连接起来
        sample_points2 = sample_points2[np.argsort(sample_points2[:, 0])]  # 根据第一列排序
        # 将样本点连接起来，只绘制相邻点之间的线条
        for i in range(len(points2) - 1):
            pt1 = (int(points2[i][1]), int(points2[i][0]))  # 转换为整数坐标
            pt2 = (int(points2[i + 1][1]), int(points2[i + 1][0]))  # 转换为整数坐标
            cv2.line(img2, pt1, pt2, color=(0, 0, 255), thickness=5)
        # 保存图像为 jpg 文件
        cv2.imwrite("points2_contour.jpg", img2)

        self.drawmatch("points1_contour.jpg", "points2_contour.jpg", points1=sample_points1, points2=sample_points2, match=match,
                        visual_num=visual_num)

    def tps_trans(self, p1, p2, gray, tps_lambda=0.2, ransac_reproj_threshold=5.0):
        '''
        Thin-Plate Spline Transform Algorithm

        input:
            p1 : feature points in the image to be transformed
            p2 : target feature points
            gray : input image
            tps_lambda : a tps parameter
            ransac_reproj_threshold : RANSAC reprojection threshold
        output:
            out_img : transformed input image
        '''
        # 使用 RANSAC 算法估计透视变换矩阵
        M, _ = cv2.findHomography(p1, p2, cv2.RANSAC, ransac_reproj_threshold)

        # 应用透视变换
        out_img = cv2.warpPerspective(gray, M, (gray.shape[1], gray.shape[0]))

        return out_img
