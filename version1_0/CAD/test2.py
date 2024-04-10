# -*- codeing=utf-8 -*-
# @Time:2023/9/16 23:12
# @Author: 杨又菁
# @File:test2.py.py
# @Software:PyCharm

import win32com.client
import csv

# 连接到AutoCAD应用程序和打开DWG文件
acad = win32com.client.Dispatch("AutoCAD.Application")
doc = acad.Documents.Open("your_file.dwg")

# 创建一个CSV文件来保存轮廓坐标
csv_filename = "contours.csv"
csv_file = open(csv_filename, "w", newline="")
csv_writer = csv.writer(csv_file)

# 获取模型空间的图形实体并遍历它们
model_space = doc.ModelSpace

for entity in model_space:
    if entity.EntityName == "AcDbPolyline":
        # 获取多段线（LWPOLYLINE或POLYLINE）的坐标
        points = entity.Coordinates
        # 将坐标写入CSV文件
        csv_writer.writerow(points)

# 关闭CSV文件
csv_file.close()

# 关闭AutoCAD应用程序
acad.Quit()
