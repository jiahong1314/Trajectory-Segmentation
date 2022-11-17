import geopandas as gpd
import pandas as pd
import geopandas as gpd
import pandas as pd
import numpy as np
from numpy import median
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shapely
import math
from shapely.geometry import Point, Polygon, shape
import csv

sns.set()


# FieldtoRoad 解决道路被错误识别为耕地轨迹
def FieldtoRoad(gdf, direction, label):
    # 计算轨迹坐标数据中中每个方向分布数量
    sel = pd.cut(direction, list(range(0, 370, 10)))
    # 获得最大值
    maxPointNum = sel.value_counts().max()
    # 最大值小于12的判定为错误分类路径，将标签改为-1
    if maxPointNum < 50:
        gdf.loc[gdf["label"].isin([label]), "label"] = -1


# DBSCANIdentification 识别耕作轨迹和行驶轨迹
def DBSCANIdentification(gdf):
    # DNSCAN训练
    columns = ['longitude', 'latitude']
    azimuth = gdf['azimuth']
    # data = np.array(gdf[columns].dropna(axis=0, how='all'))
    data = np.array(gdf[columns].dropna(axis=0, how='any'))
    '''
    k = 0
    for i in range(len(data)):
        sample = data[i]
        for j in range(len(sample)):
            if np.isnan(sample[j]):
                print(sample)
                print(k)
        k += 1
    '''
    # earth's radius in km
    kms_per_radian = 6371.0086
    # define epsilon as 0.5 kilometers, converted to radians for use by haversine
    # This uses the 'haversine' formula to calculate the great-circle distance between two points
    # that is, the shortest distance over the earth's surface
    # http://www.movable-type.co.uk/scripts/latlong.html
    epsilon = 0.5 / kms_per_radian
    # radians() Convert angles from degrees to radians
    db = DBSCAN(eps=epsilon, min_samples=13, algorithm='ball_tree', metric='haversine').fit(data)
    # db = DBSCAN(eps=0.00005, min_samples=7).fit(data)
    labels = db.labels_
    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目
    gdf['label'] = labels

    # 剔除labei为-1的点，即聚类识别出的农机行驶轨迹
    trackSum = gdf[gdf['label'] != -1].copy()
    trackSum['azimuth'] = azimuth

    # 各个聚簇区域方向分布情况
    for i in range(0, n_clusters_):
        directionLabel = trackSum[trackSum['label'] == i].copy()
        FieldtoRoad(gdf, directionLabel['azimuth'].tolist(), i)

    return gdf


# 田路交界修正
def FieldAndRoadSwitch( gdf , directionList , labelList):
    # 获取该农田作业轨迹方向分布
    sel = pd.cut(directionList, np.arange(0, 370, 10))
    # 对方向分布进行排序获取数量前三的三个方向区间
    distribute = sel.value_counts().sort_values().tail(3).index.tolist()
    count = 0
    # 从进入农田部分的田路交界处进行纠正
    for index in labelList:
        # 判断标志
        flag = False
        # 判断每个index对应的方向角是否与农田作业轨迹方向分布相似
        for i in distribute:
            azimuthNow = gdf.loc[index]['azimuth']
            if azimuthNow in i:
                flag = True
                break
        # 如果相似判定为农田 不用修改
        if flag:
            count += 1
        else:   # 否则将标签修改为-1，修正为道路轨迹点
            gdf.loc[index, 'label'] = -1
        # 当超过2个点被判定为农田即停止遍历
        if count > 2:
            break
    # 从离开农田部分的田路交界处进行纠正
    count = 0
    for index in reversed(labelList):
        # 判断标志
        flag = False
        # 判断每个index对应的方向角是否与农田作业轨迹方向分布相似
        for i in distribute:
            azimuthReverse = gdf.loc[index]['azimuth']
            if azimuthReverse in i:
                flag = True
                break
        # 如果相似判定为农田 不用修改
        if flag:
            count += 1
        else:  # 否则将标签修改为-1，修正为道路轨迹点
            gdf.loc[index, 'label'] = -1
        #   当超过10个点被判定为农田即停止遍历
        if count > 10:
            break

        return gdf


if __name__ == '__main__':
    dirPath = 'E:/桌面/data'
    name = os.listdir(dirPath)
    outPath = 'images_354'
    if not os.path.exists(outPath):
        os.makedirs(outPath)  # 创建路径
    for i in name:
        filePath = dirPath + "\\" + i
        gdf = gpd.read_file(filePath)
        # 预处理
        gdf = gdf.dropna(axis=0, how='any', subset=['longitude', 'latitude', 'azimuth']).copy()  # 清洗空值
        gdf = gdf[(gdf['latitude'] != 0) & (gdf['longitude'] != 0)].copy()
        gdf = gdf.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)  #删除重复值
        gdf = DBSCANIdentification(gdf)    # DBSCAN密度聚类
        gdf.loc[(gdf['lac'] == 0) & (gdf['speed'] < 10), "label"] = -1   # 根据lac状态识别部分道路轨迹
        g1 = gdf.groupby(gdf['label']).groups   # 根据label进行分组
        labelList = g1.keys()   # 获取分组的标签值
        #  遍历每个label对应的作业轨迹
        for label in labelList:
            x1 = g1[label].tolist()   # 将每个标签对应的作业轨迹index转换为列表
            test1 = gdf.loc[x1]    # 根据index获取作业轨迹的全部数据
            azimuth = test1['azimuth']   # 获取方向角数据
            directionList = azimuth.tolist()
            # 进行田路交界部分轨迹修正
            gdf = FieldAndRoadSwitch(gdf, directionList, x1)

        # 保存图片
        outPath = 'images_354'
        sns.lmplot(x='longitude', y='latitude', data=gdf, hue='label', fit_reg=False)
        plt.savefig(outPath + "\\" + i[:-5] + ".png")
