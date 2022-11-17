import folium
import geopandas as gpd
import pandas as pd
import geopandas as gpd
import pandas as pd
import numpy as np
from numpy import median
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import shapely
import math
from shapely.geometry import Point, Polygon, shape
import csv

# FieldtoRoad 解决道路被错误识别为耕地轨迹
def FieldtoRoad(gdf, direction, label):
    # 计算轨迹坐标数据中中每个方向分布数量
    sel = pd.cut(direction, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180,
                             190, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360])
    # 获得最大值
    maxPointNum = sel.value_counts().max()
    # 最大值小于12的判定为错误分类路径，将标签改为-1
    if maxPointNum < 20:
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
if __name__ == '__main__':
    gdf = gpd.read_file("E:/桌面/data/865074051109485_2022-05-26.json")
    gdf = gdf.dropna(axis=0, how='any', subset=['longitude', 'latitude', 'azimuth']).copy()
    gdf = gdf[(gdf['latitude'] != 0) & (gdf['longitude'] != 0)].copy()
    gdf = gdf.drop_duplicates(subset=['latitude', 'longitude'], keep='first', inplace=False)
    gdf = DBSCANIdentification(gdf)
    gdf_part = gdf.copy()
    gdf_part = gdf_part[gdf_part['label'] != -1].copy()
    columns = ['latitude', 'longitude']
    gdf_image = gdf_part[columns].values.tolist()
    tiles = 'https://webst02.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}'
    bj_map = folium.Map(
               tiles=tiles,
               position='relative',
               max_zoom=28,
               attr='高德-卫星影像图',
               zoom_start=25,
               )
    # bj_map = folium.Map(location=gdf_image[0], zoom_start=4, tiles='Stamen Terrain')

    # folium.Marker(
    #     location=gdf_image,
    #     popup='Mt. Hood Meadows',
    #     icon=folium.Icon(icon='cloud')
    # ).add_to(bj_map)
    # for loc in gdf_image:
    #     folium.Marker(
    #         location=loc,
    #         popup='Mt. Hood Meadows',
    #         icon=folium.Icon(icon='cloud')
    #     ).add_to(bj_map)
    ls = folium.PolyLine(locations=gdf_image,
                         color='red')
    ls.add_to(bj_map)
    print("test")
    bj_map.save("map.html")
