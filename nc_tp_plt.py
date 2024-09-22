# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:55:08 2024

@author: DELL
"""


import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd  # 导入pandas

# 打开.nc文件
original_path = r'/scratch/jianhao/data/5_625deg/total_precipitation/total_precipitation_2000_5.625deg.nc'
ds = xr.open_dataset(original_path)


# 选择特定的时间
selected_data = ds['tp'].sel(time='2020-08-19T23:00:00.000000000')
formatted_time = str(selected_data.time.values)

# 获取经纬度数据
lon = ds['lon'].values
lat = ds['lat'].values

# 创建图形和轴
fig, ax = plt.subplots(figsize=(10, 5))

# 绘制降水量填色图
tp_plot = ax.pcolormesh(lon, lat, selected_data.values, cmap='rainbow', shading='auto', alpha=0.5)
cbar = plt.colorbar(tp_plot, orientation='vertical')
cbar.set_label('Total Precipitation (m)')

# 设置坐标轴标签
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')


plt.title(formatted_time)

# 显示图形
plt.show()