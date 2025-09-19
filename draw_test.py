# # -*- coding: utf-8 -*-
# """
# Created on Fri Dec 20 14:56:29 2024

# @author: DELL
# """

# import matplotlib.pyplot as plt
# import cartopy.crs as ccrs
# import xarray as xr
# import os



# # 设置NC文件所在的文件夹路径
# nc_folder_path = r'C:\Users\DELL\Desktop\code_work\SDUT\txt2nc\save'

# # 获取文件夹内所有NC文件
# nc_files = [f for f in os.listdir(nc_folder_path) if f.endswith('.nc')]

# # 遍历所有NC文件
# for nc_file in nc_files:
#     # 打开NC文件
#     dataset = xr.open_dataset(os.path.join(nc_folder_path, nc_file))
    
#     # 提取降水量数据
#     precipitation = dataset['OI_merge']  # 假设'OI_merge'是您的降水量变量名
    
#     # 将小于0的值置为0，大于75的值置为70
#     precipitation = precipitation.clip(min=0, max=75)
    
#     # 设置绘图区域
#     fig = plt.figure(figsize=(10, 6))
#     ax = plt.axes(projection=ccrs.PlateCarree())
    
#     # 绘制降水量等值线图
#     contour = plt.contourf(precipitation.LON, precipitation.LAT, precipitation, levels=10, cmap='Blues')
#     ax.coastlines()
    
#     # 添加颜色条
#     plt.colorbar(contour, label='Precipitation (mm)')
    
#     # 设置标题
#     plt.title(f'Precipitation Distribution - {nc_file}')
    
#     # # 显示图形
#     # plt.show()
    
#     # 保存图形
#     plt.savefig(f'{nc_file}_precipitation_distribution.png', dpi=300)
#     plt.close()  # 关闭绘图窗口，避免显示在屏幕上

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xarray as xr
import os

# 设置NC文件所在的文件夹路径
nc_folder_path = r'/weather/Fusion/2021-2024年10minCR_nc/2024/20240508'

# 获取文件夹内所有NC文件
nc_files = [f for f in os.listdir(nc_folder_path) if f.endswith('.nc')]

# 自定义颜色
mycolors3 = ('#FFFFFF', '#48A0F0', '#64DBDD', '#5AC839', '#3F8E27', '#FFFE54', '#E1C141',
             '#F09637', '#EA3323', '#C5291C', '#B12418', '#EB3AEA', '#8920AD')

# 遍历所有NC文件
for nc_file in nc_files:
    # 打开NC文件
    dataset = xr.open_dataset(os.path.join(nc_folder_path, nc_file))
    
    # 提取雷达反射率数据
    # 假设'CR'是您的雷达反射率变量名，请根据实际情况替换
    cr_data = dataset['CR']  # 假设'CR'是您的雷达反射率变量名
    
    # 将小于0的值置为0，大于75的值置为75
    cr_data = cr_data.clip(min=0, max=75)
    
    # 设置绘图区域
    fig = plt.figure(figsize=(10, 6))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # 绘制雷达反射率等值线图
    contour = ax.contourf(cr_data.LON, cr_data.LAT, cr_data.isel(time=0), 
                          levels=[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
                          colors=mycolors3)
    ax.coastlines()
    
    # 添加颜色条
    plt.colorbar(contour, label='Radar Reflectivity (dBZ)')
    
    # 设置标题
    plt.title(f'Radar Reflectivity Distribution - {nc_file}')
    
    # 保存图形
    plt.savefig(f'/weather/Fusion/2021-2024年10minCR_png/{os.path.splitext(nc_file)[0]}.png', dpi=300)
    plt.close()  # 关闭绘图窗口，避免显示在屏幕上