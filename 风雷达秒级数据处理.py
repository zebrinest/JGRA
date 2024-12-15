# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 01:30:45 2024

@author: Administrator
"""
import numpy as np
import pandas as pd
import os
import math
from scipy.interpolate import interp1d



import os
import fnmatch
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

folder_path = 'I:\\香河风\\202312-202402\\'  # 指定文件夹路径

def read_file(file_path):
    print(f"正在读取文件: {file_path}")  # 打印文件路径
    try:
        return pd.read_table(
            file_path,
            encoding='ANSI',  # Try changing encoding if 'ANSI' doesn't work
            delimiter=';',     # Ensure this matches your file's delimiter
            on_bad_lines='warn'
        )
    except pd.errors.EmptyDataError:
        print(f"空数据错误，跳过文件: {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"读取文件时出错 {file_path}: {e}")
        return pd.DataFrame()

file_paths = []
for dirpath, dirnames, filenames in os.walk(folder_path):
    for filename in fnmatch.filter(filenames, '*.gz.txt'):
        file_paths.append(os.path.join(dirpath, filename))

# 使用四个线程读取文件，并显示进度条
all_data = []
with ThreadPoolExecutor(max_workers=6) as executor:
    for data in tqdm(executor.map(read_file, file_paths), total=len(file_paths), desc="读取进度"):
        all_data.append(data)
wind_sec = pd.concat(all_data, ignore_index=True)

wind_sec = pd.read_csv("E:\\Beijing2024\\Wind Lidar\\0805-0810\\merged_sorted_data.csv")



elevation = 1115
###################处理风雷达数据#########################
##### 合并源文件#####
wind_data_path = 'E:\\Wuhai Data\\cz-lsm wind lidar\\second data\\原始秒级数据\\rawdata\\'

# Read and concatenate data files
wind_data_raw = []
for filename in os.listdir(wind_data_path):
    file_path = os.path.join(wind_data_path, filename)  # 获取文件完整路径
    # 使用pandas读取文件，跳过前8行
    wind_data = pd.read_table(file_path, encoding='ANSI',delimiter=';',on_bad_lines='warn')
    wind_data_raw.append(wind_data)  # 将数据帧添加到列表中
wind_sec = pd.concat(all_data, axis=0, ignore_index=True)   

# Convert timestamps
wind_sec.timestamp = pd.to_datetime(wind_sec.timestamp)
wind_sec['timestamp'] = wind_sec['timestamp'] + pd.Timedelta(hours=8)
wind_sec=wind_sec.reset_index()
# Filtering invalid measurements
wind_sec.loc[(wind_sec['Confidence Index Status'] == 0) | 
             (wind_sec['Horizontal Wind Speed [m/s]'] == 0) | 
             (wind_sec['Horizontal Wind Speed [m/s]'] == 999), 
             'Horizontal Wind Speed [m/s]'] = np.NaN
wind_sec.loc[(wind_sec['Confidence Index Status'] == 0) | 
             (wind_sec['Vertical Wind Speed [m/s]'] == 0) | 
             (wind_sec['Vertical Wind Speed [m/s]'] == 999), 
             'Vertical Wind Speed [m/s]'] = np.NaN
wind_sec.loc[(wind_sec['Confidence Index Status'] == 0) | 
             (wind_sec['Horizontal Wind Direction [癩'] == 0) | 
             (wind_sec['Horizontal Wind Direction [癩'] == 999), 
             'Horizontal Wind Direction [癩'] == np.NaN

# Calculate wind components
wind_sec['u'] = -wind_sec['Horizontal Wind Speed [m/s]'] * np.sin(wind_sec['Horizontal Wind Direction [癩'] * np.pi / 180.)
wind_sec['v'] = -wind_sec['Horizontal Wind Speed [m/s]'] * np.cos(wind_sec['Horizontal Wind Direction [癩'] * np.pi / 180.)
height=sorted(wind_sec['Altitude [m]'][wind_sec['Altitude [m]'] % 25 == 0].unique())

# Aggregate data into 1-minute intervals
wind_1min = (wind_sec.groupby(['Altitude [m]', pd.Grouper(key='timestamp', freq='1min')]).mean()).reset_index()
wind_5min = (wind_1min.groupby(['Altitude [m]', pd.Grouper(key='timestamp', freq='5min')]).mean()).reset_index()
wind_15min = (wind_1min.groupby(['Altitude [m]', pd.Grouper(key='timestamp', freq='15min')]).mean()).reset_index()
wind_30min = (wind_1min.groupby(['Altitude [m]', pd.Grouper(key='timestamp', freq='30min')]).mean()).reset_index()

wind_5min['wind direction']=np.mod(180 + np.arctan2(wind_5min['u'], wind_5min['v']) * 180 / np.pi, 330)
wind_15min['wind direction']=np.mod(180 + np.arctan2(wind_15min['u'], wind_15min['v']) * 180 / np.pi, 330)
wind_30min['wind direction']=np.mod(180 + np.arctan2(wind_30min['u'], wind_30min['v']) * 180 / np.pi, 330)


#转换为igor显示的正确角度：Wind_matrix[:, 1] = (np.pi / 2 - np.arctan2(wind_1min['v'], wind_1min['u'])) % (2 * np.pi)

wind_5min['rad_igor']=np.arctan2(wind_5min['v'], wind_5min['u'])
wind_15min['rad_igor']=np.arctan2(wind_15min['v'], wind_15min['u'])
wind_30min['rad_igor']=np.arctan2(wind_30min['v'], wind_30min['u'])

wind_5min['rad']=wind_5min['wind direction']/180*np.pi
wind_15min['rad']=wind_15min['wind direction']/180*np.pi
wind_30min['rad']=wind_30min['wind direction']/180*np.pi

wind_5min['ws_synthetic']=np.sqrt(np.square(wind_5min['u'])+np.square(wind_5min['v']))
wind_15min['ws_synthetic']=np.sqrt(np.square(wind_15min['u'])+np.square(wind_15min['v']))
wind_30min['ws_synthetic']=np.sqrt(np.square(wind_30min['u'])+np.square(wind_30min['v']))

wind_5min['ws_igor_arrow']=wind_5min['ws_synthetic']*3
wind_15min['ws_igor_arrow']=wind_15min['ws_synthetic']*3
wind_30min['ws_igor_arrow']=wind_30min['ws_synthetic']*3
          
hori_speed_5min=wind_5min.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='ws_synthetic')           
hori_speed_15min=wind_15min.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='ws_synthetic')
hori_speed_30min=wind_30min.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='ws_synthetic')

wind_1min_interval_25 = wind_1min[wind_1min['Altitude [m]'].isin(height)]
wind_5min_interval_25 = wind_5min[wind_5min['Altitude [m]'].isin(height)]
wind_15min_interval_25 = wind_15min[wind_15min['Altitude [m]'].isin(height)]
wind_30min_interval_25 = wind_30min[wind_30min['Altitude [m]'].isin(height)]

wind_1min_interval_25u = wind_1min[wind_1min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='u')
wind_1min_interval_25v = wind_1min[wind_1min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='v')
wind_1min_25_ws=wind_1min[wind_1min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Horizontal Wind Speed [m/s]')
wind_1min_25_wd=wind_1min[wind_1min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Horizontal Wind Direction [癩')
wind_1min_25_vertical=wind_1min[wind_1min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Vertical Wind Speed [m/s]')


wind_5min_interval_25u = wind_5min[wind_5min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='u')
wind_5min_interval_25v = wind_5min[wind_5min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='v')
wind_5min_25_ws=wind_5min[wind_5min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Horizontal Wind Speed [m/s]')
wind_5min_25_wd=wind_5min[wind_5min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='wind direction')
wind_5min_25_vertical=wind_5min[wind_5min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Vertical Wind Speed [m/s]')

wind_15min_interval_25u = wind_15min[wind_15min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='u')
wind_15min_interval_25v = wind_15min[wind_15min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='v')
wind_15min_25_ws=wind_15min[wind_15min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Horizontal Wind Speed [m/s]')
wind_15min_25_wd=wind_15min[wind_15min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='wind direction')
wind_15min_25_vertical=wind_15min[wind_15min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Vertical Wind Speed [m/s]')

wind_30min_interval_25u = wind_30min[wind_30min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='u')
wind_30min_interval_25v = wind_30min[wind_30min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='v')
wind_30min_25_ws=wind_30min[wind_30min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Horizontal Wind Speed [m/s]')
wind_30min_25_wd=wind_30min[wind_30min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='wind direction')
wind_30min_25_vertical=wind_30min[wind_30min['Altitude [m]'].isin(height)].reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Vertical Wind Speed [m/s]')

print("风雷达原始数据处理完毕，开始补齐风场")
###############ERA5补齐到风场##########################
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
ds1 = xr.open_dataset(r"E:\Wuhai Data\wuhai era5\wind complement 202207.nc")
# 定义常量
T0 = 288.15  # 海平面温度 (K)
L = 0.0065  # 温度梯度 (K/m)
P0 = 1013.25  # 海平面标准气压 (mb)
R = 287.05  # 气体常数 (J/(kg·K))
g = 9.801394946924072  # 重力加速度 (m/s²)
#换算单位
ds1['level'] = (T0 / L) * (1 - (ds1['level']/ P0) ** (R * L / g))
# 根据高度排序
ds1 = ds1.sortby('level')
ds = ds1[['u', 'v', 'w']]

# 将时间转换为数值
time_numeric = (ds['time'] - np.datetime64('2022-07-01T00:00:00')) / np.timedelta64(1, 'h')
# 定义目标坐标和高度
target_lat = 39.6583
target_lon = 106.8286   
target_heights = np.linspace(50+elevation, 3500+elevation, num=((3500-50)//25 + 1), endpoint=True)
# 准备插值函数
def interpolate_variable(variable):
    interpolator = RegularGridInterpolator(
        (time_numeric, ds['level'], ds['latitude'], ds['longitude']),
        ds[variable].values,
        method='linear', 
        bounds_error=False,
        fill_value=None
    )
    return interpolator

# 进行插值
results = {var: [] for var in ds.data_vars}

for var in ds.data_vars:
    interpolator = interpolate_variable(var)
    for t in time_numeric:
        interpolated_values = []
        for h in target_heights:
            interpolated_values.append(interpolator((t, h, target_lat, target_lon)))
        results[var].append(interpolated_values)
# 将结果转换为DataFrame
interpolated_data = {}
for var, values in results.items():
    interpolated_data[var] = np.array(values)

for group in interpolated_data:
    globals()[group] = pd.DataFrame(interpolated_data[group])

u_era5=u
v_era5=v
w_era5=w
wind_era5=np.sqrt(np.square(u_era5)+np.square(v_era5))

def central_difference(f, h):
    
    """
    计算函数值数组f的中央差分
    
    参数：
    f -- 2D-Dataframe，表示在等间距点上的函数值
    h -- 步长
    
    返回：
    df -- 一维数组，表示f的中央差分
    """
    n = f.shape[1]
    time_steps = f.shape[0]
    step_index = int(h / 25)#25是空间分辨率
    df = np.zeros((time_steps, n))  
    
    for time in range(np.shape(f)[0]):
        for i in range(0, int(n-h/25)):
            if i>=(int(h/25)) and i <(np.shape(f)[1]-step_index):
                df[time, i] = (f.iloc[time, i + step_index] - f.iloc[time, i - step_index]) / (2 * h)
    df=pd.DataFrame(df)
    df.columns=f.columns
    df=df.iloc[:,step_index:np.shape(f)[1]-step_index]
    df.index = f.index
    return df

def windshear_index(f, h):
    from numpy import log as ln
    """
    计算函数值数组f的中央差分,h=25,则表示50米的变化
    
    参数：
    f -- 2D-Dataframe，表示在等间距点上的函数值
    h -- 步长
    
    返回：
    df -- 一维数组，表示f的中央差分
    """
    n = f.shape[1]
    time_steps = f.shape[0]
    step_index = int(h / 25)#25是空间分辨率
    df = np.zeros((time_steps, n))  
    
    for time in range(np.shape(f)[0]):
        for i in range(0, int(n-h/25)):
            if i>=(int(h/25)) and i <(np.shape(f)[1]-step_index):
                v0 = f.iloc[time, i - step_index]
                v1 = f.iloc[time, i + step_index]
                h0 = f.columns[i - step_index]
                h1 = f.columns[i + step_index]
                df[time, i] = ln(v1 / v0) / ln(h1 / h0)
                #df[time, i] = ln((f.iloc[time, i + step_index] / f.iloc[time, i - step_index]) / (f.columns[i + step_index] / f.columns[i - step_index]))
    df=pd.DataFrame(df)
    df.columns=f.columns
    df=df.iloc[:,step_index:np.shape(f)[1]-step_index]
    df.index = f.index
    return df

#####插值50米以下的高度####
#ground=pd.read_excel("E:\Wuhai Data\\CZsummer_winter.xlsx",sheet_name="summer for igor")
ground_wind_WD = pd.read_excel("E:\Wuhai Data\\乌海气象参数.xlsx",sheet_name="Sheet1").set_index('date')['WD']
ground_wind_WS = pd.read_excel("E:\Wuhai Data\\乌海气象参数.xlsx",sheet_name="Sheet1").set_index('date')['WS']
ground_wind_u = -ground_wind_WS*np.sin(np.radians(ground_wind_WD))
ground_wind_v = -ground_wind_WS*np.cos(np.radians(ground_wind_WD))

common_timestamps_u = wind_60min_interval_25u.index.intersection(ground_wind_u.index)
common_timestamps_v = wind_60min_interval_25v.index.intersection(ground_wind_v.index)

ground_wind_u = ground_wind_u[common_timestamps_u].to_frame()
ground_wind_v = ground_wind_v[common_timestamps_v].to_frame()
ground_wind_u['25']=(ground_wind_u.iloc[:,0]+wind_60min_interval_25u.iloc[:,0])/2
ground_wind_v['25']=(ground_wind_v.iloc[:,0]+wind_60min_interval_25v.iloc[:,0])/2

wind_obs_25u=pd.concat([ground_wind_u, wind_60min_interval_25u], axis=1)
wind_obs_25v=pd.concat([ground_wind_v, wind_60min_interval_25v], axis=1)
wind_obs_25u.columns=wind_obs_25u.columns.astype(int)
wind_obs_25v.columns=wind_obs_25v.columns.astype(int)

##########################################补齐风场-60min#############################################
#先计算原始数据，再平滑，然后求的所有东西都是平滑以后的
time_era5 = ds1['time'].to_dataframe()
time_era5['timestamp'] = time_era5['time'] + pd.Timedelta(hours=8)
time_era5 = time_era5.set_index('timestamp')
time_era5 = time_era5.drop(time_era5.columns[0], axis=1)
time_era5 = time_era5.reset_index()

u_era5['timeseries']=time_era5
u_era5 = u_era5.set_index('timeseries')
u_era5.columns=np.arange(50+elevation, 3501+elevation, 25)

v_era5['timeseries']=time_era5
v_era5 = v_era5.set_index('timeseries')
v_era5.columns=np.arange(50+elevation, 3501+elevation, 25)

w_era5['timeseries']=time_era5
w_era5 = w_era5.set_index('timeseries')
w_era5.columns=np.arange(50+elevation, 3501+elevation, 25)

common_timestamps_u_filled = u_era5.index.intersection(wind_obs_25u.index)
common_timestamps_v_filled = v_era5.index.intersection(wind_obs_25v.index)
common_timestamps_w_filled = w_era5.index.intersection(wind_60min_25_vertical.index)

wind_obs_25u.columns=wind_obs_25u.columns+elevation
wind_obs_25v.columns=wind_obs_25v.columns+elevation
wind_60min_25_vertical.columns=wind_60min_25_vertical.columns+elevation
###########1h补齐###########
wind_filled_u_60min = wind_obs_25u.loc[common_timestamps_u_filled].combine_first(u_era5.loc[common_timestamps_u_filled])
#wind_filled_u_60min = pd.concat([wind_obs_25u.loc[common_timestamps_u_filled,0+elevation:25+elevation], wind_filled_u_60min.loc[common_timestamps_u_filled]], axis=1)

wind_filled_v_60min = wind_obs_25v.loc[common_timestamps_v_filled].combine_first(v_era5.loc[common_timestamps_v_filled])
#wind_filled_v_60min = pd.concat([wind_obs_25v.loc[common_timestamps_v_filled,0+elevation:25+elevation], wind_filled_v_60min.loc[common_timestamps_v_filled]], axis=1)

wd_filled_60min = np.mod(180 + np.degrees(np.arctan2(wind_filled_u_60min, wind_filled_v_60min)), 360)

wind_filled_w_60min = wind_60min_25_vertical.loc[common_timestamps_w_filled].combine_first(w_era5.loc[common_timestamps_w_filled])
#wind_filled_w_60min = pd.concat([wind_60min_25_vertical.loc[common_timestamps_w_filled,0+elevation:25+elevation], wind_filled_w_60min.loc[common_timestamps_w_filled]], axis=1)
wind_filled_w_60min = wind_filled_w_60min.applymap(lambda x: np.nan if abs(x) > 4.5 else x).interpolate(method='linear')

horizontal_speed_filled_60min=np.sqrt(np.square(wind_filled_u_60min)+np.square(wind_filled_v_60min))
u_60min_origin=wind_filled_u_60min.copy()
v_60min_origin=wind_filled_v_60min.copy()

from statsmodels.nonparametric.smoothers_lowess import lowess
def loess_smoothing(data, frac=0.13, window_size=5):
    """
    使用LOESS对每个时间点在不同高度的数据进行平滑
    """
    smoothed_data = np.empty_like(data)
    for i in range(data.shape[0]):  # 遍历每个时间点
        y = data[i, :]
        x = np.arange(len(y))
        smoothed_y = lowess(y, x, frac=frac, return_sorted=False)
        smoothed_y[smoothed_y < 0] = np.nan  # Set negative values to NaN
        smoothed_data[i, :] = smoothed_y
    
    # Fill NaN values using moving average
    for i in range(smoothed_data.shape[0]):
        series = pd.Series(smoothed_data[i, :])
        filled_series = series.fillna(series.rolling(window=window_size, min_periods=1, center=True).mean())
        smoothed_data[i, :] = filled_series.values
    
    
    return smoothed_data

# 使用LOESS进行平滑处理
smoothed_data = loess_smoothing(horizontal_speed_filled_60min.values)

# 将平滑后的数据转换回DataFrame
smoothed_hori_speed_60min = pd.DataFrame(smoothed_data, index=horizontal_speed_filled_60min.index, columns=horizontal_speed_filled_60min.columns)
wind_obs=np.sqrt(np.square(ground_wind_u)+np.square(ground_wind_v))
smoothed_hori_speed_60min.iloc[:,0:2] = wind_obs

wind_filled_u_60min = smoothed_hori_speed_60min * -np.sin(np.radians(wd_filled_60min))
wind_filled_v_60min = smoothed_hori_speed_60min * -np.cos(np.radians(wd_filled_60min))

WDS_50m_60min=windshear_index(smoothed_hori_speed_60min, 25)
WDS_100m_60min=windshear_index(smoothed_hori_speed_60min, 50)
WDS_150m_60min=windshear_index(smoothed_hori_speed_60min, 75)
#计算风矢量切变指标
wind_60min_Δu_Δz_50m = central_difference(wind_filled_u_60min, 25)
wind_60min_Δv_Δz_50m = central_difference(wind_filled_v_60min, 25)
windshear_60min_50m=np.sqrt(np.square(wind_60min_Δu_Δz_50m)+np.square(wind_60min_Δv_Δz_50m))

wind_60min_Δu_Δz_100m = central_difference(wind_filled_u_60min, 50)
wind_60min_Δv_Δz_100m = central_difference(wind_filled_v_60min, 50)
windshear_60min_100m=np.sqrt(np.square(wind_60min_Δu_Δz_100m)+np.square(wind_60min_Δv_Δz_100m))

wind_60min_Δu_Δz_150m = central_difference(wind_filled_u_60min, 75)
wind_60min_Δv_Δz_150m = central_difference(wind_filled_v_60min, 75)
windshear_60min_150m=np.sqrt(np.square(wind_60min_Δu_Δz_150m)+np.square(wind_60min_Δv_Δz_150m))

#输出Igor长表
u_long=pd.melt(wind_filled_u_60min.reset_index(), id_vars=['index'], var_name="height", value_name='u').set_index('index')
v_long=pd.melt(wind_filled_v_60min.reset_index(), id_vars=['index'], var_name="height", value_name='v').set_index('index')
w_long=pd.melt(wind_filled_w_60min.reset_index(), id_vars=['index'], var_name="height", value_name='w').set_index('index')
merged_df = pd.merge(u_long, v_long, on=['index', 'height'], how='outer')
Wind_matrix_60min = pd.merge(merged_df, w_long, on=['index', 'height'], how='outer')
Wind_matrix_60min['WD_igor']=np.arctan2(Wind_matrix_60min['v'],Wind_matrix_60min['u'])
Wind_matrix_60min['wind direction']=np.mod(180 + np.degrees(np.arctan2(Wind_matrix_60min['u'], Wind_matrix_60min['v'])), 360)
Wind_matrix_60min['ws_synthetic']=np.sqrt(np.square(Wind_matrix_60min['u'])+np.square(Wind_matrix_60min['v']))
Wind_matrix_60min['igor_arrow'] = Wind_matrix_60min['ws_synthetic'].apply(lambda x: min(x * 3, 30))
Wind_matrix_60min['height'] = Wind_matrix_60min['height'] -elevation

ws_LOESS_smooth = pd.melt(smoothed_hori_speed_60min.reset_index(), id_vars=['index'], var_name="height", value_name='ws_LOESS_smooth').set_index('index')
ws_LOESS_smooth['height'] = ws_LOESS_smooth['height'] -elevation
Wind_matrix_60min['ws_LOESS_smooth'] = ws_LOESS_smooth['ws_LOESS_smooth']
Wind_matrix_60min['igor_arrow_smooth'] = Wind_matrix_60min['ws_LOESS_smooth'].apply(lambda x: min(x * 3, 30))
Wind_matrix_60min['u_smooth'] = -ws_LOESS_smooth['ws_LOESS_smooth']*np.sin(np.radians(Wind_matrix_60min['WD_igor']))
Wind_matrix_60min['v_smooth'] = -ws_LOESS_smooth['ws_LOESS_smooth']*np.cos(np.radians(Wind_matrix_60min['WD_igor']))


################5min补齐的插值######################
buqi_timestamps_5min=pd.date_range(start=Wind_matrix_60min.index.min(), end=Wind_matrix_60min.index.max(), freq='5T')

#地面观测插值成5min
ground_wind_u_interp_5min = ground_wind_u.reindex(buqi_timestamps_5min).interpolate(method='linear')
ground_wind_v_interp_5min = ground_wind_v.reindex(buqi_timestamps_5min).interpolate(method='linear')
ground_wind_u_interp_5min['25']=(ground_wind_u_interp_5min.iloc[:,0]+wind_5min_interval_25u.iloc[:,0])/2
ground_wind_v_interp_5min['25']=(ground_wind_v_interp_5min.iloc[:,0]+wind_5min_interval_25v.iloc[:,0])/2
wind_obs_25u=pd.concat([ground_wind_u_interp_5min, wind_5min_interval_25u], axis=1)
wind_obs_25v=pd.concat([ground_wind_v_interp_5min, wind_5min_interval_25v], axis=1)
wind_obs_25u.columns=wind_obs_25u.columns.astype(int)
wind_obs_25v.columns=wind_obs_25v.columns.astype(int)

u_era5_interp_5min = u_era5.reindex(buqi_timestamps_5min).interpolate(method='linear')
v_era5_interp_5min = v_era5.reindex(buqi_timestamps_5min).interpolate(method='linear')
w_era5_interp_5min = w_era5.reindex(buqi_timestamps_5min).interpolate(method='linear')

common_timestamps_u_filled=u_era5_interp_5min.index.intersection(wind_obs_25u.index)
common_timestamps_v_filled=v_era5_interp_5min.index.intersection(wind_obs_25v.index)
common_timestamps_w_filled=w_era5_interp_5min.index.intersection(wind_5min_25_vertical.index)

all_columns = wind_obs_25u.columns.intersection(u_era5_interp_5min.columns)
wind_obs_25u.columns = wind_obs_25u.columns + elevation
wind_obs_25v.columns = wind_obs_25v.columns + elevation
wind_filled_u_5min=wind_obs_25u.loc[common_timestamps_u_filled].combine_first(u_era5_interp_5min.loc[common_timestamps_u_filled])
wind_filled_u_5min=pd.concat([wind_obs_25u.loc[common_timestamps_u_filled,0:25], wind_filled_u_5min.loc[common_timestamps_u_filled]], axis=1)

wind_filled_v_5min=wind_obs_25v.loc[common_timestamps_v_filled].combine_first(v_era5_interp_5min.loc[common_timestamps_v_filled])
wind_filled_v_5min=pd.concat([wind_obs_25v.loc[common_timestamps_v_filled,0:25], wind_filled_v_5min.loc[common_timestamps_v_filled]], axis=1)

wind_5min_25_vertical.columns = wind_5min_25_vertical.columns + elevation
wind_filled_w_5min=wind_5min_25_vertical.loc[common_timestamps_w_filled].combine_first(w_era5_interp_5min.loc[common_timestamps_w_filled])
#wind_filled_w=pd.concat([wind_5min_25_vertical.loc[common_timestamps_w_filled,0:25], wind_filled_w.loc[common_timestamps_w_filled]], axis=1)
wind_filled_w_5min=wind_filled_w_5min.applymap(lambda x: np.nan if abs(x) > 4.5 else x).interpolate(method='linear')
wd_filled_5min = np.mod(180 + np.degrees(np.arctan2(wind_filled_u_5min, wind_filled_v_5min)), 360)

horizontal_speed_filled_5min=np.sqrt(np.square(wind_filled_u_5min)+np.square(wind_filled_v_5min))
u_5min_origin=wind_filled_u_5min.copy()
v_5min_origin=wind_filled_v_5min.copy()

# 使用LOESS进行平滑处理
smoothed_data = loess_smoothing(horizontal_speed_filled_5min.values)

# 将平滑后的数据转换回DataFrame
smoothed_hori_speed_5min = pd.DataFrame(smoothed_data, index=horizontal_speed_filled_5min.index, columns=horizontal_speed_filled_5min.columns)
wind_obs=np.sqrt(np.square(ground_wind_u_interp_5min)+np.square(ground_wind_v_interp_5min))
smoothed_hori_speed_5min.iloc[:,0:2] = wind_obs

wind_filled_u_5min = smoothed_hori_speed_5min * -np.sin(np.radians(wd_filled_5min))
wind_filled_v_5min = smoothed_hori_speed_5min * -np.cos(np.radians(wd_filled_5min))

#计算风切指数指标

WDS_50m_5min=windshear_index(smoothed_hori_speed_5min, 25)
WDS_100m_5min=windshear_index(smoothed_hori_speed_5min, 50)
WDS_150m_5min=windshear_index(smoothed_hori_speed_5min, 75)

#计算风矢量切变指标
wind_5min_Δu_Δz_50m = central_difference(wind_filled_u_5min, 25)
wind_5min_Δv_Δz_50m = central_difference(wind_filled_v_5min, 25)
windshear_5min_50m=np.sqrt(np.square(wind_5min_Δu_Δz_50m)+np.square(wind_5min_Δv_Δz_50m))

wind_5min_Δu_Δz_100m = central_difference(wind_filled_u_5min, 50)
wind_5min_Δv_Δz_100m = central_difference(wind_filled_v_5min, 50)
windshear_5min_100m=np.sqrt(np.square(wind_5min_Δu_Δz_100m)+np.square(wind_5min_Δv_Δz_100m))

wind_5min_Δu_Δz_150m = central_difference(wind_filled_u_5min, 75)
wind_5min_Δv_Δz_150m = central_difference(wind_filled_v_5min, 75)
windshear_5min_150m=np.sqrt(np.square(wind_5min_Δu_Δz_150m)+np.square(wind_5min_Δv_Δz_150m))

u_long=pd.melt(wind_filled_u_5min.reset_index(), id_vars=['index'], var_name="height", value_name='u').set_index('index')
v_long=pd.melt(wind_filled_v_5min.reset_index(), id_vars=['index'], var_name="height", value_name='v').set_index('index')
w_long=pd.melt(wind_filled_w_5min.reset_index(), id_vars=['index'], var_name="height", value_name='w').set_index('index')

merged_df = pd.concat([u_long, v_long], axis=1)
merged_df = merged_df.iloc[:,1:4]

Wind_matrix_5min = pd.merge(merged_df, w_long, on=['index', 'height'], how='outer')
Wind_matrix_5min['WD_igor']=np.arctan2(Wind_matrix_5min['v'],Wind_matrix_5min['u'])
Wind_matrix_5min['wind direction']=np.mod(180 + np.arctan2(Wind_matrix_5min['u'], Wind_matrix_5min['v']) * 180 / np.pi, 360)
Wind_matrix_5min['ws_synthetic']=np.sqrt(np.square(Wind_matrix_5min['u'])+np.square(Wind_matrix_5min['v']))
Wind_matrix_5min['igor_arrow'] = Wind_matrix_5min['ws_synthetic'].apply(lambda x: min(x * 3, 30))

ws_LOESS_smooth = pd.melt(smoothed_hori_speed_5min.reset_index(), id_vars=['index'], var_name="height", value_name='ws_LOESS_smooth').set_index('index')
ws_LOESS_smooth['height'] = ws_LOESS_smooth['height']
Wind_matrix_5min['ws_LOESS_smooth'] = ws_LOESS_smooth['ws_LOESS_smooth']
Wind_matrix_5min['igor_arrow_smooth'] = Wind_matrix_5min['ws_LOESS_smooth'].apply(lambda x: min(x * 3, 30))
Wind_matrix_5min['u_smooth'] = -ws_LOESS_smooth['ws_LOESS_smooth']*np.sin(np.radians(Wind_matrix_5min['WD_igor']))
Wind_matrix_5min['v_smooth'] = -ws_LOESS_smooth['ws_LOESS_smooth']*np.cos(np.radians(Wind_matrix_5min['WD_igor']))



################20min补齐的插值######################
buqi_timestamps_20min=pd.date_range(start=Wind_matrix_60min.index.min(), end=Wind_matrix_60min.index.max(), freq='20T')

ground_wind_u_interp_20min = ground_wind_u.reindex(buqi_timestamps_20min).interpolate(method='linear')
ground_wind_v_interp_20min = ground_wind_v.reindex(buqi_timestamps_20min).interpolate(method='linear')
wind_obs_25u=pd.concat([ground_wind_u_interp_20min, wind_20min_interval_25u], axis=1)
wind_obs_25v=pd.concat([ground_wind_v_interp_20min, wind_20min_interval_25v], axis=1)
wind_obs_25u.columns=wind_obs_25u.columns.astype(int)
wind_obs_25v.columns=wind_obs_25v.columns.astype(int)
wind_obs_25u.columns = wind_obs_25u.columns + elevation
wind_obs_25v.columns = wind_obs_25v.columns + elevation

u_era5_interp_20min = u_era5.reindex(buqi_timestamps_20min).interpolate(method='linear')
v_era5_interp_20min = v_era5.reindex(buqi_timestamps_20min).interpolate(method='linear')
w_era5_interp_20min = w_era5.reindex(buqi_timestamps_20min).interpolate(method='linear')

common_timestamps_u_filled=u_era5_interp_20min.index.intersection(wind_obs_25u.index)
common_timestamps_v_filled=v_era5_interp_20min.index.intersection(wind_obs_25v.index)
common_timestamps_w_filled=w_era5_interp_20min.index.intersection(wind_20min_25_vertical.index)
all_columns = wind_obs_25u.columns.intersection(u_era5_interp_20min.columns)

wind_filled_u_20min=wind_obs_25u.loc[common_timestamps_u_filled].combine_first(u_era5_interp_20min.loc[common_timestamps_u_filled])
wind_filled_u_20min=pd.concat([wind_obs_25u.loc[common_timestamps_u_filled,0:25], wind_filled_u_20min.loc[common_timestamps_u_filled]], axis=1)

wind_filled_v_20min=wind_obs_25v.loc[common_timestamps_v_filled].combine_first(v_era5_interp_20min.loc[common_timestamps_v_filled])
wind_filled_v_20min=pd.concat([wind_obs_25v.loc[common_timestamps_v_filled,0:25], wind_filled_v_20min.loc[common_timestamps_v_filled]], axis=1)

wind_20min_25_vertical.columns = wind_20min_25_vertical.columns + elevation
wind_filled_w_20min=wind_20min_25_vertical.loc[common_timestamps_w_filled].combine_first(w_era5_interp_20min.loc[common_timestamps_w_filled])
#wind_filled_w=pd.concat([wind_5min_25_vertical.loc[common_timestamps_w_filled,0:25], wind_filled_w.loc[common_timestamps_w_filled]], axis=1)
wind_filled_w_20min=wind_filled_w_20min.applymap(lambda x: np.nan if abs(x) > 4.5 else x).interpolate(method='linear')

wd_filled_20min = np.mod(180 + np.degrees(np.arctan2(wind_filled_u_20min, wind_filled_v_20min)), 360)

#计算风切指数指标
horizontal_speed_filled_20min=np.sqrt(np.square(wind_filled_u_20min)+np.square(wind_filled_v_20min))
u_20min_origin=wind_filled_u_20min.copy()
v_20min_origin=wind_filled_v_20min.copy()

# 使用LOESS进行平滑处理
smoothed_data = loess_smoothing(horizontal_speed_filled_20min.values)

# 将平滑后的数据转换回DataFrame
smoothed_hori_speed_20min = pd.DataFrame(smoothed_data, index=horizontal_speed_filled_20min.index, columns=horizontal_speed_filled_20min.columns)
wind_obs=np.sqrt(np.square(ground_wind_u_interp_20min)+np.square(ground_wind_v_interp_20min))
smoothed_hori_speed_20min.iloc[:,0:2] = wind_obs

wind_filled_u_20min = smoothed_hori_speed_20min * -np.sin(np.radians(wd_filled_20min))
wind_filled_v_20min = smoothed_hori_speed_20min * -np.cos(np.radians(wd_filled_20min))

WDS_50m_20min=windshear_index(smoothed_hori_speed_20min, 25)
WDS_100m_20min=windshear_index(smoothed_hori_speed_20min, 50)
WDS_150m_20min=windshear_index(smoothed_hori_speed_20min, 75)
#计算风矢量切变指标
wind_20min_Δu_Δz_50m = central_difference(wind_filled_u_20min, 25)
wind_20min_Δv_Δz_50m = central_difference(wind_filled_v_20min, 25)
windshear_20min_50m=np.sqrt(np.square(wind_20min_Δu_Δz_50m)+np.square(wind_20min_Δv_Δz_50m))

wind_20min_Δu_Δz_100m = central_difference(wind_filled_u_20min, 50)
wind_20min_Δv_Δz_100m = central_difference(wind_filled_v_20min, 50)
windshear_20min_100m=np.sqrt(np.square(wind_20min_Δu_Δz_100m)+np.square(wind_20min_Δv_Δz_100m))

wind_20min_Δu_Δz_150m = central_difference(wind_filled_u_20min, 75)
wind_20min_Δv_Δz_150m = central_difference(wind_filled_v_20min, 75)
windshear_20min_150m=np.sqrt(np.square(wind_20min_Δu_Δz_150m)+np.square(wind_20min_Δv_Δz_150m))

u_long=pd.melt(wind_filled_u_20min.reset_index(), id_vars=['index'], var_name="height", value_name='u').set_index('index')
v_long=pd.melt(wind_filled_v_20min.reset_index(), id_vars=['index'], var_name="height", value_name='v').set_index('index')
w_long=pd.melt(wind_filled_w_20min.reset_index(), id_vars=['index'], var_name="height", value_name='w').set_index('index')

merged_df = pd.merge(u_long, v_long, on=['index', 'height'], how='outer')

merged_df = pd.concat([u_long, v_long], axis=1)
merged_df = merged_df.iloc[:,1:4]

Wind_matrix_20min = pd.merge(merged_df, w_long, on=['index', 'height'], how='outer')
Wind_matrix_20min['WD_igor']=np.arctan2(Wind_matrix_20min['v'],Wind_matrix_20min['u'])
Wind_matrix_20min['wind direction']=np.mod(180 + np.arctan2(Wind_matrix_20min['u'], Wind_matrix_20min['v']) * 180 / np.pi, 360)
Wind_matrix_20min['ws_synthetic']=np.sqrt(np.square(Wind_matrix_20min['u'])+np.square(Wind_matrix_20min['v']))
Wind_matrix_20min['igor_arrow'] = Wind_matrix_20min['ws_synthetic'].apply(lambda x: min(x * 3, 30))

ws_LOESS_smooth = pd.melt(smoothed_hori_speed_20min.reset_index(), id_vars=['index'], var_name="height", value_name='ws_LOESS_smooth').set_index('index')
ws_LOESS_smooth['height'] = ws_LOESS_smooth['height'] -elevation
Wind_matrix_20min['ws_LOESS_smooth'] = ws_LOESS_smooth['ws_LOESS_smooth']
Wind_matrix_20min['igor_arrow_smooth'] = Wind_matrix_20min['ws_LOESS_smooth'].apply(lambda x: min(x * 3, 30))
Wind_matrix_20min['u_smooth'] = -ws_LOESS_smooth['ws_LOESS_smooth']*np.sin(np.radians(Wind_matrix_20min['WD_igor']))
Wind_matrix_20min['v_smooth'] = -ws_LOESS_smooth['ws_LOESS_smooth']*np.cos(np.radians(Wind_matrix_20min['WD_igor']))



# 使用LOESS进行平滑处理
smoothed_data = loess_smoothing(horizontal_speed_filled_20min.values)
# 将平滑后的数据转换回DataFrame
smoothed_hori_speed_20min = pd.DataFrame(smoothed_data, index=horizontal_speed_filled_20min.index, columns=horizontal_speed_filled_20min.columns)
wind_obs=np.sqrt(np.square(ground_wind_u)+np.square(ground_wind_v))
ws_LOESS_smooth = pd.melt(smoothed_hori_speed_20min.reset_index(), id_vars=['index'], var_name="height", value_name='ws_LOESS_smooth').set_index('index')
ws_LOESS_smooth['height'] = ws_LOESS_smooth['height'] - elevation
Wind_matrix_20min['ws_LOESS_smooth'] = ws_LOESS_smooth['ws_LOESS_smooth']
Wind_matrix_20min['igor_arrow_smooth'] = Wind_matrix_20min['ws_LOESS_smooth'].apply(lambda x: min(x * 3, 30))





#############风相关参数###########

wind_5min_std = (wind_5min.groupby(['Altitude [m]', pd.Grouper(key='timestamp', freq='5min')]).std()).reset_index()
wind_20min_std = (wind_20min.groupby(['Altitude [m]', pd.Grouper(key='timestamp', freq='20min')]).std()).reset_index()
wind_60min_std = (wind_60min.groupby(['Altitude [m]', pd.Grouper(key='timestamp', freq='60min')]).std()).reset_index()
# Calculate TKE wind_1min[['u','v','Altitude [m]','wind direction','Vertical Wind Speed [m/s]']]
u_TKE_5min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='u').resample('5T').std()
v_TKE_5min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='v').resample('5T').std()
w_TKE_5min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Vertical Wind Speed [m/s]').resample('5T').std()
TKE_5min=(u_TKE_5min+v_TKE_5min+w_TKE_5min)/2

u_TKE_20min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='u').resample('20T').std()
v_TKE_20min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='v').resample('20T').std()
w_TKE_20min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='Vertical Wind Speed [m/s]').resample('20T').std()
TKE_20min=(u_TKE_20min+v_TKE_20min+w_TKE_20min)/2

u_TKE_60min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='u_2').resample('60T').std()
v_TKE_60min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='v_2').resample('60T').std()
w_TKE_60min=wind_sec.reset_index().pivot(index='timestamp', columns='Altitude [m]', values='w_2').resample('60T').std()
TKE_60min=(np.square(u_TKE_60min)+np.square(v_TKE_60min)+np.square(w_TKE_60min))/2

TKE_5min = TKE_5min.loc[(TKE_5min.index >= "2022-07-02 00:00:00") & (TKE_5min.index <= "2022-07-31 23:00:00")]
TKE_20min = TKE_20min.loc[(TKE_20min.index >= "2022-07-02 00:00:00") & (TKE_20min.index <= "2022-07-31 23:00:00")]
TKE_60min = TKE_60min.loc[(TKE_60min.index >= "2022-07-02 00:00:00") & (TKE_60min.index <= "2022-07-31 23:00:00")]
print("风雷达数据再分析补齐处理完毕")

######### Raw MWR data process##############
folder_path = 'E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\temp\\'  # 指定路径
temp = pd.read_excel(folder_path+'temp.xlsx')
folder_path = 'E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\rh\\'  # 指定路径
rh = pd.read_excel(folder_path+'rh.xlsx')
rh['bjt'] = pd.to_datetime(rh['bjt'])
rh.set_index('bjt', inplace=True)

ah=pd.read_excel(folder_path+'ah.xlsx')
ah['bjt'] = pd.to_datetime(ah['bjt'])
ah.set_index('bjt', inplace=True)

temp['bjt'] = pd.to_datetime(temp['bjt'])
temp.set_index('bjt', inplace=True)

start_time = '2022-06-30 08:30:00'
end_time = '2022-08-01 07:30:00'

temp_1min_mean =  temp.resample('1T').mean().reindex(pd.date_range(start=temp.resample('1T').mean().index.min(), end=temp.resample('1T').mean().index.max(), freq='1T')).interpolate(method='time')
temp_5min_mean =  temp.resample('5T').mean().reindex(pd.date_range(start=temp.resample('5T').mean().index.min(), end=temp.resample('5T').mean().index.max(), freq='5T')).interpolate(method='time')
temp_20min_mean =  temp.resample('20T').mean().reindex(pd.date_range(start=temp.resample('20T').mean().index.min(), end=temp.resample('20T').mean().index.max(), freq='20T')).interpolate(method='time')
temp_60min_mean =  temp.resample('60T').mean().reindex(pd.date_range(start=temp.resample('60T').mean().index.min(), end=temp.resample('60T').mean().index.max(), freq='60T'))

height_temp=[col for col in temp_1min_mean.columns if isinstance(col, int)]

rh_1min_mean =  rh.resample('1T').mean().reindex(pd.date_range(start=rh.resample('1T').mean().index.min(), end=rh.resample('1T').mean().index.max(), freq='1T')).interpolate(method='time')
rh_5min_mean =  rh.resample('5T').mean().reindex(pd.date_range(start=rh.resample('5T').mean().index.min(), end=rh.resample('5T').mean().index.max(), freq='5T')).interpolate(method='time')
rh_20min_mean =  rh.resample('20T').mean().reindex(pd.date_range(start=rh.resample('20T').mean().index.min(), end=rh.resample('20T').mean().index.max(), freq='20T')).interpolate(method='time')
rh_60min_mean =  rh.resample('60T').mean().reindex(pd.date_range(start=rh.resample('60T').mean().index.min(), end=rh.resample('60T').mean().index.max(), freq='60T')).interpolate(method='time')

ah_5min_mean =  ah.resample('5T').mean().reindex(pd.date_range(start=ah.resample('5T').mean().index.min(), end=ah.resample('5T').mean().index.max(), freq='5T')).interpolate(method='time')
ah_20min_mean =  ah.resample('20T').mean().reindex(pd.date_range(start=ah.resample('20T').mean().index.min(), end=ah.resample('20T').mean().index.max(), freq='20T')).interpolate(method='time')
ah_60min_mean =  ah.resample('60T').mean().reindex(pd.date_range(start=ah.resample('60T').mean().index.min(), end=ah.resample('60T').mean().index.max(), freq='60T')).interpolate(method='time')

def goff_gratch_formula_dataframe(df):
    def goff_gratch_formula(T):
        T_ref = 373.15  # Reference temperature in Kelvin (100°C)
        log10_es = (
            -7.90298 * (T_ref / T - 1) +
            5.02808 * math.log10(T_ref / T) -
            1.3816e-7 * (10**(11.344 * (1 - T / T_ref)) - 1) +
            8.1328e-3 * (10**(-3.49149 * (T_ref / T - 1)) - 1) +
            math.log10(1013.246)
        )
        es = 10**log10_es
        return es
    
    # Apply the Goff-Gratch formula to each element in the DataFrame
    es_df = df.applymap(goff_gratch_formula)
    return es_df

es_1min = goff_gratch_formula_dataframe(temp_1min_mean)
es_5min = goff_gratch_formula_dataframe(temp_5min_mean)
es_20min = goff_gratch_formula_dataframe(temp_20min_mean)
es_60min = goff_gratch_formula_dataframe(temp_60min_mean)


folder_path = 'E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\met'  # 指定路径
all_data = []  # 创建一个空列表以存储数据帧

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)  # 获取文件完整路径
    # 使用pandas读取文件，跳过前8行
    data = pd.read_table(file_path, skiprows=19,delimiter=',', header=None, encoding='ANSI')
    all_data.append(data)  # 将数据帧添加到列表中
merged_data = pd.concat(all_data, ignore_index=True)

met=merged_data.iloc[:,:10]

met.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second','rain_flag','pressure','temperature','RH']
met['Year'] = met['Year'].apply(lambda x: 2000 + x)

# 将两位数年份转换为四位数年份
# 创建一个新的列来存储时间戳
met['timestamp'] = pd.to_datetime(met[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
met['timestamp'] = met['timestamp'] + pd.Timedelta(hours=8)

df1_met = pd.DataFrame(met)
df1_met['timestamp'] = pd.to_datetime(df1_met['timestamp'])
df1_met.set_index('timestamp', inplace=True)
met_1min=df1_met.resample('1T').mean().interpolate()
met_5min=met_1min.resample('5T').mean()
met_20min=met_1min.resample('20T').mean()
met_60min=met_1min.resample('60T').mean()

#找压高公式第一层，初始值
common_timestamps_1min = met_1min.index.intersection(temp_1min_mean.index)
common_timestamps_5min = met_5min.index.intersection(temp_5min_mean.index)
common_timestamps_20min = met_20min.index.intersection(temp_20min_mean.index)
common_timestamps_60min = met_60min.index.intersection(temp_60min_mean.index)

# 定义计算气压的函数

L = 0.0065  # 温度梯度 (K/m)
g = 9.801394946924072  # 重力加速度 (m/s^2)
M = 0.0289644  # 空气平均摩尔质量 (kg/mol)
R = 8.3144598  # 气体常数 (J/(mol·K))

def calculate_pressure_series(P0_series, T0_series, heights):
    """
    根据时间序列的海平面处的压强 P0_series 和温度 T0_series，计算给定高度数组 heights 处的压强时间序列。
    
    P0_series: 海平面处的压强时间序列 (Pa)
    T0_series: 海平面处的绝对温度时间序列 (K)
    heights: 高度数组 (m)
    """
    pressures = np.zeros((len(P0_series), len(heights)))
    
    for i in range(len(P0_series)):
        P0 = P0_series[i]
        T0 = T0_series[i]
        pressures[i, :] = P0 * (1 - (L * heights) / T0) ** (g * M / (R * L))
        
    return pressures

heights = np.array(height_temp)  # 高度数组 (m)

p_60min = pd.DataFrame(calculate_pressure_series(met_60min['pressure'].to_numpy(), met_60min['temperature'].values, heights), 
                       index=met_60min.index, columns=heights)
p_20min = pd.DataFrame(calculate_pressure_series(met_20min['pressure'].to_numpy(), met_20min['temperature'].values, heights), 
                       index=met_20min.index, columns=heights)
p_5min = pd.DataFrame(calculate_pressure_series(met_5min['pressure'].to_numpy(), met_5min['temperature'].values, heights), 
                       index=met_5min.index, columns=heights)
p_1min = pd.DataFrame(calculate_pressure_series(met_1min['pressure'].to_numpy(), met_1min['temperature'].values, heights), 
                       index=met_1min.index, columns=heights)

e_1min=np.multiply(es_1min.loc[common_timestamps_1min] , rh_1min_mean.loc[common_timestamps_1min]/100)
e_5min=np.multiply(es_5min.loc[common_timestamps_5min] , rh_5min_mean.loc[common_timestamps_5min]/100)
e_20min=np.multiply(es_20min.loc[common_timestamps_20min] , rh_20min_mean.loc[common_timestamps_20min]/100)
e_60min=np.multiply(es_60min.loc[common_timestamps_60min] , rh_60min_mean.loc[common_timestamps_60min]/100)

q_1min=0.622*e_1min/(p_1min-0.378*e_1min)
q_5min=0.622*e_5min/(p_5min-0.378*e_5min)
q_20min=0.622*e_20min/(p_20min-0.378*e_20min)
q_60min=0.622*e_60min/(p_60min-0.378*e_60min)

θ_1min=temp_1min_mean.loc[common_timestamps_1min]*(1000/p_1min) ** 0.286
θ_5min=temp_5min_mean.loc[common_timestamps_5min]*(1000/p_5min) ** 0.286
θ_20min=temp_20min_mean.loc[common_timestamps_20min]*(1000/p_20min) ** 0.286
θ_60min=temp_60min_mean.loc[common_timestamps_60min]*(1000/p_60min) ** 0.286

θv_1min=temp_1min_mean.loc[common_timestamps_1min]*(1+0.608*q_1min)*(1000/p_1min) ** 0.286
θv_5min=temp_5min_mean.loc[common_timestamps_5min]*(1+0.608*q_5min)*(1000/p_5min) ** 0.286
θv_20min=temp_20min_mean.loc[common_timestamps_20min]*(1+0.608*q_20min)*(1000/p_20min) ** 0.286
θv_60min=temp_60min_mean.loc[common_timestamps_60min]*(1+0.608*q_60min)*(1000/p_60min) ** 0.286

height_θv=[col for col in θv_1min.columns if isinstance(col, int)]

#插值到25米间隔的高度计算边界层
custom_heights = np.arange(0, 10001, 25)  # 替换为你的实际自定义高度
original_heights = height_θv  # 替换为你的实际原始高度

# 准备插值结果的列名
interpolated_columns = ['time'] + [f'{int(h)}' for h in custom_heights]

# 存储每个数据集的插值结果的字典,逐个注释运行不然会死机
datasets = {
    'θv_1min_interval25': θv_1min,
    #'θv_2min': θv_2min,
    'θv_5min_interval25': θv_5min,
    #'θv_10min': θv_10min,
    #'θv_15min': θv_15min,
    'θv_20min_interval25': θv_20min,
    #'θv_30min': θv_30min
    'θv_60min_interval25': θv_60min
}

interpolated_data_dict = {}
# 假设有多个数据集需要插值（每个数据集是一个DataFrame存储在列表中）

for key, data in datasets.items():
    # 用于存储当前数据集所有行的列表
    interpolated_rows = []

    # 当前数据集的插值过程
    for time, row in data.iterrows():
        temperatures = row.values  # 获取温度值

        # 创建插值函数，使用三次插值
        interp_func = interp1d(original_heights, temperatures, kind='cubic', bounds_error=False, fill_value='extrapolate')

        # 执行插值
        new_temps = interp_func(custom_heights)

        # 替换自定义高度与原始高度匹配的插值值
        for j, h in enumerate(custom_heights):
            if h in original_heights:
                original_index = np.where(original_heights == h)[0][0]
                new_temps[j] = temperatures[original_index]

        # 构建新行并添加到列表
        new_row = [time] + list(new_temps)
        interpolated_rows.append(new_row)
    print(key)
    # 从列表中一步创建DataFrame，并设置时间戳为索引
    interpolated_data = pd.DataFrame(interpolated_rows, columns=interpolated_columns)
    interpolated_data.set_index('time', inplace=True)
    
    # 将插值结果存储在字典中
    interpolated_data_dict[key] = interpolated_data

# 为每个插值结果重命名
renamed_data_dict = {f"{key}": df for key, df in interpolated_data_dict.items()}
interpolated_θv_1min_25 = renamed_data_dict['θv_1min_interval25']
interpolated_θv_5min_25 = renamed_data_dict['θv_5min_interval25']
interpolated_θv_20min_25 = renamed_data_dict['θv_20min_interval25']
interpolated_θv_60min_25 = renamed_data_dict['θv_60min_interval25']

###############Richardson Bulk###############

def central_delta(f, h):
    
    """
    计算函数值数组f的中央差分
    
    参数：
    f -- 2D-Dataframe，表示在等间距点上的函数值
    h -- 步长
    
    返回：
    df -- 一维数组，表示f的中央差分
    """
    n = f.shape[1]
    time_steps = f.shape[0]
    step_index = int(h / 25)#25是空间分辨率
    df = np.zeros((time_steps, n))  
    
    for time in range(np.shape(f)[0]):
        for i in range(0, int(n-h/25)):
            if i>=(int(h/25)) and i <(np.shape(f)[1]-step_index):
                df[time, i] = (f.iloc[time, i + step_index] - f.iloc[time, i - step_index])
    df=pd.DataFrame(df)
    df.columns=f.columns
    df=df.iloc[:,step_index:np.shape(f)[1]-step_index]
    df.index = f.index
    return df


# h=25

# windshear_60min_50m=np.square(central_delta(u_60min_origin, h))+np.square(central_delta(v_60min_origin, h))
# windshear_20min_50m=np.square(central_delta(u_20min_origin, h))+np.square(central_delta(v_20min_origin, h))
# windshear_5min_50m=np.square(central_delta(u_5min_origin, h))+np.square(central_delta(v_5min_origin, h))

# windshear_60min_50m.columns = windshear_60min_50m.columns - elevation
# windshear_20min_50m.columns = windshear_20min_50m.columns - elevation
# windshear_5min_50m.columns = windshear_5min_50m.columns - elevation

# θv_5min_50m_diff = central_delta(interpolated_θv_5min_25, h)
# θv_20min_50m_diff = central_delta(interpolated_θv_20min_25, h)
# θv_60min_50m_diff = central_delta(interpolated_θv_60min_25, h)

# θv_60min_50m=interpolated_θv_60min_25.iloc[:,int(h/25):np.shape(interpolated_θv_60min_25)[1]-int(h / 25)]
# θv_20min_50m=interpolated_θv_20min_25.iloc[:,int(h/25):np.shape(interpolated_θv_20min_25)[1]-int(h / 25)]
# θv_5min_50m=interpolated_θv_5min_25.iloc[:,int(h/25):np.shape(interpolated_θv_5min_25)[1]-int(h / 25)]

# common_columns_50m = windshear_60min_50m.columns.intersection(θv_60min_50m.columns.astype(int))
# common_time_60min = windshear_60min_50m.index.intersection(θv_60min_50m.index)
# common_time_20min = windshear_20min_50m.index.intersection(θv_20min_50m.index)
# common_time_5min = windshear_5min_50m.index.intersection(θv_5min_50m.index)

# θv_60min_50m_diff.columns = θv_60min_50m_diff.columns.astype(int)
# θv_20min_50m_diff.columns = θv_20min_50m.columns.astype(int)
# θv_5min_50m_diff.columns = θv_5min_50m.columns.astype(int)
# θv_60min_50m.columns = θv_60min_50m_diff.columns.astype(int)
# θv_20min_50m.columns = θv_20min_50m.columns.astype(int)
# θv_5min_50m.columns = θv_5min_50m.columns.astype(int)

# Ri_bulk_60min_50m = (g*h*2*θv_60min_50m_diff[common_columns_50m].loc[common_time_60min,])/(windshear_60min_50m*θv_60min_50m[common_columns_50m].loc[common_time_60min,])
# Ri_bulk_20min_50m = (g*h*2*θv_20min_50m_diff[common_columns_50m].loc[common_time_20min,])/(windshear_20min_50m*θv_20min_50m[common_columns_50m].loc[common_time_20min,])
# Ri_bulk_5min_50m = (g*h*2*θv_5min_50m_diff[common_columns_50m].loc[common_time_5min,])/(windshear_5min_50m*θv_5min_50m[common_columns_50m].loc[common_time_5min,])

# h=50

# windshear_60min_100m=np.square(central_delta(u_60min_origin, h))+np.square(central_delta(v_60min_origin, h))
# windshear_20min_100m=np.square(central_delta(u_20min_origin, h))+np.square(central_delta(v_20min_origin, h))
# windshear_5min_100m=np.square(central_delta(u_5min_origin, h))+np.square(central_delta(v_5min_origin, h))

# windshear_60min_100m.columns = windshear_60min_100m.columns - elevation
# windshear_20min_100m.columns = windshear_20min_100m.columns - elevation
# windshear_5min_100m.columns = windshear_5min_100m.columns - elevation

# θv_5min_100m_diff = central_delta(interpolated_θv_5min_25, h)
# θv_20min_100m_diff = central_delta(interpolated_θv_20min_25, h)
# θv_60min_100m_diff = central_delta(interpolated_θv_60min_25, h)

# θv_60min_100m=interpolated_θv_60min_25.iloc[:,int(h/25):np.shape(interpolated_θv_60min_25)[1]-int(h / 25)]
# θv_20min_100m=interpolated_θv_20min_25.iloc[:,int(h/25):np.shape(interpolated_θv_20min_25)[1]-int(h / 25)]
# θv_5min_100m=interpolated_θv_5min_25.iloc[:,int(h/25):np.shape(interpolated_θv_5min_25)[1]-int(h / 25)]

# common_columns_100m = windshear_60min_100m.columns.intersection(θv_60min_100m.columns.astype(int))
# common_time_60min = windshear_60min_100m.index.intersection(θv_60min_100m.index)
# common_time_20min = windshear_20min_100m.index.intersection(θv_20min_100m.index)
# common_time_5min = windshear_5min_100m.index.intersection(θv_5min_100m.index)

# θv_60min_100m_diff.columns = θv_60min_100m_diff.columns.astype(int)
# θv_20min_100m_diff.columns = θv_20min_100m.columns.astype(int)
# θv_5min_100m_diff.columns = θv_5min_100m.columns.astype(int)
# θv_60min_100m.columns = θv_60min_100m_diff.columns.astype(int)
# θv_20min_100m.columns = θv_20min_100m.columns.astype(int)
# θv_5min_100m.columns = θv_5min_100m.columns.astype(int)


# Ri_bulk_60min_100m = (g*h*2*θv_60min_100m_diff[common_columns_100m].loc[common_time_60min,])/(windshear_60min_100m*θv_60min_100m[common_columns_100m].loc[common_time_60min,])
# Ri_bulk_20min_100m = (g*h*2*θv_20min_100m_diff[common_columns_100m].loc[common_time_20min,])/(windshear_20min_100m*θv_20min_100m[common_columns_100m].loc[common_time_20min,])
# Ri_bulk_5min_100m = (g*h*2*θv_5min_100m_diff[common_columns_100m].loc[common_time_5min,])/(windshear_5min_100m*θv_5min_100m[common_columns_100m].loc[common_time_5min,])


# h=75

# windshear_60min_150m=np.square(central_delta(u_60min_origin, h))+np.square(central_delta(v_60min_origin, h))
# windshear_20min_150m=np.square(central_delta(u_20min_origin, h))+np.square(central_delta(v_20min_origin, h))
# windshear_5min_150m=np.square(central_delta(u_5min_origin, h))+np.square(central_delta(v_5min_origin, h))

# windshear_60min_150m.columns = windshear_60min_150m.columns - elevation
# windshear_20min_150m.columns = windshear_20min_150m.columns - elevation
# windshear_5min_150m.columns = windshear_5min_150m.columns - elevation

# θv_5min_150m_diff = central_delta(interpolated_θv_5min_25, h)
# θv_20min_150m_diff = central_delta(interpolated_θv_20min_25, h)
# θv_60min_150m_diff = central_delta(interpolated_θv_60min_25, h)

# θv_60min_150m=interpolated_θv_60min_25.iloc[:,int(h/25):np.shape(interpolated_θv_60min_25)[1]-int(h / 25)]
# θv_20min_150m=interpolated_θv_20min_25.iloc[:,int(h/25):np.shape(interpolated_θv_20min_25)[1]-int(h / 25)]
# θv_5min_150m=interpolated_θv_5min_25.iloc[:,int(h/25):np.shape(interpolated_θv_5min_25)[1]-int(h / 25)]

# common_columns_150m = windshear_60min_150m.columns.intersection(θv_60min_150m.columns.astype(int))
# common_time_60min = windshear_60min_150m.index.intersection(θv_60min_150m.index)
# common_time_20min = windshear_20min_150m.index.intersection(θv_20min_150m.index)
# common_time_5min = windshear_5min_150m.index.intersection(θv_5min_150m.index)

# θv_60min_150m_diff.columns = θv_60min_150m_diff.columns.astype(int)
# θv_20min_150m_diff.columns = θv_20min_150m.columns.astype(int)
# θv_5min_150m_diff.columns = θv_5min_150m.columns.astype(int)
# θv_60min_150m.columns = θv_60min_150m_diff.columns.astype(int)
# θv_20min_150m.columns = θv_20min_150m.columns.astype(int)
# θv_5min_150m.columns = θv_5min_150m.columns.astype(int)

# Ri_bulk_60min_150m = (g*h*2*θv_60min_150m_diff[common_columns_150m].loc[common_time_60min,])/(windshear_60min_150m*θv_60min_150m[common_columns_150m].loc[common_time_60min,])
# Ri_bulk_20min_150m = (g*h*2*θv_20min_150m_diff[common_columns_150m].loc[common_time_20min,])/(windshear_20min_150m*θv_20min_150m[common_columns_150m].loc[common_time_20min,])
# Ri_bulk_5min_150m = (g*h*2*θv_5min_150m_diff[common_columns_150m].loc[common_time_5min,])/(windshear_5min_150m*θv_5min_150m[common_columns_150m].loc[common_time_5min,])

# print("Bulk Richardson Number Completed")

# ############ 规整到风雷达的高度#########################
# # 假设自定义高度和原始高度是已知的变量
# custom_heights = height  # 替换为你的实际自定义高度
# original_heights = height_θv  # 替换为你的实际原始高度
# MWR_heights_interp25 = np.arange(0, 3001, 25)
# # 准备插值结果的列名
# interpolated_columns = ['time'] + [f'{int(h)}' for h in custom_heights]

# # 存储每个数据集的插值结果的字典,逐个注释运行不然会死机
# datasets = {
#     #'θv_1min': θv_1min,
#     #'θv_5min': θv_5min,
#     #'θv_20min': θv_20min,
#     #'θv_60min': θv_60min
# }

# interpolated_data_dict = {}
# # 假设有多个数据集需要插值（每个数据集是一个DataFrame存储在列表中）

# for key, data in datasets.items():
#     # 用于存储当前数据集所有行的列表
#     interpolated_rows = []

#     # 当前数据集的插值过程
#     for time, row in data.iterrows():
#         temperatures = row.values  # 获取温度值

#         # 创建插值函数，使用三次插值
#         interp_func = interp1d(original_heights, temperatures, kind='cubic', bounds_error=False, fill_value='extrapolate')

#         # 执行插值
#         new_temps = interp_func(custom_heights)

#         # 替换自定义高度与原始高度匹配的插值值
#         for j, h in enumerate(custom_heights):
#             if h in original_heights:
#                 original_index = np.where(original_heights == h)[0][0]
#                 new_temps[j] = temperatures[original_index]

#         # 构建新行并添加到列表
#         new_row = [time] + list(new_temps)
#         interpolated_rows.append(new_row)
#     print(key)
#     # 从列表中一步创建DataFrame，并设置时间戳为索引
#     interpolated_data = pd.DataFrame(interpolated_rows, columns=interpolated_columns)
#     interpolated_data.set_index('time', inplace=True)
    
#     # 将插值结果存储在字典中
#     interpolated_data_dict[key] = interpolated_data

# # 为每个插值结果重命名
# renamed_data_dict = {f"{key}_Ri": df for key, df in interpolated_data_dict.items()}

# interpolated_θv_1min_ri = renamed_data_dict['θv_1min_Ri']
# interpolated_θv_5min_ri = renamed_data_dict['θv_5min_Ri']
# interpolated_θv_20min_ri = renamed_data_dict['θv_20min_Ri']
# interpolated_θv_60min_ri = renamed_data_dict['θv_60min_Ri']

# Δθv_Δz_1min = (interpolated_θv_1min_ri.diff(axis=1).iloc[:, 1:])/25
# Δθv_Δz_5min = (interpolated_θv_5min_ri.diff(axis=1).iloc[:, 1:])/25
# Δθv_Δz_20min = (interpolated_θv_20min_ri.diff(axis=1).iloc[:, 1:])/25
# Δθv_Δz_60min = (interpolated_θv_60min_ri.diff(axis=1).iloc[:, 1:])/25

#####计算理查森数########


def Richardson(thetav, u, v, h, elevation):
    #THIS IS GRADIENT RICHARDSON NUMBER
    """
    Calculate Richardson number using central difference method for given arrays of
    virtual potential temperature (thetav), horizontal wind components (u, v), and height step (h).

    Parameters:
    thetav -- 2D DataFrame with virtual potential temperature values at different heights and times
    u -- 2D DataFrame with u-component of wind speed at different heights and times
    v -- 2D DataFrame with v-component of wind speed at different heights and times
    h -- Height step for central difference calculation (e.g., 25 for 50 meters)

    Returns:
    Ri -- DataFrame with Richardson number values
    """
    g = 9.81  # Gravitational acceleration (m/s^2)
    
    # Align columns and index for thetav, u, and v
    common_timestamps = thetav.index.intersection(u.index).intersection(v.index)
    
    # Create copies of the DataFrames with adjusted column names
    thetav_copy = thetav.copy()
    u_copy = u.copy()
    v_copy = v.copy()
    
    thetav_copy.columns = thetav_copy.columns.astype(int)
    u_copy.columns = u_copy.columns.astype(int) - elevation
    v_copy.columns = v_copy.columns.astype(int) - elevation
    
    common_columns = thetav_copy.columns.intersection(u_copy.columns).intersection(v_copy.columns)

    thetav_copy = thetav_copy.loc[common_timestamps, common_columns]
    u_copy = u_copy.loc[common_timestamps, common_columns]
    v_copy = v_copy.loc[common_timestamps, common_columns]

    n = len(common_columns)
    time_steps = len(common_timestamps)
    step_index = int(h / 25)  # Assuming spatial resolution is 25 meters

    Ri = np.zeros((time_steps, n))  

    for time in range(time_steps):
        for i in range(step_index, n - step_index):
            theta_a = thetav_copy.iloc[time, i - step_index]
            theta_b = thetav_copy.iloc[time, i + step_index]
            u_a = u_copy.iloc[time, i - step_index]
            u_b = u_copy.iloc[time, i + step_index]
            v_a = v_copy.iloc[time, i - step_index]
            v_b = v_copy.iloc[time, i + step_index]
            
            delta_theta = theta_a - theta_b
            delta_z = -2 * step_index * 25  # Total height difference for central difference
            delta_u = u_a - u_b
            delta_v = v_a - v_b

            Ri[time, i] = (2 * g / (theta_a + theta_b)) * (delta_theta * delta_z) / (delta_u**2 + delta_v**2)
    
    Ri = pd.DataFrame(Ri, columns=common_columns, index=common_timestamps)
    Ri = Ri.iloc[:, step_index:n - step_index]
    
    return Ri

Ri_5min_50=Richardson(interpolated_θv_5min_25, wind_filled_u_5min, wind_filled_v_5min, 25, elevation)
Ri_20min_50=Richardson(interpolated_θv_20min_25, wind_filled_u_20min, wind_filled_v_20min, 25, elevation)
Ri_60min_50=Richardson(interpolated_θv_60min_25, wind_filled_u_60min, wind_filled_v_60min, 25, elevation)

Ri_5min_100=Richardson(interpolated_θv_5min_25, wind_filled_u_5min, wind_filled_v_5min, 50, elevation)
Ri_20min_100=Richardson(interpolated_θv_20min_25, wind_filled_u_20min, wind_filled_v_20min, 50, elevation)
Ri_60min_100=Richardson(interpolated_θv_60min_25, wind_filled_u_60min, wind_filled_v_60min, 50, elevation)

Ri_5min_150=Richardson(interpolated_θv_5min_25, wind_filled_u_5min, wind_filled_v_5min, 75, elevation)
Ri_20min_150=Richardson(interpolated_θv_20min_25, wind_filled_u_20min, wind_filled_v_20min, 75, elevation)
Ri_60min_150=Richardson(interpolated_θv_60min_25, wind_filled_u_60min, wind_filled_v_60min, 75, elevation)

def merge_files_in_folder_to_dataframe(folder_path, file_extension, skip_lines, encoding='utf-8'):
    dataframes = []
    
    # 获取文件夹内所有指定后缀的文件名
    files_to_merge = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    
    for file_name in files_to_merge:
        file_path = os.path.join(folder_path, file_name)
        # 读取文件，并跳过指定行数
        df = pd.read_table(file_path, delimiter=' ,', skiprows=skip_lines, encoding=encoding, header=0, index_col=False, engine='python')
        dataframes.append(df)
    
    # 合并所有DataFrame
    merged_dataframe = pd.concat(dataframes, ignore_index=True)
    
    return merged_dataframe

# 稳定性指标
folder_path = 'E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\sta\\'  # 替换为你的文件夹路径
file_extension = '.ASC'  # 替换为你的文件后缀
skip_lines = 12  # 替换为需要跳过的行数
encoding = 'UTF-8'  # 替换为文件的编码

stability_index = merge_files_in_folder_to_dataframe(folder_path, file_extension, skip_lines, encoding)
stability_index.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',' Rain Flag', ' LI [K]', ' KOI [K]', ' TTI [K]', ' KI [K]', ' SI [K]', ' CAPE [J/kg]']
stability_index['Year'] = stability_index['Year'].apply(lambda x: 2000 + x)
stability_index['Timestamp'] = pd.to_datetime(stability_index[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
stability_index['Timestamp'] = stability_index['Timestamp'] + pd.Timedelta(hours=8)
stability_index.set_index('Timestamp', inplace=True)
stability_index=stability_index.iloc[:,7:13]

#IWV
folder_path = 'E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\iwv\\'  # 替换为你的文件夹路径
file_extension = '.ASC'  # 替换为你的文件后缀
skip_lines = 7  # 替换为需要跳过的行数
encoding = 'ANSI'  # 替换为文件的编码

IWV = merge_files_in_folder_to_dataframe(folder_path, file_extension, skip_lines, encoding)
IWV.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',' Rain Flag', ' IWV [kg/m^2]', ' Elev. Ang [癩, Azi. Ang [癩']
IWV['Year'] = IWV['Year'].apply(lambda x: 2000 + x)
IWV['Timestamp'] = pd.to_datetime(IWV[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
IWV['Timestamp'] = IWV['Timestamp'] + pd.Timedelta(hours=8)
IWV.set_index('Timestamp', inplace=True)
#IWV_1min=pd.date_range(start="2022-07-02 00:00:00", end="2022/07/31 23:00:00", freq='1T')
#IWV=IWV.reindex(IWV_1min).interpolate(method='linear')
IWV=IWV.iloc[:,7]
IWV_5min=IWV.resample('5T').mean().interpolate(method='linear')
IWV_5min=IWV_5min.loc[(IWV_5min.index >= "2022-07-02 00:00:00") & (IWV_5min.index <= "2022-07-31 23:00:00")]
IWV_20min=IWV.resample('20T').mean().interpolate(method='linear')
IWV_20min=IWV_20min.loc[(IWV_20min.index >= "2022-07-02 00:00:00") & (IWV_20min.index <= "2022-07-31 23:00:00")]
IWV_60min=IWV.resample('60T').mean().interpolate(method='linear')
IWV_60min=IWV_60min.loc[(IWV_60min.index >= "2022-07-02 00:00:00") & (IWV_60min.index <= "2022-07-31 23:00:00")]

#LWP
folder_path = 'E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\LWP\\'  # 替换为你的文件夹路径
file_extension = '.ASC'  # 替换为你的文件后缀
skip_lines = 7  # 替换为需要跳过的行数
encoding = 'ANSI'  # 替换为文件的编码

LWP = merge_files_in_folder_to_dataframe(folder_path, file_extension, skip_lines, encoding)
LWP.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',' Rain Flag', ' LWP [kg/m^2]', ' Elev. Ang [癩, Azi. Ang [癩']
LWP['Year'] = LWP['Year'].apply(lambda x: 2000 + x)
LWP['Timestamp'] = pd.to_datetime(LWP[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
LWP['Timestamp'] = LWP['Timestamp'] + pd.Timedelta(hours=8)
LWP.set_index('Timestamp', inplace=True)
#LWP_1min=pd.date_range(start="2022-07-02 00:00:00", end="2022/07/31 23:00:00", freq='1T')
#LWP=LWP.reindex(LWP_1min).interpolate(method='linear')
LWP=LWP.iloc[:,7]
LWP_5min=LWP.resample('5T').mean().interpolate(method='linear')
LWP_5min=LWP_5min.loc[(LWP_5min.index >= "2022-07-02 00:00:00") & (LWP_5min.index <= "2022-07-31 23:00:00")]
LWP_20min=LWP.resample('20T').mean().interpolate(method='linear')
LWP_20min=LWP_20min.loc[(LWP_20min.index >= "2022-07-02 00:00:00") & (LWP_20min.index <= "2022-07-31 23:00:00")]
LWP_60min=LWP.resample('60T').mean().interpolate(method='linear')
LWP_60min=LWP_60min.loc[(LWP_60min.index >= "2022-07-02 00:00:00") & (LWP_60min.index <= "2022-07-31 23:00:00")]

# def merge_files_in_folder_to_dataframe(folder_path, file_extension, skip_lines, encoding='utf-8'):
#     dataframes = []
    
#     # 获取文件夹内所有指定后缀的文件名
#     files_to_merge = [f for f in os.listdir(folder_path) if f.endswith(file_extension)]
    
#     for file_name in files_to_merge:
#         file_path = os.path.join(folder_path, file_name)
#         # 读取文件，并跳过指定行数
#         df = pd.read_table(file_path, delimiter=' ,', skiprows=skip_lines, encoding=encoding, header=None, index_col=False, engine='python')
#         dataframes.append(df)
    
#     # 合并所有DataFrame
#     merged_dataframe = pd.concat(dataframes, ignore_index=True)
    
#     return merged_dataframe

# #LWD
# folder_path = 'E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\LWD\\'  # 替换为你的文件夹路径
# file_extension = '.ASC'  # 替换为你的文件后缀
# skip_lines = 10  # 替换为需要跳过的行数
# encoding = 'UTF-8'  # 替换为文件的编码
# header=None
# LWD = merge_files_in_folder_to_dataframe(folder_path, file_extension, skip_lines, encoding)

# LWD.columns = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second',' Rain Flag', 
#                '0', '20', '40', '60', '80', '100', '120', '140', '160', '180', '200', '220', '240', '260', '280', '300', '330', '360', '390', '420', '450', 
#                '480', '520', '550', '580', '620', '660', '700', '740', '780', '820', '860', '900', '950', '1000', '1050', '1100', '1150', '1200', '1250', 
#                '1300', '1350', '1400', '1450', '1500', '1550', '1600', '1650', '1700', '1750', '1800', '1850', '1900', '1950', '2000', '2100', '2200', '2300', 
#                '2400', '2500', '2600', '2700', '2800', '2900', '3000', '3150', '3300', '3450', '3600', '3750', '3900', '4100', '4300', '4500', '4800', '5100', 
#                '5400', '5700', '6000', '6300', '6600', '6900', '7200', '7500', '7800', '8100', '8400', '8700', '9000', '9300', '9600', '10000']
# LWD['Year'] = LWD['Year'].apply(lambda x: 2000 + x)
# LWD['Timestamp'] = pd.to_datetime(LWD[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])
# LWD['Timestamp'] = LWD['Timestamp'] + pd.Timedelta(hours=8)
# LWD.set_index('Timestamp', inplace=True)
# LWD.index=LWD.index.strftime('%Y/%m/%d %H:%M:%S')
# LWD.to_csv("E:\\Wuhai Data\\微波辐射计\\原始数据\\M07\\LWD\\LWD.csv")
# #LWD_1min=pd.date_range(start="2022-07-02 00:00:00", end="2022/07/31 23:00:00", freq='1T')
# #LWD=LWD.reindex(LWD_1min).interpolate(method='linear')
# LWD_1min=LWD.resample('1T').mean().interpolate(method='linear')
# LWD_5min=LWD_1min.resample('5T').mean()
# LWD_5min=LWD_5min.loc[(LWD_5min.index >= "2022-07-02 00:00:00") & (LWD_5min.index <= "2022-07-31 23:00:00")]
# LWD_20min=LWD.resample('20T').mean().interpolate(method='linear')
# LWD_20min=LWD_20min.loc[(LWD_20min.index >= "2022-07-02 00:00:00") & (LWD_20min.index <= "2022-07-31 23:00:00")]
# LWD_60min=LWD.resample('60T').mean().interpolate(method='linear')
# LWD_60min=LWD_60min.loc[(LWD_60min.index >= "2022-07-02 00:00:00") & (LWD_60min.index <= "2022-07-31 23:00:00")]
# import matplotlib.pyplot as plt
# plt.figure(figsize=(12, 6))
# plt.contourf(LWD.index, LWD.columns, LWD.T, cmap='jet')
# plt.colorbar(label='Variable Value')
# plt.xlabel('Time')
# plt.ylabel('Height (m)')
# plt.title('Filled Contour Plot of Variable Over Time and Height')
# plt.show()
print("微波辐射计数据处理完毕，开始边界层计算")
##################################################计算热力、物质边界层################################################################
def Second_order_diff(f, h):
    
    """
    计算函数值数组f的中央差分
    `
    参数：
    f -- 2D-Dataframe，表示在等间距点上的函数值
    h -- 步长
    
    返回：
    df -- 一维数组，表示f的中央差分
    """
    n = f.shape[1]
    time_steps = f.shape[0]
    step_index = int(h / 25)#25是空间分辨率
    df = np.zeros((time_steps, n))  
    
    for time in range(np.shape(f)[0]):
        for i in range(0, int(n-h/25)):
            if i>=(int(h/25)) and i <(np.shape(f)[1]-step_index):
                df[time, i] = (f.iloc[time, i + step_index] + f.iloc[time, i - step_index] - 2 * f.iloc[time, i]) / np.square(h)
    df=pd.DataFrame(df)
    df.columns=f.columns
    df=df.iloc[:,step_index:np.shape(f)[1]-step_index]
    return df

# Initialize arrays for MLH and SLH

dθv_dz_60min_50m=central_difference(interpolated_θv_60min_25, 25)
dθv_dz_20min_50m=central_difference(interpolated_θv_20min_25, 25)
dθv_dz_5min_50m=central_difference(interpolated_θv_5min_25, 25)
dθv_dz_1min_50m=central_difference(interpolated_θv_1min_25, 25)

dθv_dz_60min_100m=central_difference(interpolated_θv_60min_25, 50)
dθv_dz_20min_100m=central_difference(interpolated_θv_20min_25, 50)
dθv_dz_5min_100m=central_difference(interpolated_θv_5min_25, 50)
dθv_dz_1min_100m=central_difference(interpolated_θv_1min_25, 50)

dθv_dz_60min_150m=central_difference(interpolated_θv_60min_25, 75)
dθv_dz_20min_150m=central_difference(interpolated_θv_20min_25, 75)
dθv_dz_5min_150m=central_difference(interpolated_θv_5min_25, 75)
dθv_dz_1min_150m=central_difference(interpolated_θv_1min_25, 75)

#插值到25米间隔的高度计算边界层到2米使计算结果更精确
custom_heights = np.arange(0, 10001, 2)  # 替换为你的实际自定义高度
original_heights = height_θv  # 替换为你的实际原始高度

# 准备插值结果的列名
interpolated_columns = ['time'] + [f'{int(h)}' for h in custom_heights]

# 存储每个数据集的插值结果的字典,逐个注释运行不然会死机
datasets = {
    'θv_1min_interval2': θv_1min,
    #'θv_2min': θv_2min,
    'θv_5min_interval2': θv_5min,
    #'θv_10min': θv_10min,
    #'θv_15min': θv_15min,
    'θv_20min_interval2': θv_20min,
    #'θv_30min': θv_30min
    'θv_60min_interval2': θv_60min
}

interpolated_data_dict = {}
# 假设有多个数据集需要插值（每个数据集是一个DataFrame存储在列表中）

for key, data in datasets.items():
    # 用于存储当前数据集所有行的列表
    interpolated_rows = []

    # 当前数据集的插值过程
    for time, row in data.iterrows():
        temperatures = row.values  # 获取温度值

        # 创建插值函数，使用三次插值
        interp_func = interp1d(original_heights, temperatures, kind='cubic', bounds_error=False, fill_value='extrapolate')

        # 执行插值
        new_temps = interp_func(custom_heights)

        # 替换自定义高度与原始高度匹配的插值值
        for j, h in enumerate(custom_heights):
            if h in original_heights:
                original_index = np.where(original_heights == h)[0][0]
                new_temps[j] = temperatures[original_index]

        # 构建新行并添加到列表
        new_row = [time] + list(new_temps)
        interpolated_rows.append(new_row)
    print(key)
    # 从列表中一步创建DataFrame，并设置时间戳为索引
    interpolated_data = pd.DataFrame(interpolated_rows, columns=interpolated_columns)
    interpolated_data.set_index('time', inplace=True)
    
    # 将插值结果存储在字典中
    interpolated_data_dict[key] = interpolated_data

# 为每个插值结果重命名
renamed_data_dict = {f"{key}": df for key, df in interpolated_data_dict.items()}
interpolated_θv_1min_2 = renamed_data_dict['θv_1min_interval2']
interpolated_θv_5min_2 = renamed_data_dict['θv_5min_interval2']
interpolated_θv_20min_2 = renamed_data_dict['θv_20min_interval2']
interpolated_θv_60min_2 = renamed_data_dict['θv_60min_interval2']

##计算CBL##
def CBL_calc(f):
    f.columns = f.columns.astype(int)
    f=f[[col for col in f.columns if col < 4001]]
    cbl = np.empty((f.shape[0], 10))*np.nan
    for time in range(f.shape[0]):  # 故意使用超出列表长度的范围
        #print(time)
        try:
            theta0 = f.iloc[time,0]
            i=0
            for height in range(1,f.shape[1]):
                if ((f.iloc[time,height]-theta0)*(f.iloc[time,height-1]-theta0)<0):
                    cbl[time][i]=f.columns[height]
                    i+=1
        except IndexError:  # 捕获下标越界异常
            print(f.index[time])
            print("下标越界，跳过当前循环")
            continue  # 跳过当前循环，继续下一个循环
    return cbl

CBL_60min=pd.concat([interpolated_θv_60min_2.reset_index()['time'],pd.DataFrame(CBL_calc(interpolated_θv_60min_2))],axis=1).set_index('time')
CBL_20min=pd.concat([interpolated_θv_20min_2.reset_index()['time'],pd.DataFrame(CBL_calc(interpolated_θv_20min_2))],axis=1).set_index('time')
CBL_5min=pd.concat([interpolated_θv_5min_2.reset_index()['time'],pd.DataFrame(CBL_calc(interpolated_θv_5min_2))],axis=1).set_index('time')
CBL_1min=pd.concat([interpolated_θv_1min_2.reset_index()['time'],pd.DataFrame(CBL_calc(interpolated_θv_1min_2))],axis=1).set_index('time')

#寻找导数的绝对值最小的点
def SBL_deri_argmin(f):
    f.columns = f.columns.astype(int)
    f=f[[col for col in f.columns if col < 3501 and col > 175]]
    index=np.argmin(f.abs(), axis=1)
    SBL_min_index=f.columns[index]
    return SBL_min_index

SBL_argmin_60min_50m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_60min_50m))],axis=1).set_index('time')
SBL_argmin_60min_100m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_60min_100m))],axis=1).set_index('time')
SBL_argmin_60min_150m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_60min_150m))],axis=1).set_index('time')

SBL_argmin_20min_50m = pd.concat([interpolated_θv_20min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_20min_50m))],axis=1).set_index('time')
SBL_argmin_20min_100m = pd.concat([interpolated_θv_20min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_20min_100m))],axis=1).set_index('time')
SBL_argmin_20min_150m = pd.concat([interpolated_θv_20min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_20min_150m))],axis=1).set_index('time')

SBL_argmin_5min_50m = pd.concat([interpolated_θv_5min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_5min_50m))],axis=1).set_index('time')
SBL_argmin_5min_100m = pd.concat([interpolated_θv_5min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_5min_100m))],axis=1).set_index('time')
SBL_argmin_5min_150m = pd.concat([interpolated_θv_5min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_5min_150m))],axis=1).set_index('time')

SBL_argmin_1min_50m = pd.concat([interpolated_θv_1min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_1min_50m))],axis=1).set_index('time')
SBL_argmin_1min_100m = pd.concat([interpolated_θv_1min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_1min_100m))],axis=1).set_index('time')
SBL_argmin_1min_150m = pd.concat([interpolated_θv_1min_25.reset_index()['time'],pd.DataFrame(SBL_deri_argmin(dθv_dz_1min_150m))],axis=1).set_index('time')


##计算SBL##-负梯度最大
def SBL_calc_neg_max(f):
    f.columns = f.columns.astype(int)
    f=f[[col for col in f.columns if col < 3501 and col > 175]]
    f=f[f<0]
    def get_min_and_next_values(row):
        sorted_row = row.sort_values()
        return pd.Series({
            #'Min_Value': sorted_row.iloc[0],
            'Min_Column': sorted_row.index[0],
            #'2nd_Min_Value': sorted_row.iloc[1],
            '2nd_Min_Column': sorted_row.index[1],
            #'3rd_Min_Value': sorted_row.iloc[2],
            '3rd_Min_Column': sorted_row.index[2],
            #'4th_Min_Value': sorted_row.iloc[3],
            '4th_Min_Column': sorted_row.index[3],
            #'5th_Min_Value': sorted_row.iloc[4],
            '5th_Min_Column': sorted_row.index[4],
        })
    sbl_neg_max=pd.DataFrame(f.apply(get_min_and_next_values, axis=1).apply(lambda x: sorted(x), axis=1).tolist())
    return sbl_neg_max

SBL_neg_max_60min_50m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_50m))],axis=1).set_index('time')
SBL_neg_max_60min_100m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_100m))],axis=1).set_index('time')
SBL_neg_max_60min_150m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_150m))],axis=1).set_index('time')

SBL_neg_max_20min_50m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_50m))],axis=1).set_index('time')
SBL_neg_max_20min_100m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_100m))],axis=1).set_index('time')
SBL_neg_max_20min_150m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_150m))],axis=1).set_index('time')

SBL_neg_max_5min_50m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_50m))],axis=1).set_index('time')
SBL_neg_max_5min_100m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_100m))],axis=1).set_index('time')
SBL_neg_max_5min_150m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dθv_dz_60min_150m))],axis=1).set_index('time')

##计算SBL##-正梯度最小
def SBL_pos_min(f):
    f.columns = f.columns.astype(int)
    f=f[[col for col in f.columns if col < 3501 and col > 175]]
    def SBL_pos_min_by_row_calc(row):
        
        # 过滤正值
        positive_row = row[row > 0]
        # 排序
        sorted_row = positive_row.sort_values()
        # 如果正值不足5个，填充NaN
        result = {
            #'Min_Value': sorted_row.iloc[0] if len(sorted_row) > 0 else float('nan'),
            'Min_Column': sorted_row.index[0] if len(sorted_row) > 0 else float('nan'),
            #'2nd_Min_Value': sorted_row.iloc[1] if len(sorted_row) > 1 else float('nan'),
            '2nd_Min_Column': sorted_row.index[1] if len(sorted_row) > 1 else float('nan'),
           # '3rd_Min_Value': sorted_row.iloc[2] if len(sorted_row) > 2 else float('nan'),
            '3rd_Min_Column': sorted_row.index[2] if len(sorted_row) > 2 else float('nan'),
            #'4th_Min_Value': sorted_row.iloc[3] if len(sorted_row) > 3 else float('nan'),
            '4th_Min_Column': sorted_row.index[3] if len(sorted_row) > 3 else float('nan'),
           # '5th_Min_Value': sorted_row.iloc[4] if len(sorted_row) > 4 else float('nan'),
            '5th_Min_Column': sorted_row.index[4] if len(sorted_row) > 4 else float('nan')
        }
        return pd.Series(result)
    
    result = f.apply(SBL_pos_min_by_row_calc, axis=1)
   
    return result

SBL_pos_min_60min_50m=pd.DataFrame(SBL_pos_min(dθv_dz_60min_50m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_60min_50m.index=interpolated_θv_60min_25.index

SBL_pos_min_20min_50m=pd.DataFrame(SBL_pos_min(dθv_dz_20min_50m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_20min_50m.index=interpolated_θv_20min_25.index

SBL_pos_min_5min_50m=pd.DataFrame(SBL_pos_min(dθv_dz_5min_50m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_5min_50m.index=interpolated_θv_5min_25.index

SBL_pos_min_60min_100m=pd.DataFrame(SBL_pos_min(dθv_dz_60min_100m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_60min_100m.index=interpolated_θv_60min_25.index

SBL_pos_min_20min_100m=pd.DataFrame(SBL_pos_min(dθv_dz_20min_100m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_20min_100m.index=interpolated_θv_20min_25.index

SBL_pos_min_5min_100m=pd.DataFrame(SBL_pos_min(dθv_dz_5min_100m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_5min_100m.index=interpolated_θv_5min_25.index

SBL_pos_min_60min_150m=pd.DataFrame(SBL_pos_min(dθv_dz_60min_150m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_60min_150m.index=interpolated_θv_60min_25.index

SBL_pos_min_20min_150m=pd.DataFrame(SBL_pos_min(dθv_dz_20min_150m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_20min_150m.index=interpolated_θv_20min_25.index

SBL_pos_min_5min_150m=pd.DataFrame(SBL_pos_min(dθv_dz_5min_150m).apply(lambda x: sorted(x), axis=1).tolist())
SBL_pos_min_5min_150m.index=interpolated_θv_5min_25.index

##############################消光系数355################################################
PExC_355_data = pd.read_excel("E:\\Wuhai Data\\原始数据\\乌海超站 - 气溶胶雷达数据\\乌海超站-气溶胶雷达数据\消光355\\merged_transposed_output.xlsx",sheet_name="Sheet1")
PExC_355_timestamps_5min=pd.date_range(start="2022-07-02 00:00:00", end="2022-07-31 23:00:00", freq='5T')
PExC_355_data['time'] = pd.to_datetime(PExC_355_data['time'])
PExC_355_data['time'] = PExC_355_data['time'] - pd.Timedelta(hours=8)
PExC_355_data=PExC_355_data.set_index("time")
#地面观测插值成5min
buqi_timestamps=pd.date_range(start="2022/06/30 00:00", end="2022/07/28 23:50:00", freq='5T')
PExC_355_data_5min = PExC_355_data.reindex(buqi_timestamps).interpolate(method='linear')
PExC_355_data_5min = PExC_355_data_5min.loc[(PExC_355_data_5min.index >= "2022-07-02 00:00:00") & (PExC_355_data_5min.index <= "2022-07-31 23:00:00")]

PExC_355_data_20min = PExC_355_data.resample('20T').mean()
PExC_355_data_20min = PExC_355_data_20min.loc[(PExC_355_data_20min.index >= "2022-07-02 00:00:00") & (PExC_355_data_20min.index <= "2022-07-31 23:00:00")]
PExC_355_data_60min = PExC_355_data.resample('60T').mean()
PExC_355_data_60min = PExC_355_data_60min.loc[(PExC_355_data_60min.index >= "2022-07-02 00:00:00") & (PExC_355_data_60min.index <= "2022-07-31 23:00:00")]

def PExC_355_central_difference(f, h):
    
    """
    计算函数值数组f的中央差分
    
    参数：
    f -- 2D-Dataframe，表示在等间距点上的函数值
    h -- 步长
    
    返回： 
    df -- 一维数组，表示f的中央差分
    """
    n = f.shape[1]
    f.columns = f.columns.astype(float)
    time_steps = f.shape[0]
    step_index = int(h / 7.5)#25是空间分辨率
    df = np.zeros((time_steps, n))  
    
    for time in range(np.shape(f)[0]):
        for i in range(0, int(n-h/25)):
            if i>=(int(h/25)) and i <(np.shape(f)[1]-step_index):
                df[time, i] = (f.iloc[time, i + step_index] - f.iloc[time, i - step_index]) / (2 * h)
    df=pd.DataFrame(df)
    df.columns=f.columns
    df=df.iloc[:,step_index:np.shape(f)[1]-step_index]
    df.index = f.index
    return df

dPExC_355_dz_60min_15m=PExC_355_central_difference(PExC_355_data_60min, 7.5)
dPExC_355_dz_60min_30m=PExC_355_central_difference(PExC_355_data_60min, 15)
dPExC_355_dz_60min_45m=PExC_355_central_difference(PExC_355_data_60min, 22.5)

dPExC_355_dz_20min_15m=PExC_355_central_difference(PExC_355_data_20min, 7.5)
dPExC_355_dz_20min_30m=PExC_355_central_difference(PExC_355_data_20min, 15)
dPExC_355_dz_20min_45m=PExC_355_central_difference(PExC_355_data_20min, 22.5)

dPExC_355_dz_5min_15m=PExC_355_central_difference(PExC_355_data_5min, 7.5)
dPExC_355_dz_5min_30m=PExC_355_central_difference(PExC_355_data_5min, 15)
dPExC_355_dz_5min_45m=PExC_355_central_difference(PExC_355_data_5min, 22.5)

##计算SBL##-负梯度最大
def SBL_calc_neg_max(f):
    f.columns = f.columns.astype(float)
    f=f[[col for col in f.columns if col < 3501 and col > 0]]
    f=f[f<0]
    def get_min_and_next_values(row):
        sorted_row = row.sort_values()
        return pd.Series({
            #'Min_Value': sorted_row.iloc[0],
            'Min_Column': sorted_row.index[0],
            #'2nd_Min_Value': sorted_row.iloc[1],
            '2nd_Min_Column': sorted_row.index[1],
            #'3rd_Min_Value': sorted_row.iloc[2],
            '3rd_Min_Column': sorted_row.index[2],
            #'4th_Min_Value': sorted_row.iloc[3],
            '4th_Min_Column': sorted_row.index[3],
            #'5th_Min_Value': sorted_row.iloc[4],
            '5th_Min_Column': sorted_row.index[4],
        })
    sbl_neg_max=pd.DataFrame(f.apply(get_min_and_next_values, axis=1).apply(lambda x: sorted(x), axis=1).tolist())
    return sbl_neg_max

PExC_355_neg_max_60min_15m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_60min_15m))],axis=1).set_index('time')
PExC_355_neg_max_60min_30m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_60min_30m))],axis=1).set_index('time')
PExC_355_neg_max_60min_45m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_60min_45m))],axis=1).set_index('time')

PExC_355_neg_max_20min_15m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_20min_15m))],axis=1).set_index('time')
PExC_355_neg_max_20min_30m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_20min_30m))],axis=1).set_index('time')
PExC_355_neg_max_20min_45m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_20min_45m))],axis=1).set_index('time')

PExC_355_neg_max_5min_15m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_20min_15m))],axis=1).set_index('time')
PExC_355_neg_max_5min_30m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_20min_30m))],axis=1).set_index('time')
PExC_355_neg_max_5min_45m = pd.concat([interpolated_θv_60min_25.reset_index()['time'],pd.DataFrame(SBL_calc_neg_max(dPExC_355_dz_20min_45m))],axis=1).set_index('time')

print("边界层计算完毕，开始绘制探空图")
#########################T-LnP Plot######################
#露点温度计算公式https://doi.org/10.1175/BAMS-86-2-225
#Td=T-(20-0.2*RH)*(300-T)^2-0.00135*(RH-84)^2+0.35

####数据准备，对齐到25米间距的高度####
import matplotlib.pyplot as plt
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units

#data preparation

def Td_calc(T,RH):
    T=T-273.15
    Td=T-(20-0.2*RH)*(T/300)**2-0.00135*(RH-84)**2+0.35
    return Td

Td_60min=Td_calc(temp_60min_mean,rh_60min_mean)
wd_60min = np.mod(180 + np.degrees(np.arctan2(wind_filled_u_60min, wind_filled_v_60min)), 360)

#插值到25米间隔的高度计算边界层到2米使计算结果更精确
custom_heights = np.arange(0, 10001, 25)  # 替换为你的实际自定义高度
original_heights = height_θv  # 替换为你的实际原始高度

# 准备插值结果的列名
interpolated_columns = ['time'] + [f'{int(h)}' for h in custom_heights]

# 存储每个数据集的插值结果的字典,逐个注释运行不然会死机
datasets = {
    #'θv_1min': θv_1min,
    'T_1min_interval25': temp_1min_mean,
    'Td_60min_interval25': Td_60min,
    'T_5min_interval25': temp_5min_mean,
    'T_20min_interval25': temp_20min_mean,
    'rh_60min_interval25': rh_60min_mean,
    'p_60min_interval25': p_60min,
    'T_60min_interval25': temp_60min_mean
}

interpolated_data_dict = {}
# 假设有多个数据集需要插值（每个数据集是一个DataFrame存储在列表中）

for key, data in datasets.items():
    # 用于存储当前数据集所有行的列表
    interpolated_rows = []

    # 当前数据集的插值过程
    for time, row in data.iterrows():
        temperatures = row.values  # 获取温度值

        # 创建插值函数，使用三次插值
        interp_func = interp1d(original_heights, temperatures, kind='cubic', bounds_error=False, fill_value='extrapolate')

        # 执行插值
        new_temps = interp_func(custom_heights)

        # 替换自定义高度与原始高度匹配的插值值
        for j, h in enumerate(custom_heights):
            if h in original_heights:
                original_index = np.where(original_heights == h)[0][0]
                new_temps[j] = temperatures[original_index]

        # 构建新行并添加到列表
        new_row = [time] + list(new_temps)
        interpolated_rows.append(new_row)
    print(key)
    # 从列表中一步创建DataFrame，并设置时间戳为索引
    interpolated_data = pd.DataFrame(interpolated_rows, columns=interpolated_columns)
    interpolated_data.set_index('time', inplace=True)
    
    # 将插值结果存储在字典中
    interpolated_data_dict[key] = interpolated_data

# 为每个插值结果重命名
renamed_data_dict = {f"{key}": df for key, df in interpolated_data_dict.items()}
T_60min_25m = renamed_data_dict['T_60min_interval25']
Td_60min_25m = renamed_data_dict['Td_60min_interval25']
rh_60min_25m = renamed_data_dict['rh_60min_interval25']
p_60min_25m = renamed_data_dict['p_60min_interval25']

T_20min_25m = renamed_data_dict['T_20min_interval25']
T_5min_25m = renamed_data_dict['T_5min_interval25']
T_1min_25m = renamed_data_dict['T_1min_interval25']


dT_dz_60min_50m = central_difference(T_60min_25m, 25)
dT_dz_20min_50m = central_difference(T_20min_25m, 25)
dT_dz_5min_50m = central_difference(T_5min_25m, 25)
dT_dz_1min_50m = central_difference(T_1min_25m, 25)


dT_dz_60min_100m = central_difference(T_60min_25m, 50)
dT_dz_20min_100m = central_difference(T_20min_25m, 50)
dT_dz_5min_100m = central_difference(T_5min_25m, 50)
dT_dz_1min_100m = central_difference(T_1min_25m, 50)



dT_dz_60min_150m = central_difference(T_60min_25m, 75)
dT_dz_20min_150m = central_difference(T_20min_25m, 75)
dT_dz_5min_150m = central_difference(T_5min_25m, 75)
dT_dz_1min_150m = central_difference(T_1min_25m, 75)


#输入的数据形状需要一致
T_60min_25m.columns=T_60min_25m.columns.astype(int)
T_60min_25m=T_60min_25m.loc[:, T_60min_25m.columns < 3001+elevation]
Td_60min_25m.columns=Td_60min_25m.columns.astype(int)
Td_60min_25m=Td_60min_25m.loc[:, Td_60min_25m.columns < 3001+elevation]
rh_60min_25m.columns=rh_60min_25m.columns.astype(int)
rh_60min_25m=rh_60min_25m.loc[:, rh_60min_25m.columns < 3001+elevation]
p_60min_25m.columns=p_60min_25m.columns.astype(int)
p_60min_25m=p_60min_25m.loc[:, p_60min_25m.columns < 3001+elevation]


def T_lnP_Plot(p, T, Td, u, v, wd, ws, time_index, index_df, outfolder):

    # Calculate the LCL pressure and temperature
    lcl_pressure, lcl_temperature = mpcalc.lcl(p[0], T[0], Td[0])
    
    # Calculate the parcel profile
    parcel_prof = mpcalc.parcel_profile(p, T[0], Td[0]).to('degC')
    
    # Create the figure for the Skew-T plot
    fig = plt.figure(figsize=(18, 12))
    skew = SkewT(fig, rotation=30)
    
    # Plot the temperature and dew point temperature
    skew.plot(p, T, 'r')
    skew.plot(p, Td, 'g')
    
    # Plot wind barbs
    skew.plot_barbs(p[::3], u[::3], v[::3], barb_increments={'half': 2, 'full': 4, 'flag': 20})
    
    # Set y-ticks, x-ticks, limits, and axis labels
    skew.ax.set_yticks(range(600, 901, 50))
    skew.ax.set_xticks(range(0, 41, 5))
    skew.ax.set_ylim(900, 600)
    skew.ax.set_xlim(0, 40)
    
    # Plot LCL temperature as black dot
    skew.plot(lcl_pressure, lcl_temperature, 'ko', markerfacecolor='black')
    
    # Plot the parcel profile as a black line
    skew.plot(p, parcel_prof, 'k', linewidth=2)
    
    # Shade areas of CAPE and CIN
    skew.shade_cin(p, T, parcel_prof, Td)
    skew.shade_cape(p, T, parcel_prof)
    
    # Plot a zero degree isotherm
    skew.ax.axvline(0, color='c', linestyle='--', linewidth=2)
    
    skew.plot_dry_adiabats()
    skew.plot_moist_adiabats()
    skew.plot_mixing_lines()
    
    # Set the title and axis labels
    skew.ax.set_title(index_df[time_index], fontsize=22,pad = 20)
    skew.ax.set_xlabel('Temperature (°C)', fontsize=22)
    skew.ax.set_ylabel('Pressure (hPa)', fontsize=22)
    
    # Set tick parameters
    skew.ax.tick_params(axis='both', which='major', labelsize=22, direction='in', width=2.5, length=10, pad=10)#
    
    # Set spine linewidth
    for spine in skew.ax.spines.values():
        spine.set_linewidth(2)
    
    # Save the plot to the specified output folder
    output_filename = str(index_df[time_index])[0:13] + '.tif'
    # if os.path.exists(outfolder + output_filename):
    #     os.remove(outfolder + output_filename)
    plt.savefig(outfolder + output_filename, format='tiff', dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    return

common_timestamps = p_60min_25m.index.intersection(smoothed_hori_speed_60min.index)

for time_index in range(0,common_timestamps.shape[0]):
    p = p_60min_25m.loc[common_timestamps,0:3000].iloc[time_index].values* units.hPa
    T = (T_60min_25m.loc[common_timestamps,0:3000].iloc[time_index].values-273.15) * units.degC
    Td = Td_60min_25m.loc[common_timestamps,0:3000].iloc[time_index].values * units.degC
    ws = smoothed_hori_speed_60min.loc[common_timestamps,0+elevation:3000+elevation].iloc[time_index].values * units.meter / units.second
    wd = wd_60min.loc[common_timestamps,0+elevation:3000+elevation].iloc[time_index].values * units.degrees
    u = wind_filled_u_60min.loc[common_timestamps,0+elevation:3000+elevation].iloc[time_index].values * units.meter / units.second
    v = wind_filled_v_60min.loc[common_timestamps,0+elevation:3000+elevation].iloc[time_index].values  * units.meter / units.second
    T_lnP_Plot(p,T,Td,u,v,wd,ws,time_index,common_timestamps,"E:\\Wuhai Data\\T-Lnp\\")

print("探空图绘制完毕，开始输出60min数据")
##################################60min data out#######################################

    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\60min raw data.xlsx', engine='openpyxl') as writer:
    
    Wind_matrix_60min.reset_index().to_excel(writer, sheet_name='Wind_60min_igor', index=False)
    
    horizontal_speed_filled_60min.reset_index().to_excel(writer, sheet_name='hori_speed_60min', index=False)
    
    smoothed_hori_speed_60min.reset_index().to_excel(writer, sheet_name='smoothed_hori_speed_60min', index=False)
    
    wind_filled_w_60min.reset_index().to_excel(writer, sheet_name='W_60min', index=False)

    PExC_355_data_60min.reset_index().to_excel(writer, sheet_name='PExC_355_60min', index=False)
    
    TKE_60min.reset_index().to_excel(writer, sheet_name='TKE_60min', index=False)
    
    interpolated_θv_60min_25.reset_index().to_excel(writer, sheet_name='θv_60min', index=False)
    
    T_60min_25m.reset_index().to_excel(writer, sheet_name='T_60min', index=False)
    
    rh_60min_mean.reset_index().to_excel(writer, sheet_name='rh_60min', index=False)
    
    ah_60min_mean.reset_index().to_excel(writer, sheet_name='ah_60min', index=False)
    
    q_60min.reset_index().to_excel(writer, sheet_name='q_60min', index=False)
    
    IWV_60min.reset_index().to_excel(writer, sheet_name='IWV_60min', index=False)
    
    LWP_60min.reset_index().reset_index().to_excel(writer, sheet_name='LWP_60min', index=False)
    
    stability_index.reset_index().to_excel(writer, sheet_name='sta_index', index=False)
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\60min diff data.xlsx', engine='openpyxl') as writer:
    
    dT_dz_60min_50m.reset_index().to_excel(writer, sheet_name='dT_dz_60min_25m', index=False)
    
    dT_dz_60min_100m.reset_index().to_excel(writer, sheet_name='dT_dz_60min_50m', index=False)
    
    dT_dz_60min_150m.reset_index().to_excel(writer, sheet_name='dT_dz_60min_75m', index=False)
    
    dθv_dz_60min_50m.reset_index().to_excel(writer, sheet_name='dθv_dz_60min_50m', index=False)
    
    dθv_dz_60min_100m.reset_index().to_excel(writer, sheet_name='dθv_dz_60min_100m', index=False)
    
    dθv_dz_60min_150m.reset_index().to_excel(writer, sheet_name='dθv_dz_60min_150m', index=False)
    
    dPExC_355_dz_60min_15m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_60min_15m', index=False)

    dPExC_355_dz_60min_30m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_60min_30m', index=False)
     
    dPExC_355_dz_60min_45m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_60min_45m', index=False)
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\60min PBLH data.xlsx', engine='openpyxl') as writer:

    Ri_60min_50.reset_index().to_excel(writer, sheet_name='Ri_60min_50m', index=False)
    
    Ri_60min_100.reset_index().to_excel(writer, sheet_name='Ri_60min_100m', index=False)
    
    Ri_60min_150.reset_index().to_excel(writer, sheet_name='Ri_60min_150m', index=False)
    
    WDS_50m_60min.reset_index().to_excel(writer, sheet_name='wds_60min_50m', index=False)
    
    WDS_100m_60min.reset_index().to_excel(writer, sheet_name='wds_60min_100m', index=False)
    
    WDS_150m_60min.reset_index().to_excel(writer, sheet_name='wds_60min_150m', index=False)
    
    windshear_60min_50m.reset_index().to_excel(writer, sheet_name='windshear_60min_50m', index=False)
    
    windshear_60min_100m.reset_index().to_excel(writer, sheet_name='windshear_60min_100m', index=False)
    
    windshear_60min_150m.reset_index().to_excel(writer, sheet_name='windshear_60min_150m', index=False)
    
    CBL_60min.reset_index().to_excel(writer, sheet_name='CBL_60min', index=False)
    
    SBL_argmin_60min_50m.reset_index().to_excel(writer, sheet_name='SBL_argmin_60min_50m', index=False)
    
    SBL_argmin_60min_100m.reset_index().to_excel(writer, sheet_name='SBL_argmin_60min_100m', index=False)
    
    SBL_argmin_60min_150m.reset_index().to_excel(writer, sheet_name='SBL_argmin_60min_150m', index=False)
    
    SBL_neg_max_60min_50m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_60min_50m', index=False)
    
    SBL_neg_max_60min_100m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_60min_100m', index=False)
    
    SBL_neg_max_60min_150m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_60min_150m', index=False)
    
    SBL_pos_min_60min_50m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_60min_50m', index=False)
    
    SBL_pos_min_60min_100m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_60min_100m', index=False)
    
    SBL_pos_min_60min_150m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_60min_150m', index=False)
    
    
##########################20min data output##################
print("输出60min数据完毕，开始输出20min数据")
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\20min raw data.xlsx', engine='openpyxl') as writer:
    
    Wind_matrix_20min.reset_index().to_excel(writer, sheet_name='Wind_20min_igor', index=False)
    
    horizontal_speed_filled_20min.reset_index().to_excel(writer, sheet_name='hori_speed_20min', index=False)
    
    smoothed_hori_speed_20min.reset_index().to_excel(writer, sheet_name='smoothed_hori_speed_20min', index=False)
    
    wind_filled_w_20min.reset_index().to_excel(writer, sheet_name='w_20min', index=False)

    PExC_355_data_20min.reset_index().to_excel(writer, sheet_name='PExC_355_20min', index=False)
    
    TKE_20min.reset_index().to_excel(writer, sheet_name='TKE_20min', index=False)
    
    interpolated_θv_20min_25.reset_index().to_excel(writer, sheet_name='θv_20min', index=False)
    
    T_20min_25m.reset_index().to_excel(writer, sheet_name='T_20min', index=False)
    
    rh_20min_mean.reset_index().to_excel(writer, sheet_name='rh_20min', index=False)
    
    ah_20min_mean.reset_index().to_excel(writer, sheet_name='ah_20min', index=False)
    
    q_20min.reset_index().to_excel(writer, sheet_name='q_20min', index=False)
    
    IWV_20min.reset_index().to_excel(writer, sheet_name='IWV_20min', index=False)
    
    LWP_20min.reset_index().reset_index().to_excel(writer, sheet_name='LWP_20min', index=False)
    
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\20min diff data.xlsx', engine='openpyxl') as writer:
    
    dT_dz_20min_50m.reset_index().to_excel(writer, sheet_name='dT_dz_20min_25m', index=False)
    
    dT_dz_20min_100m.reset_index().to_excel(writer, sheet_name='dT_dz_20min_50m', index=False)
    
    dT_dz_20min_150m.reset_index().to_excel(writer, sheet_name='dT_dz_20min_75m', index=False)
    
    dθv_dz_20min_50m.reset_index().to_excel(writer, sheet_name='dθv_dz_20min_50m', index=False)
    
    dθv_dz_20min_100m.reset_index().to_excel(writer, sheet_name='dθv_dz_20min_100m', index=False)
    
    dθv_dz_20min_150m.reset_index().to_excel(writer, sheet_name='dθv_dz_20min_150m', index=False)
    
    dPExC_355_dz_20min_15m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_20min_15m', index=False)

    dPExC_355_dz_20min_30m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_20min_30m', index=False)
     
    dPExC_355_dz_20min_45m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_20min_45m', index=False)
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\20min PBLH data.xlsx', engine='openpyxl') as writer:

    Ri_20min_50.reset_index().to_excel(writer, sheet_name='Ri_20min_50m', index=False)
    
    Ri_20min_100.reset_index().to_excel(writer, sheet_name='Ri_20min_100m', index=False)
    
    Ri_20min_150.reset_index().to_excel(writer, sheet_name='Ri_20min_150m', index=False)
    
    WDS_50m_20min.reset_index().to_excel(writer, sheet_name='wds_20min_50m', index=False)
    
    WDS_100m_20min.reset_index().to_excel(writer, sheet_name='wds_20min_100m', index=False)
    
    WDS_150m_20min.reset_index().to_excel(writer, sheet_name='wds_20min_150m', index=False)
    
    windshear_20min_50m.reset_index().to_excel(writer, sheet_name='windshear_20min_50m', index=False)
    
    windshear_20min_100m.reset_index().to_excel(writer, sheet_name='windshear_20min_100m', index=False)
    
    windshear_20min_150m.reset_index().to_excel(writer, sheet_name='windshear_20min_150m', index=False)
    
    CBL_20min.reset_index().to_excel(writer, sheet_name='CBL_20min', index=False)
    
    SBL_argmin_20min_50m.reset_index().to_excel(writer, sheet_name='SBL_argmin_20min_50m', index=False)
    
    SBL_argmin_20min_100m.reset_index().to_excel(writer, sheet_name='SBL_argmin_20min_100m', index=False)
    
    SBL_argmin_20min_150m.reset_index().to_excel(writer, sheet_name='SBL_argmin_20min_150m', index=False)
    
    SBL_neg_max_20min_50m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_20min_50m', index=False)
    
    SBL_neg_max_20min_100m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_20min_100m', index=False)
    
    SBL_neg_max_20min_150m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_20min_150m', index=False)
    
    SBL_pos_min_20min_50m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_20min_50m', index=False)
    
    SBL_pos_min_20min_100m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_20min_100m', index=False)
    
    SBL_pos_min_20min_150m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_20min_150m', index=False)
    

##########################5min data output##################
print("输出20min数据完毕，开始输出5min数据")
Wind_matrix_5min.reset_index().to_csv('E:\\Wuhai Data\\从秒级数据处理到的结果\\Wind_matrix_5min.csv', index=False)
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\5min raw data.xlsx', engine='openpyxl') as writer:
    
    horizontal_speed_filled_5min.reset_index().to_excel(writer, sheet_name='hori_speed_5min', index=False)
    
    smoothed_hori_speed_5min.reset_index().to_excel(writer, sheet_name='smoothed_hori_speed_5min', index=False)
    
    wind_filled_w_5min.reset_index().to_excel(writer, sheet_name='w_5min', index=False)

    PExC_355_data_5min.reset_index().to_excel(writer, sheet_name='PExC_355_5min', index=False)
    
    TKE_5min.reset_index().to_excel(writer, sheet_name='TKE_5min', index=False)
    
    interpolated_θv_5min_25.reset_index().to_excel(writer, sheet_name='θv_5min', index=False)
    
    T_5min_25m.reset_index().to_excel(writer, sheet_name='T_5min', index=False)
    
    rh_5min_mean.reset_index().to_excel(writer, sheet_name='rh_5min', index=False)
    
    ah_5min_mean.reset_index().to_excel(writer, sheet_name='ah_5min', index=False)
    
    q_5min.reset_index().reset_index().to_excel(writer, sheet_name='q_5min', index=False)
    
    IWV_5min.reset_index().reset_index().to_excel(writer, sheet_name='IWV_5min', index=False)
    
    LWP_5min.reset_index().reset_index().to_excel(writer, sheet_name='LWP_5min', index=False)
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\5min diff data.xlsx', engine='openpyxl') as writer:
    
    dT_dz_5min_50m.reset_index().to_excel(writer, sheet_name='dT_dz_5min_25m', index=False)
    
    dT_dz_5min_100m.reset_index().to_excel(writer, sheet_name='dT_dz_5min_50m', index=False)
    
    dT_dz_5min_150m.reset_index().to_excel(writer, sheet_name='dT_dz_5min_75m', index=False)
    
    dθv_dz_5min_50m.reset_index().to_excel(writer, sheet_name='dθv_dz_5min_50m', index=False)
    
    dθv_dz_5min_100m.reset_index().to_excel(writer, sheet_name='dθv_dz_5min_100m', index=False)
    
    dθv_dz_5min_150m.reset_index().to_excel(writer, sheet_name='dθv_dz_5min_150m', index=False)
    
    dPExC_355_dz_5min_15m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_5min_15m', index=False)

    dPExC_355_dz_5min_30m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_5min_30m', index=False)
     
    dPExC_355_dz_5min_45m.reset_index().to_excel(writer, sheet_name='dPExC_355_dz_5min_45m', index=False)
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\5min PBLH data.xlsx', engine='openpyxl') as writer:

    Ri_5min_50.reset_index().to_excel(writer, sheet_name='Ri_5min_50m', index=False)
    
    Ri_5min_100.reset_index().to_excel(writer, sheet_name='Ri_5min_100m', index=False)
    
    Ri_5min_150.reset_index().to_excel(writer, sheet_name='Ri_5min_150m', index=False)
    
    WDS_50m_5min.reset_index().to_excel(writer, sheet_name='wds_5min_50m', index=False)
    
    WDS_100m_5min.reset_index().to_excel(writer, sheet_name='wds_5min_100m', index=False)
    
    WDS_150m_5min.reset_index().to_excel(writer, sheet_name='wds_5min_150m', index=False)
    
    windshear_5min_50m.reset_index().to_excel(writer, sheet_name='windshear_5min_50m', index=False)
    
    windshear_5min_100m.reset_index().to_excel(writer, sheet_name='windshear_5min_100m', index=False)
    
    windshear_5min_150m.reset_index().to_excel(writer, sheet_name='windshear_5min_150m', index=False)
    
    CBL_5min.reset_index().to_excel(writer, sheet_name='CBL_5min', index=False)
    
    SBL_argmin_5min_50m.reset_index().to_excel(writer, sheet_name='SBL_argmin_5min_50m', index=False)
    
    SBL_argmin_5min_100m.reset_index().to_excel(writer, sheet_name='SBL_argmin_5min_100m', index=False)
    
    SBL_argmin_5min_150m.reset_index().to_excel(writer, sheet_name='SBL_argmin_5min_150m', index=False)
    
    SBL_neg_max_5min_50m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_5min_50m', index=False)
    
    SBL_neg_max_5min_100m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_5min_100m', index=False)
    
    SBL_neg_max_5min_150m.reset_index().to_excel(writer, sheet_name='SBL_neg_max_5min_150m', index=False)
    
    SBL_pos_min_5min_50m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_5min_50m', index=False)
    
    SBL_pos_min_5min_100m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_5min_100m', index=False)
    
    SBL_pos_min_5min_150m.reset_index().to_excel(writer, sheet_name='SBL_pos_min_5min_150m', index=False)
    
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\1min thermal PBLH data.xlsx', engine='openpyxl') as writer:   
    
    SBL_argmin_1min_50m.reset_index().to_excel(writer, sheet_name='SBL_argmin_1min_50m', index=False)
    
    SBL_argmin_1min_100m.reset_index().to_excel(writer, sheet_name='SBL_argmin_1min_100m', index=False)
    
    SBL_argmin_1min_150m.reset_index().to_excel(writer, sheet_name='SBL_argmin_1min_150m', index=False)
    
    CBL_1min.reset_index().to_excel(writer, sheet_name='CBL_1min', index=False)
    
    dθv_dz_1min_50m.reset_index().to_excel(writer, sheet_name='dθv_dz_1min_50m', index=False)
    
    dθv_dz_1min_100m.reset_index().to_excel(writer, sheet_name='dθv_dz_1min_100m', index=False)
    
    dθv_dz_1min_150m.reset_index().to_excel(writer, sheet_name='dθv_dz_1min_150m', index=False)
    
    θv_1min.reset_index().to_excel(writer, sheet_name='θv_1min', index=False)
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\1min dTdz data.xlsx', engine='openpyxl') as writer:       
    
    dT_dz_1min_50m.reset_index().to_excel(writer, sheet_name='dT_dz_1min_50m', index=False)
    
    dT_dz_1min_100m.reset_index().to_excel(writer, sheet_name='dT_dz_1min_100m', index=False)
    
    dT_dz_1min_150m.reset_index().to_excel(writer, sheet_name='dT_dz_1min_150m', index=False)
    
with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\WDS data.xlsx', engine='openpyxl') as writer:    
    
    WDS_50m_60min.reset_index().to_excel(writer, sheet_name='wds_60min_50m', index=False)
    
    WDS_100m_60min.reset_index().to_excel(writer, sheet_name='wds_60min_100m', index=False)
    
    WDS_150m_60min.reset_index().to_excel(writer, sheet_name='wds_60min_150m', index=False)
    
    WDS_50m_20min.reset_index().to_excel(writer, sheet_name='wds_20min_50m', index=False)
    
    WDS_100m_20min.reset_index().to_excel(writer, sheet_name='wds_20min_100m', index=False)
    
    WDS_150m_20min.reset_index().to_excel(writer, sheet_name='wds_20min_150m', index=False)
    
    WDS_50m_5min.reset_index().to_excel(writer, sheet_name='wds_5min_50m', index=False)
    
    WDS_100m_5min.reset_index().to_excel(writer, sheet_name='wds_5min_100m', index=False)
    
    WDS_150m_5min.reset_index().to_excel(writer, sheet_name='wds_5min_150m', index=False)

with pd.ExcelWriter('E:\\Wuhai Data\\从秒级数据处理到的结果\\Bulk Richardson Number.xlsx', engine='openpyxl') as writer:   
    
    Ri_bulk_60min_50m.reset_index().to_excel(writer, sheet_name='Ri_bulk_60min_50m', index=False)
    
    Ri_bulk_60min_100m.reset_index().to_excel(writer, sheet_name='Ri_bulk_60min_100m', index=False)
    
    Ri_bulk_60min_150m.reset_index().to_excel(writer, sheet_name='Ri_bulk_60min_150m', index=False)
    
    Ri_bulk_20min_50m.reset_index().to_excel(writer, sheet_name='Ri_bulk_20min_50m', index=False)
    
    Ri_bulk_20min_100m.reset_index().to_excel(writer, sheet_name='Ri_bulk_20min_100m', index=False)
    
    Ri_bulk_20min_150m.reset_index().to_excel(writer, sheet_name='Ri_bulk_20min_150m', index=False)
    
    Ri_bulk_5min_50m.reset_index().to_excel(writer, sheet_name='Ri_bulk_5min_50m', index=False)
    
    Ri_bulk_5min_100m.reset_index().to_excel(writer, sheet_name='Ri_bulk_5min_100m', index=False)
    
    Ri_bulk_5min_150m.reset_index().to_excel(writer, sheet_name='Ri_bulk_5min_150m', index=False)

with pd.ExcelWriter('G:\\2022BJ-wind_60min1.xlsx', engine='openpyxl') as writer:   

    #wind_60min.reset_index().to_excel(writer, sheet_name='wind_60min', index=False)

    TKE_60min.reset_index().to_excel(writer, sheet_name='TKE_60min', index=False)
#完成后打开输出目录
os.startfile("E:\\Wuhai Data\\从秒级数据处理到的结果\\")
print("数据处理完毕！")


wind_1min.to_csv("I:\\wind_1min.csv",index=False)
wind_5min.to_csv("I:\\wind_5min.csv",index=False)
wind_15min.to_csv("I:\\wind_15min.csv",index=False)
wind_30min.to_csv("I:\\wind_30min.csv",index=False)