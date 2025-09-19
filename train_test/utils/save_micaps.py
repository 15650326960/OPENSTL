import datetime

import numpy as np


def write_to_micaps4(save_path, data, start_time, shici, rows, cols, latitude, longitude, RowResolution, ColResolution, effectiveNum=2, title=None):
    start_lat = round(latitude,5) - ColResolution / 100000.0 * rows#* (rows-1)
    end_lat = round(latitude,5)
    start_lon = round(longitude,5)
    end_lon = round(longitude,5) + RowResolution / 100000.0 * cols #* (cols-1)


    time = start_time + datetime.timedelta(minutes=(shici + 1)*2)
    time = time.strftime('%Y%m%d%H%M')
    sc = str(shici + 1).zfill(3)
    str_time = start_time.strftime('%Y%m%d%H%M')

    title = "逐2分钟x波段CR" + str_time + "预报" + ":" + time[6:]
    nlon = 1200  # lon格点数
    nlat = 1200  # lat格点数
    slon = start_lon  # start_lon
    slat = start_lat  # start_lat
    elon = end_lon  # end_lon
    elat = end_lat  # end_lat
    dlon = (elon-slon) / 1200 #+ RowResolution / 100000.0) / 1200   # delta_lon
    dlat = (elat-slat) / 1200 #+ ColResolution / 100000.0) / 1200  # delta_lat
    level = 0  # 变量的高度层次
    year = str_time[0:4]
    month = str_time[4:6]
    day = str_time[6:8]
    hour = str_time[8:10]

    title = ("diamond 4 " + title + "\n"
             + year + " " + month + " " + day + " " + hour + " " + str(shici + 1) + " 0 \n"
             + "{:.6f}".format(dlon) + " " + "{:.6f}".format(dlat) + " " + "{:.6f}".format(
                slon) + " " + "{:.6f}".format(elon - dlon) + " "
             + "{:.6f}".format(slat) + " " + "{:.6f}".format(elat - dlat) + " " + str(nlon) + " " + str(nlat) + " "
             + str("1 0 100 1.0 0"))

    format_str = "%." + str(effectiveNum) + "f "
    np.savetxt(save_path, data, delimiter=' ', fmt=format_str, header=title, comments='')
