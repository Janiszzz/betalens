import pandas as pd
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt
import os
import datetime as dt

# 连接到Wind API
from WindPy import w

w.start()

# 连接到MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["stock_data"]
collection_daily = db["daily_data"]  # 存储日频数据的集合

# 定义要获取的股票代码列表（根据你的需求修改）
codes = ["000906.SH", "512760.SH", "588000.SH", "512530.SH",
         "159708.SZ", "159928.SZ", "159707.SZ", "512000.SH",
         "518880.SH", "511520.SH", "511090.SH"]


# 定义获取日频数据的函数
def get_daily_data(codes):
    for code in codes:
        # 获取日频数据（调整 BarSize=1 表示日线）
        _, data = w.wsi(
            code,  # 股票代码
            "open,high,low,close,volume,amt",  # 获取字段：开盘价、最高价、最低价、收盘价、成交量、成交额
            "2020-01-01 09:00:00",  # 开始时间
            dt.datetime.today().strftime("%Y-%m-%d %H:%M:%S"),  # 结束时间（当前日期）
            "BarSize=1"  # 设置为日线数据
        )

        if data is not None:
            data = data.reset_index(inplace=False)
            data['windcode'] = code

            # 计算移动平均线和标签列
            def f(group, n):
                group['ma' + str(n)] = group['close'].rolling(n).mean()
                group['label' + str(n)] = group['close'] > group['close'].shift(n)
                group['label' + str(n)] = group['label' + str(n)].astype(int)
                return group

            # 计算不同周期的MA和Label
            data = f(data, 5)
            data = f(data, 20)
            data = f(data, 60)

            # 去掉前N行未填充的数据（根据移动平均的周期）
            data = data.iloc[60:]

            # 插入到MongoDB
            records = data.to_dict("records")
            collection_daily.insert_many(records)


# 调用函数获取数据
get_daily_data(codes)
