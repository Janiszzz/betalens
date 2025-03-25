#%%By Janis 250226
# Todo：
'''
1. 单次存入csv数据
2. WIND API获取日行情并插入
3. 查询任意日期任意字段数据
研究数据库结构：
表1：个券行情数据，A股、A债、基金等的日频行情，列：入库实际时间（现实世界）=最早可用于交易时间（盘前中后）、windcode、中文名、数据性质（收盘价、成交量）、数值、备注（json，含说明等信息），同时计算close to close等收益率数据用于向量化回测和挖掘
    对于开盘价，在当天9点30出现。对于其他价量，最早15.00才可确定
表2：个券基本面数据，个券的财务等基本面，要对齐到券级别。列：入库实际时间（现实世界）=最早可用于交易时间（盘前中后）、数据理论上的发生时间（季报？）、windcode、中文名、数据性质（归母净利润）、数值、备注（json，含说明等信息）
    基本是日频，都发生在财报公布时间，注意公布时间在盘什么时间。理论上的发生时间标记统一到0331等，不关心未来函数，只作为画图展示等。
    如何处理年报次序公布：使用分析师一致预期、使用线性外推，尽量避免非原生数据的入库。
表3：宏观经济数据，入库实际时间（现实世界）=最早可用于交易时间（盘前中后）、数据理论上的发生时间（1月gdp）、windcode、中文名、数据性质（包含均线算子等）、数值、备注（json，含说明等信息）
    类似于表2.
表4：因子库数据，入库实际时间（现实世界）=最早可用于交易时间（盘前中后）、数据编制方式、数值、备注（json，含说明等信息）
投资数据库结构：
出于投资目的，实际不需要历史数据
从研究数据库拉取所需最近一个滚动窗口内的数据，其余全部在线拉取，并直接记录入库时间
'''
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import os
import itertools
from psycopg2.extras import execute_values
from functools import wraps
import time

def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer

class Datafeed(object):
    _initialized = False

    def __init__(self, sheetname):
        if not self._initialized:
            self.conn = psycopg2.connect(
                dbname="datafeed",
                user="postgres",
                password="111111",#你自己的卡密
                host="localhost",
                port="5432"
            )
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            self.sheet = sheetname
            self.__class__._initialized = True

    @staticmethod
    def insert_daily_market_data(self, data, table):

        data = pd.melt(data, id_vars=data.columns[0:3], value_vars=data.columns[3:-1])
        data.loc[data['variable'] == "开盘价(元)", '日期'] += " 09:30:01"
        data.loc[data['variable'] != "开盘价(元)", '日期'] += " 15:00:01"
        data['日期'] = pd.to_datetime(data['日期'])

        df = data[['日期','代码','简称','variable','value']]
        df.columns = ['datetime', 'code', 'name', 'metric', 'value']

        import psycopg2.extras as extras
        tuples = [tuple(x) for x in df.to_numpy()]

        cols = ','.join(list(df.columns))
        # SQL query to execute
        query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
        try:
            extras.execute_values(self.cursor, query, tuples)
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.conn.rollback()
            self.cursor.close()
            return 1
        #print("the dataframe is inserted")
        #self.cursor.close()
        return 0

    @staticmethod
    def insert_daily_index_data(self, data, table):

        data = pd.melt(data, id_vars=data.columns[0:3], value_vars=data.columns[3:-1])
        data.loc[data['variable'] == "开盘价", '日期'] += " 09:30:01"
        data.loc[data['variable'] != "开盘价", '日期'] += " 15:00:01"
        data['日期'] = pd.to_datetime(data['日期'])

        df = data[['日期', '代码', '简称', 'variable', 'value']]
        df.columns = ['datetime', 'code', 'name', 'metric', 'value']

        import psycopg2.extras as extras
        tuples = [tuple(x) for x in df.to_numpy()]

        cols = ','.join(list(df.columns))
        # SQL query to execute
        query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
        try:
            extras.execute_values(self.cursor, query, tuples)
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.conn.rollback()
            self.cursor.close()
            return 1
        # print("the dataframe is inserted")
        # self.cursor.close()
        return 0

    @staticmethod
    def insert_daily_fund_data(self, data, table):

        data = pd.melt(data, id_vars=data.columns[0:3], value_vars=data.columns[3:-1])
        data.loc[data['variable'] == "开盘价(元)", '日期'] += " 09:30:01"
        data.loc[data['variable'] != "开盘价(元)", '日期'] += " 15:00:01"
        data['日期'] = pd.to_datetime(data['日期'])

        df = data[['日期', '代码', '简称', 'variable', 'value']]
        df.columns = ['datetime', 'code', 'name', 'metric', 'value']

        import psycopg2.extras as extras
        tuples = [tuple(x) for x in df.to_numpy()]

        cols = ','.join(list(df.columns))
        # SQL query to execute
        query = "INSERT INTO %s(%s) VALUES %%s" % (table, cols)
        try:
            extras.execute_values(self.cursor, query, tuples)
            self.conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            print("Error: %s" % error)
            self.conn.rollback()
            self.cursor.close()
            return 1
        # print("the dataframe is inserted")
        # self.cursor.close()
        return 0

    def insert_files(self, folder_path, insert_func):
        # 获取文件夹中所有文件的列表
        file_list = os.listdir(folder_path)
        error_list = []
        # 读取所有CSV文件并将其存储在一个列表中
        for file in file_list:
            if file.endswith('.CSV'):
                file_path = os.path.join(folder_path, file)
                try:
                    data = pd.read_csv(file_path, encoding='gb2312')
                except:
                    print(file_path + " Error!")
                    error_list.append(file_path)
                    continue
                if(insert_func(self, data, self.sheet)):
                    print(file_path + " Error!")
                    error_list.append(file_path)
                else:
                    print(file_path+" inserted successfully")

        return error_list



    def check_result(self, result:pd.DataFrame):
        result.drop("note", axis=1, inplace=True)
        result.drop_duplicates(inplace=True)
        result.sort_values(by=['datetime'], inplace=True)
        result.reset_index(drop=True, inplace=True)

        def check(group):
            # 检查 datetime 是否有重复值
            duplicate_dates = group['datetime'].duplicated(keep=False)
            if duplicate_dates.any():
                print("Warning: 发现重复的 datetime 值！")
                # 统计每个重复的日期出现的次数
                date_counts = group['datetime'].value_counts()
                for date, count in date_counts.items():
                    if count > 1:
                        print(f"日期 {date} 出现了 {count} 次。")

        result.groupby(['code', 'metric']).apply(check, include_groups=False)
        return result

    def query_data(self, params=None):
        """
        查询 daily_market_data 表中的数据

        Args:
            conn: 数据库连接
            cursor: 数据库游标
            params (dict): 包含查询条件的字典，支持以下键：
                - 'start_date': 开始日期，格式如 '2023-01-01'
                - 'end_date': 结束日期，格式如 '2024-01-01'
                - 'code': 要查询的代码（股票代码等）[]
                - 'metric': 要查询的指标

        Returns:
            pandas DataFrame: 查询结果
        """
        conditions = []
        params_list = []

        if params is None:
            params = {}

        # 遍历字典中的每个键值对
        for key, value in params.items():
            if value is not None:
                if key == 'start_date':
                    conditions.append("datetime >= '" + value + "' :: TIMESTAMP")
                    params_list.append(value)
                elif key == 'end_date':
                    conditions.append("datetime <= '" + value + "' :: TIMESTAMP")
                    params_list.append(value)
                elif key == 'code':
                    conditions.append("(code = '" + "' OR code = '".join(value) + "')")
                    params_list.append(value)
                elif key == 'metric':
                    conditions.append("metric = '" + value + "'")
                    params_list.append(value)
                elif key == 'label_start_date':
                    conditions.append("label_datetime >= '" + value + "' :: TIMESTAMP")
                    params_list.append(value)
                elif key == 'label_end_date':
                    conditions.append("label_datetime <= '" + value + "' :: TIMESTAMP")
                    params_list.append(value)

        # 构建查询语句
        query = "SELECT * FROM " + self.sheet
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # 执行查询
        print(query)
        self.cursor.execute(query)

        # 将结果转换为 DataFrame
        result = self.check_result(pd.DataFrame(self.cursor.fetchall()))

        return result

    @func_timer
    def query_nearest_after(self, params=None):
        """
        根据输入时间戳序列查找每个时点之后最近的有效值

        Args:
            params (dict): 必须包含以下键：
                - codes: 代码列表（必须与datetimes等长）
                - datetimes: 目标时间戳列表（格式：'YYYY-MM-DD HH:MM'）
                - metric: 查询的指标名称
                - time_tolerance: 允许的最大时间间隔（单位：小时，默认不限制）

        Returns:
            DataFrame: 包含以下列：
                code | input_datetime | matched_datetime | time_diff_hours | value
        """
        # 参数校验
        required_keys = ['codes', 'datetimes', 'metric']
        if not all(k in params for k in required_keys):
            raise ValueError(f"必须提供参数: {required_keys}")

        def gen_pairs():
            for code, dt in itertools.product(params['codes'], params['datetimes']):
                yield (code, dt)

        # 生成输入数据占位符 ✅ 防注入处理
        input_tuples = list(gen_pairs())
        value_placeholders = ', '.join(['(%s, %s::TIMESTAMP)'] * len(input_tuples))

        # 构建核心查询 ✅ 包含时间间隔计算
        sql = f"""WITH input_data (code, input_ts) AS (
            VALUES {value_placeholders}
        ),
        candidate_data AS (
            SELECT
                i.code,
                i.input_ts,
                t.datetime AS matched_ts,
                EXTRACT(EPOCH FROM (t.datetime - i.input_ts))/3600 AS diff_hours,
                t.value,
                ROW_NUMBER() OVER (
                    PARTITION BY i.code, i.input_ts 
                    ORDER BY t.datetime ASC  -- ✅ 找之后最近的
                ) AS rn
            FROM input_data i
            LEFT JOIN {self.sheet} t
                ON i.code = t.code
                AND t.datetime >= i.input_ts  -- ✅ 关键过滤条件
                AND t.metric = %s
                {"AND (t.datetime - i.input_ts) <= %s * INTERVAL '1 hour'"
        if params.get('time_tolerance') else ''}
        )
        SELECT 
            code,
            input_ts,
            matched_ts,
            diff_hours,
            value
        FROM candidate_data
        WHERE rn = 1
        """

        # 构造参数列表 ✅ 安全传参
        params_list = []
        for code, dt in input_tuples:
            params_list.extend([code, dt])
        params_list.append(params['metric'])
        if 'time_tolerance' in params:
            params_list.append(params['time_tolerance'])

        # 执行查询
        self.cursor.execute(sql, params_list)
        df = pd.DataFrame(self.cursor.fetchall())
        return df
#%%
def get_interval(df, start=None, end=None):
    # 假设 df 的索引是时间序列
    if isinstance(df.index, pd.DatetimeIndex):
        if start is None and end is None:
            return df  # 返回完整数据框，因为没有时间限制

        elif start is not None and end is not None:
            return df.loc[start:end]

        elif start is not None:
            # 筛选从 start 到最后一个时间点的数据
            return df.loc[start:]

        else:  # end is not None
            # 筛选从第一个时间点到 end 的数据
            return df.loc[:end]

    # 如果没有时间索引，假设有一个时间列名为 'datetime'
    elif 'datetime' in df.columns:
        mask = []

        if start is not None and end is not None:
            mask = df['datetime'].between(start, end)
        elif start is not None:
            mask = df['datetime'] >= start
        else:  # end is not None
            mask = df['datetime'] <= end

        return df[mask]

    else:
        raise ValueError("DataFrame 必须包含时间索引或 'datetime' 列")

#%%
if __name__ == '__main__':
    #error_result = df.insert_files(r"C:\Users\Janis\Desktop\基金", df.insert_daily_fund_data)
    #pd.DataFrame(error_result).to_excel("error_result2.xlsx")

    # 虚拟的权重序列
    weights = pd.DataFrame(0, index=pd.date_range(start='2010-01-01 10:00:00', end='2025-01-01 10:00:00', freq='D'),columns=['000279.OF', '000592.OF', '000824.OF', '000916.OF', '001076.OF',
       '001188.OF', '001255.OF', '001537.OF'])

    db = Datafeed("daily_market_data")

    params = {
        'codes': ['000010.SZ','000001.SZ','000002.SZ','000003.SZ',],
        'datetimes': weights.index,
        'metric': "收盘价(元)",
        # 'time_tolerance': 48
    }
    tmp = db.query_nearest_after(params)  # panel data

    '''params = {
        'start_date': '2023-01-03 15:00:00',
        'end_date': '2024-01-01 9:00:00',
        'code': ["000010.SZ"],
        'metric': "收盘价(元)"
    }
    tmp = db.query_data(params)'''