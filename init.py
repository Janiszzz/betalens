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
import pandas as pd
import psycopg2
import psycopg2.extras
import os


class Datafeed:
    _initialized = False

    def __init__(self, sheetname):
        if not self._initialized:
            self.conn = psycopg2.connect(
                dbname="datafeed",
                user="postgres",
                password="111111",
                host="localhost",
                port="5432"
            )
            self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            self.sheet = sheetname
            self.__class__._initialized = True

    def insert_daily_market_data(self, data):
        for index, row in data.iterrows():
            self.cursor.execute(
                'INSERT INTO daily_market_data (datetime, code, name, metric, value) VALUES (%s, %s, %s, %s, %s);',
                (row['日期'], row['代码'], row['简称'], row['variable'], row['value']))
        self.conn.commit()
        return 0

    def insert_files(self, folder_path, insert_func):
        # 获取文件夹中所有文件的列表
        file_list = os.listdir(folder_path)

        # 读取所有CSV文件并将其存储在一个列表中
        for file in file_list:
            if file.endswith('.CSV'):
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path, encoding='gb2312')
                data = pd.melt(data, id_vars=data.columns[0:3], value_vars=data.columns[3:-1])
                data.loc[data['variable'] == "开盘价(元)", '日期'] += " 09:30:01"
                data.loc[data['variable'] != "开盘价(元)", '日期'] += " 15:00:01"
                data['日期'] = pd.to_datetime(data['日期'])
                insert_func(data, self.conn, self.cursor)
                print(file_path)

        return None

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

#%%
if __name__ == '__main__':
    df = Datafeed("daily_market_data")
    params = {
        'start_date': '2023-01-03 15:00:00',
        'end_date': '2024-01-01 9:00:00',
        'code': ["002431.SZ", "002430.SZ"],
        'metric': "收盘价(元)"
    }
    tmp = df.query_data(params)