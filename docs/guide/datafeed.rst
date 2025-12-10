Datafeed 使用指南
=================

`betalens.datafeed` 是一套薄封装的数据中台，负责 Excel/EDE 文件清洗、验证、数据库读写以及 Wind 数据接入。

模块结构
--------

- :class:`betalens.datafeed.Datafeed`：统一入口，维护 PostgreSQL 连接与日志
- ``excel``：读写 Excel/CSV、创建目录树、批量转换
- ``validation``：缺失值处理、日期列校验、枚举约束
- ``query``：时间序列与最近点查询、收益率计算
- ``integration``：增量写入、目录树批处理、错误回滚
- ``wind_ingest`` / ``ede_processor``：数据源适配器
- ``config``：配置文件管理

快速连接数据库
--------------

.. code-block:: python

   from betalens.datafeed import Datafeed

   # 使用默认配置（从config.json读取）
   df = Datafeed("daily_market_data")
   print(df.sheet)  # => daily_market_data

   # 自定义数据库配置
   df_dev = Datafeed(
       "factor_store",
       db_config={
           'dbname': 'beta_dev',
           'user': 'postgres',
           'password': 'your_password',
           'host': 'localhost',
           'port': '5432'
       }
   )

   # 使用完毕后关闭连接
   df.close()

`Datafeed` 会在构造时打开连接并创建日志文件。

常用查询
--------

时间范围查询
~~~~~~~~~~~~

.. code-block:: python

   # 查询时间范围内的数据
   history = df.query_time_range(
       codes=["000001.SZ", "000002.SZ"],
       start_date="2024-01-01",
       end_date="2024-03-31",
       metric="收盘价(元)"
   )

最近时点查询
~~~~~~~~~~~~

.. code-block:: python

   # 查询每个时点之后最近的有效值（用于获取调仓成本价）
   params = {
       'codes': ["000001.SZ", "000002.SZ"],
       'datetimes': ["2024-01-31 15:00:00", "2024-02-29 15:00:00"],
       'metric': "收盘价(元)",
       'time_tolerance': 24  # 时间容差（小时）
   }
   cost_price = df.query_nearest_after(params)

   # 查询每个时点之前最近的有效值（用于获取历史特征）
   params = {
       'codes': ["000001.SZ"],
       'datetimes': ["2024-03-31 10:00:00"],
       'metric': "归母净利润",
       'time_tolerance': 365 * 24  # 1年
   }
   last_report = df.query_nearest_before(params)

返回的 DataFrame 包含：

- ``code``: 代码
- ``input_ts``: 输入时间戳
- ``datetime``: 匹配到的数据时间戳
- ``diff_hours``: 时间差（小时）
- ``{metric}``: 数据值
- ``name``: 名称

辅助查询函数
~~~~~~~~~~~~

.. code-block:: python

   # 获取最新数据日期
   latest = df.get_latest_date(code="000001.SZ", metric="收盘价(元)")

   # 获取可用日期列表
   dates = df.get_available_dates(
       code="000001.SZ",
       metric="收盘价(元)",
       start_date="2024-01-01",
       end_date="2024-12-31"
   )

Excel & EDE 导入
----------------

CSV/Excel 文件导入
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   insert_result = df.insert_csv_file(
       filepath=r".\exports\margin_trading.csv",
       config={
           "key_columns": ["datetime", "code", "metric"],
           "key_value_mapping": {"价值(元)": "value"},
           "additional_fields": {"source": "WIND"},
           "apply_time_alignment": True,
       },
       mode="incremental"  # 或 "insert"
   )
   print(insert_result)

EDE 文件导入
~~~~~~~~~~~~

.. code-block:: python

   # 处理EDE格式的Excel文件
   result = df.insert_ede_file(
       filepath='EDE20251103.xlsx',
       date_from='filename',  # 或 'metric'
       mode='incremental'
   )
   print(f"新增{result['new_rows']}行，跳过{result['skipped_rows']}行")

EDE格式特征：

- 第一列：证券代码
- 第二列：证券简称
- 第三列及之后：指标列，格式为"指标名 [元数据] 值类型 [元数据] 单位"

批量导入
~~~~~~~~

.. code-block:: python

   # 批量处理目录下的文件
   summary = df.batch_process_excel_files(
       folder_path=r".\exports\macro",
       config={...},
       file_pattern="*.xlsx",
       recursive=True,
       mode='incremental'
   )
   print(summary)

Wind 数据抓取
-------------

.. code-block:: python

   # 抓取股票日行情
   result = df.ingest_wind_daily_market(
       codes=['000001.SZ', '000002.SZ'],
       start_date='2024-01-01',
       end_date='2024-01-31',
       fields=None,  # 使用默认字段
       asset_type='stock',
       mode='incremental'
   )

   # 便捷方法
   df.ingest_wind_daily_index(codes, start_date, end_date)  # 指数
   df.ingest_wind_daily_fund(codes, start_date, end_date)   # 基金
   df.ingest_wind_daily_bond(codes, start_date, end_date)   # 债券

数据验证
--------

.. code-block:: python

   # 验证并修复DataFrame
   cleaned_df, report = df.validate_dataframe(
       df=history,
       validations={
           "required_columns": ["datetime", "code", "value"],
           "datetime_column": "datetime",
           "fill_strategy": "forward_fill"
       }
   )

   # 检查Excel文件
   is_valid, errors = df.check_excel_file(
       filepath=r".\data.xlsx",
       checks={"required_columns": ["datetime", "code"]}
   )

增量更新
--------

.. code-block:: python

   import pandas as pd

   # 准备数据
   new_data = pd.DataFrame({
       'datetime': ['2024-01-01 15:00:00'],
       'code': ['000001.SZ'],
       'metric': ['收盘价(元)'],
       'value': [10.5],
       'name': ['平安银行']
   })

   # 增量更新
   new_rows, skipped_rows = df.incremental_update(
       df=new_data,
       date_column='datetime',
       code_column='code',
       metric_column='metric'
   )

交易日辅助函数
--------------

.. code-block:: python

   from betalens.datafeed import get_absolute_trade_days, trade_days_offset

   # 获取交易日序列
   trade_days = get_absolute_trade_days("2024-01-01", "2024-12-31", "D")  # 日频
   trade_days = get_absolute_trade_days("2024-01-01", "2024-12-31", "M")  # 月频
   trade_days = get_absolute_trade_days("2024-01-01", "2024-12-31", "Y")  # 年频

   # 交易日偏移
   from datetime import datetime
   next_day = trade_days_offset(datetime(2024, 1, 1, 10, 0), offset=1)
   prev_day = trade_days_offset(datetime(2024, 1, 1, 10, 0), offset=-1)

注意事项
--------

- 在离线测试或生成文档时，可通过 ``autodoc_mock_imports`` 忽略 ``psycopg2``、``WindPy``。
- 对于高频查询，优先使用 ``query_time_range`` 获取批量数据。
- 所有插入方法默认开启事务；长批次任务请定期 ``conn.commit()``。
- 配置文件 ``config.json`` 从 ``config.example.json`` 复制并修改。

更多 API 细节请参阅 :doc:`../api/datafeed`。


