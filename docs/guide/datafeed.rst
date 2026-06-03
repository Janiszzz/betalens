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

行业分类查询
------------

行业归属是 **point-in-time 状态数据**：某股票从某日起属于某行业，直到下次变更。
betalens 不另造存储模型，而是复用长格式时序表（``industry`` 表），约定：

- ``metric``：分类体系名，如 ``申万一级行业``、``申万二级行业（2021）``
- ``value``：行业代码数值部分（如 ``480301``），便于索引分组
- ``remark``（JSONB）：``{"ind_name", "ind_code", "scheme"}``，存行业中文名等
- ``datetime``：该归属关系的生效时点（最早可知日）

查询语义 = 取 ``datetime <= 查询日`` 的最近一条，天然避免前视偏差。

正查 / 反查
~~~~~~~~~~~

.. code-block:: python

   from betalens.datafeed import query_industry, get_industry_members

   # 正查：某股票在某日所属行业（注意 table_name 默认 'industry'，
   # 申万分类数据在 'industry' 表，须显式指定）
   df = query_industry(
       cursor,
       codes=["000001.SZ"],
       dates=["2023-06-30"],
       scheme="申万一级行业",          # 不带版本后缀 → 自动选版本，见下
       table_name="industry",
   )
   # 返回列：code | query_date | effective_dt | sec_name |
   #         industry_value | ind_name | ind_code | scheme

   # 反查：某日某行业的成分股
   members = get_industry_members(
       cursor, industry="银行", date="2023-06-30",
       scheme="申万一级行业", table_name="industry",
   )

.. _industry-version-autoselect:

版本自动选择（申万 2014/2021 多版本）
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

申万行业分类历经多次改版（2014-02-21、2021-07-30），**同一 6 位行业代码在不同
版本含义不同**。入库时按「计入日期」落到版本族，metric 带版本后缀区分：
``申万一级行业（旧版）`` / ``申万一级行业（2014）`` / ``申万一级行业（2021）``。

查询时无需关心版本——``scheme`` **不带版本后缀**（``申万一级行业``）即触发版本
自动选择：底层用 ``metric LIKE '申万一级行业%'`` 匹配全部版本，配合
``ORDER BY datetime DESC`` 取最近一条，结果**天然落到查询日生效的那个版本**，
无需硬编码任何版本边界日期：

.. code-block:: python

   # 同一只股票，按查询日自动落到对应版本
   df = query_industry(cursor, codes=["000001.SZ"],
                       dates=["2010-06-30", "2018-06-30", "2023-06-30"],
                       scheme="申万一级行业", table_name="industry")
   # 2010 → 旧版（effective_dt=1991-04-03）
   # 2018 → 2014 版（effective_dt=2014-02-21，银行）
   # 2023 → 2021 版（effective_dt=2021-07-30，银行）

   # 带版本后缀 → LIKE 退化为精确匹配，只查该版本
   df = query_industry(cursor, codes=["000001.SZ"], dates=["2023-06-30"],
                       scheme="申万一级行业（2021）", table_name="industry")

   # exact=True → 强制精确匹配（关闭自动选择），对旧无后缀 metric 向后兼容
   df = query_industry(cursor, codes=["000001.SZ"], dates=["2023-06-30"],
                       scheme="申万一级行业", table_name="industry", exact=True)

.. note::

   原理是「版本族记录的生效日天然落在各自版本区间内」，所以匹配全部版本 + 取最近
   一条 = 按日期选版本。将来申万出新版本，**入库时加新后缀即可，查询代码不用动**。

入库长表（申万分类示例）
~~~~~~~~~~~~~~~~~~~~~~~~

把申万长表（``股票代码 / 计入日期 / 行业代码(6位) / 更新日期``）写入 ``industry``
表的要点：

- **6 位行业代码按每两位拆三级**：前 2 位 = 一级、前 4 位 = 二级、全 6 位 = 三级，
  各自一条 metric，``value`` 存对应数值（如 ``48`` / ``4803`` / ``480301``）。
- **证券代码转 wind 格式**：首位 ``6`` → ``.SH``，``0/3`` → ``.SZ``，
  ``4/8/9`` → ``.BJ``（北交所 ``8xxxxx`` / ``9xxxxx`` / ``689xxx`` 等）。
- **中文行业名按版本字典解析**写入 ``remark.ind_name``（同一代码不同版本含义不同，
  须用对应版本字典）；旧版无字典覆盖时留空。
- **``name`` 列（证券中文简称）NOT NULL**：从含 wind 代码↔简称的现表补；历史/退市股
  无简称时用 wind 代码占位。
- ``remark`` 是 dict，``execute_values`` 不自动转 JSONB，入库前注册适配器：
  ``psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)``。
- 用 :func:`~betalens.datafeed.incremental_insert` 按 ``(datetime, code, metric)``
  去重写入，可重复运行。

指数股票池查询
--------------

指数成分股池与行业归属同属 **point-in-time 状态数据**：某指数从某调整日起拥有一组
成分股，直到下次调整。betalens 复用长格式时序表（``index_universe`` 表），**每个生效日
存为一行**，成分股列表整体放入 ``remark``：

- ``code``：指数代码（如 ``000906.SH``）
- ``name``：指数名称（如 ``中证800``）
- ``metric``：固定为 ``universe``，标识成分股池
- ``value``：成分股数量（便于校验）
- ``remark``（JSONB）：``{"index_code", "index_name", "constituents": [...]}``，成分股列表存这里
- ``datetime``：该股票池的生效时点（最早可知日）

查询语义与 :func:`~betalens.datafeed.query_nearest_before` 同构——取 ``datetime <= 查询日``
的最近一条，天然避免前视偏差。

查询成分股列表
~~~~~~~~~~~~~~

.. code-block:: python

   from betalens.datafeed import get_index_universe, get_index_universe_date

   # 返回某指数某日生效的成分股代码列表（point-in-time）
   codes = get_index_universe(cursor, "000906.SH", "2024-03-01")
   # => ['000001.SZ', '000002.SZ', ...]，约 800 只
   # 实际取的是 <=2024-03-01 的最近生效日（如 2023-12-11）

   # 查实际生效的快照日期（便于排查）
   eff_dt = get_index_universe_date(cursor, "000906.SH", "2024-03-01")
   # => Timestamp('2023-12-11 00:00:00')

   # 早于首个生效日 → 返回空列表
   get_index_universe(cursor, "000906.SH", "2000-01-01")   # => []

.. note::

   :func:`~betalens.datafeed.query_nearest_before` 的 SELECT 只返回 ``value``/``name``，
   **不返回 remark**。故 :func:`~betalens.datafeed.get_index_universe` 先用它定位最近
   生效日，再按精确 ``datetime`` 补一条小查询取出 ``remark.constituents``。
   ``cursor`` 兼容 ``RealDictCursor`` 与普通 cursor。

入库宽表（成分进出记录）
~~~~~~~~~~~~~~~~~~~~~~~~

成分进出记录通常是 **宽表快照**：第一列为序号，其余每列列名为生效日期，列下方为该日
成分股 WindCode。脚本 ``makeupdatabase/load_index_universe.py`` 负责整理入库：

- **每个日期列整理成一行**：``datetime``=生效日、``code``=指数代码、``metric='universe'``、
  ``value``=成分股数量、``remark.constituents``=该列非空去重保序的代码列表。
- ``remark`` 是 dict，``execute_values`` 不自动转 JSONB，入库前注册适配器：
  ``psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)``。
- 用 :func:`~betalens.datafeed.incremental_insert` 按 ``(datetime, code, metric)``
  去重写入，可重复运行。

.. code-block:: bash

   # 建表（复用通用 DDL 模板）
   python betalens/datafeed/makeupdatabase/create_database.py --tables index_universe

   # 入库（默认指向中证800；可换其他指数）
   python betalens/datafeed/makeupdatabase/load_index_universe.py \
       --index-code 000906.SH --index-name 中证800 \
       --excel <成分进出记录.xlsx> --sheet Sheet2

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


