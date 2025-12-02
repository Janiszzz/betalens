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

快速连接数据库
--------------

.. code-block:: python

   from betalens.datafeed import Datafeed, get_database_config

   df = Datafeed("daily_market_data")
   print(df.sheet)  # => daily_market_data

   # 临时覆盖配置
   df_dev = Datafeed(
       "factor_store",
       db_config={**get_database_config(), "dbname": "beta_dev"}
   )

`Datafeed` 会在构造时打开连接并创建日志文件。使用完成后可显式调用 ``df.close()``（如未定义，将在回收期自动关闭）。

常用查询
--------

.. code-block:: python

   from betalens.datafeed import query_time_range, query_nearest_before

   history = query_time_range(
       cursor=df.cursor,
       conn=df.conn,
       codes=["000001.SZ"],
       start_date="2024-01-01",
       end_date="2024-03-31",
       metric="收盘价(元)",
   )

   last_report = query_nearest_before(
       cursor=df.cursor,
       conn=df.conn,
       codes=["000001.SZ"],
       datetimes=["2024-03-31"],
       metric="归母净利润",
       time_tolerance=365 * 24,
   )

高阶 API（例如 :meth:`Datafeed.query_time_range`）内部委托以上函数，同时自动记录日志。

Excel & EDE 导入
----------------

.. code-block:: python

   insert_result = df.insert_csv_file(
       filepath=r".\exports\margin_trading.csv",
       config={
           "key_columns": ["datetime", "code", "metric"],
           "key_value_mapping": {"价值(元)": "value"},
           "additional_fields": {"source": "WIND"},
           "apply_time_alignment": True,
       },
       mode="incremental",
   )
   print(insert_result)

EDE 文件处理流程（``insert_ede_file``）会自动抽取列名中的日期、指标、单位与元数据，并调用 :func:`integration.incremental_insert`。

批量校验
--------

.. code-block:: python

   from betalens.datafeed import DataValidator, FillStrategy

   validator = DataValidator(
       required_columns=["datetime", "code", "value"],
       datetime_column="datetime",
       fill_strategy=FillStrategy.FORWARD_FILL,
   )
   cleaned_df, report = validator.validate(df=history)
   report.pretty_print()

`validation` 模块也提供函数式接口（``validate_and_fix``、``fix_datetime_column`` 等），便于在 ETL 脚本中串联。

数据目录批处理
--------------

.. code-block:: python

   from betalens.datafeed import process_directory_tree

   summary = process_directory_tree(
       root_dir=r".\exports\macro",
       file_patterns=["*.xlsx"],
       handler=df.insert_ede_file,
       max_workers=4,
   )
   print(summary["success"], summary["errors"])

注意事项
--------

- 在离线测试或生成文档时，可通过 ``autodoc_mock_imports`` 忽略 ``psycopg2``、``WindPy``。
- 对于高频查询，优先使用 ``query_time_range`` + ``pivot_to_wide`` 组合，避免框架在内部重复透视。
- 所有插入方法默认开启事务；在长批次任务中请定期 ``conn.commit()`` 或使用 ``with Datafeed(...) as df`` 语法。

更多 API 细节请参阅 :doc:`../api/datafeed`。


