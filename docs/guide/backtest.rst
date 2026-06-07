回测模块
========

`betalens.backtest` 聚焦「调仓权重 → 日度净值」的最短路径，以 :class:`BacktestBase` 为核心。

权重矩阵规范
------------

- 行索引为时间类型（DatetimeIndex），表示调仓时间
- 列代表证券代码；可选 ``cash`` 列保留显式现金头寸（缺失时仅告警，不影响计算）
- 各行为目标权重（多头合计 1、空头合计 -1），框架按成本价归一

.. note::

   **整数手成交**：A 股最小买入单位为一手（默认 100 股），非整数手无法成交。
   框架在每个调仓日按当期总资产把目标权重换算为可成交的整数手，凑不齐整手的
   余款（含取余 + 未投资部分）自动转入现金。真正用于计算收益/持仓的是
   ``actual_weight``（整数手股票权重 + ``cash`` 列，逐行合计为 1），而非输入的目标
   ``weight``。一手股数由 ``lot_size`` 参数控制。


.. code-block:: python

   import pandas as pd

   weights = pd.DataFrame(
       data=[[0.5, 0.5, 0.0], [-0.5, 0.5, 1.0]],
       index=pd.to_datetime(["2024-01-31 15:00", "2024-02-29 15:00"]),
       columns=["000001.SZ", "000002.SZ", "cash"],
   )

回测执行
--------

.. code-block:: python

   from betalens.backtest import BacktestBase

   engine = BacktestBase(
       weight=weights,
       symbol="Demo",
       amount=5_000_000,
       ftc=0.0002,  # 固定费用
       ptc=0.0,     # 百分比费用
       lot_size=100,  # 一手股数，整数手成交
       verbose=True,
       check_trade_status=True,        # 换仓前检查交易状态（默认开）
       trade_status_mode="to_cash",    # 停牌持仓的处理模式
       trade_status_table="trade_status",  # 交易状态表名
   )

   nav = engine.nav                   # 日度净值 Series
   daily_amount = engine.daily_amount # 每日市值
   position = engine.position         # 每日持仓数量
   actual_weight = engine.actual_weight  # 实际成交权重（整数手 + cash）
   trade_status = engine.trade_status        # 调仓日交易状态长表（审计）
   trade_status_matrix = engine.trade_status_matrix  # 调仓日 × code 状态矩阵

主要属性
--------

- ``nav``: 归一化净值序列（基于初始金额）
- ``actual_weight``: 实际成交权重 DataFrame（整数手股票权重 + ``cash`` 列，逐行合计 1）
- ``daily_amount``: 每日市值序列
- ``position``: 每日持仓数量 DataFrame（股票列均为 ``lot_size`` 的整数倍）
- ``amount``: 各调仓日（分配前）总资产 Series
- ``cost_price``: 调仓成本价 DataFrame
- ``cost_ret``: 调仓区间收益率 DataFrame
- ``start`` / ``end``: 回测起止时间
- ``initial_amount``: 初始资金
- ``lot_size``: 一手股数（默认 100）
- ``trade_status``: 调仓日交易状态长表 DataFrame（审计用，关闭检查或查询失败时为 ``None``）
- ``trade_status_matrix``: 调仓日 × code 的状态矩阵（值为 -1/0/1，index 对齐 ``weight``）

内部步骤
--------

1. **get_rebalance_data**：
   - 调用 :class:`betalens.datafeed.Datafeed` 获取调仓日收盘价
   - 验证日期匹配和标的匹配
   - 构造 ``cost_price`` 和 ``cost_ret``

2. **get_position_data**（逐期迭代）：
   - 按调仓日顺序迭代：用当期总资产把目标权重换算为整数手
     （``np.trunc(目标股数 / lot_size)``，向零截断兼容做空），余款转 ``cash``
   - 生成 ``actual_weight``（实际成交权重）与 ``amount``（各期总资产）
   - 当期手数依赖当期总资产、总资产又由上期实际持仓收益递推，存在逐期依赖，
     无法向量化（故不再用 ``cumprod``）

3. **get_daily_position_data**：
   - 用 ``actual_weight`` 而非目标 ``weight`` 计算持仓数量
   - 查询调仓区间的全部交易日，前向填充头寸
   - 现金以「份额=现金额、价=1」参与每日重估，不再被丢弃
   - 得到 ``daily_amount`` 和 ``nav``

异常处理
--------

框架提供详细的异常类和错误信息：

.. code-block:: python

   from betalens.backtest import (
       BacktestDataError,
       DateMismatchError,
       CodeMismatchError
   )

   try:
       engine = BacktestBase(weight=weights, symbol="Demo", amount=1_000_000)
   except DateMismatchError as e:
       print(f"日期不匹配: {e}")
   except CodeMismatchError as e:
       print(f"标的不匹配: {e}")
   except BacktestDataError as e:
       print(f"数据错误: {e}")

数据验证
--------

框架自动执行多层数据验证：

1. **权重输入验证**：检查DataFrame格式、索引类型、数值类型、NaN/Inf值
2. **查询结果验证**：检查数据库返回的列是否完整
3. **日期匹配检查**：确保权重日期在数据库中有对应数据
4. **标的匹配检查**：警告并置零缺失标的的权重
5. **计算前验证**：检查输入数据的完整性

交易状态处理
------------

回测前会从 ``trade_status`` 表（见 :doc:`datafeed`）提取各调仓日的个券交易状态，对**停牌**
持仓按 ``trade_status_mode`` 处理后再撮合。该流程由独立方法 ``get_trade_status`` 完成，
在 ``get_rebalance_data`` 之前执行，并把结果留存到 ``engine.trade_status``（长表）与
``engine.trade_status_matrix``（调仓日 × code 矩阵，值 -1/0/1）供审计。

仅 ``value == 0``（停牌）视为需要处理的异常；``value == -1``（未上市）交由标的匹配检查
处理。五种模式：

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 模式
     - 行为
   * - ``to_cash``
     - **默认**。停牌股当期权重置 0，资金留现金（假设买卖失败）
   * - ``hold``
     - 停牌无法调仓，沿用上一调仓日整行权重（持仓被动冻结）；首期退化为 ``to_cash``
   * - ``redistribute``
     - 停牌股权重清零后，整行剩余权重按比例重新归一到当期可交易持仓
   * - ``as_normal``
     - 忽略停牌，假设仍能正常买卖，仅统计提示，不改权重
   * - ``report_only``
     - 仅统计提示，不改动权重

.. code-block:: python

   # 关闭交易状态检查（按纯权重撮合）
   engine = BacktestBase(weight=weights, symbol="Demo", amount=1_000_000,
                         check_trade_status=False)

   # 停牌持仓权重再归一到可交易标的
   engine = BacktestBase(weight=weights, symbol="Demo", amount=1_000_000,
                         trade_status_mode="redistribute")

查询失败或表中无数据时，流程降级为「按正常交易处理」并告警，不中断回测。

常见扩展
--------

**多资产权重展开**：

.. code-block:: python

   from betalens.factor.factor import get_single_factor_weight

   # 从因子模块生成的权重可直接使用
   weights = get_single_factor_weight(labeled_pool, params)
   weights["cash"] = 0
   engine = BacktestBase(weight=weights, symbol="Factor", amount=1_000_000)

**成本拆解**：

.. code-block:: python

   # 访问每次调仓对应的成交价
   cost_price = engine.cost_price  # MultiIndex: (input_ts, datetime)

**持仓导出**：

.. code-block:: python

   # 每日持仓数量（股数）
   position = engine.position
   position.to_csv("position.csv")

**连接绩效分析**：

.. code-block:: python

   from betalens.analyst import PortfolioAnalyzer, ReportExporter

   analyzer = PortfolioAnalyzer(engine.nav)
   print(f"Sharpe: {analyzer.sharpe_ratio():.4f}")
   print(f"Max Drawdown: {analyzer.max_drawdown():.2%}")

   exporter = ReportExporter(analyzer)
   exporter.generate_annual_report()

若需要更复杂的交易撮合或期货保证金处理，可在此基础上继承 ``BacktestBase`` 扩展新的回测基类。


