回测模块
========

``BacktestBase`` 接收带时间戳的目标权重，按成交事件推进账户，输出日频漂移权重、
净值、净损益和交易审计。统一引擎使用后复权分红再投收益，原始价格用于整数成交和费用。

基本用法
--------

.. code-block:: python

   from betalens.backtest import BacktestBase

   bt = BacktestBase(
       weight=weights, symbol="Demo", amount=1_000_000,
       metric="close", end_date="2024-12-31",
       ftc=0.0, ptc=0.001,
       risk={"stop_loss": 0.08, "take_profit": 0.20, "execution_metric": "open"},
   )

阈值仅为配置示例，不是建议参数。``risk=None`` 默认关闭风控。

实验归档与离线看板
------------------

安装 ``betalens[experiments]`` 可使用 LMDB 实验库；安装
``betalens[experiment-viewer]`` 同时提供本地只读看板。归档接口只接收完成的结果，
不会发起回测或查询行情。调用方提供对象中没有的因子身份及完整有效配置。

.. code-block:: python

   from betalens.experiments import ExperimentStore, ExperimentInput

   with ExperimentStore("outputs/experiments.lmdb") as store:
       factor_id = store.register_factor(
           key="momentum", name="动量因子", category="量价",
           description="价格动量的参数与组合研究",
       )
       experiment_id = store.save_backtest(
           bt, analyst=analyst, factor_id=factor_id,
           config=effective_config, source_id="manual-20261110-01",
       )
       # FactorPipeline 的统一结果可直接保存：
       # experiment_id = store.save_run(result, factor_id=factor_id, config=effective_config)

       fixed_id = store.derive_factor(
           factor_id, key="factor_momentum_20", name="20日动量",
           locks={"factor_spec.compute_kwargs.window": 20},
       )
       records = store.query(factor_id=fixed_id, has_raw=True)
       nav = store.read_table(experiment_id, "nav")
       restored = store.to_analyst(experiment_id)
       # 可在研究脚本中更换分析参数；不会修改归档指标。
       revised = store.to_analyst(experiment_id, risk_free_rate=0.02)

       batch = store.save_batch([
           ExperimentInput(result_a, factor_id, config_a, source_id="a"),
           ExperimentInput(result_b, other_factor_id, config_b, source_id="b"),
       ], source_id="parallel-20261110-01")
       # batch["status"] 为 complete / partial / failed；检查 batch["failures"]。

``query`` 返回 ``items/total/page/page_size``；支持因子、批次、类型、参数、配置路径、
指标条件、标签、入库时间和原始数据可用性筛选。
例如 ``metrics={"夏普比率": {"gte": 1.0}}``、
``config={"run.rebal_freq": "W"}``。
``read_table`` 支持 ``offset/limit/columns/start/end``；
``read_extras`` 提取 profiling、图表及附属研究产物。

母因子下的派生条目为平铺关系，不能继续派生，也不能覆盖母因子已锁定的字段。
条目自动展示库内所有匹配实验，包括登记前已入库的记录；不扫描旧文件。
``update_factor`` 仅修改中文名称、说明和分类，不能修改锁字段。

同一 ``source_id`` 在同一因子及批次中重复保存相同内容会返回原 ID；内容变化时抛出
``ConflictError``。批量保存逐项提交并返回失败清单，修正失败项后使用同一批次来源标识重试。
原始表缺失时抛出 ``MissingArtifactError``，不会用空表伪装。
同一批次内相同母因子、有效配置及参数的重复结果合并为一个子实验；内容不一致时报告冲突。
独立的随机种子、回测区间等应包含在有效配置或参数中，跨批次结果始终独立保存。

.. code-block:: bash

   python -m betalens.experiments serve --db outputs/experiments.lmdb --port 8020

在浏览器打开 ``http://127.0.0.1:8020``。前端资源已随包提供，运行时无需 Node.js
或互联网。看板仅提供读取接口；因子命名、登记与说明维护使用 Python API。
开发前端时，在 ``experiment-dashboard/`` 执行 ``npm ci``、``npm run build``，
构建结果写入 Python 包中的静态资源目录。

多个存储实例在同一进程共享环境，跨进程独立打开；写入使用短事务排队。
默认 ``map_size`` 为 128 GiB 虚拟映射上限，不预分配等量物理内存。
容量不足时关闭全部访问进程，再用更大的 ``map_size`` 打开；不会自动跨进程扩容。
库格式带版本号，未识别的版本会拒绝打开。

目标权重和自然漂移
------------------

权重行表示调仓指令，索引必须是唯一时间戳。股票列是代码，现金按股票净敞口补足。
输入按时间排序；NaN、Inf、重复时间和重复代码拒绝。日频输入的每行仍表示一次目标重设，
即使数值相同。稀疏目标之间只传播收益、更新实际权重，不自动取整或再平衡。

无交易时，组合收益为上一时点实际权重乘以标的后复权收益之和；
实际权重随后按 ``w_i * (1+r_i) / (1+R)`` 漂移。日内成交会拆分成交前后收益，
不可直接将日末权重乘当日完整收益。

``weight`` 保存原始输入；``effective_target_weight`` 是状态处理后的目标；
``actual_weight`` 是成交事件后的快照；``daily_weight`` 是扣费后的每日收盘权重。

时间和数据口径
--------------

无时区时间按上海时间解释。``metric`` 默认 ``close``，内置时点为 ``open=09:30``、
``close=15:00``。其他价格字段须在 ``price_time_map`` 配置 ``HH:MM[:SS]`` 时点，
且数据库存在同一记录中的原始价和后复权价。成交匹配不早于目标时点的最近价格，
允许精确匹配；10:00 指定开盘价会匹配下一可用开盘。

价格从 ``market_daily_post_adjusted`` 成对读取 ``metric/raw_metric/adjustment_factor``，
按 ``trade_date`` 建立模拟时点。源 ``available_at`` 保留用于版本策略和审计，
不会被改写成开盘或收盘时间；Datafeed 的 research/strict 合同继续适用。
研究模式的来源推定时间不表示严格 PIT，模拟市场时点也不是字段级历史可得性的证明。

``end_date`` 是明确的成交和估值上界，日期字符串包含该日收盘。省略时，默认结束于
最后一批策略目标完成首次价格匹配的交易日收盘；要继续观察持仓或风控须显式给结束日。
无后续成交价的指令保持 pending；缺失估值只能沿用已知价格并写入 ``valuation_audit``。

同一证券尚未成交的旧策略目标被更新目标替代。不同证券可在不同价格时点执行，
资金按实际发生顺序变化。一次回测内批量查询，不提供预加载或持久化回测行情缓存。

整数成交和费用
--------------

板块取自 ``stock_profile.market``。主板、创业板买入按100股倍数，科创板至少200股、
超过后可按1股递增。限制作用于本次交易增减量；科创板200股增至250股的50股买单不能成交。
卖出支持一次性处理已有零股。不实现单笔申报上限拆单；其他板块/资产类型明确拒绝。

目标交易量向零截断。卖出先执行；买入会预留费用，现金不足时统一缩小买入增量并再次截断，
余款留现金，不二次分配。目标明确允许负现金时保留融资研究假设，不另计融资/借券成本。

每个证券每次非零主动成交收 ``ftc + ptc * abs(成交股数) * 原始成交价``；
无成交不收费。费用扣现金并归属该证券的日损益，首次费用计入收益。

``position`` / ``equivalent_position`` 是分红再投后的等效股数，可能包含小数；
``position_units`` 是后复权持仓单位，不能解释为券商真实股数。跨分红先传播旧仓后复权收益，
再用当时原始价及复权因子换算新交易。除息日新买入不获得此前分红；复权因子变化不记交易、
不重复加现金分红。全部退出时不足1股的虚拟余份折现金，记 ``synthetic_settlements``，不收交易费。

固定止盈止损
------------

每天收盘检查实际持仓，即使策略仅按月调仓。基准是本轮首次实际建仓的后复权价格；
同方向加减仓不重置，完全退出或反向后重建。多头价格收益为 ``close/entry-1``，
空头取其相反数，阈值不含手续费。触及止损或止盈后生成全退出指令，默认下一可用开盘执行，
也可通过 ``risk.execution_metric`` 指定其他价格字段。

收盘观察后只能匹配严格更晚的价格，不能用同一收盘价成交；跳空按实际后续价格成交。
风控只影响触发证券，其他股票继续漂移，退出资金留现金。pending 退出期间仍持仓计收益，
不重复提交或收费。退出完成后只接受目标时间严格晚于退出成交的新原始目标，旧目标不会恢复。

缺少有效新收盘价时跳过该证券检查并审计。触发/成交记录在 ``risk_events``，
``rebalance_log.source`` 区分 strategy/risk。旧信号层 state 止盈止损与统一 risk 同时启用时，
流水线配置校验拒绝；不静默迁移旧信号规则。

停牌模式
--------

交易状态只在原始目标行上处理一次，不冻结实际股数，也不强制检查成交可交易性。
``to_cash`` 置零停牌目标；``hold`` 沿用上一期整行目标；``redistribute`` 在多空同侧
重新分配；``as_normal/report_only`` 保留目标。查询失败告警并保留未知状态。

输出和迁移
----------

``nav`` 包含初始资金基准1；``daily_returns`` 包括首日损益。
``daily_pnl`` / ``daily_pnl_total`` 是含交易费的净损益，与净资产变化一致。
``daily_price`` 为后复权收盘估值价，``daily_raw_price`` 是对应原始价；
``cost_price`` / ``raw_cost_price`` 是成交事件价格快照。

``rebalance_log.price/value`` 表示原始成交价/额，``adjusted_price`` 用于复权收益比较。
``order_log`` 记录未成交、替代或风控阻止的指令；``price_audit`` 保留来源时点和板块。

``dump_to_excel`` 导出上述数据及参数；Analyst 和 Dashboard 以日频漂移权重展示持仓，
主动成交流水计算换手，排除自然漂移、分红再投和虚拟余份结算。

旧 ``time_tolerance``、``lot_size``、``preloaded_*`` 参数已移除，传入会明确报迁移错误。
挖掘不再区分 vector/exact，删除 ``evaluation.engine/time_tolerance``，
可通过 ``evaluation.risk``、因子 ``run.risk`` 传入统一风控配置。
挖掘的输入数据缓存、搜索和窗口算法仍独立存在。
