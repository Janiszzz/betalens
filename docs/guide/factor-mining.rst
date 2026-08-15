因子参数挖掘
============

``betalens.factor.mining`` 提供通用参数挖掘框架，适合把因子的 ``window``、分组数、择时参数等做系统化搜索。它复用标准流水线：缓存数据、计算因子、PIT 股票池过滤、分组、权重、回测、指标排序。

因子侧 hook
-----------

参与挖掘的因子脚本通常提供：

.. code-block:: python

   def make_mining_spec(params):
       return FactorSpec(
           name="DISP",
           inputs={"close_wide": "收盘价(元)"},
           compute=compute_disp,
           compute_kwargs={"window": int(params["window"])},
           direction="negative",
           table_name="daily_market",
           index_code="000906.SH",
       )

   def mining_gid(params):
       return f"window={params['window']}"

   def mining_warmup_days(params):
       return int(params["window"])

   def mining_weight_hook(weights, task):
       return weights

   def mining_valid_report(params, rank, output_dir, start_date, end_date):
       return None

必需的是 ``make_mining_spec``；其他 hook 只有在需要稳定命名、预热期、权重过滤或验证报告时提供。

单段参数扫描
------------

.. code-block:: python

   from betalens.factor.mining import ParameterSweepConfig, run_parameter_sweep

   config = ParameterSweepConfig(
       factor_module="factor_DISP",
       output_dir="outputs/mining/sweep_window",
       span=("2020-01-01", "2024-12-31"),
       grid={"window": [60, 120, 252]},
       spec_factory="make_mining_spec",
       gid_factory="mining_gid",
       objective="sharpe",
       engine="vector",
       workers=1,
       rebuild_cache=False,
   )
   df = run_parameter_sweep(config)

``run_parameter_sweep`` 输出结果表，默认写 ``sweep_results.csv``。单参数粗筛建议先用 ``engine="vector"``，候选参数再用 ``engine="exact"`` 复核。

滚动窗口挖掘
------------

.. code-block:: python

   from betalens.factor.mining import RollingMiningConfig, run_walk_forward

   config = RollingMiningConfig(
       factor_module="factor_DISP",
       output_dir="outputs/mining/walkforward",
       grid={"window": [60, 120, 252]},
       train=("2018-01-01", "2021-12-31"),
       test=("2022-01-01", "2023-12-31"),
       valid=("2024-01-01", "2024-12-31"),
       train_schemes=[(252, 63), (504, 126)],
       test_schemes=[(252, 63)],
       objective="sharpe",
       engine="exact",
   )
   payload = run_walk_forward(config)

滚动流程：

* TRAIN：全网格滚动评估并统计冠军。
* TEST：只复测 TRAIN 入围候选。
* VALID：对 TEST 前几名做整段验证，可调用 ``valid_report_hook``。

``RollingMiningConfig.rolling_mode`` 默认为 ``split``，保持上述兼容口径。
设置为 ``paired`` 时，使用 ``paired_schemes`` 的
``(train_length, test_length, step)`` 组合：每个 TRAIN 窗口后紧接一个不重叠
TEST 窗口，两个窗口在 ``rolling_span`` 内按 ``step`` 个交易日同步前移，
``valid`` 继续作为独立留出区间。旧 ``train`` / ``test`` 字段只在
``rolling_mode: split`` 时使用。

Optuna 因果 Walk-Forward
------------------------

设置 ``RollingMiningConfig.sampler="grid"`` 或 ``"motpe"`` 会启用 Optuna
多目标协调器。该模式要求 ``rolling_mode="paired"``：每个 TRAIN 窗拥有独立
study，只为紧随其后的 TEST 锁定参数；TEST 结果不写回 sampler。最终 VALID
候选取自结束日期最近的完整 TRAIN 窗，锁定后仅用 ``exact`` 引擎报告。

``grid`` 使用 ``GridSampler`` 穷举 YAML 明列的值。组合数超过
``max_grid_candidates`` 时在运行任务前失败，不截断。``motpe`` 使用多目标
``TPESampler``，预算由 ``n_trials`` 控制，``paper_params`` 会作为首个 trial
入队。两种 sampler 都由主进程串行调用 ``study.ask()`` / ``study.tell()``；
worker 只收到普通字典，不连接 Optuna storage。并行只由一个
``ProcessPoolExecutor`` 管理。

两个最大化目标为稳健 Rank IC 和 TRAIN Sharpe。Rank IC 使用前一交易日信号，
对应本次至下次调仓成交价收益；先计算月度 IC 均值，再计算
``median - 0.25 * 1.4826 * MAD``。默认可行约束为 IC 覆盖率不低于 80%、
最大回撤不高于 35%、平均单边调仓换手不高于 100%。锁参时先取可行 Pareto
前沿中稳健 IC 前 20%，再按 Sharpe、较小回撤、较低换手、较早 trial 排序。
无可行解时先最小化标准化约束违约，再执行同一排序。

SQLite 与恢复
-------------

默认每个因子的活动目录位于本地
``%LOCALAPPDATA%/Betalens/mining/<ALPHA>/<search_hash>``，其中创建
``study.sqlite3``、协调锁和 worker-slot 锁。只有协调主进程写库；本地 SQLite
使用 ``WAL``、``synchronous=NORMAL``、``busy_timeout=30s``、
``pool_size=1``、``max_overflow=0``。数据库锁按 1、2、4 秒退避重试。
周期 SQLite online backup 在单独线程运行并合并重复请求；OneDrive 输出目录
每 60 秒原子镜像审计文件和一致快照，不进入计算热路径。

``search_hash`` 只包含公式、搜索空间、窗口、目标、约束、数据口径和数据库
``dataset_coverage``/交易日历签名；worker
数量、运行目录和审计频率进入 ``runtime_hash``，调整性能参数不会使历史 trial
失效。重启时遗留 RUNNING trial 标记为 ``FAIL/stale_after_restart``，并按稳定
``candidate_id`` 重新入队。发现旧 OneDrive ``.study.lock`` 的 PID 仍活动时，
新协调器拒绝启动；旧任务退出后才通过 SQLite backup 认领已有 study。

``storage_url`` 可显式改为本地 scratch SQLite URL 或 PostgreSQL URL；框架
不会因 OneDrive 锁错误自动改变 storage。``Ctrl+C`` 会停止提交、取消未开始
任务、刷新审计和快照、释放协调锁，并以退出码 130 结束。

审计输出
--------

Optuna 模式同时写终端和 UTF-8 ``run.log``。``audit/`` 包含：

* ``events.jsonl``、``errors.jsonl``：逐事件结构化日志和异常路径。
* ``candidate_manifest.csv``：Grid 完整候选，或 MOTPE 已 ask 候选。
* ``trials.partial.csv``、``pareto_front.partial.csv``：每批原子刷新。
* ``oos_parameter_path.partial.csv``：各 TEST 锁参、样本外指标和参数变化。
* ``status.json``：PID、配置哈希、当前窗口、进度、快照和退出原因。
* ``task_logs/<trial>/``：worker 独立 stdout、stderr 和 traceback。

正常完成后输出 ``trials.csv``、``pareto_front.csv``、sklearn 风格
``cv_results.csv``、``oos_parameter_path.csv`` 和 ``final_candidates.yaml``；
partial 文件保留用于中断对账。
不创建 ``GridSearchCV`` estimator 适配层；``cv_results.csv`` 仅采用 sklearn
常用列名，避免把 PIT 缓存和回测对象伪装成 estimator。

缓存
----

挖掘缓存使用 immutable memmap-v3。每个数值宽表分别保存为 ``float64 .npy``，
PIT 股票池保存为布尔矩阵，交易状态保存为 ``int8``，行业标签保存为 ``int32``
编码和字典。worker 以 ``mmap_mode='r'`` 打开共享文件，只把当前窗口和预热期
切片转换为 DataFrame，不再为每个进程反序列化整份 pickle。

缓存构建使用独立 build lock，先写 staging，最后原子发布 ``READY.json``；
发布前校验 shape 与 SHA-256，日期和证券轴按内容寻址并跨字段复用；发布后读取
不加锁。v3 不迁移旧 ``mining_cache.pkl`` / ``pit_universe.pkl``，
配置不匹配时直接构建新 generation。

当 ``FactorSpec.industry_inputs`` 非空，cache 同时保存行业标签宽表；当
``mask_inputs_by_pit: true`` 时，行情和行业输入会在公式计算前按逐日 PIT
成分股 mask。``use_industry`` 的后处理中性化面板也会按行业体系缓存，
避免每个候选任务重复查询行业表。缓存元数据包含 schema 版本、输入映射、
日期范围、指数和行业体系指纹，配置不匹配时自动重建。

推荐忽略：

.. code-block:: text

   **/_sweep/_cache/
   **/_walkforward/_cache/

已有 `.gitignore` 包含这些目录。若历史缓存已被跟踪，需单独从索引移除。

Exact 预加载
------------

``BacktestBase`` 在参数末尾提供三个仅限关键字的可选输入：
``preloaded_cost_price``、``preloaded_close_price`` 和
``preloaded_trade_status``。Mining exact worker 从 memmap 分别截取成交价、收盘估值价
和交易状态后传入；
三项齐全时不创建 Datafeed 连接。未传入时仍走原数据库查询。

预加载数据与数据库结果进入相同的日期/证券/价格校验、停牌处理、整数手成交、
现金、持仓和净值计算流程。格式错误会抛出 ``BacktestDataError``，不会静默
回退数据库。``engine.data_sources`` 记录每条数据链路实际使用
``preloaded`` 还是 ``database``，供 worker 日志审计。

多窗口调度
----------

Optuna paired 模式使用单一全局事件循环和单一 ``ProcessPoolExecutor``。调度器
按窗口轮询，优先给不同 TRAIN 窗口各提交一个 trial；MOTPE 每 study 最多一个
在途 trial。Grid 仅在剩余窗口不足 worker 数时才使用同一 study 的备用容量。
TEST 只有在对应 TRAIN 锁参写入 SQLite 后才能提交，且不调用 ``tell()``。

``min_workers`` 是硬下限。资源不足时按 ``resource_check_seconds`` 等待，超过
``resource_wait_minutes`` 后失败，不静默降到下限以下；运行中低于内存低水位
会暂停补任务，达到恢复水位后继续。BLAS/NumExpr 线程固定为 1，避免进程内
嵌套并行。启动时会在短生命周期 worker 中映射真实 cache，读取 private bytes，
再预留 25% 放大、至少 8 GB 物理内存和 15% commit 余量。

引擎选择
--------

* ``vector``：用价格宽表快速近似回测，适合粗筛。
* ``exact``：调用 ``BacktestBase``，整数手、交易状态和资金递推口径更准，速度更慢。

内存保护
--------

多进程挖掘默认 ``max_memory_ratio=0.5``，会按缓存体量估算 worker 数。若单 worker 超出预算会快速失败。参数含义：

* ``workers``：目标并行数。
* ``max_memory_ratio``：可用内存占比上限。
* ``max_memory_bytes``：显式内存上限，优先级高于比例。
* ``cache_memory_multiplier``：缓存数据在 worker 内的放大系数。
* ``worker_memory_overhead_bytes``：单 worker 固定额外开销。

结果指标
--------

``metrics_from_nav`` 默认输出 ``ann_ret``、``sharpe``、``mdd`` 等指标。
worker 还输出 ``robust_rank_ic``、``mean_rank_ic``、``ic_coverage`` 和
``turnover``。旧扫描路径由 ``objective`` 指定排序字段；Optuna 路径固定使用
稳健 Rank IC 与 Sharpe 两个目标。

实践建议
--------

* 小样本先跑 smoke 网格，确认 hook、数据和输出目录。
* 滚动窗口方案用 ``window_lengths × steps`` 生成，再过滤 ``step <= window_length``。
* 先用 ``vector`` 找候选，再用 ``exact`` 对前几名复核。
* 运行日志保留配置、参数组合、任务 START 行和进度条，便于恢复和审计。
