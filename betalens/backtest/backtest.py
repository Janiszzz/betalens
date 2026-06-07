import numpy as np
import pandas as pd
import warnings
from pylab import mpl, plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'
from datafeed import Datafeed


class BacktestDataError(Exception):
    """回测数据异常基类"""
    pass


class DateMismatchError(BacktestDataError):
    """日期不匹配异常"""
    pass


class CodeMismatchError(BacktestDataError):
    """标的代码不匹配异常"""
    pass


# ==================== 数据验证函数 ====================

def format_data_sample(df, max_rows=3, max_cols=5):
    """格式化数据样本用于错误信息"""
    if df is None:
        return "None"
    if df.empty:
        return "空DataFrame"
    
    sample_info = []
    sample_info.append(f"形状: {df.shape}")
    sample_info.append(f"索引类型: {type(df.index).__name__}")
    
    if isinstance(df.index, pd.MultiIndex):
        sample_info.append(f"索引层级: {df.index.names}")
        sample_info.append(f"索引样本: {df.index[:max_rows].tolist()}")
    else:
        sample_info.append(f"索引范围: {df.index[0]} ~ {df.index[-1]}")
        sample_info.append(f"索引样本: {df.index[:max_rows].tolist()}")
    
    if isinstance(df, pd.Series):
        sample_info.append(f"类型: Series")
        if len(df) > 0:
            sample_info.append(f"数据样本:\n{df.iloc[:max_rows].to_string()}")
    else:
        sample_info.append(f"列名: {list(df.columns[:max_cols])}{'...' if len(df.columns) > max_cols else ''}")
        if len(df) > 0:
            sample_info.append(f"数据样本:\n{df.iloc[:max_rows, :max_cols].to_string()}")
    
    return "\n  ".join(sample_info)


def validate_weight_input(weight):
    """
    验证 weight 输入格式
    
    Args:
        weight: 待验证的权重数据
        
    Raises:
        BacktestDataError: 当格式不符合要求时
    """
    if not isinstance(weight, pd.DataFrame):
        raise BacktestDataError(
            f"weight 必须是 DataFrame，当前类型: {type(weight)}\n"
            f"修复建议: 使用 pd.DataFrame() 转换"
        )
    
    if weight.empty:
        raise BacktestDataError(
            f"weight 不能为空\n"
            f"修复建议: 检查输入数据源"
        )
    
    # 检查索引是否为时间类型
    if not isinstance(weight.index, (pd.DatetimeIndex, pd.Timestamp)):
        # 尝试转换
        try:
            pd.to_datetime(weight.index)
        except:
            raise BacktestDataError(
                f"weight.index 必须是时间索引，当前类型: {type(weight.index)}\n"
                f"索引样本: {weight.index[:3].tolist()}\n"
                f"修复建议: 使用 pd.to_datetime() 转换索引"
            )
    
    # 检查列名格式
    invalid_cols = [col for col in weight.columns if not isinstance(col, str)]
    if invalid_cols:
        raise BacktestDataError(
            f"weight 列名必须为字符串类型，发现非字符串列: {invalid_cols[:5]}\n"
            f"修复建议: 使用 weight.columns = weight.columns.astype(str)"
        )
    
    # 检查数据类型
    numeric_cols = weight.select_dtypes(include=[np.number]).columns
    non_numeric_cols = weight.columns.difference(numeric_cols)
    if len(non_numeric_cols) > 0 and 'cash' not in non_numeric_cols:
        # 允许 cash 列，但其他列应该是数值型
        other_non_numeric = non_numeric_cols.difference(['cash'])
        if len(other_non_numeric) > 0:
            raise BacktestDataError(
                f"weight 值必须为数值类型，发现非数值列: {list(other_non_numeric)[:5]}\n"
                f"修复建议: 使用 pd.to_numeric() 转换或删除非数值列"
            )
    
    # 检查是否包含 cash 列
    if 'cash' not in weight.columns:
        warnings.warn(
            f"weight 缺少 'cash' 列，将在后续处理中添加（值为1）\n"
            f"修复建议: 添加 cash 列: weight['cash'] = 0.0",
            UserWarning
        )
    
    # 检查是否有 NaN/Inf
    numeric_data = weight.select_dtypes(include=[np.number])
    if numeric_data.isnull().any().any():
        nan_cols = numeric_data.columns[numeric_data.isnull().any()].tolist()
        warnings.warn(
            f"weight 包含 NaN 值，列: {nan_cols[:5]}\n"
            f"修复建议: 使用 weight.fillna(0) 填充",
            UserWarning
        )
    
    if np.isinf(numeric_data.values).any():
        warnings.warn(
            f"weight 包含 Inf 值\n"
            f"修复建议: 使用 weight.replace([np.inf, -np.inf], np.nan).fillna(0)",
            UserWarning
        )


def validate_query_result(df, expected_columns, query_name="数据库查询"):
    """
    验证数据库查询结果
    
    Args:
        df: 查询返回的 DataFrame
        expected_columns: 期望的列名列表
        query_name: 查询名称（用于错误信息）
        
    Raises:
        BacktestDataError: 当格式不符合要求时
    """
    if df is None:
        raise BacktestDataError(
            f"{query_name} 返回 None\n"
            f"修复建议: 检查数据库连接和查询参数"
        )
    
    if df.empty:
        raise BacktestDataError(
            f"{query_name} 返回空结果\n"
            f"修复建议: 检查查询参数和数据库中的数据"
        )
    
    if not isinstance(df, pd.DataFrame):
        raise BacktestDataError(
            f"{query_name} 返回类型不是 DataFrame: {type(df)}\n"
            f"修复建议: 检查查询函数返回值"
        )
    
    missing_cols = set(expected_columns) - set(df.columns)
    if missing_cols:
        raise BacktestDataError(
            f"{query_name} 缺少必需的列: {missing_cols}\n"
            f"实际列名: {list(df.columns)}\n"
            f"期望列名: {expected_columns}\n"
            f"修复建议: 检查数据库表结构和查询SQL"
        )


def validate_pivot_result(df, expected_codes=None, index_levels=None):
    """
    验证 pivot_table 结果
    
    Args:
        df: pivot_table 后的 DataFrame
        expected_codes: 期望的标的代码列表
        index_levels: 期望的索引层级名称列表
        
    Raises:
        BacktestDataError: 当格式不符合要求时
    """
    if df.empty:
        raise BacktestDataError(
            f"pivot_table 结果为空\n"
            f"修复建议: 检查输入数据和 pivot_table 参数"
        )
    
    # 检查索引结构
    if index_levels is not None:
        if isinstance(df.index, pd.MultiIndex):
            actual_levels = df.index.names
            if set(actual_levels) != set(index_levels):
                warnings.warn(
                    f"pivot_table 索引层级不匹配\n"
                    f"实际: {actual_levels}\n"
                    f"期望: {index_levels}\n"
                    f"数据样本:\n{format_data_sample(df)}",
                    UserWarning
                )
        else:
            if len(index_levels) > 1:
                raise BacktestDataError(
                    f"pivot_table 结果应为 MultiIndex，但当前为单层索引\n"
                    f"期望层级: {index_levels}\n"
                    f"修复建议: 检查 pivot_table 的 index 参数"
                )
    
    # 检查是否有重复索引
    if df.index.duplicated().any():
        dup_indices = df.index[df.index.duplicated()].unique()
        warnings.warn(
            f"pivot_table 结果包含重复索引: {len(dup_indices)} 个\n"
            f"示例: {dup_indices[:3].tolist()}\n"
            f"修复建议: 检查输入数据是否有重复的 (input_ts, datetime, code) 组合",
            UserWarning
        )
    
    # 检查标的代码
    if expected_codes is not None:
        missing_codes = set(expected_codes) - set(df.columns)
        if missing_codes:
            warnings.warn(
                f"pivot_table 结果缺少部分标的: {len(missing_codes)}/{len(expected_codes)}\n"
                f"缺失示例: {list(missing_codes)[:5]}\n"
                f"修复建议: 检查数据库查询结果是否包含这些标的",
                UserWarning
            )


def validate_index_alignment(df1, df2, name1="DataFrame1", name2="DataFrame2"):
    """
    验证两个 DataFrame 的索引是否对齐
    
    Args:
        df1: 第一个 DataFrame
        df2: 第二个 DataFrame
        name1: 第一个 DataFrame 的名称
        name2: 第二个 DataFrame 的名称
        
    Returns:
        bool: 是否对齐
    """
    if df1.index.equals(df2.index):
        return True
    
    # 检查长度
    if len(df1) != len(df2):
        warnings.warn(
            f"{name1} 和 {name2} 长度不匹配: {len(df1)} vs {len(df2)}\n"
            f"{name1} 索引范围: {df1.index[0]} ~ {df1.index[-1]}\n"
            f"{name2} 索引范围: {df2.index[0]} ~ {df2.index[-1]}",
            UserWarning
        )
        return False
    
    # 检查内容
    if isinstance(df1.index, pd.MultiIndex) and isinstance(df2.index, pd.MultiIndex):
        if not df1.index.equals(df2.index):
            warnings.warn(
                f"{name1} 和 {name2} 索引内容不匹配\n"
                f"修复建议: 使用 reindex() 或 align() 对齐索引",
                UserWarning
            )
            return False
    
    return True


def validate_calculation_inputs(*args, **kwargs):
    """
    验证计算前的输入数据
    
    Args:
        *args: 要验证的 DataFrame/Series
        **kwargs: 命名参数，格式为 name=df
    """
    for i, df in enumerate(args):
        if df is None:
            raise BacktestDataError(f"计算输入参数 {i} 为 None")
        if hasattr(df, 'isnull'):
            if isinstance(df, pd.Series):
                has_nan = df.isnull().any()
            else:
                has_nan = df.isnull().any().any() if hasattr(df, 'any') else False
            if has_nan:
                warnings.warn(
                    f"计算输入参数 {i} 包含 NaN 值\n"
                    f"数据样本:\n{format_data_sample(df)}",
                    UserWarning
                )
        if hasattr(df, 'values'):
            try:
                values = df.values
                if isinstance(df, pd.Series):
                    if pd.api.types.is_numeric_dtype(df):
                        if np.isinf(values).any():
                            warnings.warn(
                                f"计算输入参数 {i} 包含 Inf 值\n"
                                f"修复建议: 使用 replace([np.inf, -np.inf], np.nan).fillna(0)",
                                UserWarning
                            )
                else:
                    numeric_data = df.select_dtypes(include=[np.number])
                    if not numeric_data.empty and np.isinf(numeric_data.values).any():
                        warnings.warn(
                            f"计算输入参数 {i} 包含 Inf 值\n"
                            f"修复建议: 使用 replace([np.inf, -np.inf], np.nan).fillna(0)",
                            UserWarning
                        )
            except (TypeError, ValueError):
                pass
    
    for name, df in kwargs.items():
        if df is None:
            raise BacktestDataError(f"计算输入参数 '{name}' 为 None")
        if hasattr(df, 'isnull'):
            if isinstance(df, pd.Series):
                has_nan = df.isnull().any()
            else:
                has_nan = df.isnull().any().any() if hasattr(df, 'any') else False
            if has_nan:
                warnings.warn(
                    f"计算输入参数 '{name}' 包含 NaN 值\n"
                    f"数据样本:\n{format_data_sample(df)}",
                    UserWarning
                )
        if hasattr(df, 'values'):
            try:
                values = df.values
                if isinstance(df, pd.Series):
                    if pd.api.types.is_numeric_dtype(df):
                        if np.isinf(values).any():
                            warnings.warn(
                                f"计算输入参数 '{name}' 包含 Inf 值\n"
                                f"修复建议: 使用 replace([np.inf, -np.inf], np.nan).fillna(0)",
                                UserWarning
                            )
                else:
                    numeric_data = df.select_dtypes(include=[np.number])
                    if not numeric_data.empty and np.isinf(numeric_data.values).any():
                        warnings.warn(
                            f"计算输入参数 '{name}' 包含 Inf 值\n"
                            f"修复建议: 使用 replace([np.inf, -np.inf], np.nan).fillna(0)",
                            UserWarning
                        )
            except (TypeError, ValueError):
                pass


class BacktestBase(object):
    def __init__(self, weight, symbol, amount,
                 ftc=0.0, ptc=0.0, verbose=True,
                 metric="收盘价(元)", time_tolerance=24,
                 table_name="daily_market",
                 check_trade_status=True,
                 trade_status_mode='to_cash',
                 trade_status_table='trade_status',
                 lot_size=100):
        # === 输入验证：weight ===
        try:
            validate_weight_input(weight)
        except BacktestDataError as e:
            error_msg = f"weight 输入验证失败:\n  {str(e)}\n  数据样本:\n  {format_data_sample(weight)}"
            raise BacktestDataError(error_msg) from e

        self.cost_ret = None
        self.weight = weight.copy()  # 使用副本避免修改原始数据
        self.symbol = symbol
        
        # 确保索引为时间类型
        if not isinstance(self.weight.index, pd.DatetimeIndex):
            try:
                self.weight.index = pd.to_datetime(self.weight.index)
            except Exception as e:
                raise BacktestDataError(
                    f"无法将 weight.index 转换为时间索引: {e}\n"
                    f"索引样本: {self.weight.index[:3].tolist()}"
                ) from e
        
        self.start = self.weight.index[0]
        self.end = self.weight.index[-1]
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        self.metric = metric
        self.time_tolerance = time_tolerance
        self.table_name = table_name
        self.check_trade_status = check_trade_status
        self.trade_status_mode = trade_status_mode
        self.trade_status_table = trade_status_table
        self.lot_size = int(lot_size)
        if self.lot_size < 1:
            raise BacktestDataError(
                f"lot_size 必须为正整数，当前: {lot_size}")

        # 提前校验模式，避免查库后才报错
        _valid_modes = ('to_cash', 'hold', 'redistribute', 'as_normal', 'report_only')
        if self.trade_status_mode not in _valid_modes:
            raise BacktestDataError(
                f"无效的 trade_status_mode: {self.trade_status_mode}，"
                f"应为 {_valid_modes} 之一"
            )

        # 提取交易状态（一等流程）：建立 self.trade_status / self.trade_status_matrix，
        # 并按 trade_status_mode 调整 self.weight，须在 get_rebalance_data 之前完成
        self.get_trade_status()
        self.get_rebalance_data()
        self.get_position_data()
        self.get_daily_position_data()

    def melt_weights(self):
        try:
            if("code" in self.weight.columns):
                self.weight = pd.pivot_table(self.weight, values='weight', index=['input_ts'], columns=['code'], )
            return 0
        except:
            return 1

    def _pivot_nearest_prices(self, raw, metric, weight_codes):
        """
        把 query_nearest_{after,before} 返回的长表转为宽表。
        每个 input_ts 一行，避免不同 code 的真实 datetime 不同导致行爆炸。

        Returns:
            prices: DataFrame, index=input_ts(DatetimeIndex), columns=code, 值=metric
            actual_dt: DataFrame, 同形状，值=每格真实成交 datetime（审计用）
        """
        required = {'code', 'input_ts', 'datetime', metric}
        missing = required - set(raw.columns)
        if missing:
            raise BacktestDataError(
                f"查询结果缺少列 {missing}\n实际列: {list(raw.columns)}"
            )

        # SQL 层已保证 (code, input_ts) 唯一；此处再 defensive drop
        raw = raw.drop_duplicates(subset=['code', 'input_ts'], keep='first')

        prices = raw.pivot(index='input_ts', columns='code', values=metric)
        actual_dt = raw.pivot(index='input_ts', columns='code', values='datetime')
        prices.columns.name = ""
        actual_dt.columns.name = ""

        # 仅保留 weight 里请求过的 code 列顺序（missing 的由上游 on_missing_code 流程处理）
        present = [c for c in weight_codes if c in prices.columns]
        prices = prices[present]
        actual_dt = actual_dt[present]

        # 共享成交日诊断：某 code 下两个不同调仓日被同一成交日兜底，提示
        # time_tolerance 过大或数据稀疏
        for code in actual_dt.columns:
            col = actual_dt[code].dropna()
            dup_mask = col.duplicated(keep=False)
            if dup_mask.any():
                sample = col[dup_mask].unique()[:3]
                warnings.warn(
                    f"[time_tolerance 过大/数据稀疏] code={code} 下有 "
                    f"{int(dup_mask.sum())} 条调仓日被同一成交日兜底，"
                    f"示例 datetime={list(sample)}，"
                    f"建议收紧 time_tolerance 或检查数据完整性",
                    UserWarning,
                )
        return prices, actual_dt

    def get_trade_status(self):
        """
        从数据库提取调仓日的个券交易状态（一等流程，与 get_rebalance_data 并列）。

        建立两份实例数据供审计与后续处理：
            self.trade_status: 长表 DataFrame，列 code/datetime/value/status_text/name
                value: 1=正常交易, 0=停牌等异常, -1=未上市/无法交易
            self.trade_status_matrix: 宽表矩阵，index=调仓日(与 weight 对齐), columns=code,
                值为 value（-1/0/1）；查询失败或关闭检查时为 None

        提取完成后，若 check_trade_status 为真，调用 _apply_trade_status 按
        trade_status_mode 调整 self.weight。本方法须在 get_rebalance_data 之前执行。
        """
        self.trade_status = None
        self.trade_status_matrix = None

        if not self.check_trade_status:
            return

        weight_codes = [c for c in self.weight.columns if c != 'cash']
        if not weight_codes:
            return
        dates = [pd.Timestamp(d).strftime('%Y-%m-%d') for d in self.weight.index]

        # === 数据库查询 ===
        try:
            ts_db = Datafeed(self.trade_status_table)
            try:
                status = ts_db.query_trade_status(
                    {'codes': weight_codes, 'dates': dates})
            finally:
                ts_db.close()
        except Exception as e:
            if self.verbose:
                warnings.warn(
                    f"交易状态提取失败（已跳过，按正常交易处理）: {e}", UserWarning)
            return

        if status is None or status.empty:
            if self.verbose:
                warnings.warn(
                    f"交易状态表 {self.trade_status_table} 无返回数据，"
                    f"按正常交易处理", UserWarning)
            return

        self.trade_status = status

        # 构造宽表矩阵（调仓日 × code），index 对齐 weight.index
        try:
            mat = status.copy()
            mat['date'] = pd.to_datetime(mat['datetime']).dt.strftime('%Y-%m-%d')
            wide = mat.pivot_table(
                index='date', columns='code', values='value', aggfunc='first')
            # 把日期索引映射回 weight 的实际时间戳索引
            date_to_ts = {}
            for ts in self.weight.index:
                date_to_ts.setdefault(pd.Timestamp(ts).strftime('%Y-%m-%d'), ts)
            wide.index = [date_to_ts.get(d, pd.Timestamp(d)) for d in wide.index]
            wide = wide.reindex(index=self.weight.index)
            wide.columns.name = ""
            self.trade_status_matrix = wide
        except Exception as e:
            if self.verbose:
                warnings.warn(f"构造交易状态矩阵失败（不影响处理）: {e}", UserWarning)

        # === 按模式应用到权重 ===
        self._apply_trade_status()

    def _apply_trade_status(self):
        """
        按 trade_status_mode 处理 self.weight 中停牌（value==0）的持仓。

        仅“停牌”(value==0) 视为异常需要处理；未上市(-1) 交由 get_rebalance_data
        的 missing_codes 逻辑处理。

        模式（trade_status_mode）：
            'to_cash'     : 默认。停牌股当期权重置0，资金留现金（假设买卖失败）
            'hold'        : 停牌无法调仓，沿用上一调仓日权重（持仓被动冻结）
            'redistribute': 停牌股权重按比例分给当期可交易持仓，整行重新归一
            'as_normal'   : 忽略停牌，假设仍能正常买卖，仅统计提示
            'report_only' : 仅统计提示，不改动权重
        """
        if self.trade_status is None or self.trade_status.empty:
            return

        mode = self.trade_status_mode

        # 停牌记录（value==0），定位到 weight 的实际时间戳
        suspended = self.trade_status[self.trade_status['value'] == 0].copy()
        if suspended.empty:
            return
        suspended['date'] = pd.to_datetime(
            suspended['datetime']).dt.strftime('%Y-%m-%d')

        date_to_ts = {}
        for ts in self.weight.index:
            date_to_ts.setdefault(pd.Timestamp(ts).strftime('%Y-%m-%d'), ts)

        index_list = list(self.weight.index)
        total_suspended = 0
        affected_dates = []

        for date_str, grp in suspended.groupby('date'):
            ts = date_to_ts.get(date_str)
            if ts is None:
                continue
            # 当期实际有持仓（权重非0）的停牌股票
            held = [c for c in grp['code'] if c in self.weight.columns
                    and self.weight.at[ts, c] != 0]
            if not held:
                continue
            total_suspended += len(held)
            affected_dates.append(date_str)

            if self.verbose:
                detail = grp[grp['code'].isin(held)][['code', 'status_text']]
                print(f"  [交易状态] {date_str} 停牌持仓 {len(held)} 只: "
                      f"{detail.to_dict('records')}")

            if mode in ('as_normal', 'report_only'):
                continue

            if mode == 'to_cash':
                # 停牌股权重置0，资金自然留现金（cash 列由后续流程补足）
                for c in held:
                    self.weight.at[ts, c] = 0.0

            elif mode == 'hold':
                # 沿用上一调仓日的整行权重（无法调仓→持仓冻结）
                pos = index_list.index(ts)
                if pos == 0:
                    # 首期无上期可沿用，退化为 to_cash
                    for c in held:
                        self.weight.at[ts, c] = 0.0
                    if self.verbose:
                        print(f"  [交易状态] {date_str} 为首期，hold 退化为 to_cash")
                else:
                    prev_ts = index_list[pos - 1]
                    self.weight.loc[ts] = self.weight.loc[prev_ts].values

            elif mode == 'redistribute':
                # 停牌股权重清零后，将整行剩余权重重新归一到可交易持仓
                for c in held:
                    self.weight.at[ts, c] = 0.0
                row = self.weight.loc[ts]
                row_sum = row.sum()
                if row_sum > 0:
                    self.weight.loc[ts] = (row / row_sum).values

        if total_suspended and self.verbose:
            note = {
                'to_cash': '，已置零转现金',
                'hold': '，已沿用上期权重',
                'redistribute': '，已重新归一到可交易持仓',
                'as_normal': '，按正常交易处理',
                'report_only': '，仅统计未改动',
            }.get(mode, '')
            print(f"  [交易状态] 模式={mode}，共 {len(affected_dates)} 个调仓日、"
                  f"{total_suspended} 个停牌持仓{note}")

    def get_rebalance_data(self):
        """
        获取调仓日数据，包含日期和标的匹配验证

        Raises:
            DateMismatchError: 当权重日期在数据库中无对应数据时
            CodeMismatchError: 当权重标的在数据库中无数据时（严格模式）
        """
        # 换仓前先检查交易状态：统计异常持仓并按 mode 处理权重
        if self.check_trade_status:
            try:
                self._apply_trade_status()
            except BacktestDataError:
                raise
            except Exception as e:
                if self.verbose:
                    warnings.warn(
                        f"交易状态检查失败（已跳过，不影响回测）: {e}", UserWarning)

        db = Datafeed(self.table_name)

        # 获取权重中的标的列表（排除cash）
        if 'cash' in self.weight.columns:
            weight_codes = list(self.weight.columns.drop('cash'))
        else:
            weight_codes = list(self.weight.columns)
        
        # 确保日期格式正确
        weight_ts = [pd.Timestamp(dt) for dt in self.weight.index]
        weight_dates = [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in weight_ts]

        # 构造区间：当前调仓日 ~ 下一调仓日；末点 = 最后调仓日 + time_tolerance 小时
        ranges = []
        for i in range(len(weight_ts) - 1):
            ranges.append((weight_dates[i], weight_dates[i + 1]))
        last_end = weight_ts[-1] + pd.Timedelta(hours=self.time_tolerance)
        ranges.append((weight_dates[-1], last_end.strftime('%Y-%m-%d %H:%M:%S')))

        params = {
            'codes': weight_codes,
            'ranges': ranges,
            'metric': self.metric,
            'time_tolerance': self.time_tolerance,
        }

        # === 数据库查询 ===
        try:
            self.cost_price = db.query_nearest_in_range_after(params)
        except Exception as e:
            raise BacktestDataError(
                f"数据库查询失败: {e}\n"
                f"查询参数: codes={len(weight_codes)}个, ranges={len(ranges)}个\n"
                f"修复建议: 检查数据库连接和查询参数"
            ) from e

        # === 验证查询结果 ===
        expected_columns = ['code', 'input_ts', 'datetime', params['metric']]
        try:
            validate_query_result(self.cost_price, expected_columns, "query_nearest_in_range_after")
        except BacktestDataError as e:
            error_msg = (
                f"数据库查询结果验证失败:\n  {str(e)}\n"
                f"查询参数: {params}\n"
                f"返回数据样本:\n  {format_data_sample(self.cost_price)}"
            )
            raise BacktestDataError(error_msg) from e
        
        # === pivot_table 转换 ===
        if params['metric'] not in self.cost_price.columns:
            raise BacktestDataError(
                f"pivot_table 找不到列 '{params['metric']}'\n"
                f"实际列名: {list(self.cost_price.columns)}\n"
                f"修复建议: 检查数据库返回的列名"
            )
        
        try:
            self.cost_price, self.actual_datetime = self._pivot_nearest_prices(
                self.cost_price, params['metric'], weight_codes,
            )
        except BacktestDataError:
            raise
        except Exception as e:
            raise BacktestDataError(
                f"_pivot_nearest_prices 失败: {e}\n"
                f"实际列名: {list(self.cost_price.columns) if isinstance(self.cost_price, pd.DataFrame) else 'N/A'}\n"
                f"修复建议: 检查数据库查询返回的列结构"
            ) from e

        # DB 返回可能为 decimal.Decimal，统一转 float 避免与 float 权重相乘报错
        self.cost_price = self.cost_price.apply(pd.to_numeric, errors='coerce')
        self.cost_price['cash'] = 1.0
        # actual_datetime 不添加 cash 列（cash 无真实成交数据）

        # === 验证 pivot 结果 ===
        try:
            validate_pivot_result(
                self.cost_price,
                expected_codes=weight_codes,
                index_levels=['input_ts'],
            )
        except BacktestDataError as e:
            error_msg = (
                f"pivot_table 结果验证失败:\n  {str(e)}\n"
                f"数据样本:\n  {format_data_sample(self.cost_price)}"
            )
            raise BacktestDataError(error_msg) from e
        
        # === 日期匹配检查 ===
        try:
            db_input_ts_raw = self.cost_price.index
            db_input_ts = set([
                pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(ts, (pd.Timestamp, pd.DatetimeIndex))
                else str(ts)
                for ts in db_input_ts_raw
            ])
        except (KeyError, AttributeError) as e:
            raise BacktestDataError(
                f"无法解析 cost_price.index: {e}\n"
                f"索引类型: {type(self.cost_price.index)}\n"
                f"修复建议: 检查 _pivot_nearest_prices 返回值"
            ) from e
        
        weight_input_ts = set(weight_dates)
        missing_dates = weight_input_ts - db_input_ts
        
        if missing_dates:
            missing_dates_sorted = sorted(missing_dates)
            raise DateMismatchError(
                f"权重中存在 {len(missing_dates)} 个日期在数据库中无数据:\n"
                f"  示例: {missing_dates_sorted[:5]}{'...' if len(missing_dates) > 5 else ''}\n"
                f"  权重日期范围: {weight_dates[0]} ~ {weight_dates[-1]}\n"
                f"  数据库日期范围: {sorted(db_input_ts)[0] if db_input_ts else 'N/A'} ~ {sorted(db_input_ts)[-1] if db_input_ts else 'N/A'}\n"
                f"  修复建议: 检查交易日历或扩大 time_tolerance 参数"
            )
        
        # === 标的匹配检查 ===
        db_codes = set(self.cost_price.columns) - {'cash'}
        weight_codes_set = set(weight_codes)
        missing_codes = weight_codes_set - db_codes
        
        if missing_codes:
            missing_codes_list = sorted(missing_codes)
            warning_msg = (
                f"部分标的在数据库中无数据 ({len(missing_codes)}/{len(weight_codes)}):\n"
                f"  缺失: {missing_codes_list[:10]}{'...' if len(missing_codes) > 10 else ''}\n"
                f"  数据样本:\n  {format_data_sample(self.cost_price)}"
            )
            
            if self.verbose:
                warnings.warn(warning_msg, UserWarning)
            
            # 将缺失标的的权重设为0（避免后续计算错误）
            for code in missing_codes:
                if code in self.weight.columns:
                    self.weight[code] = 0.0
                    if self.verbose:
                        print(f"  [警告] 标的 {code} 权重已置零")
        
        # === 长度一致性检查 ===
        if len(self.cost_price) != len(self.weight):
            raise DateMismatchError(
                f"数据长度不匹配: 权重 {len(self.weight)} 行, 价格 {len(self.cost_price)} 行\n"
                f"  权重索引范围: {self.weight.index[0]} ~ {self.weight.index[-1]}\n"
                f"  价格索引范围: {self.cost_price.index[0]} ~ {self.cost_price.index[-1]}\n"
                f"  可能原因: 部分日期查询失败或存在重复日期\n"
                f"  修复建议: 检查数据库查询结果和日期匹配逻辑"
            )
        
        # === 计算收益率 ===
        try:
            self.cost_ret = self.cost_price.pct_change().fillna(0).infer_objects(copy=False)
        except Exception as e:
            raise BacktestDataError(
                f"计算 cost_ret 失败: {e}\n"
                f"cost_price 数据样本:\n  {format_data_sample(self.cost_price)}"
            ) from e
        
        # === 更新 start/end ===
        try:
            self.start = self.cost_price.index[0]
            self.end = self.cost_price.index[-1]
        except (AttributeError, IndexError) as e:
            raise BacktestDataError(
                f"无法从 cost_price.index 取首尾: {e}\n"
                f"索引类型: {type(self.cost_price.index)}\n"
                f"修复建议: 确认 _pivot_nearest_prices 返回非空 DataFrame"
            ) from e
        
        # === 更新权重索引以匹配价格数据 ===
        # 备份原始索引信息（如果需要）
        original_weight_index = self.weight.index.copy()
        try:
            self.weight.index = self.cost_price.index
        except Exception as e:
            raise BacktestDataError(
                f"无法更新 weight.index: {e}\n"
                f"weight.index 类型: {type(original_weight_index)}\n"
                f"cost_price.index 类型: {type(self.cost_price.index)}\n"
                f"修复建议: 检查索引格式是否兼容"
            ) from e
        
        return self.cost_price

    def get_position_data(self):
        # === 计算前验证 ===
        try:
            validate_calculation_inputs(
                cost_ret=self.cost_ret,
                weight=self.weight,
                initial_amount=self.initial_amount
            )
        except BacktestDataError as e:
            error_msg = (
                f"计算前验证失败:\n  {str(e)}\n"
                f"cost_ret 样本:\n  {format_data_sample(self.cost_ret)}\n"
                f"weight 样本:\n  {format_data_sample(self.weight)}"
            )
            raise BacktestDataError(error_msg) from e
        
        # === 索引对齐检查 ===
        if not validate_index_alignment(self.cost_ret, self.weight, "cost_ret", "weight"):
            # 尝试对齐
            try:
                self.cost_ret, self.weight = self.cost_ret.align(self.weight, join='inner', axis=0)
            except Exception as e:
                raise BacktestDataError(
                    f"无法对齐 cost_ret 和 weight 的索引: {e}\n"
                    f"cost_ret 索引: {self.cost_ret.index[:3].tolist()}\n"
                    f"weight 索引: {self.weight.index[:3].tolist()}"
                ) from e
        
        # === 逐期迭代：整数手取整 → actual_weight + 现金，并递推总资产 amount ===
        # 最小买入单位 lot_size 股，非整数手无法成交，余款转入 cash。
        # 当期可买手数依赖当期总资产 A_t，而 A_t 又由上期实际持仓收益递推，
        # 存在逐期依赖，无法向量化（原 cumprod 方案），须按调仓日顺序迭代。
        try:
            stock_cols = [c for c in self.weight.columns if c != 'cash']
            all_cols = stock_cols + ['cash']
            index_list = list(self.weight.index)
            lot = self.lot_size

            A = float(self.initial_amount)
            amount_list = []
            actual_rows = []

            for pos, ts in enumerate(index_list):
                amount_list.append(A)  # 本调仓日（分配前）总资产

                target_w = self.weight.loc[ts]
                price = self.cost_price.loc[ts]
                aw = {c: 0.0 for c in all_cols}
                spent = 0.0

                if A != 0:
                    for code in stock_cols:
                        w = target_w.get(code, 0.0)
                        p = price.get(code, np.nan)
                        if w == 0 or pd.isna(w) or pd.isna(p) or p <= 0:
                            continue
                        # 目标股数 → 向零截断到整数手 → 实际成交股数/市值
                        shares = (w * A) / p
                        lots = np.trunc(shares / lot)
                        actual_value = lots * lot * p
                        aw[code] = actual_value / A
                        spent += actual_value
                    aw['cash'] = (A - spent) / A  # 余款（含未投资部分）转现金

                actual_rows.append(aw)

                # 递推到下一调仓日：实际权重 × 持有期收益（现金收益为0）
                if pos < len(index_list) - 1:
                    nxt = self.cost_ret.iloc[pos + 1]
                    period_ret = sum(
                        aw[c] * float(nxt.get(c, 0.0))
                        for c in stock_cols
                        if not pd.isna(nxt.get(c, 0.0))
                    )
                    A = A * (1.0 + period_ret)
            # __ITER_TAIL__
            # actual_weight：实际成交的整数手权重 + 余现金，供后续收益/持仓计算
            self.actual_weight = pd.DataFrame(
                actual_rows, index=self.weight.index, columns=all_cols
            ).fillna(0.0)
            # amount：各调仓日（分配前）总资产
            self.amount = pd.Series(
                amount_list, index=self.weight.index, name='amount'
            )

            if self.amount.isnull().any():
                raise BacktestDataError(
                    f"amount 计算后包含 NaN 值\n修复建议: 检查输入数据和计算过程"
                )
            if np.isinf(self.amount.values).any():
                warnings.warn(
                    f"amount 包含 Inf 值\n修复建议: 检查是否有异常大的收益率",
                    UserWarning
                )
        except BacktestDataError:
            raise
        except Exception as e:
            raise BacktestDataError(
                f"计算 amount 失败: {e}\n"
                f"cost_ret 样本:\n  {format_data_sample(self.cost_ret)}\n"
                f"weight 样本:\n  {format_data_sample(self.weight)}"
            ) from e

        return self.amount

    def get_daily_position_data(self):
        import datetime as dt
        db = Datafeed(self.table_name)
        
        # === 准备查询参数 ===
        if 'cash' in self.weight.columns:
            query_codes = list(self.weight.columns.drop('cash'))
        else:
            query_codes = list(self.weight.columns)
        
        # 确保 start/end 格式正确
        try:
            if isinstance(self.start, pd.Timestamp):
                start_date_str = self.start.strftime('%Y-%m-%d %H:%M:%S')
            else:
                start_date_str = str(self.start)
            
            if isinstance(self.end, pd.Timestamp):
                end_date_str = self.end.strftime('%Y-%m-%d %H:%M:%S')
            else:
                end_date_str = str(self.end)
        except Exception as e:
            raise BacktestDataError(
                f"无法格式化 start/end 日期: {e}\n"
                f"start 类型: {type(self.start)}, 值: {self.start}\n"
                f"end 类型: {type(self.end)}, 值: {self.end}"
            ) from e
        
        # === 数据库查询 ===
        try:
            close_price_ts = db.query_time_range(
                codes=query_codes,
                start_date=start_date_str,
                end_date=end_date_str,
                metric=self.metric,
            )
        except Exception as e:
            raise BacktestDataError(
                f"query_time_range 查询失败: {e}\n"
                f"查询参数: codes={len(query_codes)}个, start={start_date_str}, end={end_date_str}\n"
                f"修复建议: 检查数据库连接和查询参数"
            ) from e
        
        # === 验证查询结果 ===
        expected_columns = ['code', 'datetime', 'value']
        try:
            validate_query_result(close_price_ts, expected_columns, "query_time_range")
        except BacktestDataError as e:
            error_msg = (
                f"query_time_range 结果验证失败:\n  {str(e)}\n"
                f"返回数据样本:\n  {format_data_sample(close_price_ts)}"
            )
            raise BacktestDataError(error_msg) from e
        
        # === pivot_table 转换 ===
        if 'value' not in close_price_ts.columns:
            raise BacktestDataError(
                f"pivot_table 找不到列 'value'\n"
                f"实际列名: {list(close_price_ts.columns)}\n"
                f"修复建议: 检查数据库返回的列名"
            )
        
        try:
            close_price_ts = pd.pivot_table(
                close_price_ts, 
                values='value', 
                index=['datetime'], 
                columns=['code']
            )
        except KeyError as e:
            raise BacktestDataError(
                f"pivot_table 失败，缺少必需的列: {e}\n"
                f"实际列名: {list(close_price_ts.columns)}\n"
                f"修复建议: 检查数据库查询返回的列结构"
            ) from e
        
        close_price_ts.columns.name = ""
        close_price_ts['cash'] = 1
        
        # === 验证 pivot_table 结果 ===
        try:
            validate_pivot_result(
                close_price_ts,
                expected_codes=query_codes,
                index_levels=['datetime']
            )
        except BacktestDataError as e:
            error_msg = (
                f"close_price_ts pivot_table 结果验证失败:\n  {str(e)}\n"
                f"数据样本:\n  {format_data_sample(close_price_ts)}"
            )
            raise BacktestDataError(error_msg) from e
        
        # === 计算持仓 ===
        # 用 actual_weight（整数手成交权重 + 余现金）而非目标 weight，
        # 整数手取整与现金留存才能真正反映到持仓/每日净值中。
        try:
            # 验证计算输入
            validate_calculation_inputs(
                actual_weight=self.actual_weight,
                amount=self.amount,
                cost_price=self.cost_price
            )

            # 持仓金额 = 实际权重 × 当期总资产（含 cash 列）
            self.position = self.actual_weight.mul(self.amount, axis=0)
            # 检查索引对齐
            if not validate_index_alignment(self.position, self.cost_price, "position", "cost_price"):
                # 尝试对齐
                try:
                    self.position, self.cost_price = self.position.align(self.cost_price, join='inner', axis=0)
                except Exception as e:
                    raise BacktestDataError(
                        f"无法对齐 position 和 cost_price 的索引: {e}\n"
                        f"position 索引: {self.position.index[:3].tolist()}\n"
                        f"cost_price 索引: {self.cost_price.index[:3].tolist()}"
                    ) from e

            # 除以成本价得到持仓数量（cash 价=1 → 现金份额=现金额；股票为整数手）
            self.position = self.position.div(self.cost_price)
            
            # 检查除法结果
            if self.position.isnull().any().any():
                nan_count = self.position.isnull().sum().sum()
                warnings.warn(
                    f"position 计算后包含 {nan_count} 个 NaN 值\n"
                    f"修复建议: 检查 cost_price 是否包含零值或NaN",
                    UserWarning
                )
            
            try:
                numeric_data = self.position.select_dtypes(include=[np.number])
                if not numeric_data.empty and np.isinf(numeric_data.values).any():
                    warnings.warn(
                        f"position 包含 Inf 值\n"
                        f"修复建议: 检查 cost_price 是否包含零值",
                        UserWarning
                    )
            except (TypeError, ValueError):
                pass
            
        except Exception as e:
            raise BacktestDataError(
                f"计算 position 失败: {e}\n"
                f"weight 样本:\n  {format_data_sample(self.weight)}\n"
                f"amount 样本:\n  {format_data_sample(self.amount)}\n"
                f"cost_price 样本:\n  {format_data_sample(self.cost_price)}"
            ) from e
        
        # === 处理索引（pivot 后 cost_price/position 已是单层 DatetimeIndex）===
        try:
            # 历史兼容：若调用方自行构造了带 input_ts level 的 MultiIndex，降回单层
            if isinstance(self.position.index, pd.MultiIndex):
                if 'input_ts' in self.position.index.names:
                    self.position = self.position.droplevel('input_ts')
                else:
                    warnings.warn(
                        f"position.index 是 MultiIndex 但不包含 'input_ts' level\n"
                        f"索引层级: {self.position.index.names}",
                        UserWarning,
                    )

            # 索引去重（按索引值，而非按行值——避免误删持仓相同的两期）
            if self.position.index.duplicated().any():
                dup_count = int(self.position.index.duplicated().sum())
                warnings.warn(
                    f"position 包含 {dup_count} 个重复索引，保留最后一条",
                    UserWarning,
                )
                self.position = self.position[
                    ~self.position.index.duplicated(keep='last')
                ]
            
        except Exception as e:
            raise BacktestDataError(
                f"处理 position 索引失败: {e}\n"
                f"索引类型: {type(self.position.index)}\n"
                f"索引层级: {self.position.index.names if isinstance(self.position.index, pd.MultiIndex) else '单层索引'}\n"
                f"修复建议: 检查索引结构"
            ) from e
        
        # === 重新索引到日线数据 ===
        try:
            # 确保 close_price_ts 的索引是时间类型
            if not isinstance(close_price_ts.index, pd.DatetimeIndex):
                try:
                    close_price_ts.index = pd.to_datetime(close_price_ts.index)
                except Exception as e:
                    raise BacktestDataError(
                        f"无法将 close_price_ts.index 转换为时间索引: {e}\n"
                        f"索引样本: {close_price_ts.index[:3].tolist()}"
                    ) from e
            
            # 确保 self.position 的索引是时间类型
            if not isinstance(self.position.index, pd.DatetimeIndex):
                try:
                    self.position.index = pd.to_datetime(self.position.index)
                except Exception as e:
                    raise BacktestDataError(
                        f"无法将 position.index 转换为时间索引: {e}\n"
                        f"索引样本: {self.position.index[:3].tolist()}"
                    ) from e
            
            # 重新索引
            self.position = self.position.reindex(close_price_ts.index, method='ffill')
            
            # 检查重新索引后的结果
            if self.position.isnull().all().all():
                raise BacktestDataError(
                    f"重新索引后 position 全部为 NaN\n"
                    f"position 原始索引范围: {self.position.index[0]} ~ {self.position.index[-1]}\n"
                    f"close_price_ts 索引范围: {close_price_ts.index[0]} ~ {close_price_ts.index[-1]}\n"
                    f"修复建议: 检查日期范围是否重叠"
                )
            
        except Exception as e:
            raise BacktestDataError(
                f"重新索引 position 失败: {e}\n"
                f"position 索引: {self.position.index[:3].tolist()}\n"
                f"close_price_ts 索引: {close_price_ts.index[:3].tolist()}"
            ) from e
        
        # === 计算每日市值和净值 ===
        try:
            # 检查列对齐
            position_cols = set(self.position.columns) - {'cash'}
            price_cols = set(close_price_ts.columns) - {'cash'}
            missing_cols = position_cols - price_cols
            
            if missing_cols:
                warnings.warn(
                    f"position 和 close_price_ts 列不匹配，缺失 {len(missing_cols)} 个标的\n"
                    f"缺失示例: {list(missing_cols)[:5]}\n"
                    f"修复建议: 这些标的的持仓将被忽略",
                    UserWarning
                )
            
            # 对齐列
            common_cols = list(set(self.position.columns) & set(close_price_ts.columns))
            if 'cash' in common_cols:
                common_cols.remove('cash')
                common_cols.append('cash')  # cash 放在最后
            
            self.position = self.position[common_cols]
            close_price_ts = close_price_ts[common_cols]
            
            # 计算每日市值
            self.daily_amount = self.position.mul(close_price_ts, axis=0).sum(axis=1).astype(float)
            
            # 检查计算结果
            if self.daily_amount.isnull().any():
                nan_count = self.daily_amount.isnull().sum()
                warnings.warn(
                    f"daily_amount 包含 {nan_count} 个 NaN 值\n"
                    f"修复建议: 检查 position 和 close_price_ts 的数据完整性",
                    UserWarning
                )
            
            try:
                if pd.api.types.is_numeric_dtype(self.daily_amount):
                    if np.isinf(self.daily_amount).any():
                        warnings.warn(
                            f"daily_amount 包含 Inf 值\n"
                            f"修复建议: 检查数据是否有异常值",
                            UserWarning
                        )
            except (TypeError, ValueError):
                pass
            
            # 计算净值
            if self.initial_amount == 0:
                raise BacktestDataError(
                    f"initial_amount 不能为0\n"
                    f"修复建议: 设置正确的初始金额"
                )
            
            self.nav = (self.daily_amount / self.initial_amount)

            # 检查净值结果
            if self.nav.isnull().any():
                warnings.warn(
                    f"nav 包含 NaN 值\n"
                    f"修复建议: 检查 daily_amount 和 initial_amount",
                    UserWarning
                )

        except Exception as e:
            raise BacktestDataError(
                f"计算 daily_amount/nav 失败: {e}\n"
                f"position 样本:\n  {format_data_sample(self.position)}\n"
                f"close_price_ts 样本:\n  {format_data_sample(close_price_ts)}"
            ) from e

        return self.daily_amount

    def dump_to_excel(self, filepath: str) -> str:
        """
        将 bt 实例的所有数据导出到一个 Excel 文件备查（每个属性一个 sheet）

        - DataFrame / Series 直接写入对应 sheet
        - 标量参数汇总到 'meta' sheet

        Args:
            filepath: 输出 Excel 路径

        Returns:
            实际写入的文件路径
        """
        df_attrs = [
            'weight', 'actual_weight', 'cost_price', 'actual_datetime', 'cost_ret',
            'amount', 'position', 'daily_amount', 'nav',
        ]
        meta_keys = [
            'symbol', 'metric', 'table_name', 'time_tolerance',
            'initial_amount', 'ftc', 'ptc', 'start', 'end',
            'units', 'trades',
        ]

        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            meta_rows = []
            for k in meta_keys:
                if hasattr(self, k):
                    meta_rows.append({'key': k, 'value': getattr(self, k)})
            pd.DataFrame(meta_rows).to_excel(writer, sheet_name='meta', index=False)

            for name in df_attrs:
                obj = getattr(self, name, None)
                if obj is None:
                    continue
                if isinstance(obj, pd.Series):
                    obj = obj.to_frame(name=name)
                if isinstance(obj, pd.DataFrame):
                    sheet = name[:31]  # Excel sheet 名最长 31
                    obj.to_excel(writer, sheet_name=sheet)

        return filepath
