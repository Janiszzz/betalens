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
            f"修复建议: 添加 cash 列: weight['cash'] = 1.0",
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
                 ftc=0.0, ptc=0.0, verbose=True):
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

        self.get_rebalance_data()
        self.get_position_data()
        self.get_daily_position_data()
        ''' 
        self.melt_weights()
        self.get_rebalance_data()
        self.get_daily_position_data()'''

    def melt_weights(self):
        try:
            if("code" in self.weight.columns):
                self.weight = pd.pivot_table(self.weight, values='weight', index=['input_ts'], columns=['code'], )
            return 0
        except:
            return 1
    def get_rebalance_data(self):
        """
        获取调仓日数据，包含日期和标的匹配验证
        
        Raises:
            DateMismatchError: 当权重日期在数据库中无对应数据时
            CodeMismatchError: 当权重标的在数据库中无数据时（严格模式）
        """
        db = Datafeed("daily_market_data")
        
        # 获取权重中的标的列表（排除cash）
        if 'cash' in self.weight.columns:
            weight_codes = list(self.weight.columns.drop('cash'))
        else:
            weight_codes = list(self.weight.columns)
        
        # 确保日期格式正确
        weight_dates = [pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M:%S')
                       if isinstance(dt, (pd.Timestamp, pd.DatetimeIndex)) 
                       else str(dt) 
                       for dt in self.weight.index]
        
        params = {
            'codes': weight_codes,
            'datetimes': weight_dates,
            'metric': "收盘价(元)",
            'time_tolerance': 24
        }
        
        # === 数据库查询 ===
        try:
            self.cost_price = db.query_nearest_after(params)
        except Exception as e:
            raise BacktestDataError(
                f"数据库查询失败: {e}\n"
                f"查询参数: codes={len(weight_codes)}个, datetimes={len(weight_dates)}个\n"
                f"修复建议: 检查数据库连接和查询参数"
            ) from e
        
        # === 验证查询结果 ===
        expected_columns = ['code', 'input_ts', 'datetime', params['metric']]
        try:
            validate_query_result(self.cost_price, expected_columns, "query_nearest_after")
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
            self.cost_price = pd.pivot_table(
                self.cost_price, 
                values=params['metric'], 
                index=['input_ts', 'datetime'], 
                columns=['code']
            )
        except KeyError as e:
            raise BacktestDataError(
                f"pivot_table 失败，缺少必需的列: {e}\n"
                f"实际列名: {list(self.cost_price.columns)}\n"
                f"修复建议: 检查数据库查询返回的列结构"
            ) from e
        
        self.cost_price.columns.name = ""
        self.cost_price['cash'] = 1
        
        # === 验证 pivot_table 结果 ===
        try:
            validate_pivot_result(
                self.cost_price, 
                expected_codes=weight_codes,
                index_levels=['input_ts', 'datetime']
            )
        except BacktestDataError as e:
            error_msg = (
                f"pivot_table 结果验证失败:\n  {str(e)}\n"
                f"数据样本:\n  {format_data_sample(self.cost_price)}"
            )
            raise BacktestDataError(error_msg) from e
        
        # === 日期匹配检查 ===
        try:
            db_input_ts_raw = self.cost_price.index.get_level_values('input_ts')
            db_input_ts = set([
                pd.Timestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                if isinstance(ts, (pd.Timestamp, pd.DatetimeIndex))
                else str(ts)
                for ts in db_input_ts_raw
            ])
        except (KeyError, AttributeError) as e:
            raise BacktestDataError(
                f"无法获取 input_ts 层级: {e}\n"
                f"索引类型: {type(self.cost_price.index)}\n"
                f"索引层级: {self.cost_price.index.names if isinstance(self.cost_price.index, pd.MultiIndex) else '单层索引'}\n"
                f"修复建议: 检查 pivot_table 的 index 参数"
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
            datetime_level = self.cost_price.index.get_level_values('datetime')
            self.start = datetime_level[0]
            self.end = datetime_level[-1]
        except (KeyError, AttributeError, IndexError) as e:
            raise BacktestDataError(
                f"无法从 cost_price.index 提取 datetime: {e}\n"
                f"索引类型: {type(self.cost_price.index)}\n"
                f"索引层级: {self.cost_price.index.names if isinstance(self.cost_price.index, pd.MultiIndex) else '单层索引'}\n"
                f"修复建议: 检查 pivot_table 的 index 参数是否包含 'datetime'"
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
        
        # === 计算 amount ===
        try:
            # 检查 shift 操作
            weight_shifted = self.weight.shift(1)
            if weight_shifted.iloc[0].notna().any():
                warnings.warn(
                    f"weight.shift(1) 第一行包含非NaN值，这可能导致计算错误\n"
                    f"修复建议: 确保第一行应为NaN或零",
                    UserWarning
                )
            
            self.amount = self.cost_ret * weight_shifted
            
            # 检查乘法结果
            if self.amount.isnull().any().any():
                nan_count = self.amount.isnull().sum().sum()
                warnings.warn(
                    f"amount 计算后包含 {nan_count} 个 NaN 值\n"
                    f"修复建议: 检查 cost_ret 和 weight 的数据完整性",
                    UserWarning
                )
            
            # 计算累计金额
            sum_series = self.amount.sum(axis=1) + 1
            
            # 检查是否有负值或零值（可能导致 cumprod 异常）
            if (sum_series <= 0).any():
                negative_count = (sum_series <= 0).sum()
                warnings.warn(
                    f"amount.sum(axis=1)+1 包含 {negative_count} 个非正值\n"
                    f"修复建议: 检查权重和收益率数据",
                    UserWarning
                )
            
            self.amount['amount'] = sum_series.cumprod() * self.initial_amount
            
            # 检查最终结果
            if self.amount['amount'].isnull().any():
                raise BacktestDataError(
                    f"amount 计算后包含 NaN 值\n"
                    f"修复建议: 检查输入数据和计算过程"
                )
            
            try:
                if pd.api.types.is_numeric_dtype(self.amount['amount']):
                    if np.isinf(self.amount['amount']).any():
                        warnings.warn(
                            f"amount 包含 Inf 值\n"
                            f"修复建议: 检查数据是否有异常大的收益率",
                            UserWarning
                        )
            except (TypeError, ValueError):
                pass
            
            self.amount = self.amount['amount']
            
        except Exception as e:
            raise BacktestDataError(
                f"计算 amount 失败: {e}\n"
                f"cost_ret 样本:\n  {format_data_sample(self.cost_ret)}\n"
                f"weight 样本:\n  {format_data_sample(self.weight)}"
            ) from e
        
        return self.amount

    def get_daily_position_data(self):
        import datetime as dt
        db = Datafeed("daily_market_data")
        
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
                metric="收盘价(元)"
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
        try:
            # 验证计算输入
            validate_calculation_inputs(
                weight=self.weight,
                amount=self.amount,
                cost_price=self.cost_price
            )
            
            # 计算持仓金额
            self.position = self.weight.mul(self.amount, axis=0)
            
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
            
            # 除以成本价得到持仓数量
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
        
        # === 处理 MultiIndex ===
        try:
            # 检查是否有 input_ts level
            if isinstance(self.position.index, pd.MultiIndex):
                if 'input_ts' in self.position.index.names:
                    self.position = self.position.droplevel('input_ts')
                else:
                    warnings.warn(
                        f"position.index 是 MultiIndex 但不包含 'input_ts' level\n"
                        f"索引层级: {self.position.index.names}\n"
                        f"修复建议: 检查索引结构",
                        UserWarning
                    )
            
            # 删除重复索引
            if self.position.index.duplicated().any():
                dup_count = self.position.index.duplicated().sum()
                warnings.warn(
                    f"position 包含 {dup_count} 个重复索引，将删除重复项",
                    UserWarning
                )
                self.position = self.position.drop_duplicates()
            
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

#%%
'''
if __name__ == '__main__':
    # 虚拟的权重序列
    weights = pd.DataFrame(0.33333333, index=pd.date_range(start='2024-01-01 10:00:00', end='2025-01-01 10:00:00', freq='1D'),columns=['000010.SZ','000001.SZ','000002.SZ',])
    weights.index.name = "input_ts"

    bb = BacktestBase(weight=weights, symbol="", amount=1000000)
    #name = 结算持仓市值（不考虑余额）
'''