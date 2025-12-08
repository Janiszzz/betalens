#%%
'''
Todo:
    - 完成模型工具集模块modelkit
    - 完成数据工具集模块toolkit，逐步进化成Data类和方法
    - 多标的回测正确性检验
    - 迭代性质的滤波器基类开发
    - 可交易表、可交易日、是存表还是做定制化方法
    - 如果在日期和标的存在性不匹配的时候，错误的处理

'''
import pandas as pd
import datetime as dt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datafeed import Datafeed, get_absolute_trade_days
from backtest import BacktestBase
from analyst import PortfolioAnalyzer, ReportExporter


'''
单因子与双因子分组策略说明：

【单因子分组 Single Factor】
1. 生成调仓日序列，可先简化处理，后续必须手工清洗交易日序列
2. 查询得到每个调仓日的公司特征，不必展开面板
3. 有两种处理方式：都使用最近日查询，对年报未更新的，采取线性外推或删除的特殊处理
4. 选择得到top10%和bottom10%的个券，设置权重±2/n
5. 权重信号展开面板

【双因子分组 Double Sort】
双重排序用于同时考虑两个因子对投资组合的影响，分析因子间交互作用。

两种排序方法：

1. **独立排序（Independent Sort）**：
   - 对所有股票分别按两个因子独立分组
   - 例如：先按因子1对全部股票分N组，再按因子2对全部股票分M组
   - 取两个标签的交集形成N×M个组合
   - 优点：不受因子顺序影响，因子地位平等
   - 缺点：如果因子相关性高，某些组合股票数量可能极少或为空
   - 适用场景：因子之间相关性较低时

2. **条件排序（Dependent Sort）**：
   - 先按主因子分组，然后在每个主因子组内按次因子分组
   - 例如：先按因子1分N组，在每个组内再按因子2分M组
   - 形成N×M个投资组合，每组股票数量相对均匀
   - 优点：组合中股票数量均衡，避免异常值影响
   - 缺点：因子排序顺序会影响结果
   - 适用场景：因子相关性较高，或需要控制某个因子影响时

参数约定：
- date_ranges, code_ranges: 调仓日期和对应的可交易股票池
- metric1, metric2: 主因子和次因子的指标名称
- quantiles1, quantiles2: 两个因子的分组数或自定义分位数字典
- sort_method: 'independent' 或 'dependent'，选择排序方法
- 返回带有双标签的DataFrame，索引为(input_ts, code)，包含两个因子的标签列

'''

def get_tradable_pool(date_list):
    """
    获取可交易股票池
    
    Args:
        date_list: 日期列表
        
    Returns:
        date_ranges: 日期范围列表
        code_ranges: 每个日期对应的股票代码列表
    """
    from datetime import timedelta
    
    # 创建一次Datafeed实例，避免重复连接
    data = Datafeed("fundamental_data")
    df = pd.DataFrame()

    try:
        for date in date_list:
            start = date + timedelta(hours=9)
            end = date + timedelta(hours=15)
            
            # 使用 query_time_range 替代已废弃的 query_data
            request = data.query_time_range(
                codes=None,  # None表示查询所有代码
                start_date=str(start),
                end_date=str(end),
                metric="交易状态"
            )
            
            # 筛选交易状态为1的记录
            request = request.loc[request['value'] == 1]
            df = pd.concat([df, request[['datetime', 'code']]], ignore_index=True)

        date_ranges = df['datetime'].dt.date.drop_duplicates().tolist()
        grouped = df.groupby(df['datetime'].dt.date)['code'].apply(list).reset_index()
        code_ranges = grouped['code'].tolist()

        return date_ranges, code_ranges
    
    finally:
        # 确保关闭数据库连接
        data.close()


def pre_query_factor_data(date_list, metric, time_tolerance=24*2*365, table_name="fundamental_data", date_ranges=None, code_ranges=None):
    """
    批量预查询因子数据，生成符合因子排序函数要求的DataFrame
    
    该函数先通过 get_tradable_pool 验证股票交易状态，获取可交易股票池，
    然后批量查询多个调仓日期的因子数据，返回格式化的DataFrame，
    可直接用于 single_factor、double_factor、multi_factor 等排序函数。
    
    Args:
        date_list: 调仓日期列表，每个元素为 date 或 datetime 对象
        metric: 因子指标名称（字符串），如 "股息率(报告期)"
        time_tolerance: 时间容差（小时），默认 2年（24*2*365）
        table_name: 数据库表名，默认 "fundamental_data"
        date_ranges: 可选，日期范围列表（如果提供，将跳过 get_tradable_pool 调用）
        code_ranges: 可选，每个日期对应的股票代码列表（如果提供，必须与 date_ranges 同时提供且长度相等）
        
    Returns:
        DataFrame，包含以下列：
            - input_ts: 输入时间戳（调仓日期），datetime64[ns]
            - code: 股票代码，object (string)
            - {metric}: 因子值列，列名为metric参数值，float64
            - datetime: 匹配到的数据时间戳（数据库中实际数据的时间），datetime64[ns]
            - diff_hours: 时间差（小时），表示 input_ts 与 datetime 的差值，float64
            - name: 股票名称（如果数据库返回），object (string)
    
    Example:
        >>> date_list = get_absolute_trade_days("2015-04-30", "2024-04-30", "Y")
        >>> pre_queried_data = pre_query_factor_data(
        ...     date_list, "股息率(报告期)", time_tolerance=24*2*365
        ... )
        >>> labeled_pool = single_factor(pre_queried_data, "股息率(报告期)", {"股息率(报告期)": 10})
        
        性能优化示例（复用可交易池）：
        >>> date_ranges, code_ranges = get_tradable_pool(date_list)
        >>> pre_queried_data1 = pre_query_factor_data(
        ...     date_list, "市值", date_ranges=date_ranges, code_ranges=code_ranges
        ... )
        >>> pre_queried_data2 = pre_query_factor_data(
        ...     date_list, "账面市值比", date_ranges=date_ranges, code_ranges=code_ranges
        ... )
    """
    if not metric:
        raise ValueError("metric 不能为空")
    
    if (date_ranges is not None) != (code_ranges is not None):
        raise ValueError("date_ranges 和 code_ranges 必须同时提供或同时不提供")
    
    if date_ranges is not None and code_ranges is not None:
        if len(date_ranges) != len(code_ranges):
            raise ValueError("date_ranges 和 code_ranges 的长度必须相等")
    
    if date_ranges is None or code_ranges is None:
        date_ranges, code_ranges = get_tradable_pool(date_list)
    
    # 创建Datafeed实例
    data = Datafeed(table_name)
    all_results = pd.DataFrame()
    
    try:
        for i, date in enumerate(date_ranges):
            # 将 date 转换为 datetime（添加 10:00:00）
            datetime_str = f"{date} 10:00:00"
            
            # 获取该日期的股票代码列表（已通过交易状态验证）
            codes = code_ranges[i]
            
            if not codes:
                continue
            
            # 查询参数
            params = {
                'codes': codes,
                'datetimes': [datetime_str],
                'metric': metric,
                'time_tolerance': time_tolerance
            }
            
            # 查询数据
            result = data.query_nearest_before(params)
            
            # 如果查询结果为空，记录警告但不中断
            if result.empty:
                continue
            
            # 合并结果
            all_results = pd.concat([all_results, result], ignore_index=True)
        
        # 确保返回的DataFrame包含必需的列
        required_cols = ['input_ts', 'code', metric]
        missing_cols = [col for col in required_cols if col not in all_results.columns]
        if missing_cols:
            raise ValueError(f"查询结果缺少必需的列: {missing_cols}")
        
        # 确保 input_ts 是 datetime 类型
        if 'input_ts' in all_results.columns:
            all_results['input_ts'] = pd.to_datetime(all_results['input_ts'])
        
        # 确保 metric 列是数值类型
        if metric in all_results.columns:
            all_results[metric] = pd.to_numeric(all_results[metric], errors='coerce')
        
        return all_results
    
    finally:
        # 确保关闭数据库连接
        data.close()


def single_factor(pre_queried_data, metric, quantiles):
    """
    单因子分组打标签
    
    Args:
        pre_queried_data: DataFrame，包含所有日期的因子数据
            必需列：input_ts, code, {metric}（因子值列）
            可选列：datetime, diff_hours, name
        metric: 因子指标名称
        quantiles: 分位数字典，如 {"股息率(报告期)": 10}
        
    Returns:
        labeled_pool: 打标签后的DataFrame，包含因子分组标签
    """
    # 验证必需列存在
    required_cols = ['input_ts', 'code', metric]
    missing_cols = [col for col in required_cols if col not in pre_queried_data.columns]
    if missing_cols:
        raise ValueError(f"pre_queried_data 缺少必需的列: {missing_cols}")
    
    # 筛选并复制数据，确保不修改原始数据
    labeled_pool = pre_queried_data[required_cols].copy()
    
    # 可选：过滤年报时间
    # labeled_pool = labeled_pool.drop(labeled_pool[labeled_pool['年报时间'] < labeled_pool['datetime']].index)
    
    def single_sort(df, keys, quantile_dict):
        """单因子分组排序"""
        if len(keys) != len(quantile_dict):
            raise ValueError("keys 和 quantile_dict 的长度必须相等")
        for key in keys:
            df[key + '_label'] = pd.qcut(
                df[key].astype(float), 
                quantile_dict[key], 
                labels=False, 
                duplicates='drop'
            )
        return df

    # 按时间分组，对每组进行分位数分组
    labeled_pool = labeled_pool.groupby('input_ts', as_index=False).apply(
        lambda group: single_sort(group, [metric], quantiles)
    )
    labeled_pool.set_index(['input_ts', 'code'], inplace=True)

    # 将metric信息存储为列名后缀，而非DataFrame属性（更规范）
    labeled_pool.attrs['metric'] = metric
    
    return labeled_pool


def double_factor(pre_queried_data1, pre_queried_data2, metric1, metric2, quantiles1, quantiles2, sort_method='dependent'):
    """
    双因子分组打标签（Double Sort）
    支持独立排序（Independent Sort）和条件排序（Dependent Sort）
    
    Args:
        pre_queried_data1: DataFrame，包含主因子数据
            必需列：input_ts, code, {metric1}（主因子值列）
            可选列：datetime, diff_hours, name
        pre_queried_data2: DataFrame，包含次因子数据
            必需列：input_ts, code, {metric2}（次因子值列）
            可选列：datetime, diff_hours, name
        metric1: 主因子指标名称
        metric2: 次因子指标名称
        quantiles1: 主因子分位数字典，如 {"市值": 5}
        quantiles2: 次因子分位数字典，如 {"账面市值比": 5}
        sort_method: 排序方法，'independent' 或 'dependent'（默认）
                    - 'independent': 独立排序，分别对两个因子独立分组后取交集
                    - 'dependent': 条件排序，先按主因子分组，再在每组内按次因子分组
        
    Returns:
        labeled_pool: 打标签后的DataFrame，包含两个因子的分组标签
                     索引为(input_ts, code)，包含 metric1_label 和 metric2_label 列
    """
    # 验证必需列存在
    required_cols1 = ['input_ts', 'code', metric1]
    required_cols2 = ['input_ts', 'code', metric2]
    
    missing_cols1 = [col for col in required_cols1 if col not in pre_queried_data1.columns]
    missing_cols2 = [col for col in required_cols2 if col not in pre_queried_data2.columns]
    
    if missing_cols1:
        raise ValueError(f"pre_queried_data1 缺少必需的列: {missing_cols1}")
    if missing_cols2:
        raise ValueError(f"pre_queried_data2 缺少必需的列: {missing_cols2}")
    
    # 合并两个因子的数据
    labeled_pool = pd.merge(
        pre_queried_data1[required_cols1],
        pre_queried_data2[required_cols2],
        on=['input_ts', 'code'],
        how='inner'  # 只保留两个因子都有数据的股票
    )
    
    def independent_double_sort(df, factor1, factor2, quantile_dict1, quantile_dict2):
        """
        独立排序：分别对两个因子独立分组
        对全部股票分别按两个因子进行分组，然后取交集
        """
        # 对因子1独立分组（基于全部股票）
        try:
            df[factor1 + '_label'] = pd.qcut(
                df[factor1].astype(float), 
                quantile_dict1[factor1], 
                labels=False, 
                duplicates='drop'
            )
        except ValueError as e:
            print(f"警告：因子 {factor1} 独立分组失败: {e}")
            df[factor1 + '_label'] = 0
        
        # 对因子2独立分组（同样基于全部股票）
        try:
            df[factor2 + '_label'] = pd.qcut(
                df[factor2].astype(float),
                quantile_dict2[factor2],
                labels=False,
                duplicates='drop'
            )
        except ValueError as e:
            print(f"警告：因子 {factor2} 独立分组失败: {e}")
            df[factor2 + '_label'] = 0
        
        return df
    
    def dependent_double_sort(df, factor1, factor2, quantile_dict1, quantile_dict2):
        """
        条件排序：先按主因子分组，再在每组内按次因子分组
        """
        # 创建副本
        df = df.copy()
        
        # 步骤1：按主因子分组
        try:
            df[factor1 + '_label'] = pd.qcut(
                df[factor1].astype(float), 
                quantile_dict1[factor1], 
                labels=False, 
                duplicates='drop'
            )
        except ValueError as e:
            print(f"警告：主因子 {factor1} 分组失败: {e}")
            df[factor1 + '_label'] = 0
        
        # 步骤2：在每个主因子组内，对次因子分组
        # 初始化次因子标签列
        df[factor2 + '_label'] = 0
        
        for label1, group in df.groupby(factor1 + '_label'):
            try:
                labels = pd.qcut(
                    group[factor2].astype(float),
                    quantile_dict2[factor2],
                    labels=False,
                    duplicates='drop'
                )
                df.loc[group.index, factor2 + '_label'] = labels
            except ValueError as e:
                # 如果组内样本太少无法分组，保持为0
                pass
        
        return df
    
    # 根据排序方法选择对应的函数
    if sort_method == 'independent':
        sort_func = independent_double_sort
    elif sort_method == 'dependent':
        sort_func = dependent_double_sort
    else:
        raise ValueError(f"不支持的排序方法: {sort_method}，请使用 'independent' 或 'dependent'")
    
    # 按时间分组，对每组进行双因子分组
    result_list = []
    for input_ts, group in labeled_pool.groupby('input_ts'):
        sorted_group = sort_func(group, metric1, metric2, quantiles1, quantiles2)
        result_list.append(sorted_group)
    
    labeled_pool = pd.concat(result_list, ignore_index=True)
    labeled_pool.set_index(['input_ts', 'code'], inplace=True)
    
    # 将因子和排序方法信息存储为attrs（便于后续处理）
    labeled_pool.attrs['metric1'] = metric1
    labeled_pool.attrs['metric2'] = metric2
    labeled_pool.attrs['sort_method'] = sort_method
    
    return labeled_pool


def _recursive_multi_factor_sort(df, factors, current_index=0, parent_group=None):
    """
    递归多因子分组排序核心函数
    
    Args:
        df: 当前数据框
        factors: 因子配置列表，每个元素为 {'name': str, 'quantiles': int, 'method': 'independent'/'dependent'}
        current_index: 当前处理的因子索引
        parent_group: 父组标签（用于dependent排序）
        
    Returns:
        df: 添加了标签列的数据框
    """
    if current_index >= len(factors):
        return df
    
    factor_config = factors[current_index]
    factor_name = factor_config['name']
    quantiles = factor_config['quantiles']
    method = factor_config.get('method', 'dependent')
    label_col = factor_name + '_label'
    
    df = df.copy()
    
    if method == 'independent':
        # 独立排序：对所有股票独立分组
        try:
            df[label_col] = pd.qcut(
                df[factor_name].astype(float),
                quantiles,
                labels=False,
                duplicates='drop'
            )
        except ValueError as e:
            print(f"警告：因子 {factor_name} 独立分组失败: {e}")
            df[label_col] = 0
    else:
        # 条件排序：在父组内分组
        if parent_group is None:
            # 第一个因子，直接分组
            try:
                df[label_col] = pd.qcut(
                    df[factor_name].astype(float),
                    quantiles,
                    labels=False,
                    duplicates='drop'
                )
            except ValueError as e:
                print(f"警告：因子 {factor_name} 分组失败: {e}")
                df[label_col] = 0
        else:
            # 在父组内分组
            df[label_col] = 0
            for parent_label, group in df.groupby(parent_group):
                try:
                    labels = pd.qcut(
                        group[factor_name].astype(float),
                        quantiles,
                        labels=False,
                        duplicates='drop'
                    )
                    df.loc[group.index, label_col] = labels
                except ValueError:
                    pass
    
    # 递归处理下一个因子
    if method == 'dependent':
        # 条件排序：下一个因子在当前组内排序
        return _recursive_multi_factor_sort(df, factors, current_index + 1, label_col)
    else:
        # 独立排序：下一个因子独立排序
        return _recursive_multi_factor_sort(df, factors, current_index + 1, None)


def multi_factor(pre_queried_data_list, factors):
    """
    多因子分组打标签（Multi-Factor Sort）
    支持递归的独立排序和条件排序混合
    
    Args:
        pre_queried_data_list: DataFrame列表，每个DataFrame包含对应因子的数据
            列表顺序应与factors配置列表的顺序一致
            每个DataFrame必需列：input_ts, code, {metric}（因子值列）
            可选列：datetime, diff_hours, name
        factors: 因子配置列表，每个元素为字典：
            {
                'name': str,           # 因子名称，如 '市值'
                'quantiles': int,      # 分组数，如 5
                'method': str          # 排序方法：'independent' 或 'dependent'（默认'dependent'）
            }
            示例：
            [
                {'name': '市值', 'quantiles': 5, 'method': 'dependent'},
                {'name': '账面市值比', 'quantiles': 5, 'method': 'dependent'},
                {'name': '动量', 'quantiles': 3, 'method': 'independent'}
            ]
        
    Returns:
        labeled_pool: 打标签后的DataFrame，包含所有因子的分组标签
                     索引为(input_ts, code)，包含各因子的 _label 列
    """
    if len(factors) < 1:
        raise ValueError("至少需要1个因子")
    
    if len(pre_queried_data_list) != len(factors):
        raise ValueError(f"pre_queried_data_list 的长度({len(pre_queried_data_list)})必须与 factors 的长度({len(factors)})相等")
    
    # 验证并合并所有因子的数据
    merged = None
    for i, factor_config in enumerate(factors):
        metric = factor_config['name']
        pre_queried_data = pre_queried_data_list[i]
        
        # 验证必需列存在
        required_cols = ['input_ts', 'code', metric]
        missing_cols = [col for col in required_cols if col not in pre_queried_data.columns]
        if missing_cols:
            raise ValueError(f"pre_queried_data_list[{i}] (因子: {metric}) 缺少必需的列: {missing_cols}")
        
        if merged is None:
            merged = pre_queried_data[required_cols].copy()
        else:
            merged = pd.merge(
                merged,
                pre_queried_data[required_cols],
                on=['input_ts', 'code'],
                how='inner'  # 只保留所有因子都有数据的股票
            )
    
    labeled_pool = merged
    
    # 按时间分组，对每组进行多因子递归分组
    result_list = []
    for input_ts, group in labeled_pool.groupby('input_ts'):
        sorted_group = _recursive_multi_factor_sort(group, factors)
        result_list.append(sorted_group)
    
    labeled_pool = pd.concat(result_list, ignore_index=True)
    labeled_pool.set_index(['input_ts', 'code'], inplace=True)
    
    # 将因子配置信息存储为attrs（便于后续处理）
    labeled_pool.attrs['factors'] = factors
    labeled_pool.attrs['factor_count'] = len(factors)
    
    return labeled_pool


def get_single_factor_weight(labeled_pool, params):
    factor_key = params['factor_key']
    import numpy as np

    def f1(group):
        group = group.copy()
        weight_col = factor_key + '_weight'
        label_col = factor_key + '_label'
        group[weight_col] = 0
        max_label = group[label_col].max()
        min_label = group[label_col].min()
        group[weight_col] = np.where(group[label_col] == max_label, 1,
                                     np.where(group[label_col] == min_label, -1, 0))
        group = group[group[weight_col] != 0]
        return group
    
    def f2(group):
        # 步骤1: 初始化权重列
        group = group.copy()
        weight_col = factor_key + '_weight'
        label_col = factor_key + '_label'
        group[weight_col] = 0
        
        # 步骤2: 获取可选的高级参数
        group_weights = params.get('group_weights', {})  # 组间权重比例配置
        intra_group_allocation = params.get('intra_group_allocation', {})  # 组内分配方式配置
        
        # 步骤3: 获取做多做空标签列表
        long_labels = params['long']
        short_labels = params['short']
        
        # 步骤4: 提取多空两侧的权重配置
        long_group_weights = group_weights.get('long', {})
        short_group_weights = group_weights.get('short', {})
        long_intra_allocation = intra_group_allocation.get('long', {})
        short_intra_allocation = intra_group_allocation.get('short', {})
        
        # 步骤5: 计算做多组的权重总和（用于归一化）
        if long_group_weights:
            total_long_weight = sum(long_group_weights.get(label, 1) for label in long_labels)
        else:
            total_long_weight = len(long_labels)
        
        # 步骤6: 计算做空组的权重总和（用于归一化）
        if short_group_weights:
            total_short_weight = sum(short_group_weights.get(label, 1) for label in short_labels)
        else:
            total_short_weight = len(short_labels)
        
        # 步骤7: 遍历做多标签，分配权重
        for label in long_labels:
            # 步骤7.1: 筛选当前标签组的股票
            mask = group[label_col] == label
            group_stocks = group[mask]
            
            if len(group_stocks) == 0:
                continue
            
            # 步骤7.2: 计算当前组的组间权重
            if long_group_weights:
                group_weight = long_group_weights.get(label, 1) / total_long_weight
            else:
                group_weight = 1 / total_long_weight
            
            # 步骤7.3: 获取当前组的组内分配配置
            allocation_cfg = long_intra_allocation.get(label, {'method': 'equal'})
            
            # 步骤7.4: 按因子值分配组内权重
            if allocation_cfg.get('method') == 'factor_value' and len(group_stocks) > 1:
                metric = allocation_cfg.get('metric', factor_key)
                order = allocation_cfg.get('order', 'desc')
                
                if metric in group_stocks.columns:
                    # 步骤7.4.1: 提取因子值并填充缺失值
                    factor_values = group_stocks[metric].copy()
                    factor_values = factor_values.fillna(0)
                    
                    # 步骤7.4.2: 处理排序方向（升序时取负）
                    if order == 'asc':
                        factor_values = -factor_values
                    
                    # 步骤7.4.3: 将因子值平移到非负区间
                    factor_values = factor_values - factor_values.min()
                    factor_sum = factor_values.sum()
                    
                    # 步骤7.4.4: 按因子值比例分配权重
                    if factor_sum > 0:
                        stock_weights = factor_values / factor_sum
                    else:
                        stock_weights = pd.Series(1/len(group_stocks), index=group_stocks.index)
                    
                    # 步骤7.4.5: 乘以组权重得到最终权重
                    group.loc[mask, weight_col] = stock_weights * group_weight
                else:
                    # 步骤7.4.6: 因子不存在时等权分配
                    group.loc[mask, weight_col] = group_weight / len(group_stocks)
            else:
                # 步骤7.5: 等权分配组内权重
                group.loc[mask, weight_col] = group_weight / len(group_stocks)
        
        # 步骤8: 遍历做空标签，分配权重（逻辑同做多，权重取负）
        for label in short_labels:
            # 步骤8.1: 筛选当前标签组的股票
            mask = group[label_col] == label
            group_stocks = group[mask]
            
            if len(group_stocks) == 0:
                continue
            
            # 步骤8.2: 计算当前组的组间权重
            if short_group_weights:
                group_weight = short_group_weights.get(label, 1) / total_short_weight
            else:
                group_weight = 1 / total_short_weight
            
            # 步骤8.3: 获取当前组的组内分配配置
            allocation_cfg = short_intra_allocation.get(label, {'method': 'equal'})
            
            # 步骤8.4: 按因子值分配组内权重
            if allocation_cfg.get('method') == 'factor_value' and len(group_stocks) > 1:
                metric = allocation_cfg.get('metric', factor_key)
                order = allocation_cfg.get('order', 'desc')
                
                if metric in group_stocks.columns:
                    # 步骤8.4.1: 提取因子值并填充缺失值
                    factor_values = group_stocks[metric].copy()
                    factor_values = factor_values.fillna(0)
                    
                    # 步骤8.4.2: 处理排序方向（升序时取负）
                    if order == 'asc':
                        factor_values = -factor_values
                    
                    # 步骤8.4.3: 将因子值平移到非负区间
                    factor_values = factor_values - factor_values.min()
                    factor_sum = factor_values.sum()
                    
                    # 步骤8.4.4: 按因子值比例分配权重
                    if factor_sum > 0:
                        stock_weights = factor_values / factor_sum
                    else:
                        stock_weights = pd.Series(1/len(group_stocks), index=group_stocks.index)
                    
                    # 步骤8.4.5: 乘以组权重并取负得到做空权重
                    group.loc[mask, weight_col] = -stock_weights * group_weight
                else:
                    # 步骤8.4.6: 因子不存在时等权分配
                    group.loc[mask, weight_col] = -group_weight / len(group_stocks)
            else:
                # 步骤8.5: 等权分配组内权重（取负）
                group.loc[mask, weight_col] = -group_weight / len(group_stocks)
        
        # 步骤9: 过滤掉权重为0的股票
        group = group[group[weight_col] != 0]
        return group
    
    if(params['mode'] == 'classic-long-short'):
        labeled_pool = labeled_pool.groupby(labeled_pool.index.get_level_values(0),as_index=False).apply(f1)
    elif(params['mode'] == 'freeplay'):
        labeled_pool = labeled_pool.groupby(labeled_pool.index.get_level_values(0), as_index=False).apply(f2)

    weights = labeled_pool.filter(like='weight').reset_index().pivot(index="input_ts", columns="code", values = factor_key + '_weight')
    weights = weights.fillna(0)

    def normalize_row(row):
        positives = row[row > 0]
        negatives = row[row < 0]

        positive_sum = positives.sum()
        negative_abs_sum = abs(negatives.sum())

        if positive_sum > 0:
            row[row > 0] = positives / positive_sum

        if negative_abs_sum > 0:
            row[row < 0] = negatives / negative_abs_sum

        return row

    weights = weights.apply(normalize_row, axis=1)

    return weights

def describe_labeled_pool(labeled_pool):
    """
    描述打标签后的因子池统计信息
    
    Args:
        labeled_pool: 带标签的因子池DataFrame
        
    Returns:
        pivot: 透视表，包含每个标签组的样本数和均值
    """
    # 从DataFrame属性中获取metric名称
    metric = labeled_pool.attrs.get('metric')
    
    if metric is None:
        # 向后兼容：尝试从旧方式获取
        metric = getattr(labeled_pool, 'metric', None)
        if metric is None:
            raise ValueError("labeled_pool 缺少 'metric' 信息，请确保使用 single_factor 函数生成")
    
    pivot = pd.pivot_table(
        data=labeled_pool.reset_index(),
        index='input_ts',
        columns=metric + '_label',
        values=metric,
        aggfunc=['count', 'mean'],  # 同时计算数量和平均值
        margins=True,  # 添加总计行/列
        margins_name='Total'
    )
    return pivot


def describe_double_labeled_pool(labeled_pool):
    """
    描述双因子打标签后的因子池统计信息
    
    Args:
        labeled_pool: 带双标签的因子池DataFrame（由double_factor生成）
        
    Returns:
        count_pivot: 各组合的样本数统计
        mean_pivot1: 主因子在各组合中的均值
        mean_pivot2: 次因子在各组合中的均值
    """
    # 从DataFrame属性中获取两个metric名称和排序方法
    metric1 = labeled_pool.attrs.get('metric1')
    metric2 = labeled_pool.attrs.get('metric2')
    sort_method = labeled_pool.attrs.get('sort_method', 'unknown')
    
    if metric1 is None or metric2 is None:
        raise ValueError("labeled_pool 缺少 'metric1' 或 'metric2' 信息，请确保使用 double_factor 函数生成")
    
    print(f"\n排序方法: {sort_method.upper()}")
    print(f"  - Independent Sort: 独立排序，两因子分别独立分组后取交集")
    print(f"  - Dependent Sort: 条件排序，先按主因子分组，再在每组内按次因子分组\n")
    
    df_reset = labeled_pool.reset_index()
    
    # 检查列名
    label_col1 = metric1 + '_label'
    label_col2 = metric2 + '_label'
    
    if label_col1 not in df_reset.columns or label_col2 not in df_reset.columns:
        print(f"警告：缺少标签列")
        print(f"可用列: {df_reset.columns.tolist()}")
        print(f"需要的列: {label_col1}, {label_col2}")
        possible_cols = [col for col in df_reset.columns if '_label' in col]
        print(f"找到的标签列: {possible_cols}")
        raise ValueError(f"DataFrame中缺少标签列 '{label_col1}' 或 '{label_col2}'")
    
    # 确保标签列是数值类型（先填充NaN）
    df_reset[label_col1] = df_reset[label_col1].fillna(0).astype(int)
    df_reset[label_col2] = df_reset[label_col2].fillna(0).astype(int)
    
    # 统计各组合的样本数
    count_pivot = pd.crosstab(
        index=df_reset[label_col1],
        columns=df_reset[label_col2],
        margins=True,
        margins_name='Total'
    )
    
    # 主因子的均值分布
    mean_pivot1 = df_reset.groupby([label_col1, label_col2])[metric1].mean().unstack(fill_value=0)
    mean_pivot1['Total'] = df_reset.groupby(label_col1)[metric1].mean()
    mean_pivot1.loc['Total'] = df_reset.groupby(label_col2)[metric1].mean().tolist() + [df_reset[metric1].mean()]
    
    # 次因子的均值分布
    mean_pivot2 = df_reset.groupby([label_col1, label_col2])[metric2].mean().unstack(fill_value=0)
    mean_pivot2['Total'] = df_reset.groupby(label_col1)[metric2].mean()
    mean_pivot2.loc['Total'] = df_reset.groupby(label_col2)[metric2].mean().tolist() + [df_reset[metric2].mean()]
    
    return count_pivot, mean_pivot1, mean_pivot2


def describe_multi_labeled_pool(labeled_pool, max_display_dims=2):
    """
    描述多因子打标签后的因子池统计信息
    
    Args:
        labeled_pool: 带多标签的因子池DataFrame（由multi_factor生成）
        max_display_dims: 最大显示维度（默认2，即显示前2个因子的交叉统计）
        
    Returns:
        stats_dict: 包含统计信息的字典
            - 'count_pivot': 各组合的样本数统计（前max_display_dims个因子）
            - 'mean_pivots': 各因子在各组合中的均值（字典，key为因子名）
            - 'factor_info': 因子配置信息
    """
    factors = labeled_pool.attrs.get('factors', [])
    
    if not factors:
        raise ValueError("labeled_pool 缺少 'factors' 信息，请确保使用 multi_factor 函数生成")
    
    df_reset = labeled_pool.reset_index()
    
    # 获取前max_display_dims个因子用于展示
    display_factors = factors[:min(max_display_dims, len(factors))]
    
    # 构建标签列名列表
    label_cols = [f['name'] + '_label' for f in display_factors]
    
    # 检查并填充标签列
    for i, factor_config in enumerate(display_factors):
        label_col = factor_config['name'] + '_label'
        if label_col not in df_reset.columns:
            raise ValueError(f"缺少标签列: {label_col}")
        df_reset[label_col] = df_reset[label_col].fillna(0).astype(int)
    
    # 统计各组合的样本数（使用前两个因子）
    if len(display_factors) >= 2:
        count_pivot = pd.crosstab(
            index=df_reset[label_cols[0]],
            columns=df_reset[label_cols[1]],
            margins=True,
            margins_name='Total'
        )
    elif len(display_factors) == 1:
        count_pivot = df_reset[label_cols[0]].value_counts().sort_index()
        count_pivot['Total'] = count_pivot.sum()
        count_pivot = count_pivot.to_frame('Count').T
    else:
        count_pivot = pd.DataFrame()
    
    # 计算各因子在各组合中的均值
    mean_pivots = {}
    for factor_config in factors:
        factor_name = factor_config['name']
        if factor_name not in df_reset.columns:
            continue
        
        if len(display_factors) >= 2:
            mean_pivot = df_reset.groupby(label_cols)[factor_name].mean().unstack(fill_value=0)
            # 添加总计行和列
            mean_pivot['Total'] = df_reset.groupby(label_cols[0])[factor_name].mean()
            mean_pivot.loc['Total'] = df_reset.groupby(label_cols[1])[factor_name].mean().tolist() + [df_reset[factor_name].mean()]
        elif len(display_factors) == 1:
            mean_pivot = df_reset.groupby(label_cols[0])[factor_name].mean()
            mean_pivot['Total'] = df_reset[factor_name].mean()
            mean_pivot = mean_pivot.to_frame('Mean').T
        else:
            mean_pivot = pd.DataFrame()
        
        mean_pivots[factor_name] = mean_pivot
    
    return {
        'count_pivot': count_pivot,
        'mean_pivots': mean_pivots,
        'factor_info': factors,
        'display_factors': [f['name'] for f in display_factors]
    }


def get_multi_factor_weight(labeled_pool, params):
    """
    根据多因子标签生成多空权重
    
    Args:
        labeled_pool: 带多标签的因子池DataFrame（由multi_factor生成）
        params: 参数字典，包含：
            - 'mode': 'classic-long-short' 或 'freeplay'
            - 'long_combinations': 做多组合列表，如 [(0,4,2), (1,4,2)] 表示各因子标签组合
            - 'short_combinations': 做空组合列表
            - 对于 'classic-long-short' 模式，自动做多最高组，做空最低组
            
    Returns:
        weights: 权重DataFrame，索引为input_ts，列为code
    """
    import numpy as np
    
    factors = labeled_pool.attrs.get('factors', [])
    if not factors:
        raise ValueError("labeled_pool 缺少 'factors' 信息，请确保使用 multi_factor 函数生成")
    
    factor_names = [f['name'] for f in factors]
    label_cols = [name + '_label' for name in factor_names]
    
    def assign_weights(group):
        # 步骤1: 初始化权重列
        group = group.copy()
        group['weight'] = 0
        
        # 步骤2: 获取可选的高级参数
        group_weights = params.get('group_weights', {})  # 组间权重比例配置
        intra_group_allocation = params.get('intra_group_allocation', {})  # 组内分配方式配置
        
        # 步骤3: 提取多空两侧的权重配置
        long_group_weights = group_weights.get('long', {})
        short_group_weights = group_weights.get('short', {})
        long_intra_allocation = intra_group_allocation.get('long', {})
        short_intra_allocation = intra_group_allocation.get('short', {})
        
        if params['mode'] == 'classic-long-short':
            # 步骤4: 经典多空模式 - 做多最高组，做空最低组
            # 步骤4.1: 找到所有因子的最大和最小标签
            max_labels = [group[col].max() for col in label_cols]
            min_labels = [group[col].min() for col in label_cols]
            
            # 步骤4.2: 构建做多掩码（所有因子都是最高组）
            mask_long = True
            for i, col in enumerate(label_cols):
                mask_long = mask_long & (group[col] == max_labels[i])
            group.loc[mask_long, 'weight'] = 1
            
            # 步骤4.3: 构建做空掩码（所有因子都是最低组）
            mask_short = True
            for i, col in enumerate(label_cols):
                mask_short = mask_short & (group[col] == min_labels[i])
            group.loc[mask_short, 'weight'] = -1
            
        elif params['mode'] == 'freeplay':
            # 步骤5: 自由模式 - 按指定组合分配权重
            # 步骤5.1: 获取做多做空组合列表
            long_combos = params.get('long_combinations', [])
            short_combos = params.get('short_combinations', [])
            
            # 步骤5.2: 计算做多组合的权重总和（用于归一化）
            if long_group_weights:
                total_long_weight = sum(long_group_weights.get(combo, 1) for combo in long_combos)
            else:
                total_long_weight = len(long_combos) if long_combos else 1
            
            # 步骤5.3: 计算做空组合的权重总和（用于归一化）
            if short_group_weights:
                total_short_weight = sum(short_group_weights.get(combo, 1) for combo in short_combos)
            else:
                total_short_weight = len(short_combos) if short_combos else 1
            
            # 步骤5.4: 遍历做多组合，分配权重
            for combo in long_combos:
                # 步骤5.4.1: 验证组合维度
                if len(combo) != len(label_cols):
                    print(f"警告：组合 {combo} 的维度({len(combo)})与因子数({len(label_cols)})不匹配，跳过")
                    continue
                
                # 步骤5.4.2: 构建组合掩码（所有因子标签都匹配）
                mask = True
                for i, col in enumerate(label_cols):
                    mask = mask & (group[col] == combo[i])
                
                # 步骤5.4.3: 筛选当前组合的股票
                group_stocks = group[mask]
                if len(group_stocks) == 0:
                    continue
                
                # 步骤5.4.4: 计算当前组合的组间权重
                if long_group_weights:
                    group_weight = long_group_weights.get(combo, 1) / total_long_weight
                else:
                    group_weight = 1 / total_long_weight
                
                # 步骤5.4.5: 获取当前组合的组内分配配置
                allocation_cfg = long_intra_allocation.get(combo, {'method': 'equal'})
                
                # 步骤5.4.6: 按因子值分配组内权重
                if allocation_cfg.get('method') == 'factor_value' and len(group_stocks) > 1:
                    metric = allocation_cfg.get('metric')
                    order = allocation_cfg.get('order', 'desc')
                    
                    if metric and metric in group_stocks.columns:
                        # 步骤5.4.6.1: 提取因子值并填充缺失值
                        factor_values = group_stocks[metric].copy()
                        factor_values = factor_values.fillna(0)
                        
                        # 步骤5.4.6.2: 处理排序方向（升序时取负）
                        if order == 'asc':
                            factor_values = -factor_values
                        
                        # 步骤5.4.6.3: 将因子值平移到非负区间
                        factor_values = factor_values - factor_values.min()
                        factor_sum = factor_values.sum()
                        
                        # 步骤5.4.6.4: 按因子值比例分配权重
                        if factor_sum > 0:
                            stock_weights = factor_values / factor_sum
                        else:
                            stock_weights = pd.Series(1/len(group_stocks), index=group_stocks.index)
                        
                        # 步骤5.4.6.5: 乘以组权重得到最终权重
                        group.loc[mask, 'weight'] = stock_weights * group_weight
                    else:
                        # 步骤5.4.6.6: 因子不存在时等权分配
                        group.loc[mask, 'weight'] = group_weight / len(group_stocks)
                else:
                    # 步骤5.4.7: 等权分配组内权重
                    group.loc[mask, 'weight'] = group_weight / len(group_stocks)
            
            # 步骤5.5: 遍历做空组合，分配权重（逻辑同做多，权重取负）
            for combo in short_combos:
                # 步骤5.5.1: 验证组合维度
                if len(combo) != len(label_cols):
                    print(f"警告：组合 {combo} 的维度({len(combo)})与因子数({len(label_cols)})不匹配，跳过")
                    continue
                
                # 步骤5.5.2: 构建组合掩码（所有因子标签都匹配）
                mask = True
                for i, col in enumerate(label_cols):
                    mask = mask & (group[col] == combo[i])
                
                # 步骤5.5.3: 筛选当前组合的股票
                group_stocks = group[mask]
                if len(group_stocks) == 0:
                    continue
                
                # 步骤5.5.4: 计算当前组合的组间权重
                if short_group_weights:
                    group_weight = short_group_weights.get(combo, 1) / total_short_weight
                else:
                    group_weight = 1 / total_short_weight
                
                # 步骤5.5.5: 获取当前组合的组内分配配置
                allocation_cfg = short_intra_allocation.get(combo, {'method': 'equal'})
                
                # 步骤5.5.6: 按因子值分配组内权重
                if allocation_cfg.get('method') == 'factor_value' and len(group_stocks) > 1:
                    metric = allocation_cfg.get('metric')
                    order = allocation_cfg.get('order', 'desc')
                    
                    if metric and metric in group_stocks.columns:
                        # 步骤5.5.6.1: 提取因子值并填充缺失值
                        factor_values = group_stocks[metric].copy()
                        factor_values = factor_values.fillna(0)
                        
                        # 步骤5.5.6.2: 处理排序方向（升序时取负）
                        if order == 'asc':
                            factor_values = -factor_values
                        
                        # 步骤5.5.6.3: 将因子值平移到非负区间
                        factor_values = factor_values - factor_values.min()
                        factor_sum = factor_values.sum()
                        
                        # 步骤5.5.6.4: 按因子值比例分配权重
                        if factor_sum > 0:
                            stock_weights = factor_values / factor_sum
                        else:
                            stock_weights = pd.Series(1/len(group_stocks), index=group_stocks.index)
                        
                        # 步骤5.5.6.5: 乘以组权重并取负得到做空权重
                        group.loc[mask, 'weight'] = -stock_weights * group_weight
                    else:
                        # 步骤5.5.6.6: 因子不存在时等权分配
                        group.loc[mask, 'weight'] = -group_weight / len(group_stocks)
                else:
                    # 步骤5.5.7: 等权分配组内权重（取负）
                    group.loc[mask, 'weight'] = -group_weight / len(group_stocks)
        
        # 步骤6: 过滤掉权重为0的股票
        group = group[group['weight'] != 0]
        return group
    
    labeled_pool = labeled_pool.groupby(labeled_pool.index.get_level_values(0), as_index=False).apply(assign_weights)
    
    weights = labeled_pool.filter(like='weight').reset_index().pivot(
        index="input_ts", 
        columns="code", 
        values='weight'
    )
    weights = weights.fillna(0)
    
    def normalize_row(row):
        positives = row[row > 0]
        negatives = row[row < 0]
        
        positive_sum = positives.sum()
        negative_abs_sum = abs(negatives.sum())
        
        if positive_sum > 0:
            row[row > 0] = positives / positive_sum
        
        if negative_abs_sum > 0:
            row[row < 0] = negatives / negative_abs_sum
        
        return row
    
    weights = weights.apply(normalize_row, axis=1)
    
    return weights


def get_double_factor_weight(labeled_pool, params):
    """
    根据双因子标签生成多空权重
    
    Args:
        labeled_pool: 带双标签的因子池DataFrame（由double_factor生成）
        params: 参数字典，包含：
            - factor_key1: 主因子名称
            - factor_key2: 次因子名称
            - mode: 'classic-long-short' 或 'freeplay'
            - long_combinations: 做多组合列表，如 [(0,4), (1,4)] 表示主因子0/1组且次因子4组
            - short_combinations: 做空组合列表，如 [(4,0), (4,1)]
            
    Returns:
        weights: 权重DataFrame，索引为input_ts，列为code
    """
    import numpy as np
    
    factor_key1 = params['factor_key1']
    factor_key2 = params['factor_key2']
    label_col1 = factor_key1 + '_label'
    label_col2 = factor_key2 + '_label'
    
    def assign_weights(group):
        group = group.copy()
        group['weight'] = 0
        
        group_weights = params.get('group_weights', {})
        intra_group_allocation = params.get('intra_group_allocation', {})
        
        long_group_weights = group_weights.get('long', {})
        short_group_weights = group_weights.get('short', {})
        long_intra_allocation = intra_group_allocation.get('long', {})
        short_intra_allocation = intra_group_allocation.get('short', {})
        
        if params['mode'] == 'classic-long-short':
            max_label1 = group[label_col1].max()
            min_label1 = group[label_col1].min()
            max_label2 = group[label_col2].max()
            min_label2 = group[label_col2].min()
            
            group.loc[(group[label_col1] == max_label1) & (group[label_col2] == max_label2), 'weight'] = 1
            group.loc[(group[label_col1] == min_label1) & (group[label_col2] == min_label2), 'weight'] = -1
            
        elif params['mode'] == 'freeplay':
            long_combos = params.get('long_combinations', [])
            short_combos = params.get('short_combinations', [])
            
            if long_group_weights:
                total_long_weight = sum(long_group_weights.get(combo, 1) for combo in long_combos)
            else:
                total_long_weight = len(long_combos) if long_combos else 1
            
            if short_group_weights:
                total_short_weight = sum(short_group_weights.get(combo, 1) for combo in short_combos)
            else:
                total_short_weight = len(short_combos) if short_combos else 1
            
            for combo in long_combos:
                label1_val, label2_val = combo
                mask = (group[label_col1] == label1_val) & (group[label_col2] == label2_val)
                group_stocks = group[mask]
                
                if len(group_stocks) == 0:
                    continue
                
                if long_group_weights:
                    group_weight = long_group_weights.get(combo, 1) / total_long_weight
                else:
                    group_weight = 1 / total_long_weight
                
                allocation_cfg = long_intra_allocation.get(combo, {'method': 'equal'})
                
                if allocation_cfg.get('method') == 'factor_value' and len(group_stocks) > 1:
                    metric = allocation_cfg.get('metric')
                    order = allocation_cfg.get('order', 'desc')
                    
                    if metric and metric in group_stocks.columns:
                        factor_values = group_stocks[metric].copy()
                        factor_values = factor_values.fillna(0)
                        
                        if order == 'asc':
                            factor_values = -factor_values
                        
                        factor_values = factor_values - factor_values.min()
                        factor_sum = factor_values.sum()
                        
                        if factor_sum > 0:
                            stock_weights = factor_values / factor_sum
                        else:
                            stock_weights = pd.Series(1/len(group_stocks), index=group_stocks.index)
                        
                        group.loc[mask, 'weight'] = stock_weights * group_weight
                    else:
                        group.loc[mask, 'weight'] = group_weight / len(group_stocks)
                else:
                    group.loc[mask, 'weight'] = group_weight / len(group_stocks)
            
            for combo in short_combos:
                label1_val, label2_val = combo
                mask = (group[label_col1] == label1_val) & (group[label_col2] == label2_val)
                group_stocks = group[mask]
                
                if len(group_stocks) == 0:
                    continue
                
                if short_group_weights:
                    group_weight = short_group_weights.get(combo, 1) / total_short_weight
                else:
                    group_weight = 1 / total_short_weight
                
                allocation_cfg = short_intra_allocation.get(combo, {'method': 'equal'})
                
                if allocation_cfg.get('method') == 'factor_value' and len(group_stocks) > 1:
                    metric = allocation_cfg.get('metric')
                    order = allocation_cfg.get('order', 'desc')
                    
                    if metric and metric in group_stocks.columns:
                        factor_values = group_stocks[metric].copy()
                        factor_values = factor_values.fillna(0)
                        
                        if order == 'asc':
                            factor_values = -factor_values
                        
                        factor_values = factor_values - factor_values.min()
                        factor_sum = factor_values.sum()
                        
                        if factor_sum > 0:
                            stock_weights = factor_values / factor_sum
                        else:
                            stock_weights = pd.Series(1/len(group_stocks), index=group_stocks.index)
                        
                        group.loc[mask, 'weight'] = -stock_weights * group_weight
                    else:
                        group.loc[mask, 'weight'] = -group_weight / len(group_stocks)
                else:
                    group.loc[mask, 'weight'] = -group_weight / len(group_stocks)
        
        group = group[group['weight'] != 0]
        return group
    
    labeled_pool = labeled_pool.groupby(labeled_pool.index.get_level_values(0), as_index=False).apply(assign_weights)
    
    weights = labeled_pool.filter(like='weight').reset_index().pivot(
        index="input_ts", 
        columns="code", 
        values='weight'
    )
    weights = weights.fillna(0)
    
    def normalize_row(row):
        positives = row[row > 0]
        negatives = row[row < 0]
        
        positive_sum = positives.sum()
        negative_abs_sum = abs(negatives.sum())
        
        if positive_sum > 0:
            row[row > 0] = positives / positive_sum
        
        if negative_abs_sum > 0:
            row[row < 0] = negatives / negative_abs_sum
        
        return row
    
    weights = weights.apply(normalize_row, axis=1)
    
    return weights

#%%
if __name__ == '__main__':
    """
    单因子回测示例：
    1. 生成调仓日序列
    2. 获取可交易股票池
    3. 对因子分组打标签
    4. 生成多空权重
    5. 回测并输出绩效报告
    """
    
    # 1. 生成调仓日序列（年度频率）
    date_list = get_absolute_trade_days("2015-04-30", "2024-04-30", "Y")
    
    # 2. 获取可交易股票池（在流程开始处调用一次）
    date_ranges, code_ranges = get_tradable_pool(date_list)
    
    # 3. 预查询因子数据（传入可交易池，避免重复查询）
    metric = "股息率(报告期)"
    pre_queried_data = pre_query_factor_data(
        date_list, 
        metric, 
        time_tolerance=24*2*365,
        table_name="fundamental_data",
        date_ranges=date_ranges,
        code_ranges=code_ranges
    )
    
    # 4. 单因子分组
    quantiles = {
        "股息率(报告期)": 10,  # 分为10组
        # "股息率(报告期)": [0, .2, .4, .6, .8, 1]  # 或使用自定义分位数
    }
    
    labeled_pool = single_factor(pre_queried_data, metric, quantiles)
    
    # 查看因子分组统计
    pivot_stats = describe_labeled_pool(labeled_pool)
    print("因子分组统计：")
    print(pivot_stats)
    
    # 5. 生成多空权重
    params = {
        'factor_key': '股息率(报告期)',
        'mode': 'freeplay',  # 'classic-long-short' 或 'freeplay'
        'long': [1],   # 做多组（分位数编号）
        'short': [8],  # 做空组（分位数编号）
    }
    weights = get_single_factor_weight(labeled_pool, params)
    
    # 调试信息：查看权重数据
    print("\n权重数据统计：")
    print(f"权重形状: {weights.shape}")
    print(f"非零权重数量: {(weights != 0).sum().sum()}")
    print(f"权重范围: [{weights.min().min():.4f}, {weights.max().max():.4f}]")
    print(f"每行权重和范围: [{weights.sum(axis=1).min():.4f}, {weights.sum(axis=1).max():.4f}]")
    print("\n前几行权重数据：")
    print(weights.head())
    
    # 添加现金列
    weights['cash'] = 0
    
    # 6. 回测
    bb = BacktestBase(weight=weights, symbol="", amount=1000000)
    
    # 绘制净值曲线
    bb.nav.plot()
    
    # 7. 绩效分析
    analyzer = PortfolioAnalyzer(bb.nav)
    exporter = ReportExporter(analyzer)
    exporter.generate_annual_report()  # 分年度输出
    exporter.generate_custom_report('2024-01-01', '2024-12-31')  # 指定时段
    
    print("\n" + "="*80)
    print("双因子回测示例 (Double Factor Backtest)")
    print("="*80 + "\n")
    
    # ========== 双因子回测示例 ==========
    # 设置开关：是否运行双因子示例
    RUN_DOUBLE_FACTOR = False  # 改为 True 以运行双因子示例
    
    if RUN_DOUBLE_FACTOR:
        # 1. 使用相同的调仓日序列
        # date_list 已在上面定义
        
        # 2. 获取可交易股票池（在流程开始处调用一次）
        date_ranges, code_ranges = get_tradable_pool(date_list)
        
        # 3. 预查询双因子数据（传入相同的可交易池，避免重复查询）
        metric1 = "市净率"  # 主因子
        metric2 = "股息率(报告期)"  # 次因子
        
        pre_queried_data1 = pre_query_factor_data(
            date_list, 
            metric1, 
            time_tolerance=24*2*365,
            table_name="fundamental_data",
            date_ranges=date_ranges,
            code_ranges=code_ranges
        )
        
        pre_queried_data2 = pre_query_factor_data(
            date_list, 
            metric2, 
            time_tolerance=24*2*365,
            table_name="fundamental_data",
            date_ranges=date_ranges,
            code_ranges=code_ranges
        )
        
        # 3. 双因子分组 - 演示两种排序方法
        quantiles1 = {metric1: 3}  # 主因子分3组（小、中、大）
        quantiles2 = {metric2: 3}  # 次因子分3组（低、中、高）
        
        print(f"双因子设置：主因子={metric1}({quantiles1[metric1]}组), 次因子={metric2}({quantiles2[metric2]}组)")
        
        # ===== 方法1：条件排序（Dependent Sort）=====
        print("\n" + "-"*80)
        print("方法1：条件排序（Dependent Sort）")
        print("-"*80)
        
        labeled_pool_dependent = double_factor(
            pre_queried_data1, 
            pre_queried_data2, 
            metric1, 
            metric2, 
            quantiles1, 
            quantiles2,
            sort_method='dependent'  # 条件排序
        )
        
        # 查看条件排序的统计
        count_pivot_dep, mean_pivot1_dep, mean_pivot2_dep = describe_double_labeled_pool(labeled_pool_dependent)
        
        print("各组合样本数统计（行=主因子组，列=次因子组）：")
        print(count_pivot_dep)
        
        # ===== 方法2：独立排序（Independent Sort）=====
        print("\n" + "-"*80)
        print("方法2：独立排序（Independent Sort）")
        print("-"*80)
        
        labeled_pool_independent = double_factor(
            pre_queried_data1, 
            pre_queried_data2, 
            metric1, 
            metric2, 
            quantiles1, 
            quantiles2,
            sort_method='independent'  # 独立排序
        )
        
        # 查看独立排序的统计
        count_pivot_indep, mean_pivot1_indep, mean_pivot2_indep = describe_double_labeled_pool(labeled_pool_independent)
        
        print("各组合样本数统计（行=主因子组，列=次因子组）：")
        print(count_pivot_indep)
        
        # ===== 对比两种方法的差异 =====
        print("\n" + "-"*80)
        print("两种排序方法的样本数差异（Independent - Dependent）：")
        print("-"*80)
        diff_pivot = count_pivot_indep - count_pivot_dep
        print(diff_pivot)
        
        # 使用条件排序的结果进行后续回测
        labeled_pool_double = labeled_pool_dependent
        
        # 3. 查看双因子分组统计（使用条件排序）
        count_pivot, mean_pivot1, mean_pivot2 = describe_double_labeled_pool(labeled_pool_double)
        
        print("\n各组合样本数统计（行=主因子组，列=次因子组）：")
        print(count_pivot)
        
        print(f"\n主因子({metric1})在各组合中的均值：")
        print(mean_pivot1)
        
        print(f"\n次因子({metric2})在各组合中的均值：")
        print(mean_pivot2)
        
        # 4. 生成双因子多空权重
        # 策略：做多小市净率+高股息率（价值股），做空大市净率+低股息率（成长股）
        double_params = {
            'factor_key1': metric1,
            'factor_key2': metric2,
            'mode': 'freeplay',
            'long_combinations': [(0, 2)],   # 做多：主因子最小组(0) + 次因子最大组(2)
            'short_combinations': [(2, 0)],  # 做空：主因子最大组(2) + 次因子最小组(0)
        }
        
        weights_double = get_double_factor_weight(labeled_pool_double, double_params)
        
        # 调试信息
        print("\n双因子权重数据统计：")
        print(f"权重形状: {weights_double.shape}")
        print(f"非零权重数量: {(weights_double != 0).sum().sum()}")
        print(f"权重范围: [{weights_double.min().min():.4f}, {weights_double.max().max():.4f}]")
        print(f"每行权重和: [{weights_double.sum(axis=1).min():.4f}, {weights_double.sum(axis=1).max():.4f}]")
        
        # 添加现金列
        weights_double['cash'] = 0
        
        # 5. 双因子回测
        bb_double = BacktestBase(weight=weights_double, symbol="", amount=1000000)
        
        # 绘制净值曲线
        bb_double.nav.plot()
        
        # 6. 双因子绩效分析
        analyzer_double = PortfolioAnalyzer(bb_double.nav)
        exporter_double = ReportExporter(analyzer_double)
        
        print("\n双因子策略绩效报告：")
        exporter_double.generate_annual_report()
        exporter_double.generate_custom_report('2024-01-01', '2024-12-31')
        
        print("\n双因子回测完成！")
    else:
        print("提示：将 RUN_DOUBLE_FACTOR 设为 True 以运行双因子示例")
    
    print("\n" + "="*80)
    print("多因子回测示例 (Multi-Factor Backtest)")
    print("="*80 + "\n")
    
    # ========== 多因子回测示例 ==========
    # 设置开关：是否运行多因子示例
    RUN_MULTI_FACTOR = False  # 改为 True 以运行多因子示例
    
    if RUN_MULTI_FACTOR:
        # 1. 使用相同的调仓日序列
        # date_list 已在上面定义
        
        # 2. 获取可交易股票池（在流程开始处调用一次）
        date_ranges, code_ranges = get_tradable_pool(date_list)
        
        # 3. 预查询多因子数据 - 3因子示例：市值、账面市值比、股息率
        factors_config = [
            {'name': '市值', 'quantiles': 3, 'method': 'dependent'},      # 先按市值分3组
            {'name': '账面市值比', 'quantiles': 3, 'method': 'dependent'},  # 在市值组内按账面市值比分3组
            {'name': '股息率(报告期)', 'quantiles': 2, 'method': 'independent'},  # 独立按股息率分2组
        ]
        
        print(f"多因子设置：")
        for i, f in enumerate(factors_config):
            print(f"  因子{i+1}: {f['name']} ({f['quantiles']}组, {f['method']})")
        total_combos = 1
        for f in factors_config:
            total_combos *= f['quantiles']
        print(f"  预期形成: {total_combos} 个组合\n")
        
        # 预查询每个因子的数据（传入相同的可交易池，避免重复查询）
        pre_queried_data_list = []
        for factor_config in factors_config:
            metric = factor_config['name']
            pre_queried_data = pre_query_factor_data(
                date_list, 
                metric, 
                time_tolerance=24*2*365,
                table_name="fundamental_data",
                date_ranges=date_ranges,
                code_ranges=code_ranges
            )
            pre_queried_data_list.append(pre_queried_data)
        
        # 3. 多因子分组
        labeled_pool_multi = multi_factor(
            pre_queried_data_list, 
            factors_config
        )
        
        # 4. 查看多因子分组统计
        stats = describe_multi_labeled_pool(labeled_pool_multi, max_display_dims=2)
        
        print("各组合样本数统计（前2个因子交叉）：")
        print(stats['count_pivot'])
        
        print(f"\n各因子在各组合中的均值：")
        for factor_name, mean_pivot in stats['mean_pivots'].items():
            print(f"\n{factor_name}:")
            print(mean_pivot)
        
        # 5. 生成多因子多空权重
        # 策略：做多小市值+高账面市值比+高股息率，做空大市值+低账面市值比+低股息率
        multi_params = {
            'mode': 'freeplay',
            'long_combinations': [
                (0, 2, 1),  # 小市值(0) + 高账面市值比(2) + 高股息率(1)
            ],
            'short_combinations': [
                (2, 0, 0),  # 大市值(2) + 低账面市值比(0) + 低股息率(0)
            ],
        }
        
        weights_multi = get_multi_factor_weight(labeled_pool_multi, multi_params)
        
        # 调试信息
        print("\n多因子权重数据统计：")
        print(f"权重形状: {weights_multi.shape}")
        print(f"非零权重数量: {(weights_multi != 0).sum().sum()}")
        print(f"权重范围: [{weights_multi.min().min():.4f}, {weights_multi.max().max():.4f}]")
        print(f"每行权重和: [{weights_multi.sum(axis=1).min():.4f}, {weights_multi.sum(axis=1).max():.4f}]")
        
        # 添加现金列
        weights_multi['cash'] = 0
        
        # 6. 多因子回测
        bb_multi = BacktestBase(weight=weights_multi, symbol="", amount=1000000)
        
        # 绘制净值曲线
        bb_multi.nav.plot()
        
        # 7. 多因子绩效分析
        analyzer_multi = PortfolioAnalyzer(bb_multi.nav)
        exporter_multi = ReportExporter(analyzer_multi)
        
        print("\n多因子策略绩效报告：")
        exporter_multi.generate_annual_report()
        exporter_multi.generate_custom_report('2024-01-01', '2024-12-31')
        
        print("\n多因子回测完成！")
    else:
        print("提示：将 RUN_MULTI_FACTOR 设为 True 以运行多因子示例")