#%%By Janis 250226
"""
数据验证和异常检查工具模块（函数式）
功能：
- 检查空值、NaN、None
- 检查日期列的格式、重复、排序、频率
- 提供多种修复策略（替换、填充、删除、抛出错误）
"""

import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict, Callable, Tuple, Any
from datetime import datetime
import logging
from enum import Enum


class FillStrategy(Enum):
    """填充策略枚举"""
    RAISE_ERROR = "raise_error"  # 抛出错误并停止
    DROP = "drop"  # 删除含有问题的行
    FILL_FORWARD = "ffill"  # 向前填充
    FILL_BACKWARD = "bfill"  # 向后填充
    FILL_VALUE = "fill_value"  # 用指定值填充
    FILL_MEAN = "mean"  # 用均值填充
    FILL_MEDIAN = "median"  # 用中位数填充
    FILL_MODE = "mode"  # 用众数填充
    INTERPOLATE = "interpolate"  # 插值填充


def _get_default_logger():
    """获取默认logger"""
    logger = logging.getLogger('DataValidator')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def check_null_values(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    check_types: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    检查空值、NaN、None
    
    Args:
        df: 待检查的DataFrame
        columns: 要检查的列名列表，None表示检查所有列
        check_types: 检查类型列表，可选['null', 'nan', 'none', 'empty_string']
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        检查结果字典
    """
    if logger is None:
        logger = _get_default_logger()
    
    if columns is None:
        columns = df.columns.tolist()
    
    if check_types is None:
        check_types = ['null', 'nan', 'none', 'empty_string']
    
    results = {}
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"列 '{col}' 不存在于DataFrame中")
            continue
        
        col_results = {
            'total_rows': len(df),
            'issues': {}
        }
        
        # 检查各种类型的空值
        if 'null' in check_types or 'nan' in check_types:
            null_mask = df[col].isnull()
            null_count = null_mask.sum()
            if null_count > 0:
                col_results['issues']['null/nan'] = {
                    'count': int(null_count),
                    'percentage': float(null_count / len(df) * 100),
                    'indices': df[null_mask].index.tolist()[:10]  # 只保留前10个
                }
        
        if 'none' in check_types:
            none_mask = df[col] == None
            none_count = none_mask.sum()
            if none_count > 0:
                col_results['issues']['none'] = {
                    'count': int(none_count),
                    'percentage': float(none_count / len(df) * 100),
                    'indices': df[none_mask].index.tolist()[:10]
                }
        
        if 'empty_string' in check_types:
            if df[col].dtype == object:
                empty_mask = df[col].astype(str).str.strip() == ''
                empty_count = empty_mask.sum()
                if empty_count > 0:
                    col_results['issues']['empty_string'] = {
                        'count': int(empty_count),
                        'percentage': float(empty_count / len(df) * 100),
                        'indices': df[empty_mask].index.tolist()[:10]
                    }
        
        if col_results['issues']:
            results[col] = col_results
            logger.warning(f"列 '{col}' 发现空值问题: {col_results['issues']}")
    
    return results


def check_datetime_column(
    df: pd.DataFrame,
    date_column: str,
    expected_freq: Optional[str] = None,
    check_sorted: bool = True,
    check_duplicates: bool = True,
    check_format: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    检查日期列的各种问题
    
    Args:
        df: DataFrame
        date_column: 日期列名
        expected_freq: 期望的频率，如'D'(日), 'W'(周), 'M'(月), 'Q'(季度), 'Y'(年)
        check_sorted: 是否检查排序
        check_duplicates: 是否检查重复
        check_format: 是否检查格式
        logger: 日志记录器，如果为None则使用默认logger
            
    Returns:
        检查结果字典
    """
    if logger is None:
        logger = _get_default_logger()
    
    results = {
        'column': date_column,
        'total_rows': len(df),
        'issues': {}
    }
    
    if date_column not in df.columns:
        results['issues']['column_not_found'] = f"列 '{date_column}' 不存在"
        logger.error(results['issues']['column_not_found'])
        return results
    
    # 1. 检查格式和类型
    if check_format:
        try:
            # 尝试转换为datetime
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                date_series = pd.to_datetime(df[date_column], errors='coerce')
                
                # 检查转换失败的值
                invalid_mask = date_series.isnull() & df[date_column].notnull()
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    results['issues']['invalid_format'] = {
                        'count': int(invalid_count),
                        'percentage': float(invalid_count / len(df) * 100),
                        'examples': df[invalid_mask][date_column].head(5).tolist()
                    }
                    logger.warning(f"列 '{date_column}' 有 {invalid_count} 个无效日期格式")
            else:
                date_series = df[date_column]
        except Exception as e:
            results['issues']['format_error'] = str(e)
            logger.error(f"日期格式检查失败: {e}")
            return results
    else:
        date_series = df[date_column]
    
    # 移除NaT值进行后续检查
    valid_dates = date_series.dropna()
    
    if len(valid_dates) == 0:
        results['issues']['all_null'] = "所有日期值都为空"
        logger.error(results['issues']['all_null'])
        return results
    
    # 2. 检查重复
    if check_duplicates:
        duplicates = valid_dates[valid_dates.duplicated(keep=False)]
        if len(duplicates) > 0:
            dup_values = duplicates.value_counts()
            results['issues']['duplicates'] = {
                'count': int(len(duplicates)),
                'unique_duplicate_dates': int(len(dup_values)),
                'examples': dup_values.head(5).to_dict()
            }
            logger.warning(f"列 '{date_column}' 有 {len(duplicates)} 个重复日期")
    
    # 3. 检查排序
    if check_sorted:
        is_sorted_asc = valid_dates.is_monotonic_increasing
        is_sorted_desc = valid_dates.is_monotonic_decreasing
        
        if not (is_sorted_asc or is_sorted_desc):
            # 找出乱序的位置
            diff = valid_dates.diff()
            unsorted_indices = diff[diff < pd.Timedelta(0)].index.tolist()
            
            results['issues']['unsorted'] = {
                'is_sorted': False,
                'unsorted_positions': unsorted_indices[:10],  # 只保留前10个
                'count': len(unsorted_indices)
            }
            logger.warning(f"列 '{date_column}' 未正确排序，发现 {len(unsorted_indices)} 处乱序")
        else:
            results['sort_order'] = 'ascending' if is_sorted_asc else 'descending'
    
    # 4. 检查频率
    if expected_freq is not None and len(valid_dates) > 1:
        try:
            # 推断实际频率
            inferred_freq = pd.infer_freq(valid_dates)
            
            if inferred_freq is None:
                # 频率不一致，计算时间间隔
                time_diffs = valid_dates.diff().dropna()
                
                results['issues']['irregular_frequency'] = {
                    'expected': expected_freq,
                    'inferred': None,
                    'min_interval': str(time_diffs.min()),
                    'max_interval': str(time_diffs.max()),
                    'mean_interval': str(time_diffs.mean()),
                    'unique_intervals': int(time_diffs.nunique())
                }
                logger.warning(f"列 '{date_column}' 频率不规则，期望 {expected_freq}")
            elif inferred_freq != expected_freq:
                results['issues']['frequency_mismatch'] = {
                    'expected': expected_freq,
                    'inferred': inferred_freq
                }
                logger.warning(f"列 '{date_column}' 频率不匹配: 期望 {expected_freq}, 实际 {inferred_freq}")
            else:
                results['frequency'] = inferred_freq
                
        except Exception as e:
            results['issues']['frequency_check_error'] = str(e)
            logger.error(f"频率检查失败: {e}")
    
    # 5. 统计信息
    results['stats'] = {
        'min_date': str(valid_dates.min()),
        'max_date': str(valid_dates.max()),
        'date_range': str(valid_dates.max() - valid_dates.min()),
        'unique_dates': int(valid_dates.nunique()),
        'null_count': int(date_series.isnull().sum())
    }
    
    return results


def fix_null_values(
    df: pd.DataFrame,
    strategy: Union[FillStrategy, str],
    columns: Optional[List[str]] = None,
    fill_value: Any = None,
    inplace: bool = False,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    修复空值
    
    Args:
        df: DataFrame
        strategy: 填充策略
        columns: 要处理的列，None表示所有列
        fill_value: 当strategy为FILL_VALUE时使用的填充值
        inplace: 是否原地修改
        logger: 日志记录器，如果为None则使用默认logger
            
    Returns:
        修复后的DataFrame
    """
    if logger is None:
        logger = _get_default_logger()
    
    if not inplace:
        df = df.copy()
    
    if isinstance(strategy, str):
        try:
            strategy = FillStrategy(strategy)
        except ValueError:
            logger.error(f"无效的填充策略: {strategy}")
            raise
    
    if columns is None:
        columns = df.columns.tolist()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        null_count = df[col].isnull().sum()
        if null_count == 0:
            continue
        
        logger.info(f"处理列 '{col}' 的 {null_count} 个空值，策略: {strategy.value}")
        
        try:
            if strategy == FillStrategy.RAISE_ERROR:
                raise ValueError(f"列 '{col}' 包含 {null_count} 个空值")
            
            elif strategy == FillStrategy.DROP:
                df.dropna(subset=[col], inplace=True)
            
            elif strategy == FillStrategy.FILL_FORWARD:
                df[col] = df[col].ffill()
            
            elif strategy == FillStrategy.FILL_BACKWARD:
                df[col] = df[col].bfill()
            
            elif strategy == FillStrategy.FILL_VALUE:
                if fill_value is None:
                    logger.warning(f"未指定fill_value，使用0")
                    fill_value = 0
                df[col] = df[col].fillna(fill_value)
            
            elif strategy == FillStrategy.FILL_MEAN:
                if pd.api.types.is_numeric_dtype(df[col]):
                    mean_val = df[col].mean()
                    df[col] = df[col].fillna(mean_val)
                else:
                    logger.warning(f"列 '{col}' 不是数值类型，无法使用均值填充")
            
            elif strategy == FillStrategy.FILL_MEDIAN:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
                else:
                    logger.warning(f"列 '{col}' 不是数值类型，无法使用中位数填充")
            
            elif strategy == FillStrategy.FILL_MODE:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val[0])
                else:
                    logger.warning(f"列 '{col}' 无法计算众数")
            
            elif strategy == FillStrategy.INTERPOLATE:
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].interpolate()
                else:
                    logger.warning(f"列 '{col}' 不是数值类型，无法使用插值填充")
        
        except Exception as e:
            logger.error(f"修复列 '{col}' 失败: {e}")
            raise
    
    return df


def drop_duplicates_strict(
    df: pd.DataFrame,
    subset: Optional[List[str]] = None,
    keep: str = 'first',
    verify_all_fields: bool = True,
    ignore_cols: Optional[List[str]] = None,
    inplace: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    严格去重：确保只有完全相同的行才会被删除
    
    Args:
        df: DataFrame
        subset: 用于判断重复的列，None表示所有列
        keep: 'first', 'last', False（删除所有重复）
        verify_all_fields: 是否验证subset外的其他字段也相同
        ignore_cols: 验证时忽略的列（如索引、时间戳等）
        inplace: 是否原地修改
        logger: 日志记录器，如果为None则使用默认logger
            
    Returns:
        (修复后的DataFrame, 去重报告)
    """
    if logger is None:
        logger = _get_default_logger()
    
    if not inplace:
        df = df.copy()
    
    report = {
        'total_rows': len(df),
        'duplicates_found': 0,
        'duplicates_removed': 0,
        'inconsistent_duplicates': [],
        'removed_indices': []
    }
    
    if subset is None:
        subset = df.columns.tolist()
    
    # 检查subset列是否存在
    missing_cols = [col for col in subset if col not in df.columns]
    if missing_cols:
        logger.warning(f"subset中的列不存在: {missing_cols}")
        subset = [col for col in subset if col in df.columns]
    
    if not subset:
        logger.warning("没有有效的subset列，跳过去重")
        return df, report
    
    # 找出重复行
    duplicated_mask = df.duplicated(subset=subset, keep=False)
    num_duplicated = duplicated_mask.sum()
    report['duplicates_found'] = int(num_duplicated)
    
    if num_duplicated == 0:
        logger.info("未发现重复数据")
        return df, report
    
    logger.info(f"发现 {num_duplicated} 行在 {subset} 上有重复")
    
    # 如果需要验证所有字段
    if verify_all_fields and num_duplicated > 0:
        # 准备验证列（排除ignore_cols）
        verify_cols = df.columns.tolist()
        if ignore_cols:
            verify_cols = [col for col in verify_cols if col not in ignore_cols]
        
        # 找出subset重复但其他字段不一致的记录
        duplicated_groups = df[duplicated_mask].groupby(subset)
        
        inconsistent_groups = []
        for name, group in duplicated_groups:
            if len(group) > 1:
                # 检查组内是否所有行完全相同（在verify_cols上）
                first_row = group.iloc[0]
                for idx, row in group.iloc[1:].iterrows():
                    # 比较所有verify_cols
                    differences = []
                    for col in verify_cols:
                        if col not in subset:  # subset已经相同，不需要再比较
                            val1 = first_row[col]
                            val2 = row[col]
                            # 处理NaN的比较
                            if pd.isna(val1) and pd.isna(val2):
                                continue
                            if val1 != val2:
                                differences.append({
                                    'column': col,
                                    'value1': val1,
                                    'value2': val2
                                })
                    
                    if differences:
                        inconsistent_groups.append({
                            'subset_values': dict(zip(subset, name if isinstance(name, tuple) else [name])),
                            'indices': [group.index[0], idx],
                            'differences': differences
                        })
        
        report['inconsistent_duplicates'] = inconsistent_groups
        
        if inconsistent_groups:
            logger.warning(
                f"发现 {len(inconsistent_groups)} 组重复：在 {subset} 上相同但其他字段不同，将保留这些记录"
            )
            for i, group_info in enumerate(inconsistent_groups[:3]):  # 只显示前3个
                logger.warning(
                    f"  示例 {i+1}: {group_info['subset_values']} "
                    f"有 {len(group_info['differences'])} 个字段不同"
                )
            if len(inconsistent_groups) > 3:
                logger.warning(f"  ... 还有 {len(inconsistent_groups)-3} 组类似情况")
            
            # 从duplicated_mask中移除这些不一致的重复
            inconsistent_indices = set()
            for group_info in inconsistent_groups:
                inconsistent_indices.update(group_info['indices'])
            
            # 只删除完全一致的重复
            for idx in inconsistent_indices:
                duplicated_mask.loc[idx] = False
    
    # 执行去重
    original_len = len(df)
    
    if keep == 'first':
        drop_mask = df.duplicated(subset=subset, keep='first')
    elif keep == 'last':
        drop_mask = df.duplicated(subset=subset, keep='last')
    elif keep == False:
        drop_mask = df.duplicated(subset=subset, keep=False)
    else:
        logger.warning(f"未知的keep参数: {keep}，使用'first'")
        drop_mask = df.duplicated(subset=subset, keep='first')
    
    # 只删除完全一致的重复（已经通过verify_all_fields过滤）
    if verify_all_fields and report['inconsistent_duplicates']:
        # 重新计算drop_mask，排除不一致的
        inconsistent_indices = set()
        for group_info in report['inconsistent_duplicates']:
            inconsistent_indices.update(group_info['indices'])
        for idx in inconsistent_indices:
            drop_mask.loc[idx] = False
    
    report['removed_indices'] = df[drop_mask].index.tolist()
    
    df.drop(df[drop_mask].index, inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    removed = original_len - len(df)
    report['duplicates_removed'] = removed
    
    if removed > 0:
        logger.info(f"删除了 {removed} 个完全相同的重复行")
    
    return df, report


def fix_datetime_column(
    df: pd.DataFrame,
    date_column: str,
    fix_format: bool = True,
    fix_duplicates: Optional[str] = 'keep_first',
    fix_sort: bool = True,
    sort_order: str = 'ascending',
    dedupe_subset: Optional[List[str]] = None,
    verify_all_fields: bool = True,
    inplace: bool = False,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    修复日期列的问题
    
    Args:
        df: DataFrame
        date_column: 日期列名
        fix_format: 是否修复格式（转换为datetime）
        fix_duplicates: 如何处理重复，None表示不处理
        fix_sort: 是否排序
        sort_order: 排序顺序，'ascending'或'descending'
        dedupe_subset: 去重时使用的列组合，None则使用[date_column]
                      推荐: ['code', 'metric', date_column] 避免误删不同metric的数据
        verify_all_fields: 是否验证subset外的其他字段也相同（严格模式）
        inplace: 是否原地修改
        logger: 日志记录器，如果为None则使用默认logger
            
    Returns:
        修复后的DataFrame
    """
    if logger is None:
        logger = _get_default_logger()
    
    if not inplace:
        df = df.copy()
    
    if date_column not in df.columns:
        logger.error(f"列 '{date_column}' 不存在")
        return df
    
    # 1. 修复格式
    if fix_format:
        try:
            if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
                original_count = len(df)
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
                
                # 统计转换失败的数量
                invalid_count = df[date_column].isnull().sum()
                if invalid_count > 0:
                    logger.warning(f"日期转换失败 {invalid_count} 个值，已设置为NaT")
                else:
                    logger.info(f"成功将列 '{date_column}' 转换为datetime类型")
        except Exception as e:
            logger.error(f"日期格式修复失败: {e}")
    
    # 2. 处理重复（使用严格去重）
    if fix_duplicates is not None:
        # 确定去重列
        if dedupe_subset is None:
            dedupe_subset = [date_column]
        else:
            # 确保date_column在subset中
            if date_column not in dedupe_subset:
                dedupe_subset = dedupe_subset + [date_column]
        
        logger.info(f"使用列组合进行去重: {dedupe_subset}")
        
        # 使用严格去重
        df, report = drop_duplicates_strict(
            df=df,
            subset=dedupe_subset,
            keep=fix_duplicates if fix_duplicates in ['first', 'last', False] else 'first',
            verify_all_fields=verify_all_fields,
            logger=logger,
            inplace=True
        )
        
        # 记录不一致的重复
        if report['inconsistent_duplicates']:
            logger.warning(
                f"发现 {len(report['inconsistent_duplicates'])} 组数据：" 
                f"在 {dedupe_subset} 上相同但其他字段不同，已保留（未删除）"
            )
    
    # 3. 排序
    if fix_sort:
        ascending = (sort_order == 'ascending')
        df.sort_values(by=date_column, ascending=ascending, inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info(f"已按 '{date_column}' {sort_order} 排序")
    
    return df


def validate_and_fix(
    df: pd.DataFrame,
    validations: Dict[str, Dict],
    inplace: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    综合验证和修复
    
    Args:
        df: DataFrame
        validations: 验证配置字典，格式如：
            {
                'null_check': {
                    'columns': ['col1', 'col2'],
                    'fix_strategy': 'ffill'
                },
                'datetime_check': {
                    'column': 'date',
                    'expected_freq': 'D',
                    'fix_format': True,
                    'fix_duplicates': 'keep_first',
                    'fix_sort': True
                }
            }
        inplace: 是否原地修改
        logger: 日志记录器，如果为None则使用默认logger
            
    Returns:
        (修复后的DataFrame, 验证报告)
    """
    if logger is None:
        logger = _get_default_logger()
    
    if not inplace:
        df = df.copy()
    
    report = {}
    
    # 1. 空值检查和修复
    if 'null_check' in validations:
        config = validations['null_check']
        columns = config.get('columns', None)
        
        # 检查
        null_results = check_null_values(df, columns=columns, logger=logger)
        report['null_check'] = null_results
        
        # 修复
        if 'fix_strategy' in config and null_results:
            df = fix_null_values(
                df,
                strategy=config['fix_strategy'],
                columns=columns,
                fill_value=config.get('fill_value', None),
                logger=logger,
                inplace=True
            )
            report['null_fix'] = f"已使用策略 '{config['fix_strategy']}' 修复空值"
    
    # 2. 日期列检查和修复
    if 'datetime_check' in validations:
        config = validations['datetime_check']
        date_column = config.get('column')
        
        if date_column:
            # 检查
            datetime_results = check_datetime_column(
                df,
                date_column=date_column,
                expected_freq=config.get('expected_freq'),
                check_sorted=config.get('check_sorted', True),
                check_duplicates=config.get('check_duplicates', True),
                check_format=config.get('check_format', True),
                logger=logger
            )
            report['datetime_check'] = datetime_results
            
            # 修复
            if datetime_results.get('issues'):
                df = fix_datetime_column(
                    df,
                    date_column=date_column,
                    fix_format=config.get('fix_format', True),
                    fix_duplicates=config.get('fix_duplicates', 'keep_first'),
                    fix_sort=config.get('fix_sort', True),
                    sort_order=config.get('sort_order', 'ascending'),
                    dedupe_subset=config.get('dedupe_subset', None),
                    verify_all_fields=config.get('verify_all_fields', True),
                    logger=logger,
                    inplace=True
                )
                report['datetime_fix'] = "已修复日期列问题"
    
    return df, report


# 为了向后兼容，保留DataValidator作为函数集合的命名空间
class DataValidator:
    """
    DataValidator - 已弃用，请直接使用模块级函数
    
    此类已弃用，方法不再实现。请使用模块级函数：
    - check_null_values()
    - check_datetime_column()
    - fix_null_values()
    - drop_duplicates_strict()
    - fix_datetime_column()
    - validate_and_fix()
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        import warnings
        warnings.warn(
            "DataValidator类已弃用，请直接使用模块级函数，如 check_null_values(), fix_null_values() 等",
            DeprecationWarning,
            stacklevel=2
        )
    
    def check_null_values(self, df, columns=None, check_types=None):
        raise NotImplementedError(
            "DataValidator.check_null_values() 已弃用。\n"
            "请使用: from datafeed.validation import check_null_values\n"
            "         results = check_null_values(df, columns, check_types)"
        )
    
    def check_datetime_column(self, df, date_column, expected_freq=None, check_sorted=True, check_duplicates=True, check_format=True):
        raise NotImplementedError(
            "DataValidator.check_datetime_column() 已弃用。\n"
            "请使用: from datafeed.validation import check_datetime_column\n"
            "         results = check_datetime_column(df, date_column, expected_freq, check_sorted, check_duplicates, check_format)"
        )
    
    def fix_null_values(self, df, strategy, columns=None, fill_value=None, inplace=False):
        raise NotImplementedError(
            "DataValidator.fix_null_values() 已弃用。\n"
            "请使用: from datafeed.validation import fix_null_values\n"
            "         fixed_df = fix_null_values(df, strategy, columns, fill_value, inplace)"
        )
    
    def drop_duplicates_strict(self, df, subset=None, keep='first', verify_all_fields=True, ignore_cols=None, inplace=False):
        raise NotImplementedError(
            "DataValidator.drop_duplicates_strict() 已弃用。\n"
            "请使用: from datafeed.validation import drop_duplicates_strict\n"
            "         cleaned_df, report = drop_duplicates_strict(df, subset, keep, verify_all_fields, ignore_cols, inplace)"
        )
    
    def fix_datetime_column(self, df, date_column, fix_format=True, fix_duplicates='keep_first', fix_sort=True, sort_order='ascending', dedupe_subset=None, verify_all_fields=True, inplace=False):
        raise NotImplementedError(
            "DataValidator.fix_datetime_column() 已弃用。\n"
            "请使用: from datafeed.validation import fix_datetime_column\n"
            "         fixed_df = fix_datetime_column(df, date_column, fix_format, fix_duplicates, fix_sort, sort_order, dedupe_subset, verify_all_fields, inplace)"
        )
    
    def validate_and_fix(self, df, validations, inplace=False):
        raise NotImplementedError(
            "DataValidator.validate_and_fix() 已弃用。\n"
            "请使用: from datafeed.validation import validate_and_fix\n"
            "         fixed_df, report = validate_and_fix(df, validations, inplace)"
        )
