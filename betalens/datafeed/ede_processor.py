#%%By Janis 251103
"""
EDE格式Excel文件处理工具模块（函数式）
功能：
- 处理特定的EDE格式Excel文件（来自Wind等数据源）
- 识别证券代码、名称和指标列
- 从列名中提取metric和元数据（日期、单位等）
- 转换为数据库标准格式并插入
"""

import pandas as pd
import numpy as np
import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging

from .excel import read_file
from .config import get_ede_config


def _get_default_logger():
    """获取默认logger"""
    logger = logging.getLogger('EDEProcessor')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def extract_date_from_filename(
    filepath: Union[str, Path],
    pattern: Optional[str] = None,
    default_time: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    从文件名中提取日期
    
    Args:
        filepath: 文件路径
        pattern: 日期匹配正则表达式，默认从config.json读取
        default_time: 默认时间，默认从config.json读取
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        日期时间字符串（格式：YYYY-MM-DD HH:MM:SS），如果未找到则返回None
    """
    if logger is None:
        logger = _get_default_logger()
    
    # 从配置文件获取默认参数
    ede_config = get_ede_config()
    date_extraction_config = ede_config.get('date_extraction', {})
    
    if pattern is None:
        pattern = date_extraction_config.get('pattern', r'(\d{8})')
    if default_time is None:
        default_time = date_extraction_config.get('default_time', '15:30:00')
    
    filepath = Path(filepath)
    filename = filepath.stem  # 获取不带扩展名的文件名
    
    # 匹配日期
    match = re.search(pattern, filename)
    if match:
        date_str = match.group(1)
        # 转换为标准日期格式
        try:
            if len(date_str) == 8:
                date_obj = datetime.strptime(date_str, '%Y%m%d')
                datetime_str = f"{date_obj.strftime('%Y-%m-%d')} {default_time}"
                logger.info(f"从文件名提取日期: {datetime_str}")
                return datetime_str
            else:
                logger.warning(f"日期格式不正确: {date_str}")
                return None
        except ValueError as e:
            logger.warning(f"日期解析失败: {date_str}, 错误: {str(e)}")
            return None
    else:
        logger.warning(f"未在文件名中找到日期: {filename}")
        return None


def parse_metric_column(
    column_name: str,
    logger: Optional[logging.Logger] = None
) -> Tuple[str, Dict[str, str]]:
    """
    解析EDE格式的指标列名，提取metric名称和元数据
    
    EDE格式示例：
        "流通A股 [交易日期] 最新 [单位] 股"
        "流通市值 [交易日期] 最新 [单位] 万元"
    
    Args:
        column_name: 列名
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        (metric名称, 元数据字典)
        
    Example:
        >>> parse_metric_column("流通A股 [交易日期] 最新 [单位] 股")
        ('流通A股', {'日期说明': '交易日期', '值类型': '最新', '单位说明': '单位', '单位': '股'})
    """
    if logger is None:
        logger = _get_default_logger()
    
    # 提取方括号内的内容
    bracket_pattern = r'\[([^\]]+)\]'
    brackets = re.findall(bracket_pattern, column_name)
    
    # 移除方括号及其内容，提取核心metric名称和其余部分
    cleaned = re.sub(bracket_pattern, '', column_name).strip()
    
    # 分割清理后的字符串
    parts = [p.strip() for p in cleaned.split() if p.strip()]
    
    if not parts:
        logger.warning(f"无法解析列名: {column_name}")
        return column_name, {}
    
    # 第一部分作为metric名称
    metric_name = parts[0]
    
    # 构建元数据
    metadata = {
        '原始列名': column_name,
    }
    
    # 提取方括号内的信息
    if len(brackets) >= 1:
        metadata['日期说明'] = brackets[0]
    
    if len(parts) >= 2:
        metadata['值类型'] = parts[1]
    
    if len(brackets) >= 2:
        metadata['单位说明'] = brackets[1]
    
    if len(parts) >= 3:
        metadata['单位'] = parts[2]
    
    logger.info(f"解析列名: '{column_name}' -> metric='{metric_name}', metadata={metadata}")
    
    return metric_name, metadata


def extract_date_from_metric_metadata(
    metadata: Dict[str, str],
    column_name: str,
    default_time: Optional[str] = None,
    logger: Optional[logging.Logger] = None
) -> Optional[str]:
    """
    从metric元数据中提取日期
    
    在某些情况下，列名中可能包含具体日期，如：
        "流通A股 [20251103] 最新 [单位] 股"
    
    Args:
        metadata: metric元数据字典
        column_name: 原始列名
        default_time: 默认时间，默认从config.json读取
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        日期时间字符串，如果未找到则返回None
    """
    if logger is None:
        logger = _get_default_logger()
    
    # 从配置文件获取默认时间
    if default_time is None:
        ede_config = get_ede_config()
        date_extraction_config = ede_config.get('date_extraction', {})
        default_time = date_extraction_config.get('default_time', '15:30:00')
    
    # 尝试从日期说明中提取日期
    if '日期说明' in metadata:
        date_str = metadata['日期说明']
        # 尝试匹配8位数字日期
        match = re.search(r'(\d{8})', date_str)
        if match:
            date_digits = match.group(1)
            try:
                date_obj = datetime.strptime(date_digits, '%Y%m%d')
                datetime_str = f"{date_obj.strftime('%Y-%m-%d')} {default_time}"
                logger.info(f"从列名元数据提取日期: {datetime_str}")
                return datetime_str
            except ValueError:
                pass
    
    # 尝试直接从原始列名中提取日期
    match = re.search(r'\[(\d{8})\]', column_name)
    if match:
        date_digits = match.group(1)
        try:
            date_obj = datetime.strptime(date_digits, '%Y%m%d')
            datetime_str = f"{date_obj.strftime('%Y-%m-%d')} {default_time}"
            logger.info(f"从列名提取日期: {datetime_str}")
            return datetime_str
        except ValueError:
            pass
    
    logger.debug(f"未从元数据中找到日期: {metadata}")
    return None


def clean_ede_dataframe(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    清理EDE格式的DataFrame
    
    操作：
    1. 删除完全空白的行
    2. 删除包含"数据来源"等无关字符串的行
    3. 删除完全空白的列
    
    Args:
        df: 原始DataFrame
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        清理后的DataFrame
    """
    if logger is None:
        logger = _get_default_logger()
    
    original_shape = df.shape
    
    # 1. 删除完全空白的行
    df = df.dropna(how='all')
    
    # 2. 删除包含特定字符串的行
    # 从配置文件获取要删除的关键词
    ede_config = get_ede_config()
    data_cleaning_config = ede_config.get('data_cleaning', {})
    keywords_to_remove = data_cleaning_config.get('keywords_to_remove', [
        '数据来源', 'Wind', '来源:', '注:', '说明:', 
        'Source:', 'Note:', 'Remark:'
    ])
    
    for keyword in keywords_to_remove:
        # 检查每一列
        mask = pd.Series([False] * len(df), index=df.index)
        for col in df.columns:
            if df[col].dtype == 'object':  # 只检查字符串列
                mask = mask | df[col].astype(str).str.contains(keyword, case=False, na=False)
        
        if mask.any():
            removed_count = mask.sum()
            logger.info(f"删除包含'{keyword}'的行: {removed_count}行")
            df = df[~mask]
    
    # 3. 删除完全空白的列
    df = df.dropna(axis=1, how='all')
    
    # 4. 重置索引
    df = df.reset_index(drop=True)
    
    cleaned_shape = df.shape
    logger.info(f"数据清理完成: {original_shape} -> {cleaned_shape}")
    
    return df


def identify_code_name_columns(
    df: pd.DataFrame,
    code_column_names: Optional[List[str]] = None,
    name_column_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[str], Optional[str]]:
    """
    识别DataFrame中的代码列和名称列
    
    Args:
        df: DataFrame
        code_column_names: 可能的代码列名列表，默认从config.json读取
        name_column_names: 可能的名称列名列表，默认从config.json读取
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        (代码列名, 名称列名)，如果未找到则返回None
    """
    if logger is None:
        logger = _get_default_logger()
    
    # 从配置文件获取默认列名列表
    ede_config = get_ede_config()
    column_names_config = ede_config.get('column_names', {})
    
    if code_column_names is None:
        code_column_names = column_names_config.get('code_columns', [
            '证券代码', 'code', 'windcode', '代码', 'Code', 'WindCode'
        ])
    
    if name_column_names is None:
        name_column_names = column_names_config.get('name_columns', [
            '证券简称', 'name', 'sec_name', '简称', '名称', 'Name', 'SecName'
        ])
    
    # 查找代码列
    code_col = None
    for col_name in code_column_names:
        if col_name in df.columns:
            code_col = col_name
            logger.info(f"识别代码列: {code_col}")
            break
    
    # 如果没找到，尝试使用第一列
    if code_col is None and len(df.columns) > 0:
        code_col = df.columns[0]
        logger.warning(f"未找到标准代码列，使用第一列作为代码列: {code_col}")
    
    # 查找名称列
    name_col = None
    for col_name in name_column_names:
        if col_name in df.columns:
            name_col = col_name
            logger.info(f"识别名称列: {name_col}")
            break
    
    # 如果没找到，尝试使用第二列
    if name_col is None and len(df.columns) > 1:
        name_col = df.columns[1]
        logger.warning(f"未找到标准名称列，使用第二列作为名称列: {name_col}")
    
    return code_col, name_col


def process_ede_file(
    filepath: Union[str, Path],
    date_from: str = 'filename',  # 'filename' 或 'metric'
    default_datetime: Optional[str] = None,
    code_column_names: Optional[List[str]] = None,
    name_column_names: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[pd.DataFrame], List[Dict]]:
    """
    处理EDE格式的Excel文件并转换为数据库格式
    
    EDE格式特征：
        - 第一列：证券代码
        - 第二列：证券简称
        - 第三列及之后：指标列，格式为"指标名 [元数据] 值类型 [元数据] 单位"
    
    Args:
        filepath: 文件路径
        date_from: 日期来源，'filename'（从文件名提取）或'metric'（从列名提取）
        default_datetime: 默认日期时间（当无法从文件名或列名提取时使用）
        code_column_names: 可能的代码列名列表
        name_column_names: 可能的名称列名列表
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        (处理后的DataFrame, 错误列表)
        
    处理后的DataFrame格式：
        - code: 证券代码
        - name: 证券简称
        - metric: 指标名称
        - value: 数值
        - datetime: 日期时间
        - note: 元数据（JSON格式）
    """
    if logger is None:
        logger = _get_default_logger()
    
    errors = []
    
    try:
        # 1. 读取文件
        logger.info(f"开始处理EDE文件: {filepath}")
        df = read_file(filepath, logger=logger)
        
        if df.empty:
            errors.append({'type': 'empty_file', 'message': '文件为空'})
            return None, errors
        
        # 2. 清理数据
        df = clean_ede_dataframe(df, logger=logger)
        
        if df.empty:
            errors.append({'type': 'empty_after_clean', 'message': '清理后数据为空'})
            return None, errors
        
        # 3. 识别代码列和名称列
        code_col, name_col = identify_code_name_columns(
            df, code_column_names, name_column_names, logger=logger
        )
        
        if code_col is None:
            errors.append({'type': 'missing_code_column', 'message': '未找到代码列'})
            return None, errors
        
        # 4. 确定日期
        datetime_value = None
        
        if date_from == 'filename':
            # 从文件名提取日期
            datetime_value = extract_date_from_filename(filepath, logger=logger)
        
        if datetime_value is None and default_datetime:
            datetime_value = default_datetime
            logger.info(f"使用默认日期: {datetime_value}")
        
        # 5. 识别指标列（除了代码列和名称列之外的所有列）
        key_columns = [code_col]
        if name_col:
            key_columns.append(name_col)
        
        metric_columns = [col for col in df.columns if col not in key_columns]
        
        if not metric_columns:
            errors.append({'type': 'no_metric_columns', 'message': '未找到指标列'})
            return None, errors
        
        logger.info(f"识别到 {len(metric_columns)} 个指标列")
        
        # 6. 转换为长格式
        result_rows = []
        
        for metric_col in metric_columns:
            # 解析指标列名
            metric_name, metadata = parse_metric_column(metric_col, logger=logger)
            
            # 如果需要从metric提取日期
            metric_datetime = None
            if date_from == 'metric' or datetime_value is None:
                metric_datetime = extract_date_from_metric_metadata(
                    metadata, metric_col, logger=logger
                )
            
            # 确定最终使用的日期
            final_datetime = metric_datetime if metric_datetime else datetime_value
            
            if final_datetime is None:
                logger.warning(f"无法确定日期，跳过指标列: {metric_col}")
                errors.append({
                    'type': 'missing_datetime',
                    'column': metric_col,
                    'message': '无法确定日期'
                })
                continue
            
            # 遍历每一行数据
            for idx, row in df.iterrows():
                code = row[code_col]
                name = row[name_col] if name_col else None
                value = row[metric_col]
                
                # 跳过空值
                if pd.isna(value) or value == '':
                    continue
                
                # 清理数值（去除千位分隔符等）
                if isinstance(value, str):
                    value = value.replace(',', '').strip()
                    try:
                        value = float(value)
                    except ValueError:
                        logger.warning(f"无法转换为数值: {value}, 代码: {code}, 指标: {metric_name}")
                        continue
                
                # 构建行数据
                row_data = {
                    'code': code,
                    'name': name,
                    'metric': metric_name,
                    'value': value,
                    'datetime': final_datetime,
                    'note': json.dumps(metadata, ensure_ascii=False)
                }
                
                result_rows.append(row_data)
        
        if not result_rows:
            errors.append({'type': 'no_valid_data', 'message': '没有有效数据'})
            return None, errors
        
        # 7. 创建结果DataFrame
        result_df = pd.DataFrame(result_rows)
        
        # 8. 数据类型转换
        result_df['datetime'] = pd.to_datetime(result_df['datetime'])
        result_df['value'] = result_df['value'].astype(float)
        
        # 9. 排序
        result_df = result_df.sort_values(by=['datetime', 'code', 'metric']).reset_index(drop=True)
        
        logger.info(f"EDE文件处理完成: {len(result_rows)} 行数据")
        
        return result_df, errors
        
    except Exception as e:
        logger.error(f"处理EDE文件失败: {str(e)}")
        errors.append({
            'type': 'processing_error',
            'message': str(e)
        })
        return None, errors

