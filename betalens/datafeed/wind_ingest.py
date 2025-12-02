#%%By Janis 250226
"""
Wind数据抓取模块（函数式）
功能：
- 从WindPy获取日行情数据
- 转换为数据库标准格式
- 支持多种资产类型（股票、指数、基金、债券）
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import logging

from .excel import apply_time_alignment
from .config import get_wind_config


def _get_default_logger():
    """获取默认logger"""
    logger = logging.getLogger('WindIngest')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def fetch_daily_market(
    codes: List[str],
    start_date: str,
    end_date: str,
    fields: Optional[List[str]] = None,
    asset_type: str = 'stock',
    apply_time_stamps: bool = True,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    从Wind获取日行情数据并转换为数据库格式
    
    Args:
        codes: 代码列表，如['000001.SZ', '000002.SZ']
        start_date: 开始日期，格式'YYYY-MM-DD'
        end_date: 结束日期，格式'YYYY-MM-DD'
        fields: 字段列表，如['open', 'high', 'low', 'close', 'volume']
               默认为None，使用预设字段
        asset_type: 资产类型，'stock'(股票), 'index'(指数), 'fund'(基金), 'bond'(债券)
        apply_time_stamps: 是否应用交易时间戳（开盘价09:30，其他15:00）
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        DataFrame，格式为：
            datetime | code | name | metric | value
            
    Example:
        >>> df = fetch_daily_market(
        ...     codes=['000001.SZ'],
        ...     start_date='2024-01-01',
        ...     end_date='2024-01-31'
        ... )
        >>> df.head()
           datetime            code   name      metric  value
        0  2024-01-02 09:30:01  000001.SZ  平安银行  开盘价(元)   10.50
        1  2024-01-02 15:00:01  000001.SZ  平安银行  收盘价(元)   10.55
    """
    if logger is None:
        logger = _get_default_logger()
    
    try:
        from WindPy import w
    except ImportError:
        logger.error("无法导入WindPy，请确保已安装Wind金融终端")
        raise ImportError("需要安装WindPy: pip install WindPy")
    
    # 启动Wind
    w.start()
    logger.info("Wind连接已启动")
    
    # 从配置文件获取资产类型的默认字段
    wind_config = get_wind_config()
    asset_fields_config = wind_config.get('asset_fields', {})
    
    # 根据资产类型设置默认字段
    if fields is None:
        if asset_type in asset_fields_config:
            type_config = asset_fields_config[asset_type]
            fields = type_config.get('fields', ['open', 'high', 'low', 'close', 'volume'])
            field_names = type_config.get('field_names', fields)
        else:
            logger.warning(f"未知资产类型 '{asset_type}'，使用默认股票字段")
            default_stock = asset_fields_config.get('stock', {})
            fields = default_stock.get('fields', ['open', 'high', 'low', 'close', 'volume'])
            field_names = default_stock.get('field_names', fields)
    else:
        # 用户自定义字段，字段名与Wind字段名相同
        field_names = fields
    
    # 构建字段映射
    field_mapping = dict(zip(fields, field_names))
    
    logger.info(f"准备获取数据: 代码数={len(codes)}, 日期范围={start_date}~{end_date}, 字段={fields}")
    
    # 获取数据
    all_data = []
    
    for code in codes:
        try:
            # 调用Wind API
            result = w.wsd(code, fields, start_date, end_date, "")
            
            # 检查错误
            if result.ErrorCode != 0:
                logger.error(f"获取 {code} 数据失败: {result.Data}")
                continue
            
            # 获取代码名称
            name_result = w.wss(code, "sec_name")
            if name_result.ErrorCode == 0 and name_result.Data and name_result.Data[0]:
                code_name = name_result.Data[0][0]
            else:
                code_name = code
            
            # 转换为DataFrame
            dates = result.Times
            data_dict = {'日期': dates, '代码': code, '简称': code_name}
            
            for i, field in enumerate(fields):
                data_dict[field_names[i]] = result.Data[i]
            
            df_code = pd.DataFrame(data_dict)
            all_data.append(df_code)
            
            logger.info(f"成功获取 {code} ({code_name}) 数据: {len(df_code)} 行")
            
        except Exception as e:
            logger.error(f"获取 {code} 数据时发生异常: {str(e)}")
            continue
    
    if not all_data:
        logger.warning("未获取到任何数据")
        return pd.DataFrame(columns=['datetime', 'code', 'name', 'metric', 'value'])
    
    # 合并所有数据
    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"合并完成，共 {len(df_all)} 行")
    
    # 转换为数据库格式（长格式）
    # 值列 = 除了日期、代码、简称之外的所有列
    value_columns = [col for col in df_all.columns if col not in ['日期', '代码', '简称']]
    
    df_melted = pd.melt(
        df_all,
        id_vars=['日期', '代码', '简称'],
        value_vars=value_columns,
        var_name='metric',
        value_name='value'
    )
    
    # 重命名列
    df_melted.rename(columns={'代码': 'code', '简称': 'name'}, inplace=True)
    
    # 应用时间戳（开盘价09:30，其他15:00）
    if apply_time_stamps:
        df_melted = apply_time_alignment(
            df_melted,
            date_column='日期',
            metric_column='metric',
            inplace=True,
            logger=logger
        )
    else:
        # 不应用时间戳，统一使用15:00:01
        df_melted['日期'] = df_melted['日期'].astype(str) + " 15:00:01"
        df_melted['日期'] = pd.to_datetime(df_melted['日期'])
    
    # 重命名日期列为datetime
    df_melted.rename(columns={'日期': 'datetime'}, inplace=True)
    
    # 调整列顺序
    df_melted = df_melted[['datetime', 'code', 'name', 'metric', 'value']]
    
    # 删除空值行
    df_melted.dropna(subset=['value'], inplace=True)
    
    logger.info(f"数据转换完成: {len(df_melted)} 行，格式: datetime|code|name|metric|value")
    
    return df_melted


def fetch_daily_index(
    codes: List[str],
    start_date: str,
    end_date: str,
    fields: Optional[List[str]] = None,
    apply_time_stamps: bool = True,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    获取指数日行情数据
    
    这是fetch_daily_market的便捷封装，asset_type='index'
    
    Args:
        codes: 指数代码列表，如['000001.SH', '399001.SZ']
        start_date: 开始日期
        end_date: 结束日期
        fields: 字段列表，默认为None
        apply_time_stamps: 是否应用交易时间戳
        logger: 日志记录器
        
    Returns:
        DataFrame
    """
    return fetch_daily_market(
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        asset_type='index',
        apply_time_stamps=apply_time_stamps,
        logger=logger
    )


def fetch_daily_fund(
    codes: List[str],
    start_date: str,
    end_date: str,
    fields: Optional[List[str]] = None,
    apply_time_stamps: bool = True,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    获取基金日行情数据
    
    这是fetch_daily_market的便捷封装，asset_type='fund'
    
    Args:
        codes: 基金代码列表
        start_date: 开始日期
        end_date: 结束日期
        fields: 字段列表，默认为None
        apply_time_stamps: 是否应用交易时间戳
        logger: 日志记录器
        
    Returns:
        DataFrame
    """
    return fetch_daily_market(
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        asset_type='fund',
        apply_time_stamps=apply_time_stamps,
        logger=logger
    )


def fetch_daily_bond(
    codes: List[str],
    start_date: str,
    end_date: str,
    fields: Optional[List[str]] = None,
    apply_time_stamps: bool = True,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    获取债券日行情数据
    
    这是fetch_daily_market的便捷封装，asset_type='bond'
    
    Args:
        codes: 债券代码列表
        start_date: 开始日期
        end_date: 结束日期
        fields: 字段列表，默认为None
        apply_time_stamps: 是否应用交易时间戳
        logger: 日志记录器
        
    Returns:
        DataFrame
    """
    return fetch_daily_market(
        codes=codes,
        start_date=start_date,
        end_date=end_date,
        fields=fields,
        asset_type='bond',
        apply_time_stamps=apply_time_stamps,
        logger=logger
    )

