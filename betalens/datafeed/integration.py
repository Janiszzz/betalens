#%%By Janis 250226
"""
数据库-Excel交互功能模块（函数式）
功能：
- 按照目录树结构读取和处理Excel文件
- 将处理后的数据插入数据库
- 增量更新功能（只插入新数据）
- 错误检查和日志记录
- 保存错误数据和源文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import logging
import json
import shutil
import psycopg2
import psycopg2.extras

from .excel import read_file, cross_section_to_db_format
from .validation import validate_and_fix


def _get_default_logger():
    """获取默认logger"""
    logger = logging.getLogger('DBExcelIntegration')
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def process_excel_to_db_format(
    filepath: Union[str, Path],
    config: Dict[str, Any],
    logger: Optional[logging.Logger] = None
) -> Tuple[Optional[pd.DataFrame], List[Dict]]:
    """
    处理Excel文件为数据库格式
    
    Args:
        filepath: Excel文件路径
        config: 处理配置，包含：
            - key_columns: 键列列表
            - value_columns: 值列列表（如果为None，则自动推断）
            - key_value_mapping: 列名映射字典
            - additional_fields: 额外字段字典
            - validation: 验证配置
            - read_kwargs: 读取参数
        logger: 日志记录器，如果为None则使用默认logger
            
    Returns:
        (处理后的DataFrame, 错误列表)
    """
    if logger is None:
        logger = _get_default_logger()
    
    errors = []
    
    try:
        # 1. 读取文件
        read_kwargs = config.get('read_kwargs', {})
        df = read_file(filepath, logger=logger, **read_kwargs)
        
        # 2. 自动推断value_columns（如果未指定）
        key_columns = config.get('key_columns', [])
        value_columns = config.get('value_columns')
        
        if value_columns is None:
            # 自动推断：除了key_columns之外的所有列
            value_columns = [col for col in df.columns if col not in key_columns]
            logger.info(f"自动推断value_columns: {len(value_columns)}列")
        
        # 3. 转换格式
        key_value_mapping = config.get('key_value_mapping', {})
        additional_fields = config.get('additional_fields', {})
        
        df_converted = cross_section_to_db_format(
            df=df,
            key_columns=key_columns,
            value_columns=value_columns,
            key_value_mapping=key_value_mapping,
            additional_fields=additional_fields,
            logger=logger
        )
        
        # 4. 数据验证
        validation_config = config.get('validation', {})
        if validation_config:
            df_converted, validation_report = validate_and_fix(
                df_converted,
                validations=validation_config,
                logger=logger,
                inplace=True
            )
            
            # 检查是否有严重错误
            if 'null_check' in validation_report:
                null_results = validation_report['null_check']
                if null_results:
                    for col, info in null_results.items():
                        if info['issues']:
                            errors.append({
                                'type': 'validation_error',
                                'column': col,
                                'details': info['issues']
                            })
            
            if 'datetime_check' in validation_report:
                datetime_results = validation_report['datetime_check']
                if datetime_results.get('issues'):
                    errors.append({
                        'type': 'datetime_error',
                        'details': datetime_results['issues']
                    })
        
        return df_converted, errors
        
    except Exception as e:
        logger.error(f"处理文件失败 {filepath}: {str(e)}")
        errors.append({
            'type': 'processing_error',
            'message': str(e)
        })
        return None, errors


def check_existing_rows(
    cursor,
    df: pd.DataFrame,
    table: str,
    key_columns: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None
) -> pd.DataFrame:
    """
    检查数据库中是否已存在相关数据行
    
    Args:
        cursor: 数据库游标
        df: 待检查的DataFrame
        table: 目标表名
        key_columns: 用于判断重复的关键列，默认使用['code', 'metric', 'datetime']
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        包含已存在行的DataFrame（只包含key_columns列），如果没有则返回空DataFrame
    """
    if logger is None:
        logger = _get_default_logger()
    
    if df.empty:
        return pd.DataFrame()
    
    # 确定关键列
    if key_columns is None:
        # 默认关键列
        default_keys = ['code', 'metric', 'datetime']
        key_columns = [col for col in default_keys if col in df.columns]
        if not key_columns:
            # 如果默认列不存在，使用前几列作为关键列
            key_columns = df.columns[:min(3, len(df.columns))].tolist()
            logger.warning(f"使用前{len(key_columns)}列作为关键列: {key_columns}")
    
    # 检查关键列是否存在
    missing_cols = [col for col in key_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"关键列不存在: {missing_cols}")
        return pd.DataFrame()
    
    try:
        # 构建查询条件
        conditions = []
        params_list = []
        
        # 为每行构建OR条件
        for idx, row in df.iterrows():
            row_conditions = []
            for col in key_columns:
                row_conditions.append(f"{col} = %s")
                val = row[col]
                # 处理datetime类型
                if pd.api.types.is_datetime64_any_dtype(df[col]) or isinstance(val, pd.Timestamp):
                    params_list.append(val)
                else:
                    params_list.append(val)
            conditions.append(f"({' AND '.join(row_conditions)})")
        
        # 如果有太多行，分批查询（避免SQL过长）
        if len(df) > 1000:
            logger.info(f"数据量较大({len(df)}行)，分批检查重复...")
            all_existing = []
            
            for i in range(0, len(df), 1000):
                batch_df = df.iloc[i:i+1000]
                batch_conditions = []
                batch_params = []
                
                for idx, row in batch_df.iterrows():
                    row_conditions = []
                    for col in key_columns:
                        row_conditions.append(f"{col} = %s")
                        val = row[col]
                        if pd.api.types.is_datetime64_any_dtype(df[col]) or isinstance(val, pd.Timestamp):
                            batch_params.append(val)
                        else:
                            batch_params.append(val)
                    batch_conditions.append(f"({' AND '.join(row_conditions)})")
                
                query = f"""
                SELECT DISTINCT {', '.join(key_columns)}
                FROM {table}
                WHERE {' OR '.join(batch_conditions)}
                """
                
                cursor.execute(query, tuple(batch_params))
                results = cursor.fetchall()
                
                if results:
                    batch_existing = pd.DataFrame(results, columns=key_columns)
                    all_existing.append(batch_existing)
            
            if all_existing:
                return pd.concat(all_existing, ignore_index=True).drop_duplicates()
            else:
                return pd.DataFrame(columns=key_columns)
        else:
            # 一次性查询
            query = f"""
            SELECT DISTINCT {', '.join(key_columns)}
            FROM {table}
            WHERE {' OR '.join(conditions)}
            """
            
            cursor.execute(query, tuple(params_list))
            results = cursor.fetchall()
            
            if results:
                return pd.DataFrame(results, columns=key_columns)
            else:
                return pd.DataFrame(columns=key_columns)
                
    except Exception as e:
        logger.error(f"检查已存在数据失败: {str(e)}")
        # 返回空DataFrame，让插入继续（保守策略）
        return pd.DataFrame(columns=key_columns)


def insert_dataframe(
    cursor,
    conn,
    df: pd.DataFrame,
    table: str,
    batch_size: int = 1000,
    check_duplicates: bool = True,
    key_columns: Optional[List[str]] = None,
    skip_duplicates: bool = True,
    logger: Optional[logging.Logger] = None
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    将DataFrame插入数据库
    
    Args:
        cursor: 数据库游标
        conn: 数据库连接
        df: 待插入的DataFrame
        table: 目标表名
        batch_size: 批量插入大小
        check_duplicates: 是否检查数据库中已存在的数据（默认True）
        key_columns: 用于判断重复的关键列，默认使用['code', 'metric', 'datetime']
        skip_duplicates: 是否跳过重复数据（默认True），如果为False，重复数据会导致插入失败
        logger: 日志记录器，如果为None则使用默认logger
            
    Returns:
        (是否成功, 消息, 统计信息字典)
    """
    if logger is None:
        logger = _get_default_logger()
    
    stats = {
        'total_rows': len(df),
        'new_rows': 0,
        'duplicate_rows': 0,
        'inserted_rows': 0
    }
    
    if df.empty:
        return True, "DataFrame为空，跳过插入", stats
    
    try:
        # 检查重复
        if check_duplicates:
            logger.info("检查数据库中是否已存在相关数据...")
            existing_df = check_existing_rows(cursor, df, table, key_columns, logger=logger)
            
            if not existing_df.empty:
                # 找出重复的行
                # 合并待插入数据和已存在数据，找出重复
                df_keys = df[key_columns].copy()
                existing_keys = existing_df[key_columns].copy()
                
                # 确保数据类型一致（特别是datetime）
                for col in key_columns:
                    if col in df_keys.columns and col in existing_keys.columns:
                        if pd.api.types.is_datetime64_any_dtype(df_keys[col]):
                            df_keys[col] = pd.to_datetime(df_keys[col])
                            existing_keys[col] = pd.to_datetime(existing_keys[col])
                
                # 标记重复
                df_keys['_temp_index'] = df.index
                existing_keys['_exists'] = True
                
                # 合并找出重复
                merged = df_keys.merge(
                    existing_keys,
                    on=key_columns,
                    how='left',
                    indicator=True
                )
                
                duplicate_mask = merged['_merge'] == 'both'
                duplicate_indices = merged[duplicate_mask]['_temp_index'].values
                
                stats['duplicate_rows'] = len(duplicate_indices)
                
                if stats['duplicate_rows'] > 0:
                    logger.warning(
                        f"发现 {stats['duplicate_rows']} 行数据在数据库中已存在"
                    )
                    
                    if skip_duplicates:
                        # 跳过重复的行
                        df_to_insert = df[~df.index.isin(duplicate_indices)].copy()
                        stats['new_rows'] = len(df_to_insert)
                        
                        if df_to_insert.empty:
                            logger.info(
                                f"所有 {stats['total_rows']} 行数据都已存在于数据库中，跳过插入"
                            )
                            stats['inserted_rows'] = 0
                            return True, f"跳过 {stats['total_rows']} 行重复数据", stats
                    else:
                        # 不跳过，插入会失败（由数据库约束决定）
                        logger.warning("发现重复数据，插入可能会失败（取决于数据库约束）")
                        df_to_insert = df.copy()
                        stats['new_rows'] = len(df_to_insert)
                else:
                    df_to_insert = df.copy()
                    stats['new_rows'] = len(df_to_insert)
            else:
                df_to_insert = df.copy()
                stats['new_rows'] = len(df_to_insert)
                logger.info("未发现重复数据，将插入所有行")
        else:
            df_to_insert = df.copy()
            stats['new_rows'] = len(df_to_insert)
        
        if df_to_insert.empty:
            return True, f"跳过 {stats['duplicate_rows']} 行重复数据", stats
        
        # 准备数据
        tuples = [tuple(x) for x in df_to_insert.to_numpy()]
        cols = ','.join(list(df_to_insert.columns))
        
        # SQL语句
        query = f"INSERT INTO {table}({cols}) VALUES %s"
        
        # 批量插入
        total_rows = len(tuples)
        inserted_rows = 0
        
        for i in range(0, total_rows, batch_size):
            batch = tuples[i:i+batch_size]
            
            psycopg2.extras.execute_values(cursor, query, batch)
            conn.commit()
            
            inserted_rows += len(batch)
            logger.info(f"已插入 {inserted_rows}/{total_rows} 行")
        
        stats['inserted_rows'] = inserted_rows
        
        msg_parts = [f"成功插入 {inserted_rows} 行"]
        if stats['duplicate_rows'] > 0:
            msg_parts.append(f"，跳过 {stats['duplicate_rows']} 行重复数据")
        
        message = "".join(msg_parts)
        logger.info(f"插入完成: {message}")
        return True, message, stats
        
    except Exception as e:
        logger.error(f"插入数据失败: {str(e)}")
        conn.rollback()
        error_msg = str(e)
        # 检查是否是重复键错误
        if 'duplicate' in error_msg.lower() or 'unique' in error_msg.lower():
            error_msg = f"发现重复数据: {error_msg}"
        return False, error_msg, stats


def get_existing_dates(
    cursor,
    table: str,
    code: str,
    metric: str,
    logger: Optional[logging.Logger] = None
) -> List[datetime]:
    """
    获取数据库中已存在的日期
    
    Args:
        cursor: 数据库游标
        table: 表名
        code: 代码
        metric: 指标
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        日期列表
    """
    if logger is None:
        logger = _get_default_logger()
    
    try:
        query = f"""
        SELECT DISTINCT datetime 
        FROM {table} 
        WHERE code = %s AND metric = %s 
        ORDER BY datetime
        """
        
        cursor.execute(query, (code, metric))
        results = cursor.fetchall()
        
        dates = [row['datetime'] for row in results]
        return dates
        
    except Exception as e:
        logger.error(f"获取已存在日期失败: {str(e)}")
        return []


def incremental_insert(
    cursor,
    conn,
    df: pd.DataFrame,
    table: str,
    date_column: str = 'datetime',
    code_column: str = 'code',
    metric_column: str = 'metric',
    logger: Optional[logging.Logger] = None
) -> Tuple[int, int]:
    """
    增量插入：只插入数据库中不存在的数据
    
    Args:
        cursor: 数据库游标
        conn: 数据库连接
        df: 待插入的DataFrame
        table: 目标表名
        date_column: 日期列名
        code_column: 代码列名
        metric_column: 指标列名
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        (新增行数, 重复行数)
    """
    if logger is None:
        logger = _get_default_logger()
    
    if df.empty:
        return 0, 0
    
    try:
        # 获取所有唯一的(code, metric)组合
        unique_combinations = df[[code_column, metric_column]].drop_duplicates()
        
        # 对每个组合，获取已存在的日期
        new_data_list = []
        duplicate_count = 0
        
        for _, row in unique_combinations.iterrows():
            code = row[code_column]
            metric = row[metric_column]
            
            # 获取已存在的日期
            existing_dates = get_existing_dates(cursor, table, code, metric, logger=logger)
            existing_dates_set = set(existing_dates)
            
            # 筛选该组合的数据
            mask = (df[code_column] == code) & (df[metric_column] == metric)
            subset = df[mask].copy()
            
            # 找出新数据
            if existing_dates_set:
                # 确保datetime列是datetime类型
                if not pd.api.types.is_datetime64_any_dtype(subset[date_column]):
                    subset[date_column] = pd.to_datetime(subset[date_column])
                
                new_mask = ~subset[date_column].isin(existing_dates_set)
                new_data = subset[new_mask]
                
                duplicate_count += len(subset) - len(new_data)
            else:
                new_data = subset
            
            if not new_data.empty:
                new_data_list.append(new_data)
        
        # 合并所有新数据
        if new_data_list:
            new_df = pd.concat(new_data_list, ignore_index=True)
            
            # 插入新数据（关闭重复检查，因为已经在增量逻辑中处理了）
            success, message, stats = insert_dataframe(
                cursor,
                conn,
                new_df, 
                table,
                check_duplicates=False,  # 增量插入已经检查过了
                logger=logger
            )
            
            if success:
                logger.info(f"增量插入完成: 新增 {len(new_df)} 行, 跳过 {duplicate_count} 行重复数据")
                return len(new_df), duplicate_count
            else:
                logger.error(f"增量插入失败: {message}")
                return 0, duplicate_count
        else:
            logger.info(f"没有新数据需要插入，所有 {duplicate_count} 行数据已存在")
            return 0, duplicate_count
            
    except Exception as e:
        logger.error(f"增量插入失败: {str(e)}")
        conn.rollback()
        return 0, 0


def save_error_file(
    filepath: Union[str, Path],
    df: Optional[pd.DataFrame],
    errors: List[Dict],
    error_dir: Union[str, Path] = "./errors",
    error_subdir: str = "failed_files",
    logger: Optional[logging.Logger] = None
) -> str:
    """
    保存错误文件和日志
    
    Args:
        filepath: 原始文件路径
        df: 处理后的DataFrame（可能为None）
        errors: 错误列表
        error_dir: 错误文件保存目录
        error_subdir: 错误子目录
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        错误记录ID
    """
    if logger is None:
        logger = _get_default_logger()
    
    filepath = Path(filepath)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    error_id = f"{filepath.stem}_{timestamp}"
    
    # 创建错误文件目录
    error_dir_path = Path(error_dir)
    error_dir_path.mkdir(parents=True, exist_ok=True)
    
    # 创建错误子目录
    error_path = error_dir_path / error_subdir
    error_path.mkdir(parents=True, exist_ok=True)
    
    # 复制源文件
    source_copy = error_path / f"{error_id}_source{filepath.suffix}"
    shutil.copy2(filepath, source_copy)
    logger.info(f"已保存源文件: {source_copy}")
    
    # 保存处理后的数据（如果有）
    if df is not None and not df.empty:
        processed_file = error_path / f"{error_id}_processed.csv"
        df.to_csv(processed_file, index=False, encoding='utf_8_sig')
        logger.info(f"已保存处理后数据: {processed_file}")
    
    # 保存错误日志
    error_log_file = error_path / f"{error_id}_errors.json"
    error_info = {
        'error_id': error_id,
        'source_file': str(filepath),
        'timestamp': timestamp,
        'errors': errors
    }
    
    with open(error_log_file, 'w', encoding='utf-8') as f:
        json.dump(error_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"已保存错误日志: {error_log_file}")
    
    return error_id


def process_directory_tree(
    cursor,
    conn,
    root_dir: Union[str, Path],
    table: str,
    config: Dict[str, Any],
    file_pattern: str = "*.csv",
    recursive: bool = True,
    mode: str = 'insert',  # 'insert' or 'incremental'
    error_dir: Union[str, Path] = "./errors",
    logger: Optional[logging.Logger] = None
) -> Dict[str, Any]:
    """
    按照目录树结构处理和插入Excel文件
    
    Args:
        cursor: 数据库游标
        conn: 数据库连接
        root_dir: 根目录
        table: 目标表名
        config: 处理配置
        file_pattern: 文件匹配模式
        recursive: 是否递归搜索
        mode: 插入模式，'insert'（直接插入）或'incremental'（增量插入）
        error_dir: 错误文件保存目录
        logger: 日志记录器，如果为None则使用默认logger
        
    Returns:
        处理统计字典
    """
    if logger is None:
        logger = _get_default_logger()
    
    root_dir = Path(root_dir)
    
    # 统计信息
    stats = {
        'total_files': 0,
        'success_files': 0,
        'error_files': 0,
        'total_rows_inserted': 0,
        'total_rows_skipped': 0,
        'error_details': []
    }
    
    # 错误日志收集器
    error_log = []
    
    # 查找所有文件
    if recursive:
        files = list(root_dir.rglob(file_pattern))
    else:
        files = list(root_dir.glob(file_pattern))
    
    stats['total_files'] = len(files)
    logger.info(f"找到 {len(files)} 个文件需要处理")
    
    # 处理每个文件
    for file in files:
        logger.info(f"处理文件: {file}")
        
        try:
            # 1. 处理文件
            df, errors = process_excel_to_db_format(file, config, logger=logger)
            
            if df is None or errors:
                # 有错误，保存错误文件
                error_id = save_error_file(
                    file, df, errors, 
                    error_dir=error_dir, 
                    logger=logger
                )
                error_log.append({
                    'error_id': error_id,
                    'source_file': str(file),
                    'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                    'errors': errors
                })
                stats['error_files'] += 1
                stats['error_details'].append({
                    'file': str(file),
                    'error_id': error_id,
                    'errors': errors
                })
                logger.warning(f"文件处理失败: {file}, 错误ID: {error_id}")
                continue
            
            # 2. 插入数据库
            if mode == 'insert':
                # 使用重复检查（默认开启）
                success, message, insert_stats = insert_dataframe(
                    cursor,
                    conn,
                    df, 
                    table,
                    check_duplicates=True,
                    skip_duplicates=True,
                    logger=logger
                )
                
                if success:
                    stats['success_files'] += 1
                    stats['total_rows_inserted'] += insert_stats.get('inserted_rows', 0)
                    stats['total_rows_skipped'] += insert_stats.get('duplicate_rows', 0)
                    logger.info(f"文件处理成功: {file} - {message}")
                else:
                    error_id = save_error_file(
                        file, df, [{
                            'type': 'insert_error',
                            'message': message
                        }],
                        error_dir=error_dir,
                        logger=logger
                    )
                    error_log.append({
                        'error_id': error_id,
                        'source_file': str(file),
                        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                        'errors': [{'type': 'insert_error', 'message': message}]
                    })
                    stats['error_files'] += 1
                    stats['error_details'].append({
                        'file': str(file),
                        'error_id': error_id,
                        'errors': [{'type': 'insert_error', 'message': message}]
                    })
            
            elif mode == 'incremental':
                new_rows, skipped_rows = incremental_insert(
                    cursor,
                    conn,
                    df, 
                    table,
                    logger=logger
                )
                
                if new_rows > 0 or skipped_rows > 0:
                    stats['success_files'] += 1
                    stats['total_rows_inserted'] += new_rows
                    stats['total_rows_skipped'] += skipped_rows
                    logger.info(f"文件增量插入成功: {file}, 新增{new_rows}行, 跳过{skipped_rows}行")
                else:
                    logger.info(f"文件无新数据: {file}")
                    stats['success_files'] += 1
            
        except Exception as e:
            logger.error(f"处理文件异常 {file}: {str(e)}")
            error_id = save_error_file(
                file, None, [{
                    'type': 'exception',
                    'message': str(e)
                }],
                error_dir=error_dir,
                logger=logger
            )
            error_log.append({
                'error_id': error_id,
                'source_file': str(file),
                'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
                'errors': [{'type': 'exception', 'message': str(e)}]
            })
            stats['error_files'] += 1
            stats['error_details'].append({
                'file': str(file),
                'error_id': error_id,
                'errors': [{'type': 'exception', 'message': str(e)}]
            })
    
    # 保存总错误日志
    if error_log:
        error_dir_path = Path(error_dir)
        error_dir_path.mkdir(parents=True, exist_ok=True)
        summary_log = error_dir_path / f"error_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_log, 'w', encoding='utf-8') as f:
            json.dump({
                'stats': stats,
                'errors': error_log
            }, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存总错误日志: {summary_log}")
    
    # 输出统计
    logger.info("="*50)
    logger.info(f"处理完成统计:")
    logger.info(f"  总文件数: {stats['total_files']}")
    logger.info(f"  成功: {stats['success_files']}")
    logger.info(f"  失败: {stats['error_files']}")
    logger.info(f"  插入行数: {stats['total_rows_inserted']}")
    logger.info(f"  跳过行数: {stats['total_rows_skipped']}")
    logger.info("="*50)
    
    return stats

