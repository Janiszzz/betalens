#%%By Janis 250226
'''
## 研究数据库结构（摘要）

### 表1 个券行情（日频）
- 列：入库实际时间（=最早可交易时间）、windcode、中文名、数据性质（收盘价/成交量等）、数值、备注（json）
- 规则：开盘价最早 09:30；其余价量最早 15:00 可确定
- 处理：计算日频收益率（如 close-to-close）用于回测与挖掘

### 表2 个券基本面（日频入库，事件驱动）
- 列：入库实际时间（=最早可交易时间）、数据理论发生时间（报告期，如 0331/0630/0930/1231）、windcode、中文名、数据性质（如归母净利润）、数值、备注（json）
- 规则：按公告时点入库，关注盘前/盘中/盘后；理论发生时间仅用于展示，不做因果外推
- 年报次序不一：优先一致预期或线性外推作占位，尽量避免非原生数据入库

### 表3 宏观经济（事件驱动）
- 列：入库实际时间（=最早可交易时间）、数据理论发生时间（如“1月GDP”）、windcode、中文名、数据性质（可含均线算子）、数值、备注（json）
- 规则：与表2一致（事件驱动、区分公告时点与发生时点）

### 表4 因子库
- 列：入库实际时间（=最早可交易时间）、数据编制方式、数值、备注（json）

## 投资数据库结构（摘要）
- 不保留完整历史
- 从研究数据库按需拉取最近滚动窗口数据；其余数据在线拉取
- 在线数据仅记录入库时间（可交易可用时点）

## 投资数据库结构：
- 出于投资目的，实际不需要历史数据
- 从研究数据库拉取所需最近一个滚动窗口内的数据，其余全部在线拉取，并直接记录入库时间

更新日志：
2025-10-31: 重构datafeed.py，新增工具模块
- excel.py: Excel文件处理模块
- validation.py: 数据验证和异常检查工具
- query.py: 数据库查询功能重构
- integration.py: 数据库与Excel交互功能
- wind_ingest.py: Wind数据抓取模块

2025-11-03: 简化core.py为薄封装层
- 移除所有业务逻辑到子模块
- core.py仅保留连接管理和函数组合
- 所有数据转换、插入逻辑位于子模块
'''
import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import os
import itertools
from psycopg2.extras import execute_values
from functools import wraps
import time
import logging
from pathlib import Path
from datetime import datetime
import warnings

# 导入新的工具模块（函数式）
from .excel import read_file, cross_section_to_db_format, check_excel_errors, apply_time_alignment
from .validation import validate_and_fix
from .query import (
    query_nearest_after as _query_nearest_after,
    query_nearest_before as _query_nearest_before,
    query_time_range as _query_time_range,
    get_available_dates as _get_available_dates,
    get_latest_date as _get_latest_date,
    build_query as _build_query
)
from .integration import (
    process_directory_tree as _process_directory_tree,
    incremental_insert as _incremental_insert,
    insert_dataframe as _insert_dataframe,
    process_excel_to_db_format as _process_excel_to_db_format
)
from .wind_ingest import (
    fetch_daily_market as _fetch_daily_market,
    fetch_daily_index as _fetch_daily_index,
    fetch_daily_fund as _fetch_daily_fund,
    fetch_daily_bond as _fetch_daily_bond
)
from .ede_processor import (
    process_ede_file as _process_ede_file
)
from .config import get_database_config, get_logging_config

def func_timer(function):
    '''
    用装饰器实现函数计时
    :param function: 需要计时的函数
    :return: None
    '''
    @wraps(function)
    def function_timer(*args, **kwargs):
        print('[Function: {name} start...]'.format(name = function.__name__))
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[Function: {name} finished, spent time: {time:.2f}s]'.format(name = function.__name__,time = t1 - t0))
        return result
    return function_timer

class Datafeed():
    """
    数据管理主类（薄封装层）
    
    职责：
    - 管理数据库连接
    - 提供统一的日志记录
    - 组合子模块功能为高层API
    - 不包含业务逻辑（所有逻辑在子模块中）
    
    使用示例：
        # 创建实例（使用默认配置）
        df = Datafeed("daily_market_data")
        
        # 创建实例（自定义数据库配置）
        df = Datafeed(
            table_name="daily_market_data",
            db_config={
                'dbname': 'my_database',
                'user': 'my_user',
                'password': 'my_password',
                'host': 'localhost',
                'port': '5432'
            }
        )
        
        # 单文件插入
        df.insert_csv_file("data.csv", config={...})
        
        # Wind数据抓取
        df.ingest_wind_daily_market(
            codes=['000001.SZ'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # 查询
        data = df.query_time_range(
            codes=['000001.SZ'],
            start_date='2024-01-01',
            metric='收盘价(元)'
        )
        
        # 关闭
        df.close()
    """
    _initialized = False

    def __init__(
        self, 
        table_name, 
        db_config=None,
        log_dir=None
    ):
        """
        初始化Datafeed实例
        
        Args:
            table_name: 数据库表名
            db_config: 数据库配置字典，包含以下键：
                - dbname: 数据库名
                - user: 用户名
                - password: 密码
                - host: 主机地址
                - port: 端口
                如果为None，使用config.json中的配置
            log_dir: 日志目录，如果为None，使用config.json中的配置
        """
        # 从配置文件获取默认配置
        default_db_config = get_database_config()
        default_log_config = get_logging_config()
        
        # 合并用户配置和默认配置
        if db_config is None:
            db_config = default_db_config
        else:
            # 使用用户提供的配置，未提供的使用默认值
            db_config = {**default_db_config, **db_config}
        
        # 设置日志目录
        if log_dir is None:
            log_dir = default_log_config.get('log_dir', './logs')
        
        # 建立数据库连接
        self.conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        self.cursor = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        self.sheet = table_name
        
        # 设置logger（用于函数式调用）
        log_dir_path = Path(log_dir)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_file = log_dir_path / f"datafeed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        self.logger = logging.getLogger('Datafeed')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            fh = logging.FileHandler(log_file, encoding='utf-8')
            fh.setLevel(logging.INFO)
            ch = logging.StreamHandler()
            ch.setLevel(logging.WARNING)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            ch.setFormatter(formatter)
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)
        
        self.__class__._initialized = True

    # ========== 插入功能 ==========
    
    def insert_csv_file(
        self,
        filepath: str,
        config: dict,
        mode: str = 'incremental'
    ) -> dict:
        """
        单文件CSV插入（薄封装）
        
        组合流程：
        1. excel.read_file - 读取文件
        2. excel.cross_section_to_db_format - 转换格式（如果需要）
        3. validation.validate_and_fix - 验证和修复（如果配置）
        4. integration.insert_dataframe/incremental_insert - 插入
        
        Args:
            filepath: 文件路径
            config: 配置字典，包含：
                - key_columns: 键列列表（用于cross_section转换）
                - value_columns: 值列列表（可选，自动推断）
                - key_value_mapping: 列名映射
                - additional_fields: 额外字段
                - validation: 验证配置
                - apply_time_alignment: 是否应用时间对齐（开盘09:30，其他15:00）
            mode: 'insert'（直接插入）或'incremental'（增量插入，默认）
            
        Returns:
            统计信息字典
        """
        self.logger.info(f"开始处理文件: {filepath}, 模式: {mode}")
        
        # 1. 处理Excel为DB格式
        df, errors = _process_excel_to_db_format(
            filepath=filepath,
            config=config,
            logger=self.logger
        )
        
        if df is None or errors:
            self.logger.error(f"文件处理失败: {errors}")
            return {'success': False, 'errors': errors}
        
        # 2. 可选：应用时间对齐
        if config.get('apply_time_alignment', False):
            df = apply_time_alignment(
                df,
                date_column=config.get('date_column', 'datetime'),
                metric_column=config.get('metric_column', 'metric'),
                logger=self.logger
            )
        
        # 3. 插入数据库
        if mode == 'insert':
            success, message, stats = _insert_dataframe(
                cursor=self.cursor,
                conn=self.conn,
                df=df,
                table=self.sheet,
                logger=self.logger
            )
            return {'success': success, 'message': message, 'stats': stats}
        
        elif mode == 'incremental':
            new_rows, skipped_rows = _incremental_insert(
                cursor=self.cursor,
                conn=self.conn,
                df=df,
                table=self.sheet,
                logger=self.logger
            )
            return {
                'success': True,
                'new_rows': new_rows,
                'skipped_rows': skipped_rows
            }
        
        else:
            raise ValueError(f"未知的插入模式: {mode}")
    
    def insert_ede_file(
        self,
        filepath: str,
        date_from: str = 'filename',
        default_datetime: str = None,
        mode: str = 'incremental'
    ) -> dict:
        """
        处理并插入EDE格式的Excel文件（薄封装）
        
        EDE格式特征：
        - 第一列：证券代码
        -第二列：证券简称
        - 第三列及之后：指标列，格式为"指标名 [元数据] 值类型 [元数据] 单位"
        
        示例EDE格式：
            证券代码    证券简称    流通A股 [交易日期] 最新 [单位] 股
            002460.SZ  赣锋锂业    1,211,379,763.0000
            1772.HK    赣锋锂业    1,211,379,763.0000
        
        处理流程：
        1. 读取Excel文件
        2. 清理数据（去除空值、"数据来源：Wind"等）
        3. 识别code、name列
        4. 解析metric列，提取指标名称和元数据
        5. 构建日期列（从文件名或列名中提取）
        6. 转换为数据库格式
        7. 插入数据库
        
        Args:
            filepath: EDE格式Excel文件路径
            date_from: 日期来源，可选值：
                - 'filename': 从文件名提取日期（如EDE20251103.xlsx -> 2025-11-03 15:30:00）
                - 'metric': 从列名中的[日期]部分提取
            default_datetime: 默认日期时间（当无法从文件名或列名提取时使用）
                格式：'YYYY-MM-DD HH:MM:SS'，如'2025-11-03 15:30:00'
            mode: 插入模式
                - 'incremental': 增量插入（默认），只插入新数据
                - 'insert': 直接插入，会检查重复并跳过
            
        Returns:
            统计信息字典，包含：
                - success: 是否成功
                - new_rows: 新增行数
                - skipped_rows: 跳过行数
                - errors: 错误列表（如果有）
        
        Example:
            >>> df = Datafeed('daily_market_data')
            >>> result = df.insert_ede_file(
            ...     'EDE20251103.xlsx',
            ...     date_from='filename',
            ...     mode='incremental'
            ... )
            >>> print(f"新增{result['new_rows']}行，跳过{result['skipped_rows']}行")
        """
        self.logger.info(f"开始处理EDE文件: {filepath}, 日期来源: {date_from}, 模式: {mode}")
        
        # 1. 处理EDE文件为DB格式
        df, errors = _process_ede_file(
            filepath=filepath,
            date_from=date_from,
            default_datetime=default_datetime,
            logger=self.logger
        )
        
        if df is None or errors:
            self.logger.error(f"EDE文件处理失败: {errors}")
            return {'success': False, 'errors': errors}
        
        self.logger.info(f"EDE文件转换完成: {len(df)} 行数据")
        
        # 2. 插入数据库
        if mode == 'insert':
            success, message, stats = _insert_dataframe(
                cursor=self.cursor,
                conn=self.conn,
                df=df,
                table=self.sheet,
                logger=self.logger
            )
            return {
                'success': success,
                'message': message,
                'new_rows': stats.get('inserted_rows', 0),
                'skipped_rows': stats.get('duplicate_rows', 0),
                'stats': stats
            }
        
        elif mode == 'incremental':
            new_rows, skipped_rows = _incremental_insert(
                cursor=self.cursor,
                conn=self.conn,
                df=df,
                table=self.sheet,
                logger=self.logger
            )
            return {
                'success': True,
                'new_rows': new_rows,
                'skipped_rows': skipped_rows
            }
        
        else:
            raise ValueError(f"未知的插入模式: {mode}")
    
    def batch_process_excel_files(
        self,
        folder_path: str,
        config: dict,
        file_pattern: str = "*.csv",
        recursive: bool = True,
        mode: str = 'insert'
    ):
        """
        批量处理Excel文件并插入数据库（薄封装）
        
        直接调用 integration.process_directory_tree
        
        Args:
            folder_path: 文件夹路径
            config: 处理配置
            file_pattern: 文件匹配模式
            recursive: 是否递归搜索
            mode: 插入模式，'insert'或'incremental'
            
        Returns:
            处理统计字典
        """
        return _process_directory_tree(
            cursor=self.cursor,
            conn=self.conn,
            root_dir=folder_path,
            table=self.sheet,
            config=config,
            file_pattern=file_pattern,
            recursive=recursive,
            mode=mode,
            logger=self.logger
        )
    
    def incremental_update(
        self,
        df: pd.DataFrame,
        date_column: str = 'datetime',
        code_column: str = 'code',
        metric_column: str = 'metric'
    ):
        """
        增量更新数据到数据库（薄封装）
        
        直接调用 integration.incremental_insert
        
        Args:
            df: 待更新的DataFrame
            date_column: 日期列名
            code_column: 代码列名
            metric_column: 指标列名
            
        Returns:
            (新增行数, 重复行数)
        """
        new_rows, skipped_rows = _incremental_insert(
            cursor=self.cursor,
            conn=self.conn,
            df=df,
            table=self.sheet,
            date_column=date_column,
            code_column=code_column,
            metric_column=metric_column,
            logger=self.logger
        )
        return new_rows, skipped_rows
    
    # ========== Wind数据抓取 ==========
    
    def ingest_wind_daily_market(
        self,
        codes: list,
        start_date: str,
        end_date: str,
        fields: list = None,
        asset_type: str = 'stock',
        mode: str = 'incremental'
    ) -> dict:
        """
        从Wind获取日行情并插入数据库（薄封装）
        
        组合流程：
        1. wind_ingest.fetch_daily_market - 获取Wind数据
        2. integration.incremental_insert/insert_dataframe - 插入
        
        Args:
            codes: 代码列表
            start_date: 开始日期，格式'YYYY-MM-DD'
            end_date: 结束日期
            fields: 字段列表，None使用默认字段
            asset_type: 资产类型，'stock', 'index', 'fund', 'bond'
            mode: 插入模式，'incremental'（默认）或'insert'
            
        Returns:
            统计信息字典
        """
        self.logger.info(
            f"开始从Wind获取数据: codes={len(codes)}, "
            f"date_range={start_date}~{end_date}, asset_type={asset_type}"
        )
        
        # 1. 从Wind获取数据
        df = _fetch_daily_market(
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            fields=fields,
            asset_type=asset_type,
            apply_time_stamps=True,
            logger=self.logger
        )
        
        if df.empty:
            self.logger.warning("未获取到任何数据")
            return {'success': False, 'message': 'No data fetched'}
        
        # 2. 插入数据库
        if mode == 'incremental':
            new_rows, skipped_rows = _incremental_insert(
                cursor=self.cursor,
                conn=self.conn,
                df=df,
                table=self.sheet,
                logger=self.logger
            )
            return {
                'success': True,
                'new_rows': new_rows,
                'skipped_rows': skipped_rows,
                'total_fetched': len(df)
            }
        else:
            success, message, stats = _insert_dataframe(
                cursor=self.cursor,
                conn=self.conn,
                df=df,
                table=self.sheet,
                logger=self.logger
            )
            return {'success': success, 'message': message, 'stats': stats}
    
    def ingest_wind_daily_index(self, codes: list, start_date: str, end_date: str, 
                                  fields: list = None, mode: str = 'incremental') -> dict:
        """Wind指数数据抓取（便捷封装）"""
        return self.ingest_wind_daily_market(codes, start_date, end_date, fields, 'index', mode)
    
    def ingest_wind_daily_fund(self, codes: list, start_date: str, end_date: str,
                                fields: list = None, mode: str = 'incremental') -> dict:
        """Wind基金数据抓取（便捷封装）"""
        return self.ingest_wind_daily_market(codes, start_date, end_date, fields, 'fund', mode)
    
    def ingest_wind_daily_bond(self, codes: list, start_date: str, end_date: str,
                                fields: list = None, mode: str = 'incremental') -> dict:
        """Wind债券数据抓取（便捷封装）"""
        return self.ingest_wind_daily_market(codes, start_date, end_date, fields, 'bond', mode)
    
    # ========== 查询功能（薄封装）==========
    
    @func_timer
    def run_query(self, conditions: list = None, params: list = None, select_columns: str = '*'):
        """
        执行自定义SQL查询（薄封装）
        
        替代原 query_data 方法，使用 query.build_query
        
        Args:
            conditions: SQL条件列表，如['datetime >= %s', 'code = %s']
            params: 参数列表
            select_columns: 要选择的列
            
        Returns:
            DataFrame
        """
        query, params = _build_query(
            table_name=self.sheet,
            conditions=conditions,
            params=params,
            select_columns=select_columns
        )
        
        self.logger.info(f"执行查询: {query}")
        self.cursor.execute(query, params)
        
        result = pd.DataFrame(self.cursor.fetchall())
        return result
    
    @func_timer
    def query_time_range(
        self,
        codes: list = None,
        start_date: str = None,
        end_date: str = None,
        metric: str = None
    ):
        """
        查询指定时间范围的数据（薄封装）
        
        直接调用 query.query_time_range
        
        Args:
            codes: 代码列表
            start_date: 开始日期
            end_date: 结束日期
            metric: 指标
            
        Returns:
            DataFrame
        """
        return _query_time_range(
            cursor=self.cursor,
            table_name=self.sheet,
            codes=codes,
            start_date=start_date,
            end_date=end_date,
            metric=metric,
            logger=self.logger
        )
    
    @func_timer
    def query_nearest_after(self, params=None):
        """
        根据输入时间戳序列查找每个时点之后最近的有效值（薄封装）
        
        主要用于回测时提取价格
        直接调用 query.query_nearest_after
        
        Args:
            params (dict): 必须包含以下键：
                - codes: 代码列表
                - datetimes: 目标时间戳列表（格式：'YYYY-MM-DD HH:MM'）
                - metric: 查询的指标名称
                - time_tolerance: 允许的最大时间间隔（单位：小时，默认不限制）

        Returns:
            DataFrame: 包含以下列：
                code | input_ts | datetime | diff_hours | value | name
        """
        # 参数校验
        required_keys = ['codes', 'datetimes', 'metric']
        if not all(k in params for k in required_keys):
            raise ValueError(f"必须提供参数: {required_keys}")

        # 使用函数式查询
        df = _query_nearest_after(
            cursor=self.cursor,
            table_name=self.sheet,
            codes=params['codes'],
            datetimes=params['datetimes'],
            metric=params['metric'],
            time_tolerance=params.get('time_tolerance'),
            logger=self.logger
        )
        
        return df

    @func_timer
    def query_nearest_before(self, params=None):
        """
        根据输入时间戳序列查找每个时点之前最近的有效值（薄封装）
        
        主要用于回测时提取历史价格特征
        直接调用 query.query_nearest_before
        
        Args:
            params (dict): 必须包含以下键：
                - codes: 代码列表
                - datetimes: 目标时间戳列表（格式：'YYYY-MM-DD HH:MM'）
                - metric: 查询的指标名称
                - time_tolerance: 允许的最大时间间隔（单位：小时，默认不限制）

        Returns:
            DataFrame: 包含以下列：
                code | input_ts | datetime | diff_hours | value | name
        """
        # 参数校验
        required_keys = ['codes', 'datetimes', 'metric']
        if not all(k in params for k in required_keys):
            raise ValueError(f"必须提供参数: {required_keys}")

        # 使用函数式查询
        df = _query_nearest_before(
            cursor=self.cursor,
            table_name=self.sheet,
            codes=params['codes'],
            datetimes=params['datetimes'],
            metric=params['metric'],
            time_tolerance=params.get('time_tolerance'),
            logger=self.logger
        )
        
        return df
    
    def get_latest_date(self, code: str = None, metric: str = None):
        """
        获取数据库中的最新日期（薄封装）
        
        直接调用 query.get_latest_date
        
        Args:
            code: 代码，None表示所有代码
            metric: 指标，None表示所有指标
            
        Returns:
            最新日期
        """
        return _get_latest_date(
            cursor=self.cursor,
            table_name=self.sheet,
            code=code,
            metric=metric,
            logger=self.logger
        )
    
    def get_available_dates(
        self,
        code: str,
        metric: str,
        start_date: str = None,
        end_date: str = None
    ):
        """
        获取指定代码和指标的可用日期列表（薄封装）
        
        直接调用 query.get_available_dates
        
        Args:
            code: 代码
            metric: 指标
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            日期列表
        """
        return _get_available_dates(
            cursor=self.cursor,
            table_name=self.sheet,
            code=code,
            metric=metric,
            start_date=start_date,
            end_date=end_date,
            logger=self.logger
        )
    
    # ========== 验证功能（薄封装）==========
    
    def validate_dataframe(
        self,
        df: pd.DataFrame,
        validations: dict
    ):
        """
        验证和修复DataFrame（薄封装）
        
        直接调用 validation.validate_and_fix
        
        Args:
            df: 待验证的DataFrame
            validations: 验证配置
            
        Returns:
            (修复后的DataFrame, 验证报告)
        """
        return validate_and_fix(
            df,
            validations=validations,
            logger=self.logger,
            inplace=False
        )
    
    def check_excel_file(
        self,
        filepath: str,
        checks: dict = None
    ):
        """
        检查Excel文件中的错误（薄封装）
        
        直接调用 excel.check_excel_errors
        
        Args:
            filepath: 文件路径
            checks: 检查配置
            
        Returns:
            (是否通过, 错误列表)
        """
        df = read_file(filepath, logger=self.logger)
        return check_excel_errors(df, checks, logger=self.logger)
    
    # ========== 连接管理 ==========
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        self.logger.info("数据库连接已关闭")
    


# ========== 辅助函数（保留）==========

def get_absolute_trade_days(begin_date, end_date, period):
    """
    获取交易日序列
    
    Args:
        begin_date: 开始日期，字符串格式
        end_date: 结束日期，字符串格式
        period: 周期，如'D'(日), 'W'(周), 'M'(月)
        
    Returns:
        交易日列表
    """
    from WindPy import w
    w.start()
    trade_days = w.tdays(begin_date, end_date, "Period="+period).Data[0]
    return trade_days

def trade_days_offset(begin_datetime, offset, period = 'D'):
    """
    交易日偏移计算
    
    Args:
        begin_datetime: 起始datetime对象
        offset: 偏移量（整数）
        period: 周期，默认'D'
        
    Returns:
        偏移后的datetime对象
    """
    from datetime import datetime, timedelta
    from WindPy import w
    time_part = begin_datetime.time()  # datetime.time对象

    # 格式化日期为字符串
    begin_date_str = begin_datetime.strftime('%Y-%m-%d')

    # 启动 WindPy
    w.start()

    # 获取偏移后的交易日（仅日期部分）
    end_date = w.tdaysoffset(offset, begin_date_str, "Period=" + period).Data[0][0]

    # 将时间部分加回去
    final_datetime = datetime.combine(end_date, time_part)

    return final_datetime
