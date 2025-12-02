#%%By Janis 250226
"""
Datafeed工具包
包含Excel处理、数据验证、数据库查询和集成等功能模块

子模块：
- core: 核心Datafeed类及辅助函数
- excel: Excel文件处理工具
- validation: 数据验证和异常检查工具
- query: 数据库查询功能
- integration: 数据库与Excel交互功能

"""

# 导入核心类和函数
from .core import (
    Datafeed,
    func_timer,
    get_absolute_trade_days,
    trade_days_offset
)

# 导入Excel处理模块（函数式）
from .excel import (
    read_file,
    read_csv_with_encoding,
    cross_section_to_db_format,
    batch_read_files,
    batch_write_files,
    create_directory_tree,
    check_excel_errors
)

# 导入数据验证模块（函数式）
from .validation import (
    FillStrategy,
    check_null_values,
    check_datetime_column,
    fix_null_values,
    drop_duplicates_strict,
    fix_datetime_column,
    validate_and_fix,
    DataValidator  # 向后兼容
)

# 导入数据库查询模块（函数式）
from .query import (
    build_query,
    generate_input_pairs,
    build_nearest_query,
    query_nearest_after,
    query_nearest_before,
    query_time_range,
    get_available_dates,
    get_latest_date,
    pivot_to_wide,
    align_to_dates,
    calculate_returns
)

# 导入数据库Excel集成模块（函数式）
from .integration import (
    process_excel_to_db_format,
    check_existing_rows,
    insert_dataframe,
    get_existing_dates,
    incremental_insert,
    save_error_file,
    process_directory_tree
)

# 导入配置管理模块
from .config import (
    ConfigManager,
    get_config,
    reset_config,
    get_database_config,
    get_logging_config,
    get_excel_config,
    get_wind_config,
    get_ede_config
)

# 定义公开的API
__all__ = [
    # 核心类
    'Datafeed',
    
    # Excel处理（函数式）
    'read_file',
    'read_csv_with_encoding',
    'cross_section_to_db_format',
    'batch_read_files',
    'batch_write_files',
    'create_directory_tree',
    'check_excel_errors',
    
    # 数据验证（函数式）
    'FillStrategy',
    'check_null_values',
    'check_datetime_column',
    'fix_null_values',
    'drop_duplicates_strict',
    'fix_datetime_column',
    'validate_and_fix',
    'DataValidator',  # 向后兼容
    
    # 数据库查询（函数式）
    'build_query',
    'generate_input_pairs',
    'build_nearest_query',
    'query_nearest_after',
    'query_nearest_before',
    'query_time_range',
    'get_available_dates',
    'get_latest_date',
    'pivot_to_wide',
    'align_to_dates',
    'calculate_returns',
    
    # 数据库Excel集成（函数式）
    'process_excel_to_db_format',
    'check_existing_rows',
    'insert_dataframe',
    'get_existing_dates',
    'incremental_insert',
    'save_error_file',
    'process_directory_tree',
    
    # 配置管理
    'ConfigManager',
    'get_config',
    'reset_config',
    'get_database_config',
    'get_logging_config',
    'get_excel_config',
    'get_wind_config',
    'get_ede_config',
    
    # 辅助函数
    'func_timer',
    'get_absolute_trade_days',
    'trade_days_offset',
]

__version__ = '2.3.1'
__author__ = 'Janis'
__date__ = '2025-11-04'

