#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库初始化脚本
作者: Janis
日期: 2026-01-10
描述: 为 datafeed 模块创建 PostgreSQL 数据库和表结构

使用方法:
    python create_database.py [--force] [--no-indexes] [--no-comments]

选项:
    --force: 强制删除已存在的表并重新创建（危险！）
    --no-indexes: 不创建索引（用于大批量数据导入前）
    --no-comments: 不添加表和列注释
"""

import sys
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import psycopg2
import psycopg2.extras
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# 导入配置管理器
# 添加父目录到路径，以便导入 datafeed 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datafeed.config import get_database_config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DatabaseCreator')


# ============================================================================
# DDL 语句定义
# ============================================================================

# 表名列表
TABLES = ['daily_market', 'fundamentals', 'macro', 'factors']

# 表说明
TABLE_DESCRIPTIONS = {
    'daily_market': '个券行情（日频）- 开盘价最早09:30，其余价量最早15:00',
    'fundamentals': '个券基本面（日频入库，事件驱动）- 按公告时点入库',
    'macro': '宏观经济数据（事件驱动）- 区分公告时点与发生时点',
    'factors': '因子库 - 存储计算好的因子数据'
}

# 列注释（所有表共享）
COLUMN_COMMENTS = {
    'datetime': '入库实际时间（最早可交易时间）',
    'code': '证券代码（WindCode格式，如000001.SZ）',
    'name': '证券中文名称',
    'metric': '指标名称（如：收盘价(元)、成交量(股)）',
    'value': '数值',
    'remark': '备注信息（JSON格式）'
}

# fundamentals表特殊列注释
FUNDAMENTALS_COLUMN_COMMENTS = {
    'datetime': '入库实际时间（最早可交易时间）',
    'remark': '备注信息，可包含理论发生时间（报告期：0331/0630/0930/1231）'
}

# macro表特殊列注释
MACRO_COLUMN_COMMENTS = {
    'code': '指标代码（WindCode格式）',
    'name': '指标名称',
    'remark': '备注信息，可包含理论发生时间（如"2024年1月GDP"）'
}

# factors表特殊列注释
FACTORS_COLUMN_COMMENTS = {
    'metric': '因子名称/数据编制方式',
    'remark': '备注信息，可包含因子计算参数和元数据'
}


def get_create_table_sql(table_name: str) -> str:
    """
    生成CREATE TABLE语句

    Args:
        table_name: 表名

    Returns:
        SQL语句
    """
    return f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        datetime TIMESTAMP NOT NULL,
        code VARCHAR(20) NOT NULL,
        name VARCHAR(100) NOT NULL,
        metric VARCHAR(100) NOT NULL,
        value NUMERIC(50, 6),
        remark JSONB,
        CONSTRAINT uq_{table_name}_datetime_code_metric
            UNIQUE (datetime, code, metric)
    );
    """


def get_create_indexes_sql(table_name: str) -> List[str]:
    """
    生成CREATE INDEX语句列表

    Args:
        table_name: 表名

    Returns:
        SQL语句列表
    """
    return [
        f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_code_metric_datetime
            ON {table_name} (code, metric, datetime);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_datetime
            ON {table_name} (datetime);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_code_datetime
            ON {table_name} (code, datetime);
        """,
        f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_metric
            ON {table_name} (metric);
        """
    ]


def get_comment_sql(table_name: str) -> List[Tuple[str, tuple]]:
    """
    生成COMMENT语句列表（使用参数化查询）

    Args:
        table_name: 表名

    Returns:
        (SQL语句模板, 参数元组) 的列表
    """
    statements = []

    # 表注释
    table_comment = TABLE_DESCRIPTIONS.get(table_name, '')
    statements.append((
        sql.SQL("COMMENT ON TABLE {} IS %s").format(sql.Identifier(table_name)),
        (table_comment,)
    ))

    # 列注释
    # 选择合适的列注释字典
    if table_name == 'fundamentals':
        comments = {**COLUMN_COMMENTS, **FUNDAMENTALS_COLUMN_COMMENTS}
    elif table_name == 'macro':
        comments = {**COLUMN_COMMENTS, **MACRO_COLUMN_COMMENTS}
    elif table_name == 'factors':
        comments = {**COLUMN_COMMENTS, **FACTORS_COLUMN_COMMENTS}
    else:
        comments = COLUMN_COMMENTS

    for column, comment in comments.items():
        statements.append((
            sql.SQL("COMMENT ON COLUMN {}.{} IS %s").format(
                sql.Identifier(table_name),
                sql.Identifier(column)
            ),
            (comment,)
        ))

    return statements


# ============================================================================
# 数据库操作函数
# ============================================================================

def validate_table_name(table_name: str) -> bool:
    """
    验证表名是否在允许的列表中（防止SQL注入）

    Args:
        table_name: 表名

    Returns:
        是否有效
    """
    return table_name in TABLES


def validate_database_name(dbname: str) -> bool:
    """
    验证数据库名（防止SQL注入）
    仅允许字母、数字、下划线

    Args:
        dbname: 数据库名

    Returns:
        是否有效
    """
    return bool(re.match(r'^[a-zA-Z0-9_]+$', dbname))


def database_exists(db_config: Dict, dbname: str) -> bool:
    """
    检查数据库是否存在

    Args:
        db_config: 数据库配置
        dbname: 数据库名

    Returns:
        是否存在
    """
    try:
        # 连接到postgres数据库（使用context manager）
        with psycopg2.connect(
            dbname='postgres',
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        ) as conn:
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            with conn.cursor() as cursor:
                # 检查数据库是否存在
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (dbname,)
                )
                exists = cursor.fetchone() is not None
                return exists

    except Exception as e:
        logger.error(f"检查数据库是否存在时出错: {str(e)}")
        return False


def create_database(db_config: Dict, dbname: str) -> bool:
    """
    创建数据库

    Args:
        db_config: 数据库配置
        dbname: 数据库名

    Returns:
        是否成功
    """
    # 验证数据库名（防止SQL注入）
    if not validate_database_name(dbname):
        logger.error(f"无效的数据库名: {dbname}（仅允许字母、数字、下划线）")
        return False

    conn = None
    try:
        # 连接到postgres数据库
        conn = psycopg2.connect(
            dbname='postgres',
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        # 设置自动提交模式（CREATE DATABASE 不能在事务块中运行）
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()
        # 创建数据库（使用psycopg2的标识符引用，避免SQL注入）
        # 使用 template0 避免 collation 版本不匹配问题
        logger.info(f"正在创建数据库: {dbname}")
        cursor.execute(
            sql.SQL("CREATE DATABASE {} ENCODING 'UTF8' TEMPLATE template0").format(
                sql.Identifier(dbname)
            )
        )
        cursor.close()
        logger.info(f"数据库 {dbname} 创建成功")
        return True

    except Exception as e:
        logger.error(f"创建数据库失败: {str(e)}")
        return False
    finally:
        if conn:
            conn.close()


def table_exists(cursor, table_name: str) -> bool:
    """
    检查表是否存在

    Args:
        cursor: 数据库游标
        table_name: 表名

    Returns:
        是否存在
    """
    cursor.execute(
        """
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name = %s
        )
        """,
        (table_name,)
    )
    return cursor.fetchone()[0]


def drop_table(cursor, table_name: str):
    """
    删除表

    Args:
        cursor: 数据库游标
        table_name: 表名
    """
    # 验证表名（防止SQL注入）
    if not validate_table_name(table_name):
        raise ValueError(f"无效的表名: {table_name}")

    logger.warning(f"正在删除表: {table_name}")
    # 使用psycopg2的标识符引用，避免SQL注入
    cursor.execute(
        sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(
            sql.Identifier(table_name)
        )
    )
    logger.info(f"表 {table_name} 已删除")


def create_table(cursor, table_name: str, create_indexes: bool = True, create_comments: bool = True):
    """
    创建表

    Args:
        cursor: 数据库游标
        table_name: 表名
        create_indexes: 是否创建索引
        create_comments: 是否添加注释
    """
    # 验证表名（防止SQL注入）
    if not validate_table_name(table_name):
        raise ValueError(f"无效的表名: {table_name}")

    # 创建表
    logger.info(f"正在创建表: {table_name}")
    sql = get_create_table_sql(table_name)
    cursor.execute(sql)
    logger.info(f"表 {table_name} 创建成功")

    # 创建索引
    if create_indexes:
        logger.info(f"正在为表 {table_name} 创建索引")
        for index_sql in get_create_indexes_sql(table_name):
            cursor.execute(index_sql)
        logger.info(f"表 {table_name} 的索引创建完成")

    # 添加注释
    if create_comments:
        logger.info(f"正在为表 {table_name} 添加注释")
        for comment_sql, params in get_comment_sql(table_name):
            cursor.execute(comment_sql, params)
        logger.info(f"表 {table_name} 的注释添加完成")


def verify_schema(db_config: Dict) -> Dict:
    """
    验证数据库schema

    Args:
        db_config: 数据库配置

    Returns:
        验证结果字典
    """
    result = {
        'database_exists': False,
        'tables': {},
        'errors': []
    }

    try:
        # 连接到数据库（使用context manager）
        with psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        ) as conn:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cursor:
                result['database_exists'] = True

                # 检查每个表
                for table_name in TABLES:
                    table_info = {
                        'exists': False,
                        'columns': [],
                        'indexes': [],
                        'constraints': []
                    }

                    # 检查表是否存在
                    if table_exists(cursor, table_name):
                        table_info['exists'] = True

                        # 获取列信息
                        cursor.execute(
                            """
                            SELECT column_name, data_type, is_nullable
                            FROM information_schema.columns
                            WHERE table_schema = 'public'
                            AND table_name = %s
                            ORDER BY ordinal_position
                            """,
                            (table_name,)
                        )
                        table_info['columns'] = [dict(row) for row in cursor.fetchall()]

                        # 获取索引信息
                        cursor.execute(
                            """
                            SELECT indexname
                            FROM pg_indexes
                            WHERE tablename = %s
                            ORDER BY indexname
                            """,
                            (table_name,)
                        )
                        table_info['indexes'] = [row['indexname'] for row in cursor.fetchall()]

                        # 获取约束信息
                        cursor.execute(
                            """
                            SELECT conname, contype
                            FROM pg_constraint
                            WHERE conrelid = %s::regclass
                            ORDER BY conname
                            """,
                            (table_name,)
                        )
                        table_info['constraints'] = [dict(row) for row in cursor.fetchall()]

                    result['tables'][table_name] = table_info

    except Exception as e:
        result['errors'].append(str(e))
        logger.error(f"验证schema时出错: {str(e)}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")

    return result


def print_verification_report(result: Dict):
    """
    打印验证报告

    Args:
        result: 验证结果
    """
    print("\n" + "="*70)
    print("数据库Schema验证报告")
    print("="*70)

    if result['errors']:
        print("\n❌ 错误:")
        for error in result['errors']:
            print(f"  - {error}")
        return

    print(f"\n✓ 数据库存在: {result['database_exists']}")

    print("\n表信息:")
    for table_name, info in result['tables'].items():
        status = "✓" if info['exists'] else "✗"
        print(f"\n  {status} {table_name}:")

        if info['exists']:
            print(f"    列数: {len(info['columns'])}")
            print(f"    索引数: {len(info['indexes'])}")
            print(f"    约束数: {len(info['constraints'])}")

            # 打印列信息
            print("\n    列定义:")
            for col in info['columns']:
                nullable = "NULL" if col['is_nullable'] == 'YES' else "NOT NULL"
                print(f"      - {col['column_name']}: {col['data_type']} {nullable}")

            # 打印索引信息
            if info['indexes']:
                print("\n    索引:")
                for idx in info['indexes']:
                    print(f"      - {idx}")

            # 打印约束信息
            if info['constraints']:
                print("\n    约束:")
                for con in info['constraints']:
                    con_type_map = {'u': 'UNIQUE', 'p': 'PRIMARY KEY', 'f': 'FOREIGN KEY', 'c': 'CHECK'}
                    con_type = con_type_map.get(con['contype'], con['contype'])
                    print(f"      - {con['conname']}: {con_type}")

    print("\n" + "="*70)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='创建 datafeed 数据库和表结构')
    parser.add_argument('--force', action='store_true',
                        help='强制删除已存在的表并重新创建（危险！）')
    parser.add_argument('--no-indexes', action='store_true',
                        help='不创建索引（用于大批量数据导入前）')
    parser.add_argument('--no-comments', action='store_true',
                        help='不添加表和列注释')
    parser.add_argument('--verify-only', action='store_true',
                        help='仅验证schema，不创建表')

    args = parser.parse_args()

    # 获取数据库配置
    try:
        db_config = get_database_config()
    except Exception as e:
        logger.error(f"读取数据库配置失败: {str(e)}")
        sys.exit(1)

    logger.info("="*70)
    logger.info("Datafeed 数据库初始化脚本")
    logger.info("="*70)
    logger.info(f"数据库: {db_config['dbname']}")
    logger.info(f"主机: {db_config['host']}:{db_config['port']}")
    logger.info(f"用户: {db_config['user']}")
    logger.info("="*70)

    # 仅验证模式
    if args.verify_only:
        logger.info("执行验证模式...")
        result = verify_schema(db_config)
        print_verification_report(result)
        return

    # 检查数据库是否存在
    if not database_exists(db_config, db_config['dbname']):
        logger.warning(f"数据库 {db_config['dbname']} 不存在")
        response = input(f"是否创建数据库 '{db_config['dbname']}'? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            if not create_database(db_config, db_config['dbname']):
                logger.error("数据库创建失败，退出")
                sys.exit(1)
        else:
            logger.info("用户取消操作，退出")
            sys.exit(0)
    else:
        logger.info(f"数据库 {db_config['dbname']} 已存在")

    # 连接到目标数据库
    try:
        # 使用context manager管理连接和事务
        with psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        ) as conn:
            with conn.cursor() as cursor:
                logger.info(f"已连接到数据库 {db_config['dbname']}")

                # 处理每个表
                for table_name in TABLES:
                    logger.info(f"\n处理表: {table_name}")

                    # 检查表是否存在
                    exists = table_exists(cursor, table_name)

                    if exists:
                        if args.force:
                            logger.warning(f"表 {table_name} 已存在，--force 模式将删除并重建")
                            response = input(f"⚠️  确认删除表 '{table_name}' 及其所有数据? (yes/no): ")
                            if response.lower() in ['yes', 'y']:
                                drop_table(cursor, table_name)
                                conn.commit()
                            else:
                                logger.info(f"跳过表 {table_name}")
                                continue
                        else:
                            logger.info(f"表 {table_name} 已存在，跳过创建（使用 --force 强制重建）")
                            continue

                    # 创建表（在事务中）
                    try:
                        create_table(
                            cursor,
                            table_name,
                            create_indexes=not args.no_indexes,
                            create_comments=not args.no_comments
                        )
                        conn.commit()
                        logger.info(f"表 {table_name} 处理完成")
                    except Exception as e:
                        logger.error(f"创建表 {table_name} 时出错: {str(e)}")
                        conn.rollback()
                        raise

        logger.info("\n" + "="*70)
        logger.info("✓ 所有表创建完成")
        logger.info("="*70)

        # 验证创建结果
        logger.info("\n正在验证创建结果...")
        result = verify_schema(db_config)
        print_verification_report(result)

    except Exception as e:
        logger.error(f"执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
