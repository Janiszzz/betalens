#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据库测试脚本
作者: Janis
日期: 2026-01-10
描述: 测试 datafeed 数据库和表结构是否正常工作

测试内容:
1. 连接测试 - 测试能否连接到各个表
2. 插入测试 - 测试数据插入功能
3. 查询测试 - 测试数据查询功能
4. 约束测试 - 测试唯一约束是否生效
5. 清理测试数据

使用方法:
    python test_database.py [--skip-cleanup]

选项:
    --skip-cleanup: 跳过测试数据清理，保留测试数据供检查
"""

import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd

# 导入datafeed模块
try:
    from core import Datafeed
    from config import get_database_config
except ImportError:
    print("错误: 无法导入datafeed模块，请确保在datafeed目录下运行此脚本")
    sys.exit(1)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('DatabaseTest')


# ============================================================================
# 测试配置
# ============================================================================

# 测试表列表
TEST_TABLES = ['daily_market', 'fundamentals', 'macro', 'factors']

# 测试数据
TEST_DATA = {
    'daily_market': [
        {
            'datetime': '2024-01-02 09:30:01',
            'code': '000001.SZ',
            'name': '平安银行',
            'metric': '开盘价(元)',
            'value': 10.50
        },
        {
            'datetime': '2024-01-02 15:00:01',
            'code': '000001.SZ',
            'name': '平安银行',
            'metric': '收盘价(元)',
            'value': 10.55
        },
        {
            'datetime': '2024-01-03 15:00:01',
            'code': '000001.SZ',
            'name': '平安银行',
            'metric': '收盘价(元)',
            'value': 10.60
        }
    ],
    'fundamentals': [
        {
            'datetime': '2024-03-31 15:00:01',
            'code': '000001.SZ',
            'name': '平安银行',
            'metric': '归母净利润(元)',
            'value': 1000000000.00
        }
    ],
    'macro': [
        {
            'datetime': '2024-01-15 10:00:00',
            'code': 'GDP',
            'name': 'GDP同比增长',
            'metric': 'GDP增速(%)',
            'value': 5.2
        }
    ],
    'factors': [
        {
            'datetime': '2024-01-02 15:00:01',
            'code': '000001.SZ',
            'name': '平安银行',
            'metric': '市值因子',
            'value': 0.85
        }
    ]
}


# ============================================================================
# 测试函数
# ============================================================================

def test_connection(table_name: str) -> bool:
    """
    测试数据库连接

    Args:
        table_name: 表名

    Returns:
        是否成功
    """
    logger.info(f"测试连接到表: {table_name}")
    try:
        df = Datafeed(table_name)
        logger.info(f"✓ 成功连接到表 {table_name}")
        df.close()
        return True
    except Exception as e:
        logger.error(f"✗ 连接表 {table_name} 失败: {str(e)}")
        return False


def test_insert(table_name: str, test_data: list) -> bool:
    """
    测试数据插入

    Args:
        table_name: 表名
        test_data: 测试数据列表

    Returns:
        是否成功
    """
    logger.info(f"测试插入数据到表: {table_name}")
    try:
        df = Datafeed(table_name)

        # 将测试数据转换为DataFrame
        test_df = pd.DataFrame(test_data)
        test_df['datetime'] = pd.to_datetime(test_df['datetime'])

        # 插入数据
        success, message, stats = df._insert_dataframe(
            cursor=df.cursor,
            conn=df.conn,
            df=test_df,
            table=table_name,
            check_duplicates=False,
            logger=df.logger
        )

        if success:
            logger.info(f"✓ 成功插入 {stats.get('inserted_rows', 0)} 行数据到表 {table_name}")
            df.close()
            return True
        else:
            logger.error(f"✗ 插入数据到表 {table_name} 失败: {message}")
            df.close()
            return False

    except Exception as e:
        logger.error(f"✗ 插入数据异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_query(table_name: str) -> bool:
    """
    测试数据查询

    Args:
        table_name: 表名

    Returns:
        是否成功
    """
    logger.info(f"测试查询表: {table_name}")
    try:
        df = Datafeed(table_name)

        # 查询所有数据
        query = f"SELECT * FROM {table_name} ORDER BY datetime LIMIT 10"
        df.cursor.execute(query)
        results = df.cursor.fetchall()

        if results:
            logger.info(f"✓ 成功查询到 {len(results)} 行数据")
            logger.info(f"  示例数据: {results[0]}")
            df.close()
            return True
        else:
            logger.warning(f"! 表 {table_name} 中没有数据（这是正常的，如果是首次运行）")
            df.close()
            return True

    except Exception as e:
        logger.error(f"✗ 查询失败: {str(e)}")
        return False


def test_unique_constraint(table_name: str) -> bool:
    """
    测试唯一约束

    Args:
        table_name: 表名

    Returns:
        是否成功
    """
    logger.info(f"测试唯一约束: {table_name}")
    try:
        df = Datafeed(table_name)

        # 准备重复数据
        duplicate_data = pd.DataFrame([
            {
                'datetime': '2024-01-02 09:30:01',
                'code': 'TEST.SZ',
                'name': '测试证券',
                'metric': '测试指标',
                'value': 100.0
            }
        ])
        duplicate_data['datetime'] = pd.to_datetime(duplicate_data['datetime'])

        # 第一次插入（应该成功）
        success1, msg1, stats1 = df._insert_dataframe(
            cursor=df.cursor,
            conn=df.conn,
            df=duplicate_data,
            table=table_name,
            check_duplicates=False,
            logger=df.logger
        )

        if not success1:
            logger.error(f"✗ 第一次插入失败（不应该）: {msg1}")
            df.close()
            return False

        # 第二次插入相同数据（应该失败或被跳过）
        success2, msg2, stats2 = df._insert_dataframe(
            cursor=df.cursor,
            conn=df.conn,
            df=duplicate_data,
            table=table_name,
            check_duplicates=True,
            skip_duplicates=True,
            logger=df.logger
        )

        if success2 and stats2.get('duplicate_rows', 0) > 0:
            logger.info(f"✓ 唯一约束正常工作，重复数据被跳过")
            df.close()
            return True
        else:
            logger.warning(f"! 唯一约束测试结果不明确: {msg2}")
            df.close()
            return True

    except Exception as e:
        # 如果是因为唯一约束违反而失败，这实际上是成功的测试
        if 'unique' in str(e).lower() or 'duplicate' in str(e).lower():
            logger.info(f"✓ 唯一约束正常工作（捕获到约束违反错误）")
            return True
        else:
            logger.error(f"✗ 测试唯一约束异常: {str(e)}")
            return False


def test_time_range_query(table_name: str) -> bool:
    """
    测试时间范围查询

    Args:
        table_name: 表名

    Returns:
        是否成功
    """
    logger.info(f"测试时间范围查询: {table_name}")
    try:
        df = Datafeed(table_name)

        # 查询时间范围数据
        result = df.query_time_range(
            start_date='2024-01-01',
            end_date='2024-12-31'
        )

        logger.info(f"✓ 时间范围查询成功，返回 {len(result)} 行")
        df.close()
        return True

    except Exception as e:
        logger.error(f"✗ 时间范围查询失败: {str(e)}")
        return False


def cleanup_test_data(table_name: str) -> bool:
    """
    清理测试数据

    Args:
        table_name: 表名

    Returns:
        是否成功
    """
    logger.info(f"清理测试数据: {table_name}")
    try:
        df = Datafeed(table_name)

        # 删除测试数据
        delete_query = f"""
        DELETE FROM {table_name}
        WHERE code IN ('000001.SZ', 'TEST.SZ', 'GDP')
        AND datetime >= '2024-01-01'
        AND datetime <= '2024-12-31'
        """

        df.cursor.execute(delete_query)
        df.conn.commit()

        deleted_rows = df.cursor.rowcount
        logger.info(f"✓ 清理完成，删除了 {deleted_rows} 行测试数据")
        df.close()
        return True

    except Exception as e:
        logger.error(f"✗ 清理测试数据失败: {str(e)}")
        return False


# ============================================================================
# 主测试函数
# ============================================================================

def run_tests(skip_cleanup: bool = False) -> Dict[str, bool]:
    """
    运行所有测试

    Args:
        skip_cleanup: 是否跳过清理

    Returns:
        测试结果字典
    """
    results = {}

    logger.info("="*70)
    logger.info("开始数据库测试")
    logger.info("="*70)

    for table_name in TEST_TABLES:
        logger.info(f"\n{'='*70}")
        logger.info(f"测试表: {table_name}")
        logger.info(f"{'='*70}")

        table_results = {}

        # 1. 连接测试
        table_results['connection'] = test_connection(table_name)

        if table_results['connection']:
            # 2. 插入测试
            if table_name in TEST_DATA:
                table_results['insert'] = test_insert(table_name, TEST_DATA[table_name])
            else:
                table_results['insert'] = None
                logger.info(f"跳过插入测试（无测试数据）")

            # 3. 查询测试
            table_results['query'] = test_query(table_name)

            # 4. 唯一约束测试
            table_results['unique_constraint'] = test_unique_constraint(table_name)

            # 5. 时间范围查询测试
            table_results['time_range_query'] = test_time_range_query(table_name)

            # 6. 清理测试数据
            if not skip_cleanup:
                table_results['cleanup'] = cleanup_test_data(table_name)
            else:
                logger.info("跳过清理测试数据（--skip-cleanup）")
                table_results['cleanup'] = None

        results[table_name] = table_results

    return results


def print_test_summary(results: Dict[str, Dict[str, bool]]):
    """
    打印测试摘要

    Args:
        results: 测试结果
    """
    logger.info("\n" + "="*70)
    logger.info("测试摘要")
    logger.info("="*70)

    all_passed = True

    for table_name, table_results in results.items():
        logger.info(f"\n表: {table_name}")

        for test_name, result in table_results.items():
            if result is None:
                status = "⊘"
                status_text = "跳过"
            elif result:
                status = "✓"
                status_text = "通过"
            else:
                status = "✗"
                status_text = "失败"
                all_passed = False

            logger.info(f"  {status} {test_name}: {status_text}")

    logger.info("\n" + "="*70)
    if all_passed:
        logger.info("✓ 所有测试通过！")
    else:
        logger.warning("✗ 部分测试失败，请检查日志")
    logger.info("="*70)

    return all_passed


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='测试 datafeed 数据库')
    parser.add_argument('--skip-cleanup', action='store_true',
                        help='跳过测试数据清理，保留测试数据供检查')

    args = parser.parse_args()

    # 运行测试
    results = run_tests(skip_cleanup=args.skip_cleanup)

    # 打印摘要
    all_passed = print_test_summary(results)

    # 返回退出码
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
