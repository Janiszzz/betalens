#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指数历史股票池入库脚本
作者: Janis
日期: 2026-06-03
描述: 把指数成分进出记录（宽表快照）写入 index_universe 表

输入 Excel 为宽表：
    - 第一列为序号（将被丢弃）
    - 其余每列列名为生效日期（如 2007-01-15），列下方为该日成分股 WindCode
每个生效日整理成一行长格式记录：
    datetime=生效日, code=指数代码, name=指数名, metric='universe',
    value=成分股数量, remark={index_code, index_name, constituents:[...]}

使用方法:
    python load_index_universe.py
    python load_index_universe.py --index-code 000906.SH --index-name 中证800 \
        --excel <路径> --sheet Sheet2
"""

import sys
import argparse
import logging
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
import psycopg2.extensions

# 添加父目录到路径，以便导入 datafeed 模块
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from datafeed.config import get_database_config
from datafeed.integration import incremental_insert

# remark 是 python dict，execute_values 入库前必须注册 Json 适配器
psycopg2.extensions.register_adapter(dict, psycopg2.extras.Json)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('IndexUniverseLoader')

# 默认指向中证800 的成分进出记录
DEFAULT_EXCEL = (Path(__file__).parent.parent.parent.parent
                 / 'test' / 'universe' / '000906.SH-成分进出记录-20260603.xlsx')


def build_universe_records(
    excel_path: str,
    index_code: str,
    index_name: str,
    sheet_name: str = 'Sheet2',
    seq_col: str = '序号',
) -> pd.DataFrame:
    """
    读取宽表 Excel，整理成可直接 incremental_insert 的长格式（每生效日一行）。

    Args:
        excel_path: Excel 文件路径
        index_code: 指数代码，写入 code
        index_name: 指数名称，写入 name
        sheet_name: 工作表名
        seq_col: 序号列名（将被丢弃）

    Returns:
        长格式 DataFrame（datetime, code, name, metric, value, remark）
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    logger.info(f"读取 {excel_path} [{sheet_name}]: {df.shape[0]}行 × {df.shape[1]}列")

    # 丢弃序号列，剩余列均为生效日期
    date_cols = [c for c in df.columns if c != seq_col]
    logger.info(f"识别到 {len(date_cols)} 个生效日期列")

    records = []
    for col in date_cols:
        eff_dt = pd.to_datetime(col)
        # 该列非空、去重并保序的成分股代码
        codes = [str(x).strip() for x in df[col].tolist() if pd.notna(x)]
        codes = list(dict.fromkeys(codes))  # 去重保序
        if not codes:
            logger.warning(f"生效日 {eff_dt.date()} 无有效成分股，跳过")
            continue
        records.append({
            'datetime': eff_dt,
            'code': index_code,
            'name': index_name,
            'metric': 'universe',
            'value': len(codes),
            'remark': {
                'index_code': index_code,
                'index_name': index_name,
                'constituents': codes,
            },
        })

    out = pd.DataFrame(
        records,
        columns=['datetime', 'code', 'name', 'metric', 'value', 'remark']
    )
    logger.info(f"整理出 {len(out)} 条生效日记录")
    return out


def main():
    parser = argparse.ArgumentParser(description='指数历史股票池入库')
    parser.add_argument('--index-code', default='000906.SH', help='指数代码')
    parser.add_argument('--index-name', default='中证800', help='指数名称')
    parser.add_argument('--excel', default=str(DEFAULT_EXCEL), help='Excel 路径')
    parser.add_argument('--sheet', default='Sheet2', help='工作表名')
    parser.add_argument('--table', default='index_universe', help='目标表名')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("指数历史股票池入库脚本")
    logger.info(f"指数: {args.index_code} ({args.index_name})")
    logger.info(f"文件: {args.excel} [{args.sheet}]")
    logger.info("=" * 70)

    # 整理记录
    try:
        df = build_universe_records(
            args.excel, args.index_code, args.index_name, sheet_name=args.sheet)
    except Exception as e:
        logger.error(f"读取/整理 Excel 失败: {str(e)}")
        sys.exit(1)

    if df.empty:
        logger.warning("无记录可入库，退出")
        return

    # 入库
    db_config = get_database_config()
    try:
        with psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        ) as conn:
            with conn.cursor() as cursor:
                new_rows, dup_rows = incremental_insert(
                    cursor, conn, df, args.table, logger=logger)
                logger.info("=" * 70)
                logger.info(f"✓ 入库完成: 新增 {new_rows} 行，跳过 {dup_rows} 行重复")
                logger.info("=" * 70)
    except Exception as e:
        logger.error(f"入库失败: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
