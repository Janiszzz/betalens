#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
个券交易状态入库脚本（稀疏存储）
作者: Janis
日期: 2026-06-06
描述: 把交易状态宽表快照写入 trade_status 表

输入 Excel 为宽表（见 test/trade_status/）：
    - 第 0 行：股票简称
    - 第 1 行：Date + WindCode（如 000001.SZ）
    - 第 2 行起：首列为日期，其余每格为该股该日交易状态文本（NaN=非交易日）

稀疏存储策略（节省空间）：
    - 异常状态（停牌一天/暂停上市/...）：入库 value=0，remark.status 存文本
    - 首次正常交易日：入库 value=1，remark.first_normal=true 作为锚点
    - 首次正常之后的正常交易日：不入库（查询时推断为正常）
    - NaN（节假日/周末）：不入库
    - 首次正常之前：不入库（查询时推断为 -1 无法交易）

使用方法:
    python load_trade_status.py
    python load_trade_status.py --excel <路径> --sheet Sheet1 --table trade_status
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
logger = logging.getLogger('TradeStatusLoader')

# 正常交易状态文本（其余均视为异常）
NORMAL_STATUS = '交易'

DEFAULT_EXCEL = (Path(__file__).parent.parent.parent.parent
                 / 'test' / 'trade_status' / '交易状态0506-1506-快照1.xlsx')


def build_trade_status_records(
    excel_path: str,
    sheet_name: str = 'Sheet1',
) -> pd.DataFrame:
    """
    读取交易状态宽表，整理成稀疏长格式（仅异常状态 + 首次正常交易日）。

    宽表结构：第0行=简称，第1行=Date+WindCode，第2行起=日期+状态文本。

    Args:
        excel_path: Excel 文件路径
        sheet_name: 工作表名

    Returns:
        长格式 DataFrame（datetime, code, name, metric, value, remark）
    """
    # header=1 用第二行（Date + WindCode）作列名；names 行单独读取
    names_row = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, nrows=1)
    df = pd.read_excel(excel_path, sheet_name=sheet_name, header=1)
    logger.info(f"读取 {excel_path} [{sheet_name}]: {df.shape[0]}行 × {df.shape[1]}列")

    date_col = df.columns[0]  # 'Date'
    code_cols = list(df.columns[1:])
    # 列序对齐简称（names_row 第0列为'简称'占位，与 Date 对齐）
    name_map = {code: names_row.iloc[0, i + 1] for i, code in enumerate(code_cols)}

    # 过滤无效列：列名为 NaN 或 'Unnamed'（Excel 尾部空列）
    valid_code_cols = [
        c for c in code_cols
        if pd.notna(c) and not str(c).startswith('Unnamed')
    ]
    dropped = len(code_cols) - len(valid_code_cols)
    if dropped:
        logger.warning(f"跳过 {dropped} 个无效代码列（列名为空/Unnamed）")
    code_cols = valid_code_cols

    dates = pd.to_datetime(df[date_col])
    records = []
    for code in code_cols:
        name = name_map.get(code)
        col = df[code]
        first_normal_done = False
        for idx, status in col.items():
            if pd.isna(status):
                continue  # 非交易日，不入库
            status = str(status).strip()
            dt = dates.iloc[idx].normalize() + pd.Timedelta(hours=15, seconds=1)
            if status == NORMAL_STATUS:
                if not first_normal_done:
                    # 首次正常交易日：入库做锚点
                    records.append({
                        'datetime': dt, 'code': code, 'name': name,
                        'metric': '交易状态', 'value': 1,
                        'remark': {'status': status, 'first_normal': True},
                    })
                    first_normal_done = True
                # 后续正常日不入库（查询时推断）
            else:
                # 异常状态：入库
                records.append({
                    'datetime': dt, 'code': code, 'name': name,
                    'metric': '交易状态', 'value': 0,
                    'remark': {'status': status},
                })

    out = pd.DataFrame(
        records,
        columns=['datetime', 'code', 'name', 'metric', 'value', 'remark']
    )
    logger.info(f"整理出 {len(out)} 条稀疏记录（异常 + 首次正常锚点）")
    return out


def main():
    parser = argparse.ArgumentParser(description='个券交易状态入库')
    parser.add_argument('--excel', default=str(DEFAULT_EXCEL), help='Excel 路径')
    parser.add_argument('--sheet', default='Sheet1', help='工作表名')
    parser.add_argument('--table', default='trade_status', help='目标表名')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("个券交易状态入库脚本（稀疏存储）")
    logger.info(f"文件: {args.excel} [{args.sheet}]")
    logger.info("=" * 70)

    try:
        df = build_trade_status_records(args.excel, sheet_name=args.sheet)
    except Exception as e:
        logger.error(f"读取/整理 Excel 失败: {str(e)}")
        sys.exit(1)

    if df.empty:
        logger.warning("无记录可入库，退出")
        return

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

