import streamlit as st
import pandas as pd
from betalens import Datafeed

# 常用表名列表（可按需扩展）
TABLES = ["daily_market", "daily_index", "daily_fund", "daily_bond"]


def render():
    st.header("数据库查询")

    mode = st.radio("查询方式", ["结构化查询", "SQL 直接查询"], horizontal=True)

    if mode == "结构化查询":
        col1, col2 = st.columns(2)
        with col1:
            table = st.selectbox("数据表", TABLES)
            codes_input = st.text_input("标的代码（逗号分隔，留空查全部）", placeholder="000905.SH,000300.SH")
        with col2:
            start = st.date_input("开始日期")
            end = st.date_input("结束日期")
            metric = st.text_input("指标（留空查所有列）", placeholder="收盘价")

        if st.button("查询", key="struct_query"):
            codes = [c.strip() for c in codes_input.split(",") if c.strip()] or None
            metric_val = metric.strip() or None
            try:
                df = Datafeed(table)
                result = df.query_time_range(
                    codes=codes,
                    start_date=str(start),
                    end_date=str(end),
                    metric=metric_val,
                )
                df.close()
                st.success(f"共 {len(result)} 行")
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"查询失败: {e}")

    else:  # SQL 直接查询
        sql = st.text_area(
            "SQL 语句",
            height=120,
            placeholder="SELECT * FROM daily_market WHERE code = '000905.SH' LIMIT 100",
        )
        table_for_conn = st.selectbox("连接到数据表（用于建立连接）", TABLES)

        if st.button("执行", key="sql_query"):
            if not sql.strip():
                st.warning("请输入 SQL 语句")
                return
            try:
                df = Datafeed(table_for_conn)
                df.cursor.execute(sql)
                rows = df.cursor.fetchall()
                df.close()
                result = pd.DataFrame(rows)
                st.success(f"共 {len(result)} 行")
                st.dataframe(result, use_container_width=True)
            except Exception as e:
                st.error(f"执行失败: {e}")
