import streamlit as st
import pandas as pd
import tempfile
import os
from pathlib import Path

import psycopg2
import psycopg2.extras

from betalens import Datafeed
from betalens.datafeed.config import get_database_config
from betalens.datafeed.makeupdatabase.create_database import (
    TABLES, TABLE_DESCRIPTIONS,
    database_exists, create_database, table_exists,
    create_table, drop_table, verify_schema,
    get_create_table_sql,
)
from betalens.datafeed.ede_processor import process_ede_file
from betalens.datafeed.excel import read_file

# 查询页面用的表列表（包含用户可能手动创建的表）
ALL_TABLES = ["daily_market", "daily_index", "daily_fund", "daily_bond",
              "fundamentals", "macro", "factors"]

# 查询行数上限，防止大表卡顿
MAX_QUERY_ROWS = 50000


def render():
    st.header("数据库管理")
    tab_manage, tab_upload, tab_query = st.tabs(
        ["🛠️ 数据库管理", "📤 数据上传", "🔍 数据查询"]
    )
    with tab_manage:
        _render_manage_tab()
    with tab_upload:
        _render_upload_tab()
    with tab_query:
        _render_query_tab()


# =====================================================================
# 标签页 1：数据库管理
# =====================================================================

def _render_manage_tab():
    db_config = get_database_config()
    st.info(
        f"**连接信息**　主机: `{db_config['host']}:{db_config['port']}`　"
        f"数据库: `{db_config['dbname']}`　用户: `{db_config['user']}`"
    )

    col_check, col_create_db = st.columns(2)
    with col_check:
        if st.button("检查数据库状态", key="dbm_check"):
            with st.spinner("正在检查..."):
                try:
                    db_exists = database_exists(db_config, db_config['dbname'])
                    if db_exists:
                        schema = verify_schema(db_config)
                        st.session_state["dbm_schema"] = schema
                    else:
                        st.session_state["dbm_schema"] = None
                        st.warning(f"数据库 `{db_config['dbname']}` 不存在")
                except Exception as e:
                    st.error(f"检查失败: {e}")

    with col_create_db:
        if st.button("创建数据库", key="dbm_create_db"):
            with st.spinner("正在创建数据库..."):
                try:
                    if database_exists(db_config, db_config['dbname']):
                        st.info("数据库已存在，无需创建")
                    else:
                        ok = create_database(db_config, db_config['dbname'])
                        if ok:
                            st.success(f"数据库 `{db_config['dbname']}` 创建成功")
                        else:
                            st.error("数据库创建失败")
                except Exception as e:
                    st.error(f"创建失败: {e}")

    # 展示 schema 验证结果
    schema = st.session_state.get("dbm_schema")
    if schema:
        if schema.get("errors"):
            st.error(f"验证错误: {schema['errors']}")
        else:
            for tbl_name in TABLES:
                info = schema["tables"].get(tbl_name, {})
                exists = info.get("exists", False)
                icon = "✅" if exists else "❌"
                desc = TABLE_DESCRIPTIONS.get(tbl_name, "")
                with st.expander(f"{icon} {tbl_name}　—　{desc}", expanded=False):
                    if exists:
                        if info["columns"]:
                            st.caption("列信息")
                            st.dataframe(
                                pd.DataFrame(info["columns"]),
                                width="stretch", hide_index=True,
                            )
                        if info["indexes"]:
                            st.caption("索引")
                            st.write(", ".join(info["indexes"]))
                        if info["constraints"]:
                            con_type_map = {'u': 'UNIQUE', 'p': 'PRIMARY KEY',
                                            'f': 'FOREIGN KEY', 'c': 'CHECK'}
                            st.caption("约束")
                            for con in info["constraints"]:
                                ct = con_type_map.get(con['contype'], con['contype'])
                                st.write(f"- {con['conname']}: {ct}")
                    else:
                        st.write("表不存在")

    # 表操作区
    st.divider()
    st.subheader("表操作")
    col_sel, col_ops = st.columns([1, 3])
    with col_sel:
        sel_table = st.selectbox("选择表", TABLES, key="dbm_sel_table")
    with col_ops:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("查看建表SQL语句", key="dbm_ddl"):
                st.code(get_create_table_sql(sel_table), language="sql")
        with c2:
            if st.button("创建表", key="dbm_create_tbl"):
                try:
                    conn = psycopg2.connect(**db_config)
                    cur = conn.cursor()
                    if table_exists(cur, sel_table):
                        st.info(f"表 `{sel_table}` 已存在")
                    else:
                        create_table(cur, sel_table)
                        conn.commit()
                        st.success(f"表 `{sel_table}` 创建成功")
                    cur.close()
                    conn.close()
                except Exception as e:
                    st.error(f"创建失败: {e}")
        with c3:
            confirm_name = st.text_input(
                f"输入表名 `{sel_table}` 以确认删除",
                key="dbm_drop_confirm_name",
            )
            if st.button("删除表", key="dbm_drop_tbl",
                         disabled=confirm_name != sel_table):
                try:
                    conn = psycopg2.connect(**db_config)
                    cur = conn.cursor()
                    drop_table(cur, sel_table)
                    conn.commit()
                    cur.close()
                    conn.close()
                    st.success(f"表 `{sel_table}` 已删除")
                except Exception as e:
                    st.error(f"删除失败: {e}")


# =====================================================================
# 标签页 2：数据上传
# =====================================================================

def _save_uploaded_file(uploaded_file) -> str:
    """将 st.file_uploader 的文件保存到临时路径，返回路径"""
    suffix = Path(uploaded_file.name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return tmp.name


def _cleanup_temp(key="dbm_temp_path"):
    """清理旧的临时文件"""
    old = st.session_state.get(key)
    if old and os.path.exists(old):
        try:
            os.unlink(old)
        except OSError:
            pass


def _render_upload_tab():
    # --- 配置区 ---
    col_left, col_right = st.columns(2)
    with col_left:
        fmt = st.radio(
            "文件格式",
            ["EDE 格式（Wind导出）", "普通 CSV/Excel"],
            horizontal=True, key="dbm_fmt",
        )
        target_table = st.selectbox("目标数据表", ALL_TABLES, key="dbm_target")
    with col_right:
        insert_mode = st.radio(
            "插入模式",
            ["增量插入", "冲突检查插入", "更新插入(upsert)"],
            key="dbm_insert_mode",
        )
        if fmt == "EDE 格式（Wind导出）":
            date_source = st.radio(
                "日期来源",
                ["从文件名提取", "从列名提取", "手动指定"],
                key="dbm_date_src",
            )
            manual_dt = None
            if date_source == "手动指定":
                dc1, dc2 = st.columns(2)
                with dc1:
                    manual_date = st.date_input("日期", key="dbm_manual_date")
                with dc2:
                    manual_time = st.time_input("时间", key="dbm_manual_time")
                manual_dt = f"{manual_date} {manual_time}"

    # --- 文件上传 ---
    uploaded = st.file_uploader(
        "上传文件", type=["xlsx", "xls", "csv"],
        key="dbm_uploader",
    )

    if uploaded is None:
        # 清理状态
        for k in ["dbm_raw_df", "dbm_converted_df", "dbm_convert_errors",
                   "dbm_upload_result"]:
            st.session_state.pop(k, None)
        _cleanup_temp()
        return

    # 保存临时文件
    if st.session_state.get("dbm_temp_name") != uploaded.name:
        _cleanup_temp()
        tmp_path = _save_uploaded_file(uploaded)
        st.session_state["dbm_temp_path"] = tmp_path
        st.session_state["dbm_temp_name"] = uploaded.name
        # 清除旧的预览
        for k in ["dbm_raw_df", "dbm_converted_df", "dbm_convert_errors",
                   "dbm_upload_result"]:
            st.session_state.pop(k, None)

    tmp_path = st.session_state.get("dbm_temp_path")
    if not tmp_path:
        return

    # --- 原始数据预览 ---
    if "dbm_raw_df" not in st.session_state:
        try:
            raw_df = read_file(tmp_path)
            st.session_state["dbm_raw_df"] = raw_df
        except Exception as e:
            st.error(f"读取文件失败: {e}")
            return

    raw_df = st.session_state["dbm_raw_df"]
    st.subheader("📋 原始数据预览")
    prev_col, stat_col = st.columns([3, 2])
    with prev_col:
        st.dataframe(raw_df.head(20), width="stretch", hide_index=True)
    with stat_col:
        st.metric("行数", raw_df.shape[0])
        st.metric("列数", raw_df.shape[1])
        st.caption("列名")
        st.write(", ".join(raw_df.columns.tolist()))
        null_counts = raw_df.isnull().sum()
        null_counts = null_counts[null_counts > 0]
        if not null_counts.empty:
            st.caption("空值统计")
            st.dataframe(null_counts.rename("空值数"), width="stretch")

    # --- 转换预览 ---
    if st.button("预览转换结果", key="dbm_preview_convert"):
        with st.spinner("正在转换..."):
            try:
                if fmt == "EDE 格式（Wind导出）":
                    date_from_map = {
                        "从文件名提取": "filename",
                        "从列名提取": "metric",
                        "手动指定": "filename",
                    }
                    date_from = date_from_map.get(date_source, "filename")
                    default_dt = manual_dt if date_source == "手动指定" else None
                    converted_df, errors = process_ede_file(
                        tmp_path,
                        date_from=date_from,
                        default_datetime=default_dt,
                    )
                    st.session_state["dbm_converted_df"] = converted_df
                    st.session_state["dbm_convert_errors"] = errors
                else:
                    # 普通格式：直接读取，假设已经是长格式或需要用户确认列映射
                    converted_df = raw_df.copy()
                    st.session_state["dbm_converted_df"] = converted_df
                    st.session_state["dbm_convert_errors"] = []
            except Exception as e:
                st.error(f"转换失败: {e}")

    converted_df = st.session_state.get("dbm_converted_df")
    convert_errors = st.session_state.get("dbm_convert_errors", [])

    if converted_df is not None:
        st.subheader("🔄 转换结果预览")
        st.dataframe(converted_df.head(30), width="stretch", hide_index=True)

        # 统计卡片
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("总行数", len(converted_df))
        if "code" in converted_df.columns:
            mc2.metric("唯一代码数", converted_df["code"].nunique())
        if "metric" in converted_df.columns:
            mc3.metric("唯一指标数", converted_df["metric"].nunique())
        if "datetime" in converted_df.columns:
            dt_col = pd.to_datetime(converted_df["datetime"], errors="coerce")
            mc4.metric("日期范围", f"{dt_col.min():%Y-%m-%d} ~ {dt_col.max():%Y-%m-%d}")

        if convert_errors:
            with st.expander(f"⚠️ 转换警告 ({len(convert_errors)} 条)", expanded=False):
                for err in convert_errors:
                    st.warning(f"[{err.get('type', '')}] {err.get('message', '')} "
                               f"{err.get('column', '')}")

    # --- 普通格式列映射 ---
    if fmt == "普通 CSV/Excel" and converted_df is not None:
        required = {"datetime", "code", "name", "metric", "value"}
        existing = set(converted_df.columns)
        missing = required - existing
        if missing:
            st.subheader("列映射")
            st.caption(f"目标列: datetime, code, name, metric, value。缺少: {', '.join(missing)}")
            mapping = {}
            cols_list = converted_df.columns.tolist()
            for col_need in sorted(missing):
                mapping[col_need] = st.selectbox(
                    f"映射到 `{col_need}`", ["(不映射)"] + cols_list,
                    key=f"dbm_map_{col_need}",
                )
            if st.button("应用映射", key="dbm_apply_map"):
                df_mapped = converted_df.copy()
                for target, source in mapping.items():
                    if source != "(不映射)" and source in df_mapped.columns:
                        df_mapped = df_mapped.rename(columns={source: target})
                st.session_state["dbm_converted_df"] = df_mapped
                st.rerun()

    # --- 确认上传 ---
    if converted_df is not None:
        st.divider()
        if st.button("✅ 确认上传到数据库", key="dbm_do_upload", type="primary"):
            _do_upload(converted_df, target_table, insert_mode, fmt, tmp_path)


def _do_upload(converted_df, target_table, insert_mode, fmt, tmp_path):
    """执行数据上传"""
    with st.status("上传中...", expanded=True) as status:
        try:
            st.write(f"正在连接数据库，目标表: `{target_table}`")
            df_feed = Datafeed(target_table)

            if fmt == "EDE 格式（Wind导出）" and converted_df is not None:
                # EDE 格式：直接用转换好的 DataFrame 插入
                n_rows = len(converted_df)
                st.write(f"正在插入 {n_rows} 行数据...")

                if insert_mode == "增量插入":
                    new_rows, skipped = df_feed.incremental_update(converted_df)
                    st.session_state["dbm_upload_result"] = {
                        "new": new_rows, "skipped": skipped, "conflicts": []
                    }
                elif insert_mode == "冲突检查插入":
                    new_rows, skipped, conflicts = df_feed.insert_with_conflict_check(
                        converted_df
                    )
                    st.session_state["dbm_upload_result"] = {
                        "new": new_rows, "skipped": skipped, "conflicts": conflicts
                    }
                else:  # 更新插入
                    updated = df_feed.update_data(converted_df)
                    st.session_state["dbm_upload_result"] = {
                        "new": updated, "skipped": 0, "conflicts": []
                    }

            else:
                # 普通格式
                n_rows = len(converted_df)
                st.write(f"正在插入 {n_rows} 行数据...")

                if insert_mode == "增量插入":
                    new_rows, skipped = df_feed.incremental_update(converted_df)
                    st.session_state["dbm_upload_result"] = {
                        "new": new_rows, "skipped": skipped, "conflicts": []
                    }
                elif insert_mode == "冲突检查插入":
                    new_rows, skipped, conflicts = df_feed.insert_with_conflict_check(
                        converted_df
                    )
                    st.session_state["dbm_upload_result"] = {
                        "new": new_rows, "skipped": skipped, "conflicts": conflicts
                    }
                else:
                    updated = df_feed.update_data(converted_df)
                    st.session_state["dbm_upload_result"] = {
                        "new": updated, "skipped": 0, "conflicts": []
                    }

            df_feed.close()
            status.update(label="上传完成", state="complete")

        except Exception as e:
            status.update(label="上传失败", state="error")
            st.error(f"上传失败: {e}")
            return

    # 展示结果
    result = st.session_state.get("dbm_upload_result", {})
    rc1, rc2, rc3 = st.columns(3)
    rc1.metric("新增行数", result.get("new", 0))
    rc2.metric("跳过行数", result.get("skipped", 0))
    rc3.metric("冲突数", len(result.get("conflicts", [])))

    conflicts = result.get("conflicts", [])
    if conflicts:
        st.subheader("⚠️ 冲突详情")
        st.dataframe(pd.DataFrame(conflicts), width="stretch", hide_index=True)

    if result.get("new", 0) > 0:
        st.success(f"成功插入 {result['new']} 行数据")


# =====================================================================
# 标签页 3：数据查询
# =====================================================================

def _render_query_tab():
    # 表元数据概览
    st.subheader("表概览")
    tbl = st.selectbox("选择表", ALL_TABLES, key="dbm_q_table")
    if st.button("查看表信息", key="dbm_q_info"):
        try:
            df_feed = Datafeed(tbl)
            df_feed.cursor.execute(f"SELECT COUNT(*) as cnt FROM {tbl}")
            row = df_feed.cursor.fetchone()
            count = row["cnt"] if row else 0
            df_feed.cursor.execute(
                f"SELECT MIN(datetime) as min_dt, MAX(datetime) as max_dt FROM {tbl}"
            )
            date_range = df_feed.cursor.fetchone() or {}
            df_feed.cursor.execute(
                f"SELECT COUNT(DISTINCT code) as n_code, "
                f"COUNT(DISTINCT metric) as n_metric FROM {tbl}"
            )
            distinct = df_feed.cursor.fetchone() or {}
            df_feed.close()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("总行数", f"{count:,}")
            min_dt = date_range["min_dt"]
            max_dt = date_range["max_dt"]
            c2.metric("最早日期", str(min_dt)[:10] if min_dt else "-")
            c3.metric("最新日期", str(max_dt)[:10] if max_dt else "-")
            c4.metric("代码数 / 指标数",
                       f"{distinct['n_code']} / {distinct['n_metric']}")
        except Exception as e:
            st.error(f"查询失败: {e}")

    st.divider()

    # 查询功能
    mode = st.radio("查询方式",
                     ["时间范围查询", "最近时点(After)", "最近时点(Before)",
                      "可用日期查询", "SQL 直接查询"],
                     horizontal=True, key="dbm_q_mode")

    if mode == "时间范围查询":
        col1, col2 = st.columns(2)
        with col1:
            q_table = st.selectbox("数据表", ALL_TABLES, key="dbm_q_tbl2")
            codes_input = st.text_input(
                "标的代码（逗号分隔，留空查全部）",
                value="000300.SH", key="dbm_q_codes",
            )
        with col2:
            start = st.date_input("开始日期", value="2024-01-01",key="dbm_q_start")
            end = st.date_input("结束日期", value="2024-05-21", key="dbm_q_end")
            metric = st.text_input("指标（留空查所有列）",
                                    value="收盘价", key="dbm_q_metric")
        limit = st.number_input(
            "最大返回行数", min_value=100, max_value=MAX_QUERY_ROWS,
            value=5000, step=1000, key="dbm_q_limit",
        )

        bc1, bc2 = st.columns(2)
        btn_query = bc1.button("查询", key="dbm_q_exec")
        btn_full = bc2.button("全量查询（忽略行数限制）", key="dbm_q_exec_full")

        if btn_query or btn_full:
            codes = [c.strip() for c in codes_input.split(",") if c.strip()] or None
            metric_val = metric.strip() or None
            query_limit = None if btn_full else limit
            try:
                df_feed = Datafeed(q_table)
                result = df_feed.query_time_range(
                    codes=codes,
                    start_date=str(start),
                    end_date=str(end),
                    metric=metric_val,
                    limit=query_limit,
                )
                df_feed.close()
                total = len(result)
                st.success(f"共 {total:,} 行")
                st.dataframe(result, width="stretch")
            except Exception as e:
                st.error(f"查询失败: {e}")

    elif mode == "最近时点(After)":
        _render_nearest_query("after")

    elif mode == "最近时点(Before)":
        _render_nearest_query("before")

    elif mode == "可用日期查询":
        _render_available_dates_query()

    else:  # SQL 直接查询
        sql = st.text_area(
            "SQL 语句", height=120,
            placeholder="SELECT * FROM daily_market WHERE code = '000905.SH' LIMIT 100",
            key="dbm_q_sql",
        )
        conn_table = st.selectbox("连接到数据表（用于建立连接）",
                                   ALL_TABLES, key="dbm_q_conn_tbl")
        sql_limit = st.number_input(
            "最大返回行数", min_value=100, max_value=MAX_QUERY_ROWS,
            value=5000, step=1000, key="dbm_q_sql_limit",
        )
        bc1, bc2 = st.columns(2)
        btn_exec = bc1.button("执行", key="dbm_q_sql_exec")
        btn_full = bc2.button("全量查询（忽略行数限制）", key="dbm_q_sql_exec_full")

        if btn_exec or btn_full:
            if not sql.strip():
                st.warning("请输入 SQL 语句")
                return
            try:
                df_feed = Datafeed(conn_table)
                df_feed.cursor.execute(sql)
                if btn_full:
                    rows = df_feed.cursor.fetchall()
                else:
                    rows = df_feed.cursor.fetchmany(sql_limit + 1)
                df_feed.close()
                result = pd.DataFrame(rows)
                total = len(result)
                truncated = not btn_full and total > sql_limit
                if truncated:
                    result = result.head(sql_limit)
                    st.warning(f"结果超过 {sql_limit:,} 行，已截断显示")
                st.success(f"返回 {len(result):,} 行" + ("（已截断）" if truncated else ""))
                st.dataframe(result, width="stretch")
            except Exception as e:
                st.error(f"执行失败: {e}")


# =====================================================================
# 最近时点查询 (After / Before)
# =====================================================================

def _render_nearest_query(direction: str):
    """渲染 query_nearest_after / query_nearest_before 的 UI"""
    label = "之后" if direction == "after" else "之前"
    key_pfx = f"dbm_q_n{direction[0]}"  # dbm_q_na / dbm_q_nb

    col1, col2 = st.columns(2)
    with col1:
        n_table = st.selectbox("数据表", ALL_TABLES, key=f"{key_pfx}_tbl")
        codes_input = st.text_input(
            "标的代码（逗号分隔）",
            placeholder="000300.SH", key=f"{key_pfx}_codes",
        )
        metric = st.text_input("指标", placeholder="收盘价(元)", key=f"{key_pfx}_metric")
    with col2:
        datetimes_input = st.text_area(
            f"时间戳列表（每行一个，查找{label}最近值）",
            placeholder="2024-01-02 15:00:00\n2024-01-03 15:00:00",
            height=120, key=f"{key_pfx}_dts",
        )
        tolerance = st.number_input(
            "时间容差（小时，0 = 不限制）",
            min_value=0, value=0, step=1, key=f"{key_pfx}_tol",
        )
    limit = st.number_input(
        "最大返回行数", min_value=100, max_value=MAX_QUERY_ROWS,
        value=5000, step=1000, key=f"{key_pfx}_limit",
    )

    bc1, bc2 = st.columns(2)
    btn_query = bc1.button("查询", key=f"{key_pfx}_exec")
    btn_full = bc2.button("全量查询（忽略行数限制）", key=f"{key_pfx}_exec_full")

    if btn_query or btn_full:
        codes = [c.strip() for c in codes_input.split(",") if c.strip()]
        datetimes = [d.strip() for d in datetimes_input.strip().splitlines() if d.strip()]
        metric_val = metric.strip()

        if not codes:
            st.warning("请输入至少一个标的代码")
            return
        if not datetimes:
            st.warning("请输入至少一个时间戳")
            return
        if not metric_val:
            st.warning("请输入指标名称")
            return

        params = {
            "codes": codes,
            "datetimes": datetimes,
            "metric": metric_val,
        }
        if tolerance > 0:
            params["time_tolerance"] = tolerance

        try:
            df_feed = Datafeed(n_table)
            if direction == "after":
                result = df_feed.query_nearest_after(params)
            else:
                result = df_feed.query_nearest_before(params)
            df_feed.close()

            total = len(result)
            if not btn_full and total > limit:
                st.warning(f"查询结果共 {total:,} 行，仅显示前 {limit:,} 行")
                result = result.head(limit)
            st.success(f"共 {total:,} 行" + (f"（显示 {limit:,}）" if not btn_full and total > limit else ""))
            st.dataframe(result, width="stretch")
        except Exception as e:
            st.error(f"查询失败: {e}")


# =====================================================================
# 可用日期查询
# =====================================================================

def _render_available_dates_query():
    col1, col2 = st.columns(2)
    with col1:
        ad_table = st.selectbox("数据表", ALL_TABLES, key="dbm_q_ad_tbl")
        code = st.text_input("标的代码（单个）", placeholder="000300.SH",
                              key="dbm_q_ad_code")
    with col2:
        metric = st.text_input("指标", placeholder="收盘价(元)",
                                key="dbm_q_ad_metric")
        ad_start = st.date_input("开始日期（可选）", value=None, key="dbm_q_ad_start")
        ad_end = st.date_input("结束日期（可选）", value=None, key="dbm_q_ad_end")

    if st.button("查询", key="dbm_q_ad_exec"):
        code_val = code.strip()
        metric_val = metric.strip()
        if not code_val:
            st.warning("请输入标的代码")
            return
        if not metric_val:
            st.warning("请输入指标名称")
            return

        try:
            df_feed = Datafeed(ad_table)
            kwargs = {"code": code_val, "metric": metric_val}
            if ad_start:
                kwargs["start_date"] = str(ad_start)
            if ad_end:
                kwargs["end_date"] = str(ad_end)
            dates = df_feed.get_available_dates(**kwargs)
            df_feed.close()

            st.metric("可用日期数", len(dates))
            if dates:
                dates_df = pd.DataFrame({"datetime": dates})
                st.dataframe(dates_df, width="stretch", hide_index=True)
            else:
                st.info("未找到匹配的日期")
        except Exception as e:
            st.error(f"查询失败: {e}")
