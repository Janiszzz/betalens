"""
事件研究示例脚本
1. 生成精确到秒的事件序列样例并保存到 events.xlsx
2. 对 600030.SH 进行事件研究分析
3. 收益计算确保在事件时间之后

文件输出控制：
- 默认不保存Excel和图表文件
- 如需保存，将下方 save_results 设置为 True
"""
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from datafeed import Datafeed
from eventstudy.eventstudy import EventStudy

# ========== 文件保存控制开关 ==========
save_results = False  # 设为True时保存所有Excel和图表文件
# =====================================

output_file = 'events.xlsx'
# 读取事件序列（保持精确到秒的datetime）
events_df = pd.read_excel(output_file)
events_df['date'] = pd.to_datetime(events_df['date'])
events = events_df.set_index('date')['event']
print(f"[OK] 已读取事件序列: {events.sum()}个事件（时间戳精确到秒）")
print(f"  - 示例时间戳: {events[events==1].index[0]}")

# 初始化 Datafeed 和 EventStudy
df_market = Datafeed("daily_market")


close_price_metric = "收盘价"

event_study = EventStudy(df_market)

# 事件研究参数
code = '000905.SH'
window_before = 10  # 事件前10天
window_after = 10   # 事件后10天

print(f"\n事件研究参数:")
print(f"  - 标的代码: {code}")
print(f"  - 价格指标: {repr(close_price_metric)}")
print(f"  - 事件前窗口: {window_before}天")
print(f"  - 事件后窗口: {window_after}天")

# 执行事件研究
print(f"\n正在分析...")
result = event_study.analyze(
    events=events,
    code=code,
    window_before=window_before,
    window_after=window_after,
    metric=close_price_metric
)

# ========== 步骤3: 展示分析结果 ==========
print("\n" + "=" * 60)
print("步骤3: 分析结果")
print("=" * 60)

if 'error' in result:
    print(f"[ERROR] 分析失败: {result['error']}")
else:
    print(f"[OK] 成功分析 {result['event_count']} 个事件")

    # 每日收益率统计
    print("\n【每日平均收益率统计】")
    daily_stats = result['daily_stats']
    print(daily_stats.to_string())

    # 累积收益率统计
    print("\n【累积收益率统计】")
    cumulative_stats = result['cumulative_stats']
    print(cumulative_stats.to_string())

    # 可选：保存详细结果（默认不保存）
    if save_results:
        with pd.ExcelWriter('eventstudy_results.xlsx') as writer:
            daily_stats.to_excel(writer, sheet_name='daily_stats')
            cumulative_stats.to_excel(writer, sheet_name='cumulative_stats')
            result['returns_matrix'].to_excel(writer, sheet_name='returns_matrix')
        print(f"\n[OK] 详细结果已保存到: eventstudy_results.xlsx")

    # 关键发现
    print("\n【关键发现】")

    # Day 0: 事件后首个收益点（严格在事件时间之后的第一个收益率）
    if 0 in daily_stats.index:
        day0_stats = daily_stats.loc[0]
        print(f"事件后首个收益点(Day 0):")
        print(f"  - 平均收益率: {day0_stats['mean']:.4%}")
        print(f"  - 上涨概率: {day0_stats['positive_prob']:.2%}")
        print(f"  - t统计量: {day0_stats['t_stat']:.2f}")

    # 事件后累积收益
    if window_after in cumulative_stats.index:
        final_stats = cumulative_stats.loc[window_after]
        print(f"\n事件后{window_after}天累积收益:")
        print(f"  - 平均收益率: {final_stats['mean']:.4%}")
        print(f"  - 上涨概率: {final_stats['positive_prob']:.2%}")
        print(f"  - t统计量: {final_stats['t_stat']:.2f}")

    # 【新增功能示例】偏移持有起点
    print("\n" + "=" * 60)
    print("【偏移持有起点示例】")
    print("=" * 60)
    print("比较从事件发生当天 vs 提前3天开始持有的收益差异\n")

    # 从事件发生后第3天开始持有
    result_offset = event_study.analyze(
        events=events,
        code=code,
        window_before=window_before,
        window_after=window_after,
        metric=close_price_metric,
        holding_start_offset= -3  # 从Day -3开始持有
    )

    if 'error' not in result_offset:
        daily_stats_offset = result_offset['daily_stats']
        cumulative_stats_offset = result_offset['cumulative_stats']

        print(f"[OK] 成功分析 {result_offset['event_count']} 个事件（从Day 3开始持有）")

        # 展示偏移后的每日收益率统计
        print("\n【从Day 3开始 - 每日平均收益率统计】")
        print(daily_stats_offset.to_string())

        # 展示偏移后的累积收益率统计
        print("\n【从Day 3开始 - 累积收益率统计】")
        print(cumulative_stats_offset.to_string())

        # 比较分析
        print("\n【对比分析】从Day 0开始持有 vs 从Day 3开始持有:")
        for day in [5, 10]:
            if day in cumulative_stats.index and day in cumulative_stats_offset.index:
                ret_day0 = cumulative_stats.loc[day, 'mean']
                ret_day3 = cumulative_stats_offset.loc[day, 'mean']
                print(f"\nDay {day}累积收益:")
                print(f"  - Day 0开始: {ret_day0:.4%}")
                print(f"  - Day 3开始: {ret_day3:.4%}")
                print(f"  - 差异: {ret_day0 - ret_day3:.4%}")

        # 可选：保存比较结果（默认不保存）
        if save_results:
            with pd.ExcelWriter('eventstudy_offset_comparison.xlsx') as writer:
                daily_stats.to_excel(writer, sheet_name='day0_daily')
                cumulative_stats.to_excel(writer, sheet_name='day0_cumulative')
                daily_stats_offset.to_excel(writer, sheet_name='day3_daily')
                cumulative_stats_offset.to_excel(writer, sheet_name='day3_cumulative')
                result['returns_matrix'].to_excel(writer, sheet_name='day0_returns')
                result_offset['returns_matrix'].to_excel(writer, sheet_name='day3_returns')
            print(f"\n[OK] 偏移对比结果已保存到: eventstudy_offset_comparison.xlsx")

    # 【新增功能示例】业绩比较基准模式
    print("\n" + "=" * 60)
    print("【业绩比较基准模式示例】")
    print("=" * 60)
    print("计算超额收益 = 持有标的收益 - 基准收益\n")

    # 定义基准代码
    benchmark_code = '000905.SH'  # 沪深300指数
    print(f"持有标的: {code}")
    print(f"比较基准: {benchmark_code}\n")

    # 使用基准模式分析
    result_benchmark = event_study.analyze(
        events=events,
        code=code,
        benchmark_code=benchmark_code,  # 传入基准代码
        window_before=window_before,
        window_after=window_after,
        metric=close_price_metric
    )

    if 'error' not in result_benchmark:
        daily_stats_benchmark = result_benchmark['daily_stats']
        cumulative_stats_benchmark = result_benchmark['cumulative_stats']

        print(f"[OK] 成功分析 {result_benchmark['event_count']} 个事件（超额收益模式）")

        # 展示超额收益的每日统计
        print("\n【超额收益 - 每日平均收益率统计】")
        print(daily_stats_benchmark.to_string())

        # 展示超额收益的累积统计
        print("\n【超额收益 - 累积收益率统计】")
        print(cumulative_stats_benchmark.to_string())

        # 对比分析：绝对收益 vs 超额收益
        print("\n【对比分析】绝对收益 vs 超额收益:")
        for day in [0, 5, 10]:
            if day in cumulative_stats.index and day in cumulative_stats_benchmark.index:
                abs_ret = cumulative_stats.loc[day, 'mean']
                excess_ret = cumulative_stats_benchmark.loc[day, 'mean']
                print(f"\nDay {day}:")
                print(f"  - 绝对收益: {abs_ret:.4%}")
                print(f"  - 超额收益: {excess_ret:.4%}")
                print(f"  - 基准收益(估算): {abs_ret - excess_ret:.4%}")

        # 可选：保存基准对比结果（默认不保存）
        if save_results:
            with pd.ExcelWriter('eventstudy_benchmark_comparison.xlsx') as writer:
                daily_stats.to_excel(writer, sheet_name='absolute_daily')
                cumulative_stats.to_excel(writer, sheet_name='absolute_cumulative')
                daily_stats_benchmark.to_excel(writer, sheet_name='excess_daily')
                cumulative_stats_benchmark.to_excel(writer, sheet_name='excess_cumulative')
                result['returns_matrix'].to_excel(writer, sheet_name='absolute_returns')
                result_benchmark['returns_matrix'].to_excel(writer, sheet_name='excess_returns')
            print(f"\n[OK] 基准对比结果已保存到: eventstudy_benchmark_comparison.xlsx")

    # 【新增功能示例】多标的均值模式
    print("\n" + "=" * 60)
    print("【多标的均值模式示例】")
    print("=" * 60)
    print("在每个时间点计算多只股票的平均收益率\n")

    # 定义股票池
    stock_pool = ['000905.SH']
    print(f"股票池: {stock_pool}\n")

    # 使用多标的模式分析
    result_multi = event_study.analyze(
        events=events,
        code=stock_pool,  # 传入股票列表
        window_before=window_before,
        window_after=window_after,
        metric=close_price_metric
    )

    if 'error' not in result_multi:
        daily_stats_multi = result_multi['daily_stats']
        cumulative_stats_multi = result_multi['cumulative_stats']

        print(f"[OK] 成功分析 {len(result_multi['valid_codes'])} 只股票")
        print(f"有效股票: {result_multi['valid_codes']}")

        # 展示多标的平均收益统计
        print("\n【多标的平均 - 每日收益率统计】")
        print(daily_stats_multi.head(10).to_string())

        print("\n【多标的平均 - 累积收益率统计】")
        print(cumulative_stats_multi.to_string())

        # 可选：保存结果（默认不保存）
        if save_results:
            with pd.ExcelWriter('eventstudy_multi_stock.xlsx') as writer:
                daily_stats_multi.to_excel(writer, sheet_name='daily_stats')
                cumulative_stats_multi.to_excel(writer, sheet_name='cumulative_stats')
                result_multi['returns_matrix'].to_excel(writer, sheet_name='returns_matrix')
                # 保存每只股票的收益矩阵
                for stock_code, stock_df in result_multi['stock_returns_dict'].items():
                    sheet_name = stock_code.replace('.', '_')
                    stock_df.to_excel(writer, sheet_name=sheet_name)
            print(f"\n[OK] 多标的结果已保存到: eventstudy_multi_stock.xlsx")

    # 【画图功能示例】
    print("\n" + "=" * 60)
    print("【画图功能示例】")
    print("=" * 60)

    # 类型1：柱状图展示平均收益率
    print("\n1. 柱状图 - 展示事件前后平均收益率")
    save_chart_path = 'eventstudy_bar_chart.png' if save_results else None
    event_study.plot_bar(
        daily_stats=daily_stats,
        title=f'{code} 事件前后平均收益率',
        save_path=save_chart_path
    )

    # 类型2：折线图展示平均累积收益率
    print("\n2. 折线图 - 展示所有事件的平均累积收益率")
    save_line_path = 'eventstudy_line_chart.png' if save_results else None
    event_study.plot_lines(
        cumulative_stats=cumulative_stats,
        title=f'{code} 事件前后平均累积收益率',
        save_path=save_line_path
    )

    # 类型3：折线图展示同一标的多个事件的累积收益曲线
    print("\n3. 折线图 - 展示同一标的多个事件的累积收益曲线（以Day 0为原点）")
    save_events_path = 'eventstudy_events_chart.png' if save_results else None
    event_study.plot_events_lines(
        events=events,
        code=code,
        window_before=window_before,
        window_after=window_after,
        metric=close_price_metric,
        max_events=10,  # 限制展示前10个事件
        title=f'{code} - 多事件累积收益对比',
        save_path=save_events_path
    )

    # 类型4：折线图展示多标的累积收益对比
    print("\n4. 折线图 - 展示多个标的的平均累积收益率对比")

    # 分别分析多个标的
    stock_codes_compare = ['000905.SH', '000300.SH']  # 可以对比不同标的
    cumulative_stats_dict = {}

    print(f"正在分析 {len(stock_codes_compare)} 个标的...")
    for stock_code in stock_codes_compare:
        result_temp = event_study.analyze(
            events=events,
            code=stock_code,
            window_before=window_before,
            window_after=window_after,
            metric=close_price_metric
        )
        if 'error' not in result_temp:
            cumulative_stats_dict[stock_code] = result_temp['cumulative_stats']

    if cumulative_stats_dict:
        save_multi_line_path = 'eventstudy_multi_line_chart.png' if save_results else None
        event_study.plot_lines(
            cumulative_stats=cumulative_stats_dict,  # 传入字典
            title='多标的平均累积收益率对比',
            save_path=save_multi_line_path,
            show_std=False  # 多标的模式下不显示标准差
        )
        print(f"[OK] 成功对比 {len(cumulative_stats_dict)} 个标的")
    else:
        print("[ERROR] 无法获取标的数据进行对比")

# 关闭数据库连接
df_market.close()
print("\n" + "=" * 60)
print("分析完成")
print("=" * 60)
