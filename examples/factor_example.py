#%%
"""
单因子、双因子、多因子回测示例
"""
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datafeed import Datafeed, get_absolute_trade_days
from backtest import BacktestBase
from analyst import PortfolioAnalyzer, ReportExporter
from factor import (
    get_tradable_pool, pre_query_factor_data, single_factor, double_factor,
    multi_factor, get_single_factor_weight, get_double_factor_weight,
    get_multi_factor_weight, describe_labeled_pool, describe_double_labeled_pool,
    describe_multi_labeled_pool
)


if __name__ == '__main__':
    """
    单因子回测示例：
    1. 生成调仓日序列
    2. 获取可交易股票池
    3. 对因子分组打标签
    4. 生成多空权重
    5. 回测并输出绩效报告
    """
    
    # 1. 生成调仓日序列（年度频率）
    date_list = get_absolute_trade_days("2015-04-30", "2024-04-30", "Y")
    
    # 2. 获取可交易股票池（在流程开始处调用一次）
    date_ranges, code_ranges = get_tradable_pool(date_list)
    
    # 3. 预查询因子数据（传入可交易池，避免重复查询）
    metric = "股息率(报告期)"
    pre_queried_data = pre_query_factor_data(
        date_list, 
        metric, 
        time_tolerance=24*2*365,
        table_name="fundamental_data",
        date_ranges=date_ranges,
        code_ranges=code_ranges
    )
    
    # 4. 单因子分组
    quantiles = {
        "股息率(报告期)": 10,  # 分为10组
        # "股息率(报告期)": [0, .2, .4, .6, .8, 1]  # 或使用自定义分位数
    }
    
    labeled_pool = single_factor(pre_queried_data, metric, quantiles)
    
    # 查看因子分组统计
    pivot_stats = describe_labeled_pool(labeled_pool)
    print("因子分组统计：")
    print(pivot_stats)
    
    # 5. 生成多空权重
    params = {
        'factor_key': '股息率(报告期)',
        'mode': 'freeplay',  # 'classic-long-short' 或 'freeplay'
        'long': [1],   # 做多组（分位数编号）
        'short': [8],  # 做空组（分位数编号）
    }
    weights = get_single_factor_weight(labeled_pool, params)
    
    # 调试信息：查看权重数据
    print("\n权重数据统计：")
    print(f"权重形状: {weights.shape}")
    print(f"非零权重数量: {(weights != 0).sum().sum()}")
    print(f"权重范围: [{weights.min().min():.4f}, {weights.max().max():.4f}]")
    print(f"每行权重和范围: [{weights.sum(axis=1).min():.4f}, {weights.sum(axis=1).max():.4f}]")
    print("\n前几行权重数据：")
    print(weights.head())
    
    # 添加现金列
    weights['cash'] = 0
    
    # 6. 回测
    bb = BacktestBase(weight=weights, symbol="", amount=1000000)
    
    # 绘制净值曲线
    bb.nav.plot()
    
    # 7. 绩效分析
    analyzer = PortfolioAnalyzer(bb.nav)
    exporter = ReportExporter(analyzer)
    exporter.generate_annual_report()  # 分年度输出
    exporter.generate_custom_report('2024-01-01', '2024-12-31')  # 指定时段
    
    print("\n" + "="*80)
    print("双因子回测示例 (Double Factor Backtest)")
    print("="*80 + "\n")
    
    # ========== 双因子回测示例 ==========
    # 设置开关：是否运行双因子示例
    RUN_DOUBLE_FACTOR = False  # 改为 True 以运行双因子示例
    
    if RUN_DOUBLE_FACTOR:
        # 1. 使用相同的调仓日序列
        # date_list 已在上面定义
        
        # 2. 获取可交易股票池（在流程开始处调用一次）
        date_ranges, code_ranges = get_tradable_pool(date_list)
        
        # 3. 预查询双因子数据（传入相同的可交易池，避免重复查询）
        metric1 = "市净率"  # 主因子
        metric2 = "股息率(报告期)"  # 次因子
        
        pre_queried_data1 = pre_query_factor_data(
            date_list, 
            metric1, 
            time_tolerance=24*2*365,
            table_name="fundamental_data",
            date_ranges=date_ranges,
            code_ranges=code_ranges
        )
        
        pre_queried_data2 = pre_query_factor_data(
            date_list, 
            metric2, 
            time_tolerance=24*2*365,
            table_name="fundamental_data",
            date_ranges=date_ranges,
            code_ranges=code_ranges
        )
        
        # 3. 双因子分组 - 演示两种排序方法
        quantiles1 = {metric1: 3}  # 主因子分3组（小、中、大）
        quantiles2 = {metric2: 3}  # 次因子分3组（低、中、高）
        
        print(f"双因子设置：主因子={metric1}({quantiles1[metric1]}组), 次因子={metric2}({quantiles2[metric2]}组)")
        
        # ===== 方法1：条件排序（Dependent Sort）=====
        print("\n" + "-"*80)
        print("方法1：条件排序（Dependent Sort）")
        print("-"*80)
        
        labeled_pool_dependent = double_factor(
            pre_queried_data1, 
            pre_queried_data2, 
            metric1, 
            metric2, 
            quantiles1, 
            quantiles2,
            sort_method='dependent'  # 条件排序
        )
        
        # 查看条件排序的统计
        count_pivot_dep, mean_pivot1_dep, mean_pivot2_dep = describe_double_labeled_pool(labeled_pool_dependent)
        
        print("各组合样本数统计（行=主因子组，列=次因子组）：")
        print(count_pivot_dep)
        
        # ===== 方法2：独立排序（Independent Sort）=====
        print("\n" + "-"*80)
        print("方法2：独立排序（Independent Sort）")
        print("-"*80)
        
        labeled_pool_independent = double_factor(
            pre_queried_data1, 
            pre_queried_data2, 
            metric1, 
            metric2, 
            quantiles1, 
            quantiles2,
            sort_method='independent'  # 独立排序
        )
        
        # 查看独立排序的统计
        count_pivot_indep, mean_pivot1_indep, mean_pivot2_indep = describe_double_labeled_pool(labeled_pool_independent)
        
        print("各组合样本数统计（行=主因子组，列=次因子组）：")
        print(count_pivot_indep)
        
        # ===== 对比两种方法的差异 =====
        print("\n" + "-"*80)
        print("两种排序方法的样本数差异（Independent - Dependent）：")
        print("-"*80)
        diff_pivot = count_pivot_indep - count_pivot_dep
        print(diff_pivot)
        
        # 使用条件排序的结果进行后续回测
        labeled_pool_double = labeled_pool_dependent
        
        # 3. 查看双因子分组统计（使用条件排序）
        count_pivot, mean_pivot1, mean_pivot2 = describe_double_labeled_pool(labeled_pool_double)
        
        print("\n各组合样本数统计（行=主因子组，列=次因子组）：")
        print(count_pivot)
        
        print(f"\n主因子({metric1})在各组合中的均值：")
        print(mean_pivot1)
        
        print(f"\n次因子({metric2})在各组合中的均值：")
        print(mean_pivot2)
        
        # 4. 生成双因子多空权重
        # 策略：做多小市净率+高股息率（价值股），做空大市净率+低股息率（成长股）
        double_params = {
            'factor_key1': metric1,
            'factor_key2': metric2,
            'mode': 'freeplay',
            'long_combinations': [(0, 2)],   # 做多：主因子最小组(0) + 次因子最大组(2)
            'short_combinations': [(2, 0)],  # 做空：主因子最大组(2) + 次因子最小组(0)
        }
        
        weights_double = get_double_factor_weight(labeled_pool_double, double_params)
        
        # 调试信息
        print("\n双因子权重数据统计：")
        print(f"权重形状: {weights_double.shape}")
        print(f"非零权重数量: {(weights_double != 0).sum().sum()}")
        print(f"权重范围: [{weights_double.min().min():.4f}, {weights_double.max().max():.4f}]")
        print(f"每行权重和: [{weights_double.sum(axis=1).min():.4f}, {weights_double.sum(axis=1).max():.4f}]")
        
        # 添加现金列
        weights_double['cash'] = 0
        
        # 5. 双因子回测
        bb_double = BacktestBase(weight=weights_double, symbol="", amount=1000000)
        
        # 绘制净值曲线
        bb_double.nav.plot()
        
        # 6. 双因子绩效分析
        analyzer_double = PortfolioAnalyzer(bb_double.nav)
        exporter_double = ReportExporter(analyzer_double)
        
        print("\n双因子策略绩效报告：")
        exporter_double.generate_annual_report()
        exporter_double.generate_custom_report('2024-01-01', '2024-12-31')
        
        print("\n双因子回测完成！")
    else:
        print("提示：将 RUN_DOUBLE_FACTOR 设为 True 以运行双因子示例")
    
    print("\n" + "="*80)
    print("多因子回测示例 (Multi-Factor Backtest)")
    print("="*80 + "\n")
    
    # ========== 多因子回测示例 ==========
    # 设置开关：是否运行多因子示例
    RUN_MULTI_FACTOR = False  # 改为 True 以运行多因子示例
    
    if RUN_MULTI_FACTOR:
        # 1. 使用相同的调仓日序列
        # date_list 已在上面定义
        
        # 2. 获取可交易股票池（在流程开始处调用一次）
        date_ranges, code_ranges = get_tradable_pool(date_list)
        
        # 3. 预查询多因子数据 - 3因子示例：市值、账面市值比、股息率
        factors_config = [
            {'name': '市值', 'quantiles': 3, 'method': 'dependent'},      # 先按市值分3组
            {'name': '账面市值比', 'quantiles': 3, 'method': 'dependent'},  # 在市值组内按账面市值比分3组
            {'name': '股息率(报告期)', 'quantiles': 2, 'method': 'independent'},  # 独立按股息率分2组
        ]
        
        print(f"多因子设置：")
        for i, f in enumerate(factors_config):
            print(f"  因子{i+1}: {f['name']} ({f['quantiles']}组, {f['method']})")
        total_combos = 1
        for f in factors_config:
            total_combos *= f['quantiles']
        print(f"  预期形成: {total_combos} 个组合\n")
        
        # 预查询每个因子的数据（传入相同的可交易池，避免重复查询）
        pre_queried_data_list = []
        for factor_config in factors_config:
            metric = factor_config['name']
            pre_queried_data = pre_query_factor_data(
                date_list, 
                metric, 
                time_tolerance=24*2*365,
                table_name="fundamental_data",
                date_ranges=date_ranges,
                code_ranges=code_ranges
            )
            pre_queried_data_list.append(pre_queried_data)
        
        # 3. 多因子分组
        labeled_pool_multi = multi_factor(
            pre_queried_data_list, 
            factors_config
        )
        
        # 4. 查看多因子分组统计
        stats = describe_multi_labeled_pool(labeled_pool_multi, max_display_dims=2)
        
        print("各组合样本数统计（前2个因子交叉）：")
        print(stats['count_pivot'])
        
        print(f"\n各因子在各组合中的均值：")
        for factor_name, mean_pivot in stats['mean_pivots'].items():
            print(f"\n{factor_name}:")
            print(mean_pivot)
        
        # 5. 生成多因子多空权重
        # 策略：做多小市值+高账面市值比+高股息率，做空大市值+低账面市值比+低股息率
        multi_params = {
            'mode': 'freeplay',
            'long_combinations': [
                (0, 2, 1),  # 小市值(0) + 高账面市值比(2) + 高股息率(1)
            ],
            'short_combinations': [
                (2, 0, 0),  # 大市值(2) + 低账面市值比(0) + 低股息率(0)
            ],
        }
        
        weights_multi = get_multi_factor_weight(labeled_pool_multi, multi_params)
        
        # 调试信息
        print("\n多因子权重数据统计：")
        print(f"权重形状: {weights_multi.shape}")
        print(f"非零权重数量: {(weights_multi != 0).sum().sum()}")
        print(f"权重范围: [{weights_multi.min().min():.4f}, {weights_multi.max().max():.4f}]")
        print(f"每行权重和: [{weights_multi.sum(axis=1).min():.4f}, {weights_multi.sum(axis=1).max():.4f}]")
        
        # 添加现金列
        weights_multi['cash'] = 0
        
        # 6. 多因子回测
        bb_multi = BacktestBase(weight=weights_multi, symbol="", amount=1000000)
        
        # 绘制净值曲线
        bb_multi.nav.plot()
        
        # 7. 多因子绩效分析
        analyzer_multi = PortfolioAnalyzer(bb_multi.nav)
        exporter_multi = ReportExporter(analyzer_multi)
        
        print("\n多因子策略绩效报告：")
        exporter_multi.generate_annual_report()
        exporter_multi.generate_custom_report('2024-01-01', '2024-12-31')
        
        print("\n多因子回测完成！")
    else:
        print("提示：将 RUN_MULTI_FACTOR 设为 True 以运行多因子示例")

