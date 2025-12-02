#%%
'''
Todo:
    - 完成模型工具集模块modelkit
    - 完成数据工具集模块toolkit，逐步进化成Data类和方法
    - 多标的回测正确性检验
    - 迭代性质的滤波器基类开发
    - 可交易表、可交易日、是存表还是做定制化方法
    - 如果在日期和标的存在性不匹配的时候，错误的处理

'''
import pandas as pd
import datetime as dt
import betalens


'''
1. 生成调仓日序列，可先简化处理，后续必须手工清洗交易日序列
2. 查询得到每个调仓日的公司特征，不必展开面板
3. 有两种处理方式：都使用最近日查询，对年报未更新的，采取线性外推或删除的特殊处理
4. 选择得到top10%和bottom10%的个券，设置权重±2/n
5. 权重信号展开面板

'''

def get_tradable_pool(date_list):
    #用时间区间去卡交易状态。交易状态存到盘中某个时点上。
    data = betalens.datafeed.Datafeed("fundamental_data")
    df = pd.DataFrame()
    from datetime import timedelta

    for date in date_list:
        start = date + timedelta(hours=9)
        end = date + timedelta(hours=15)
        params = {
            'start_date': str(start),
            'end_date': str(end),
            'metric': "交易状态",
        }
        request = data.query_data(params)
        request = request.loc[request['value'] == 1]
        df = pd.concat([df, request[['datetime','code']]])

    date_ranges = df['datetime'].dt.date.drop_duplicates().tolist()

    grouped = df.groupby(df['datetime'].dt.date)['code'].apply(list).reset_index()
    #grouped = df.groupby(df['datetime'].dt.date).apply(lambda x: list(x.reset_index()))
    code_ranges = grouped['code'].tolist()

    return date_ranges, code_ranges
 

def single_factor(date_ranges, code_ranges, metric, quantiles):
    labeled_pool = pd.DataFrame()
    for i in range(len(code_ranges)):
        params = {
            'codes' : code_ranges[i],
            'datetimes':[date_ranges[i]],
            'metric': metric,
            'time_tolerance': 24*2*365
        }
        data = betalens.datafeed.Datafeed("fundamental_data")
        labeled_pool = pd.concat([labeled_pool,data.query_nearest_before(params)])
    
    #labeled_pool = labeled_pool.drop(labeled_pool[labeled_pool['年报时间'] < labeled_pool['datetime']].index)
    
    def single_sort(df,keys,quantile_dict):
        if len(keys) != len(quantile_dict):
            raise ValueError("keys 和 quantile_dict 的长度必须相等")
        for key in keys:
            df[key + '_label'] = pd.qcut(df[key].astype(float), quantile_dict[key], labels=False, duplicates='drop')
        return df

    labeled_pool = labeled_pool.groupby('input_ts', as_index=False).apply(lambda group: single_sort(group,[metric],quantiles))
    labeled_pool.set_index(['input_ts', 'code'], inplace=True)

    labeled_pool.metric = metric
    return labeled_pool



def get_single_factor_weight(labeled_pool, params):
    factor_key = params['factor_key']
    import numpy as np

    def f1(group):
        group = group.copy()
        weight_col = factor_key + '_weight'
        label_col = factor_key + '_label'
        group[weight_col] = 0
        max_label = group[label_col].max()
        min_label = group[label_col].min()
        group[weight_col] = np.where(group[label_col] == max_label, 1,
                                     np.where(group[label_col] == min_label, -1, 0))
        group = group[group[weight_col] != 0]
        return group
    def f2(group):
        group = group.copy()
        weight_col = factor_key + '_weight'
        label_col = factor_key + '_label'
        group[weight_col] = 0
        for i in params['long']:
            group.loc[group[label_col] == i, weight_col] = 1
        for i in params['short']:
            group.loc[group[label_col] == i, weight_col] = -1
        group = group[group[weight_col] != 0]
        return group
    if(params['mode'] == 'classic-long-short'):
        labeled_pool = labeled_pool.groupby(labeled_pool.index.get_level_values(0),as_index=False).apply(f1)
    elif(params['mode'] == 'freeplay'):
        labeled_pool = labeled_pool.groupby(labeled_pool.index.get_level_values(0), as_index=False).apply(f2)

    weights = labeled_pool.filter(like='weight').reset_index().pivot(index="input_ts", columns="code", values = factor_key + '_weight')
    weights = weights.fillna(0)

    def normalize_row(row):
        """处理单行数据：正数除以正数之和，负数除以负数之和的绝对值"""
        # 提取正数和负数
        positives = row[row > 0]
        negatives = row[row < 0]

        # 计算正数和与负数绝对值之和
        positive_sum = positives.sum()
        negative_abs_sum = abs(negatives.sum())

        # 处理正数部分
        if positive_sum > 0:
            row[row > 0] = positives / positive_sum

        # 处理负数部分
        if negative_abs_sum > 0:
            row[row < 0] = negatives / negative_abs_sum

        return row

    weights = weights.apply(normalize_row, axis=1)

    return weights

def describe_labeled_pool(labeled_pool):

    pivot = pd.pivot_table(
        data=labeled_pool.reset_index(),
        index='input_ts',
        columns=labeled_pool.metric+'_label',
        values=labeled_pool.metric,
        aggfunc=['count', 'mean'],  # 同时计算总和和平均值
        margins=True,  # 添加总计行/列
        margins_name='Total'
    )
    return pivot

#%%
if __name__ == '__main__':
    data = betalens.datafeed.Datafeed("daily_market_data")

    date_list = betalens.datafeed.get_absolute_trade_days("2015-04-30","2024-04-30","Y")

    date_ranges, code_ranges = get_tradable_pool(date_list)

    metric = "股息率(报告期)"
    quantiles = {
        "股息率(报告期)": 1000,
        # "股息率(报告期)": [0,.2,.4,.6,.8,1]
    }

    labeled_pool = single_factor(date_ranges, code_ranges, metric, quantiles)

    describe_labeled_pool(labeled_pool)

    params = {
        'factor_key' : '股息率(报告期)',
        'mode' : 'freeplay', #'classic-long-short','freeplay',
        'long' : [200,201],
        'short' : [800,801],
    }
    #1 极值组long short
    weights = get_single_factor_weight(labeled_pool, params)

    weights['cash'] = 0
    bb = betalens.backtest.BacktestBase(weight=weights, symbol="", amount=1000000)

    bb.nav.plot()

    analyzer = betalens.analyst.PortfolioAnalyzer(bb.nav)
    exporter = betalens.analyst.ReportExporter(analyzer)
    exporter.generate_annual_report()  # 分年度输出
    exporter.generate_custom_report('2024-01-01', '2024-12-31')  # 指定时段'''