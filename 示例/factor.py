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
import betalens.datafeed
import betalens.backtest
import betalens.analyst

data = betalens.datafeed.Datafeed("daily_market_data")

'''
1. 生成调仓日序列，可先简化处理，后续必须手工清洗交易日序列
2. 查询得到每个调仓日的公司特征，不必展开面板
3. 有两种处理方式：都使用最近日查询，对年报未更新的，采取线性外推或删除的特殊处理
4. 选择得到top10%和bottom10%的个券，设置权重±2/n
5. 权重信号展开面板

'''
date_ranges = pd.date_range(start='2015-04-30 10:00:00', end='2025-04-30 10:00:00', freq='1YE')#按信号生成频率
def get_code_ranges(date_ranges):
    params = {
        'datetimes': date_ranges,
        'metric': "是否可交易",
        # 'time_tolerance': 48
    }
    data = betalens.datafeed.Datafeed("公司特征")
    df = data.query_nearest_before(params)
    df = df.loc[df['value'] == 1]
    df = df[['input_ts','code']]
    
    date_ranges = df['input_ts'].dt.date.drop_duplicates().tolist()

    grouped = df.groupby(df['input_ts'].dt.date)['code'].apply(list).reset_index()
    code_ranges = grouped['code'].tolist()

    return date_ranges, code_ranges

  
date_ranges, code_ranges = get_code_ranges(date_ranges)
    

def single_factor(date_ranges, code_ranges, metric):
    
    params = {
        'code' : code_ranges,
        'datetimes': date_ranges,
        'metric': metric,
        # 'time_tolerance': 48
    }
    data = betalens.datafeed.Datafeed("公司特征")
    firm_cht = data.query_nearest_before(params)
    
    firm_cht = firm_cht.drop(firm_cht[firm_cht['年报时间'] < firm_cht['datetime']].index)
    
    def f(group):
        group[metric + '_weight'] = 0
        group.loc[group[metric + '_label'] == group.loc[:, metric + '_label'].max(),metric + '_weight'] = 1
        group.loc[group[metric + '_label'] == group.loc[:, metric + '_label'].min(), metric + '_weight'] = -1
        return group

    firm_cht = firm_cht.groupby('datetime').apply(f)
    
    return firm_cht

params = {
    'codes': code_ranges,
    'datetimes': date_ranges,
    'metric': "收盘价(元)",
    # 'time_tolerance': 48
}
price = data.query_nearest_after(params)
price.melt()
price = price[['datetime','value']].drop_duplicates()
price.set_index('datetime', inplace=True)
price = price.astype(float)
#%%
def DDI_filter(s):
    s5 = s.rolling(5).mean().fillna(0)
    s20 = s.rolling(20).mean().fillna(0)
    sig = s20 - s5
    sig[sig > 0] = 1
    sig[sig < 0] = -1
    sig = sig.cumsum()/100
    sig[sig > 1] = 1
    sig[sig < -1] = -1
    return sig

def make_weights(weights, codes):
    weights.index.name = "input_ts"
    weights.columns = codes
    return weights

weights = DDI_filter(price).fillna(0)
weights = make_weights(weights,['000010.SZ'])
#%%
bb = betalens.backtest.BacktestBase(weight=weights, symbol="", amount=1000000)

bb.nav.plot()

analyzer = betalens.analyst.PortfolioAnalyzer(bb.nav)
exporter = betalens.analyst.ReportExporter(analyzer)
exporter.generate_annual_report()  # 分年度输出
exporter.generate_custom_report('2024-01-01', '2024-12-31')  # 指定时段