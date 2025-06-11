##%%
import pandas as pd
import betalens.datafeed
import betalens.backtest
import betalens.analyst

data = betalens.datafeed.Datafeed("daily_market_data")
date_ranges = pd.date_range(start='2015-04-30 10:00:00', end='2025-04-30 10:00:00', freq='1W')
params = {
    'codes': ['000001.SZ'],
    'datetimes': date_ranges,
    'metric': "收盘价(元)",
    'time_tolerance': 48 #注意数据缺失带来的错误
}
price = data.query_nearest_before(params)  # panel data


price = price.dropna()
price = price[['datetime','收盘价(元)']].drop_duplicates()
price.set_index('datetime', inplace=True)
price = price.astype(float)
#%%展示容错。中间环节可以不符合betalens规范，只要保证输入规范即可。
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
    weights['cash'] = 1- weights.sum(axis=1)
    return weights

weights = DDI_filter(price).fillna(0)
weights = make_weights(weights,['000001.SZ'])
#%%
bb = betalens.backtest.BacktestBase(weight=weights, symbol="", amount=1000000)

bb.nav.plot()

analyzer = betalens.analyst.PortfolioAnalyzer(bb.nav)
exporter = betalens.analyst.ReportExporter(analyzer)
exporter.generate_annual_report()  # 分年度输出
exporter.generate_custom_report('2019-01-01', '2019-12-31')  # 指定时段