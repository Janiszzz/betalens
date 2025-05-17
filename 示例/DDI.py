#%%
import pandas as pd
import betalens.datafeed
import betalens.backtest
import betalens.analyst

data = betalens.datafeed.Datafeed("daily_market_data")
date_ranges = pd.date_range(start='2015-01-01 10:00:00', end='2025-01-01 10:00:00', freq='1D')
params = {
    'codes': ['000010.SZ'],
    'datetimes': date_ranges,
    'metric': "收盘价(元)",
    # 'time_tolerance': 48
}
price = data.query_nearest_after(params)  # panel data
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