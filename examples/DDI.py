##%%
import sys
import os
import pandas as pd

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from betalens.datafeed import Datafeed, get_absolute_trade_days
from betalens.backtest import BacktestBase
from betalens.analyst import PortfolioAnalyzer, ReportExporter

data = Datafeed("daily_market_data")
date_ranges = get_absolute_trade_days('2020-04-30 10:00:00', '2024-04-30 10:00:00', 'W')
params = {
    'codes': ['512480.SH'],
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
    sig[sig < 0] = -1
    return sig

def make_weights(weights, codes):
    weights.index.name = "input_ts"
    weights.columns = codes
    weights['cash'] = 1 - weights.sum(axis=1)
    return weights

weights = DDI_filter(price).fillna(0)
weights = make_weights(weights,['512480.SH'])
#%%
#包含空头会特别奇怪,净值线和weight重合
bb = BacktestBase(weight=weights, symbol="", amount=1000000)

bb.nav.plot()
bb.weight.plot()
bb.cost_price.plot()
bb.daily_amount.plot()
bb.amount.plot()

analyzer = PortfolioAnalyzer(bb.nav)
exporter = ReportExporter(analyzer)
exporter.generate_annual_report()  # 分年度输出
exporter.generate_custom_report('2021-01-01', '2021-12-31')  # 指定时段（需在数据范围内）