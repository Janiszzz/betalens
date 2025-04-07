import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'
from datafeed import Datafeed

class BacktestBase(object):
    def __init__(self, weight, symbol, amount,
                 ftc=0.0, ptc=0.0, verbose=True):
        self.weight = weight
        self.symbol = symbol
        self.start = weight.index[0]
        self.end = weight.index[-1]
        self.initial_amount = amount
        self.amount = amount
        self.ftc = ftc
        self.ptc = ptc
        self.units = 0
        self.position = 0
        self.trades = 0
        self.verbose = verbose
        ''' 
        self.melt_weights()
        self.get_rebalance_data()
        self.get_daily_position_data()'''

    def melt_weights(self):
        return
    def get_rebalance_data(self):


        db = Datafeed("daily_market_data")
        params = {
            'codes': self.weight.columns,
            'datetimes': self.weight.index,
            'metric': "收盘价(元)",
            # 'time_tolerance': 48
        }
        cost = db.query_nearest_after(params)
        return cost


    def get_daily_position_data(self):
        return


#%%
if __name__ == '__main__':
    # 虚拟的权重序列
    weights = pd.DataFrame(0.333, index=pd.date_range(start='2024-01-01 10:00:00', end='2025-01-01 10:00:00', freq='1W'),columns=['000010.SZ','000001.SZ','000002.SZ',])
    weights.index.name = "input_ts"
    bb = BacktestBase(weight=weights, symbol="", amount=1000000)
    cost = bb.get_rebalance_data()
    tmp = pd.pivot_table(cost, values='value', index=['input_ts','datetime'],columns=['code'],)
    #tmp.index.name = "input_ts"
    tmp.columns.name = ""
    tmp = tmp.pct_change().fillna(0)
    tmpp = tmp*weights
    tmpp = (tmpp.sum(axis=1).cumsum()+1)*1000000
    #name = 结算持仓市值（不考虑余额）

