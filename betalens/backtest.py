import numpy as np
import pandas as pd
from pylab import mpl, plt
plt.style.use('seaborn-v0_8')
mpl.rcParams['font.family'] = 'serif'
from betalens.datafeed import Datafeed

class BacktestBase(object):
    def __init__(self, weight, symbol, amount,
                 ftc=0.0, ptc=0.0, verbose=True):
        self.cost_ret = None
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

        self.get_rebalance_data()
        self.get_position_data()
        self.get_daily_position_data()
        ''' 
        self.melt_weights()
        self.get_rebalance_data()
        self.get_daily_position_data()'''

    def melt_weights(self):
        try:
            if("code" in self.weight.columns):
                self.weight = pd.pivot_table(self.weight, values='weight', index=['input_ts'], columns=['code'], )
            return 0
        except:
            return 1
    def get_rebalance_data(self):

        db = Datafeed("daily_market_data")
        params = {
            'codes': self.weight.columns.drop('cash'),
            'datetimes': self.weight.index,
            'metric': "收盘价(元)",
            # 'time_tolerance': 48
        }
        #当天开始调仓，并不一定是当天就调。只需控制偏离不晚于下一次调仓。
        self.cost_price = db.query_nearest_after(params)
        self.cost_price = pd.pivot_table(self.cost_price, values='value', index=['input_ts', 'datetime'], columns=['code'], )
        self.cost_price.columns.name = ""
        self.cost_price['cash'] = 1
        self.cost_ret = self.cost_price.pct_change().fillna(0)
        self.start = self.cost_price.index.get_level_values('datetime')[0]
        self.end = self.cost_price.index.get_level_values('datetime')[-1]
        self.weight.index = self.cost_price.index
        return self.cost_price

    def get_position_data(self):
        self.amount = self.cost_ret * self.weight
        self.amount['amount'] = (self.amount.sum(axis=1).cumsum() + 1) * self.initial_amount
        #self.amount.reset_index(inplace=True)
        #self.amount = self.amount[['datetime','amount']].set_index('datetime')
        self.amount = self.amount['amount']
        self.amount = self.amount.shift(1).fillna(self.initial_amount)
        return self.amount

    def get_daily_position_data(self):
        
        db = Datafeed("daily_market_data")
        params = {
            'start_date': str(self.start),
            'end_date': str(self.end),
            'code': self.weight.columns.drop('cash'),
            'metric': "收盘价(元)"
        }
        close_price_ts = db.query_data(params)
        #close_price_ts.set_index("datetime", inplace=True)
        close_price_ts = pd.pivot_table(close_price_ts, values='value', index=['datetime'], columns=['code'], )
        close_price_ts.columns.name = ""
        close_price_ts['cash'] = 1
        self.position = self.weight.mul(self.amount, axis=0)
        self.position = self.position.div(self.cost_price)
        self.position = self.position.droplevel('input_ts').drop_duplicates()
        self.position = self.position.reindex(close_price_ts.index, method='ffill')
        self.daily_amount = self.position.mul(close_price_ts, axis=0).sum(axis=1).astype(float)
        self.nav = (self.daily_amount / self.initial_amount)
        return self.daily_amount

#%%
if __name__ == '__main__':
    # 虚拟的权重序列
    weights = pd.DataFrame(0.33333333, index=pd.date_range(start='2024-01-01 10:00:00', end='2025-01-01 10:00:00', freq='1D'),columns=['000010.SZ','000001.SZ','000002.SZ',])
    weights.index.name = "input_ts"

    bb = BacktestBase(weight=weights, symbol="", amount=1000000)
    #name = 结算持仓市值（不考虑余额）

