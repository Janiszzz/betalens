#ToDo：
'''
这是收益检测模块，对抗过拟合的最后环节，基于Lucky Factor
设计上应该与挖掘模块高度隔离，处于pipeline最下游
输入多个因子收益序列，分析其中一个因子中性化之后的有效程度，输出检验结果、新因子都被哪几个老因子解释
'''
#%%Lucky Factors, Harvey and Liu (2021)
import pandas as pd
import numpy as np
import concurrent.futures
import statsmodels.api as sm
from sympy.codegen.cfunctions import isnan


class RobustTest(object):
    def __init__(self, fund, factor):
        data = pd.concat([fund, factor], axis=1).dropna()
        X = data.iloc[:-1, 1:].reset_index(drop=True)
        y = data.iloc[1:, 0].reset_index(drop=True)

        self.X = X  # 因子数据
        self.y = y  # 资产收益数据
        self.OX = pd.DataFrame()  # 正交化后的因子数据
        self.T = pd.DataFrame()  # t统计量
        self.fund_name = fund.name

    @staticmethod
    def create_sample_dataframes():
        # 创建一个包含时间戳的日期范围
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')

        # 假设我们有两个DataFrame，一个包含资产收益率，一个包含因子值
        asset_returns = pd.DataFrame({
            'date': dates,
            'asset_1': np.random.randn(1000),
            'asset_2': np.random.randn(1000),
            'asset_3': np.random.randn(1000)
        }).set_index('date')

        factor_values = pd.DataFrame({
            'date': dates,
            'factor_1': np.random.randn(1000),
            'factor_2': np.random.randn(1000),
            'factor_3': np.random.randn(1000),
            'factor_4': np.random.randn(1000),
            'factor_5': np.random.randn(1000),
            'factor_6': np.random.randn(1000)
        }).set_index('date')
        return asset_returns, factor_values

    def neu(self):
        #逐个进行单因子回归
        self.T = {}
        for i in range(self.X.shape[1]):
            Xi = self.X.iloc[:, i]
            model = sm.OLS(Xi, sm.add_constant(self.y)).fit()
            self.OX = pd.concat([self.OX,model.resid],axis=1)

            model = sm.OLS(self.y, sm.add_constant(Xi)).fit()
            self.T[self.X.columns[i]] = model.tvalues.iloc[1]

        self.OX.columns = self.X.columns
        self.T = pd.DataFrame(self.T,index = [self.y.name]).T.abs()
        return self.OX,self.T

    
    def bootstrap_resample(self, data):
        indices = np.sort(np.random.choice(range(data.shape[0]),  data.shape[0], replace=True))
        bootstrapped_data = data.iloc[indices,:]
        return bootstrapped_data

    def max_statistic(self, data):
        bs_y = data.iloc[:,0]
        bs_OX = data.iloc[:,1:]

        t_statistics = []

        for i in range(bs_OX.shape[1]):
            bs_OXi = bs_OX.iloc[:, i]
            model = sm.OLS(bs_y, sm.add_constant(bs_OXi)).fit()
            t_statistics.append(model.tvalues.iloc[1])

        return np.max(t_statistics)

    def bootstrap_once(self, n_bootstraps=1000):

        max_statistic_pdf = []
        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            data = pd.concat([self.y, self.OX], axis=1)
            for i in range(n_bootstraps):
                future = executor.submit(self.max_statistic, self.bootstrap_resample(data))
                futures.append(future)
            for future in concurrent.futures.as_completed(futures):
                max_statistic_pdf.append(future.result())

        max_statistic_pdf = np.sort(np.abs(np.array(max_statistic_pdf)))
        modifd_P = self.T.abs().apply(lambda x : np.searchsorted(max_statistic_pdf,x)/len(max_statistic_pdf))
        modifd_P = 1- modifd_P

        eff_fct_name = modifd_P.loc[modifd_P[y.name]<0.1].index

        return eff_fct_name,modifd_P,max_statistic_pdf
    def work(self):
        while (1):
            if (self.X.shape[0] < 100 or self.X.shape[1] == 0):
                print("out")
                break
            self.neu()

            eff_fct_name, modifd_P, max_statistic_pdf = self.bootstrap_once(OX, y, T)
            not_eff_fct_name = list(set(self.X.columns) - set(eff_fct_name))

            if (len(not_eff_fct_name) == 0 or len(eff_fct_name) == 0):
                modifd_P.columns = [self.fund_name]
                #ans = pd.concat([ans, modifd_P], axis=1)
                #print(modifd_P)
                return modifd_P

            oy = self.y
            for i in eff_fct_name:
                EXi = self.X.loc[:, i]
                model = sm.OLS(oy, sm.add_constant(EXi)).fit()
                # print(model.summary())
                oy = model.resid

            self.X = self.X[not_eff_fct_name]
            self.y = oy
        print("Finish!")
        print()

#%%
'''
    import init
    codes = ["000906.SH", "512760.SH", "588000.SH", "512530.SH", "159708.SZ", "159928.SZ", "159707.SZ", "512000.SH",
                 "518880.SH", "511520.SH", "511090.SH"]
    data = pd.DataFrame()
    for code in codes:
        data = pd.concat([data,init.get_mongo(code)])
    '''

code = pd.read_excel(r".\红利主题.xlsx")
code = code['代码'].str.cat(sep=',')

ret = pd.read_excel(r".\ret.xlsx").set_index("date")

fct = pd.read_excel(r".\指数行情序列.xlsx").set_index("date")
fct = fct.pct_change()
# ret = fct
# %%
print("start")
result = []
for fund in ret.columns:
    # fund = '005833.OF'
    new = RobustTest(ret[fund], fct)
    result.append(new.work())
result.to_excel("result.xlsx")
#%%
import pandas as pd
import numpy as np
import concurrent.futures
import statsmodels.api as sm

fct = pd.read_excel(r".\betaplus-1000-indexdaily.xlsx").set_index("date")
fct = fct.pct_change()
ret = pd.read_excel(r".\ret.xlsx").set_index("date")


def panel(X,y):
    model = sm.OLS(y, sm.add_constant(X)).fit()
    B = model.params
    B = pd.DataFrame(B).T
    OX = model.resid
    T = model.tvalues
    T = pd.DataFrame(T).T
    return B,OX,T

def fake_fund(X,B,OX):
    indices = np.random.choice(range(OX.shape[0]),  OX.shape[0], replace=True)
    bootstrapped_OX = OX.iloc[indices].reset_index(drop=True) 
    fake_y = X.mul(B.iloc[0,:],axis=1).drop('const',axis=1).sum(axis=1) + bootstrapped_OX
    return fake_y
    
def bootstrap_fake_fund(X,B,OX,T, n_bootstraps=1000):
    max_statistic_pdf = []
    '''    for i in range(0,n_bootstraps):
        fake_y = fake_fund(X,B,OX)
        _, _, bootstrapped_T = panel(X, fake_y)
        max_statistic_pdf.append(bootstrapped_T.loc[0,'const'])
        '''
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:

        for i in range(n_bootstraps):
            future = executor.submit(panel, X, fake_fund(X,B,OX))
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            max_statistic_pdf.append(future.result()[2].loc[0,'const'])


    max_statistic_pdf = np.sort(np.abs(np.array(max_statistic_pdf)))
    
    modifd_P = T.abs().apply(lambda x : np.searchsorted(max_statistic_pdf,x)/len(max_statistic_pdf))
    modifd_P = 1 - modifd_P

    return modifd_P['const'],max_statistic_pdf
#%%
ans = pd.DataFrame()
print("start")
for fund in ret.columns:
    #fund = '005833.OF'
    data = pd.concat([ret[fund],fct],axis=1).dropna()

    X = data.iloc[:-1,1:].reset_index(drop=True)
    y = data.iloc[1:,0].reset_index(drop=True)
    if(y.shape[0] == 0):
        continue
    #X = data.iloc[:,1:].reset_index(drop=True)
    #y = data.iloc[:,0].reset_index(drop=True)

    B,OX,T = panel(X, y)
    modifd_P,max_statistic_pdf = bootstrap_fake_fund(X,B,OX,T)
    modifd_P.index = [fund]
    print(modifd_P)
    ans = pd.concat([ans, modifd_P])

#%%
def get_interval(df, start=None, end=None):
    # 假设 df 的索引是时间序列
    if isinstance(df.index, pd.DatetimeIndex):
        if start is None and end is None:
            return df  # 返回完整数据框，因为没有时间限制

        elif start is not None and end is not None:
            return df.loc[start:end]

        elif start is not None:
            # 筛选从 start 到最后一个时间点的数据
            return df.loc[start:]

        else:  # end is not None
            # 筛选从第一个时间点到 end 的数据
            return df.loc[:end]

    # 如果没有时间索引，假设有一个时间列名为 'datetime'
    elif 'datetime' in df.columns:
        mask = []

        if start is not None and end is not None:
            mask = df['datetime'].between(start, end)
        elif start is not None:
            mask = df['datetime'] >= start
        else:  # end is not None
            mask = df['datetime'] <= end

        return df[mask]

    else:
        raise ValueError("DataFrame 必须包含时间索引或 'datetime' 列")

def work(fund,fct):
    data = pd.concat([fund,fct],axis=1).dropna()
    X = data.iloc[:-1,1:].reset_index(drop=True)
    y = data.iloc[1:,0].reset_index(drop=True)
    if(y.shape[0] == 0):
        return -1
    #X = data.iloc[:,1:].reset_index(drop=True)
    #y = data.iloc[:,0].reset_index(drop=True)

    B,OX,T = panel(X, y)
    modifd_P,max_statistic_pdf = bootstrap_fake_fund(X,B,OX,T)

    return modifd_P

def parse_name_dates(s):
    """
    将字符串格式 '姓名(开始日期-结束日期)' 拆解为姓名和两个datetime对象

    Args:
        s (str): 输入字符串，例如 '盛丰衍(20180711-20250101)'

    Returns:
        dict: 包含以下键的字典：
            - 'name': 提取的姓名
            - 'start_date': 开始日期 datetime 对象
            - 'end_date': 结束日期 datetime 对象

    Raises:
        ValueError: 如果字符串格式不正确或无法解析日期
    """
    from datetime import datetime
    try:
        # 拆分名称和日期部分
        name_part = s.split('(')[0]
        date_part = s.split(')')[0].split('(')[1]  # 获取括号内的内容

        # 解析开始和结束日期
        start_date_str, end_date_str = date_part.split('-')
        date_format = '%Y%m%d'

        start_date = datetime.strptime(start_date_str, date_format)
        end_date = datetime.strptime(end_date_str, date_format)

        return {
            'name': name_part,
            'start_date': start_date,
            'end_date': end_date
        }

    except (IndexError, ValueError) as e:
        raise ValueError(f"无法解析字符串 '{s}': {e}")

intervals = pd.read_excel(r".\datepairs.xlsx")
ans = []
for fund in intervals.columns:
    for datepairs in intervals[fund]:
        if(pd.isna(datepairs)):
            continue
        tmp = parse_name_dates(datepairs)
        a,b = get_interval(ret[fund],tmp['start_date'],tmp['end_date']),get_interval(fct,tmp['start_date'],tmp['end_date'])
        result = work(a.fillna(0),b.fillna(0))
        print(fund,datepairs,result)
        ans.append([fund,datepairs,result])

        
        
        
pd.DataFrame(ans).to_excel(r".\250318.xlsx")





