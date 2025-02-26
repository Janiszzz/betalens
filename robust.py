#ToDo：
#%%Lucky Factors, Harvey and Liu (2021)
import pandas as pd
import numpy as np
import concurrent.futures
import statsmodels.api as sm
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

def neu(X,y):
    OX = pd.DataFrame()
    T = {}
    #逐个进行单因子回归
    for i in range(X.shape[1]):
        Xi = X.iloc[:, i]
        model = sm.OLS(Xi, sm.add_constant(y)).fit()
        OX = pd.concat([OX,model.resid],axis=1)
        
        model = sm.OLS(y, sm.add_constant(Xi)).fit()
        T[X.columns[i]] = model.tvalues.iloc[1]
        
        
    OX.columns = X.columns
    T = pd.DataFrame(T,index = [y.name]).T.abs()
    return OX,T

    
def bootstrap_resample(data):
    indices = np.random.choice(range(data.shape[0]),  data.shape[0], replace=True)
    bootstrapped_data = data.iloc[indices,:]     
    return bootstrapped_data

def max_statistic(data):
    bs_y = data.iloc[:,0]
    bs_OX = data.iloc[:,1:]
    
    t_statistics = []
    
    for i in range(bs_OX.shape[1]):
        bs_OXi = bs_OX.iloc[:, i]
        model = sm.OLS(bs_y, sm.add_constant(bs_OXi)).fit()
        t_statistics.append(model.tvalues.iloc[1])
        
    return np.max(t_statistics)

def bootstrap_once(OX, y, T, n_bootstraps=1000):
    
    max_statistic_pdf = []
    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = pd.concat([y, OX], axis=1)
        for i in range(n_bootstraps):
            future = executor.submit(max_statistic, bootstrap_resample(data))
            futures.append(future)
        for future in concurrent.futures.as_completed(futures):
            max_statistic_pdf.append(future.result())

    max_statistic_pdf = np.sort(np.abs(np.array(max_statistic_pdf)))
    modifd_P = T.abs().apply(lambda x : np.searchsorted(max_statistic_pdf,x)/len(max_statistic_pdf))
    modifd_P = 1- modifd_P

    eff_fct_name = modifd_P.loc[modifd_P[y.name]<0.1].index
    
    return eff_fct_name,modifd_P,max_statistic_pdf
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
ans = pd.DataFrame()
print("start")
for fund in ret.columns:
    # fund = '005833.OF'
    data = pd.concat([ret[fund], fct], axis=1).dropna()

    X = data.iloc[:-1, 1:].reset_index(drop=True)
    y = data.iloc[1:, 0].reset_index(drop=True)

    while (1):
        if (X.shape[0] < 100 or X.shape[1] == 0):
            print("out")
            break
        OX, T = neu(X, y)

        eff_fct_name, modifd_P, max_statistic_pdf = bootstrap_once(OX, y, T)
        not_eff_fct_name = list(set(X.columns) - set(eff_fct_name))

        if (len(not_eff_fct_name) == 0 or len(eff_fct_name) == 0):
            modifd_P.columns = [fund]
            ans = pd.concat([ans, modifd_P], axis=1)
            print(modifd_P)
            break

        oy = y
        for i in eff_fct_name:
            EXi = X.loc[:, i]
            model = sm.OLS(oy, sm.add_constant(EXi)).fit()
            # print(model.summary())
            oy = model.resid

        X = X[not_eff_fct_name]
        y = oy

ans.to_excel("ans.xlsx")
#%%
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
    fake_y = X.mul(B.iloc[0,:],axis=1).drop('const',axis=1).sum(axis=1)+bootstrapped_OX
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
    #X = data.iloc[:,1:].reset_index(drop=True)
    #y = data.iloc[:,0].reset_index(drop=True)

    B,OX,T = panel(X, y)
    modifd_P,max_statistic_pdf = bootstrap_fake_fund(X,B,OX,T)
    modifd_P.index = [fund]
    print(modifd_P)
    ans = pd.concat([ans, modifd_P])
    
        
        
        
        
        
        
        
        
        
        
        
        






